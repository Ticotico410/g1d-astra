from __future__ import annotations

import os
from pathlib import Path
import signal
import sys
import threading
import time

import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
import tyro

from .config import CONFIG_DIR, config_path_for_robot, load_config
from .direct_vr import DirectVRReader
from .feedforward import PinocchioRNEAFeedforward
from .math_utils import apply_relative_pose, pose7_to_pos_quat_xyzw, relative_pose, transform_pico_pose_to_robot
from .mujoco_robot import MujocoVisualizer
from .placo_robot import IKTarget, PlacoIKSolver
from .publisher import make_publisher


class TeleopRunner:
    def __init__(
        self,
        config_path: str,
        common_config_path: str | None,
        robot: str,
        mode: str,
        net: str,
        scale: float,
        body_regularization_weight: float | None = None,
        placo_regularization_weight: float | None = None,
        lz_mt_weight: float | None = None,
        lz_it_weight: float | None = None,
        yaw_joint_weight: float | None = None,
        torso_joint_weight: float | None = None,
        max_hand_delta: float | None = None,
        max_joint_step: float | None = None,
        teleop_control_mode: str | None = None,
        record: bool = False,
        record_dir: str | None = None,
        record_task_name: str | None = None,
        record_fps: int | None = None,
        record_images: bool | None = None,
        image_server_host: str | None = None,
        image_server_port: int | None = None,
        record_tts: bool | None = None,
    ) -> None:
        self.robot = robot
        self.cfg = load_config(config_path, common_config_path)
        self.cfg.robot_name = robot
        if teleop_control_mode is not None:
            self.cfg.teleop_control_mode = teleop_control_mode
        self.cfg.teleop_control_mode = self._normalize_control_mode(self.cfg.teleop_control_mode)
        if body_regularization_weight is not None:
            self.cfg.body_regularization_weight = body_regularization_weight
        if placo_regularization_weight is not None:
            self.cfg.placo_regularization_weight = placo_regularization_weight
        if max_hand_delta is not None:
            self.cfg.max_hand_delta = max_hand_delta
        if max_joint_step is not None:
            self.cfg.max_joint_step = max_joint_step
        for joint_name, weight in {
            "LZ_mt_Joint": lz_mt_weight,
            "LZ_it_Joint": lz_it_weight,
            "Yaw_Joint": yaw_joint_weight,
            "torso_Joint": torso_joint_weight,
        }.items():
            if weight is not None:
                self.cfg.body_joint_regularization_weights[joint_name] = weight
        self.mode = mode
        self.scale = scale
        # A joint owned by the joystick in manual mode must be masked out of IK,
        # otherwise the solver pulls it back to its whole-body solution every
        # frame and fights the integrated joystick target.
        self.manual_joystick_joint = self._resolve_manual_joystick_joint()
        ik_masked_joints = list(self.cfg.ik_masked_joints)
        if self.manual_joystick_joint and self.manual_joystick_joint not in ik_masked_joints:
            ik_masked_joints.append(self.manual_joystick_joint)
        self.ik_robot = PlacoIKSolver(
            self.cfg.urdf_path,
            self.cfg.controlled_joints,
            self.cfg.left_ee_body,
            self.cfg.right_ee_body,
            dt=self.cfg.ik_dt,
            regularization_weight=self.cfg.placo_regularization_weight,
            body_joints=self.cfg.body_joints,
            body_regularization_weight=self.cfg.body_regularization_weight,
            body_joint_regularization_weights=self.cfg.body_joint_regularization_weights,
            masked_joints=ik_masked_joints,
            max_joint_step=self.cfg.max_joint_step,
            reset_joint_positions=self.cfg.reset_joint_positions,
        )
        self.viz_robot = MujocoVisualizer(
            self.cfg.urdf_path,
            self.cfg.controlled_joints,
            floating_base=self.cfg.floating_base,
            floating_base_child_link=self.cfg.floating_base_child_link or None,
        )
        self.rnea = self._make_rnea_feedforward()
        self.publisher = make_publisher(mode, self.cfg, net)
        # Recording config overrides, then wrap the publisher.
        if record_dir is not None:
            self.cfg.record_dir = record_dir
        if record_task_name is not None:
            self.cfg.record_task_name = record_task_name
        if record_fps is not None:
            self.cfg.record_fps = record_fps
        if record_images is not None:
            self.cfg.record_images = record_images
        if image_server_host is not None:
            self.cfg.image_server_host = image_server_host
        if image_server_port is not None:
            self.cfg.image_server_port = image_server_port
        if record_tts is not None:
            self.cfg.record_tts = record_tts
        self.recorder = None
        if record:
            self.cfg.record_enabled = True
            from .recorder import RecordingPublisher

            self.publisher = RecordingPublisher(self.publisher, self.cfg)
            self.recorder = self.publisher
        self.reader = DirectVRReader(
            self.cfg.direct_vr.tcp_port,
            self.cfg.direct_vr.broadcast_port,
            head_pose_index=self.cfg.direct_vr.head_pose_index,
        )
        self.left_ref: tuple[np.ndarray, np.ndarray] | None = None
        self.right_ref: tuple[np.ndarray, np.ndarray] | None = None
        self.left_robot_ref = self.ik_robot.frame_pose(self.cfg.left_ee_body)
        self.right_robot_ref = self.ik_robot.frame_pose(self.cfg.right_ee_body)
        self.base_cmd = [0.0] * (3 + len(self.cfg.base_cmd_joints))
        self.head_ref_z: float | None = None
        self.head_delta_z = 0.0
        self.head_ref_quat: np.ndarray | None = None
        self.head_joint_ref_q: dict[str, float] = {}
        self.lz_height = float(np.clip(self.cfg.lz_height_offset, self.cfg.lz_joint_min, self.cfg.lz_joint_max))
        self.manual_joint_value = self._manual_joystick_reset_value()
        self._last_debug_time = 0.0
        self._last_a_pressed = False
        self._last_y_pressed = False
        self._last_x_pressed = False
        self._last_b_pressed = False
        self._stop_requested = threading.Event()
        self._sync_viewer_from_ik()
        self._update_base_cmd_joints()

    def request_stop(self) -> None:
        if not self._stop_requested.is_set():
            print("\n[teleop] Ctrl+C received, stopping...")
        self._stop_requested.set()

    @staticmethod
    def _normalize_control_mode(value: str) -> str:
        mode = str(value).strip().lower()
        aliases = {"1": "wbc", "wbc": "wbc", "2": "manual", "manual": "manual"}
        if mode not in aliases:
            raise ValueError("teleop_control_mode must be 'wbc'/'1' or 'manual'/'2'")
        return aliases[mode]

    def _reset_joint_targets(self) -> dict[str, float]:
        targets = {name: float(self.cfg.reset_joint_positions.get(name, 0.0)) for name in self.cfg.controlled_joints}
        for name in self.cfg.lz_height_joints:
            targets[name] = float(np.clip(self.cfg.lz_height_offset, self.cfg.lz_joint_min, self.cfg.lz_joint_max))
        return targets

    def _resolve_manual_joystick_joint(self) -> str:
        """Name of the joint owned by the joystick, or "" when not applicable."""
        name = str(self.cfg.manual_joystick_joint).strip()
        if not name or self.cfg.teleop_control_mode != "manual":
            return ""
        if name not in self.cfg.controlled_joints:
            raise ValueError(f"manual_joystick_joint {name!r} not in controlled_joints")
        return name

    def _manual_joystick_reset_value(self) -> float:
        if not self.manual_joystick_joint:
            return 0.0
        return float(np.clip(
            self.cfg.reset_joint_positions.get(self.manual_joystick_joint, 0.0),
            self.cfg.manual_joystick_min,
            self.cfg.manual_joystick_max,
        ))

    def _sync_manual_joystick_joint(self) -> None:
        if not self.manual_joystick_joint:
            return
        self.ik_robot.set_joint_positions({self.manual_joystick_joint: float(self.manual_joint_value)})

    def _update_manual_joystick_joint(self, buttons) -> None:
        """Integrate one joystick axis into a joint target; releasing holds it.

        Matches the reference waist_joystick behaviour: axis values outside the
        deadband are integrated at `manual_joystick_gain` (rad/s) and clamped to
        [manual_joystick_min, manual_joystick_max]. The joint is already masked
        out of IK, so writing it here is not overwritten by the next solve.
        """
        if not self.manual_joystick_joint:
            return
        axis = buttons.axis_l if str(self.cfg.manual_joystick_side).lower() == "left" else buttons.axis_r
        value = float(axis[0] if str(self.cfg.manual_joystick_axis).lower() == "x" else axis[1])
        deadband = abs(float(self.cfg.manual_joystick_deadband))
        if abs(value) <= deadband:
            value = 0.0
        period = 1.0 / max(self.cfg.fps, 1)
        self.manual_joint_value = float(np.clip(
            self.manual_joint_value + value * self.cfg.manual_joystick_gain * period,
            self.cfg.manual_joystick_min,
            self.cfg.manual_joystick_max,
        ))
        self._sync_manual_joystick_joint()

    def _sync_viewer_from_ik(self) -> None:
        q = self.ik_robot.joint_qpos(self.cfg.controlled_joints)
        self.viz_robot.set_joint_qpos(self.cfg.controlled_joints, q)

    def _sync_lz_height(self) -> None:
        if not self.cfg.lz_height_joints:
            return
        value = float(np.clip(self.lz_height, self.cfg.lz_joint_min, self.cfg.lz_joint_max))
        self.ik_robot.set_joint_positions({name: value for name in self.cfg.lz_height_joints})

    def _debug_state(self, active_left: bool, active_right: bool, solved: bool) -> None:
        now = time.perf_counter()
        if now - self._last_debug_time < 1.0:
            return
        self._last_debug_time = now
        buttons = self.reader.latest_buttons()
        arm_q = self.ik_robot.arm_command_14(self.cfg.left_arm_joints, self.cfg.right_arm_joints)
        body_q = dict(zip(self.cfg.base_cmd_joints, self.base_cmd[3:]))
        print(
            "[teleop] "
            f"body={self.reader.latest_body().available} "
            f"grip=({buttons.left_grip:.2f},{buttons.right_grip:.2f}) "
            f"active=({active_left},{active_right}) "
            f"solved={solved} "
            f"arm_norm={float(np.linalg.norm(arm_q)):.4f} "
            f"body_q={{ {', '.join(f'{k}={v:.4f}' for k, v in body_q.items())} }}"
        )

    def _update_base_cmd_joints(self) -> None:
        # base_cmd = [vx, 0.0, vyaw] + cfg.base_cmd_joints qpos values.
        if self.cfg.base_cmd_joints:
            self.base_cmd[3:] = self.ik_robot.joint_qpos(self.cfg.base_cmd_joints).tolist()

    def _make_rnea_feedforward(self) -> PinocchioRNEAFeedforward | None:
        if not self.cfg.rnea_feedforward_joints:
            return None
        command_joints = list(dict.fromkeys(self.cfg.controlled_joints + self.cfg.rnea_feedforward_joints))
        try:
            return PinocchioRNEAFeedforward(
                self.cfg.urdf_path,
                command_joints=command_joints,
                output_joints=self.cfg.rnea_feedforward_joints,
            )
        except Exception as exc:
            print(f"[teleop] warning: Pinocchio RNEA feedforward disabled: {exc}")
            return None

    def _compute_arm_tau(self, arm_joints: list[str], body_positions: dict[str, float]) -> np.ndarray:
        if self.rnea is None:
            return np.zeros(len(arm_joints), dtype=float)
        joint_positions = dict(zip(self.cfg.controlled_joints, self.ik_robot.joint_qpos(self.cfg.controlled_joints)))
        tau_map = self.rnea.compute(joint_positions)
        arm_tau = np.asarray([tau_map.get(name, 0.0) for name in arm_joints], dtype=float)
        if self.cfg.g1d_waist_pitch_joint:
            waist_pitch_tau = tau_map.get(self.cfg.g1d_waist_pitch_joint)
            if waist_pitch_tau is not None and self.cfg.g1d_waist_pitch_joint in body_positions:
                arm_tau = np.concatenate([arm_tau, np.asarray([waist_pitch_tau], dtype=float)])
        return arm_tau

    def _publish_current_command(self) -> None:
        arm_joints = self.cfg.left_arm_joints + self.cfg.right_arm_joints
        arm_q = self.ik_robot.arm_command_14(self.cfg.left_arm_joints, self.cfg.right_arm_joints)
        body_positions = dict(zip(self.cfg.base_cmd_joints, self.base_cmd[3:]))
        arm_tau = self._compute_arm_tau(arm_joints, body_positions)
        gripper_joints = self.cfg.left_gripper_joints + self.cfg.right_gripper_joints
        gripper_q = self.ik_robot.joint_qpos(gripper_joints) if gripper_joints else None
        head_q = self.ik_robot.joint_qpos(self.cfg.head_joints) if self.cfg.head_joints else None
        self.publisher.publish(arm_q=arm_q, arm_tau=arm_tau, base_cmd=self.base_cmd, gripper_q=gripper_q, head_q=head_q, lift_position=self.lz_height)

    def _publish_stop_command(self) -> None:
        self.base_cmd[0:3] = [0.0, 0.0, 0.0]
        self._update_base_cmd_joints()
        try:
            self._publish_current_command()
        except Exception as exc:
            print(f"[teleop] failed to publish stop command: {exc}")

    def _update_gripper_joints(self, buttons) -> None:
        span = self.cfg.gripper_closed_position - self.cfg.gripper_open_position
        values: dict[str, float] = {}
        left_q = float(self.cfg.gripper_open_position + np.clip(buttons.left_trigger, 0.0, 1.0) * span)
        right_q = float(self.cfg.gripper_open_position + np.clip(buttons.right_trigger, 0.0, 1.0) * span)
        values.update({name: left_q for name in self.cfg.left_gripper_joints})
        values.update({name: right_q for name in self.cfg.right_gripper_joints})
        if values:
            self.ik_robot.set_joint_positions(values)

    def reset_grip_references(self, *, quiet: bool = False) -> None:
        self.left_ref = None
        self.right_ref = None
        self.left_robot_ref = self.ik_robot.frame_pose(self.cfg.left_ee_body)
        self.right_robot_ref = self.ik_robot.frame_pose(self.cfg.right_ee_body)
        if not quiet:
            print("[teleop] reset gripper IK references")

    def _interpolate_to_reset(self, viewer=None) -> None:
        targets = self._reset_joint_targets()
        start = np.array(self.ik_robot.joint_qpos(self.cfg.controlled_joints), dtype=float)
        goal = np.asarray([targets[name] for name in self.cfg.controlled_joints], dtype=float)
        # The lift is not interpolated: snap its start to the goal so it moves
        # immediately, since the hardware itself is already slow.
        lz_joints = set(self.cfg.lz_height_joints)
        for idx, name in enumerate(self.cfg.controlled_joints):
            if name in lz_joints:
                start[idx] = goal[idx]
        steps = max(1, int(round(self.cfg.reset_duration * max(self.cfg.fps, 1))))
        period = 1.0 / max(self.cfg.fps, 1)
        for i in range(1, steps + 1):
            if self._stop_requested.is_set():
                break
            alpha = i / steps
            q = start + (goal - start) * alpha
            self.ik_robot.set_joint_positions(dict(zip(self.cfg.controlled_joints, q)))
            self._sync_viewer_from_ik()
            self._update_base_cmd_joints()
            self._publish_current_command()
            if viewer is not None:
                viewer.sync()
            if self._stop_requested.wait(period):
                break
        self.lz_height = float(np.clip(self.cfg.lz_height_offset, self.cfg.lz_joint_min, self.cfg.lz_joint_max))
        self.manual_joint_value = self._manual_joystick_reset_value()
        self.reset_grip_references(quiet=True)

    def reset_all_state(self, viewer=None, reason: str = "A pressed") -> None:
        self.base_cmd = [0.0] * (3 + len(self.cfg.base_cmd_joints))
        self.head_ref_z = None
        self.head_delta_z = 0.0
        self.head_ref_quat = None
        self.head_joint_ref_q = {}
        self.lz_height = float(np.clip(self.cfg.lz_height_offset, self.cfg.lz_joint_min, self.cfg.lz_joint_max))
        self.left_ref = None
        self.right_ref = None
        self._interpolate_to_reset(viewer)
        if self._stop_requested.is_set():
            return
        self._sync_viewer_from_ik()
        self._update_base_cmd_joints()
        self._publish_current_command()
        print(f"[teleop] {reason}: interpolated reset pose")

    def _handle_record_buttons(self, buttons) -> None:
        # VR-Y: start/save (toggle). VR-X: delete the last saved episode.
        # VR-B: pause/resume writing frames when the recorder supports it.
        if self.recorder is not None:
            if buttons.y_pressed and not self._last_y_pressed:
                self.recorder.toggle_episode()
            if buttons.x_pressed and not self._last_x_pressed:
                self.recorder.delete_last_episode()
            if buttons.b_pressed and not self._last_b_pressed:
                toggle_pause = getattr(self.recorder, "toggle_pause", None)
                if callable(toggle_pause):
                    toggle_pause()
        self._last_y_pressed = buttons.y_pressed
        self._last_x_pressed = buttons.x_pressed
        self._last_b_pressed = buttons.b_pressed

    def _update_velocity_cmd(self) -> None:
        buttons = self.reader.latest_buttons()
        forward_axis = buttons.axis_l[1]
        yaw_axis = buttons.axis_l[0]
        self.base_cmd[0] = float(forward_axis * 0.5)
        self.base_cmd[1] = 0.0
        self.base_cmd[2] = float(-yaw_axis * 1.5)

    def _update_manual_lz_height(self, buttons) -> None:
        if not self.cfg.lz_height_joints:
            return
        period = 1.0 / max(self.cfg.fps, 1)
        lz_axis = buttons.axis_r[1]
        # Position target: the stick integrates a target height and releasing it
        # holds that target. The publisher closes the loop with its velocity
        # interface.
        self.lz_height = float(np.clip(
            self.lz_height + lz_axis * self.cfg.manual_lz_speed * period,
            self.cfg.lz_joint_min,
            self.cfg.lz_joint_max,
        ))
        self._sync_lz_height()

    def _update_head_height(self, body) -> None:
        if body.head_pose is None:
            return
        head_pos_pico, _ = pose7_to_pos_quat_xyzw(body.head_pose)
        head_pos, _ = transform_pico_pose_to_robot(head_pos_pico, np.array([0.0, 0.0, 0.0, 1.0]))
        if self.head_ref_z is None:
            self.head_ref_z = float(head_pos[2])
        self.head_delta_z = float(head_pos[2] - self.head_ref_z)
        self.lz_height = float(np.clip(self.cfg.lz_height_offset + self.head_delta_z * self.cfg.lz_height_scale, self.cfg.lz_joint_min, self.cfg.lz_joint_max))
        self._sync_lz_height()

    def _clip_head_joint(self, name: str, value: float) -> float:
        limit = self.cfg.head_joint_limits.get(name)
        if limit is None or len(limit) != 2:
            return float(value)
        return float(np.clip(value, float(limit[0]), float(limit[1])))

    def _reset_head_motion_reference(self) -> None:
        self.head_ref_quat = None
        self.head_joint_ref_q = {}

    def _update_head_motion(self, body, active: bool) -> None:
        if not self.cfg.head_joints:
            return
        if not active:
            self._reset_head_motion_reference()
            return
        if body.head_pose is None:
            return
        _, head_quat_pico = pose7_to_pos_quat_xyzw(body.head_pose)
        _, head_quat = transform_pico_pose_to_robot(np.zeros(3), head_quat_pico)
        if self.head_ref_quat is None:
            self.head_ref_quat = head_quat
            self.head_joint_ref_q = dict(zip(self.cfg.head_joints, self.ik_robot.joint_qpos(self.cfg.head_joints)))
            return
        rel_rot = Rotation.from_quat(self.head_ref_quat).inv() * Rotation.from_quat(head_quat)
        rel_yaw, rel_pitch, _ = rel_rot.as_euler("zyx", degrees=False)
        values: dict[str, float] = {}
        if self.cfg.head_yaw_joint:
            yaw = self.head_joint_ref_q.get(self.cfg.head_yaw_joint, 0.0) + rel_yaw * self.cfg.head_yaw_scale
            values[self.cfg.head_yaw_joint] = self._clip_head_joint(self.cfg.head_yaw_joint, yaw)
        if self.cfg.head_pitch_joint:
            pitch = self.head_joint_ref_q.get(self.cfg.head_pitch_joint, 0.0) + rel_pitch * self.cfg.head_pitch_scale
            values[self.cfg.head_pitch_joint] = self._clip_head_joint(self.cfg.head_pitch_joint, pitch)
        if values:
            self.ik_robot.set_joint_positions(values)

    def _limit_delta(self, delta: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(delta))
        if norm > self.cfg.max_hand_delta > 0.0:
            return delta * (self.cfg.max_hand_delta / norm)
        return delta

    def targets_from_vr(self) -> tuple[list[IKTarget], bool, bool]:
        self._update_velocity_cmd()
        body = self.reader.latest_body()
        buttons = self.reader.latest_buttons()
        threshold = self.cfg.activation_grip_threshold
        active_left = buttons.left_grip > threshold
        active_right = buttons.right_grip > threshold
        if self.cfg.teleop_control_mode == "manual":
            self._update_manual_lz_height(buttons)
            self._update_manual_joystick_joint(buttons)
        if not body.available:
            return [], active_left, active_right
        if not active_left and not active_right:
            self.head_ref_z = None
            self.head_delta_z = 0.0
            self._reset_head_motion_reference()
            self.left_ref = None
            self.right_ref = None
            self.left_robot_ref = self.ik_robot.frame_pose(self.cfg.left_ee_body)
            self.right_robot_ref = self.ik_robot.frame_pose(self.cfg.right_ee_body)
            return [], active_left, active_right
        if self.cfg.teleop_control_mode == "wbc":
            self._update_head_height(body)

        targets: list[IKTarget] = []
        if active_left:
            left_pos_pico, left_quat_pico = pose7_to_pos_quat_xyzw(body.poses[self.cfg.direct_vr.left_pose_index])
            left_pos, left_quat = transform_pico_pose_to_robot(left_pos_pico, left_quat_pico)
            if self.left_ref is None:
                self.left_ref = (left_pos, left_quat)
                self.left_robot_ref = self.ik_robot.frame_pose(self.cfg.left_ee_body)
            else:
                left_rel = relative_pose(*self.left_ref, left_pos, left_quat)
                left_target = apply_relative_pose(
                    self.left_robot_ref[0],
                    self.left_robot_ref[1],
                    self._limit_delta(left_rel[0]),
                    left_rel[1],
                    scale=self.scale,
                )
                targets.append(IKTarget(self.cfg.left_ee_body, left_target[0], left_target[1]))
        else:
            self.left_ref = None
            self.left_robot_ref = self.ik_robot.frame_pose(self.cfg.left_ee_body)

        if active_right:
            right_pos_pico, right_quat_pico = pose7_to_pos_quat_xyzw(body.poses[self.cfg.direct_vr.right_pose_index])
            right_pos, right_quat = transform_pico_pose_to_robot(right_pos_pico, right_quat_pico)
            if self.right_ref is None:
                self.right_ref = (right_pos, right_quat)
                self.right_robot_ref = self.ik_robot.frame_pose(self.cfg.right_ee_body)
            else:
                right_rel = relative_pose(*self.right_ref, right_pos, right_quat)
                right_target = apply_relative_pose(
                    self.right_robot_ref[0],
                    self.right_robot_ref[1],
                    self._limit_delta(right_rel[0]),
                    right_rel[1],
                    scale=self.scale,
                )
                targets.append(IKTarget(self.cfg.right_ee_body, right_target[0], right_target[1]))
        else:
            self.right_ref = None
            self.right_robot_ref = self.ik_robot.frame_pose(self.cfg.right_ee_body)

        return targets, active_left, active_right

    def step(self, viewer=None) -> None:
        buttons = self.reader.latest_buttons()
        if buttons.a_pressed and not self._last_a_pressed:
            self.reset_all_state(viewer, reason="A pressed")
            self._last_a_pressed = True
            self._debug_state(False, False, False)
            return
        self._last_a_pressed = buttons.a_pressed
        self._handle_record_buttons(buttons)

        targets, active_left, active_right = self.targets_from_vr()
        solved = False
        if targets:
            self.ik_robot.solve_ik(
                targets,
                active_left=active_left,
                active_right=active_right,
                left_ee_frame=self.cfg.left_ee_body,
                right_ee_frame=self.cfg.right_ee_body,
                iters=self.cfg.ik_iters,
            )
            solved = True
        self._update_head_motion(self.reader.latest_body(), active_left or active_right)
        self._update_gripper_joints(buttons)
        self._sync_viewer_from_ik()
        self.viz_robot.integrate_base_velocity(self.base_cmd[0], self.base_cmd[1], self.base_cmd[2], 1.0 / max(self.cfg.fps, 1))
        self._update_base_cmd_joints()
        self._debug_state(active_left, active_right, solved)
        self._publish_current_command()

    def run(self) -> None:
        self.reader.start()
        period = 1.0 / max(self.cfg.fps, 1)
        previous_sigint = signal.getsignal(signal.SIGINT)
        viewer = None

        def handle_sigint(signum, frame) -> None:
            if self._stop_requested.is_set():
                raise KeyboardInterrupt
            self.request_stop()

        signal.signal(signal.SIGINT, handle_sigint)
        print(f"[teleop] robot={self.robot}, mode={self.mode}, control_mode={self.cfg.teleop_control_mode}, urdf={self.cfg.urdf_path}")
        print("[teleop] hold left/right gripper to enable Placo IK; VR-A reset; VR-Y record/save, VR-X delete last; Ctrl+C to exit")
        try:
            viewer = mujoco.viewer.launch_passive(self.viz_robot.model, self.viz_robot.data)
            self.reset_all_state(viewer, reason="startup")
            while viewer.is_running() and not self._stop_requested.is_set():
                start = time.perf_counter()
                self.step(viewer)
                viewer.sync()
                elapsed = time.perf_counter() - start
                if elapsed < period and self._stop_requested.wait(period - elapsed):
                    break
        except KeyboardInterrupt:
            self.request_stop()
        finally:
            ctrl_c_exit = self._stop_requested.is_set()
            signal.signal(signal.SIGINT, previous_sigint)
            self._stop_requested.set()
            self._publish_stop_command()
            self.reader.stop()
            close = getattr(self.publisher, "close", None)
            if callable(close):
                close()
            if viewer is not None and not ctrl_c_exit:
                viewer.close()
            print("[teleop] stopped", flush=True)
            if ctrl_c_exit:
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(0)


def main(
    robot: str = "g1_d",
    config: str | None = None,
    common_config: str = str(CONFIG_DIR / "common.yaml"),
    mode: str = "sim",
    net: str = "eno2",
    scale: float = 1.0,
    body_regularization_weight: float | None = None,
    placo_regularization_weight: float | None = None,
    lz_mt_weight: float | None = None,
    lz_it_weight: float | None = None,
    yaw_joint_weight: float | None = None,
    torso_joint_weight: float | None = None,
    max_hand_delta: float | None = None,
    max_joint_step: float | None = None,
    teleop_control_mode: str | None = None,
    record: bool = False,
    record_dir: str | None = None,
    record_task_name: str | None = None,
    record_fps: int | None = None,
    record_images: bool | None = None,
    image_server_host: str | None = None,
    image_server_port: int | None = None,
    record_tts: bool | None = None,
) -> None:
    """Run teleop for a configured robot.

    robot:
      Robot config name under config/<robot>.yaml (G1-D: g1_d).
    config:
      Optional robot YAML override. If omitted, resolved from --robot.
    common_config:
      Shared YAML merged before the robot YAML.
    mode:
      sim   - MuJoCo viewer + print sampled commands.
      robot - MuJoCo viewer + publish using the selected hardware_interface.
    teleop_control_mode:
      wbc/1    - Headset height delta controls the lift; waist joins whole-body IK.
      manual/2 - Right stick Y controls the lift, or manual_joystick_joint.
    body_regularization_weight:
      Default zero-posture regularization for body joints. Larger means body joints move less.
    placo_regularization_weight:
      Zero-posture regularization for all controlled joints.
    lz_mt_weight / lz_it_weight / yaw_joint_weight / torso_joint_weight:
      Per-body-joint zero-posture regularization overrides. Only applies to joints present in the selected robot YAML.
    max_hand_delta / max_joint_step:
      Safety clamps for unreachable targets and sudden IK jumps.
    record:
      Record commands, measured state, images and gripper values to MCAP under
      <record_dir>/<task>/.
    record_dir / record_task_name / record_fps / record_images / image_server_host:
      Output directory, task name, recording rate, whether to record images, and
      the image_server IP. All override the YAML.
    image_server_port:
      image_server port (overrides YAML, default 5555).
    record_tts:
      Announce record start/save/delete through the robot speaker (overrides
      YAML; falls back to printing when no speaker is available).
    """
    if mode not in {"sim", "robot"}:
        raise ValueError("mode must be 'sim' or 'robot'")
    config_path = config if config is not None else str(config_path_for_robot(robot))
    TeleopRunner(
        config_path=config_path,
        common_config_path=common_config,
        robot=robot,
        mode=mode,
        net=net,
        scale=scale,
        body_regularization_weight=body_regularization_weight,
        placo_regularization_weight=placo_regularization_weight,
        lz_mt_weight=lz_mt_weight,
        lz_it_weight=lz_it_weight,
        yaw_joint_weight=yaw_joint_weight,
        torso_joint_weight=torso_joint_weight,
        max_hand_delta=max_hand_delta,
        max_joint_step=max_joint_step,
        teleop_control_mode=teleop_control_mode,
        record=record,
        record_dir=record_dir,
        record_task_name=record_task_name,
        record_fps=record_fps,
        record_images=record_images,
        image_server_host=image_server_host,
        image_server_port=image_server_port,
        record_tts=record_tts,
    ).run()


def cli() -> None:
    tyro.cli(main)


if __name__ == "__main__":
    cli()
