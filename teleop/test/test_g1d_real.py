#!/usr/bin/env python3
"""Replay a G1-D CSV trajectory to the real robot.

Dry-run by default and never touches hardware; pass --publish explicitly to
send commands through the G1-D DDS interface.

Examples:
  python test/test_g1d_real.py --net eth0 --dry-run
  python test/test_g1d_real.py --net eth0 --publish
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import signal
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "robot_teleop" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robot_teleop.config import CONFIG_DIR, config_path_for_robot, load_config
from robot_teleop.feedforward import PinocchioRNEAFeedforward
from robot_teleop.publisher import PrintPublisher, make_publisher


DEFAULT_CSV = ROOT / "test" / "csv" / "eval_action_episode_0.csv"

CSV_TO_G1D_JOINTS = {
    "robot_29dof.waist_yaw": "torso_Joint",
    "robot_29dof.waist_pitch": "Yaw_Joint",
    "robot_29dof.left_arm_shoulder_pitch": "left_shoulder_pitch_joint",
    "robot_29dof.left_arm_shoulder_roll": "left_shoulder_roll_joint",
    "robot_29dof.left_arm_shoulder_yaw": "left_shoulder_yaw_joint",
    "robot_29dof.left_arm_elbow": "left_elbow_joint",
    "robot_29dof.left_arm_wrist_roll": "left_wrist_roll_joint",
    "robot_29dof.left_arm_wrist_pitch": "left_wrist_pitch_joint",
    "robot_29dof.left_arm_wrist_yaw": "left_wrist_yaw_joint",
    "robot_29dof.right_arm_shoulder_pitch": "right_shoulder_pitch_joint",
    "robot_29dof.right_arm_shoulder_roll": "right_shoulder_roll_joint",
    "robot_29dof.right_arm_shoulder_yaw": "right_shoulder_yaw_joint",
    "robot_29dof.right_arm_elbow": "right_elbow_joint",
    "robot_29dof.right_arm_wrist_roll": "right_wrist_roll_joint",
    "robot_29dof.right_arm_wrist_pitch": "right_wrist_pitch_joint",
    "robot_29dof.right_arm_wrist_yaw": "right_wrist_yaw_joint",
}


class CsvRealReplay:
    def __init__(
        self,
        csv_path: Path,
        hz: float,
        net: str,
        publish: bool,
        robot: str,
        config: str | None,
        common_config: str,
        gripper_input_min: float,
        gripper_input_max: float,
        gripper_mode: str,
        zero_base_velocity: bool,
        use_height: bool,
        warmup_seconds: float,
        loop: bool,
        rnea: bool,
    ) -> None:
        self.csv_path = csv_path.expanduser().resolve()
        self.hz = hz
        self.period = 1.0 / hz
        self.net = net
        self.publish_to_robot = bool(publish)
        self.zero_base_velocity = zero_base_velocity
        self.use_height = use_height
        self.warmup_seconds = max(warmup_seconds, 0.0)
        self.loop = loop
        self.stop = False

        config_path = Path(config).expanduser().resolve() if config else config_path_for_robot(robot)
        self.cfg = load_config(config_path, common_config)
        self.cfg.hardware_interface = "g1_d_dds"

        self.gripper_input_min = gripper_input_min
        self.gripper_input_max = gripper_input_max
        self.gripper_mode = gripper_mode
        self.rows = self._load_rows(self.csv_path)
        self.arm_joints = self.cfg.left_arm_joints + self.cfg.right_arm_joints
        self.gripper_joints = self.cfg.left_gripper_joints + self.cfg.right_gripper_joints
        self.publisher = make_publisher("robot", self.cfg, net) if self.publish_to_robot else PrintPublisher(every=max(1, int(round(hz))))
        self.ff = None
        if rnea:
            self.ff = PinocchioRNEAFeedforward(
                self.cfg.urdf_path,
                command_joints=self.cfg.controlled_joints,
                output_joints=self.cfg.rnea_feedforward_joints,
            )

    @staticmethod
    def _load_rows(path: Path) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                row: dict[str, float] = {}
                for key, value in raw.items():
                    if key is None or value is None or value == "":
                        continue
                    row[key] = float(value)
                rows.append(row)
        if not rows:
            raise ValueError(f"CSV has no rows: {path}")
        return rows

    def _on_sigint(self, _signum, _frame) -> None:
        self.stop = True
        print("\n[real-csv] Ctrl+C, stopping...")

    def _map_gripper_value(self, raw: float) -> float:
        if self.gripper_mode == "direct":
            return float(raw)
        span = max(self.gripper_input_max - self.gripper_input_min, 1e-9)
        alpha = float(np.clip((float(raw) - self.gripper_input_min) / span, 0.0, 1.0))
        if self.gripper_mode == "motor":
            # Dex1 hardware and CSV: 0 is closed, max is open.
            # URDF: open_position is open, closed_position is closed.
            return float(self.cfg.gripper_closed_position + alpha * (self.cfg.gripper_open_position - self.cfg.gripper_closed_position))
        # closed_high: 0 is open, max is closed.
        return float(self.cfg.gripper_open_position + alpha * (self.cfg.gripper_closed_position - self.cfg.gripper_open_position))

    def _joint_map_from_row(self, row: dict[str, float]) -> dict[str, float]:
        joint_map: dict[str, float] = {}
        for csv_name, joint_name in CSV_TO_G1D_JOINTS.items():
            if csv_name in row:
                joint_map[joint_name] = float(row[csv_name])
        return joint_map

    def _arm_q(self, joint_map: dict[str, float]) -> np.ndarray:
        return np.asarray(
            [float(joint_map.get(name, self.cfg.reset_joint_positions.get(name, 0.0))) for name in self.arm_joints],
            dtype=float,
        )

    def _gripper_q(self, row: dict[str, float]) -> np.ndarray | None:
        if not self.gripper_joints:
            return None
        left_q = self._map_gripper_value(row.get("gripper.left_gripper_gripper_pos", 0.0))
        right_q = self._map_gripper_value(row.get("gripper.right_gripper_gripper_pos", 0.0))
        values = [left_q] * len(self.cfg.left_gripper_joints) + [right_q] * len(self.cfg.right_gripper_joints)
        return np.asarray(values, dtype=float)

    def _base_cmd(self, row: dict[str, float], joint_map: dict[str, float]) -> list[float]:
        vx = 0.0 if self.zero_base_velocity else float(row.get("base_command.base_command_vx", 0.0))
        vyaw = 0.0 if self.zero_base_velocity else float(row.get("base_command.base_command_angle_z", 0.0))
        body: dict[str, float] = {}
        if self.use_height:
            h = float(row.get("base_command.base_command_height", self.cfg.reset_joint_positions.get("LZ_mt_Joint", 0.0)))
            # The CSV height looks like a body height (around 0.74 m) rather
            # than a lift joint value, so clamp it into the joint range.
            h = float(np.clip(h, self.cfg.lz_joint_min, self.cfg.lz_joint_max))
            for name in self.cfg.g1d_lift_joints:
                body[name] = h
        else:
            for name in self.cfg.g1d_lift_joints:
                body[name] = float(self.cfg.reset_joint_positions.get(name, 0.0))
        if self.cfg.g1d_waist_pitch_joint:
            body[self.cfg.g1d_waist_pitch_joint] = float(joint_map.get(self.cfg.g1d_waist_pitch_joint, 0.0))
        if self.cfg.g1d_waist_yaw_joint:
            body[self.cfg.g1d_waist_yaw_joint] = float(joint_map.get(self.cfg.g1d_waist_yaw_joint, 0.0))
        return [vx, 0.0, vyaw] + [float(body.get(name, 0.0)) for name in self.cfg.base_cmd_joints]

    def _arm_tau(self, joint_map: dict[str, float]) -> np.ndarray:
        tau = np.zeros(len(self.arm_joints) + 1, dtype=float)  # 14 arm + waist pitch ff slot.
        if self.ff is None:
            return tau
        positions = dict(self.cfg.reset_joint_positions)
        positions.update(joint_map)
        try:
            tau_map = self.ff.compute(positions)
        except Exception as exc:
            print(f"[real-csv] warning: RNEA failed, using zero tau: {exc}")
            return tau
        for i, name in enumerate(self.arm_joints):
            tau[i] = float(tau_map.get(name, 0.0))
        if self.cfg.g1d_waist_pitch_joint:
            tau[len(self.arm_joints)] = float(tau_map.get(self.cfg.g1d_waist_pitch_joint, 0.0))
        return tau

    def _publish_row(self, row: dict[str, float]) -> None:
        joint_map = self._joint_map_from_row(row)
        arm_q = self._arm_q(joint_map)
        arm_tau = self._arm_tau(joint_map)
        base_cmd = self._base_cmd(row, joint_map)
        gripper_q = self._gripper_q(row)
        lift_position = base_cmd[3] if self.use_height and len(base_cmd) > 3 else None
        self.publisher.publish(
            arm_q=arm_q,
            arm_tau=arm_tau,
            base_cmd=base_cmd,
            gripper_q=gripper_q,
            lift_position=lift_position,
        )

    def _warmup(self) -> None:
        if not self.rows or self.warmup_seconds <= 0.0:
            return
        first = self.rows[0]
        joint_map = self._joint_map_from_row(first)
        target = self._arm_q(joint_map)
        start = target.copy()
        if self.publish_to_robot:
            try:
                snap = self.publisher.state_snapshot()
                motor_q = snap.get("motor_q")
                if motor_q and len(motor_q) >= 29:
                    start = np.asarray([float(motor_q[i]) for i in range(15, 29)], dtype=float)
            except Exception:
                pass
        arm_tau = self._arm_tau(joint_map)
        base_cmd = self._base_cmd(first, joint_map)
        base_cmd[0:3] = [0.0, 0.0, 0.0]
        gripper_q = self._gripper_q(first)
        lift_position = base_cmd[3] if self.use_height and len(base_cmd) > 3 else None
        steps = max(1, int(round(self.warmup_seconds / self.period)))
        print(f"[real-csv] warmup {self.warmup_seconds:.2f}s, steps={steps}")
        for i in range(1, steps + 1):
            if self.stop:
                return
            alpha = i / steps
            arm_q = start + (target - start) * alpha
            self.publisher.publish(arm_q, arm_tau, base_cmd, gripper_q=gripper_q, lift_position=lift_position)
            time.sleep(self.period)

    def run(self) -> None:
        signal.signal(signal.SIGINT, self._on_sigint)
        mode = "PUBLISH" if self.publish_to_robot else "DRY-RUN"
        print(f"[real-csv] mode={mode}, csv={self.csv_path}, frames={len(self.rows)}, hz={self.hz}")
        print(f"[real-csv] net={self.net or '<none>'}, zero_base_velocity={self.zero_base_velocity}, use_height={self.use_height}")
        if self.publish_to_robot:
            print("[real-csv] WARNING: sending commands to real G1-D. Keep robot clear.")
        self._warmup()
        try:
            while not self.stop:
                next_at = time.monotonic()
                for i, row in enumerate(self.rows):
                    if self.stop:
                        break
                    self._publish_row(row)
                    if i % int(max(self.hz, 1.0)) == 0:
                        print(f"[real-csv] frame={i}/{len(self.rows)}")
                    next_at += self.period
                    delay = next_at - time.monotonic()
                    if delay > 0.0:
                        time.sleep(delay)
                if not self.loop:
                    break
        finally:
            self._safe_stop()

    def _safe_stop(self) -> None:
        try:
            if self.rows:
                joint_map = self._joint_map_from_row(self.rows[-1])
                arm_q = self._arm_q(joint_map)
                arm_tau = np.zeros(len(self.arm_joints) + 1, dtype=float)
                base_cmd = self._base_cmd(self.rows[-1], joint_map)
                base_cmd[0:3] = [0.0, 0.0, 0.0]
                gripper_q = self._gripper_q(self.rows[-1])
                for _ in range(max(1, int(round(0.3 / self.period)))):
                    self.publisher.publish(arm_q, arm_tau, base_cmd, gripper_q=gripper_q, lift_position=None)
                    time.sleep(self.period)
        finally:
            self.publisher.close()
            print("[real-csv] stopped; base/lift zeroed by publisher.close()")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay eval CSV to real G1-D using robot_teleop G1-D DDS interface.")
    parser.add_argument("csv", nargs="?", default=str(DEFAULT_CSV), help="CSV path to replay.")
    parser.add_argument("--net", default="", help="Network interface, e.g. eno2.")
    parser.add_argument("--publish", action="store_true", help="Actually send commands to the robot. Default is dry-run.")
    parser.add_argument("--dry-run", action="store_true", help="Dry-run only; same as omitting --publish.")
    parser.add_argument("--robot", default="g1_d", help="Robot config name, default g1_d.")
    parser.add_argument("--config", default=None, help="Optional robot yaml path.")
    parser.add_argument("--common-config", default=str(CONFIG_DIR / "common.yaml"), help="Common yaml path.")
    parser.add_argument("--hz", type=float, default=30.0, help="Replay frequency, default 30 Hz.")
    parser.add_argument("--loop", action="store_true", help="Loop replay until Ctrl+C.")
    parser.add_argument("--zero-base-velocity", action="store_true", help="Do not move mobile base.")
    parser.add_argument("--use-height", action="store_true", help="Use base_command.base_command_height for lift target; default keeps lift at reset.")
    parser.add_argument("--warmup-seconds", type=float, default=1.5, help="Ramp current arm pose to first frame before replay.")
    parser.add_argument("--no-rnea", action="store_true", help="Disable Pinocchio RNEA feedforward tau.")
    parser.add_argument(
        "--gripper-mode",
        choices=("motor", "closed_high", "direct"),
        default="motor",
        help="Gripper mapping. motor: CSV 0=closed/max=open; closed_high: 0=open/max=closed; direct: CSV is URDF q.",
    )
    parser.add_argument("--gripper-input-min", type=float, default=0.0)
    parser.add_argument("--gripper-input-max", type=float, default=5.0)
    args = parser.parse_args()

    if args.hz <= 0.0:
        raise ValueError("--hz must be positive")
    if args.publish and not args.net:
        raise ValueError("--publish requires --net, e.g. --net eno2")
    CsvRealReplay(
        csv_path=Path(args.csv),
        hz=args.hz,
        net=args.net,
        publish=bool(args.publish and not args.dry_run),
        robot=args.robot,
        config=args.config,
        common_config=args.common_config,
        gripper_input_min=args.gripper_input_min,
        gripper_input_max=args.gripper_input_max,
        gripper_mode=args.gripper_mode,
        zero_base_velocity=args.zero_base_velocity,
        use_height=args.use_height,
        warmup_seconds=args.warmup_seconds,
        loop=args.loop,
        rnea=not args.no_rnea,
    ).run()


if __name__ == "__main__":
    main()
