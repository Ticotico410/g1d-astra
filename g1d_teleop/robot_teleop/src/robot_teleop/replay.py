"""Replay a recorded MCAP episode to the robot or into MuJoCo.

The G1-D ``teleop_frame`` format stores actions, measured state and images.

With ``--net`` the episode is published to the real robot through the same
publisher used for teleoperation, so PD gains, the command ramp and the
motion-mode release all behave identically. Without ``--net`` (or with
``--no-publish``) it only animates the MuJoCo viewer.

Usage::

    # MuJoCo only
    python robot_teleop/replay_robot.py --robot g1_d --input <episode.mcap>
    # Dry run: loads the hardware config but sends nothing
    python robot_teleop/replay_robot.py --robot g1_d --input <episode.mcap> \\
        --net eth0 --no-publish
    # Replay to the real robot
    python robot_teleop/replay_robot.py --robot g1_d --input <episode.mcap> --net enp4s0
"""

from __future__ import annotations

import base64
import json
import signal
import time

import numpy as np
import tyro

from .config import CONFIG_DIR, TeleopConfig, config_path_for_robot, load_config

# LowState motor indices, matching publisher.G1DDSPublisher.
_ARM_MOTOR_IDX = list(range(15, 29))   # 14: 7 left + 7 right
_WAIST_YAW_MOTOR_IDX = 12              # cfg.g1d_waist_yaw_joint
_WAIST_PITCH_MOTOR_IDX = 13            # cfg.g1d_waist_pitch_joint

_TOPIC_TELEOP_FRAME = "/teleop/frame"
N_ARM = 14


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_frames(path: str) -> tuple[list[dict], dict, str]:
    """Read an MCAP file and return (frames, info_meta, detected_format)."""
    from mcap.reader import make_reader

    frames: list[dict] = []
    info: dict = {}
    fmt = ""
    with open(path, "rb") as f:
        reader = make_reader(f)
        for _schema, channel, message in reader.iter_messages(
            topics=[_TOPIC_TELEOP_FRAME, "/episode/meta"]
        ):
            payload = json.loads(message.data)
            if channel.topic == _TOPIC_TELEOP_FRAME:
                frames.append(payload)
                fmt = "teleop_frame"
            elif payload.get("kind") == "info":
                info = payload
    frames.sort(key=lambda fr: fr.get("idx", 0))
    return frames, info, fmt


def _episode_summary(info: dict, fmt: str) -> tuple[float, str]:
    """Return the requested recording rate and task label."""
    return float(info.get("record_fps", 30) or 30), ""


def _measured_fps(frames: list[dict]) -> float:
    """Actual capture rate from the frame timestamps, or 0.0 if unavailable.

    The metadata rate is what was requested, not what was achieved: encoding
    JPEG encoding can push a nominal recording rate down substantially.
    Replaying at the requested rate would then run faster than the operator
    actually moved, so the measured rate is what keeps playback at real speed.
    """
    key = "sample_time_ns" if "sample_time_ns" in (frames[0] if frames else {}) else "t_ns"
    stamps = [int(fr.get(key, 0)) for fr in frames]
    if len(stamps) < 2 or any(s <= 0 for s in stamps):
        return 0.0
    deltas = np.diff(np.asarray(stamps, dtype=float))
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return 0.0
    median_ns = float(np.median(deltas))
    if median_ns <= 0.0:
        return 0.0
    return 1e9 / median_ns


# ---------------------------------------------------------------------------
# Frame to publish arguments
# ---------------------------------------------------------------------------
def _args_from_action(fr: dict) -> tuple | None:
    """Unpack the commanded values from a teleop_frame frame."""
    a = fr.get("actions", {})
    aq = a.get("arm_q")
    if not aq:
        return None
    arm_q = np.asarray(aq, dtype=float)
    at = a.get("arm_tau")
    arm_tau = np.asarray(at, dtype=float) if at else np.zeros(arm_q.size)
    base_cmd = [float(x) for x in (a.get("base_cmd") or [])]
    gq = a.get("gripper_q")
    gripper_q = np.asarray(gq, dtype=float) if gq is not None else None
    lift = a.get("lift_position")
    hq = a.get("head_q")
    head_q = np.asarray(hq, dtype=float) if hq else None
    return arm_q, arm_tau, base_cmd, gripper_q, (float(lift) if lift is not None else None), head_q


def _args_from_state(fr: dict, cfg: TeleopConfig) -> tuple | None:
    """Rebuild publish arguments from the measured state of a teleop_frame frame.

    The measured state has no base velocity command (zeroed) and no clean URDF
    gripper units, so grippers and torques fall back to the recorded actions.
    """
    st = fr.get("states", {})
    mq = st.get("motor_q")
    if not mq or len(mq) <= _ARM_MOTOR_IDX[-1]:
        return None
    arm_q = np.asarray([mq[i] for i in _ARM_MOTOR_IDX], dtype=float)
    yaw = float(mq[_WAIST_YAW_MOTOR_IDX])
    pitch = float(mq[_WAIST_PITCH_MOTOR_IDX])
    # Lift: invert the measured height (m) back to a joint value; fall back to
    # the recorded lift_position when no measurement exists.
    mh = st.get("measured_height_m")
    if mh is not None and cfg.g1d_lift_meas_scale:
        lift = float(np.clip(
            (float(mh) - cfg.g1d_lift_meas_offset) / cfg.g1d_lift_meas_scale,
            cfg.lz_joint_min, cfg.lz_joint_max,
        ))
    else:
        av = fr.get("actions", {}).get("lift_position")
        lift = float(av) if av is not None else None
    body_map = {
        cfg.g1d_waist_pitch_joint: pitch,
        cfg.g1d_waist_yaw_joint: yaw,
    }
    for name in cfg.lz_height_joints:
        body_map[name] = lift if lift is not None else 0.0
    base_body = [float(body_map.get(j, 0.0)) for j in cfg.base_cmd_joints]
    base_cmd = [0.0, 0.0, 0.0] + base_body
    a = fr.get("actions", {})
    gq = a.get("gripper_q")
    gripper_q = np.asarray(gq, dtype=float) if gq is not None else None
    at = a.get("arm_tau")
    arm_tau = np.asarray(at, dtype=float) if at else np.zeros(arm_q.size)
    hq = a.get("head_q")
    head_q = np.asarray(hq, dtype=float) if hq else None
    return arm_q, arm_tau, base_cmd, gripper_q, lift, head_q


def _publish_args(fr: dict, source: str, cfg: TeleopConfig, fmt: str) -> tuple | None:
    if source == "state":
        out = _args_from_state(fr, cfg)
        if out is not None:
            return out
        # No measured state in this frame (e.g. recorded in sim): use actions.
    return _args_from_action(fr)


# ---------------------------------------------------------------------------
# Image display
# ---------------------------------------------------------------------------
def _make_image_shower(show: bool, fmt: str):
    """Return (show_frame, close). No-op when disabled or OpenCV is missing."""
    if not show:
        return (lambda _fr: None), (lambda: None)
    try:
        import cv2
    except Exception:
        print("[replay] OpenCV not installed, skipping image display (pip install opencv-python)")
        return (lambda _fr: None), (lambda: None)

    def decode(record) -> np.ndarray | None:
        data = record.get("data") if isinstance(record, dict) else None
        if not data:
            return None
        try:
            raw = base64.b64decode(data)
            return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            return None

    window = "replay: head L/R, wrist L/R"

    def show_single(fr: dict) -> None:
        for key, record in (fr.get("images") or {}).items():
            img = decode(record)
            if img is not None:
                cv2.imshow(f"replay:{key}", img)
        cv2.waitKey(1)

    def close() -> None:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    return show_single, close


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def replay(
    input: str,
    net: str = "",
    robot: str = "g1_d",
    config: str | None = None,
    common_config: str = str(CONFIG_DIR / "common.yaml"),
    source: str = "action",
    target_fps: float = 0.0,
    no_publish: bool = False,
    zero_base_velocity: bool = False,
    show_images: bool = True,
    loop: bool = False,
    speed: float = 1.0,
    start_frame: int = 0,
    end_frame: int = -1,
) -> None:
    """Replay a recorded MCAP episode to the robot or into MuJoCo.

    input:
      Path to a G1-D episode_XXXX.mcap file in teleop_frame format.
    robot:
      Robot config name. Required when publishing, since it selects the hardware interface.
    net:
      Network interface (e.g. eth0). If set and not --no-publish, commands go to
      the real robot. If empty, visualize in MuJoCo only.
    source:
      action - replay the recorded commands. state - replay measured joints.
    target_fps:
      Playback rate. 0 (the default) replays at real speed, derived from the
      recorded frame timestamps rather than the nominal rate in the metadata.
    no_publish:
      Never touch hardware; visualize in MuJoCo even if --net is given.
    zero_base_velocity:
      Zero the recorded base velocity so the robot does not drive.
    show_images:
      Display recorded camera frames with OpenCV (needs opencv-python).
    loop:
      Loop playback until Ctrl+C.
    speed:
      Playback speed multiplier; 0.5 is half speed.
    start_frame / end_frame:
      Replay only a slice. end_frame is inclusive; -1 means the final frame.
    """
    source = str(source).lower()
    if source not in {"action", "state"}:
        raise SystemExit("--source must be 'action' or 'state'")
    if speed <= 0.0:
        raise SystemExit("--speed must be positive")

    frames, info, fmt = _load_frames(input)
    if not frames:
        raise SystemExit(f"[replay] no frame messages found in {input}")
    fmt = fmt or "teleop_frame"

    start = max(0, int(start_frame))
    end = len(frames) if end_frame < 0 else min(len(frames), int(end_frame) + 1)
    if start >= end:
        raise SystemExit("Selected frame range is empty")
    frames = frames[start:end]

    record_fps, goal = _episode_summary(info, fmt)
    measured_fps = _measured_fps(frames)
    if target_fps > 0:
        fps = target_fps
        rate_note = "from --target-fps"
    elif measured_fps > 0:
        # Default to real speed: use the rate the episode was actually captured
        # at, not the rate its metadata asked for.
        fps = measured_fps
        rate_note = f"measured from timestamps (metadata says {record_fps:.1f})"
    else:
        fps = record_fps
        rate_note = "from metadata"
    period = 1.0 / max(fps * speed, 1e-3)
    print(f"[replay] {input}")
    print(
        f"[replay] format={fmt}, frames={len(frames)}, fps={fps:.1f} ({rate_note}), "
        f"speed={speed:g}, source={source}"
    )
    if goal:
        print(f"[replay] goal={goal!r}")

    config_path = config if config is not None else str(config_path_for_robot(robot))
    cfg = load_config(config_path, common_config)
    cfg.robot_name = robot

    publish_to_robot = bool(net) and not no_publish

    stop = {"flag": False}

    def _on_sig(_signum, _frame):
        stop["flag"] = True
        print("\n[replay] Ctrl+C, stopping...")

    signal.signal(signal.SIGINT, _on_sig)

    show_frame, close_images = _make_image_shower(show_images, fmt)
    try:
        if publish_to_robot:
            _replay_to_robot(frames, cfg, net, period, source, fmt, zero_base_velocity,
                             loop, stop, show_frame)
        else:
            _replay_in_mujoco(frames, cfg, period, source, fmt, loop, stop, show_frame)
    finally:
        close_images()


def _measured_arm_q(publisher) -> np.ndarray | None:
    """Measured 14 arm joint angles, used as the warm-up interpolation start."""
    try:
        snap = publisher.state_snapshot()
    except Exception:
        return None
    arm = snap.get("arm_q_meas")
    if arm:
        return np.asarray(arm, dtype=float)
    mq = snap.get("motor_q")
    if not mq or len(mq) <= _ARM_MOTOR_IDX[-1]:
        return None
    return np.asarray([mq[i] for i in _ARM_MOTOR_IDX], dtype=float)


def _ramp_in(publisher, first_args, period, stop, seconds: float = 1.5) -> None:
    """Interpolate from the robot's current pose to the first frame.

    Hardware needs a stretch of commands close to its measured pose before it
    will follow: jumping straight to a distant first frame can leave joints
    disabled, which looks like the robot simply not moving.
    """
    arm_q0, arm_tau0, base_cmd0, gripper_q0, lift0, head_q0 = first_args
    start_arm = _measured_arm_q(publisher)
    if start_arm is None or start_arm.size != arm_q0.size:
        start_arm = arm_q0  # no measurement available: skip interpolation
    steps = max(1, int(round(seconds / max(period, 1e-3))))
    print(f"[replay] warm-up: {seconds:.1f}s from the measured pose to frame 0 ({steps} steps)")
    # The base stays still during warm-up; lift and grippers go to frame 0.
    warm_base = [0.0, 0.0, 0.0] + [float(x) for x in base_cmd0[3:]]
    for i in range(1, steps + 1):
        if stop["flag"]:
            return
        alpha = i / steps
        arm_q = start_arm + (arm_q0 - start_arm) * alpha
        publisher.publish(
            arm_q=arm_q,
            arm_tau=arm_tau0,
            base_cmd=warm_base,
            gripper_q=gripper_q0,
            head_q=head_q0,
            lift_position=lift0,
        )
        time.sleep(period)


def _replay_to_robot(frames, cfg, net, period, source, fmt, zero_base_velocity,
                     loop, stop, show_frame) -> None:
    from .publisher import make_publisher

    # hardware_interface comes from the robot YAML and is not overridden here.
    publisher = make_publisher("robot", cfg, net)
    first_args = _publish_args(frames[0], source, cfg, fmt)
    if first_args is not None:
        _ramp_in(publisher, first_args, period, stop)
    print("[replay] replaying to the robot... (Ctrl+C to stop)")
    try:
        while not stop["flag"]:
            next_at = time.monotonic()
            for fr in frames:
                if stop["flag"]:
                    break
                args = _publish_args(fr, source, cfg, fmt)
                if args is None:
                    continue
                arm_q, arm_tau, base_cmd, gripper_q, lift, head_q = args
                if zero_base_velocity and len(base_cmd) >= 3:
                    base_cmd[0:3] = [0.0, 0.0, 0.0]
                publisher.publish(
                    arm_q=arm_q,
                    arm_tau=arm_tau,
                    base_cmd=base_cmd,
                    gripper_q=gripper_q,
                    head_q=head_q,
                    lift_position=lift,
                )
                show_frame(fr)
                next_at += period
                delay = next_at - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            if not loop:
                break
    finally:
        # The lift and base take velocity commands, so they must be explicitly
        # zeroed (lift_position=None gives z=0), otherwise the robot keeps
        # drifting after exit. Hold the final arm pose briefly, then close.
        try:
            args = _publish_args(frames[-1], source, cfg, fmt)
            if args is not None:
                arm_q, arm_tau, base_cmd, gripper_q, _lift, head_q = args
                if len(base_cmd) >= 3:
                    base_cmd[0:3] = [0.0, 0.0, 0.0]
                hold_steps = max(1, int(round(0.5 / max(period, 1e-3))))
                for _ in range(hold_steps):
                    publisher.publish(arm_q=arm_q, arm_tau=arm_tau, base_cmd=base_cmd,
                                      gripper_q=gripper_q, head_q=head_q, lift_position=None)
                    time.sleep(period)
        except Exception:
            pass
        publisher.close()
        print("[replay] stopped (lift and base velocities zeroed)")


def _replay_in_mujoco(frames, cfg, period, source, fmt, loop, stop, show_frame) -> None:
    import mujoco.viewer

    from .mujoco_robot import MujocoVisualizer

    viz = MujocoVisualizer(
        cfg.urdf_path,
        cfg.controlled_joints,
        floating_base=cfg.floating_base,
        floating_base_child_link=cfg.floating_base_child_link or None,
    )
    arm_joints = cfg.left_arm_joints + cfg.right_arm_joints
    gripper_joints = cfg.left_gripper_joints + cfg.right_gripper_joints

    def _apply(fr: dict) -> None:
        args = _publish_args(fr, source, cfg, fmt)
        if args is None:
            return
        arm_q, _tau, base_cmd, gripper_q, _lift, head_q = args
        if arm_q.size == len(arm_joints):
            viz.set_joint_qpos(arm_joints, arm_q)
        if gripper_q is not None and gripper_q.size == len(gripper_joints) and gripper_joints:
            viz.set_joint_qpos(gripper_joints, gripper_q)
        if head_q is not None and cfg.head_joints and head_q.size == len(cfg.head_joints):
            viz.set_joint_qpos(cfg.head_joints, head_q)
        if cfg.base_cmd_joints and len(base_cmd) >= 3 + len(cfg.base_cmd_joints):
            viz.set_joint_qpos(cfg.base_cmd_joints, np.asarray(base_cmd[3:], dtype=float))

    print("[replay] MuJoCo playback... (close the window or Ctrl+C to stop)")
    with mujoco.viewer.launch_passive(viz.model, viz.data) as viewer:
        while viewer.is_running() and not stop["flag"]:
            next_at = time.monotonic()
            for fr in frames:
                if not viewer.is_running() or stop["flag"]:
                    break
                _apply(fr)
                show_frame(fr)
                viewer.sync()
                next_at += period
                delay = next_at - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            if not loop:
                break
    print("[replay] playback finished")


def cli() -> None:
    tyro.cli(replay)


if __name__ == "__main__":
    cli()
