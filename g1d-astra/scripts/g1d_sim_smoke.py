#!/usr/bin/env python3
"""Exercise the G1-D DirectVR -> Placo IK -> MuJoCo visualization path offline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teleop-root",
        type=Path,
        default=Path('/home/karthus_chen/ycb_ws/g1d_teleop'),
        help="Path to the G1-D teleoperation checkout",
    )
    args = parser.parse_args()
    teleop_root = args.teleop_root.expanduser().resolve()
    src = teleop_root / "robot_teleop" / "src"
    if not (src / "robot_teleop" / "main.py").is_file():
        parser.error(f"robot_teleop package not found under {teleop_root}")
    sys.path.insert(0, str(src))

    from robot_teleop.direct_vr import BodySample, ButtonState
    from robot_teleop.main import TeleopRunner

    runner = TeleopRunner(
        config_path=str(teleop_root / "robot_teleop/config/g1_d.yaml"),
        common_config_path=str(teleop_root / "robot_teleop/config/common.yaml"),
        robot="g1_d",
        mode="sim",
        net="",
        scale=1.0,
        teleop_control_mode="wbc",
    )
    body = BodySample()
    body.available = True
    buttons = ButtonState(left_grip=1.0, right_grip=1.0)
    runner.reader._body = body
    runner.reader._buttons = buttons

    # Controller translations are expressed in PICO coordinates. These map to
    # the six cardinal directions in the robot frame through the real input path.
    cases = {
        "forward": (0.0, 0.0, -0.04),
        "backward": (0.0, 0.0, 0.04),
        "left": (-0.04, 0.0, 0.0),
        "right": (0.04, 0.0, 0.0),
        "up": (0.0, 0.04, 0.0),
        "down": (0.0, -0.04, 0.0),
    }
    signs = {
        "forward": (0, 1), "backward": (0, -1),
        "left": (1, 1), "right": (1, -1),
        "up": (2, 1), "down": (2, -1),
    }
    failures: list[str] = []

    try:
        for label, controller_delta in cases.items():
            runner.ik_robot.reset()
            body.poses[:] = 0.0
            body.poses[:, 6] = 1.0
            runner.left_ref = runner.right_ref = None
            runner.left_robot_ref = runner.ik_robot.frame_pose(runner.cfg.left_ee_body)
            runner.right_robot_ref = runner.ik_robot.frame_pose(runner.cfg.right_ee_body)
            runner.targets_from_vr()  # capture the controller and robot references

            body.poses[20, :3] = controller_delta
            body.poses[21, :3] = controller_delta
            targets, active_left, active_right = runner.targets_from_vr()
            before = np.stack([
                runner.ik_robot.frame_pose(runner.cfg.left_ee_body)[0],
                runner.ik_robot.frame_pose(runner.cfg.right_ee_body)[0],
            ])
            for _ in range(20):
                runner.ik_robot.solve_ik(
                    targets,
                    active_left,
                    active_right,
                    runner.cfg.left_ee_body,
                    runner.cfg.right_ee_body,
                    runner.cfg.ik_iters,
                )
            after = np.stack([
                runner.ik_robot.frame_pose(runner.cfg.left_ee_body)[0],
                runner.ik_robot.frame_pose(runner.cfg.right_ee_body)[0],
            ])
            delta = after - before
            axis, sign = signs[label]
            projected = delta[:, axis] * sign
            passed = len(targets) == 2 and bool(np.all(projected > 0.012))
            print(f"{label:8s} L={np.round(delta[0], 3)} R={np.round(delta[1], 3)} {'PASS' if passed else 'FAIL'}")
            if not passed:
                failures.append(label)
    finally:
        runner.reader.stop()
        runner.publisher.close()

    if failures:
        print("Failed directions:", ", ".join(failures))
        return 1
    print("PASS: bilateral G1-D IK responds in all six Cartesian directions (offline; no DDS).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
