#!/usr/bin/env python3

#python test/test_g1d_sim.py test/csv/eval_action_episode_0.csv --hz 30 

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "robot_teleop" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robot_teleop.config import CONFIG_DIR, load_config
from robot_teleop.mujoco_robot import MujocoVisualizer


CSV_TO_G1D_JOINTS = {
    # From the CSV robot_29dof columns, only waist_yaw and waist_pitch are
    # replayed; legs and waist_roll are ignored.
    "robot_29dof.waist_yaw": "torso_Joint",
    "robot_29dof.waist_pitch": "Yaw_Joint",
    # Left arm.
    "robot_29dof.left_arm_shoulder_pitch": "left_shoulder_pitch_joint",
    "robot_29dof.left_arm_shoulder_roll": "left_shoulder_roll_joint",
    "robot_29dof.left_arm_shoulder_yaw": "left_shoulder_yaw_joint",
    "robot_29dof.left_arm_elbow": "left_elbow_joint",
    "robot_29dof.left_arm_wrist_roll": "left_wrist_roll_joint",
    "robot_29dof.left_arm_wrist_pitch": "left_wrist_pitch_joint",
    "robot_29dof.left_arm_wrist_yaw": "left_wrist_yaw_joint",
    # Right arm.
    "robot_29dof.right_arm_shoulder_pitch": "right_shoulder_pitch_joint",
    "robot_29dof.right_arm_shoulder_roll": "right_shoulder_roll_joint",
    "robot_29dof.right_arm_shoulder_yaw": "right_shoulder_yaw_joint",
    "robot_29dof.right_arm_elbow": "right_elbow_joint",
    "robot_29dof.right_arm_wrist_roll": "right_wrist_roll_joint",
    "robot_29dof.right_arm_wrist_pitch": "right_wrist_pitch_joint",
    "robot_29dof.right_arm_wrist_yaw": "right_wrist_yaw_joint",
}


class CsvReplay:
    def __init__(
        self,
        csv_path: Path,
        hz: float,
        gripper_input_min: float,
        gripper_input_max: float,
        gripper_mode: str,
        loop: bool,
    ) -> None:
        self.csv_path = csv_path.expanduser().resolve()
        self.hz = hz
        self.dt = 1.0 / hz
        self.gripper_input_min = gripper_input_min
        self.gripper_input_max = gripper_input_max
        self.gripper_mode = gripper_mode
        self.loop = loop
        self.rows = self._read_rows(self.csv_path)
        if not self.rows:
            raise ValueError(f"CSV has no data rows: {self.csv_path}")

        self.cfg = load_config(CONFIG_DIR / "g1_d.yaml", CONFIG_DIR / "common.yaml")
        self.robot = MujocoVisualizer(
            self.cfg.urdf_path,
            self.cfg.controlled_joints,
            floating_base=self.cfg.floating_base,
            floating_base_child_link=self.cfg.floating_base_child_link or None,
        )
        self._init_reset_pose()
        self._check_columns()

    @staticmethod
    def _read_rows(csv_path: Path) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                row: dict[str, float] = {}
                for key, value in raw.items():
                    if key is None or value is None or value == "":
                        continue
                    row[key] = float(value)
                rows.append(row)
        return rows

    def _check_columns(self) -> None:
        header = set(self.rows[0].keys())
        needed = set(CSV_TO_G1D_JOINTS) | {
            "gripper.left_gripper_gripper_pos",
            "gripper.right_gripper_gripper_pos",
            "base_command.base_command_vx",
            "base_command.base_command_angle_z",
        }
        missing = sorted(needed - header)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

    def _init_reset_pose(self) -> None:
        values = []
        for name in self.cfg.controlled_joints:
            values.append(float(self.cfg.reset_joint_positions.get(name, 0.0)))
        self.robot.set_joint_qpos(self.cfg.controlled_joints, np.asarray(values, dtype=float))

    def _set_csv_joints(self, row: dict[str, float]) -> None:
        joint_names: list[str] = []
        q: list[float] = []
        for csv_name, joint_name in CSV_TO_G1D_JOINTS.items():
            if csv_name in row:
                joint_names.append(joint_name)
                q.append(row[csv_name])
        if joint_names:
            self.robot.set_joint_qpos(joint_names, np.asarray(q, dtype=float))

    def _map_gripper_value(self, raw: float) -> float:
        if self.gripper_mode == "direct":
            return float(np.clip(raw, self.cfg.gripper_open_position, self.cfg.gripper_closed_position))

        denom = max(self.gripper_input_max - self.gripper_input_min, 1e-6)
        alpha = float(np.clip((raw - self.gripper_input_min) / denom, 0.0, 1.0))
        if self.gripper_mode == "motor":
            # G1-D/Dex1 motor value: 0 = closed, larger value = open.
            return self.cfg.gripper_closed_position + alpha * (
                self.cfg.gripper_open_position - self.cfg.gripper_closed_position
            )
        if self.gripper_mode == "closed_high":
            # Normalized/action value: 0 = open, larger value = closed.
            return self.cfg.gripper_open_position + alpha * (
                self.cfg.gripper_closed_position - self.cfg.gripper_open_position
            )
        raise ValueError(f"Unsupported gripper mode: {self.gripper_mode}")

    def _set_gripper(self, row: dict[str, float]) -> None:
        left_raw = row.get("gripper.left_gripper_gripper_pos", 0.0)
        right_raw = row.get("gripper.right_gripper_gripper_pos", 0.0)
        left_q = self._map_gripper_value(left_raw)
        right_q = self._map_gripper_value(right_raw)
        joint_names = self.cfg.left_gripper_joints + self.cfg.right_gripper_joints
        if not joint_names:
            return
        q = [left_q] * len(self.cfg.left_gripper_joints) + [right_q] * len(self.cfg.right_gripper_joints)
        self.robot.set_joint_qpos(joint_names, np.asarray(q, dtype=float))

    def _integrate_base(self, row: dict[str, float]) -> None:
        vx = row.get("base_command.base_command_vx", 0.0)
        vyaw = row.get("base_command.base_command_angle_z", 0.0)
        self.robot.integrate_base_velocity(vx, 0.0, vyaw, self.dt)

    def apply_row(self, row: dict[str, float]) -> None:
        self._set_csv_joints(row)
        self._set_gripper(row)
        self._integrate_base(row)
        mujoco.mj_forward(self.robot.model, self.robot.data)

    def run(self) -> None:
        print(f"[replay] csv={self.csv_path}")
        print(f"[replay] frames={len(self.rows)}, hz={self.hz}, dt={self.dt:.4f}s")
        print(
            f"[replay] gripper_mode={self.gripper_mode}, "
            f"input_range=[{self.gripper_input_min}, {self.gripper_input_max}]"
        )
        print("[replay] ignore legs and waist_roll; replay waist_yaw/waist_pitch/arms/grippers/base vx+vyaw")
        with mujoco.viewer.launch_passive(self.robot.model, self.robot.data) as viewer:
            while viewer.is_running():
                for i, row in enumerate(self.rows):
                    if not viewer.is_running():
                        break
                    t0 = time.perf_counter()
                    self.apply_row(row)
                    viewer.sync()
                    if i % int(max(self.hz, 1.0)) == 0:
                        print(f"[replay] frame={i}/{len(self.rows)}")
                    elapsed = time.perf_counter() - t0
                    sleep_time = self.dt - elapsed
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)
                if not self.loop:
                    break
            print("[replay] done")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a G1-D action CSV in MuJoCo at 30 Hz by default.")
    parser.add_argument(
        "csv",
        nargs="?",
        default=str(ROOT / "test" / "csv" / "eval_action_episode_0.csv"),
        help="CSV path to replay.",
    )
    parser.add_argument("--hz", type=float, default=30.0, help="Replay frequency, default 30 Hz.")
    parser.add_argument(
        "--gripper-mode",
        choices=("motor", "closed_high", "direct"),
        default="motor",
        help="Gripper mapping. motor: 0=closed/max=open; closed_high: 0=open/max=closed; direct: CSV is URDF q in meters.",
    )
    parser.add_argument(
        "--gripper-input-min",
        type=float,
        default=0.0,
        help="Minimum CSV gripper input for motor/closed_high modes; default 0.0.",
    )
    parser.add_argument(
        "--gripper-input-max",
        type=float,
        default=5.0,
        help="Maximum CSV gripper input for motor/closed_high modes; default 5.0 for G1-D/Dex1 motor value.",
    )
    parser.add_argument("--loop", action="store_true", help="Loop replay until the viewer is closed.")
    args = parser.parse_args()
    if args.hz <= 0.0:
        raise ValueError("--hz must be positive")
    CsvReplay(
        Path(args.csv),
        hz=args.hz,
        gripper_input_min=args.gripper_input_min,
        gripper_input_max=args.gripper_input_max,
        gripper_mode=args.gripper_mode,
        loop=args.loop,
    ).run()


if __name__ == "__main__":
    main()
