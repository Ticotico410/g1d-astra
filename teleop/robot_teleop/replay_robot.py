#!/usr/bin/env python3
"""Replay a recorded MCAP episode from the workspace root.

Usage:
    python robot_teleop/replay_robot.py --robot g1_d \
        --input data/demo/draw_picture/episode_0001.mcap
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robot_teleop.replay import cli


if __name__ == "__main__":
    cli()
