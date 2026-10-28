#!/usr/bin/env python3
"""Run the multi-robot MuJoCo teleop app from the workspace root.

Usage:
    python robot_teleop/teleop_robot.py --robot g1_d --mode sim
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robot_teleop.main import cli


if __name__ == "__main__":
    cli()
