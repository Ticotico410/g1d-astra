#!/usr/bin/env python3
"""Astra-gated, slow, stepwise G1-D reference-trajectory execution."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
TELEOP_ROOT = Path(os.environ.get('G1D_TELEOP_ROOT', '/home/karthus_chen/ycb_ws/g1d_teleop'))
IMAGE_ROOT = TELEOP_ROOT / 'image' / 'robot_driver' / 'image_server'
ROBOT_SRC = TELEOP_ROOT / 'robot_teleop' / 'src'
sys.path[:0] = [str(ROOT), str(IMAGE_ROOT), str(ROBOT_SRC)]

from image_client import ImageClient
from hikvision_image_client import HikvisionImageClient
from core.decider import Astra
from robot_teleop.config import config_path_for_robot, load_config
from robot_teleop.replay import (
    _load_frames,
    _measured_fps,
    _publish_args,
    _ramp_in,
)
from robot_teleop.publisher import make_publisher


DEMO = TELEOP_ROOT / 'data/demo/draw_picture/episode_0001.mcap'
SEGMENT_FRAMES = 25
SLOWDOWN = 2.0
SKILLS = {
    'advance_reference_step': (
        'Continue the validated drawing action at controlled speed, then inspect fresh images '
        'before the next decision.'
    ),
    'hold_and_look': 'Do not move; inspect the fresh camera observations again.',
    'done': 'The requested task is visibly complete on the board.',
}
INSTRUCTIONS = (
    'Follow the supplied drawing task and the operator-validated reference motion. At '
    'each decision inspect all four fresh images (right head, left wrist, right wrist, and '
    'third-person view). Choose advance_reference_step only when the robot is stable, no '
    'person or obstacle is in either arm workspace, and the next short segment is safe. '
    'Each segment is followed by a new observation. The reference is executed at half its '
    'recorded speed, with mobile-base velocity fixed to zero. Do not judge intermediate '
    'partial strokes as failure. Never expose internal execution details in the UI. After '
    'all reference motion, choose done only when the requested result is visible. Write concise Chinese notes.'
)


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def append_log(path: Path, value: dict) -> None:
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def execute_segment(publisher, frames: list[dict], cfg, start: int, end: int,
                    period: float, stop: dict, stop_path: Path) -> int:
    next_at = time.monotonic()
    cursor = start
    for cursor in range(start, end):
        if stop_path.exists():
            stop['flag'] = True
            break
        args = _publish_args(frames[cursor], 'action', cfg, 'teleop_frame')
        if args is not None:
            arm_q, arm_tau, base_cmd, gripper_q, lift, head_q = args
            base_cmd[0:3] = [0.0, 0.0, 0.0]
            publisher.publish(arm_q=arm_q, arm_tau=arm_tau, base_cmd=base_cmd,
                              gripper_q=gripper_q, head_q=head_q, lift_position=lift)
        next_at += period
        delay = next_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
    return cursor + 1 if not stop['flag'] else cursor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--demo', type=Path, default=DEMO)
    parser.add_argument('--task-text', default='左臂画太阳，右臂画简化山体与飞鸟')
    parser.add_argument('--agent-instructions', default=None)
    parser.add_argument('--phase-label', default='双臂绘画')
    parser.add_argument('--net', default='enp4s0')
    parser.add_argument('--robot-image-host', default='192.168.123.164')
    parser.add_argument('--third-camera-host', default='127.0.0.1')
    parser.add_argument('--third-camera-port', type=int, default=5560)
    parser.add_argument('--max-decisions', type=int, default=80)
    parser.add_argument('--segment-frames', type=int, default=SEGMENT_FRAMES)
    parser.add_argument('--slowdown', type=float, default=SLOWDOWN)
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.run_dir / 'status.json'
    stop_path = args.run_dir / 'stop_requested'
    log_path = args.run_dir / 'decisions.jsonl'
    load_dotenv(ROOT / '.env', override=True)
    key = (os.environ.get('OPENAI_API_KEY') or os.environ.get('UNITREE_OPENAI_API_KEY') or '').strip()
    if not key:
        raise RuntimeError('OPENAI_API_KEY is missing')
    os.environ['OPENAI_API_KEY'] = key
    base = (os.environ.get('OPENAI_BASE_URL') or '').strip()
    if base:
        os.environ['OPENAI_BASE_URL'] = base.rstrip('/')

    frames, _, fmt = _load_frames(str(args.demo))
    if fmt != 'teleop_frame' or not frames:
        raise RuntimeError(f'Unexpected demonstration format/length: {fmt}, {len(frames)}')
    if args.segment_frames < 1 or args.slowdown < 1.0:
        raise ValueError('--segment-frames must be positive and --slowdown must be >= 1')
    selected = frames
    fps = _measured_fps(selected) or 25.0
    period = args.slowdown / fps
    total_segments = math.ceil(len(selected) / args.segment_frames)
    cfg = load_config(config_path_for_robot('g1_d'),
                      TELEOP_ROOT / 'robot_teleop/config/common.yaml')
    cfg.robot_name = 'g1_d'

    images = ImageClient(image_show=False, server_address=args.robot_image_host,
                         mode='teleimager', camera_streams=('head', 'left', 'right'))
    receiver = threading.Thread(target=images.receive_process, daemon=True)
    receiver.start()
    third = HikvisionImageClient(args.third_camera_host, args.third_camera_port).start()
    astra = Astra(model='gpt-6-astra', effort='high')
    stop = {'flag': False}
    watcher_done = threading.Event()
    publisher = None
    cursor = 0
    ramped_in = False
    def watch_stop_file():
        while not watcher_done.wait(0.05):
            if stop_path.exists():
                stop['flag'] = True
                return
    watcher = threading.Thread(target=watch_stop_file, daemon=True)
    watcher.start()
    try:
        publisher = make_publisher('robot', cfg, args.net)
        for decision_step in range(args.max_decisions):
            if stop['flag']:
                break
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                latest = images.get_latest()
                third_rgb = third.get_rgb()
                if (latest.get('head_right') is not None and latest.get('left_wrist') is not None
                        and latest.get('right_wrist') is not None
                        and third_rgb is not None and latest.get('head_age_s', 99) < 2
                        and latest.get('left_wrist_age_s', 99) < 2
                        and latest.get('right_wrist_age_s', 99) < 2
                        and third.age_s() < 2):
                    break
                time.sleep(0.05)
            else:
                raise TimeoutError('Fresh head-right, left-wrist, right-wrist, or third-camera frame missing')
            obs = {
                'head_right': cv2.cvtColor(latest['head_right'], cv2.COLOR_BGR2RGB),
                'left_wrist': cv2.cvtColor(latest['left_wrist'], cv2.COLOR_BGR2RGB),
                'right_wrist': cv2.cvtColor(latest['right_wrist'], cv2.COLOR_BGR2RGB),
                'third_camera': third_rgb,
            }
            atomic_json(status_path, {
                'step': min(total_segments, cursor // args.segment_frames),
                'max_steps': total_segments,
                'progress': cursor / len(selected),
                'task_text': args.task_text,
                'progress_text': args.phase_label, 'status': 'ANALYZING',
                'execution': 'Astra 正在检查任务状态',
            })
            allowed = dict(SKILLS)
            if cursor < len(selected):
                allowed.pop('done')
            else:
                allowed.pop('advance_reference_step')
            choice = astra.choose({
                'task': args.task_text,
                'execution_progress': cursor / len(selected),
                'reference_fps_hz': fps,
                'execution_fps_hz': fps / args.slowdown,
                'reference_direction': 'as manually validated by the operator',
                'reference_descent_m': 0.0,
            }, allowed, args.agent_instructions or (
                f'Task: {args.task_text}. {INSTRUCTIONS}'
            ), obs)
            append_log(log_path, {
                'time': datetime.now().isoformat(), 'step': decision_step,
                'skill': choice.skill, 'note': choice.note,
                'decision': 'astra_segment_gate', 'executed': None,
                'execution_progress': cursor / len(selected),
                'observation_keys': list(obs),
            })
            if stop['flag']:
                break
            if choice.skill == 'advance_reference_step' and cursor < len(selected):
                if not ramped_in:
                    first = _publish_args(selected[0], 'action', cfg, fmt)
                    if first is None:
                        raise RuntimeError('Reference trajectory first frame has no action')
                    _ramp_in(publisher, first, period, stop, seconds=3.0)
                    ramped_in = True
                end = min(cursor + args.segment_frames, len(selected))
                atomic_json(status_path, {
                    'step': min(total_segments, cursor // args.segment_frames),
                    'max_steps': total_segments,
                    'progress': cursor / len(selected),
                    'task_text': args.task_text,
                    'progress_text': args.phase_label,
                    'status': 'AUTO_EXECUTING', 'skill': choice.skill,
                    'observation': '已检查当前画面，正在继续执行任务动作。',
                    'execution': f'任务执行中，进度 {cursor / len(selected) * 100:.0f}%',
                })
                cursor = execute_segment(publisher, selected, cfg, cursor, end,
                                         period, stop, stop_path)
                progress = cursor / len(selected)
                atomic_json(status_path, {
                    'step': min(total_segments, math.ceil(cursor / args.segment_frames)),
                    'max_steps': total_segments, 'progress': progress,
                    'task_text': args.task_text,
                    'progress_text': args.phase_label,
                    'status': 'OBSERVING' if not stop['flag'] else 'STOPPED',
                    'skill': choice.skill,
                    'observation': '当前动作已执行，等待新图像复核。',
                    'execution': f'任务进度 {progress * 100:.0f}%',
                })
                if stop['flag']:
                    break
                continue
            if choice.skill == 'done' and cursor >= len(selected):
                atomic_json(status_path, {
                    'step': total_segments, 'max_steps': total_segments, 'progress': 1.0,
                    'progress_text': f'{args.phase_label}已完成', 'status': 'COMPLETE',
                    'skill': 'done', 'observation': choice.note,
                    'execution': '停止发送动作',
                })
                return 0
            atomic_json(status_path, {
                'step': min(total_segments, math.ceil(cursor / args.segment_frames)),
                'max_steps': total_segments,
                'progress': cursor / len(selected),
                'task_text': args.task_text,
                'progress_text': args.phase_label, 'status': 'OBSERVING',
                'skill': 'hold_and_look', 'observation': choice.note,
                'execution': '保持当前位置，重新观察',
            })
        if stop['flag']:
            atomic_json(status_path, {
                'step': min(total_segments, math.ceil(cursor / args.segment_frames)),
                'max_steps': total_segments, 'progress': cursor / len(selected),
                'progress_text': '已停止', 'status': 'STOPPED',
                'execution': '停止发送新动作，保持当前位置',
            })
            return 0
        atomic_json(status_path, {
            'step': min(total_segments, math.ceil(cursor / args.segment_frames)),
            'max_steps': total_segments,
            'progress': cursor / len(selected), 'progress_text': '达到决策步数上限',
            'status': 'ACTION_LIMIT', 'execution': '停止发送新动作',
        })
        return 1
    except KeyboardInterrupt:
        atomic_json(status_path, {
            'step': min(total_segments, math.ceil(cursor / args.segment_frames)),
            'max_steps': total_segments, 'progress': cursor / len(selected),
            'progress_text': '已停止', 'status': 'STOPPED',
            'execution': '停止发送新动作，保持当前位置',
        })
        return 130
    except Exception as exc:
        atomic_json(status_path, {
            'step': min(total_segments, math.ceil(cursor / args.segment_frames)),
            'max_steps': total_segments, 'progress': cursor / len(selected),
            'progress_text': '异常停止', 'status': 'ERROR',
            'observation': f'{type(exc).__name__}: {exc}',
            'execution': '流程异常，停止发送动作',
        })
        raise
    finally:
        watcher_done.set()
        watcher.join(timeout=1)
        if publisher is not None:
            publisher.close()
        images.running = False
        receiver.join(timeout=2)
        third.stop()


if __name__ == '__main__':
    raise SystemExit(main())
