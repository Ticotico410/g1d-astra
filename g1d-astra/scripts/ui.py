#!/usr/bin/env python3
"""深色 G1-D 双臂绘画操作界面。
python scripts/ui.py /tmp/g1d_ui_preview
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TELEOP_ROOT = Path(os.environ.get('G1D_TELEOP_ROOT', '/home/karthus_chen/ycb_ws/g1d_teleop'))
IMAGE_ROOT = TELEOP_ROOT / 'image' / 'robot_driver' / 'image_server'
sys.path.insert(0, str(IMAGE_ROOT))
from image_client import ImageClient
from hikvision_image_client import HikvisionImageClient


WIDTH, HEIGHT = 2240, 960
THIRD_W = 600
BG = (17, 20, 24)
PANEL = (29, 34, 40)
PANEL_ALT = (37, 43, 50)
TEXT = (231, 236, 240)
MUTED = (153, 164, 174)
CYAN = (211, 174, 73)
GREEN = (88, 200, 128)
AMBER = (55, 176, 238)
RED = (78, 82, 235)
STATUS_ZH = {
    'ANALYZING': '分析图像中', 'AUTO_EXECUTING': '自主执行中',
    'OBSERVING': '重新观察中', 'COMPLETE': '已完成',
    'ACTION_LIMIT': '达到步数上限', 'ERROR': '异常停止', 'STOPPED': '已停止',
}
SKILL_ZH = {
    'x_forward_5cm': '末端向前 5 cm', 'x_back_5cm': '末端向后 5 cm',
    'y_left_5cm': '末端向左 5 cm', 'y_right_5cm': '末端向右 5 cm',
    'z_up_5cm': '末端抬高 5 cm', 'z_down_5cm': '末端下降 5 cm',
    'draw_circle_3cm': '连续绘制半径 3 cm 圆形',
    'advance_reference_step': '继续执行任务动作',
    'hold_and_look': '保持并观察', 'done': '任务完成',
}
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONTS = {size: ImageFont.truetype(FONT_PATH, size) for size in (14, 16, 18, 20, 22, 24)}
TEXT_OPS = []


def fit(image: np.ndarray | None, width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), np.uint8)
    if image is None:
        return canvas
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return canvas
    scale = min(width / w, height / h)
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
    y, x = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def text(canvas, x, y, value, scale=0.55, color=TEXT, thickness=1):
    size = min(FONTS, key=lambda candidate: abs(candidate - round(scale * 40)))
    TEXT_OPS.append((x, y, str(value), size, color))


def paint_text(canvas):
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    for x, y, value, size, color in TEXT_OPS:
        draw.text((x, y), value[:100], font=FONTS[size], fill=color[::-1], anchor='ls')
    TEXT_OPS.clear()
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def draw_view(canvas, image, title, x, y, width, height, age=None):
    cv2.rectangle(canvas, (x, y), (x + width, y + height), PANEL, -1)
    label = title if age is None else f'{title}   {age:.2f}s'
    text(canvas, x + 14, y + 26, label, 0.48, MUTED)
    y0, y1 = y + 38, y + height - 10
    x0, x1 = x + 10, x + width - 10
    # Destination slice size must match fit() output exactly.
    canvas[y0:y1, x0:x1] = fit(image, x1 - x0, y1 - y0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--robot-image-host', default=os.environ.get('G1D_IMAGE_HOST', '192.168.123.164'))
    parser.add_argument('--third-camera-host', default=os.environ.get('G1D_THIRD_CAMERA_HOST', '127.0.0.1'))
    parser.add_argument('--third-camera-port', type=int, default=int(os.environ.get('G1D_THIRD_CAMERA_PORT', '5560')))
    parser.add_argument('--head-port', type=int, default=55555)
    parser.add_argument('--left-wrist-port', type=int, default=55556)
    parser.add_argument('--right-wrist-port', type=int, default=55557)
    parser.add_argument('--record-fps', type=float, default=10.0)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)

    images = ImageClient(image_show=False, server_address=args.robot_image_host,
                         mode='teleimager', head_port=args.head_port,
                         left_wrist_port=args.left_wrist_port,
                         right_wrist_port=args.right_wrist_port,
                         camera_streams=('head', 'left', 'right'))
    import threading
    receiver = threading.Thread(target=images.receive_process, name='robot-image-client', daemon=True)
    receiver.start()
    hikvision = HikvisionImageClient(host=args.third_camera_host, port=args.third_camera_port).start()

    window = 'G1-D | 双臂绘画操作界面'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, WIDTH, HEIGHT)
    video_path = args.run_dir / 'operator_view.mp4'
    writer = None
    last_state = {}
    last_read = 0.0
    terminal_since = None
    try:
        while True:
            now = time.monotonic()
            if now - last_read >= 0.1:
                try:
                    last_state = json.loads((args.run_dir / 'status.json').read_text(encoding='utf-8'))
                except (OSError, ValueError):
                    pass
                last_read = now

            frames = images.get_latest()
            third = hikvision.get_bgr()
            head_right = frames.get('head_right')
            wrist_left = frames.get('left_wrist')
            wrist_right = frames.get('right_wrist')
            frame = np.full((HEIGHT, WIDTH, 3), BG, np.uint8)
            if last_state.get('status') in {'COMPLETE', 'ACTION_LIMIT', 'ERROR', 'STOPPED'}:
                terminal_since = terminal_since or now
                if now - terminal_since >= 3.0:
                    break

            # Third-person image fills a full-height rail at the far left.
            text(frame, 20, 32, '第三视角', 0.62, TEXT)
            age = hikvision.age_s()
            draw_view(frame, third, '第三视角画面', 14, 48, THIRD_W - 28, HEIGHT - 62, age)

            content_x = THIRD_W + 14
            content_w = WIDTH - content_x - 14
            gap = 12
            info_w = 450
            camera_w = (content_w - info_w - 2 * gap) // 3
            tile_h = 430
            draw_view(frame, head_right, '头部右目', content_x, 14, camera_w, tile_h,
                      frames.get('head_age_s'))
            draw_view(frame, wrist_left, '左腕相机', content_x + camera_w + gap,
                      14, camera_w, tile_h, frames.get('left_wrist_age_s'))
            draw_view(frame, wrist_right, '右腕相机', content_x + (camera_w + gap) * 2,
                      14, camera_w, tile_h, frames.get('right_wrist_age_s'))

            info_x = content_x + camera_w * 3 + gap * 3
            cv2.rectangle(frame, (info_x, 14), (WIDTH - 14, HEIGHT - 14), PANEL, -1)
            task_title = last_state.get('task_text', 'G1-D 白板任务')
            text(frame, info_x + 20, 48, task_title, 0.48, TEXT)
            status = STATUS_ZH.get(last_state.get('status'), last_state.get('status', '等待启动'))
            text(frame, info_x + 20, 91, f'状态  {status}',
                 0.49, GREEN if last_state.get('status') == 'COMPLETE' else AMBER)
            text(frame, info_x + 20, 127, f"步骤  {last_state.get('step', 0)} / {last_state.get('max_steps', 100)}",
                 0.48, CYAN)
            text(frame, info_x + 20, 175, '当前决策', 0.46, MUTED)
            skill = SKILL_ZH.get(last_state.get('skill'), last_state.get('skill', '等待决策'))
            text(frame, info_x + 20, 207, skill, 0.55, CYAN)
            text(frame, info_x + 20, 244, '机器人观测 / Astra 说明', 0.46, MUTED)
            obs = str(last_state.get('observation', last_state.get('note', '等待相机图像')))
            for i in range(5):
                text(frame, info_x + 20, 274 + i * 26, obs[i * 27:(i + 1) * 27], 0.39, TEXT)
            text(frame, info_x + 20, 440, '执行状态', 0.46, MUTED)
            text(frame, info_x + 20, 474, last_state.get('execution', '等待 agent 启动'), 0.44, TEXT)
            text(frame, info_x + 20, 518, '任务 / 步骤进度', 0.46, MUTED)
            progress = float(last_state.get('progress', 0.0) or 0.0)
            bx, by, bw = info_x + 20, 542, info_w - 42
            cv2.rectangle(frame, (bx, by), (bx + bw, by + 14), PANEL_ALT, -1)
            if progress > 0:
                cv2.rectangle(frame, (bx, by), (bx + int(bw * min(progress, 1)), by + 14), GREEN, -1)
            text(frame, info_x + 20, 583, f'{progress * 100:.0f}% 任务进度  |  {last_state.get("progress_text", "画圆")}', 0.42, TEXT)
            text(frame, info_x + 20, 650, '自主模式：Astra 决策并执行', 0.44, CYAN)
            text(frame, info_x + 20, 682, 'Q / Esc  停止', 0.44, RED)
            text(frame, info_x + 20, 733, '相机状态', 0.40, MUTED)
            text(frame, info_x + 20, 762, '头部右目 / 左腕 / 右腕 / 第三视角',
                 0.36, MUTED)

            lower_y = tile_h + 28
            lower_h = HEIGHT - lower_y - 14
            cv2.rectangle(frame, (content_x, lower_y),
                          (content_x + camera_w * 3 + gap * 2, HEIGHT - 14), PANEL, -1)
            text(frame, content_x + 18, lower_y + 36, '任务', 0.46, MUTED)
            text(frame, content_x + 18, lower_y + 74, task_title, 0.50, TEXT)
            text(frame, content_x + 18, lower_y + 125,
                 task_title, 0.45, TEXT)
            text(frame, content_x + 18, lower_y + 172,
                 'Astra 输入：头部右目、左腕、右腕、第三视角', 0.40, MUTED)

            frame = paint_text(frame)
            if writer is None:
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'),
                                         args.record_fps, (WIDTH, HEIGHT))
                if not writer.isOpened():
                    raise RuntimeError(f'Cannot create dashboard video: {video_path}')
            writer.write(frame)
            cv2.imshow(window, frame)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                (args.run_dir / 'stop_requested').touch()
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                (args.run_dir / 'stop_requested').touch()
                break
    finally:
        (args.run_dir / 'stop_requested').touch()
        if writer is not None:
            writer.release()
        images.running = False
        receiver.join(timeout=2.0)
        hikvision.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
