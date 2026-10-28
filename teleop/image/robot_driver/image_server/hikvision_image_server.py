#!/usr/bin/env python3
"""Local Hikvision (USB V4L2) image server for agent / operator third-person view.

Captures from the workstation USB camera and publishes JPEG frames over ZMQ PUB.
Default port **5560** (keeps teleimager 55555–55557 free).

Run on the PC that has the Hikvision plugged in::

    conda activate robot_teleop
    cd ~/ycb_ws/unitree_robot_teleop
    python image/robot_driver/image_server/hikvision_image_server.py
    python image/robot_driver/image_server/hikvision_image_server.py --device /dev/video0 --port 5560

Wire format: raw JPEG bytes (same style as teleimager single-camera PUB).
"""

from __future__ import annotations

import argparse
import signal
import time

import cv2
import numpy as np
import zmq


def _device_candidates(device: str | int) -> list[str | int]:
    """Prefer real /dev paths; UVC often exposes video0=metadata, video1=capture."""
    import glob
    import os

    raw = str(device)
    out: list[str | int] = []

    def add(x: str | int) -> None:
        if x not in out:
            out.append(x)

    existing = sorted(glob.glob("/dev/video*"))

    if raw in {"auto", ""}:
        # Only real nodes — do not invent /dev/video0 when the box has video1/video2.
        for path in existing:
            add(path)
        return out

    if raw.isdigit():
        idx = int(raw)
        add(f"/dev/video{idx}")
        add(f"/dev/video{idx + 1}")
        if idx > 0:
            add(f"/dev/video{idx - 1}")
        # Prefer path opens; numeric index only as last resorts matching existing nodes.
        for path in existing:
            add(path)
        add(idx)
        return out

    if raw.startswith("/dev/"):
        add(raw)
        # UVC sibling (metadata vs capture)
        if raw[-1].isdigit():
            twin = raw[:-1] + str((int(raw[-1]) + 1) % 10)
            add(twin)
    else:
        add(raw)

    for path in existing:
        add(path)
    return out


def _open_hint(device: str | int) -> str:
    import glob
    import os
    import subprocess

    nodes = sorted(glob.glob("/dev/video*"))
    lines = [
        f"cannot open Hikvision device {device!r}",
        f"  seen nodes: {nodes or '(none)'}",
        f"  user={os.getuid()} groups={os.getgroups()}",
    ]
    try:
        import grp

        in_video = os.getuid() == 0 or any(
            grp.getgrgid(g).gr_name == "video" for g in os.getgroups()
        )
        lines.append(f"  in 'video' group: {in_video}")
    except Exception:
        pass

    # Who holds the camera? (most common failure after a leftover server)
    holders = []
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", "hikvision_image_server|opencv|ffmpeg|guvcview|cheese"],
            text=True,
            timeout=2,
        ).strip()
        if out:
            holders.append(out)
    except Exception:
        pass
    if nodes:
        try:
            cmd = ["fuser", "-v", *nodes]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
            text = (out.stdout or "") + (out.stderr or "")
            if text.strip():
                holders.append(text.strip())
        except Exception:
            pass
    if holders:
        lines.append("  possible holders (kill these if stale):")
        for block in holders:
            lines.extend("    " + x for x in block.splitlines()[:12])
        lines.append("  fix busy:  pkill -f hikvision_image_server.py")

    lines.append("  try:  python .../hikvision_image_server.py --device auto")
    lines.append("  or:   python .../hikvision_image_server.py --device /dev/video1")
    lines.append("  fix perms:  sudo usermod -aG video $USER && newgrp video")
    return "\n".join(lines)


def open_capture(device: str | int, width: int, height: int, fps: float) -> cv2.VideoCapture:
    last_err = ""
    for cand in _device_candidates(device):
        for api, api_name in ((cv2.CAP_V4L2, "V4L2"), (cv2.CAP_ANY, "ANY")):
            cap = cv2.VideoCapture(cand, api)
            if not cap.isOpened():
                cap.release()
                last_err = f"{cand!r}/{api_name}: open failed"
                continue
            if width > 0:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height > 0:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps > 0:
                cap.set(cv2.CAP_PROP_FPS, fps)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                last_err = f"{cand!r}/{api_name}: opened but read failed"
                continue
            print(
                f"[hikvision-server] device={cand!r} api={api_name} "
                f"shape={frame.shape} prop_fps={cap.get(cv2.CAP_PROP_FPS):.1f}"
            )
            return cap
    raise RuntimeError(_open_hint(device) + (f"\n  last: {last_err}" if last_err else ""))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--device",
        default="auto",
        help="V4L2 path/index, or 'auto' to probe /dev/video*",
    )
    p.add_argument("--bind", default="0.0.0.0", help="ZMQ bind host")
    p.add_argument("--port", type=int, default=5560, help="ZMQ PUB port")
    p.add_argument("--width", type=int, default=0, help="requested width (0=driver default)")
    p.add_argument("--height", type=int, default=0, help="requested height (0=driver default)")
    p.add_argument("--fps", type=float, default=30.0, help="publish rate cap")
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--show", action="store_true", help="local preview window")
    args = p.parse_args()

    cap = open_capture(args.device, args.width, args.height, args.fps)
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 2)
    sock.setsockopt(zmq.LINGER, 0)
    endpoint = f"tcp://{args.bind}:{args.port}"
    sock.bind(endpoint)
    print(f"[hikvision-server] PUB {endpoint}  (Ctrl+C to stop)")

    running = True

    def _stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    period = 1.0 / max(args.fps, 1e-3)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(args.jpeg_quality, 30, 95))]
    frames = 0
    t0 = time.monotonic()
    next_t = t0

    try:
        while running:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            ok_enc, buf = cv2.imencode(".jpg", frame, encode_param)
            if not ok_enc:
                continue
            try:
                sock.send(buf.tobytes(), zmq.NOBLOCK)
            except zmq.Again:
                pass
            frames += 1
            if args.show:
                cv2.imshow("Hikvision Server Preview", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    running = False
            now = time.monotonic()
            if now - t0 >= 2.0:
                print(f"[hikvision-server] ~{frames / (now - t0):.1f} FPS published")
                frames = 0
                t0 = now
            next_t += period
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.monotonic()
    finally:
        cap.release()
        sock.close(0)
        if args.show:
            cv2.destroyAllWindows()
        print("[hikvision-server] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
