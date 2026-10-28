#!/usr/bin/env python3
"""Hikvision third-person image client for agent code.

Subscribe to ``hikvision_image_server.py`` (default ``tcp://127.0.0.1:5560``).
Robot head/wrist cameras stay in ``image_client.py`` (separate).

Agent usage::

    from hikvision_image_client import HikvisionImageClient

    with HikvisionImageClient(host="127.0.0.1", port=5560) as cam:
        rgb = cam.wait_rgb(timeout_s=2.0)
        snap = cam.get_latest()  # bgr / rgb / age_s / ok

Live preview::

    python image/robot_driver/image_server/hikvision_image_client.py --show
"""

from __future__ import annotations

import argparse
import threading
import time
from typing import Optional

import cv2
import numpy as np
import zmq


class HikvisionImageClient:
    """Background ZMQ SUB client for the local Hikvision JPEG stream."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5560,
        image_show: bool = False,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.endpoint = f"tcp://{host}:{self.port}"
        self._image_show = bool(image_show)
        self._lock = threading.Lock()
        self._latest_bgr: Optional[np.ndarray] = None
        self._latest_stamp: float = 0.0
        self._frame_id = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._ok = False

    def start(self) -> "HikvisionImageClient":
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="hikvision-image-sub", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._image_show:
            try:
                cv2.destroyWindow("Hikvision Client Stream")
            except Exception:
                pass

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> "HikvisionImageClient":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    @property
    def ready(self) -> bool:
        return self._ok

    @property
    def frame_id(self) -> int:
        return self._frame_id

    def age_s(self) -> float:
        with self._lock:
            if self._latest_stamp <= 0:
                return float("inf")
            return time.monotonic() - self._latest_stamp

    def get_bgr(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_bgr is None:
                return None
            return self._latest_bgr.copy()

    def get_rgb(self, timeout_s: float = 0.0) -> Optional[np.ndarray]:
        bgr = self.wait_bgr(timeout_s=timeout_s) if timeout_s > 0 else self.get_bgr()
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def wait_bgr(self, timeout_s: float = 2.0) -> Optional[np.ndarray]:
        deadline = time.monotonic() + max(timeout_s, 0.0)
        while time.monotonic() < deadline:
            frame = self.get_bgr()
            if frame is not None:
                return frame
            time.sleep(0.02)
        return self.get_bgr()

    def wait_rgb(self, timeout_s: float = 2.0) -> Optional[np.ndarray]:
        bgr = self.wait_bgr(timeout_s=timeout_s)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def get_latest(self) -> dict:
        with self._lock:
            bgr = None if self._latest_bgr is None else self._latest_bgr.copy()
            stamp = self._latest_stamp
            fid = self._frame_id
        rgb = None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return {
            "bgr": bgr,
            "rgb": rgb,
            "frame_id": fid,
            "stamp_monotonic": stamp,
            "age_s": (time.monotonic() - stamp) if stamp > 0 else float("inf"),
            "endpoint": self.endpoint,
            "ok": self._ok,
        }

    def _run(self) -> None:
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.RCVHWM, 2)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 200)
        try:
            sock.setsockopt(zmq.CONFLATE, 1)
        except Exception:
            pass
        try:
            sock.connect(self.endpoint)
        except Exception as exc:
            print(f"[hikvision-client] connect failed {self.endpoint}: {exc}")
            self._running = False
            sock.close(0)
            return
        print(f"[hikvision-client] SUB {self.endpoint}")
        while self._running:
            try:
                message = sock.recv()
            except zmq.Again:
                if self._image_show and cv2.waitKey(1) & 0xFF == ord("q"):
                    self._running = False
                continue
            except zmq.ZMQError:
                if not self._running:
                    break
                continue
            img = cv2.imdecode(np.frombuffer(message, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            with self._lock:
                self._latest_bgr = img
                self._latest_stamp = time.monotonic()
                self._frame_id += 1
                self._ok = True
            if self._image_show:
                cv2.imshow("Hikvision Client Stream", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._running = False
        sock.close(0)
        print("[hikvision-client] closed")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5560)
    p.add_argument(
        "--show",
        dest="show",
        action="store_true",
        default=True,
        help="live preview window (default: on)",
    )
    p.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="smoke test only: one frame then exit",
    )
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--save", default="")
    args = p.parse_args()

    client = HikvisionImageClient(host=args.host, port=args.port, image_show=args.show)
    client.start()
    frame = client.wait_bgr(timeout_s=args.timeout)
    if frame is None:
        print(f"FAIL: no frame from {client.endpoint} within {args.timeout}s")
        print("Start server: python .../hikvision_image_server.py --device /dev/video1")
        client.stop()
        return 1
    print(f"OK shape={frame.shape} age={client.age_s():.3f}s endpoint={client.endpoint}")
    if args.save:
        cv2.imwrite(args.save, frame)
        print(f"saved {args.save}")
    if args.show:
        print("live preview — press q in the window (or Ctrl+C) to quit")
        try:
            while client._running:
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
    client.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
