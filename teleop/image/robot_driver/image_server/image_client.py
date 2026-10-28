"""ZMQ image client for robot camera streams.

Two server modes:

- ``teleimager`` (default): separate JPEG PUB ports
    head 55555 (stereo 480x1280), left wrist 55556, right wrist 55557
- ``legacy``: one concatenated JPEG on a single port (default 5555)

Hikvision third-person is separate: ``hikvision_image_client.py``.

Run live preview::

    python image/robot_driver/image_server/image_client.py
    python image/robot_driver/image_server/image_client.py --host 192.168.123.164 --mode teleimager
"""

from __future__ import annotations

import argparse
import struct
import threading
import time
from collections import deque
from multiprocessing import shared_memory
from typing import Optional

import cv2
import numpy as np
import zmq


class ImageClient:
    def __init__(
        self,
        tv_img_shape=None,
        tv_img_shm_name=None,
        wrist_img_shape=None,
        wrist_img_shm_name=None,
        image_show=False,
        server_address="192.168.123.164",
        port=5555,
        mode="teleimager",
        head_port=55555,
        left_wrist_port=55556,
        right_wrist_port=55557,
        Unit_Test=False,
        camera_streams=("head", "left", "right"),
    ):
        self.running = True
        self._image_show = image_show
        self._server_address = server_address
        self._port = int(port)
        self._mode = str(mode).lower()
        self._head_port = int(head_port)
        self._left_wrist_port = int(left_wrist_port)
        self._right_wrist_port = int(right_wrist_port)
        self._camera_streams = tuple(camera_streams)
        if self._mode == "teleimager":
            invalid = set(self._camera_streams) - {"head", "left", "right"}
            if invalid or "head" not in self._camera_streams:
                raise ValueError("teleimager camera_streams must include 'head' and only known stream names")

        self.tv_img_shape = tv_img_shape
        self.wrist_img_shape = wrist_img_shape

        # Per-camera flip (legacy combined layout names kept for API compat).
        self._flip_settings = {
            "color_0": False,  # Head Left
            "color_1": False,  # Head Right
            "color_2": False,  # Wrist Left
            "color_3": False,  # Wrist Right
        }
        self._flip_lock = threading.Lock()

        self.tv_enable_shm = False
        if self.tv_img_shape is not None and tv_img_shm_name is not None:
            self.tv_image_shm = shared_memory.SharedMemory(name=tv_img_shm_name)
            self.tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=self.tv_image_shm.buf)
            self.tv_enable_shm = True

        self.wrist_enable_shm = False
        if self.wrist_img_shape is not None and wrist_img_shm_name is not None:
            self.wrist_image_shm = shared_memory.SharedMemory(name=wrist_img_shm_name)
            self.wrist_img_array = np.ndarray(
                wrist_img_shape, dtype=np.uint8, buffer=self.wrist_image_shm.buf
            )
            self.wrist_enable_shm = True

        self._latest = {
            "head": None,
            "head_left": None,
            "head_right": None,
            "left_wrist": None,
            "right_wrist": None,
            "combined": None,
        }
        self._latest_stamps = {"head": 0.0, "left": 0.0, "right": 0.0}
        self._latest_lock = threading.Lock()

        self._enable_performance_eval = Unit_Test
        if self._enable_performance_eval:
            self._init_performance_metrics()

        self._sockets: list[zmq.Socket] = []
        self._context: zmq.Context | None = None

    def set_flip(self, settings: Optional[dict] = None, **kwargs):
        with self._flip_lock:
            if settings:
                for key, value in settings.items():
                    if key in self._flip_settings:
                        self._flip_settings[key] = value
            for key, value in kwargs.items():
                if key in self._flip_settings:
                    self._flip_settings[key] = value
        print(f"[ImageClient] Flip settings updated: {self._flip_settings}")

    def get_flip_settings(self) -> dict:
        with self._flip_lock:
            return self._flip_settings.copy()

    def get_latest(self) -> dict:
        """Copy of latest decoded frames (BGR). Keys may be None until first packet."""
        now = time.monotonic()
        with self._latest_lock:
            result = {k: (None if v is None else v.copy()) for k, v in self._latest.items()}
            stamps = self._latest_stamps.copy()
        result["head_age_s"] = now - stamps["head"] if stamps["head"] else float("inf")
        result["left_wrist_age_s"] = now - stamps["left"] if stamps["left"] else float("inf")
        result["right_wrist_age_s"] = now - stamps["right"] if stamps["right"] else float("inf")
        # Keep the historical alias for clients that only display the right wrist.
        result["wrist_age_s"] = now - stamps["right"] if stamps["right"] else float("inf")
        return result

    def _init_performance_metrics(self):
        self._frame_count = 0
        self._last_frame_id = -1
        self._time_window = 1.0
        self._frame_times = deque()
        self._latencies = deque()
        self._lost_frames = 0
        self._total_frames = 0

    def _update_performance_metrics(self, timestamp, frame_id, receive_time):
        latency = receive_time - timestamp
        self._latencies.append(latency)
        while self._latencies and self._frame_times and self._latencies[0] < receive_time - self._time_window:
            self._latencies.popleft()
        self._frame_times.append(receive_time)
        while self._frame_times and self._frame_times[0] < receive_time - self._time_window:
            self._frame_times.popleft()
        expected_frame_id = self._last_frame_id + 1 if self._last_frame_id != -1 else frame_id
        if frame_id != expected_frame_id:
            lost = frame_id - expected_frame_id
            if lost < 0:
                print(f"[Image Client] Received out-of-order frame ID: {frame_id}")
            else:
                self._lost_frames += lost
        self._last_frame_id = frame_id
        self._total_frames = frame_id + 1
        self._frame_count += 1

    def _print_performance_metrics(self, receive_time):
        if self._frame_count % 30 == 0:
            real_time_fps = len(self._frame_times) / self._time_window if self._time_window > 0 else 0
            if self._latencies:
                avg_latency = sum(self._latencies) / len(self._latencies)
                max_latency = max(self._latencies)
                min_latency = min(self._latencies)
                jitter = max_latency - min_latency
            else:
                avg_latency = max_latency = min_latency = jitter = 0
            lost_frame_rate = (self._lost_frames / self._total_frames) * 100 if self._total_frames else 0
            print(
                f"[Image Client] Real-time FPS: {real_time_fps:.2f}, "
                f"Avg Latency: {avg_latency*1000:.2f} ms, Max Latency: {max_latency*1000:.2f} ms, "
                f"Min Latency: {min_latency*1000:.2f} ms, Jitter: {jitter*1000:.2f} ms, "
                f"Lost Frame Rate: {lost_frame_rate:.2f}%"
            )

    def _make_sub(self, port: int) -> zmq.Socket:
        assert self._context is not None
        sock = self._context.socket(zmq.SUB)
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.setsockopt(zmq.RCVHWM, 2)
        sock.setsockopt(zmq.RCVTIMEO, 200)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://{self._server_address}:{port}")
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sockets.append(sock)
        return sock

    @staticmethod
    def _decode_jpeg(message: bytes, unit_test: bool = False):
        timestamp = frame_id = None
        jpg_bytes = message
        if unit_test:
            header_size = struct.calcsize("dI")
            header = message[:header_size]
            jpg_bytes = message[header_size:]
            timestamp, frame_id = struct.unpack("dI", header)
        # teleimager / legacy color may be raw JPEG, or II + jpeg
        img = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            return img, timestamp, frame_id
        if len(jpg_bytes) >= 8:
            color_size, ir_size = struct.unpack("<II", jpg_bytes[:8])
            end = 8 + color_size
            if 0 < color_size and end <= len(jpg_bytes) and (
                ir_size == 0 or end + ir_size <= len(jpg_bytes)
            ):
                img = cv2.imdecode(
                    np.frombuffer(jpg_bytes[8:end], dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if img is not None:
                    return img, timestamp, frame_id
        return None, timestamp, frame_id

    def _apply_flip_regions(self, image: np.ndarray) -> np.ndarray:
        head_width = self.tv_img_shape[1] if self.tv_img_shape is not None else min(1280, image.shape[1])
        wrist_width = self.wrist_img_shape[1] if self.wrist_img_shape is not None else 640
        head_left_end = head_width // 2
        head_right_end = min(head_width, image.shape[1])
        wrist_left_end = min(head_width + wrist_width // 2, image.shape[1])
        wrist_right_end = min(head_width + wrist_width, image.shape[1])
        with self._flip_lock:
            flip = self._flip_settings.copy()
        out = image
        if flip.get("color_0") and head_left_end > 0:
            out[:, :head_left_end] = cv2.flip(out[:, :head_left_end], 1)
        if flip.get("color_1") and head_right_end > head_left_end:
            out[:, head_left_end:head_right_end] = cv2.flip(out[:, head_left_end:head_right_end], 1)
        if flip.get("color_2") and wrist_left_end > head_right_end:
            out[:, head_right_end:wrist_left_end] = cv2.flip(out[:, head_right_end:wrist_left_end], 1)
        if flip.get("color_3") and wrist_right_end > wrist_left_end:
            out[:, wrist_left_end:wrist_right_end] = cv2.flip(out[:, wrist_left_end:wrist_right_end], 1)
        return out

    def _update_shm_and_latest_from_combined(self, combined: np.ndarray) -> None:
        head_w = self.tv_img_shape[1] if self.tv_img_shape is not None else min(1280, combined.shape[1])
        head = combined[:, :head_w]
        wrists = combined[:, head_w:] if combined.shape[1] > head_w else None
        head_left = head_right = None
        if head.shape[1] >= 2:
            head_left, head_right = np.hsplit(head[:, : head.shape[1] - (head.shape[1] % 2)], 2)
        left_wrist = right_wrist = None
        if wrists is not None and wrists.shape[1] >= 2:
            left_wrist, right_wrist = np.hsplit(wrists[:, : wrists.shape[1] - (wrists.shape[1] % 2)], 2)

        if self.tv_enable_shm:
            np.copyto(self.tv_img_array, head[:, : self.tv_img_shape[1]])
        if self.wrist_enable_shm and wrists is not None:
            np.copyto(self.wrist_img_array, wrists[:, -self.wrist_img_shape[1] :])

        with self._latest_lock:
            self._latest.update(
                head=head,
                head_left=head_left,
                head_right=head_right,
                left_wrist=left_wrist,
                right_wrist=right_wrist,
                combined=combined,
            )
            stamp = time.monotonic()
            for name, image in (("head", head), ("left", left_wrist), ("right", right_wrist)):
                if image is not None:
                    self._latest_stamps[name] = stamp

    def _update_latest_teleimager(
        self,
        head: np.ndarray | None,
        left_wrist: np.ndarray | None,
        right_wrist: np.ndarray | None,
    ) -> None:
        head_left = head_right = None
        if head is not None and head.shape[1] >= 2:
            even = head[:, : head.shape[1] - (head.shape[1] % 2)]
            head_left, head_right = np.hsplit(even, 2)

        # Build a combined strip for SHM / preview: head | L wrist | R wrist
        parts = [p for p in (head, left_wrist, right_wrist) if p is not None]
        combined = None
        if parts:
            h = max(p.shape[0] for p in parts)

            def pad(img):
                if img.shape[0] == h:
                    return img
                out = np.zeros((h, img.shape[1], 3), dtype=np.uint8)
                out[: img.shape[0]] = img
                return out

            combined = np.hstack([pad(p) for p in parts])

        if self.tv_enable_shm and head is not None:
            w = self.tv_img_shape[1]
            canvas = np.zeros(self.tv_img_shape, dtype=np.uint8)
            hh, ww = head.shape[:2]
            canvas[: min(hh, canvas.shape[0]), : min(ww, w)] = head[
                : min(hh, canvas.shape[0]), : min(ww, w)
            ]
            np.copyto(self.tv_img_array, canvas)
        if self.wrist_enable_shm:
            wl = left_wrist if left_wrist is not None else np.zeros((480, 640, 3), np.uint8)
            wr = right_wrist if right_wrist is not None else np.zeros((480, 640, 3), np.uint8)
            strip = np.hstack([wl, wr])
            canvas = np.zeros(self.wrist_img_shape, dtype=np.uint8)
            hh, ww = strip.shape[:2]
            canvas[: min(hh, canvas.shape[0]), : min(ww, canvas.shape[1])] = strip[
                : min(hh, canvas.shape[0]), : min(ww, canvas.shape[1])
            ]
            np.copyto(self.wrist_img_array, canvas)

        with self._latest_lock:
            self._latest.update(
                head=head,
                head_left=head_left,
                head_right=head_right,
                left_wrist=left_wrist,
                right_wrist=right_wrist,
                combined=combined,
            )

    def _show_frame(self, combined: np.ndarray | None) -> None:
        if not self._image_show or combined is None:
            return
        height, width = combined.shape[:2]
        scale = 0.5 if width >= 1280 else 1.0
        view = cv2.resize(combined, (max(1, int(width * scale)), max(1, int(height * scale))))
        cv2.imshow("Image Client Stream", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.running = False

    def _close(self):
        for sock in self._sockets:
            try:
                sock.close(0)
            except Exception:
                pass
        self._sockets.clear()
        if self._context is not None:
            self._context.term()
            self._context = None
        if self._image_show:
            cv2.destroyAllWindows()
        print("Image client has been closed.")

    def receive_process(self):
        self._context = zmq.Context()
        try:
            if self._mode == "legacy":
                self._receive_legacy()
            else:
                self._receive_teleimager()
        except KeyboardInterrupt:
            print("Image client interrupted by user.")
        except Exception as e:
            print(f"[Image Client] An error occurred while receiving data: {e}")
        finally:
            self._close()

    def _receive_legacy(self):
        sock = self._make_sub(self._port)
        print(f"\nImage client has started, connecting to {self._server_address}:{self._port} (legacy)...")
        while self.running:
            try:
                message = sock.recv()
            except zmq.Again:
                continue
            receive_time = time.time()
            try:
                current_image, timestamp, frame_id = self._decode_jpeg(
                    message, self._enable_performance_eval
                )
            except struct.error as e:
                print(f"[Image Client] Error unpacking header: {e}, discarding message.")
                continue
            if current_image is None:
                print("[Image Client] Failed to decode image.")
                continue
            current_image = self._apply_flip_regions(current_image)
            self._update_shm_and_latest_from_combined(current_image)
            self._show_frame(current_image)
            if self._enable_performance_eval and timestamp is not None and frame_id is not None:
                self._update_performance_metrics(timestamp, frame_id, receive_time)
                self._print_performance_metrics(receive_time)

    def _receive_teleimager(self):
        ports = {"head": self._head_port, "left": self._left_wrist_port,
                 "right": self._right_wrist_port}
        sockets = [(self._make_sub(ports[slot]), slot) for slot in self._camera_streams]
        print(
            f"\nImage client has started (teleimager): "
            f"{self._server_address} "
            + " ".join(f"{slot}={ports[slot]}" for slot in self._camera_streams)
        )
        print("Press q in the window to quit.")

        head = left = right = None
        while self.running:
            updated = False
            for sock, slot in sockets:
                try:
                    message = sock.recv()
                except zmq.Again:
                    continue
                img, _, _ = self._decode_jpeg(message, False)
                if img is None:
                    print(f"[Image Client] Failed to decode {slot} JPEG.")
                    continue
                if slot == "head":
                    head = img
                elif slot == "left":
                    left = img
                else:
                    right = img
                with self._latest_lock:
                    self._latest_stamps[slot] = time.monotonic()
                updated = True

            if not updated:
                if self._image_show:
                    # keep UI responsive while waiting
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.running = False
                continue

            # Optional per-half flips on head stereo / wrists
            with self._flip_lock:
                flip = self._flip_settings.copy()
            if head is not None and head.shape[1] >= 2:
                mid = head.shape[1] // 2
                if flip.get("color_0"):
                    head[:, :mid] = cv2.flip(head[:, :mid], 1)
                if flip.get("color_1"):
                    head[:, mid:] = cv2.flip(head[:, mid:], 1)
            if left is not None and flip.get("color_2"):
                left = cv2.flip(left, 1)
            if right is not None and flip.get("color_3"):
                right = cv2.flip(right, 1)

            self._update_latest_teleimager(head, left, right)
            with self._latest_lock:
                combined = None if self._latest["combined"] is None else self._latest["combined"]
            self._show_frame(combined)


def _parse_args():
    p = argparse.ArgumentParser(description="Robot image client live preview")
    p.add_argument("--host", default="192.168.123.164")
    p.add_argument(
        "--mode",
        default="teleimager",
        choices=("teleimager", "legacy"),
        help="teleimager=3 PUB ports; legacy=single concatenated port",
    )
    p.add_argument("--port", type=int, default=5555, help="legacy single port")
    p.add_argument("--head-port", type=int, default=55555)
    p.add_argument("--left-wrist-port", type=int, default=55556)
    p.add_argument("--right-wrist-port", type=int, default=55557)
    p.add_argument("--unit-test", action="store_true", help="legacy timing header metrics")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    client = ImageClient(
        image_show=True,
        server_address=args.host,
        port=args.port,
        mode=args.mode,
        head_port=args.head_port,
        left_wrist_port=args.left_wrist_port,
        right_wrist_port=args.right_wrist_port,
        Unit_Test=args.unit_test,
    )
    client.receive_process()
