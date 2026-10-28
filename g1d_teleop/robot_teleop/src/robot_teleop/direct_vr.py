from __future__ import annotations

from dataclasses import dataclass, field
import json
import socket
import struct
import threading
import time
from typing import Optional

import numpy as np


@dataclass
class BodySample:
    poses: np.ndarray = field(default_factory=lambda: np.zeros((24, 7), dtype=float))
    timestamp_ns: int = 0
    available: bool = False
    head_pose: np.ndarray | None = None


@dataclass
class ButtonState:
    x_pressed: bool = False
    a_pressed: bool = False
    b_pressed: bool = False
    y_pressed: bool = False
    axis_l: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    axis_r: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    left_trigger: float = 0.0
    right_trigger: float = 0.0
    left_grip: float = 0.0
    right_grip: float = 0.0


class DirectVRReader:
    """DirectVR reader for PICO / XRoboToolkit headsets.

    Listens on TCP 63901 for device-state JSON and broadcasts discovery packets
    on UDP 29888 so the headset can find this host. Incoming body data is an
    array of XRoboToolkit poses [x, y, z, qx, qy, qz, qw]; no xrobotoolkit
    dependency is required.

    `latest_body()` reports `available=False` until either a full 24-joint body
    array or both controller poses have been parsed. Buttons and triggers are
    updated independently, so grippers keep working even without body tracking.
    """

    SERVER_HEAD = 0xCF
    CLIENT_HEAD = 0x3F
    TAIL = 0xA5
    UDP_TCP_IP_CMD = 0x7E
    CLIENT_DEVICE_STATE_JSON = 0x6D

    def __init__(self, tcp_port: int = 63901, broadcast_port: int = 29888, head_pose_index: int = 0) -> None:
        self.tcp_port = tcp_port
        self.broadcast_port = broadcast_port
        self.head_pose_index = head_pose_index
        self._body = BodySample()
        self._buttons = ButtonState()
        self._lock = threading.Lock()
        self._thread_lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._listen_sock: Optional[socket.socket] = None
        self._client_sockets: set[socket.socket] = set()

    def start(self) -> None:
        self._stop.clear()
        tcp_thread = threading.Thread(target=self._tcp_loop, daemon=True)
        udp_thread = threading.Thread(target=self._udp_broadcast_loop, daemon=True)
        with self._thread_lock:
            self._threads = [tcp_thread, udp_thread]
        tcp_thread.start()
        udp_thread.start()

    def stop(self) -> None:
        self._stop.set()
        listen_sock = self._listen_sock
        if listen_sock is not None:
            try:
                listen_sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                listen_sock.close()
            except OSError:
                pass
        with self._thread_lock:
            client_sockets = list(self._client_sockets)
            threads = list(self._threads)
        for client in client_sockets:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                client.close()
            except OSError:
                pass
        current = threading.current_thread()
        for thread in threads:
            if thread is not current:
                thread.join(timeout=0.2)
        with self._thread_lock:
            self._threads.clear()
            self._client_sockets.clear()
        self._listen_sock = None

    def latest_body(self) -> BodySample:
        with self._lock:
            head_pose = None if self._body.head_pose is None else self._body.head_pose.copy()
            return BodySample(self._body.poses.copy(), self._body.timestamp_ns, self._body.available, head_pose)

    def latest_buttons(self) -> ButtonState:
        with self._lock:
            return ButtonState(
                self._buttons.x_pressed,
                self._buttons.a_pressed,
                self._buttons.b_pressed,
                self._buttons.y_pressed,
                self._buttons.axis_l.copy(),
                self._buttons.axis_r.copy(),
                self._buttons.left_trigger,
                self._buttons.right_trigger,
                self._buttons.left_grip,
                self._buttons.right_grip,
            )

    def _tcp_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listen_sock = sock
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", self.tcp_port))
            sock.listen(8)
            sock.settimeout(0.2)
            print(f"[DirectVR] listening on TCP 0.0.0.0:{self.tcp_port}")
            while not self._stop.is_set():
                try:
                    client, addr = sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                print(f"[DirectVR] client connected: {addr}")
                thread = threading.Thread(target=self._client_loop, args=(client,), daemon=True)
                with self._thread_lock:
                    self._threads.append(thread)
                thread.start()
        finally:
            try:
                sock.close()
            except OSError:
                pass

    def _client_loop(self, client: socket.socket) -> None:
        buffer = bytearray()
        with self._thread_lock:
            self._client_sockets.add(client)
        try:
            with client:
                client.settimeout(0.2)
                while not self._stop.is_set():
                    try:
                        data = client.recv(65536)
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    if not data:
                        break
                    buffer.extend(data)
                    self._parse_buffer(buffer)
        finally:
            with self._thread_lock:
                self._client_sockets.discard(client)

    def _parse_buffer(self, buffer: bytearray) -> None:
        header_size = struct.calcsize("<BBI")
        tail_size = struct.calcsize("<QB")
        while len(buffer) >= header_size + tail_size:
            if buffer[0] != self.CLIENT_HEAD:
                del buffer[0]
                continue
            _, cmd, length = struct.unpack_from("<BBI", buffer, 0)
            packet_size = header_size + length + tail_size
            if len(buffer) < packet_size:
                return
            tail = buffer[packet_size - 1]
            if tail != self.TAIL:
                del buffer[0]
                continue
            payload = bytes(buffer[header_size : header_size + length]).decode("utf-8", errors="ignore")
            del buffer[:packet_size]
            if cmd == self.CLIENT_DEVICE_STATE_JSON:
                self._handle_state_json(payload)

    def _handle_state_json(self, payload: str) -> None:
        try:
            data = json.loads(payload)
            value = data.get("value", data)
            if isinstance(value, str):
                value = json.loads(value)
        except json.JSONDecodeError:
            return
        if not isinstance(value, dict):
            return

        def pick_float(source: dict, *names: str) -> float:
            for name in names:
                raw = source.get(name)
                if raw is not None:
                    return float(raw or 0.0)
            return 0.0

        def pick_bool(source: dict, *names: str) -> bool:
            for name in names:
                raw = source.get(name)
                if raw is not None:
                    return bool(raw)
            return False

        buttons = self.latest_buttons()
        controller = value.get("Controller", value.get("controller", {})) if isinstance(value, dict) else {}
        left = controller.get("left", controller.get("Left", {})) if isinstance(controller, dict) else {}
        right = controller.get("right", controller.get("Right", {})) if isinstance(controller, dict) else {}
        head_source = value.get("Headset", value.get("headset", value.get("Head", value.get("head", {}))))
        buttons.left_trigger = pick_float(left, "trigger", "Trigger", "triggerValue", "indexTrigger")
        buttons.left_grip = pick_float(left, "grip", "Grip", "gripValue", "squeeze", "gripButton")
        buttons.axis_l[:] = [pick_float(left, "axisX", "axis_x", "thumbstickX"), pick_float(left, "axisY", "axis_y", "thumbstickY")]
        buttons.x_pressed = pick_bool(left, "primaryButton", "xButton", "X")
        buttons.y_pressed = pick_bool(left, "secondaryButton", "yButton", "Y")
        buttons.right_trigger = pick_float(right, "trigger", "Trigger", "triggerValue", "indexTrigger")
        buttons.right_grip = pick_float(right, "grip", "Grip", "gripValue", "squeeze", "gripButton")
        buttons.axis_r[:] = [pick_float(right, "axisX", "axis_x", "thumbstickX"), pick_float(right, "axisY", "axis_y", "thumbstickY")]
        buttons.a_pressed = pick_bool(right, "primaryButton", "aButton", "A")
        buttons.b_pressed = pick_bool(right, "secondaryButton", "bButton", "B")

        body = self.latest_body()
        body_json = value.get("Body", value.get("body", {})) if isinstance(value, dict) else {}
        joints = body_json.get("joints", body_json.get("Joints", [])) if isinstance(body_json, dict) else []
        if isinstance(joints, list) and len(joints) >= 24:
            for i, joint in enumerate(joints[:24]):
                pose = self._extract_pose7(joint, "p", "pose", "Pose", "position", "Position") if isinstance(joint, dict) else None
                if pose is not None:
                    body.poses[i] = pose
            body.timestamp_ns = int(value.get("timeStampNs", body_json.get("timeStampNs", 0)) or 0)
            body.available = True
        else:
            left_pose = self._extract_controller_pose7(left)
            right_pose = self._extract_controller_pose7(right)
            if left_pose is not None and right_pose is not None:
                body.poses[20] = left_pose
                body.poses[21] = right_pose
                body.timestamp_ns = int(value.get("timeStampNs", body_json.get("timeStampNs", 0)) or 0)
                body.available = True

        if isinstance(head_source, dict):
            body.head_pose = self._extract_controller_pose7(head_source)
        if body.head_pose is None and body.available and 0 <= self.head_pose_index < len(body.poses):
            body.head_pose = body.poses[self.head_pose_index].copy()

        with self._lock:
            self._buttons = buttons
            self._body = body

    def _extract_controller_pose7(self, source: dict) -> np.ndarray | None:
        pose = self._extract_pose7(source, "pose", "Pose", "p")
        if pose is not None:
            return pose
        pos = self._extract_vec3(source, "position", "Position", "pos", "translation", "localPosition")
        quat = self._extract_quat_xyzw(source, "rotation", "Rotation", "orientation", "quat", "localRotation")
        if pos is not None and quat is not None:
            return np.concatenate([pos, quat])
        return None

    def _extract_pose7(self, source: dict, *names: str) -> np.ndarray | None:
        for name in names:
            raw = source.get(name)
            if raw is None:
                continue
            if isinstance(raw, str):
                vals = np.fromstring(raw, sep=",", dtype=float)
                if vals.size == 7:
                    return vals
            if isinstance(raw, list) and len(raw) >= 7:
                return np.asarray(raw[:7], dtype=float)
            if isinstance(raw, dict):
                pos = self._extract_vec3(raw, "pos", "position", "translation", "localPosition")
                quat = self._extract_quat_xyzw(raw, "quat", "rotation", "orientation", "localRotation")
                if pos is not None and quat is not None:
                    return np.concatenate([pos, quat])
        return None

    def _extract_vec3(self, source: dict, *names: str) -> np.ndarray | None:
        for name in names:
            raw = source.get(name)
            if isinstance(raw, str):
                vals = np.fromstring(raw, sep=",", dtype=float)
                if vals.size >= 3:
                    return vals[:3]
            if isinstance(raw, list) and len(raw) >= 3:
                return np.asarray(raw[:3], dtype=float)
            if isinstance(raw, dict):
                if all(k in raw for k in ("x", "y", "z")):
                    return np.asarray([raw["x"], raw["y"], raw["z"]], dtype=float)
        if all(k in source for k in ("x", "y", "z")):
            return np.asarray([source["x"], source["y"], source["z"]], dtype=float)
        return None

    def _extract_quat_xyzw(self, source: dict, *names: str) -> np.ndarray | None:
        for name in names:
            raw = source.get(name)
            if isinstance(raw, str):
                vals = np.fromstring(raw, sep=",", dtype=float)
                if vals.size >= 4:
                    return vals[:4]
            if isinstance(raw, list) and len(raw) >= 4:
                return np.asarray(raw[:4], dtype=float)
            if isinstance(raw, dict):
                if all(k in raw for k in ("x", "y", "z", "w")):
                    return np.asarray([raw["x"], raw["y"], raw["z"], raw["w"]], dtype=float)
        if all(k in source for k in ("qx", "qy", "qz", "qw")):
            return np.asarray([source["qx"], source["qy"], source["qz"], source["qw"]], dtype=float)
        return None

    def _udp_broadcast_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        while not self._stop.is_set():
            for ip, broadcast in self._broadcast_targets():
                packet = self._make_server_packet(self.UDP_TCP_IP_CMD, ip.encode())
                try:
                    sock.sendto(packet, (broadcast, self.broadcast_port))
                except OSError:
                    pass
            self._stop.wait(5.0)
        sock.close()

    def _broadcast_targets(self) -> list[tuple[str, str]]:
        targets: list[tuple[str, str]] = []
        try:
            hostname = socket.gethostname()
            for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
                ip = info[4][0]
                if ip.startswith("127."):
                    continue
                parts = ip.split(".")
                if len(parts) == 4:
                    targets.append((ip, ".".join(parts[:3] + ["255"])))
        except OSError:
            pass
        if not targets:
            targets.append(("0.0.0.0", "255.255.255.255"))
        return targets

    def _make_server_packet(self, cmd: int, payload: bytes) -> bytes:
        head = struct.pack("<BBI", self.SERVER_HEAD, cmd, len(payload))
        tail = struct.pack("<QB", int(time.time()), self.TAIL)
        return head + payload + tail
