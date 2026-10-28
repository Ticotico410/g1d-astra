"""Record and replay support for the legacy single-image MCAP format.

Writes every frame's commands (actions), measured state (states) and camera
images to MCAP:
- Uses the official mcap Python library with plain JSON payloads (json /
  jsonschema encoding), readable by the mcap CLI and Foxglove.
- Images arrive over ZMQ from image_server (PUB, JPEG). Frames are not decoded
  here; the raw JPEG is stored base64-encoded.
- RecordingPublisher is a pass-through decorator around the real publisher, so
  the control loop is untouched.

"""

from __future__ import annotations

import base64
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import TeleopConfig
from .publisher import CommandPublisher


# ---------------------------------------------------------------------------
# Speech announcements through the robot's built-in speaker
# ---------------------------------------------------------------------------
class TtsSpeaker:
    """Announce recording events through the G1 AudioClient from a worker thread.

    TtsMaker blocks until the robot finishes speaking, so calls are queued on a
    dedicated thread and never stall the control or recording loops. Requires an
    initialized ChannelFactory (the DDS publisher does this in robot mode); in
    simulation, or if setup fails, it degrades to printing.
    """

    def __init__(self, enabled: bool = True, speaker_id: int = 0) -> None:
        self._enabled = bool(enabled)
        self._speaker_id = int(speaker_id)
        self._client = None
        self._queue: "queue.Queue[str | None]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if not self._enabled:
            return
        try:
            from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

            client = AudioClient()
            client.SetTimeout(3.0)
            client.Init()
            self._client = client
        except Exception as exc:  # no SDK, uninitialized ChannelFactory, or sim
            print(f"[record][TTS] disabled, setup failed: {exc}")
            self._client = None
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def push(self, msg: str) -> None:
        if self._client is None:
            print(f"[record][TTS](off) {msg}")
            return
        self._queue.put(msg)

    def _loop(self) -> None:
        while self._running:
            try:
                msg = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if msg is None:
                break
            try:
                print(f"[record][TTS] {msg}")
                self._client.TtsMaker(msg, self._speaker_id)
            except Exception as exc:
                print(f"[record][TTS] announcement failed: {exc}")

    def close(self) -> None:
        self._running = False
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Image receiver: ZMQ SUB to image_server
# ---------------------------------------------------------------------------
class ImageZMQReceiver:
    """Subscribe to JPEG frames from image_server on a worker thread.

    standard (USB cameras): the whole message is one combined JPEG
        -> {"color_0": bytes}
    dex (RealSense): color_size(4) + ir_size(4) + color + ir
        -> {"color_0": ..., "ir_0": ...}

    Degrades safely when pyzmq is missing or the connection fails: snapshot()
    returns {} and joint/state recording continues unaffected.
    """

    _DEX_CANDIDATES = ((8, 0, 4), (20, 12, 16))  # (header_size, color_size_off, ir_size_off)

    def __init__(self, host: str, port: int, client_type: str = "standard") -> None:
        self._host = host
        self._port = int(port)
        self._client_type = str(client_type).lower()
        self._latest: dict[str, bytes] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._ok = False

    def start(self) -> None:
        try:
            import zmq  # noqa: F401
        except Exception as exc:  # pragma: no cover - env dependent
            print(f"[record] image recording disabled, pyzmq unavailable ({exc})")
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import zmq

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.RCVHWM, 2)
        sock.setsockopt(zmq.LINGER, 0)
        try:
            sock.setsockopt(zmq.CONFLATE, 1)
        except Exception:
            pass
        endpoint = f"tcp://{self._host}:{self._port}"
        try:
            sock.connect(endpoint)
        except Exception as exc:
            print(f"[record] image subscribe failed {endpoint}: {exc}")
            sock.close(0)
            return
        print(f"[record] image client ready: {endpoint} | mode={self._client_type}")
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        while self._running:
            try:
                events = dict(poller.poll(100))
            except Exception:
                break
            if sock not in events:
                continue
            try:
                msg = sock.recv(zmq.NOBLOCK)
            except Exception:
                continue
            parsed = self._parse(bytes(msg))
            if parsed:
                self._ok = True
                with self._lock:
                    self._latest = parsed
        sock.close(0)

    def _parse(self, message: bytes) -> dict[str, bytes]:
        if not message:
            return {}
        if self._client_type == "dex":
            return self._parse_dex(message)
        return {"color_0": message}

    def _parse_dex(self, message: bytes) -> dict[str, bytes]:
        for header_size, color_off, ir_off in self._DEX_CANDIDATES:
            if len(message) < header_size:
                continue
            color_size = int.from_bytes(message[color_off:color_off + 4], "little")
            ir_size = int.from_bytes(message[ir_off:ir_off + 4], "little")
            color_start = header_size
            color_end = color_start + color_size
            ir_end = color_end + ir_size
            if color_size == 0 or color_end > len(message) or ir_end > len(message):
                continue
            out = {"color_0": message[color_start:color_end]}
            if ir_size > 0:
                out["ir_0"] = message[color_end:ir_end]
            return out
        return {}

    def snapshot(self) -> dict[str, bytes]:
        with self._lock:
            return dict(self._latest)

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)


# ---------------------------------------------------------------------------
# MCAP writing
# ---------------------------------------------------------------------------
_META_SCHEMA = {
    "type": "object",
    "title": "teleop.meta",
    "properties": {"kind": {"type": "string"}},
}
_FRAME_SCHEMA = {
    "type": "object",
    "title": "teleop.frame",
    "properties": {
        "idx": {"type": "integer"},
        "t_ns": {"type": "integer"},
        "actions": {"type": "object"},
        "states": {"type": "object"},
        "images": {"type": "object"},
    },
}


class MCAPRecorder:
    """Write metadata plus per-frame data into one MCAP file (JSON payloads)."""

    def __init__(self, path: Path) -> None:
        from mcap.writer import Writer

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "wb")
        self._writer = Writer(self._file)
        self._writer.start(profile="", library="unitree_robot_teleop record 1.0")
        meta_schema = self._writer.register_schema(
            name="teleop.meta", encoding="jsonschema", data=json.dumps(_META_SCHEMA).encode()
        )
        frame_schema = self._writer.register_schema(
            name="teleop.frame", encoding="jsonschema", data=json.dumps(_FRAME_SCHEMA).encode()
        )
        self._meta_channel = self._writer.register_channel(
            topic="/episode/meta", message_encoding="json", schema_id=meta_schema
        )
        self._frame_channel = self._writer.register_channel(
            topic="/teleop/frame", message_encoding="json", schema_id=frame_schema
        )
        self._seq = 0
        self._frame_count = 0
        self._start_ns = time.time_ns()
        self._closed = False

    def write_meta(self, payload: dict) -> None:
        t = time.time_ns()
        self._writer.add_message(
            channel_id=self._meta_channel,
            log_time=t,
            publish_time=t,
            sequence=0,
            data=json.dumps(payload, ensure_ascii=False).encode(),
        )

    def write_frame(self, t_ns: int, actions: dict, states: dict, images: dict) -> None:
        payload = {
            "idx": self._frame_count,
            "t_ns": int(t_ns),
            "actions": actions,
            "states": states,
            "images": images,
        }
        self._writer.add_message(
            channel_id=self._frame_channel,
            log_time=int(t_ns),
            publish_time=int(t_ns),
            sequence=self._seq,
            data=json.dumps(payload, ensure_ascii=False).encode(),
        )
        self._seq += 1
        self._frame_count += 1

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        duration_s = (time.time_ns() - self._start_ns) / 1e9
        try:
            self.write_meta(
                {"kind": "summary", "total_frames": self._frame_count, "duration_s": duration_s}
            )
            self._writer.finish()
        finally:
            self._file.close()
        print(f"[record] saved {self._frame_count} frames, {duration_s:.1f}s -> {self.path}")


def _next_episode_path(cfg: TeleopConfig) -> Path:
    day = datetime.now().strftime("%Y_%m_%d")
    base = Path(cfg.record_dir).expanduser() / cfg.record_task_name / day
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(base.glob("episode_*.mcap"))
    idx = 0
    if existing:
        try:
            idx = max(int(p.stem.split("_")[-1]) for p in existing) + 1
        except ValueError:
            idx = len(existing)
    return base / f"episode_{idx:04d}.mcap"


def _to_list(value) -> list | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    return arr.tolist()


# ---------------------------------------------------------------------------
# Pass-through decorator: record while teleoperating
# ---------------------------------------------------------------------------
class RecordingPublisher(CommandPublisher):
    """Pass-through decorator that records episodes driven by controller buttons.

    - When idle it only forwards commands and writes nothing.
    - start_episode(): open a new episode file and begin recording.
    - save_episode(): close and keep the current episode.
    - delete_last_episode(): remove the previously saved episode; usable while
      recording without interrupting the current one.

    Button bindings live in main.py: VR-Y toggles start/save, VR-X deletes the
    last episode. Each action is announced through the speaker when record_tts
    is enabled.
    """

    def __init__(self, inner: CommandPublisher, cfg: TeleopConfig) -> None:
        self._inner = inner
        self._cfg = cfg
        self._period = 1.0 / max(int(cfg.record_fps), 1)
        self._next_at = time.monotonic()
        self._recorder: MCAPRecorder | None = None
        self._cur_id = -1
        self._last_saved: Path | None = None
        self._images: ImageZMQReceiver | None = None
        if cfg.record_images:
            self._images = ImageZMQReceiver(
                cfg.image_server_host, cfg.image_server_port, cfg.image_client_type
            )
            self._images.start()
        self._tts = TtsSpeaker(cfg.record_tts, cfg.record_tts_speaker_id)
        self._tts.start()
        print(
            f"[record] ready (fps={cfg.record_fps}, images={cfg.record_images}, "
            f"tts={cfg.record_tts}) | VR-Y=start/save, VR-X=delete last"
        )

    # -- Episode control, driven by controller buttons ----------------------
    @property
    def is_recording(self) -> bool:
        return self._recorder is not None

    @staticmethod
    def _episode_num(path) -> int:
        try:
            return int(Path(path).stem.split("_")[-1])
        except (ValueError, IndexError):
            return -1

    def start_episode(self) -> None:
        if self._recorder is not None:
            return
        path = _next_episode_path(self._cfg)
        self._cur_id = self._episode_num(path)
        self._recorder = MCAPRecorder(path)
        self._recorder.write_meta(self._build_info_meta())
        self._next_at = time.monotonic()
        print(f"[record] START -> {path}")
        self._tts.push("start recording")

    def save_episode(self) -> None:
        if self._recorder is None:
            return
        path = self._recorder.path
        self._recorder.close()  # prints frame count and duration
        self._recorder = None
        self._last_saved = path
        self._tts.push(f"saved episode {self._cur_id}")

    def delete_last_episode(self) -> None:
        # Deletes the previously saved episode, never the one being recorded, so
        # it stays usable mid-episode.
        target = self._last_saved
        if target is None or not Path(target).exists():
            target = self._scan_last_saved()
        if target is None:
            print("[record] no episode available to delete")
            self._tts.push("nothing to delete")
            return
        num = self._episode_num(target)
        self._remove_file(target, "deleted last episode")
        self._tts.push(f"deleted episode {num}")
        if self._last_saved is not None and Path(self._last_saved) == Path(target):
            self._last_saved = None

    def _scan_last_saved(self) -> Path | None:
        """Find the newest episode in today's directory, excluding the open one."""
        day = datetime.now().strftime("%Y_%m_%d")
        base = Path(self._cfg.record_dir).expanduser() / self._cfg.record_task_name / day
        if not base.is_dir():
            return None
        cur = self._recorder.path if self._recorder is not None else None
        files = sorted(p for p in base.glob("episode_*.mcap") if cur is None or p != Path(cur))
        return files[-1] if files else None

    @staticmethod
    def _remove_file(path: Path, what: str) -> None:
        try:
            Path(path).unlink()
            print(f"[record] {what}: {path}")
        except OSError as exc:
            print(f"[record] delete failed {path}: {exc}")

    def toggle_episode(self) -> None:
        if self.is_recording:
            self.save_episode()
        else:
            self.start_episode()

    def _build_info_meta(self) -> dict:
        cfg = self._cfg
        return {
            "kind": "info",
            "robot": cfg.robot_name or "unknown",
            "hardware_interface": cfg.hardware_interface,
            "teleop_control_mode": cfg.teleop_control_mode,
            "record_fps": cfg.record_fps,
            "teleop_fps": cfg.fps,
            "record_images": cfg.record_images,
            "image_client_type": cfg.image_client_type,
            "left_arm_joints": cfg.left_arm_joints,
            "right_arm_joints": cfg.right_arm_joints,
            "left_gripper_joints": cfg.left_gripper_joints,
            "right_gripper_joints": cfg.right_gripper_joints,
            "head_joints": cfg.head_joints,
            "base_cmd_joints": cfg.base_cmd_joints,
            "waist_pitch_joint": cfg.g1d_waist_pitch_joint,
            "manual_joystick_joint": cfg.manual_joystick_joint,
            "units": {
                "arm_q": "rad",
                "gripper_q": "urdf_joint(pre-conversion)",
                "lift_position": "lift joint value (NOT meters)",
                "measured_height_m": "meters",
            },
            "lift_meas_scale": cfg.g1d_lift_meas_scale,
            "lift_meas_offset": cfg.g1d_lift_meas_offset,
            "lz_joint_min": cfg.lz_joint_min,
            "lz_joint_max": cfg.lz_joint_max,
        }

    def state_snapshot(self) -> dict:
        return self._inner.state_snapshot()

    def publish(
        self,
        arm_q: np.ndarray,
        arm_tau: np.ndarray,
        base_cmd: list[float],
        gripper_q: np.ndarray | None = None,
        head_q: np.ndarray | None = None,
        lift_position: float | None = None,
    ) -> None:
        # Forward to the real publisher first: recording must never block control.
        self._inner.publish(
            arm_q=arm_q,
            arm_tau=arm_tau,
            base_cmd=base_cmd,
            gripper_q=gripper_q,
            head_q=head_q,
            lift_position=lift_position,
        )
        if self._recorder is None:  # idle: forward only
            return
        now = time.monotonic()
        if now < self._next_at:
            return
        self._next_at = max(self._next_at + self._period, now)
        try:
            self._record_frame(arm_q, arm_tau, base_cmd, gripper_q, head_q, lift_position)
        except Exception as exc:  # a recording error must not take down teleop
            print(f"[record] write frame failed: {exc}")

    def _record_frame(self, arm_q, arm_tau, base_cmd, gripper_q, head_q, lift_position) -> None:
        cfg = self._cfg
        base_list = [float(x) for x in base_cmd]
        body = dict(zip(cfg.base_cmd_joints, base_list[3:]))
        waist_pitch_target = (
            body.get(cfg.g1d_waist_pitch_joint) if cfg.g1d_waist_pitch_joint else None
        )
        actions = {
            "arm_q": _to_list(arm_q),
            "arm_tau": _to_list(arm_tau),
            "base_cmd": base_list,
            "gripper_q": _to_list(gripper_q),
            "head_q": _to_list(head_q),
            "lift_position": float(lift_position) if lift_position is not None else None,
            "waist_pitch_target": float(waist_pitch_target) if waist_pitch_target is not None else None,
        }
        states = self._inner.state_snapshot()
        images: dict = {}
        if self._images is not None:
            for key, raw in self._images.snapshot().items():
                images[key] = {
                    "encoding": "jpeg",
                    "data": base64.b64encode(raw).decode("ascii"),
                }
        self._recorder.write_frame(time.time_ns(), actions, states, images)

    def close(self) -> None:
        try:
            if self._recorder is not None:
                self.save_episode()  # keep whatever was being recorded on exit
        finally:
            self._tts.close()
            if self._images is not None:
                self._images.close()
            self._inner.close()
