"""VBAN network audio output and receiver health monitoring for SilentVoice."""

from __future__ import annotations

from collections import deque
import ctypes
from dataclasses import dataclass
import ipaddress
import math
import socket
import threading
import time
from typing import Callable, Optional

import numpy as np

RATE = 48_000
SAMPLE_RATE_INDEX = 3  # 48 kHz, audio sub-protocol (VBAN SR index 3)
CHANNELS = 2
FRAMES_PER_PACKET = 256
HEADER_SIZE = 28
BIT_FORMAT_PCM16 = 1

OVERDUE_SKIP_THRESHOLD = 0.050  # 50 ms
FINE_WAIT_SECS = 0.002

# VBAN SERVICE_IDENTIFICATION (PING0)
SERVICE_REPLY_PORT = 6980
PING0_PACKET_SIZE = 704
_SERVICE_PROTOCOL = 0x60
_SERVICE_FUNCTION_PING0 = 0x00
_SERVICE_FUNCTION_REPLY = 0x80
_SERVICE_TYPE_IDENTIFICATION = 0x00
_PING0_BIT_TYPE_OFFSET = 0
_PING0_BIT_FEATURE_OFFSET = 4
_PING0_VERSION_OFFSET = 28
_PING0_APP_NAME_OFFSET = 292
_PING0_APP_NAME_LEN = 64
_PING0_LABEL = b"SilentVoice PING"

StreamKey = tuple[str, int, bytes]


def _target(address: str, port: int, stream_name: str) -> StreamKey:
    host = str(ipaddress.IPv4Address(str(address).strip()))
    port_int = int(port)
    if not 1 <= port_int <= 65535:
        raise ValueError("Port must be between 1 and 65535")
    name = str(stream_name).strip()
    try:
        encoded = name.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("Stream name must use ASCII characters") from exc
    if not 1 <= len(encoded) <= 16 or any(b < 32 or b > 126 for b in encoded):
        raise ValueError("Stream name must contain 1–16 printable ASCII characters")
    return host, port_int, encoded


def prepare_stereo_48k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Convert mono or stereo float audio into contiguous float32 stereo at 48 kHz."""
    sr = int(sample_rate)
    if sr <= 0:
        raise ValueError("Sample rate must be positive")
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 0 or data.ndim > 2:
        raise ValueError("Audio samples must be a 1D or 2D array")
    if data.ndim == 2:
        if data.shape[1] == 0:
            return np.zeros((0, CHANNELS), dtype=np.float32)
        if data.shape[1] == 1:
            data = data[:, 0]
        elif data.shape[1] > 2:
            data = np.mean(data, axis=1, dtype=np.float32)
    if len(data) == 0:
        return np.zeros((0, CHANNELS), dtype=np.float32)
    if not np.all(np.isfinite(data)):
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if sr != RATE:
        from scipy.signal import resample_poly

        divisor = math.gcd(sr, RATE)
        up = RATE // divisor
        down = sr // divisor
        if data.ndim == 1:
            data = resample_poly(data, up, down).astype(np.float32, copy=False)
        else:
            data = resample_poly(data, up, down, axis=0).astype(np.float32, copy=False)
    if data.ndim == 1:
        stereo = np.repeat(data[:, None], CHANNELS, axis=1)
    else:
        stereo = data
    return np.ascontiguousarray(stereo, dtype=np.float32)


def encode_header(stream_name: bytes, count: int, nu_frame: int) -> bytes:
    """Encode a 28-byte VBAN audio header (48 kHz, stereo, signed PCM16 LE)."""
    if not 1 <= count <= FRAMES_PER_PACKET:
        raise ValueError(f"Packet frame count must be 1..{FRAMES_PER_PACKET}")
    header = bytearray(HEADER_SIZE)
    header[:4] = b"VBAN"
    header[4] = SAMPLE_RATE_INDEX
    header[5] = count - 1
    header[6] = CHANNELS - 1
    header[7] = BIT_FORMAT_PCM16
    copied = min(len(stream_name), 16)
    header[8 : 8 + copied] = stream_name[:copied]
    header[24:28] = (nu_frame & 0xFFFFFFFF).to_bytes(4, "little")
    return bytes(header)


@dataclass(frozen=True)
class DecodedHeader:
    samples_per_packet: int
    channels: int
    sample_rate_index: int
    bit_format: int
    stream_name: str
    nu_frame: int


def decode_header(packet: bytes) -> Optional[DecodedHeader]:
    """Decode a 28-byte VBAN header; returns None if magic is invalid."""
    if len(packet) < HEADER_SIZE or packet[:4] != b"VBAN":
        return None
    raw_name = packet[8:24]
    zero_pos = raw_name.find(b"\x00")
    if zero_pos != -1:
        raw_name = raw_name[:zero_pos]
    return DecodedHeader(
        samples_per_packet=packet[5] + 1,
        channels=packet[6] + 1,
        sample_rate_index=packet[4] & 0x1F,
        bit_format=packet[7],
        stream_name=raw_name.decode("ascii", errors="replace"),
        nu_frame=int.from_bytes(packet[24:28], "little"),
    )


class VbanPlayback:
    """Handle to an active or completed VBAN track."""

    def __init__(self, track_id: int, output: "VbanOutput") -> None:
        self.track_id = track_id
        self._output = output
        self.done_event = threading.Event()
        self._lock = threading.Lock()
        self._status = "pending"
        self._error: Optional[str] = None

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def error(self) -> Optional[str]:
        with self._lock:
            return self._error

    def _set_state(self, status: str, error: Optional[str] = None, terminal: bool = False) -> None:
        with self._lock:
            self._status = status
            if error is not None:
                self._error = error
        if terminal:
            self.done_event.set()

    def pause(self) -> None:
        self._output._enqueue(("command", self.track_id, "pause"))

    def resume(self) -> None:
        self._output._enqueue(("command", self.track_id, "resume"))

    def stop(self) -> None:
        self._output._enqueue(("command", self.track_id, "stop"))

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self.done_event.wait(timeout=timeout)


@dataclass
class _TrackRuntime:
    track_id: int
    kind: str
    stream: StreamKey
    audio: np.ndarray
    gain: float
    repeat: bool
    handle: VbanPlayback
    cursor: int = 0
    paused: bool = False


@dataclass
class _StreamRuntime:
    next_send: Optional[float] = None
    nu_frame: int = 0


@dataclass
class _FinishTail:
    deadline: float
    stream: StreamKey
    kind: str
    handle: VbanPlayback


class _WindowsTimerPeriod:
    """Scope Windows 1 ms multimedia timer resolution to active streaming."""

    def __init__(self) -> None:
        self._active = False

    def acquire(self) -> None:
        if self._active:
            return
        try:
            self._active = ctypes.windll.winmm.timeBeginPeriod(1) == 0
        except (AttributeError, OSError):
            self._active = False

    def release(self) -> None:
        if not self._active:
            return
        try:
            ctypes.windll.winmm.timeEndPeriod(1)
        except (AttributeError, OSError):
            pass
        self._active = False


class VbanOutput:
    """Shared VBAN UDP worker supporting persistent nuFrame, mixing, and prompt stop."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._commands: deque[tuple] = deque()
        self._tracks: dict[int, _TrackRuntime] = {}
        self._streams: dict[StreamKey, _StreamRuntime] = {}
        self._finishes: dict[int, _FinishTail] = {}
        self._next_track_id = 1
        self._shutdown = False
        self._worker: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None
        self._timer_period = _WindowsTimerPeriod()
        self._last_error: Optional[str] = None

    def _ensure_worker(self) -> None:
        with self._cond:
            if self._shutdown:
                self._shutdown = False
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._run,
                    name="vban-output",
                    daemon=True,
                )
                self._worker.start()

    def _enqueue(self, cmd: tuple) -> None:
        with self._cond:
            if self._shutdown:
                return
            self._commands.append(cmd)
            self._cond.notify()

    @property
    def last_error(self) -> Optional[str]:
        with self._cond:
            return self._last_error

    def start(
        self,
        samples: np.ndarray,
        sample_rate: int,
        address: str,
        port: int,
        stream_name: str,
        *,
        kind: str = "speech",
        gain: float = 1.0,
        repeat: bool = False,
    ) -> VbanPlayback:
        """Queue an audio track on the VBAN worker without blocking on network I/O."""
        key = _target(address, port, stream_name)
        sr = int(sample_rate)
        if sr <= 0:
            raise ValueError("Sample rate must be positive")
        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim == 0 or arr.ndim > 2:
            raise ValueError("Audio samples must be a 1D or 2D array")
        gain_f = float(gain)
        if not math.isfinite(gain_f):
            raise ValueError("Gain must be finite")

        self._ensure_worker()

        reply_event: Optional[threading.Event] = None
        reply_error: list[Optional[str]] = [None]
        if kind == "test":
            reply_event = threading.Event()

        with self._cond:
            track_id = self._next_track_id
            self._next_track_id += 1
            handle = VbanPlayback(track_id, self)
            self._commands.append(
                (
                    "start",
                    track_id,
                    kind,
                    key,
                    arr,
                    sr,
                    gain_f,
                    bool(repeat),
                    handle,
                    reply_event,
                    reply_error,
                )
            )
            self._cond.notify()

        if reply_event is not None:
            reply_event.wait(timeout=2.0)
            if reply_error[0] is not None:
                raise RuntimeError(reply_error[0])
        return handle

    def shutdown(self, timeout: float = 2.0) -> None:
        """Stop all tracks and shut down the worker thread cleanly."""
        with self._cond:
            self._shutdown = True
            self._commands.append(("shutdown",))
            self._cond.notify_all()
            worker = self._worker
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=timeout)
        with self._cond:
            if self._worker is worker:
                self._worker = None

    def _close_socket(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _ensure_socket(self) -> Optional[socket.socket]:
        if self._sock is None:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setblocking(False)
                self._sock = sock
            except OSError as exc:
                self._last_error = f"VBAN socket error: {exc}"
                return None
        return self._sock

    def _remove_track(self, track_id: int, status: str, error: Optional[str] = None) -> None:
        track = self._tracks.pop(track_id, None)
        if track is not None:
            track.handle._set_state(status, error=error, terminal=True)
            return
        tail = self._finishes.pop(track_id, None)
        if tail is not None:
            tail.handle._set_state(status, error=error, terminal=True)

    def _has_production_on(self, key: StreamKey) -> bool:
        if any(t.kind != "test" and t.stream == key for t in self._tracks.values()):
            return True
        return any(f.kind != "test" and f.stream == key for f in self._finishes.values())

    def _cancel_tests_on(self, key: StreamKey) -> None:
        test_ids = [tid for tid, t in self._tracks.items() if t.kind == "test" and t.stream == key]
        for tid in test_ids:
            self._remove_track(tid, "stopped")

    def _cancel_kind(self, kind: str) -> None:
        ids = [tid for tid, t in self._tracks.items() if t.kind == kind]
        for tid in ids:
            self._remove_track(tid, "stopped")
        tail_ids = [tid for tid, f in self._finishes.items() if f.kind == kind]
        for tid in tail_ids:
            self._remove_track(tid, "stopped")

    def _handle_start(self, cmd: tuple, now: float) -> None:
        (
            _,
            track_id,
            kind,
            key,
            arr,
            sr,
            gain,
            repeat,
            handle,
            reply_event,
            reply_error,
        ) = cmd

        if kind == "test":
            if self._has_production_on(key):
                msg = "VBAN production output is active on this stream"
                handle._set_state("failed", error=msg, terminal=True)
                reply_error[0] = msg
                if reply_event is not None:
                    reply_event.set()
                return
            self._cancel_tests_on(key)
        else:
            self._cancel_tests_on(key)
            self._cancel_kind(kind)

        try:
            stereo = prepare_stereo_48k(arr, sr)
        except Exception as exc:
            msg = str(exc)
            handle._set_state("failed", error=msg, terminal=True)
            if reply_error is not None:
                reply_error[0] = msg
            if reply_event is not None:
                reply_event.set()
            return

        if len(stereo) == 0:
            handle._set_state("finished", terminal=True)
            if reply_event is not None:
                reply_event.set()
            return

        stream = self._streams.get(key)
        if stream is None:
            stream = _StreamRuntime()
            self._streams[key] = stream

        # Drop overdue debt accumulated by existing tracks before inserting fresh audio.
        if stream.next_send is not None and now - stream.next_send > OVERDUE_SKIP_THRESHOLD:
            self._skip_overdue_on_stream(key, int((now - stream.next_send) * RATE))
            stream.next_send = now
        elif stream.next_send is None:
            stream.next_send = now

        self._tracks[track_id] = _TrackRuntime(
            track_id=track_id,
            kind=kind,
            stream=key,
            audio=stereo,
            gain=gain,
            repeat=repeat,
            handle=handle,
        )
        if reply_event is not None:
            reply_event.set()

    def _skip_overdue_on_stream(self, key: StreamKey, skip_frames: int) -> None:
        if skip_frames <= 0:
            return
        finished_ids: list[int] = []
        for tid, track in self._tracks.items():
            if track.stream != key or track.paused:
                continue
            total = len(track.audio)
            if track.repeat and total > 0:
                track.cursor = (track.cursor + skip_frames) % total
            else:
                track.cursor = min(total, track.cursor + skip_frames)
                if track.cursor >= total:
                    finished_ids.append(tid)
        for tid in finished_ids:
            self._remove_track(tid, "finished")

    def _handle_command(self, track_id: int, action: str, now: float) -> None:
        if action == "stop":
            self._remove_track(track_id, "stopped")
            return
        track = self._tracks.get(track_id)
        if track is None:
            return
        if action == "pause" and not track.paused:
            track.paused = True
            track.handle._set_state("paused")
        elif action == "resume" and track.paused:
            track.paused = False
            track.handle._set_state("playing")
            stream = self._streams.get(track.stream)
            if stream is not None and stream.next_send is None:
                stream.next_send = now

    def _check_finishes(self, now: float) -> None:
        due = [tid for tid, tail in self._finishes.items() if tail.deadline <= now]
        for tid in due:
            tail = self._finishes.pop(tid, None)
            if tail is not None:
                tail.handle._set_state("finished", terminal=True)

    def _pump_due_streams(self, now: float) -> None:
        due_keys = [
            key
            for key, stream in self._streams.items()
            if stream.next_send is not None and stream.next_send <= now
        ]
        for key in due_keys:
            stream = self._streams[key]
            assert stream.next_send is not None
            if now - stream.next_send > OVERDUE_SKIP_THRESHOLD:
                skip_frames = int((now - stream.next_send) * RATE)
                self._skip_overdue_on_stream(key, skip_frames)
                schedule_base = now
            else:
                schedule_base = stream.next_send

            active = [
                t
                for t in self._tracks.values()
                if t.stream == key and not t.paused and (t.repeat or t.cursor < len(t.audio))
            ]
            if not active:
                stream.next_send = None
                continue

            if any(t.repeat for t in active):
                packet_frames = FRAMES_PER_PACKET
            else:
                max_remaining = max(len(t.audio) - t.cursor for t in active)
                packet_frames = min(FRAMES_PER_PACKET, max_remaining)
                if packet_frames <= 0:
                    stream.next_send = None
                    continue

            mix = np.zeros((packet_frames, CHANNELS), dtype=np.float32)
            takes: list[tuple[_TrackRuntime, int]] = []
            for track in active:
                total = len(track.audio)
                if track.repeat:
                    written = 0
                    while written < packet_frames:
                        avail = total - track.cursor
                        take = min(packet_frames - written, avail)
                        chunk = track.audio[track.cursor : track.cursor + take]
                        if track.gain == 1.0:
                            mix[written : written + take] += chunk
                        else:
                            mix[written : written + take] += chunk * track.gain
                        track.cursor = (track.cursor + take) % total
                        written += take
                    takes.append((track, packet_frames))
                else:
                    take = min(packet_frames, total - track.cursor)
                    chunk = track.audio[track.cursor : track.cursor + take]
                    if track.gain == 1.0:
                        mix[:take] += chunk
                    else:
                        mix[:take] += chunk * track.gain
                    track.cursor += take
                    takes.append((track, take))

            np.clip(mix, -1.0, 1.0, out=mix)
            pcm = np.rint(mix * 32767.0).astype("<i2", copy=False).tobytes()
            nu_frame = (stream.nu_frame + 1) & 0xFFFFFFFF
            header = encode_header(key[2], packet_frames, nu_frame)
            packet = header + pcm

            sock = self._ensure_socket()
            send_ok = True
            send_err: Optional[str] = None
            if sock is None:
                send_ok = False
                send_err = self._last_error or "VBAN UDP socket is unavailable"
            else:
                try:
                    sock.sendto(packet, (key[0], key[1]))
                except BlockingIOError:
                    # OS send buffer temporarily saturated; skip slot rather than stalling.
                    pass
                except OSError as exc:
                    send_ok = False
                    send_err = str(exc)

            if not send_ok:
                assert send_err is not None
                with self._cond:
                    self._last_error = send_err
                for track, _ in takes:
                    self._remove_track(track.track_id, "failed", error=send_err)
                stream.next_send = None
                continue

            with self._cond:
                self._last_error = None
            stream.nu_frame = nu_frame
            stream.next_send = schedule_base + (packet_frames / RATE)

            send_now = time.perf_counter()
            still_active = False
            for track, take in takes:
                if track.handle.status == "pending":
                    track.handle._set_state("playing")
                if not track.repeat and track.cursor >= len(track.audio):
                    self._tracks.pop(track.track_id, None)
                    self._finishes[track.track_id] = _FinishTail(
                        deadline=send_now + (take / RATE),
                        stream=track.stream,
                        kind=track.kind,
                        handle=track.handle,
                    )
                else:
                    still_active = True
            if not still_active:
                stream.next_send = None

    def _next_deadline(self) -> Optional[float]:
        deadlines = [s.next_send for s in self._streams.values() if s.next_send is not None]
        deadlines.extend(f.deadline for f in self._finishes.values())
        return min(deadlines) if deadlines else None

    def _shutdown_all(self) -> None:
        for tid in list(self._tracks.keys()):
            self._remove_track(tid, "stopped")
        for tid in list(self._finishes.keys()):
            self._remove_track(tid, "stopped")
        self._timer_period.release()
        self._close_socket()

    def _run(self) -> None:
        try:
            while True:
                pending: list[tuple] = []
                with self._cond:
                    while self._commands:
                        pending.append(self._commands.popleft())

                halt = False
                for cmd in pending:
                    tag = cmd[0]
                    now = time.perf_counter()
                    if tag == "shutdown":
                        halt = True
                        break
                    if tag == "start":
                        self._handle_start(cmd, now)
                    elif tag == "command":
                        self._handle_command(cmd[1], cmd[2], now)

                if halt:
                    break

                now = time.perf_counter()
                self._check_finishes(now)
                self._pump_due_streams(now)

                has_active_work = any(not t.paused for t in self._tracks.values())
                if has_active_work:
                    self._timer_period.acquire()
                else:
                    self._timer_period.release()

                deadline = self._next_deadline()
                with self._cond:
                    if self._commands:
                        continue
                    if self._shutdown:
                        break
                    if deadline is None:
                        self._cond.wait()
                        continue
                    remaining = deadline - time.perf_counter()
                    if remaining > FINE_WAIT_SECS:
                        self._cond.wait(timeout=remaining - FINE_WAIT_SECS)
                        continue

                if deadline is not None:
                    while True:
                        now = time.perf_counter()
                        if now >= deadline:
                            break
                        with self._cond:
                            if self._commands or self._shutdown:
                                break
                        time.sleep(min(0.0005, max(0.0, deadline - now)))
        finally:
            with self._cond:
                while self._commands:
                    rem = self._commands.popleft()
                    if rem[0] == "start":
                        rem_handle = rem[8]
                        rem_reply = rem[9]
                        rem_err = rem[10]
                        rem_handle._set_state("stopped", terminal=True)
                        if rem_err is not None:
                            rem_err[0] = "VBAN output is shutting down"
                        if rem_reply is not None:
                            rem_reply.set()
            self._shutdown_all()


_DEFAULT_OUTPUT_LOCK = threading.Lock()
_DEFAULT_OUTPUT: Optional[VbanOutput] = None


def get_vban_output() -> VbanOutput:
    global _DEFAULT_OUTPUT
    with _DEFAULT_OUTPUT_LOCK:
        if _DEFAULT_OUTPUT is None:
            _DEFAULT_OUTPUT = VbanOutput()
        return _DEFAULT_OUTPUT


def start_samples(
    samples: np.ndarray,
    sample_rate: int,
    address: str,
    port: int,
    stream_name: str,
    *,
    kind: str = "speech",
    gain: float = 1.0,
    repeat: bool = False,
) -> VbanPlayback:
    """Start sending audio on the shared VBAN worker and return a playback handle."""
    return get_vban_output().start(
        samples,
        sample_rate,
        address,
        port,
        stream_name,
        kind=kind,
        gain=gain,
        repeat=repeat,
    )


def send_samples(
    samples: np.ndarray,
    sample_rate: int,
    address: str,
    port: int,
    stream_name: str,
    stop_event: threading.Event | None = None,
    repeat: bool = False,
    gain: float = 1.0,
    kind: str = "speech",
) -> None:
    """Send float audio as paced VBAN PCM16 stereo 48 kHz packets."""
    handle = start_samples(
        samples,
        sample_rate,
        address,
        port,
        stream_name,
        kind=kind,
        gain=gain,
        repeat=repeat,
    )
    if stop_event is None:
        handle.wait()
    else:
        while not handle.wait(timeout=0.02):
            if stop_event.is_set():
                handle.stop()
                handle.wait(timeout=0.5)
                break
    if handle.status == "failed" and handle.error:
        raise RuntimeError(handle.error)


def send_test(
    address: str,
    port: int,
    stream_name: str,
    stop_event: threading.Event,
    gain: float = 1.0,
) -> None:
    """Repeat a short 660 Hz test beep over VBAN until stop_event is set."""
    duration = 1.5
    t = np.arange(int(RATE * duration), dtype=np.float32) / RATE
    active = (t < 0.45).astype(np.float32)
    # Apply short 10 ms edge ramps so the beep has no clicks.
    ramp = int(RATE * 0.010)
    on_frames = int(RATE * 0.45)
    if ramp > 0 and on_frames > 2 * ramp:
        env = np.zeros_like(t)
        env[:on_frames] = 1.0
        fade = 0.5 * (1.0 - np.cos(np.pi * np.arange(ramp, dtype=np.float32) / ramp))
        env[:ramp] = fade
        env[on_frames - ramp : on_frames] = fade[::-1]
    else:
        env = active
    samples = (0.18 * np.sin(2.0 * np.pi * 660.0 * t) * env).astype(np.float32)
    send_samples(
        samples,
        RATE,
        address,
        port,
        stream_name,
        stop_event=stop_event,
        repeat=True,
        gain=gain,
        kind="test",
    )


def shutdown_vban() -> None:
    """Shut down the shared VBAN output runtime."""
    global _DEFAULT_OUTPUT
    with _DEFAULT_OUTPUT_LOCK:
        output = _DEFAULT_OUTPUT
        _DEFAULT_OUTPUT = None
    if output is not None:
        output.shutdown()


# ---------------------------------------------------------------------------
# VBAN SERVICE_IDENTIFICATION (PING0) receiver health check
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Ping0Reply:
    request_id_echo: int
    device_type: int
    features: int
    version: tuple[int, int, int, int]
    application_name: str

    @property
    def version_str(self) -> str:
        return ".".join(str(part) for part in self.version)


def encode_ping0_request(request_id: int = 1) -> bytes:
    """Build a 704-byte VBAN SERVICE_IDENTIFICATION (PING0) request."""
    pkt = bytearray(PING0_PACKET_SIZE)
    pkt[0:4] = b"VBAN"
    pkt[4] = _SERVICE_PROTOCOL
    pkt[5] = _SERVICE_FUNCTION_PING0
    pkt[6] = _SERVICE_TYPE_IDENTIFICATION
    pkt[7] = 0
    pkt[8 : 8 + len(_PING0_LABEL)] = _PING0_LABEL[:16]
    pkt[24:28] = (request_id & 0xFFFFFFFF).to_bytes(4, "little")
    return bytes(pkt)


def decode_ping0_reply(
    data: bytes,
    expected_ip: str,
    expected_port: int,
    source_addr: tuple[str, int],
) -> Optional[Ping0Reply]:
    """Validate and parse a 704-byte VBAN PING0 reply from the expected endpoint."""
    if len(data) != PING0_PACKET_SIZE:
        return None
    if data[0:4] != b"VBAN":
        return None
    if data[4] != _SERVICE_PROTOCOL:
        return None
    if (data[5] & _SERVICE_FUNCTION_REPLY) != _SERVICE_FUNCTION_REPLY:
        return None
    if (data[5] & ~_SERVICE_FUNCTION_REPLY) != _SERVICE_FUNCTION_PING0:
        return None
    if data[6] != _SERVICE_TYPE_IDENTIFICATION or data[7] != 0:
        return None
    if source_addr[0] != expected_ip or int(source_addr[1]) != int(expected_port):
        return None

    payload = data[HEADER_SIZE:]
    raw_app = payload[_PING0_APP_NAME_OFFSET : _PING0_APP_NAME_OFFSET + _PING0_APP_NAME_LEN]
    zero_pos = raw_app.find(b"\x00")
    if zero_pos != -1:
        raw_app = raw_app[:zero_pos]
    app_name = raw_app.decode("ascii", errors="replace").strip()

    ver = payload[_PING0_VERSION_OFFSET : _PING0_VERSION_OFFSET + 4]
    return Ping0Reply(
        request_id_echo=int.from_bytes(data[24:28], "little"),
        device_type=int.from_bytes(payload[_PING0_BIT_TYPE_OFFSET : _PING0_BIT_TYPE_OFFSET + 4], "little"),
        features=int.from_bytes(payload[_PING0_BIT_FEATURE_OFFSET : _PING0_BIT_FEATURE_OFFSET + 4], "little"),
        version=(ver[0], ver[1], ver[2], ver[3]),
        application_name=app_name,
    )


def ping_receiver(
    address: str,
    port: int = SERVICE_REPLY_PORT,
    timeout: float = 1.0,
    reply_port: int = SERVICE_REPLY_PORT,
) -> tuple[str, Optional[Ping0Reply], Optional[str]]:
    """
    Send one PING0 packet and wait for a reply on `reply_port`.

    Returns `(status, reply, error_detail)` where `status` is one of:
    - `"alive"`: valid PING0 reply received
    - `"no_reply"`: timeout or send error (receiver unreachable)
    - `"local_socket"`: failed to bind `reply_port` locally
    """
    host = str(ipaddress.IPv4Address(str(address).strip()))
    port_int = int(port)
    if not 1 <= port_int <= 65535:
        raise ValueError("Port must be between 1 and 65535")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        try:
            sock.bind(("0.0.0.0", int(reply_port)))
        except OSError as exc:
            return "local_socket", None, str(exc)

        request = encode_ping0_request(1)
        try:
            sock.sendto(request, (host, port_int))
        except OSError as exc:
            return "no_reply", None, str(exc)

        deadline = time.monotonic() + max(0.05, float(timeout))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "no_reply", None, None
            sock.settimeout(remaining)
            try:
                data, source = sock.recvfrom(2048)
            except socket.timeout:
                return "no_reply", None, None
            except OSError as exc:
                return "no_reply", None, str(exc)
            reply = decode_ping0_reply(data, host, port_int, (source[0], source[1]))
            if reply is not None:
                return "alive", reply, None
    finally:
        sock.close()


HealthCallback = Callable[[str, str, str], None]


class VbanHealthMonitor:
    """Periodic background PING0 monitor for the configured VBAN receiver."""

    def __init__(
        self,
        on_change: Optional[HealthCallback] = None,
        reply_port: int = SERVICE_REPLY_PORT,
        probe_timeout: float = 1.0,
    ) -> None:
        self._cond = threading.Condition()
        self._on_change = on_change
        self._reply_port = int(reply_port)
        self._probe_timeout = float(probe_timeout)
        self._target: Optional[tuple[str, int, float]] = None
        self._generation = 0
        self._state = "unknown"
        self._reason = ""
        self._detail = ""
        self._consecutive_failures = 0
        self._sock: Optional[socket.socket] = None
        self._shutdown = False
        self._force_probe = False
        self._manual_probes: deque[tuple[str, int, float, threading.Event, list]] = deque()
        self._thread: Optional[threading.Thread] = None

    @property
    def state(self) -> tuple[str, str, str]:
        with self._cond:
            return self._state, self._reason, self._detail

    def set_callback(self, callback: Optional[HealthCallback]) -> None:
        with self._cond:
            self._on_change = callback

    def set_target(
        self,
        address: Optional[str],
        port: int = SERVICE_REPLY_PORT,
        interval_secs: float = 15.0,
    ) -> None:
        new_target: Optional[tuple[str, int, float]] = None
        if address and str(address).strip():
            host = str(ipaddress.IPv4Address(str(address).strip()))
            port_int = int(port)
            if not 1 <= port_int <= 65535:
                raise ValueError("Port must be between 1 and 65535")
            interval = max(0.1, float(interval_secs))
            new_target = (host, port_int, interval)

        cb: Optional[HealthCallback] = None
        emit_args: Optional[tuple[str, str, str]] = None
        with self._cond:
            if new_target == self._target:
                return
            same_endpoint = (
                new_target is not None
                and self._target is not None
                and new_target[0] == self._target[0]
                and new_target[1] == self._target[1]
            )
            self._target = new_target
            self._generation += 1
            if not same_endpoint:
                self._consecutive_failures = 0
                if (self._state, self._reason, self._detail) != ("unknown", "", ""):
                    self._state = "unknown"
                    self._reason = ""
                    self._detail = ""
                    cb = self._on_change
                    emit_args = ("unknown", "", "")
            if new_target is not None and (self._thread is None or not self._thread.is_alive()) and not self._shutdown:
                self._thread = threading.Thread(
                    target=self._run,
                    name="vban-health",
                    daemon=True,
                )
                self._thread.start()
            self._cond.notify_all()

        if cb is not None and emit_args is not None:
            cb(*emit_args)

    def probe_now(self) -> None:
        with self._cond:
            self._force_probe = True
            self._cond.notify_all()

    def probe_once_sync(
        self,
        address: str,
        port: int = SERVICE_REPLY_PORT,
        timeout: float = 1.0,
    ) -> tuple[str, Optional[Ping0Reply], Optional[str]]:
        """Run one PING0 probe on the monitor thread (avoids reply-port conflicts)."""
        host = str(ipaddress.IPv4Address(str(address).strip()))
        port_int = int(port)
        if not 1 <= port_int <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        done = threading.Event()
        result: list = [("no_reply", None, None)]
        with self._cond:
            if self._shutdown:
                self._shutdown = False
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run,
                    name="vban-health",
                    daemon=True,
                )
                self._thread.start()
            self._manual_probes.append((host, port_int, max(0.05, float(timeout)), done, result))
            self._cond.notify_all()
        done.wait(timeout=max(0.2, float(timeout) + 1.0))
        return result[0]

    def shutdown(self, timeout: float = 2.0) -> None:
        with self._cond:
            self._shutdown = True
            self._target = None
            self._generation += 1
            self._cond.notify_all()
            thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        with self._cond:
            if self._thread is thread:
                self._thread = None

    def _close_socket(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _probe_once(
        self,
        host: str,
        port: int,
        request_id: int = 1,
        timeout: Optional[float] = None,
    ) -> tuple[str, Optional[Ping0Reply], Optional[str]]:
        if self._sock is None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.bind(("0.0.0.0", self._reply_port))
            except OSError as exc:
                sock.close()
                return "local_socket", None, str(exc)
            self._sock = sock

        sock = self._sock
        # Drain stale datagrams from earlier probe windows.
        sock.setblocking(False)
        while True:
            try:
                sock.recvfrom(2048)
            except OSError:
                break
        sock.setblocking(True)

        req = encode_ping0_request(request_id)
        try:
            sock.sendto(req, (host, port))
        except OSError as exc:
            return "no_reply", None, str(exc)

        wait_secs = self._probe_timeout if timeout is None else float(timeout)
        deadline = time.monotonic() + wait_secs
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "no_reply", None, None
            sock.settimeout(remaining)
            try:
                data, source = sock.recvfrom(2048)
            except socket.timeout:
                return "no_reply", None, None
            except OSError as exc:
                return "no_reply", None, str(exc)
            reply = decode_ping0_reply(data, host, port, (source[0], source[1]))
            if reply is not None:
                return "alive", reply, None

    def _run(self) -> None:
        last_endpoint: Optional[tuple[str, int]] = None
        next_probe_at = 0.0
        request_seq = 0
        try:
            while True:
                manual = None
                with self._cond:
                    if self._shutdown:
                        break
                    if self._manual_probes:
                        manual = self._manual_probes.popleft()
                        target = self._target
                    else:
                        target = self._target
                        gen = self._generation
                        if target is None:
                            self._close_socket()
                            last_endpoint = None
                            self._cond.wait()
                            continue
                        endpoint = (target[0], target[1])
                        interval = target[2]
                        if endpoint != last_endpoint:
                            self._close_socket()
                            last_endpoint = endpoint
                            next_probe_at = 0.0
                        now = time.monotonic()
                        if not self._force_probe and now < next_probe_at:
                            self._cond.wait(timeout=next_probe_at - now)
                            continue
                        self._force_probe = False

                if manual is not None:
                    m_host, m_port, m_timeout, m_done, m_result = manual
                    request_seq = (request_seq + 1) & 0xFFFFFFFF
                    res = self._probe_once(m_host, m_port, request_seq, timeout=m_timeout)
                    m_result[0] = res
                    m_done.set()
                    cb: Optional[HealthCallback] = None
                    emit_args: Optional[tuple[str, str, str]] = None
                    with self._cond:
                        if self._target is None:
                            self._close_socket()
                        elif self._target[0] == m_host and self._target[1] == m_port and res[0] == "alive":
                            self._consecutive_failures = 0
                            reply_obj = res[1]
                            detail = ""
                            if reply_obj is not None and reply_obj.application_name:
                                detail = f"{reply_obj.application_name} {reply_obj.version_str}".strip()
                            new_tuple = ("alive", "", detail)
                            if new_tuple != (self._state, self._reason, self._detail):
                                self._state, self._reason, self._detail = new_tuple
                                cb = self._on_change
                                emit_args = new_tuple
                    if cb is not None and emit_args is not None:
                        cb(*emit_args)
                    continue

                request_seq = (request_seq + 1) & 0xFFFFFFFF
                status, reply, err_detail = self._probe_once(endpoint[0], endpoint[1], request_seq)

                cb: Optional[HealthCallback] = None
                emit_args: Optional[tuple[str, str, str]] = None
                with self._cond:
                    if self._shutdown:
                        break
                    if gen != self._generation:
                        continue
                    threshold = 3 if interval < 10.0 else 2
                    if status == "alive":
                        self._consecutive_failures = 0
                        detail = ""
                        if reply is not None and reply.application_name:
                            detail = f"{reply.application_name} {reply.version_str}".strip()
                        new_tuple = ("alive", "", detail)
                    elif status == "local_socket":
                        self._consecutive_failures = 0
                        new_tuple = ("failed", "local_socket", err_detail or "")
                    else:
                        self._consecutive_failures += 1
                        if self._consecutive_failures >= threshold:
                            new_tuple = ("failed", "no_reply", err_detail or "")
                        else:
                            new_tuple = None

                    if new_tuple is not None and new_tuple != (self._state, self._reason, self._detail):
                        self._state, self._reason, self._detail = new_tuple
                        cb = self._on_change
                        emit_args = new_tuple
                    next_probe_at = time.monotonic() + interval

                if cb is not None and emit_args is not None:
                    cb(*emit_args)
        finally:
            self._close_socket()
