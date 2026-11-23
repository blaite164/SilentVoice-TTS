"""
core/voice_cloner.py — OpenVoice v2 ONNX zero-shot voice timbre cloner.

Supports:
1. Isolated DirectML GPU worker (`python-embed` + `onnxruntime-directml`)
   installed on-demand into `data/engines/openvoice/gpu_runtime` so the main
   executable stays lightweight and works on NVIDIA / AMD / Intel GPUs.
2. Automatic in-process CPU (`onnxruntime`) fallback when GPU runtime is not
   present or the user selects CPU mode.
"""

from __future__ import annotations

import atexit
import os
import struct
import subprocess
import threading
import zipfile
from math import gcd
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from core.config import ENGINES_DIR, config
from core.win_api import assign_process_to_kill_on_close_job

OPENVOICE_DIR = ENGINES_DIR / "openvoice"
REF_ENCODER_FILE = "tone_ref_encoder.onnx"
CONVERTER_FILE = "tone_converter.onnx"
GPU_RUNTIME_DIR = OPENVOICE_DIR / "gpu_runtime"
GPU_PYTHON_DIR = GPU_RUNTIME_DIR / "python_embed"
GPU_PYTHON_EXE = GPU_PYTHON_DIR / "python.exe"
GPU_UV_EXE = GPU_RUNTIME_DIR / "uv.exe"
GPU_WORKER_SCRIPT = GPU_RUNTIME_DIR / "gpu_worker.py"
GPU_READY_MARKER = GPU_RUNTIME_DIR / "gpu_ready.marker"

REF_ENCODER_URL = "https://huggingface.co/TigreGotico/voiceclonnx-openvoice-v2/resolve/main/tone_ref_encoder.onnx"
CONVERTER_URL = "https://huggingface.co/TigreGotico/voiceclonnx-openvoice-v2/resolve/main/tone_converter.onnx"
PYTHON_EMBED_URL = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip"
UV_WIN_URL = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"

TARGET_SR = 22050
_HANN_1024 = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(1024, dtype=np.float32) / 1024.0)

_GPU_WORKER_SOURCE = r'''import msvcrt
import os
import struct
import sys

import numpy as np
import onnxruntime as ort

msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
stdin = sys.stdin.buffer
stdout = sys.stdout.buffer


def read_exact(n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stdin.read(n - len(buf))
        if not chunk:
            raise EOFError("stdin closed")
        buf.extend(chunk)
    return bytes(buf)


def send_err(msg: str) -> None:
    data = msg.encode("utf-8", errors="replace")
    stdout.write(b"ER" + struct.pack("<I", len(data)) + data)
    stdout.flush()


def main() -> None:
    enc_path, conv_path = sys.argv[1], sys.argv[2]
    so = ort.SessionOptions()
    so.enable_mem_pattern = False
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    enc = ort.InferenceSession(enc_path, sess_options=so, providers=providers)
    conv = ort.InferenceSession(conv_path, sess_options=so, providers=providers)
    enc_in = enc.get_inputs()[0].name
    conv_in = [i.name for i in conv.get_inputs()]
    is_dml = "DmlExecutionProvider" in conv.get_providers()

    stdout.write(b"RD" + (b"\x01" if is_dml else b"\x00"))
    stdout.flush()

    while True:
        cmd = stdin.read(1)
        if not cmd:
            break
        try:
            if cmd == b"P":
                stdout.write(b"OK" + (b"\x01" if is_dml else b"\x00"))
                stdout.flush()
            elif cmd == b"E":
                (n_frames,) = struct.unpack("<I", read_exact(4))
                raw = read_exact(n_frames * 513 * 4)
                spec_T = np.frombuffer(raw, dtype=np.float32).reshape(1, n_frames, 513)
                emb = enc.run(None, {enc_in: spec_T})[0].astype(np.float32).reshape(256)
                payload = emb.tobytes()
                stdout.write(b"OK" + struct.pack("<I", len(payload)) + payload)
                stdout.flush()
            elif cmd == b"C":
                (n_frames,) = struct.unpack("<I", read_exact(4))
                spec_raw = read_exact(n_frames * 513 * 4)
                tgt_raw = read_exact(256 * 4)
                spec_T = np.frombuffer(spec_raw, dtype=np.float32).reshape(1, n_frames, 513)
                tgt_g = np.frombuffer(tgt_raw, dtype=np.float32).reshape(1, 256, 1)
                src_g = enc.run(None, {enc_in: spec_T})[0].astype(np.float32).reshape(1, 256, 1)
                spec_conv = np.ascontiguousarray(spec_T.transpose(0, 2, 1))
                spec_len = np.array([n_frames], dtype=np.int64)
                out = conv.run(
                    None,
                    {
                        conv_in[0]: spec_conv,
                        conv_in[1]: spec_len,
                        conv_in[2]: src_g,
                        conv_in[3]: tgt_g,
                    },
                )[0].astype(np.float32).reshape(-1)
                payload = out.tobytes()
                stdout.write(b"OK" + struct.pack("<I", len(payload)) + payload)
                stdout.flush()
            elif cmd == b"Q":
                break
        except Exception as exc:
            send_err(str(exc))


if __name__ == "__main__":
    main()
'''


def _hidden_process_kwargs() -> Dict[str, Any]:
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}


def to_22k_mono(audio: np.ndarray, sr: int) -> np.ndarray:
    """Convert any PCM float/int array at `sr` to normalized 22050 Hz mono float32."""
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr > 0 and sr != TARGET_SR and len(y) > 0:
        from scipy.signal import resample_poly

        g = gcd(int(sr), TARGET_SR)
        y = resample_poly(y, TARGET_SR // g, int(sr) // g).astype(np.float32)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)


def compute_linear_spec(y: np.ndarray, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """Compute OpenVoice v2 linear magnitude STFT spectrogram `(T, 513)` float32."""
    pad = (n_fft - hop) // 2  # 384
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    y_pad = np.pad(y, (pad, pad), mode="reflect")
    n_frames = 1 + (len(y_pad) - n_fft) // hop
    strides = (y_pad.strides[0] * hop, y_pad.strides[0])
    frames = np.lib.stride_tricks.as_strided(y_pad, shape=(n_frames, n_fft), strides=strides)
    stft = np.fft.rfft(frames * _HANN_1024, n=n_fft, axis=1)
    return np.ascontiguousarray(np.sqrt((stft.real ** 2) + (stft.imag ** 2) + 1e-6), dtype=np.float32)


def trim_audio_edges(audio: np.ndarray, sr: int, threshold: float = 0.004) -> np.ndarray:
    """Trim dead silence beyond short natural padding so converter skips wasted frames."""
    if len(audio) < max(64, int(0.15 * sr)):
        return audio
    nz = np.where(np.abs(audio) > threshold)[0]
    if len(nz) == 0:
        return audio
    lead = int(0.03 * sr)
    tail = int(0.06 * sr)
    start = max(0, int(nz[0]) - lead)
    end = min(len(audio), int(nz[-1]) + tail)
    return audio[start:end] if end > start else audio


def extract_voiced_speech(audio: np.ndarray, sr: int, target_peak: float = 0.75) -> np.ndarray:
    """Strip silent pauses via simple RMS VAD and normalize peak level for cleaner embedding."""
    y = np.asarray(audio, dtype=np.float32)
    if len(y) == 0:
        return y
    win = max(1, int(sr * 0.03))
    n_blocks = len(y) // win
    if n_blocks >= 4:
        blocks = y[: n_blocks * win].reshape(n_blocks, win)
        rms = np.sqrt(np.mean(blocks * blocks, axis=1))
        peak_rms = float(np.max(rms)) if len(rms) > 0 else 0.0
        if peak_rms > 1e-4:
            keep = rms >= max(peak_rms * 0.14, 0.004)
            # Keep 1 neighbor block around speech so consonant onsets aren't clipped
            dilated = keep.copy()
            dilated[1:] |= keep[:-1]
            dilated[:-1] |= keep[1:]
            voiced = blocks[dilated].reshape(-1)
            if len(voiced) >= int(0.35 * sr):
                y = voiced
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 1e-4:
        y = y * (target_peak / peak)
    return y.astype(np.float32)


def _read_exact(stream: Any, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError("GPU worker stream closed")
        buf.extend(chunk)
    return bytes(buf)


def extract_zip_bytes(zip_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    root = target_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            destination = (target_dir / member.filename).resolve()
            if destination != root and root not in destination.parents:
                raise RuntimeError("Archive contains an invalid or unsafe file path (path traversal detected).")
        zf.extractall(target_dir)


def patch_embed_pth(python_dir: Path) -> None:
    for pth in python_dir.glob("python*._pth"):
        txt = pth.read_text(encoding="utf-8")
        if "Lib\\site-packages" not in txt:
            txt = txt.replace("#import site", "Lib\\site-packages\nimport site")
            pth.write_text(txt, encoding="utf-8")


def extract_uv_exe(zip_path: Path, dest_exe: Path) -> None:
    dest_exe.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith("uv.exe"):
                dest_exe.write_bytes(zf.read(name))
                return
    raise RuntimeError("uv.exe was not found inside downloaded archive")


class _GpuWorkerProcess:
    """Persistent child process running OpenVoice v2 ONNX on DirectML GPU."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._proc: Optional[subprocess.Popen] = None
        self.using_directml: bool = False

    def is_running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def ensure_started(self, enc_path: Path, conv_path: Path) -> bool:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return True
            self.stop()
            if not (GPU_PYTHON_EXE.is_file() and GPU_WORKER_SCRIPT.is_file() and GPU_READY_MARKER.is_file()):
                return False
            proc = subprocess.Popen(
                [str(GPU_PYTHON_EXE), "-u", str(GPU_WORKER_SCRIPT), str(enc_path), str(conv_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                **_hidden_process_kwargs(),
            )
            assign_process_to_kill_on_close_job(proc)
            try:
                assert proc.stdout is not None
                # Give GPU shader/DML compile enough time on cold start via thread timeout
                hdr_box: list[bytes] = []
                err_box: list[Exception] = []

                def _wait_ready() -> None:
                    try:
                        hdr_box.append(_read_exact(proc.stdout, 3))
                    except Exception as exc:
                        err_box.append(exc)

                t = threading.Thread(target=_wait_ready, daemon=True)
                t.start()
                t.join(timeout=45.0)
                if not hdr_box:
                    raise RuntimeError(str(err_box[0]) if err_box else "GPU worker startup timed out")
                hdr = hdr_box[0]
                if hdr[:2] != b"RD":
                    raise RuntimeError(f"Unexpected GPU worker handshake: {hdr!r}")
                self.using_directml = (hdr[2:3] == b"\x01")
                self._proc = proc
                return True
            except Exception as exc:
                print(f"[OpenVoice] GPU worker failed to start, falling back to CPU: {exc}")
                try:
                    proc.kill()
                except Exception:
                    pass
                self._proc = None
                self.using_directml = False
                return False

    def _request(self, payload: bytes) -> bytes:
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None or proc.stdin is None or proc.stdout is None:
                raise RuntimeError("GPU worker is not running")
            proc.stdin.write(payload)
            proc.stdin.flush()
            tag = _read_exact(proc.stdout, 2)
            (size,) = struct.unpack("<I", _read_exact(proc.stdout, 4))
            body = _read_exact(proc.stdout, size)
            if tag == b"ER":
                raise RuntimeError(body.decode("utf-8", errors="replace"))
            if tag != b"OK":
                raise RuntimeError(f"Invalid GPU worker response tag: {tag!r}")
            return body

    def encode_ref(self, enc_path: Path, conv_path: Path, spec_T: np.ndarray) -> np.ndarray:
        with self._lock:
            if not self.ensure_started(enc_path, conv_path):
                raise RuntimeError("GPU runtime unavailable")
            n_frames = int(spec_T.shape[0])
            req = b"E" + struct.pack("<I", n_frames) + np.ascontiguousarray(spec_T, dtype=np.float32).tobytes()
            raw = self._request(req)
            return np.frombuffer(raw, dtype=np.float32).copy().reshape(1, 256)

    def convert(self, enc_path: Path, conv_path: Path, spec_T: np.ndarray, tgt_g: np.ndarray) -> np.ndarray:
        with self._lock:
            if not self.ensure_started(enc_path, conv_path):
                raise RuntimeError("GPU runtime unavailable")
            n_frames = int(spec_T.shape[0])
            req = (
                b"C"
                + struct.pack("<I", n_frames)
                + np.ascontiguousarray(spec_T, dtype=np.float32).tobytes()
                + np.ascontiguousarray(tgt_g, dtype=np.float32).reshape(256).tobytes()
            )
            raw = self._request(req)
            return np.frombuffer(raw, dtype=np.float32).copy()

    def stop(self) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None
            self.using_directml = False
            if proc is not None:
                try:
                    if proc.poll() is None and proc.stdin is not None:
                        proc.stdin.write(b"Q")
                        proc.stdin.flush()
                        proc.wait(timeout=1.5)
                except Exception:
                    pass
                try:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait(timeout=1.5)
                except Exception:
                    pass


class OpenVoiceCloner:
    """Thread-safe OpenVoice v2 voice timbre upgrader with GPU DirectML + CPU fallback."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cpu_enc: Optional[Any] = None
        self._cpu_conv: Optional[Any] = None
        self._gpu_worker = _GpuWorkerProcess()
        self._ref_cache_key: Optional[Tuple[str, int, int, str]] = None
        self._ref_cache_emb: Optional[np.ndarray] = None
        self._warmed_key: Optional[Tuple[str, str]] = None
        atexit.register(self.shutdown)

    @staticmethod
    def models_present(base_dir: Optional[Path] = None) -> bool:
        d = base_dir or OPENVOICE_DIR
        return (d / REF_ENCODER_FILE).is_file() and (d / CONVERTER_FILE).is_file()

    @staticmethod
    def gpu_runtime_present(base_dir: Optional[Path] = None) -> bool:
        d = base_dir or OPENVOICE_DIR
        rt = d / "gpu_runtime"
        return (rt / "python_embed" / "python.exe").is_file() and (rt / "gpu_worker.py").is_file() and (rt / "gpu_ready.marker").is_file()

    @staticmethod
    def is_installed(base_dir: Optional[Path] = None) -> bool:
        return OpenVoiceCloner.models_present(base_dir) and OpenVoiceCloner.gpu_runtime_present(base_dir)

    @staticmethod
    def get_ref_cache_signature(ref_path: str) -> str:
        p_str = str(ref_path or "").strip()
        if not p_str:
            return ""
        p = Path(p_str)
        if not p.is_file():
            return ""
        try:
            st = p.stat()
            return f"{p.resolve()}|{st.st_mtime_ns}|{st.st_size}"
        except OSError:
            return p_str

    def shutdown(self) -> None:
        with self._lock:
            self._gpu_worker.stop()
            self._cpu_enc = None
            self._cpu_conv = None
            self._ref_cache_key = None
            self._ref_cache_emb = None
            self._warmed_key = None

    def clear_ref_cache(self) -> None:
        with self._lock:
            self._ref_cache_key = None
            self._ref_cache_emb = None
            self._warmed_key = None

    def _ensure_cpu_sessions(self) -> Tuple[Any, Any]:
        if self._cpu_enc is not None and self._cpu_conv is not None:
            return self._cpu_enc, self._cpu_conv
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = max(2, (os.cpu_count() or 8) // 2)
        so.inter_op_num_threads = 1
        self._cpu_enc = ort.InferenceSession(str(OPENVOICE_DIR / REF_ENCODER_FILE), sess_options=so, providers=["CPUExecutionProvider"])
        self._cpu_conv = ort.InferenceSession(str(OPENVOICE_DIR / CONVERTER_FILE), sess_options=so, providers=["CPUExecutionProvider"])
        return self._cpu_enc, self._cpu_conv

    def _use_gpu(self) -> bool:
        pref = str(config.get("openvoice_device", "gpu") or "gpu").strip().lower()
        return pref != "cpu" and self.gpu_runtime_present()

    def _get_target_embedding(self, ref_path: Path, use_gpu: bool) -> np.ndarray:
        st = ref_path.stat()
        backend = "gpu" if use_gpu else "cpu"
        key = (str(ref_path.resolve()), int(st.st_mtime_ns), int(st.st_size), backend)
        if self._ref_cache_key == key and self._ref_cache_emb is not None:
            return self._ref_cache_emb

        import soundfile as sf

        audio, sr = sf.read(str(ref_path), dtype="float32", always_2d=False)
        ref_y = extract_voiced_speech(to_22k_mono(audio, int(sr)), TARGET_SR)
        # Up to 20s of active reference speech gives a crisp speaker embedding without excess silence
        if len(ref_y) > TARGET_SR * 20:
            ref_y = ref_y[: TARGET_SR * 20]
        if len(ref_y) < int(0.4 * TARGET_SR):
            raise RuntimeError("Reference voice audio is too short (need at least 0.5 seconds)")

        ref_spec_T = compute_linear_spec(ref_y)
        enc_path = OPENVOICE_DIR / REF_ENCODER_FILE
        conv_path = OPENVOICE_DIR / CONVERTER_FILE

        emb: Optional[np.ndarray] = None
        if use_gpu:
            try:
                emb = self._gpu_worker.encode_ref(enc_path, conv_path, ref_spec_T)
            except Exception as exc:
                print(f"[OpenVoice] GPU ref encode fallback to CPU: {exc}")
                self._gpu_worker.stop()
        if emb is None:
            enc, _ = self._ensure_cpu_sessions()
            emb = enc.run(None, {enc.get_inputs()[0].name: ref_spec_T[None, :, :]})[0].astype(np.float32)

        self._ref_cache_key = key
        self._ref_cache_emb = emb
        return emb

    def upgrade_audio(self, samples: np.ndarray, sr: int, ref_path_str: Optional[str] = None) -> Tuple[np.ndarray, int]:
        """Convert synthesized speech `samples` at `sr` to match the timbre of `ref_path`."""
        if not self.models_present():
            return samples, sr
        ref_str = str(ref_path_str if ref_path_str is not None else config.get("openvoice_ref_path", "")).strip()
        if not ref_str:
            return samples, sr
        ref_path = Path(ref_str)
        if not ref_path.is_file():
            return samples, sr
        if len(samples) == 0:
            return samples, sr

        trimmed = trim_audio_edges(np.asarray(samples, dtype=np.float32), int(sr))
        src_22k = to_22k_mono(trimmed, int(sr))
        if len(src_22k) < 1024:
            return samples, sr
        orig_rms = float(np.sqrt(np.mean(src_22k ** 2)))
        src_peak = float(np.max(np.abs(src_22k)))
        if src_peak > 1e-4:
            src_22k_norm = src_22k * (0.75 / src_peak)
        else:
            src_22k_norm = src_22k
        src_spec_T = compute_linear_spec(src_22k_norm)

        enc_path = OPENVOICE_DIR / REF_ENCODER_FILE
        conv_path = OPENVOICE_DIR / CONVERTER_FILE

        with self._lock:
            use_gpu = self._use_gpu()
            tgt_g = self._get_target_embedding(ref_path, use_gpu=use_gpu)

            converted: Optional[np.ndarray] = None
            if use_gpu:
                try:
                    converted = self._gpu_worker.convert(enc_path, conv_path, src_spec_T, tgt_g)
                except Exception as exc:
                    print(f"[OpenVoice] GPU conversion failed, retrying on CPU: {exc}")
                    self._gpu_worker.stop()
                    # Re-use already computed tgt_g (same 256-dim vector!)
            if converted is None:
                enc, conv = self._ensure_cpu_sessions()
                src_g = enc.run(None, {enc.get_inputs()[0].name: src_spec_T[None, :, :]})[0].astype(np.float32)
                in_names = [i.name for i in conv.get_inputs()]
                converted = conv.run(
                    None,
                    {
                        in_names[0]: np.ascontiguousarray(src_spec_T.T[None, :, :]),
                        in_names[1]: np.array([src_spec_T.shape[0]], dtype=np.int64),
                        in_names[2]: src_g[:, :, None],
                        in_names[3]: tgt_g[:, :, None],
                    },
                )[0].astype(np.float32).reshape(-1)

        if converted is None or len(converted) == 0 or not np.isfinite(converted).all():
            return samples, sr

        # Preserve comfortable RMS level comparable to original source speech
        src_rms = orig_rms
        out_rms = float(np.sqrt(np.mean(converted ** 2)))
        if out_rms > 1e-5 and src_rms > 1e-5:
            gain = min(3.0, max(0.35, src_rms / out_rms))
            converted = converted * gain
        converted = np.clip(converted, -0.98, 0.98).astype(np.float32)
        return converted, TARGET_SR

    def warmup(self) -> None:
        if not bool(config.get("openvoice_enabled", False)):
            return
        if not self.models_present():
            return
        ref_str = str(config.get("openvoice_ref_path", "") or "").strip()
        use_gpu = self._use_gpu()
        backend_tag = "gpu" if use_gpu else "cpu"
        warm_sig = (backend_tag, self.get_ref_cache_signature(ref_str))
        enc_path = OPENVOICE_DIR / REF_ENCODER_FILE
        conv_path = OPENVOICE_DIR / CONVERTER_FILE
        with self._lock:
            if self._warmed_key == warm_sig:
                return
            active_gpu = False
            if use_gpu:
                active_gpu = self._gpu_worker.ensure_started(enc_path, conv_path)
            if active_gpu:
                try:
                    dummy_spec = np.full((64, 513), 1e-4, dtype=np.float32)
                    dummy_tgt = np.zeros((1, 256), dtype=np.float32)
                    self._gpu_worker.convert(enc_path, conv_path, dummy_spec, dummy_tgt)
                except Exception:
                    pass
            else:
                enc, conv = self._ensure_cpu_sessions()
                try:
                    dummy_spec = np.full((64, 513), 1e-4, dtype=np.float32)
                    src_g = enc.run(None, {enc.get_inputs()[0].name: dummy_spec[None, :, :]})[0].astype(np.float32)
                    in_names = [i.name for i in conv.get_inputs()]
                    conv.run(
                        None,
                        {
                            in_names[0]: np.ascontiguousarray(dummy_spec.T[None, :, :]),
                            in_names[1]: np.array([64], dtype=np.int64),
                            in_names[2]: src_g[:, :, None],
                            in_names[3]: np.zeros((1, 256, 1), dtype=np.float32),
                        },
                    )
                except Exception:
                    pass
            if ref_str and Path(ref_str).is_file():
                try:
                    self._get_target_embedding(Path(ref_str), use_gpu=active_gpu)
                except Exception:
                    pass
            self._warmed_key = warm_sig


def install_gpu_runtime(
    progress_cb: Callable[[int, str], None],
    download_fn: Callable[[str, Path, str, int, int, str], None],
    cancel_check: Callable[[], None],
    proxy_env: Optional[Dict[str, str]] = None,
) -> None:
    """Install isolated python-embed + uv + onnxruntime-directml into GPU_RUNTIME_DIR."""
    from core.i18n import tr

    GPU_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    GPU_READY_MARKER.unlink(missing_ok=True)

    py_zip = GPU_RUNTIME_DIR / "python_embed.zip"
    if not GPU_PYTHON_EXE.is_file():
        cancel_check()
        download_fn(PYTHON_EMBED_URL, py_zip, "openvoice", 60, 12, tr("ov.dl.python", "Python 3.12 Embed (~11 МБ)"))
        cancel_check()
        progress_cb(72, tr("ov.dl.extract_python", "Розпакування Python GPU runtime..."))
        extract_zip_bytes(py_zip, GPU_PYTHON_DIR)
        patch_embed_pth(GPU_PYTHON_DIR)
        py_zip.unlink(missing_ok=True)
    else:
        patch_embed_pth(GPU_PYTHON_DIR)

    uv_zip = GPU_RUNTIME_DIR / "uv.zip"
    if not GPU_UV_EXE.is_file():
        cancel_check()
        download_fn(UV_WIN_URL, uv_zip, "openvoice", 73, 10, tr("ov.dl.uv", "Менеджер пакетів uv (~15 МБ)"))
        cancel_check()
        progress_cb(83, tr("ov.dl.extract_uv", "Розпакування uv..."))
        extract_uv_exe(uv_zip, GPU_UV_EXE)
        uv_zip.unlink(missing_ok=True)

    site_dir = GPU_PYTHON_DIR / "Lib" / "site-packages"
    site_dir.mkdir(parents=True, exist_ok=True)
    GPU_WORKER_SCRIPT.write_text(_GPU_WORKER_SOURCE, encoding="utf-8")

    cancel_check()
    progress_cb(85, tr("ov.dl.directml", "Встановлення onnxruntime-directml для GPU (~40 МБ)..."))
    env = os.environ.copy()
    if proxy_env:
        for k, v in proxy_env.items():
            if v:
                env[k] = v

    res = subprocess.run(
        [
            str(GPU_UV_EXE),
            "pip",
            "install",
            "--python",
            str(GPU_PYTHON_EXE),
            "--target",
            str(site_dir),
            "--no-cache",
            "onnxruntime-directml",
            "numpy",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=600,
        **_hidden_process_kwargs(),
    )
    if res.returncode != 0:
        err_tail = (res.stderr or res.stdout or "uv pip install failed").strip()[-800:]
        raise RuntimeError(err_tail)

    cancel_check()
    progress_cb(96, tr("ov.dl.verify_gpu", "Перевірка GPU DirectML..."))
    chk = subprocess.run(
        [str(GPU_PYTHON_EXE), "-c", "import onnxruntime as ort; print(','.join(ort.get_available_providers()))"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        **_hidden_process_kwargs(),
    )
    if chk.returncode != 0:
        raise RuntimeError((chk.stderr or chk.stdout or "DirectML check failed").strip()[-500:])
    GPU_READY_MARKER.write_text((chk.stdout or "ok").strip(), encoding="utf-8")


voice_cloner = OpenVoiceCloner()
