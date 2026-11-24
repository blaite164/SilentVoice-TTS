"""
core/llama_tts.py — Local GGUF TTS runtime for Qwen3-TTS 1.7B Base.

Uses official `ggml-org/llama.cpp` Windows binaries (`llama-server.exe`
with NVIDIA CUDA or Vulkan GPU support) downloaded on-demand into
`data/engines/llama_tts` so the main executable stays compact.
"""

from __future__ import annotations

import atexit
from collections import deque
import hashlib
import math
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Set, Tuple

import numpy as np
import requests
from requests.adapters import HTTPAdapter
import soundfile as sf
from urllib3.connection import HTTPConnection

from core.config import DATA_DIR, ENGINES_DIR, SAMPLES_DIR, config
from core.i18n import tr
from core.win_api import assign_process_to_kill_on_close_job, has_hardware_gpu, has_nvidia_gpu

LLAMA_TTS_DIR = ENGINES_DIR / "llama_tts"
LLAMA_TTS_EXE = LLAMA_TTS_DIR / "llama-tts.exe"
LLAMA_TTS_SERVER_EXE = LLAMA_TTS_DIR / "llama-server.exe"
LLAMA_TTS_READY_MARKER = LLAMA_TTS_DIR / "ready.marker"

LLAMA_TTS_ENGINE_IDS: Tuple[str, ...] = ("qwen3_tts", "zonos3_mini", "kokoro_82m", "pocket_tts")
LLAMA_STREAMING_ENGINE_IDS: Tuple[str, ...] = ("qwen3_tts", "zonos3_mini")

QWEN3_TTS_DIR = ENGINES_DIR / "qwen3_tts"

# ZONOS3-Mini (ggml-org/ZONOS3-Mini-GGUF, 44.1 kHz, 43 languages, built-in voices + zero-shot cloning)
ZONOS3_DIR = ENGINES_DIR / "zonos3_mini"
ZONOS3_MODEL_FILE = "ZONOS3-Mini-Q8_0.gguf"
ZONOS3_MMPROJ_FILE = "mmproj-ZONOS3-Mini-F32.gguf"
ZONOS3_MODEL_URL = "https://huggingface.co/ggml-org/ZONOS3-Mini-GGUF/resolve/main/ZONOS3-Mini-Q8_0.gguf"
ZONOS3_MMPROJ_URL = "https://huggingface.co/ggml-org/ZONOS3-Mini-GGUF/resolve/main/mmproj-ZONOS3-Mini-F32.gguf"
ZONOS3_NORM_LANGS = frozenset(("en", "fr", "de", "es", "it", "pt", "ja", "zh", "cmn"))

# Kokoro-82M (ggml-org/Kokoro-82M-GGUF, 24 kHz, English US/UK built-in voices, CPU/GPU)
KOKORO_DIR = ENGINES_DIR / "kokoro_82m"
KOKORO_MODEL_FILE = "Kokoro-82M-F32.gguf"
KOKORO_MMPROJ_FILE = "mmproj-Kokoro-82M-F16.gguf"
KOKORO_MODEL_URL = "https://huggingface.co/ggml-org/Kokoro-82M-GGUF/resolve/main/Kokoro-82M-F32.gguf"
KOKORO_MMPROJ_URL = "https://huggingface.co/ggml-org/Kokoro-82M-GGUF/resolve/main/mmproj-Kokoro-82M-F16.gguf"

# Pocket TTS (EryriLabs/pocket-tts-GGUF, 24 kHz, per-language EN/DE/ES/FR, CPU/GPU + zero-shot cloning)
POCKET_TTS_DIR = ENGINES_DIR / "pocket_tts"
POCKET_TTS_HF_BASE = "https://huggingface.co/EryriLabs/pocket-tts-GGUF/resolve/main"
POCKET_TTS_LANG_FILES: Dict[str, Tuple[str, str, str, str]] = {
    "en": (
        "pocket-tts-en.gguf",
        f"{POCKET_TTS_HF_BASE}/pocket-tts-en.gguf",
        "mmproj-pocket-tts-en.gguf",
        f"{POCKET_TTS_HF_BASE}/mmproj-pocket-tts-en.gguf",
    ),
    "de": (
        "pocket-tts-german.gguf",
        f"{POCKET_TTS_HF_BASE}/german/pocket-tts-german.gguf",
        "mmproj-pocket-tts-german.gguf",
        f"{POCKET_TTS_HF_BASE}/german/mmproj-pocket-tts-german.gguf",
    ),
    "es": (
        "pocket-tts-spanish.gguf",
        f"{POCKET_TTS_HF_BASE}/spanish/pocket-tts-spanish.gguf",
        "mmproj-pocket-tts-spanish.gguf",
        f"{POCKET_TTS_HF_BASE}/spanish/mmproj-pocket-tts-spanish.gguf",
    ),
    "fr": (
        "pocket-tts-french.gguf",
        f"{POCKET_TTS_HF_BASE}/french/pocket-tts-french.gguf",
        "mmproj-pocket-tts-french.gguf",
        f"{POCKET_TTS_HF_BASE}/french/mmproj-pocket-tts-french.gguf",
    ),
}
QWEN3_QUANT_SPECS: Dict[str, Dict[str, str]] = {
    "q8": {
        "id": "q8",
        "label": "Q8_0",
        "file": "Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
        "url": "https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF/resolve/main/Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
    },
}
QWEN3_LEGACY_MODEL_FILES: Tuple[str, ...] = (
    "Qwen3-TTS-12Hz-1.7B-Base.Q5_K_M.gguf",
    "Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf",
)
QWEN3_MODEL_FILE = QWEN3_QUANT_SPECS["q8"]["file"]
QWEN3_MMPROJ_FILE = "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf"
QWEN3_MODEL_URL = QWEN3_QUANT_SPECS["q8"]["url"]
QWEN3_MMPROJ_URL = "https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF/resolve/main/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf"
QWEN3_AUTO_Q8_VRAM_GB = 9.0  # При < 9 ГБ VRAM (4/6/8 ГБ карти) пропонуємо ZONOS3-Mini чи Qwen3 Q8; при >= 9 ГБ одразу ставимо Qwen3 Q8
QWEN3_NATIVE_SR = 24000
QWEN3_CTX_SIZE = 2048
QWEN3_SUPPORTED_LANGS = frozenset(("zh", "en", "de", "it", "pt", "es", "ja", "ko", "fr", "ru"))

FALLBACK_VULKAN_ZIP_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b12608/llama-b12608-bin-win-vulkan-x64.zip"
FALLBACK_CUDA_ZIP_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b12608/llama-b12608-bin-win-cuda-12.4-x64.zip"
FALLBACK_CUDART_ZIP_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b12608/cudart-llama-bin-win-cuda-12.4-x64.zip"
FALLBACK_CPU_ZIP_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b12608/llama-b12608-bin-win-cpu-x64.zip"
CUDA_REQUIRED_DLLS = ("ggml-cuda.dll", "cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll")

_REF_WAV_CACHE_DIR = DATA_DIR / "tts_cache" / "ref_wavs"
_PCM_PARAM_RE = re.compile(r"([a-zA-Z0-9_-]+)\s*=\s*([^;\s]+)")


class Qwen3SynthesisCancelledError(RuntimeError):
    """Піднімається, коли активний запит синтезу Qwen3-TTS було негайно перервано користувачем."""


def estimate_qwen3_n_predict(
    text: str,
    *,
    ref_text: str = "",
    ctx_size: int = QWEN3_CTX_SIZE,
) -> int:
    """
    Розраховує динамічний ліміт кадрів генерації (12.5 кадр/с) під довжину тексту із запасом,
    щоб модель не зависала на десятки секунд після завершення фрази при пропуску EOS.
    """
    clean = str(text or "").strip()
    if not clean:
        return 64
    n_chars = len(clean)
    n_pauses = sum(1 for ch in clean if ch in ",.;:!?—–-\n")
    est = int(math.ceil(n_chars * 2.0 + n_pauses * 8 + 64))
    clean_ref = str(ref_text or "").strip()
    ref_tokens = (320 + len(clean_ref)) if clean_ref else 0
    prompt_est = 16 + n_chars + ref_tokens
    ctx_cap = max(96, int(ctx_size) - prompt_est - 64)
    return max(96, min(1200, ctx_cap, est))


def _hidden_process_kwargs() -> Dict[str, Any]:
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}


def is_runtime_installed() -> bool:
    return (
        LLAMA_TTS_EXE.is_file()
        and LLAMA_TTS_SERVER_EXE.is_file()
        and LLAMA_TTS_READY_MARKER.is_file()
    )


def normalize_qwen3_quant(quant: Optional[str] = None) -> str:
    """Повертає єдине підтримуване квантування Qwen3-TTS 1.7B ('q8')."""
    _ = quant
    return "q8"


def get_qwen3_model_spec(quant: Optional[str] = None) -> Dict[str, str]:
    """Повертає специфікацію GGUF-файлу (file/url/label) для Qwen3-TTS 1.7B (Q8_0)."""
    return dict(QWEN3_QUANT_SPECS[normalize_qwen3_quant(quant)])


def should_prompt_qwen3_quant_for_vram(vram_gb: Optional[float]) -> bool:
    """
    Повертає True, якщо на ПК менше 9 ГБ відеопам'яті (або обсяг VRAM невідомий),
    щоб запропонувати вибір між ZONOS3-Mini та Qwen3-TTS 1.7B (Q8). При >= 9 ГБ VRAM одразу ставиться Qwen3 Q8.
    """
    if vram_gb is None:
        return True
    try:
        return float(vram_gb) < float(QWEN3_AUTO_Q8_VRAM_GB)
    except (TypeError, ValueError):
        return True


def find_installed_qwen3_model_file(preferred_quant: Optional[str] = None) -> Optional[Path]:
    """Знаходить встановлений GGUF-файл моделі Qwen3-TTS (Q8_0 або legacy-файли)."""
    _ = preferred_quant
    candidates = [
        QWEN3_MODEL_FILE,
        *QWEN3_LEGACY_MODEL_FILES,
    ]
    seen: Set[str] = set()
    for fname in candidates:
        if not fname or fname in seen:
            continue
        seen.add(fname)
        path = QWEN3_TTS_DIR / fname
        if path.is_file():
            return path
    return None


def is_qwen3_tts_installed() -> bool:
    return (
        is_runtime_installed()
        and find_installed_qwen3_model_file() is not None
        and (QWEN3_TTS_DIR / QWEN3_MMPROJ_FILE).is_file()
    )


def is_zonos3_installed() -> bool:
    return (
        is_runtime_installed()
        and (ZONOS3_DIR / ZONOS3_MODEL_FILE).is_file()
        and (ZONOS3_DIR / ZONOS3_MMPROJ_FILE).is_file()
    )


def is_kokoro_installed() -> bool:
    return (
        is_runtime_installed()
        and (KOKORO_DIR / KOKORO_MODEL_FILE).is_file()
        and (KOKORO_DIR / KOKORO_MMPROJ_FILE).is_file()
    )


def _normalize_pocket_lang(lang: Optional[str] = None) -> str:
    raw = str(lang if lang is not None else config.get("app_language", "en") or "en").strip().lower()
    return raw if raw in POCKET_TTS_LANG_FILES else "en"


def is_pocket_tts_installed(lang: Optional[str] = None) -> bool:
    if not is_runtime_installed():
        return False
    code = _normalize_pocket_lang(lang)
    m_file, _, mm_file, _ = POCKET_TTS_LANG_FILES[code]
    return (POCKET_TTS_DIR / m_file).is_file() and (POCKET_TTS_DIR / mm_file).is_file()


def is_llama_engine_installed(engine_id: str, lang: Optional[str] = None) -> bool:
    eid = str(engine_id or "").strip()
    if eid == "qwen3_tts":
        return is_qwen3_tts_installed()
    if eid == "zonos3_mini":
        return is_zonos3_installed()
    if eid == "kokoro_82m":
        return is_kokoro_installed()
    if eid == "pocket_tts":
        return is_pocket_tts_installed(lang=lang)
    return False


def is_any_llama_tts_installed() -> bool:
    if is_qwen3_tts_installed() or is_zonos3_installed() or is_kokoro_installed():
        return True
    for code in POCKET_TTS_LANG_FILES:
        if is_pocket_tts_installed(code):
            return True
    return False


def get_llama_engine_downloads(engine_id: str, lang: Optional[str] = None) -> list[Tuple[str, Path, str, int, int]]:
    """Повертає список файлів (url, dest_path, fname, base_pct, span_pct) для завантаження GGUF-рушія."""
    eid = str(engine_id or "").strip()
    if eid == "zonos3_mini":
        return [
            (ZONOS3_MODEL_URL, ZONOS3_DIR / ZONOS3_MODEL_FILE, ZONOS3_MODEL_FILE, 25, 35),
            (ZONOS3_MMPROJ_URL, ZONOS3_DIR / ZONOS3_MMPROJ_FILE, ZONOS3_MMPROJ_FILE, 60, 39),
        ]
    if eid == "kokoro_82m":
        return [
            (KOKORO_MODEL_URL, KOKORO_DIR / KOKORO_MODEL_FILE, KOKORO_MODEL_FILE, 25, 25),
            (KOKORO_MMPROJ_URL, KOKORO_DIR / KOKORO_MMPROJ_FILE, KOKORO_MMPROJ_FILE, 50, 49),
        ]
    if eid == "pocket_tts":
        code = _normalize_pocket_lang(lang)
        m_file, m_url, mm_file, mm_url = POCKET_TTS_LANG_FILES[code]
        return [
            (m_url, POCKET_TTS_DIR / m_file, m_file, 25, 50),
            (mm_url, POCKET_TTS_DIR / mm_file, mm_file, 75, 24),
        ]
    q_spec = get_qwen3_model_spec()
    return [
        (q_spec["url"], QWEN3_TTS_DIR / q_spec["file"], q_spec["file"], 25, 45),
        (QWEN3_MMPROJ_URL, QWEN3_TTS_DIR / QWEN3_MMPROJ_FILE, QWEN3_MMPROJ_FILE, 70, 29),
    ]


def resolve_llama_cpp_cuda_asset_urls(
    proxies: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
    """Find the newest official Windows x64 CUDA 12.x llama.cpp zip URL and matching cudart zip URL."""
    try:
        resp = requests.get(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases",
            params={"per_page": 8},
            proxies=proxies,
            timeout=10.0,
        )
        if resp.status_code == 200:
            releases = resp.json()
            if isinstance(releases, list):
                for rel in releases:
                    cuda_url = ""
                    cudart_url = ""
                    for asset in rel.get("assets", []) or []:
                        name = str(asset.get("name", "") or "").lower()
                        url = str(asset.get("browser_download_url", "") or "").strip()
                        if not url or not name.endswith("-x64.zip") or "bin-win-cuda-12" not in name:
                            continue
                        if name.startswith("cudart-llama-"):
                            cudart_url = url
                        elif name.startswith("llama-"):
                            cuda_url = url
                    if cuda_url:
                        return cuda_url, (cudart_url or FALLBACK_CUDART_ZIP_URL)
    except Exception:
        pass
    return FALLBACK_CUDA_ZIP_URL, FALLBACK_CUDART_ZIP_URL


def resolve_llama_cpp_win_asset_url(
    prefer_vulkan: bool = True,
    proxies: Optional[Dict[str, str]] = None,
    prefer_cuda: Optional[bool] = None,
) -> str:
    """Find the newest official Windows x64 zip URL containing llama-tts/llama-server."""
    if prefer_cuda:
        return resolve_llama_cpp_cuda_asset_urls(proxies=proxies)[0]
    token = "win-vulkan-x64.zip" if prefer_vulkan else "win-cpu-x64.zip"
    try:
        resp = requests.get(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases",
            params={"per_page": 8},
            proxies=proxies,
            timeout=10.0,
        )
        if resp.status_code == 200:
            releases = resp.json()
            if isinstance(releases, list):
                for rel in releases:
                    for asset in rel.get("assets", []) or []:
                        name = str(asset.get("name", "") or "").lower()
                        url = str(asset.get("browser_download_url", "") or "").strip()
                        if token in name and url:
                            return url
    except Exception:
        pass
    return FALLBACK_VULKAN_ZIP_URL if prefer_vulkan else FALLBACK_CPU_ZIP_URL


def _extract_binaries_from_zip(zip_path: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            base_name = Path(member.filename).name
            if not base_name.lower().endswith((".exe", ".dll")):
                continue
            target = target_dir / base_name
            with zf.open(member, "r") as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def install_llama_tts_runtime(
    engine_id: str,
    progress_cb: Callable[[int, str], None],
    download_fn: Callable[[str, Path, str, int, int, str], None],
    cancel_check: Callable[[], None],
    proxies: Optional[Dict[str, str]] = None,
) -> None:
    """Download and extract official llama.cpp CUDA (for NVIDIA) or Vulkan (for AMD/Intel) binaries."""
    use_cuda = has_nvidia_gpu()
    if is_runtime_installed():
        try:
            marker_txt = LLAMA_TTS_READY_MARKER.read_text(encoding="utf-8", errors="ignore").strip().lower()
        except Exception:
            marker_txt = ""
        if use_cuda:
            has_all_cuda_dlls = all((LLAMA_TTS_DIR / dll_name).is_file() for dll_name in CUDA_REQUIRED_DLLS)
            if marker_txt == "cuda-ok" and has_all_cuda_dlls:
                return
            if marker_txt == "ok" and not (LLAMA_TTS_DIR / "ggml-vulkan.dll").is_file():
                return
        else:
            if marker_txt in ("vulkan-ok", "ok"):
                return

    shutdown_llama_tts()
    LLAMA_TTS_DIR.mkdir(parents=True, exist_ok=True)
    LLAMA_TTS_READY_MARKER.unlink(missing_ok=True)

    if use_cuda:
        cuda_url, cudart_url = resolve_llama_cpp_cuda_asset_urls(proxies=proxies)
        zip_path = LLAMA_TTS_DIR / "llama_cuda_bin.zip"
        cancel_check()
        download_fn(
            cuda_url,
            zip_path,
            engine_id,
            2,
            10,
            tr("tts.dl.llama_runtime_cuda", "Завантаження рушія llama.cpp (NVIDIA CUDA)..."),
        )
        cancel_check()
        progress_cb(12, tr("tts.dl.extract_runtime", "Розпакування бінарників llama.cpp..."))
        _extract_binaries_from_zip(zip_path, LLAMA_TTS_DIR)
        zip_path.unlink(missing_ok=True)

        need_cudart = not all(
            (LLAMA_TTS_DIR / dll_name).is_file()
            for dll_name in ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll")
        )
        if need_cudart and cudart_url:
            cudart_zip = LLAMA_TTS_DIR / "cudart_bin.zip"
            cancel_check()
            download_fn(
                cudart_url,
                cudart_zip,
                engine_id,
                13,
                8,
                tr("tts.dl.llama_cudart", "Завантаження бібліотек NVIDIA CUDA Runtime..."),
            )
            cancel_check()
            progress_cb(21, tr("tts.dl.extract_runtime", "Розпакування бінарників llama.cpp..."))
            _extract_binaries_from_zip(cudart_zip, LLAMA_TTS_DIR)
            cudart_zip.unlink(missing_ok=True)
        marker_value = "cuda-ok"
    else:
        zip_path = LLAMA_TTS_DIR / "llama_bin.zip"
        url = resolve_llama_cpp_win_asset_url(prefer_vulkan=True, proxies=proxies)
        cancel_check()
        download_fn(
            url,
            zip_path,
            engine_id,
            2,
            18,
            tr("tts.dl.llama_runtime", "Завантаження рушія llama.cpp (Vulkan GPU) (~37 МБ)..."),
        )
        cancel_check()
        progress_cb(21, tr("tts.dl.extract_runtime", "Розпакування бінарників llama.cpp..."))
        _extract_binaries_from_zip(zip_path, LLAMA_TTS_DIR)
        zip_path.unlink(missing_ok=True)
        marker_value = "vulkan-ok"

    if not (LLAMA_TTS_EXE.is_file() and LLAMA_TTS_SERVER_EXE.is_file()):
        raise RuntimeError("Downloaded llama.cpp archive does not contain llama-tts.exe / llama-server.exe")
    LLAMA_TTS_READY_MARKER.write_text(marker_value, encoding="utf-8")


def prepare_reference_wav(audio_path: Path) -> Path:
    """Ensure any reference audio (.wav/.mp3/.ogg/.flac) is stored as mono 24 kHz 16-bit WAV <= 20s."""
    if not audio_path.is_file():
        raise FileNotFoundError(f"Reference audio not found: {audio_path}")
    st = audio_path.stat()
    digest = hashlib.sha256(
        f"{audio_path.resolve()}|{st.st_mtime_ns}|{st.st_size}|20s|24k".encode("utf-8")
    ).hexdigest()[:16]
    _REF_WAV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached_wav = _REF_WAV_CACHE_DIR / f"ref_{digest}.wav"
    if cached_wav.is_file() and cached_wav.stat().st_size > 100:
        return cached_wav

    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    src_sr = max(1, int(sr))
    max_samples = int(src_sr * 20)
    if len(arr) > max_samples:
        arr = arr[:max_samples]
    out_sr = src_sr
    if src_sr != QWEN3_NATIVE_SR and len(arr) > 0:
        from scipy.signal import resample_poly

        g = math.gcd(src_sr, QWEN3_NATIVE_SR)
        arr = resample_poly(arr, QWEN3_NATIVE_SR // g, src_sr // g).astype(np.float32, copy=False)
        out_sr = QWEN3_NATIVE_SR
    peak = float(np.max(np.abs(arr))) if len(arr) else 0.0
    if peak > 0.98:
        arr = arr * (0.95 / peak)
    tmp_wav = cached_wav.with_suffix(".tmp.wav")
    sf.write(str(tmp_wav), arr, int(out_sr), subtype="PCM_16")
    os.replace(tmp_wav, cached_wav)
    return cached_wav


def _resolve_default_sample_for_lang(lang: str) -> Optional[Path]:
    lang_low = str(lang or "en").strip().lower()
    candidates = {
        "en": ("piper/en_amy.ogg", "silero/en_0.ogg"),
        "ru": ("silero/ru_xenia.ogg", "piper/ru_irina.ogg"),
        "de": ("piper/de_eva.ogg", "silero/de_eva.ogg"),
        "fr": ("piper/fr_siwis.ogg",),
        "es": ("piper/es_sharvard_f.ogg",),
        "pl": ("piper/pl_gosia.ogg",),
    }.get(lang_low, ("piper/en_amy.ogg",))
    for rel in candidates:
        p = SAMPLES_DIR / rel
        if p.is_file():
            return p
    return None


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _read_wav_bytes(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    import io

    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    return arr, int(sr)


def _parse_pcm_content_type(content_type: str) -> Tuple[int, str, int, int]:
    """Parse `audio/pcm;rate=24000;encoding=float;bits=32;channels=1` headers."""
    rate = QWEN3_NATIVE_SR
    encoding = "float"
    bits = 32
    channels = 1
    for key, raw_val in _PCM_PARAM_RE.findall(str(content_type or "")):
        k = key.strip().lower()
        v = raw_val.strip().strip('"').lower()
        if k == "rate":
            try:
                rate = max(1, int(v))
            except ValueError:
                pass
        elif k == "encoding":
            encoding = v
        elif k == "bits":
            try:
                bits = int(v)
            except ValueError:
                pass
        elif k == "channels":
            try:
                channels = max(1, int(v))
            except ValueError:
                pass
    return rate, encoding, bits, channels


def _pcm_bytes_to_float32(raw_bytes: bytes, encoding: str, bits: int, channels: int) -> np.ndarray:
    sample_bytes = 2 if (encoding == "signed-int" and bits == 16) else 4
    frame_bytes = sample_bytes * max(1, int(channels))
    usable = (len(raw_bytes) // frame_bytes) * frame_bytes
    if usable <= 0:
        return np.zeros(0, dtype=np.float32)
    view = memoryview(raw_bytes)[:usable]
    if sample_bytes == 2:
        arr = np.frombuffer(view, dtype="<i2").astype(np.float32) / 32768.0
    else:
        arr = np.frombuffer(view, dtype="<f4").copy()
    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    return arr


def _decode_audio_response(raw_bytes: bytes, content_type: str) -> Tuple[np.ndarray, int]:
    ct_low = str(content_type or "").lower()
    if "audio/pcm" in ct_low:
        sr, encoding, bits, channels = _parse_pcm_content_type(ct_low)
        return _pcm_bytes_to_float32(raw_bytes, encoding, bits, channels), sr
    return _read_wav_bytes(raw_bytes)


def _decode_llama_tts_failure(res: subprocess.CompletedProcess, engine_name: str) -> str:
    tail = (res.stderr or res.stdout or "").strip()
    if len(tail) > 500:
        tail = tail[-500:]
    return tail or f"{engine_name} exited with code {res.returncode}"


def _force_close_socket(sock: Any) -> None:
    """Force-abort an in-flight blocking socket immediately on Windows (even with makefile refs)."""
    if sock is None:
        return
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    real_close = getattr(sock, "_real_close", None)
    if callable(real_close):
        try:
            real_close()
            return
        except OSError:
            pass
    try:
        sock.close()
    except OSError:
        pass


class _CancellableHTTPAdapter(HTTPAdapter):
    """HTTPAdapter that tracks sockets of in-flight /tts requests for instant cancellation."""

    def __init__(self, manager: "_Qwen3ServerManager") -> None:
        self._manager = manager
        super().__init__()

        class _TrackedHTTPConnection(HTTPConnection):
            def connect(conn_self) -> None:
                super().connect()
                manager._register_current_thread_socket(conn_self.sock)

            def endheaders(conn_self, *args, **kwargs) -> None:
                manager._register_current_thread_socket(conn_self.sock)
                super().endheaders(*args, **kwargs)

            def close(conn_self) -> None:
                manager._unregister_socket(conn_self.sock)
                super().close()

        self._conn_cls = _TrackedHTTPConnection

    def get_connection_with_tls_context(self, request, verify, proxies=None, cert=None):
        pool = super().get_connection_with_tls_context(request, verify, proxies=proxies, cert=cert)
        pool.ConnectionCls = self._conn_cls
        return pool


class _Qwen3ServerManager:
    """Manages a persistent `llama-server.exe` process for Qwen3-TTS (0.6B and 1.7B)."""

    IDLE_UNLOAD_SEC = 30.0

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._startup_lock = threading.Lock()
        self._req_lock = threading.Lock()
        self._req_local = threading.local()
        self._proc: Optional[subprocess.Popen] = None
        self._port: int = 0
        self._device_mode: str = ""
        self._engine_id: str = ""
        self._model_path: str = ""
        self._target_engine_id: str = "qwen3_tts"
        self._start_gen: int = 0
        self._cancel_gen: int = 0
        self._warmed_port: int = 0
        self._active_user_requests: int = 0
        self._active_sockets: Dict[int, Tuple[Set[Any], bool]] = {}
        self._active_responses: Dict[int, Tuple[Any, bool]] = {}
        self._session = requests.Session()
        self._session.mount("http://", _CancellableHTTPAdapter(self))
        self._stderr_tail: deque[str] = deque(maxlen=30)
        self._idle_timer: Optional[threading.Timer] = None
        self._idle_timeout_sec: float = self.IDLE_UNLOAD_SEC
        self._ref_bytes_cache: Tuple[str, bytes] = ("", b"")

    def _register_current_thread_socket(self, sock: Any) -> None:
        if sock is None or not getattr(self._req_local, "active", False):
            return
        tid = threading.get_ident()
        is_warmup = bool(getattr(self._req_local, "is_warmup", False))
        with self._req_lock:
            entry = self._active_sockets.get(tid)
            if entry is None:
                self._active_sockets[tid] = ({sock}, is_warmup)
            else:
                entry[0].add(sock)

    def _unregister_socket(self, sock: Any) -> None:
        if sock is None:
            return
        with self._req_lock:
            empty_tids = []
            for tid, (socks, _is_w) in self._active_sockets.items():
                socks.discard(sock)
                if not socks:
                    empty_tids.append(tid)
            for tid in empty_tids:
                self._active_sockets.pop(tid, None)

    def _clear_thread_tracking(self, tid: int) -> None:
        with self._req_lock:
            self._active_sockets.pop(tid, None)
            self._active_responses.pop(tid, None)

    def cancel_active(self, *, only_warmup: bool = False) -> None:
        """Immediately abort active /tts HTTP requests and sockets so llama-server halts GPU work."""
        socks_to_close = []
        resps_to_close = []
        with self._req_lock:
            if not only_warmup:
                self._cancel_gen += 1
            for tid, (socks, is_w) in list(self._active_sockets.items()):
                if only_warmup and not is_w:
                    continue
                socks_to_close.extend(socks)
                self._active_sockets.pop(tid, None)
            for tid, (resp, is_w) in list(self._active_responses.items()):
                if only_warmup and not is_w:
                    continue
                resps_to_close.append(resp)
                self._active_responses.pop(tid, None)
        for sock in socks_to_close:
            _force_close_socket(sock)
        for resp in resps_to_close:
            try:
                resp.close()
            except Exception:
                pass

    def is_running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def _cancel_idle_timer(self) -> None:
        if self._idle_timer is not None:
            try:
                self._idle_timer.cancel()
            except Exception:
                pass
            self._idle_timer = None

    def _is_qwen3_selected_in_config(self) -> bool:
        active = str(config.get("active_engine", "") or "").strip()
        fallback = str(config.get("fallback_engine", "") or "").strip()
        cur_eng = self._engine_id or "qwen3_tts"
        return active == cur_eng or fallback == cur_eng

    def _on_idle_timeout(self) -> None:
        if self._is_qwen3_selected_in_config():
            return
        self.stop()

    def _reset_idle_timer(self) -> None:
        with self._lock:
            self._cancel_idle_timer()
            if self._proc is None or self._proc.poll() is not None or self._idle_timeout_sec <= 0:
                return
            if self._is_qwen3_selected_in_config():
                return
            t = threading.Timer(self._idle_timeout_sec, self._on_idle_timeout)
            t.daemon = True
            t.name = "Qwen3IdleUnload"
            self._idle_timer = t
            t.start()

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        pipe = proc.stderr
        if pipe is None:
            return
        try:
            for raw_line in iter(pipe.readline, b""):
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line:
                    self._stderr_tail.append(line)
        except Exception:
            pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def stop(self, *, _bump_cancel: bool = True) -> None:
        if _bump_cancel:
            self.cancel_active(only_warmup=False)
        else:
            self.cancel_active(only_warmup=True)
        with self._lock:
            self._start_gen += 1
            self._warmed_port = 0
            self._cancel_idle_timer()
            proc = self._proc
            self._proc = None
            self._port = 0
            self._device_mode = ""
            self._engine_id = ""
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=1.5)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=1.0)
                except Exception:
                    pass

    def _wait_ready(
        self,
        proc: subprocess.Popen,
        port: int,
        timeout: float = 90.0,
        start_gen: Optional[int] = None,
    ) -> None:
        deadline = time.monotonic() + timeout
        url = f"http://127.0.0.1:{port}/health"
        while time.monotonic() < deadline:
            if start_gen is not None and (self._start_gen != start_gen or self._proc is not proc):
                raise RuntimeError("Qwen3-TTS server startup cancelled")
            if proc.poll() is not None:
                tail = " | ".join(self._stderr_tail)[-300:]
                suffix = f": {tail}" if tail else ""
                raise RuntimeError(f"Qwen3-TTS server exited early (code {proc.returncode}){suffix}")
            try:
                r = self._session.get(url, timeout=0.6)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.15)
        raise RuntimeError("Timed out waiting for Qwen3-TTS server startup")

    @staticmethod
    def _resolve_engine_files(eng_id: str, lang: str = "") -> Tuple[Path, Path, str]:
        if eng_id == "zonos3_mini":
            if not is_zonos3_installed():
                raise RuntimeError(
                    tr("tts.err.model_not_installed", "Модель {name} не встановлена. Натисніть «Встановити» у списку локальних TTS.", name="ZONOS3-Mini")
                )
            return ZONOS3_DIR / ZONOS3_MODEL_FILE, ZONOS3_DIR / ZONOS3_MMPROJ_FILE, "ZONOS3-Mini"
        if eng_id == "kokoro_82m":
            if not is_kokoro_installed():
                raise RuntimeError(
                    tr("tts.err.model_not_installed", "Модель {name} не встановлена. Натисніть «Встановити» у списку локальних TTS.", name="Kokoro-82M")
                )
            return KOKORO_DIR / KOKORO_MODEL_FILE, KOKORO_DIR / KOKORO_MMPROJ_FILE, "Kokoro-82M"
        if eng_id == "pocket_tts":
            code = _normalize_pocket_lang(lang)
            if not is_pocket_tts_installed(code):
                raise RuntimeError(
                    tr("tts.err.model_not_installed", "Модель {name} не встановлена. Натисніть «Встановити» у списку локальних TTS.", name="Pocket TTS")
                )
            m_file, _, mm_file, _ = POCKET_TTS_LANG_FILES[code]
            return POCKET_TTS_DIR / m_file, POCKET_TTS_DIR / mm_file, "Pocket TTS"
        if not is_qwen3_tts_installed():
            raise RuntimeError(
                tr("tts.err.model_not_installed", "Модель {name} не встановлена. Натисніть «Встановити» у списку локальних TTS.", name="Qwen3-TTS")
            )
        model_path = find_installed_qwen3_model_file() or (QWEN3_TTS_DIR / QWEN3_MODEL_FILE)
        return model_path, QWEN3_TTS_DIR / QWEN3_MMPROJ_FILE, "Qwen3-TTS"

    def ensure_started(self, device_mode: str, engine_id: Optional[str] = None, lang: str = "") -> int:
        with self._startup_lock:
            with self._lock:
                eng_id = str(engine_id or self._target_engine_id or "qwen3_tts").strip() or "qwen3_tts"
                model_path, mmproj_path, _title = self._resolve_engine_files(eng_id, lang=lang)
                eff_device = "cpu" if (device_mode == "cpu" or (eng_id in ("kokoro_82m", "pocket_tts") and not has_hardware_gpu())) else "gpu"

                if (
                    self._proc is not None
                    and self._proc.poll() is None
                    and self._device_mode == eff_device
                    and self._engine_id == eng_id
                    and self._model_path == str(model_path)
                    and self._port > 0
                ):
                    return self._port
                self.stop(_bump_cancel=False)
                self._stderr_tail.clear()
                port = _pick_free_port()
                ngl = "0" if eff_device == "cpu" else "99"
                threads = str(max(2, (os.cpu_count() or 8) // 2))
                cmd = [
                    str(LLAMA_TTS_SERVER_EXE),
                    "--host", "127.0.0.1",
                    "--port", str(port),
                    "-m", str(model_path),
                    "-mm", str(mmproj_path),
                    "-ngl", ngl,
                    "-t", threads,
                    "-c", str(QWEN3_CTX_SIZE),
                    "-np", "1",
                    "-fa", "on",
                ]
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(LLAMA_TTS_DIR),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    **_hidden_process_kwargs(),
                )
                assign_process_to_kill_on_close_job(proc)
                self._proc = proc
                self._port = 0
                self._device_mode = eff_device
                self._engine_id = eng_id
                self._model_path = str(model_path)
                my_gen = self._start_gen
                threading.Thread(target=self._drain_stderr, args=(proc,), daemon=True, name="LlamaTTSStderrDrain").start()

            try:
                self._wait_ready(proc, port, timeout=90.0, start_gen=my_gen)
            except Exception:
                with self._lock:
                    if self._proc is proc:
                        self._proc = None
                        self._port = 0
                        self._device_mode = ""
                        self._engine_id = ""
                try:
                    proc.kill()
                except Exception:
                    pass
                raise

            with self._lock:
                if self._proc is not proc or self._start_gen != my_gen:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    raise RuntimeError("Llama-TTS server stopped during startup")
                self._port = port
                self._warmed_port = 0
                self._reset_idle_timer()
                return port

    def _get_ref_wav_bytes(self, ref_wav: Path) -> bytes:
        ref_key = str(ref_wav)
        if self._ref_bytes_cache[0] == ref_key and self._ref_bytes_cache[1]:
            return self._ref_bytes_cache[1]
        raw = ref_wav.read_bytes()
        self._ref_bytes_cache = (ref_key, raw)
        return raw

    def _build_tts_request_kwargs(
        self,
        text: str,
        lang: str,
        ref_wav: Optional[Path],
        ref_text: str,
        *,
        n_predict: Optional[int],
        stream: bool,
        voice: str = "",
        use_lang: bool = True,
    ) -> Dict[str, Any]:
        clean_ref_text = str(ref_text or "").strip()
        clean_voice = str(voice or "").strip()
        if ref_wav is not None and ref_wav.is_file():
            wav_bytes = self._get_ref_wav_bytes(ref_wav)
            form_data: Dict[str, str] = {
                "input": text,
                "response_format": "pcm",
                "stream": "true" if stream else "false",
            }
            if use_lang and lang:
                form_data["lang"] = lang
            if clean_voice:
                form_data["voice"] = clean_voice
            if n_predict is not None:
                form_data["n_predict"] = str(int(n_predict))
            if clean_ref_text:
                form_data["speaker_ref_text"] = clean_ref_text
            return {
                "data": form_data,
                "files": {"speaker_ref": ("ref.wav", wav_bytes, "audio/wav")},
            }
        payload: Dict[str, Any] = {
            "input": text,
            "response_format": "pcm",
            "stream": bool(stream),
        }
        if use_lang and lang:
            payload["lang"] = lang
        if clean_voice:
            payload["voice"] = clean_voice
        if n_predict is not None:
            payload["n_predict"] = int(n_predict)
        return {"json": payload}

    def _is_cancelled(self, my_cancel_gen: int) -> bool:
        with self._req_lock:
            return self._cancel_gen != my_cancel_gen

    def warmup_inference(
        self,
        *,
        port: int,
        ref_wav: Optional[Path],
        lang: str,
        ref_text: str = "",
        voice: str = "",
        use_lang: bool = True,
        n_predict: Optional[int] = 2,
    ) -> bool:
        """Run a tiny warmup /tts request to compile GPU shaders/graphs before first phrase."""
        if port <= 0:
            return False
        with self._req_lock:
            if self._warmed_port == port or self._active_user_requests > 0:
                return False
        tid = threading.get_ident()
        self._req_local.active = True
        self._req_local.is_warmup = True
        resp = None
        try:
            req_kwargs = self._build_tts_request_kwargs(
                "a.",
                lang,
                ref_wav,
                ref_text,
                n_predict=n_predict,
                stream=False,
                voice=voice,
                use_lang=use_lang,
            )
            resp = self._session.post(
                f"http://127.0.0.1:{port}/tts",
                timeout=30.0,
                **req_kwargs,
            )
            with self._req_lock:
                self._active_responses[tid] = (resp, True)
            ok = getattr(resp, "status_code", 0) == 200
            if ok:
                with self._req_lock:
                    self._warmed_port = port
            return ok
        except Exception:
            return False
        finally:
            self._req_local.active = False
            self._req_local.is_warmup = False
            self._clear_thread_tracking(tid)
            if resp is not None:
                try:
                    resp.close()
                except Exception:
                    pass
            self._reset_idle_timer()

    def synthesize(
        self,
        text: str,
        lang: str,
        ref_wav: Optional[Path],
        ref_text: str,
        device_mode: str,
        engine_id: str = "qwen3_tts",
        *,
        voice: str = "",
        use_lang: bool = True,
        include_n_predict: bool = True,
        n_predict_fps: float = 12.5,
    ) -> Tuple[np.ndarray, int]:
        self.cancel_active(only_warmup=True)
        with self._req_lock:
            my_cancel_gen = self._cancel_gen
            self._active_user_requests += 1
        tid = threading.get_ident()
        resp = None
        try:
            self._target_engine_id = engine_id
            try:
                port = self.ensure_started(device_mode) if engine_id == "qwen3_tts" else self.ensure_started(device_mode, engine_id=engine_id, lang=lang)
            finally:
                self._target_engine_id = "qwen3_tts"
            if self._is_cancelled(my_cancel_gen):
                raise Qwen3SynthesisCancelledError("Llama-TTS synthesis cancelled")

            n_predict: Optional[int] = None
            if include_n_predict:
                scale = max(1.0, float(n_predict_fps) / 12.5)
                base_n = estimate_qwen3_n_predict(text, ref_text=ref_text, ctx_size=QWEN3_CTX_SIZE)
                n_predict = base_n if abs(scale - 1.0) < 0.01 else min(1400, int(math.ceil(base_n * scale)))
            req_kwargs = self._build_tts_request_kwargs(
                text,
                lang,
                ref_wav,
                ref_text,
                n_predict=n_predict,
                stream=False,
                voice=voice,
                use_lang=use_lang,
            )
            self._req_local.active = True
            self._req_local.is_warmup = False
            try:
                resp = self._session.post(
                    f"http://127.0.0.1:{port}/tts",
                    timeout=120.0,
                    **req_kwargs,
                )
            except Exception as exc:
                if self._is_cancelled(my_cancel_gen):
                    raise Qwen3SynthesisCancelledError("Llama-TTS synthesis cancelled") from exc
                raise

            with self._req_lock:
                self._active_responses[tid] = (resp, False)
            if self._is_cancelled(my_cancel_gen):
                raise Qwen3SynthesisCancelledError("Llama-TTS synthesis cancelled")
            if resp.status_code != 200:
                raise RuntimeError(f"Llama-TTS server error ({resp.status_code}): {resp.text[:300]}")
            with self._req_lock:
                self._warmed_port = port
            headers = getattr(resp, "headers", None)
            content_type = str(headers.get("Content-Type", "") or "") if hasattr(headers, "get") else ""
            return _decode_audio_response(resp.content, content_type)
        finally:
            self._req_local.active = False
            self._req_local.is_warmup = False
            self._clear_thread_tracking(tid)
            with self._req_lock:
                self._active_user_requests = max(0, self._active_user_requests - 1)
            if resp is not None:
                try:
                    resp.close()
                except Exception:
                    pass
            self._reset_idle_timer()

    def synthesize_stream(
        self,
        text: str,
        lang: str,
        ref_wav: Optional[Path],
        ref_text: str,
        device_mode: str,
        engine_id: str = "qwen3_tts",
        *,
        voice: str = "",
        use_lang: bool = True,
        include_n_predict: bool = True,
        n_predict_fps: float = 12.5,
    ) -> Iterator[Tuple[np.ndarray, int]]:
        """Stream raw float32 PCM audio chunks from llama-server /tts as soon as frames decode."""
        self.cancel_active(only_warmup=True)
        with self._req_lock:
            my_cancel_gen = self._cancel_gen
            self._active_user_requests += 1
        tid = threading.get_ident()
        resp = None
        try:
            self._target_engine_id = engine_id
            try:
                port = self.ensure_started(device_mode) if engine_id == "qwen3_tts" else self.ensure_started(device_mode, engine_id=engine_id, lang=lang)
            finally:
                self._target_engine_id = "qwen3_tts"
            if self._is_cancelled(my_cancel_gen):
                raise Qwen3SynthesisCancelledError("Llama-TTS synthesis cancelled")

            n_predict: Optional[int] = None
            if include_n_predict:
                scale = max(1.0, float(n_predict_fps) / 12.5)
                base_n = estimate_qwen3_n_predict(text, ref_text=ref_text, ctx_size=QWEN3_CTX_SIZE)
                n_predict = base_n if abs(scale - 1.0) < 0.01 else min(1400, int(math.ceil(base_n * scale)))
            req_kwargs = self._build_tts_request_kwargs(
                text,
                lang,
                ref_wav,
                ref_text,
                n_predict=n_predict,
                stream=True,
                voice=voice,
                use_lang=use_lang,
            )
            self._req_local.active = True
            self._req_local.is_warmup = False
            try:
                resp = self._session.post(
                    f"http://127.0.0.1:{port}/tts",
                    timeout=120.0,
                    stream=True,
                    **req_kwargs,
                )
            except Exception as exc:
                if self._is_cancelled(my_cancel_gen):
                    raise Qwen3SynthesisCancelledError("Qwen3-TTS synthesis cancelled") from exc
                raise

            with self._req_lock:
                self._active_responses[tid] = (resp, False)
            if self._is_cancelled(my_cancel_gen):
                raise Qwen3SynthesisCancelledError("Qwen3-TTS synthesis cancelled")
            if resp.status_code != 200:
                raise RuntimeError(f"Qwen3-TTS server error ({resp.status_code}): {resp.text[:300]}")
            with self._req_lock:
                self._warmed_port = port

            headers = getattr(resp, "headers", None)
            content_type = str(headers.get("Content-Type", "") or "") if hasattr(headers, "get") else ""
            if "audio/pcm" not in content_type.lower() or not hasattr(resp, "iter_content"):
                samples, sr = _decode_audio_response(resp.content, content_type)
                if len(samples) > 0:
                    yield samples, sr
                return

            sr, encoding, bits, channels = _parse_pcm_content_type(content_type)
            sample_bytes = 2 if (encoding == "signed-int" and bits == 16) else 4
            frame_bytes = sample_bytes * max(1, int(channels))
            carry = bytearray()
            try:
                for raw_part in resp.iter_content(chunk_size=32768):
                    if self._is_cancelled(my_cancel_gen):
                        raise Qwen3SynthesisCancelledError("Qwen3-TTS synthesis cancelled")
                    if not raw_part:
                        continue
                    carry.extend(raw_part)
                    usable = (len(carry) // frame_bytes) * frame_bytes
                    if usable <= 0:
                        continue
                    chunk_bytes = bytes(carry[:usable])
                    del carry[:usable]
                    arr = _pcm_bytes_to_float32(chunk_bytes, encoding, bits, channels)
                    if len(arr) > 0:
                        yield arr, sr
            except Qwen3SynthesisCancelledError:
                raise
            except Exception as exc:
                if self._is_cancelled(my_cancel_gen):
                    raise Qwen3SynthesisCancelledError("Qwen3-TTS synthesis cancelled") from exc
                raise
        finally:
            self._req_local.active = False
            self._req_local.is_warmup = False
            self._clear_thread_tracking(tid)
            with self._req_lock:
                self._active_user_requests = max(0, self._active_user_requests - 1)
            if resp is not None:
                try:
                    resp.close()
                except Exception:
                    pass
            self._reset_idle_timer()


_qwen3_server = _Qwen3ServerManager()
atexit.register(_qwen3_server.stop)


def is_qwen3_server_running() -> bool:
    return _qwen3_server.is_running()


def cancel_qwen3_synthesis() -> None:
    """Immediately cancel any active Qwen3-TTS HTTP synthesis request."""
    _qwen3_server.cancel_active(only_warmup=False)


def shutdown_llama_tts() -> None:
    _qwen3_server.stop()


def _resolve_qwen3_warmup_lang() -> str:
    active_voice = str(config.get("active_voice", "") or "").strip().lower()
    if active_voice.startswith("qwen3_") and "_" in active_voice:
        cand = active_voice.rsplit("_", 1)[-1]
        if cand in QWEN3_SUPPORTED_LANGS:
            return cand
    app_lang = str(config.get("app_language", "en") or "en").strip().lower()
    if app_lang in QWEN3_SUPPORTED_LANGS:
        return app_lang
    return "en"


def _resolve_qwen3_reference(lang: str) -> Tuple[Optional[Path], str]:
    ref_raw = str(config.get("qwen3_tts_ref_path", "") or "").strip()
    ref_text = str(config.get("qwen3_tts_ref_text", "") or "").strip()
    ref_wav: Optional[Path] = None
    if ref_raw and Path(ref_raw).is_file():
        ref_wav = prepare_reference_wav(Path(ref_raw))
    else:
        default_sample = _resolve_default_sample_for_lang(lang)
        if default_sample is not None:
            ref_wav = prepare_reference_wav(default_sample)
            ref_text = ""
    return ref_wav, ref_text


def warmup_qwen3_tts() -> None:
    if not is_qwen3_tts_installed():
        return
    warmup_lang = _resolve_qwen3_warmup_lang()
    ref_wav: Optional[Path] = None
    ref_text = ""
    try:
        ref_wav, ref_text = _resolve_qwen3_reference(warmup_lang)
    except Exception:
        pass
    try:
        port = _qwen3_server.ensure_started("gpu")
    except Exception:
        return
    try:
        _qwen3_server.warmup_inference(
            port=port,
            ref_wav=ref_wav,
            lang=warmup_lang,
            ref_text=ref_text,
        )
    except Exception:
        pass


def synthesize_qwen3_tts(
    text: str,
    lang: str,
    speed: float = 1.0,
) -> Tuple[np.ndarray, int]:
    if not is_qwen3_tts_installed():
        raise RuntimeError(tr("tts.err.model_not_installed", "Модель {name} не встановлена. Натисніть «Встановити» у списку локальних TTS.", name="Qwen3-TTS"))

    ref_wav, ref_text = _resolve_qwen3_reference(lang)
    lang_code = str(lang or "en").strip().lower()
    samples, sr = _qwen3_server.synthesize(text, lang_code, ref_wav, ref_text, "gpu")

    if abs(speed - 1.0) > 0.04 and len(samples) > 100:
        from scipy.signal import resample

        new_len = max(1, int(len(samples) / speed))
        samples = resample(samples, new_len).astype(np.float32)
    return samples, sr


def synthesize_qwen3_tts_stream(
    text: str,
    lang: str,
    speed: float = 1.0,
) -> Iterator[Tuple[np.ndarray, int]]:
    if not is_qwen3_tts_installed():
        raise RuntimeError(tr("tts.err.model_not_installed", "Модель {name} не встановлена. Натисніть «Встановити» у списку локальних TTS.", name="Qwen3-TTS"))

    if abs(speed - 1.0) > 0.04:
        samples, sr = synthesize_qwen3_tts(text, lang=lang, speed=speed)
        if len(samples) > 0:
            yield samples, sr
        return

    ref_wav, ref_text = _resolve_qwen3_reference(lang)
    lang_code = str(lang or "en").strip().lower()
    yield from _qwen3_server.synthesize_stream(text, lang_code, ref_wav, ref_text, "gpu")


def is_llama_server_running() -> bool:
    return _qwen3_server.is_running()


def _resolve_optional_ref_wav(engine_id: str, lang: str, fallback_to_default_sample: bool = False) -> Optional[Path]:
    ref_raw = str(config.get(f"{engine_id}_ref_path", "") or "").strip()
    if ref_raw and Path(ref_raw).is_file():
        return prepare_reference_wav(Path(ref_raw))
    if fallback_to_default_sample:
        sample = _resolve_default_sample_for_lang(lang)
        if sample is not None:
            return prepare_reference_wav(sample)
    return None


def _resolve_llama_call_params(
    engine_id: str,
    text: str,
    lang: str,
    voice: str,
) -> Tuple[str, str, Optional[Path], str, str, bool, bool, float]:
    """Повертає (clean_text, req_lang, ref_wav, req_voice, device_mode, use_lang, include_n_predict, fps)."""
    eid = str(engine_id or "").strip()
    clean_text = str(text or "").strip()
    lang_code = str(lang or "en").strip().lower() or "en"
    clean_voice = str(voice or "").strip()

    if eid == "zonos3_mini":
        ref_wav = _resolve_optional_ref_wav("zonos3_mini", lang_code, fallback_to_default_sample=False)
        req_voice = "" if ref_wav is not None else clean_voice
        req_lang = lang_code if lang_code in ZONOS3_NORM_LANGS else ""
        return clean_text, req_lang, ref_wav, req_voice, "gpu", bool(req_lang), True, 15.0

    if eid == "kokoro_82m":
        # У Kokoro 'b*' — British English (en-gb), 'a*' — American English (en-us)
        req_voice = clean_voice or "af_heart"
        req_lang = "en-gb" if req_voice.startswith("b") else "en-us"
        dev = "gpu" if has_hardware_gpu() else "cpu"
        return clean_text, req_lang, None, req_voice, dev, True, False, 12.5

    if eid == "pocket_tts":
        pocket_lang = _normalize_pocket_lang(lang_code)
        # Як зазначено у EryriLabs/pocket-tts-GGUF README: для іспанської початковий '¡' може обривати генерацію
        if pocket_lang == "es":
            clean_text = clean_text.lstrip("¡").strip() or clean_text
        ref_wav = _resolve_optional_ref_wav("pocket_tts", pocket_lang, fallback_to_default_sample=True)
        dev = "gpu" if has_hardware_gpu() else "cpu"
        return clean_text, pocket_lang, ref_wav, "", dev, False, True, 12.5

    raise RuntimeError(f"Unsupported llama.cpp TTS engine: {eid}")


def warmup_llama_tts(engine_id: str, lang: str = "", voice: str = "") -> None:
    eid = str(engine_id or "").strip()
    if eid == "qwen3_tts":
        warmup_qwen3_tts()
        return
    eff_lang = str(lang or config.get("app_language", "en") or "en").strip().lower() or "en"
    if not is_llama_engine_installed(eid, lang=eff_lang):
        return
    try:
        _, req_lang, ref_wav, req_voice, dev, use_lang, include_n_predict, _ = _resolve_llama_call_params(
            eid, "a.", eff_lang, voice
        )
        port = _qwen3_server.ensure_started(dev, engine_id=eid, lang=eff_lang)
        _qwen3_server.warmup_inference(
            port=port,
            ref_wav=ref_wav,
            lang=req_lang,
            ref_text="",
            voice=req_voice,
            use_lang=use_lang,
            n_predict=2 if include_n_predict else None,
        )
    except Exception:
        pass


def synthesize_llama_tts(
    engine_id: str,
    text: str,
    lang: str = "en",
    voice: str = "",
    speed: float = 1.0,
) -> Tuple[np.ndarray, int]:
    eid = str(engine_id or "").strip()
    if eid == "qwen3_tts":
        return synthesize_qwen3_tts(text, lang=lang, speed=speed)

    clean_text, req_lang, ref_wav, req_voice, dev, use_lang, include_n_predict, fps = _resolve_llama_call_params(
        eid, text, lang, voice
    )
    samples, sr = _qwen3_server.synthesize(
        clean_text,
        req_lang or lang,
        ref_wav,
        "",
        dev,
        engine_id=eid,
        voice=req_voice,
        use_lang=use_lang,
        include_n_predict=include_n_predict,
        n_predict_fps=fps,
    )
    if abs(speed - 1.0) > 0.04 and len(samples) > 100:
        from scipy.signal import resample

        new_len = max(1, int(len(samples) / speed))
        samples = resample(samples, new_len).astype(np.float32)
    return samples, sr


def synthesize_llama_tts_stream(
    engine_id: str,
    text: str,
    lang: str = "en",
    voice: str = "",
    speed: float = 1.0,
) -> Iterator[Tuple[np.ndarray, int]]:
    eid = str(engine_id or "").strip()
    if eid == "qwen3_tts":
        yield from synthesize_qwen3_tts_stream(text, lang=lang, speed=speed)
        return
    if abs(speed - 1.0) > 0.04:
        samples, sr = synthesize_llama_tts(eid, text, lang=lang, voice=voice, speed=speed)
        if len(samples) > 0:
            yield samples, sr
        return
    clean_text, req_lang, ref_wav, req_voice, dev, use_lang, include_n_predict, fps = _resolve_llama_call_params(
        eid, text, lang, voice
    )
    yield from _qwen3_server.synthesize_stream(
        clean_text,
        req_lang or lang,
        ref_wav,
        "",
        dev,
        engine_id=eid,
        voice=req_voice,
        use_lang=use_lang,
        include_n_predict=include_n_predict,
        n_predict_fps=fps,
    )
