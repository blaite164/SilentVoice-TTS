"""
core/omnivoice_tts.py — Local GGUF TTS runtime for OmniVoice (`omnivoice.cpp`).

Features:
- On-demand download of patched Windows x64 `tts-server.exe` (CUDA for NVIDIA or Vulkan for AMD/Intel/NVIDIA)
  and pinned GGUF weights from `Serveurperso/OmniVoice-GGUF` (`Q8_0` default, optional `BF16` / `Q4_K_M`).
- Resident local HTTP server on `127.0.0.1` with per-launch `--api-key` authentication,
  configurable MaskGIT decode steps (`8 / 16 / 24 / 32`), and in-memory voice prompt caching.
- Pause-aware smart reference audio trimming (~10s max, RMS pause detection + fade-out + tail silence).
- Automatic one-time local reference transcription via `whisper.cpp` (`whisper-cli.exe` + `ggml-small-q5_1.bin`),
  cached on disk by audio SHA-256 so every voice clip clones accurately without manual transcription.
- Native Voice Design (6 official attribute categories + `whisper` + pinned `seed`) and non-verbal tags (`[laughter]`, `[sigh]`).
"""

from __future__ import annotations

import atexit
import base64
from collections import deque
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
import zipfile

import numpy as np
import requests
from requests.adapters import HTTPAdapter
import soundfile as sf
from urllib3.connection import HTTPConnection

from core.config import DATA_DIR, ENGINES_DIR, config
from core.i18n import tr
from core.win_api import assign_process_to_kill_on_close_job, has_nvidia_gpu

# ----------------------------------------------------------------------
# Paths, pinned revisions & download specs
# ----------------------------------------------------------------------

OMNIVOICE_DIR = ENGINES_DIR / "omnivoice"
OMNIVOICE_RUNTIME_DIR = ENGINES_DIR / "omnivoice_runtime"
OMNIVOICE_SERVER_EXE = OMNIVOICE_RUNTIME_DIR / "tts-server.exe"
OMNIVOICE_RUNTIME_MARKER = OMNIVOICE_RUNTIME_DIR / "ready.marker"

OMNIVOICE_WHISPER_DIR = ENGINES_DIR / "omnivoice_whisper"
OMNIVOICE_WHISPER_EXE = OMNIVOICE_WHISPER_DIR / "whisper-cli.exe"
OMNIVOICE_WHISPER_MODEL_FILE = "ggml-small-q5_1.bin"
OMNIVOICE_WHISPER_MODEL_PATH = OMNIVOICE_WHISPER_DIR / OMNIVOICE_WHISPER_MODEL_FILE
OMNIVOICE_WHISPER_MARKER = OMNIVOICE_WHISPER_DIR / "ready.marker"

OMNIVOICE_REF_CACHE_DIR = DATA_DIR / "tts_cache" / "omnivoice_refs"

OMNIVOICE_NATIVE_SR = 24000
OMNIVOICE_MAX_REF_SEC = 10.0
OMNIVOICE_HF_REPO = "Serveurperso/OmniVoice-GGUF"
OMNIVOICE_HF_REV = "361609388ae572a820d085185bbbe2a2aac4b30e"

OMNIVOICE_RUNTIME_TAG = "omnivoice-runtime-v1"
OMNIVOICE_GITHUB_REPO = "blaite164/SilentVoice-TTS"
OMNIVOICE_GITLAB_PROJECT = "blaite164%2Fsilentvoice-tts"

FALLBACK_WHISPER_WIN_ZIP_URL = (
    "https://github.com/ggml-org/whisper.cpp/releases/download/b5907/whisper-bin-x64.zip"
)
WHISPER_MODEL_URL = (
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q5_1.bin"
)

CUDA_REQUIRED_DLLS = ("ggml-cuda.dll", "cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll")

OMNIVOICE_QUANT_SPECS: Dict[str, Dict[str, str]] = {
    "Q8_0": {
        "id": "Q8_0",
        "label": "Q8_0 (~1.0 GB)",
        "base_file": "omnivoice-base-Q8_0.gguf",
        "codec_file": "omnivoice-tokenizer-Q8_0.gguf",
        "base_url": f"https://huggingface.co/{OMNIVOICE_HF_REPO}/resolve/{OMNIVOICE_HF_REV}/omnivoice-base-Q8_0.gguf",
        "codec_url": f"https://huggingface.co/{OMNIVOICE_HF_REPO}/resolve/{OMNIVOICE_HF_REV}/omnivoice-tokenizer-Q8_0.gguf",
    },
    "BF16": {
        "id": "BF16",
        "label": "BF16 (~1.6 GB)",
        "base_file": "omnivoice-base-BF16.gguf",
        "codec_file": "omnivoice-tokenizer-BF16.gguf",
        "base_url": f"https://huggingface.co/{OMNIVOICE_HF_REPO}/resolve/{OMNIVOICE_HF_REV}/omnivoice-base-BF16.gguf",
        "codec_url": f"https://huggingface.co/{OMNIVOICE_HF_REPO}/resolve/{OMNIVOICE_HF_REV}/omnivoice-tokenizer-BF16.gguf",
    },
    "Q4_K_M": {
        "id": "Q4_K_M",
        "label": "Q4_K_M (~660 MB)",
        "base_file": "omnivoice-base-Q4_K_M.gguf",
        "codec_file": "omnivoice-tokenizer-Q4_K_M.gguf",
        "base_url": f"https://huggingface.co/{OMNIVOICE_HF_REPO}/resolve/{OMNIVOICE_HF_REV}/omnivoice-base-Q4_K_M.gguf",
        "codec_url": f"https://huggingface.co/{OMNIVOICE_HF_REPO}/resolve/{OMNIVOICE_HF_REV}/omnivoice-tokenizer-Q4_K_M.gguf",
    },
}

OMNIVOICE_LANG_LABELS: Dict[str, str] = {
    "uk": "Ukrainian",
    "ua": "Ukrainian",
    "en": "English",
    "ru": "Russian",
    "pl": "Polish",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "auto": "None",
}

CLONE_GUIDE_SAMPLE_TEXTS: Dict[str, str] = {
    "uk": (
        "Сьогодні чудова сонячна погода, і легкий вітер тихо шелестить у гіллі. "
        "Цей запис чисто передає мій природний голос."
    ),
    "en": (
        "The morning air feels bright and crisp today, with a gentle breeze in the trees. "
        "This clip captures my natural voice."
    ),
    "ru": (
        "Сегодня ясная, тёплая погода, и лёгкий ветер тихо шелестит в листве. "
        "Эта запись чисто передаёт мой естественный голос."
    ),
    "pl": (
        "Dzisiaj jest piękna, słoneczna pogoda, a wiatr cicho szeleści wśród liści. "
        "To nagranie czysto oddaje mój naturalny głos."
    ),
    "de": (
        "Heute ist das Wetter wunderbar klar, und eine sanfte Brise raschelt in den Bäumen. "
        "Diese Aufnahme zeigt meine natürliche Stimme."
    ),
    "fr": (
        "Le matin est clair et lumineux aujourd'hui, avec une douce brise dans les arbres. "
        "Cet enregistrement capte ma voix naturelle."
    ),
    "es": (
        "Hoy la mañana está clara y fresca, con una brisa suave en las hojas. "
        "Esta grabación refleja mi voz natural."
    ),
}

_CLONE_SAMPLE_LANG_ALIASES: Dict[str, str] = {
    "ua": "uk",
    "uk": "uk",
    "en": "en",
    "ru": "ru",
    "pl": "pl",
    "de": "de",
    "fr": "fr",
    "es": "es",
}


def detect_bundled_clone_sample_lang(ref_path: Optional[str | Path]) -> Optional[str]:
    """Detect language code if ref_path is one of the bundled clone voice samples."""
    raw = str(ref_path or "").strip()
    if not raw:
        return None
    p = Path(raw)
    parts_low = [part.lower() for part in p.parts]
    in_clone_catalog = False
    for idx, part in enumerate(parts_low[:-1]):
        if part in ("clone_voices", "omnivoice", "qwen3_tts", "zonos3_mini") and (
            idx > 0 and parts_low[idx - 1] == "samples"
        ):
            in_clone_catalog = True
            if idx + 1 < len(parts_low) - 1:
                lang_dir = _CLONE_SAMPLE_LANG_ALIASES.get(parts_low[idx + 1])
                if lang_dir:
                    return lang_dir
            break
    if in_clone_catalog and "_" in p.stem:
        prefix = p.stem.split("_", 1)[0].lower()
        lang_pref = _CLONE_SAMPLE_LANG_ALIASES.get(prefix)
        if lang_pref:
            return lang_pref
    return None


def get_bundled_clone_sample_guide_text(ref_path: Optional[str | Path]) -> str:
    """Return matching guide sample text if ref_path belongs to bundled clone samples."""
    lang = detect_bundled_clone_sample_lang(ref_path)
    if not lang:
        return ""
    return CLONE_GUIDE_SAMPLE_TEXTS.get(lang, "")


def get_default_omnivoice_ref_text(ref_path: Optional[str | Path]) -> str:
    """Return sidecar .txt transcript if present, otherwise bundled guide text for known clone samples."""
    raw = str(ref_path or "").strip()
    if not raw:
        return ""
    p = Path(raw)
    sidecar = p.with_suffix(".txt")
    if sidecar.is_file():
        try:
            stxt = sidecar.read_text(encoding="utf-8", errors="ignore").strip()
            if stxt:
                return stxt
        except OSError:
            pass
    return get_bundled_clone_sample_guide_text(p)

OMNIVOICE_STEPS_CHOICES: Tuple[int, ...] = (8, 16, 24, 32)
OMNIVOICE_DEFAULT_STEPS = 16

# Exact vocabulary accepted by omnivoice.cpp src/voice-design.h
OMNIVOICE_GENDERS: Tuple[str, ...] = ("", "female", "male")
OMNIVOICE_AGES: Tuple[str, ...] = ("", "child", "teenager", "young adult", "middle-aged", "elderly")
OMNIVOICE_PITCHES: Tuple[str, ...] = (
    "",
    "very low pitch",
    "low pitch",
    "moderate pitch",
    "high pitch",
    "very high pitch",
)
OMNIVOICE_ACCENTS: Tuple[str, ...] = (
    "",
    "american accent",
    "british accent",
    "australian accent",
    "canadian accent",
    "indian accent",
    "chinese accent",
    "japanese accent",
    "korean accent",
    "portuguese accent",
    "russian accent",
)


class OmniVoiceSynthesisCancelledError(RuntimeError):
    """Raised when an in-flight OmniVoice HTTP request is cancelled by the user."""


# ----------------------------------------------------------------------
# Config helpers & Voice Design validation
# ----------------------------------------------------------------------


def normalize_omnivoice_quant(quant: Optional[str] = None) -> str:
    raw = str(quant if quant is not None else config.get("omnivoice_quant", "Q8_0") or "Q8_0").strip().upper()
    if raw in OMNIVOICE_QUANT_SPECS:
        return raw
    return "Q8_0"


def get_omnivoice_quant_spec(quant: Optional[str] = None) -> Dict[str, str]:
    return dict(OMNIVOICE_QUANT_SPECS[normalize_omnivoice_quant(quant)])


def normalize_omnivoice_steps(steps: Optional[Any] = None) -> int:
    raw = config.get("omnivoice_steps", OMNIVOICE_DEFAULT_STEPS) if steps is None else steps
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return OMNIVOICE_DEFAULT_STEPS
    if val < 4:
        return 8
    if val > 64:
        return 64
    return val


def normalize_omnivoice_mode(mode: Optional[str] = None) -> str:
    raw = str(mode if mode is not None else config.get("omnivoice_mode", "clone") or "clone").strip().lower()
    return "design" if raw == "design" else "clone"


def normalize_omnivoice_lang_label(lang: Optional[str]) -> str:
    code = str(lang or "").strip().lower()
    if code in OMNIVOICE_LANG_LABELS:
        return OMNIVOICE_LANG_LABELS[code]
    if code in ("ukrainian", "english", "russian", "polish", "german", "french", "spanish"):
        return code.capitalize()
    return "None"


def build_omnivoice_instruct(
    *,
    gender: str = "",
    age: str = "",
    pitch: str = "",
    whisper: bool = False,
    accent: str = "",
) -> str:
    """Build a valid comma-separated instruct string matching omnivoice.cpp voice-design.h."""
    parts: List[str] = []
    g = str(gender or "").strip().lower()
    if g in OMNIVOICE_GENDERS and g:
        parts.append(g)
    a = str(age or "").strip().lower()
    if a in OMNIVOICE_AGES and a:
        parts.append(a)
    p = str(pitch or "").strip().lower()
    if p in OMNIVOICE_PITCHES and p:
        parts.append(p)
    if bool(whisper):
        parts.append("whisper")
    acc = str(accent or "").strip().lower()
    if acc in OMNIVOICE_ACCENTS and acc:
        parts.append(acc)
    return ", ".join(parts)


def get_active_omnivoice_design_instruct() -> str:
    return build_omnivoice_instruct(
        gender=str(config.get("omnivoice_design_gender", "female") or ""),
        age=str(config.get("omnivoice_design_age", "young adult") or ""),
        pitch=str(config.get("omnivoice_design_pitch", "moderate pitch") or ""),
        whisper=bool(config.get("omnivoice_whisper", False)),
        accent=str(config.get("omnivoice_design_accent", "") or ""),
    )


def format_omnivoice_design_summary_localized() -> str:
    """Return localized human-readable summary of active Voice Design attributes for the UI card."""
    g = str(config.get("omnivoice_design_gender", "female") or "").strip().lower()
    a = str(config.get("omnivoice_design_age", "young adult") or "").strip().lower()
    p = str(config.get("omnivoice_design_pitch", "moderate pitch") or "").strip().lower()
    acc = str(config.get("omnivoice_design_accent", "") or "").strip().lower()
    whisper = bool(config.get("omnivoice_whisper", False))

    gender_map = {
        "female": tr("omnivoice.gender.female", "Жіночий"),
        "male": tr("omnivoice.gender.male", "Чоловічий"),
    }
    age_map = {
        "child": tr("omnivoice.age.child", "Дитина"),
        "teenager": tr("omnivoice.age.teenager", "Підліток"),
        "young adult": tr("omnivoice.age.young_adult", "Молодий"),
        "middle-aged": tr("omnivoice.age.middle_aged", "Середній"),
        "elderly": tr("omnivoice.age.elderly", "Літній"),
    }
    pitch_map = {
        "very low pitch": tr("omnivoice.pitch.very_low", "Дуже низький"),
        "low pitch": tr("omnivoice.pitch.low", "Низький"),
        "moderate pitch": tr("omnivoice.pitch.moderate", "Середній"),
        "high pitch": tr("omnivoice.pitch.high", "Високий"),
        "very high pitch": tr("omnivoice.pitch.very_high", "Дуже високий"),
    }

    parts: List[str] = []
    if g in gender_map:
        parts.append(gender_map[g])
    if a in age_map:
        parts.append(age_map[a])
    if p in pitch_map:
        parts.append(pitch_map[p])
    if acc in OMNIVOICE_ACCENTS and acc:
        parts.append(acc.replace(" accent", "").title())
    if whisper:
        parts.append("🤫")
    if not parts:
        return ", ".join((gender_map["female"], age_map["young adult"], pitch_map["moderate pitch"]))
    return ", ".join(parts)


def get_active_omnivoice_seed() -> int:
    raw = config.get("omnivoice_design_seed", 42)
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return 42
    return val if val >= 0 else 42


def get_omnivoice_cache_signature() -> str:
    """Return a deterministic signature of all OmniVoice settings affecting audio output."""
    mode = normalize_omnivoice_mode()
    steps = normalize_omnivoice_steps()
    quant = normalize_omnivoice_quant()
    denoise = 1 if bool(config.get("omnivoice_denoise", True)) else 0
    whisper = 1 if bool(config.get("omnivoice_whisper", False)) else 0
    ref_path_val = str(config.get("omnivoice_ref_path", "") or "").strip()
    has_ref = bool(ref_path_val and Path(ref_path_val).is_file())

    if mode == "clone" and has_ref:
        try:
            st = Path(ref_path_val).stat()
            ref_stat = f"{ref_path_val}|{st.st_size}|{st.st_mtime_ns}"
        except OSError:
            ref_stat = ref_path_val
        ref_txt = str(config.get("omnivoice_ref_text", "") or "").strip()
        return f"ov:clone:{quant}:s{steps}:d{denoise}:w{whisper}:{ref_stat}:{ref_txt}"

    instruct = get_active_omnivoice_design_instruct()
    seed = get_active_omnivoice_seed()
    return f"ov:design:{quant}:s{steps}:d{denoise}:seed{seed}:{instruct}"


# ----------------------------------------------------------------------
# Installation status checks
# ----------------------------------------------------------------------


def _hidden_process_kwargs() -> Dict[str, Any]:
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}


def is_omnivoice_runtime_installed() -> bool:
    return OMNIVOICE_SERVER_EXE.is_file() and OMNIVOICE_RUNTIME_MARKER.is_file()


def is_omnivoice_whisper_installed() -> bool:
    return (
        OMNIVOICE_WHISPER_EXE.is_file()
        and OMNIVOICE_WHISPER_MODEL_PATH.is_file()
        and OMNIVOICE_WHISPER_MODEL_PATH.stat().st_size > 1_000_000
        and OMNIVOICE_WHISPER_MARKER.is_file()
    )


def find_installed_omnivoice_weights(preferred_quant: Optional[str] = None) -> Optional[Tuple[Path, Path, str]]:
    """Return (base_gguf, codec_gguf, quant_id) for the installed quant pair, or None."""
    pref = normalize_omnivoice_quant(preferred_quant)
    order = [pref] + [q for q in ("Q8_0", "BF16", "Q4_K_M") if q != pref]
    for q in order:
        spec = OMNIVOICE_QUANT_SPECS[q]
        base_p = OMNIVOICE_DIR / spec["base_file"]
        codec_p = OMNIVOICE_DIR / spec["codec_file"]
        if base_p.is_file() and codec_p.is_file() and base_p.stat().st_size > 1_000_000 and codec_p.stat().st_size > 1_000_000:
            return base_p, codec_p, q
    return None


def is_omnivoice_installed() -> bool:
    return is_omnivoice_runtime_installed() and find_installed_omnivoice_weights() is not None


# ----------------------------------------------------------------------
# Runtime & Whisper downloads
# ----------------------------------------------------------------------


def _extract_binaries_from_zip(zip_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            base_name = Path(member.filename).name
            if not base_name.lower().endswith((".exe", ".dll")):
                continue
            dest = target_dir / base_name
            with zf.open(member, "r") as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)


def resolve_omnivoice_runtime_urls(
    *,
    prefer_cuda: bool,
    proxies: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Return ordered list of candidate download URLs (GitHub release + GitLab mirror)."""
    asset_name = (
        "omnivoice-runtime-win-cuda-x64.zip"
        if prefer_cuda
        else "omnivoice-runtime-win-vulkan-x64.zip"
    )
    urls: List[str] = []
    # 1) Query GitHub Releases for newest omnivoice-runtime-v* tag
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{OMNIVOICE_GITHUB_REPO}/releases",
            params={"per_page": 10},
            proxies=proxies,
            timeout=10.0,
        )
        if resp.status_code == 200:
            releases = resp.json()
            if isinstance(releases, list):
                for rel in releases:
                    tag = str(rel.get("tag_name", "") or "")
                    if not tag.startswith("omnivoice-runtime-"):
                        continue
                    for asset in rel.get("assets", []) or []:
                        if str(asset.get("name", "") or "").lower() == asset_name:
                            u = str(asset.get("browser_download_url", "") or "").strip()
                            if u and u not in urls:
                                urls.append(u)
                    if urls:
                        break
    except Exception:
        pass

    # 2) Static fallback URLs (GitHub pinned tag + GitLab Generic Package mirror)
    gh_static = f"https://github.com/{OMNIVOICE_GITHUB_REPO}/releases/download/{OMNIVOICE_RUNTIME_TAG}/{asset_name}"
    gl_static = (
        f"https://gitlab.com/api/v4/projects/{OMNIVOICE_GITLAB_PROJECT}"
        f"/packages/generic/omnivoice-runtime/1/{asset_name}"
    )
    for cand in (gh_static, gl_static):
        if cand not in urls:
            urls.append(cand)
    return urls


def resolve_whisper_cpp_win_asset_url(proxies: Optional[Dict[str, str]] = None) -> str:
    """Resolve the newest official ggml-org/whisper.cpp Windows x64 zip URL."""
    try:
        resp = requests.get(
            "https://api.github.com/repos/ggml-org/whisper.cpp/releases",
            params={"per_page": 6},
            proxies=proxies,
            timeout=10.0,
        )
        if resp.status_code == 200:
            releases = resp.json()
            if isinstance(releases, list):
                for rel in releases:
                    for asset in rel.get("assets", []) or []:
                        name = str(asset.get("name", "") or "")
                        url = str(asset.get("browser_download_url", "") or "").strip()
                        if name.lower() == "whisper-bin-x64.zip" and url:
                            return url
    except Exception:
        pass
    return FALLBACK_WHISPER_WIN_ZIP_URL


def install_omnivoice_runtime(
    engine_id: str,
    progress_cb: Callable[[int, str], None],
    download_fn: Callable[[str, Path, str, int, int, str], None],
    cancel_check: Callable[[], None],
    proxies: Optional[Dict[str, str]] = None,
) -> None:
    """Download and extract patched Windows x64 omnivoice.cpp `tts-server.exe` runtime."""
    use_cuda = has_nvidia_gpu()
    if is_omnivoice_runtime_installed():
        try:
            marker = OMNIVOICE_RUNTIME_MARKER.read_text(encoding="utf-8", errors="ignore").strip().lower()
        except Exception:
            marker = ""
        if use_cuda:
            has_cuda = all((OMNIVOICE_RUNTIME_DIR / d).is_file() for d in CUDA_REQUIRED_DLLS)
            if marker == "cuda-ok" and has_cuda:
                return
        else:
            if marker == "vulkan-ok" and (OMNIVOICE_RUNTIME_DIR / "ggml-vulkan.dll").is_file():
                return

    shutdown_omnivoice_tts()
    OMNIVOICE_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    OMNIVOICE_RUNTIME_MARKER.unlink(missing_ok=True)

    # Try preferred backend first; if NVIDIA CUDA package is unavailable, fall back to Vulkan
    attempts: List[Tuple[bool, str, str]] = []
    if use_cuda:
        attempts.append((True, "cuda-ok", tr("tts.dl.omnivoice_runtime_cuda", "Завантаження рушія OmniVoice (NVIDIA CUDA)...")))
        attempts.append((False, "vulkan-ok", tr("tts.dl.omnivoice_runtime_vulkan", "Завантаження рушія OmniVoice (Vulkan GPU)...")))
    else:
        attempts.append((False, "vulkan-ok", tr("tts.dl.omnivoice_runtime_vulkan", "Завантаження рушія OmniVoice (Vulkan GPU)...")))

    last_err: Optional[Exception] = None
    for want_cuda, marker_val, dl_label in attempts:
        zip_path = OMNIVOICE_RUNTIME_DIR / (
            "omnivoice_runtime_cuda.zip" if want_cuda else "omnivoice_runtime_vulkan.zip"
        )
        urls = resolve_omnivoice_runtime_urls(prefer_cuda=want_cuda, proxies=proxies)
        # Primary GitHub release URL already includes GitLab Generic Package mirror in TTSManager Auto-Race;
        # only fall back to additional URLs if the primary URL is not a GitHub release URL.
        primary_urls = urls[:1] if (urls and "github.com/" in urls[0] and "/releases/download/" in urls[0]) else urls
        for url in primary_urls:
            cancel_check()
            try:
                zip_path.unlink(missing_ok=True)
                download_fn(url, zip_path, engine_id, 2, 16, dl_label)
                cancel_check()
                progress_cb(19, tr("tts.dl.omnivoice_extract", "Розпакування рушія OmniVoice..."))
                # Clean stale backend DLLs so CUDA/Vulkan modules don't conflict
                for old_dll in OMNIVOICE_RUNTIME_DIR.glob("ggml-*.dll"):
                    try:
                        old_dll.unlink()
                    except OSError:
                        pass
                _extract_binaries_from_zip(zip_path, OMNIVOICE_RUNTIME_DIR)
                for cleanup_p in (
                    zip_path,
                    zip_path.with_suffix(zip_path.suffix + ".part"),
                    zip_path.with_suffix(zip_path.suffix + ".part.meta"),
                ):
                    cleanup_p.unlink(missing_ok=True)
                if not OMNIVOICE_SERVER_EXE.is_file():
                    raise RuntimeError("Downloaded OmniVoice runtime archive does not contain tts-server.exe")
                OMNIVOICE_RUNTIME_MARKER.write_text(marker_val, encoding="utf-8")
                return
            except Exception as exc:
                zip_path.unlink(missing_ok=True)
                last_err = exc
                # Propagate explicit user cancellation immediately (preserve .part for resume)
                if "cancel" in type(exc).__name__.lower():
                    raise
                continue

    raise RuntimeError(
        tr(
            "omnivoice.err.runtime_download",
            "Не вдалося завантажити бінарник рушія OmniVoice (Windows x64): {err}",
            err=last_err or "release asset not published yet",
        )
    )


def install_omnivoice_whisper(
    engine_id: str,
    progress_cb: Callable[[int, str], None],
    download_fn: Callable[[str, Path, str, int, int, str], None],
    cancel_check: Callable[[], None],
    proxies: Optional[Dict[str, str]] = None,
) -> None:
    """Download lightweight `whisper.cpp` CLI + `ggml-small-q5_1.bin` (~190 MB) for auto ref_text."""
    if is_omnivoice_whisper_installed():
        return

    OMNIVOICE_WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    OMNIVOICE_WHISPER_MARKER.unlink(missing_ok=True)

    if not OMNIVOICE_WHISPER_EXE.is_file():
        cancel_check()
        w_url = resolve_whisper_cpp_win_asset_url(proxies=proxies)
        zip_path = OMNIVOICE_WHISPER_DIR / "whisper_bin.zip"
        try:
            download_fn(
                w_url,
                zip_path,
                engine_id,
                20,
                4,
                tr("tts.dl.omnivoice_whisper_bin", "Завантаження whisper.cpp для авторозпізнавання голосу..."),
            )
            cancel_check()
            progress_cb(24, tr("tts.dl.omnivoice_whisper_extract", "Розпакування whisper.cpp..."))
            _extract_binaries_from_zip(zip_path, OMNIVOICE_WHISPER_DIR)
        finally:
            zip_path.unlink(missing_ok=True)

    if not OMNIVOICE_WHISPER_MODEL_PATH.is_file() or OMNIVOICE_WHISPER_MODEL_PATH.stat().st_size < 1_000_000:
        cancel_check()
        download_fn(
            WHISPER_MODEL_URL,
            OMNIVOICE_WHISPER_MODEL_PATH,
            engine_id,
            25,
            13,
            tr("tts.dl.omnivoice_whisper_model", "Завантаження Whisper Small (~190 МБ) для розпізнавання референсу..."),
        )

    if OMNIVOICE_WHISPER_EXE.is_file() and OMNIVOICE_WHISPER_MODEL_PATH.is_file():
        OMNIVOICE_WHISPER_MARKER.write_text("ok", encoding="utf-8")


def get_omnivoice_weight_downloads(quant: Optional[str] = None) -> List[Tuple[str, Path, str, int, int]]:
    spec = get_omnivoice_quant_spec(quant)
    return [
        (spec["base_url"], OMNIVOICE_DIR / spec["base_file"], spec["base_file"], 38, 42),
        (spec["codec_url"], OMNIVOICE_DIR / spec["codec_file"], spec["codec_file"], 80, 19),
    ]


# ----------------------------------------------------------------------
# Pause-aware reference audio preparation & Whisper ASR caching
# ----------------------------------------------------------------------


def _load_any_audio_mono(audio_path: Path) -> Tuple[np.ndarray, int]:
    try:
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=1, dtype=np.float32)
        return arr, max(1, int(sr))
    except Exception:
        from core.tts_manager import _find_ffmpeg_exe

        ffmpeg_exe = _find_ffmpeg_exe()
        if not ffmpeg_exe:
            raise
        cmd = [
            ffmpeg_exe,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-f",
            "wav",
            "-c:a",
            "pcm_f32le",
            "pipe:1",
        ]
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            **_hidden_process_kwargs(),
        )
        if res.returncode != 0 or not res.stdout:
            raise RuntimeError(f"ffmpeg failed to decode {audio_path.name}: {res.stderr.decode('utf-8', errors='replace')[:200]}")
        data, sr = sf.read(io.BytesIO(res.stdout), dtype="float32", always_2d=False)
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=1, dtype=np.float32)
        return arr, max(1, int(sr))


def _resample_clean(arr: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or len(arr) == 0:
        return arr.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    g = math.gcd(int(src_sr), int(dst_sr))
    return resample_poly(arr, int(dst_sr) // g, int(src_sr) // g).astype(np.float32, copy=False)


def prepare_omnivoice_reference_wav(
    audio_path: Path,
    *,
    max_sec: float = OMNIVOICE_MAX_REF_SEC,
    target_sr: int = OMNIVOICE_NATIVE_SR,
) -> Tuple[Path, str]:
    """
    Prepare a clean mono 24 kHz PCM16 reference WAV for OmniVoice:
    - Edge silence trimming (50 ms padding)
    - Pause-aware smart trim around `max_sec` (~10s, RMS 80 ms window) to never cut mid-word
    - 30 ms fade-out + 100 ms tail silence + gentle peak normalization
    Returns `(cached_wav_path, digest16)`.
    """
    if not audio_path.is_file():
        raise FileNotFoundError(f"Reference audio not found: {audio_path}")

    dst_sr = max(8000, int(target_sr))
    clamped_max_s = max(3.0, min(15.0, float(max_sec)))
    st = audio_path.stat()
    digest = hashlib.sha256(
        f"{audio_path.resolve()}|{st.st_mtime_ns}|{st.st_size}|smart_{clamped_max_s:.1f}s|{dst_sr}".encode("utf-8")
    ).hexdigest()[:16]

    OMNIVOICE_REF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached_wav = OMNIVOICE_REF_CACHE_DIR / f"ovref_{digest}.wav"
    if cached_wav.is_file() and cached_wav.stat().st_size > 100:
        return cached_wav, digest

    arr, src_sr = _load_any_audio_mono(audio_path)
    if len(arr) < int(src_sr * 0.5):
        raise ValueError(tr("omnivoice.err.ref_too_short", "Файл голосу занадто короткий (< 0.5 с). Оберіть запис 6–12 секунд."))

    # 1. Trim leading/trailing silence with 50 ms margin
    nz = np.where(np.abs(arr) > 0.01)[0]
    if len(nz) > 0:
        pad = int(src_sr * 0.05)
        arr = arr[max(0, int(nz[0]) - pad) : min(len(arr), int(nz[-1]) + pad)]

    # 2. Pause-aware smart trim if longer than max_sec
    if len(arr) > int(src_sr * clamped_max_s):
        win = max(1, int(src_sr * 0.08))  # 80 ms RMS window
        hop = max(1, int(src_sr * 0.01))  # 10 ms hop
        lo_i = max(int(src_sr * 2.5), int(src_sr * (clamped_max_s - 3.0)))
        hi_i = min(len(arr) - win, int(src_sr * (clamped_max_s + 1.5)))
        cut_idx = int(src_sr * clamped_max_s)
        if hi_i > lo_i:
            best_rms = float("inf")
            best_pos = cut_idx
            for pos in range(lo_i, hi_i, hop):
                seg = arr[pos : pos + win]
                rms = float(np.sqrt(np.mean(seg * seg)))
                if rms < best_rms:
                    best_rms = rms
                    best_pos = pos + win // 2
            cut_idx = best_pos
        arr = arr[:cut_idx]

    # 3. Smooth 30 ms fade-out + 100 ms clean tail silence
    fade_n = min(len(arr), int(src_sr * 0.03))
    if fade_n > 1:
        arr = arr.copy()
        arr[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
    tail_silence = np.zeros(int(src_sr * 0.10), dtype=np.float32)
    arr = np.concatenate([arr, tail_silence], axis=0)

    # 4. Safe peak normalization
    peak = float(np.max(np.abs(arr))) if len(arr) else 0.0
    if peak < 1e-4:
        raise ValueError(tr("omnivoice.err.ref_silent", "Вибраний файл голосу містить лише тишу."))
    if peak > 0.98 or peak < 0.25:
        target_peak = 0.92 if peak > 0.98 else min(0.85, peak * 2.5)
        arr = arr * (target_peak / peak)

    arr = np.clip(_resample_clean(arr, src_sr, dst_sr), -1.0, 1.0).astype(np.float32, copy=False)
    tmp_wav = cached_wav.with_suffix(".tmp.wav")
    sf.write(str(tmp_wav), arr, dst_sr, subtype="PCM_16")
    os.replace(tmp_wav, cached_wav)
    _prune_ref_cache(max_entries=20)
    return cached_wav, digest


def _prune_ref_cache(max_entries: int = 20) -> None:
    """Keep only the most recent prepared reference WAVs and their transcript JSONs."""
    try:
        wavs = sorted(
            OMNIVOICE_REF_CACHE_DIR.glob("ovref_*.wav"),
            key=lambda p: p.stat().st_mtime_ns,
            reverse=True,
        )
        for stale_wav in wavs[max(1, int(max_entries)) :]:
            stale_wav.unlink(missing_ok=True)
            stale_wav.with_suffix(".json").unlink(missing_ok=True)
    except OSError:
        pass


def _transcript_cache_path(digest: str) -> Path:
    return OMNIVOICE_REF_CACHE_DIR / f"ovref_{digest}.json"


def load_cached_ref_transcript(digest: str) -> str:
    p = _transcript_cache_path(digest)
    if not p.is_file():
        return ""
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return str(data.get("text", "") or "").strip()
    except Exception:
        return ""


def save_cached_ref_transcript(digest: str, text: str, *, source: str = "whisper") -> None:
    clean = str(text or "").strip()
    if not clean:
        return
    OMNIVOICE_REF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _transcript_cache_path(digest)
    try:
        p.write_text(
            json.dumps({"text": clean, "source": source}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


def transcribe_wav_with_whisper_cpp(
    wav_24k_path: Path,
    *,
    lang: str = "auto",
    timeout_sec: float = 90.0,
) -> str:
    """Transcribe a prepared 24 kHz WAV using local `whisper-cli.exe` (CPU, 0 MB VRAM)."""
    if not is_omnivoice_whisper_installed():
        raise RuntimeError(
            tr(
                "omnivoice.err.whisper_missing",
                "Компонент авторозпізнавання Whisper не встановлено. Впишіть текст референсу вручну у налаштуваннях OmniVoice або перевстановіть OmniVoice.",
            )
        )

    # Whisper.cpp expects 16 kHz mono PCM16 WAV
    arr, sr = sf.read(str(wav_24k_path), dtype="float32", always_2d=False)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1, dtype=np.float32)
    arr_16k = np.clip(_resample_clean(arr, int(sr), 16000), -1.0, 1.0).astype(np.float32, copy=False)

    tmp_16k = wav_24k_path.with_suffix(".w16k.wav")
    out_prefix = wav_24k_path.with_suffix(".wout")
    out_txt = Path(f"{out_prefix}.txt")
    try:
        sf.write(str(tmp_16k), arr_16k, 16000, subtype="PCM_16")
        out_txt.unlink(missing_ok=True)

        w_lang = str(lang or "auto").strip().lower()
        if w_lang == "ua":
            w_lang = "uk"
        if w_lang not in ("auto", "uk", "en", "ru", "pl", "de", "fr", "es"):
            w_lang = "auto"

        threads = str(max(2, min(6, (os.cpu_count() or 4) // 2 or 2)))
        cmd = [
            str(OMNIVOICE_WHISPER_EXE),
            "-m",
            str(OMNIVOICE_WHISPER_MODEL_PATH),
            "-f",
            str(tmp_16k),
            "-l",
            w_lang,
            "-t",
            threads,
            "-nt",
            "-np",
            "-otxt",
            "-of",
            str(out_prefix),
        ]
        res = subprocess.run(
            cmd,
            cwd=str(OMNIVOICE_WHISPER_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            **_hidden_process_kwargs(),
        )
        text = ""
        if out_txt.is_file():
            text = out_txt.read_text(encoding="utf-8", errors="replace")
        if not text.strip() and res.stdout:
            text = res.stdout.decode("utf-8", errors="replace")
        # Clean timestamps / bracketed artifacts if any leaked
        cleaned_lines = []
        for ln in text.splitlines():
            s = re.sub(r"^\s*\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]\s*", "", ln).strip()
            if s:
                cleaned_lines.append(s)
        final_text = re.sub(r"\s+", " ", " ".join(cleaned_lines)).strip()
        if res.returncode != 0 and not final_text:
            err_tail = res.stderr.decode("utf-8", errors="replace").strip()[-300:]
            raise RuntimeError(f"whisper-cli failed (rc={res.returncode}): {err_tail}")
        return final_text
    finally:
        tmp_16k.unlink(missing_ok=True)
        out_txt.unlink(missing_ok=True)


def resolve_omnivoice_ref_text(
    ref_audio_path: Path,
    *,
    lang: str = "auto",
    force_whisper: bool = False,
) -> Tuple[Path, str, str]:
    """
    Prepare smart-trimmed reference WAV and resolve its matching `ref_text`.
    Order:
    1) Explicit user text in `config["omnivoice_ref_text"]` (if it belongs to this `ref_audio_path` and not `force_whisper`)
    2) Sidecar `.txt` next to `ref_audio_path` (when file wasn't trimmed or as fallback)
    3) Disk Whisper cache by trimmed WAV SHA-256
    4) Local `whisper.cpp` transcription (saved to disk cache & config)
    Returns `(prepared_wav_path, ref_digest, ref_text)`.
    """
    prepared_wav, digest = prepare_omnivoice_reference_wav(ref_audio_path)

    if not force_whisper:
        saved_ref_for = str(config.get("omnivoice_ref_text_path", "") or "").strip()
        saved_text = str(config.get("omnivoice_ref_text", "") or "").strip()
        try:
            same_file = bool(saved_ref_for) and Path(saved_ref_for).resolve() == ref_audio_path.resolve()
        except OSError:
            same_file = bool(saved_ref_for) and saved_ref_for.lower() == str(ref_audio_path).lower()
        if saved_text and (same_file or not saved_ref_for):
            return prepared_wav, digest, saved_text

        cached_txt = load_cached_ref_transcript(digest)
        if cached_txt:
            return prepared_wav, digest, cached_txt

        sidecar_txt = ref_audio_path.with_suffix(".txt")
        if sidecar_txt.is_file():
            try:
                stxt = sidecar_txt.read_text(encoding="utf-8", errors="ignore").strip()
                if stxt:
                    # Only trust sidecar blindly when audio wasn't truncated substantially
                    orig_info = sf.info(str(ref_audio_path))
                    if orig_info.duration <= OMNIVOICE_MAX_REF_SEC + 2.0:
                        save_cached_ref_transcript(digest, stxt, source="sidecar")
                        return prepared_wav, digest, stxt
            except Exception:
                pass

    # Run local whisper.cpp on the exact prepared (smart-trimmed) WAV
    transcribed = transcribe_wav_with_whisper_cpp(prepared_wav, lang=lang)
    if not transcribed:
        raise RuntimeError(
            tr(
                "omnivoice.err.ref_text_empty",
                "Не вдалося розпізнати текст із файлу голосу. Введіть текст референсу вручну у налаштуваннях OmniVoice.",
            )
        )
    save_cached_ref_transcript(digest, transcribed, source="whisper")
    try:
        cur_cfg_ref = str(config.get("omnivoice_ref_path", "") or "").strip()
        if cur_cfg_ref and Path(cur_cfg_ref).resolve() == ref_audio_path.resolve():
            config.set("omnivoice_ref_text", transcribed)
            config.set("omnivoice_ref_text_path", str(ref_audio_path))
    except Exception:
        pass
    return prepared_wav, digest, transcribed


# ----------------------------------------------------------------------
# Resident HTTP server manager for `omnivoice.cpp` `tts-server.exe`
# ----------------------------------------------------------------------


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _force_close_socket(sock: Any) -> None:
    if sock is None:
        return
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except Exception:
        pass
    try:
        sock.close()
    except Exception:
        pass


class _TrackableHTTPConnection(HTTPConnection):
    """HTTPConnection subclass that exposes its raw socket for instant cancellation."""

    _manager: Optional["_OmniVoiceServerManager"] = None

    def connect(self) -> None:
        super().connect()
        mgr = self._manager
        if mgr is not None and self.sock is not None:
            mgr._register_thread_socket(self.sock)


class _TrackableHTTPAdapter(HTTPAdapter):
    def __init__(self, manager: "_OmniVoiceServerManager", *args: Any, **kwargs: Any) -> None:
        self._manager = manager
        super().__init__(*args, **kwargs)

    def init_poolmanager(self, *args: Any, **kwargs: Any) -> None:
        super().init_poolmanager(*args, **kwargs)
        conn_cls = type("_OVTrackableConn", (_TrackableHTTPConnection,), {"_manager": self._manager})
        self.poolmanager.pool_classes_by_scheme = dict(self.poolmanager.pool_classes_by_scheme)
        http_pool = self.poolmanager.pool_classes_by_scheme["http"]
        self.poolmanager.pool_classes_by_scheme["http"] = type(
            "_OVTrackablePool",
            (http_pool,),
            {"ConnectionCls": conn_cls},
        )


def _read_wav_bytes(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1, dtype=np.float32)
    return arr, int(sr)


def _s16le_pcm_to_float32(raw_bytes: bytes) -> np.ndarray:
    usable = (len(raw_bytes) // 2) * 2
    if usable <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(memoryview(raw_bytes)[:usable], dtype="<i2").astype(np.float32) / 32768.0


class _OmniVoiceServerManager:
    """Manages a single resident `tts-server.exe` process with cached voice prompts & instant cancel."""

    IDLE_TIMEOUT_SEC = 300.0
    READY_TIMEOUT_SEC = 120.0

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._req_lock = threading.Lock()
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._port: int = 0
        self._api_key: str = ""
        self._supports_steps: bool = False
        self._supports_api_key: bool = False
        self._loaded_quant: str = ""
        self._force_cpu_session: bool = False
        self._registered_voices: Dict[Tuple[str, str], str] = {}
        self._log_tail: deque[str] = deque(maxlen=80)
        self._idle_timer: Optional[threading.Timer] = None

        self._cancel_gen: int = 0
        self._active_user_requests: int = 0
        self._req_local = threading.local()
        self._active_sockets: Dict[int, Tuple[Any, bool]] = {}
        self._active_responses: Dict[int, Tuple[Any, bool]] = {}

        self._session = requests.Session()
        self._session.trust_env = False
        adapter = _TrackableHTTPAdapter(self, pool_connections=4, pool_maxsize=4)
        self._session.mount("http://", adapter)

    def _register_thread_socket(self, sock: Any) -> None:
        if not getattr(self._req_local, "active", False):
            return
        tid = threading.get_ident()
        is_w = bool(getattr(self._req_local, "is_warmup", False))
        with self._req_lock:
            self._active_sockets[tid] = (sock, is_w)

    def _clear_thread_tracking(self, tid: int) -> None:
        with self._req_lock:
            self._active_sockets.pop(tid, None)
            self._active_responses.pop(tid, None)

    def _is_cancelled(self, my_gen: int) -> bool:
        with self._req_lock:
            return self._cancel_gen != my_gen

    def cancel_active(self, *, only_warmup: bool = False) -> None:
        socks_to_close: List[Any] = []
        resps_to_close: List[Any] = []
        with self._req_lock:
            if not only_warmup:
                self._cancel_gen += 1
            for tid, (sock, is_w) in list(self._active_sockets.items()):
                if not only_warmup or is_w:
                    socks_to_close.append(sock)
                    self._active_sockets.pop(tid, None)
            for tid, (resp, is_w) in list(self._active_responses.items()):
                if not only_warmup or is_w:
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

    def _probe_binary_capabilities(self) -> Tuple[bool, bool]:
        try:
            r = subprocess.run(
                [str(OMNIVOICE_SERVER_EXE), "--help"],
                cwd=str(OMNIVOICE_RUNTIME_DIR),
                capture_output=True,
                text=True,
                timeout=10.0,
                **_hidden_process_kwargs(),
            )
            txt = (r.stdout or "") + "\n" + (r.stderr or "")
            return ("supports 'steps'" in txt), ("--api-key" in txt)
        except Exception:
            return False, False

    def _auth_headers(self) -> Dict[str, str]:
        if self._api_key and self._supports_api_key:
            return {"X-Api-Key": self._api_key}
        return {}

    def _drain_output(self, proc: subprocess.Popen[bytes]) -> None:
        pipe = proc.stdout
        if pipe is None:
            return
        for raw in iter(pipe.readline, b""):
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                self._log_tail.append(line)

    def _cancel_idle_timer(self) -> None:
        with self._lock:
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None

    def _reset_idle_timer(self) -> None:
        with self._lock:
            if self._idle_timer is not None:
                self._idle_timer.cancel()
            timer = threading.Timer(self.IDLE_TIMEOUT_SEC, self._on_idle_timeout)
            timer.daemon = True
            self._idle_timer = timer
            timer.start()

    def _on_idle_timeout(self) -> None:
        with self._req_lock:
            if self._active_user_requests > 0:
                return
        self.stop()

    def stop(self) -> None:
        self.cancel_active(only_warmup=False)
        with self._lock:
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None
            proc = self._proc
            self._proc = None
            self._port = 0
            self._api_key = ""
            self._loaded_quant = ""
            self._registered_voices.clear()
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=5.0)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=3.0)
                except Exception:
                    pass

    def _spawn_and_wait(self, base_gguf: Path, codec_gguf: Path, *, force_cpu: bool) -> int:
        supports_steps, supports_api_key = self._probe_binary_capabilities()
        port = _pick_free_port()
        api_key = secrets.token_urlsafe(24) if supports_api_key else ""
        cmd = [
            str(OMNIVOICE_SERVER_EXE),
            "--model",
            str(base_gguf),
            "--codec",
            str(codec_gguf),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        if supports_api_key and api_key:
            cmd += ["--api-key", api_key]

        env = dict(os.environ)
        if force_cpu:
            env["GGML_BACKEND"] = "CPU"
        else:
            env.pop("GGML_BACKEND", None)

        self._log_tail.clear()
        proc = subprocess.Popen(
            cmd,
            cwd=str(OMNIVOICE_RUNTIME_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            **_hidden_process_kwargs(),
        )
        assign_process_to_kill_on_close_job(proc)
        threading.Thread(target=self._drain_output, args=(proc,), daemon=True).start()

        headers = {"X-Api-Key": api_key} if (supports_api_key and api_key) else {}
        deadline = time.monotonic() + self.READY_TIMEOUT_SEC
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                tail = " | ".join(list(self._log_tail)[-12:])
                raise RuntimeError(f"OmniVoice tts-server exited with code {proc.returncode}: {tail[-350:]}")
            try:
                r = self._session.get(f"http://127.0.0.1:{port}/health", timeout=0.8)
                if r.status_code == 200:
                    r_m = self._session.get(f"http://127.0.0.1:{port}/v1/models", headers=headers, timeout=1.0)
                    if r_m.status_code == 200:
                        self._proc = proc
                        self._port = port
                        self._api_key = api_key
                        self._supports_steps = supports_steps
                        self._supports_api_key = supports_api_key
                        self._registered_voices.clear()
                        return port
            except Exception:
                pass
            time.sleep(0.15)

        try:
            proc.terminate()
            proc.wait(timeout=3.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        tail = " | ".join(list(self._log_tail)[-10:])
        raise TimeoutError(f"OmniVoice tts-server startup timeout: {tail[-300:]}")

    def ensure_started(self, preferred_quant: Optional[str] = None) -> int:
        with self._lock:
            weights = find_installed_omnivoice_weights(preferred_quant)
            if not is_omnivoice_runtime_installed() or weights is None:
                raise RuntimeError(
                    tr(
                        "tts.err.model_not_installed",
                        "Модель {name} не встановлена. Натисніть «Встановити» у списку локальних TTS.",
                        name="OmniVoice",
                    )
                )
            base_gguf, codec_gguf, quant_id = weights
            if self._proc is not None and self._proc.poll() is None and self._loaded_quant == quant_id:
                self._cancel_idle_timer()
                return self._port

            self.stop()
            try:
                port = self._spawn_and_wait(base_gguf, codec_gguf, force_cpu=self._force_cpu_session)
                self._loaded_quant = quant_id
                return port
            except Exception as first_exc:
                if self._force_cpu_session:
                    raise
                print(
                    f"[OmniVoice] GPU backend failed ({first_exc}); retrying on CPU fallback...",
                    file=sys.stderr,
                    flush=True,
                )
                self._force_cpu_session = True
                port = self._spawn_and_wait(base_gguf, codec_gguf, force_cpu=True)
                self._loaded_quant = quant_id
                return port

    def ensure_voice_registered(self, port: int, wav_path: Path, ref_digest: str, ref_text: str) -> str:
        clean_text = str(ref_text or "").strip()
        if not clean_text:
            raise ValueError(
                tr(
                    "omnivoice.err.ref_text_empty",
                    "Не вдалося розпізнати текст із файлу голосу. Введіть текст референсу вручну у налаштуваннях OmniVoice.",
                )
            )
        key = (ref_digest, clean_text)
        with self._lock:
            existing = self._registered_voices.get(key)
            if existing:
                return existing

        text_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()[:8]
        voice_name = f"clone_{ref_digest[:8]}_{text_hash}"
        wav_b64 = base64.b64encode(wav_path.read_bytes()).decode("ascii")
        resp = self._session.post(
            f"http://127.0.0.1:{port}/v1/audio/voices",
            json={"name": voice_name, "ref_text": clean_text, "wav_b64": wav_b64},
            headers=self._auth_headers(),
            timeout=120.0,
        )
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"OmniVoice voice register failed (HTTP {resp.status_code}): {resp.text[:300]}")
        with self._lock:
            self._registered_voices[key] = voice_name
        return voice_name

    def _build_speech_payload(
        self,
        *,
        text: str,
        lang_label: str,
        voice_name: str,
        instruct: str,
        seed: Optional[int],
        steps: int,
        denoise: bool,
        response_format: str,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "input": str(text or "").rstrip("\r\n"),
            "response_format": response_format,
        }
        if voice_name:
            payload["voice"] = voice_name
        if lang_label and lang_label != "None":
            payload["language"] = lang_label
        if instruct:
            payload["instructions"] = instruct
        if seed is not None and seed >= 0:
            payload["seed"] = int(seed)
        if self._supports_steps:
            payload["steps"] = int(steps)
            payload["denoise"] = bool(denoise)
        return payload

    def synthesize(
        self,
        *,
        text: str,
        lang_label: str,
        ref_wav: Optional[Path] = None,
        ref_digest: str = "",
        ref_text: str = "",
        instruct: str = "",
        seed: Optional[int] = None,
        steps: int = OMNIVOICE_DEFAULT_STEPS,
        denoise: bool = True,
        is_warmup: bool = False,
    ) -> Tuple[np.ndarray, int]:
        if not is_warmup:
            self.cancel_active(only_warmup=True)
        with self._req_lock:
            my_gen = self._cancel_gen
            if not is_warmup:
                self._active_user_requests += 1
        tid = threading.get_ident()
        resp = None
        try:
            port = self.ensure_started()
            if self._is_cancelled(my_gen):
                raise OmniVoiceSynthesisCancelledError("OmniVoice synthesis cancelled")

            voice_name = ""
            if ref_wav is not None and ref_digest:
                voice_name = self.ensure_voice_registered(port, ref_wav, ref_digest, ref_text)
                if self._is_cancelled(my_gen):
                    raise OmniVoiceSynthesisCancelledError("OmniVoice synthesis cancelled")

            payload = self._build_speech_payload(
                text=text,
                lang_label=lang_label,
                voice_name=voice_name,
                instruct=instruct,
                seed=seed,
                steps=steps,
                denoise=denoise,
                response_format="wav",
            )
            self._req_local.active = True
            self._req_local.is_warmup = is_warmup
            try:
                resp = self._session.post(
                    f"http://127.0.0.1:{port}/v1/audio/speech",
                    json=payload,
                    headers=self._auth_headers(),
                    timeout=180.0,
                    stream=True,
                )
            except Exception as exc:
                if self._is_cancelled(my_gen):
                    raise OmniVoiceSynthesisCancelledError("OmniVoice synthesis cancelled") from exc
                raise

            with self._req_lock:
                self._active_responses[tid] = (resp, is_warmup)
            if self._is_cancelled(my_gen):
                raise OmniVoiceSynthesisCancelledError("OmniVoice synthesis cancelled")
            if resp.status_code != 200:
                raise RuntimeError(f"OmniVoice tts-server error ({resp.status_code}): {resp.text[:300]}")

            chunks: List[bytes] = []
            try:
                if hasattr(resp, "iter_content"):
                    for part in resp.iter_content(chunk_size=32768):
                        if self._is_cancelled(my_gen):
                            raise OmniVoiceSynthesisCancelledError("OmniVoice synthesis cancelled")
                        if part:
                            chunks.append(part)
                    raw_bytes = b"".join(chunks)
                else:
                    raw_bytes = resp.content
            except OmniVoiceSynthesisCancelledError:
                raise
            except Exception as exc:
                if self._is_cancelled(my_gen):
                    raise OmniVoiceSynthesisCancelledError("OmniVoice synthesis cancelled") from exc
                raise

            if not raw_bytes.startswith(b"RIFF"):
                raise RuntimeError(f"OmniVoice tts-server error ({resp.status_code}): {raw_bytes[:300]!r}")
            return _read_wav_bytes(raw_bytes)
        finally:
            self._req_local.active = False
            self._req_local.is_warmup = False
            self._clear_thread_tracking(tid)
            if not is_warmup:
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
        *,
        text: str,
        lang_label: str,
        ref_wav: Optional[Path] = None,
        ref_digest: str = "",
        ref_text: str = "",
        instruct: str = "",
        seed: Optional[int] = None,
        steps: int = OMNIVOICE_DEFAULT_STEPS,
        denoise: bool = True,
    ) -> Iterator[Tuple[np.ndarray, int]]:
        """Stream 24 kHz mono PCM chunks from `tts-server` as soon as chunks decode."""
        self.cancel_active(only_warmup=True)
        with self._req_lock:
            my_gen = self._cancel_gen
            self._active_user_requests += 1
        tid = threading.get_ident()
        resp = None
        try:
            port = self.ensure_started()
            if self._is_cancelled(my_gen):
                raise OmniVoiceSynthesisCancelledError("OmniVoice synthesis cancelled")

            voice_name = ""
            if ref_wav is not None and ref_digest:
                voice_name = self.ensure_voice_registered(port, ref_wav, ref_digest, ref_text)
                if self._is_cancelled(my_gen):
                    raise OmniVoiceSynthesisCancelledError("OmniVoice synthesis cancelled")

            payload = self._build_speech_payload(
                text=text,
                lang_label=lang_label,
                voice_name=voice_name,
                instruct=instruct,
                seed=seed,
                steps=steps,
                denoise=denoise,
                response_format="pcm",
            )
            self._req_local.active = True
            self._req_local.is_warmup = False
            try:
                resp = self._session.post(
                    f"http://127.0.0.1:{port}/v1/audio/speech",
                    json=payload,
                    headers=self._auth_headers(),
                    timeout=180.0,
                    stream=True,
                )
            except Exception as exc:
                if self._is_cancelled(my_gen):
                    raise OmniVoiceSynthesisCancelledError("OmniVoice synthesis cancelled") from exc
                raise

            with self._req_lock:
                self._active_responses[tid] = (resp, False)
            if self._is_cancelled(my_gen):
                raise OmniVoiceSynthesisCancelledError("OmniVoice synthesis cancelled")
            if resp.status_code != 200:
                raise RuntimeError(f"OmniVoice tts-server error ({resp.status_code}): {resp.text[:300]}")

            ct = str(resp.headers.get("Content-Type", "") or "").lower()
            if "audio/wav" in ct or not hasattr(resp, "iter_content"):
                samples, sr = _read_wav_bytes(resp.content)
                if len(samples) > 0:
                    yield samples, sr
                return

            carry = bytearray()
            try:
                for raw_part in resp.iter_content(chunk_size=16384):
                    if self._is_cancelled(my_gen):
                        raise OmniVoiceSynthesisCancelledError("OmniVoice synthesis cancelled")
                    if not raw_part:
                        continue
                    carry.extend(raw_part)
                    usable = (len(carry) // 2) * 2
                    if usable <= 0:
                        continue
                    chunk_bytes = bytes(carry[:usable])
                    del carry[:usable]
                    arr = _s16le_pcm_to_float32(chunk_bytes)
                    if len(arr) > 0:
                        yield arr, OMNIVOICE_NATIVE_SR
            except OmniVoiceSynthesisCancelledError:
                raise
            except Exception as exc:
                if self._is_cancelled(my_gen):
                    raise OmniVoiceSynthesisCancelledError("OmniVoice synthesis cancelled") from exc
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


_omnivoice_server = _OmniVoiceServerManager()
atexit.register(_omnivoice_server.stop)


def is_omnivoice_server_running() -> bool:
    return _omnivoice_server.is_running()


def cancel_omnivoice_synthesis() -> None:
    _omnivoice_server.cancel_active(only_warmup=False)


def shutdown_omnivoice_tts() -> None:
    _omnivoice_server.stop()


def _resolve_active_synth_params(lang: str) -> Dict[str, Any]:
    """Resolve active OmniVoice mode (Clone vs Design), reference, instruct, seed, steps & denoise."""
    mode = normalize_omnivoice_mode()
    steps = normalize_omnivoice_steps()
    denoise = bool(config.get("omnivoice_denoise", True))
    whisper = bool(config.get("omnivoice_whisper", False))
    lang_label = normalize_omnivoice_lang_label(lang)

    ref_raw = str(config.get("omnivoice_ref_path", "") or "").strip()
    if mode == "clone" and ref_raw and Path(ref_raw).is_file():
        ref_wav, ref_digest, ref_text = resolve_omnivoice_ref_text(Path(ref_raw), lang="auto")
        # Whisper style can optionally be combined on top of a cloned voice
        instruct = "whisper" if whisper else ""
        return {
            "lang_label": lang_label,
            "ref_wav": ref_wav,
            "ref_digest": ref_digest,
            "ref_text": ref_text,
            "instruct": instruct,
            "seed": None,
            "steps": steps,
            "denoise": denoise,
        }

    # Voice Design mode (or Clone mode when no reference clip has been chosen yet)
    instruct = get_active_omnivoice_design_instruct() or "female, young adult, moderate pitch"
    seed = get_active_omnivoice_seed()
    return {
        "lang_label": lang_label,
        "ref_wav": None,
        "ref_digest": "",
        "ref_text": "",
        "instruct": instruct,
        "seed": seed,
        "steps": steps,
        "denoise": denoise,
    }


def warmup_omnivoice_tts(lang: str = "") -> None:
    if not is_omnivoice_installed():
        return
    eff_lang = str(lang or config.get("app_language", "uk") or "uk").strip().lower() or "uk"
    try:
        params = _resolve_active_synth_params(eff_lang)
        probe = "Привіт." if eff_lang in ("uk", "ua") else ("Привет." if eff_lang == "ru" else "Hello.")
        _omnivoice_server.synthesize(
            text=probe,
            lang_label=params["lang_label"],
            ref_wav=params["ref_wav"],
            ref_digest=params["ref_digest"],
            ref_text=params["ref_text"],
            instruct=params["instruct"],
            seed=params["seed"],
            steps=params["steps"],
            denoise=params["denoise"],
            is_warmup=True,
        )
    except Exception:
        pass


def synthesize_omnivoice_tts(
    text: str,
    lang: str = "uk",
    speed: float = 1.0,
) -> Tuple[np.ndarray, int]:
    if not is_omnivoice_installed():
        raise RuntimeError(
            tr(
                "tts.err.model_not_installed",
                "Модель {name} не встановлена. Натисніть «Встановити» у списку локальних TTS.",
                name="OmniVoice",
            )
        )
    params = _resolve_active_synth_params(lang)
    samples, sr = _omnivoice_server.synthesize(
        text=str(text or "").strip(),
        lang_label=params["lang_label"],
        ref_wav=params["ref_wav"],
        ref_digest=params["ref_digest"],
        ref_text=params["ref_text"],
        instruct=params["instruct"],
        seed=params["seed"],
        steps=params["steps"],
        denoise=params["denoise"],
    )
    if abs(speed - 1.0) > 0.04 and len(samples) > 100:
        from scipy.signal import resample

        new_len = max(1, int(len(samples) / float(speed)))
        samples = resample(samples, new_len).astype(np.float32)
    return samples, sr


def synthesize_omnivoice_tts_stream(
    text: str,
    lang: str = "uk",
    speed: float = 1.0,
) -> Iterator[Tuple[np.ndarray, int]]:
    if not is_omnivoice_installed():
        raise RuntimeError(
            tr(
                "tts.err.model_not_installed",
                "Модель {name} не встановлена. Натисніть «Встановити» у списку локальних TTS.",
                name="OmniVoice",
            )
        )
    if abs(speed - 1.0) > 0.04:
        samples, sr = synthesize_omnivoice_tts(text, lang=lang, speed=speed)
        if len(samples) > 0:
            yield samples, sr
        return
    params = _resolve_active_synth_params(lang)
    yield from _omnivoice_server.synthesize_stream(
        text=str(text or "").strip(),
        lang_label=params["lang_label"],
        ref_wav=params["ref_wav"],
        ref_digest=params["ref_digest"],
        ref_text=params["ref_text"],
        instruct=params["instruct"],
        seed=params["seed"],
        steps=params["steps"],
        denoise=params["denoise"],
    )
