"""
core/tts_manager.py — Асинхронний менеджер синтезу мовлення (TTS) та керування моделями:
- Локальні движки: Piper, Silero, HolosTTS, TeraTTS
- Онлайн-голоси: Microsoft Edge-TTS (за замовчуванням, можна вимкнути) + додавання Google Cloud TTS API, ElevenLabs API, OpenAI TTS API
- Прослуховування доступних .wav-прикладів для кожного движка
- Встановлення та видалення локальних моделей в 1 клік
"""

from __future__ import annotations

import asyncio
import base64
from collections import deque
import concurrent.futures
import gc
import gzip
import importlib
import io
import json
import os
import queue
import random
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import types
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
import soundfile as sf
from PySide6.QtCore import QObject, Signal

from core.ai_corrector import correct_text_with_ai
from core.audio_router import audio_router
from core.config import BASE_DIR, BUNDLE_DIR, DATA_DIR, ENGINES_DIR, SAMPLES_DIR, TTS_CACHE_DIR, config
from core.i18n import get_ui_language, tr
from core.obs_server import obs_server
from core.proxy_helper import (
    get_active_api_proxies,
    get_active_socks5_url,
    get_edge_tts_connector,
    get_edge_tts_proxy_config,
)
from core.text_normalizer import (
    adapt_text_for_local_tts,
    apply_default_stress_fixes,
    apply_piper_uk_stress_phonemes,
    format_stress_marks,
    normalize_tts_text,
    strip_plus_stress_marks,
)
from core.tts_cache import TTSAudioCache

# ──────────────────────────────────────────────────────────────────────
# 0. Індивідуальні налаштування покращення звучання (DSP) для кожного TTS
# ──────────────────────────────────────────────────────────────────────

DEFAULT_DSP_SETTINGS: Dict[str, Any] = {
    "volume": 100,
    "enabled": False,
    "speed_enabled": False,
    "speed_rate": 1.00,
    "pitch_enabled": False,
    "pitch_steps": 0.0,
    "denoise_enabled": False,
    "denoise_prop": 0.30,
    "deess_enabled": False,
    "deess_db": -6.0,
    "gate_enabled": False,
    "gate_db": -45.0,
}

CUSTOM_EDGE_UK_DSP: Dict[str, Any] = {
    "volume": 180,
    "enabled": True,
    "speed_enabled": True,
    "speed_rate": 1.07,
    "pitch_enabled": True,
    "pitch_steps": 0.9,
    "denoise_enabled": True,
    "denoise_prop": 0.50,
    "deess_enabled": True,
    "deess_db": -8.0,
    "gate_enabled": True,
    "gate_db": -45.0,
}


_GLOBAL_IMPORT_LOCK = threading.RLock()


def _get_torch():
    """Лінивий потокобезпечний імпорт PyTorch (заощаджує ~7.5 с при старті програми)."""
    if "torch" not in sys.modules:
        with _GLOBAL_IMPORT_LOCK:
            import torch
            return torch
    import torch
    return torch


def _resolve_dsp_key_and_defaults(engine_id: str, voice_id: str = "") -> Tuple[str, Dict[str, Any]]:
    """Визначає ключ збереження DSP та дефолтний пресет (для Chirp та кастомного Edge-голосу)."""
    v_str = str(voice_id or "").strip()
    if v_str.endswith("::custom"):
        defaults = dict(CUSTOM_EDGE_UK_DSP)
        return f"{engine_id}::voice::{v_str}", defaults
    if "chirp" in v_str.lower():
        defaults = dict(DEFAULT_DSP_SETTINGS)
        defaults["enabled"] = True
        defaults["gate_enabled"] = True
        defaults["gate_db"] = -45.0
        return f"{engine_id}::chirp", defaults
    return engine_id, dict(DEFAULT_DSP_SETTINGS)


def get_engine_dsp_settings(engine_id: str, voice_id: str = "") -> Dict[str, Any]:
    """Повертає словник налаштувань DSP для вказаного TTS-движка (та/або конкретного голосу)."""
    key, merged = _resolve_dsp_key_and_defaults(engine_id, voice_id)
    if not engine_id:
        return merged
    all_dsp = config.get("engine_dsp_settings", {})
    if isinstance(all_dsp, dict) and isinstance(all_dsp.get(key), dict):
        merged.update(all_dsp[key])
    return merged


def save_engine_dsp_settings(engine_id: str, settings: Dict[str, Any], voice_id: str = "") -> None:
    """Зберігає словник налаштувань DSP для вказаного TTS-движка (та/або конкретного голосу)."""
    if not engine_id:
        return
    key, merged = _resolve_dsp_key_and_defaults(engine_id, voice_id)
    all_dsp = dict(config.get("engine_dsp_settings", {}) or {})
    if isinstance(all_dsp.get(key), dict):
        merged.update(all_dsp[key])
    merged.update(settings)
    all_dsp[key] = merged
    config.set("engine_dsp_settings", all_dsp)


def _fx_highpass(x: np.ndarray, sr: int, cutoff: float = 80.0, order: int = 2) -> np.ndarray:
    """Зрізає низькочастотний гул нижче 80 Гц."""
    if len(x) < 32:
        return x
    nyquist = sr / 2.0
    if cutoff <= 0 or cutoff >= nyquist:
        return x
    try:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(max(1, int(order)), cutoff, btype="highpass", fs=sr, output="sos")
        return sosfiltfilt(sos, x).astype(np.float32)
    except Exception:
        return x


def _fx_deesser(
    x: np.ndarray,
    sr: int,
    low: float = 5000.0,
    high: float = 8000.0,
    threshold_db: float = -25.0,
    reduction_db: float = -6.0,
) -> np.ndarray:
    """Пом'якшує різкі шиплячі та свистячі звуки («с», «ш», «щ», «ц»)."""
    if len(x) < 32 or reduction_db >= 0.0:
        return x
    nyquist = sr / 2.0
    if low >= nyquist:
        return x
    high_clamped = min(high, nyquist - 100.0)
    if high_clamped <= low:
        return x
    try:
        from scipy.ndimage import uniform_filter1d
        from scipy.signal import butter, sosfiltfilt
        sos = butter(4, [low, high_clamped], btype="bandpass", fs=sr, output="sos")
        band = sosfiltfilt(sos, x)
        rest = x - band
        env = np.abs(band)
        thresh = 10.0 ** (threshold_db / 20.0)
        reduction = 10.0 ** (reduction_db / 20.0)
        gain = np.where(env > thresh, reduction, 1.0)
        smooth_win = max(3, int(sr * 0.005))
        gain = uniform_filter1d(gain.astype(np.float32), size=smooth_win, mode="nearest")
        return (rest + band * gain).astype(np.float32)
    except Exception:
        return x


def _fx_denoise(x: np.ndarray, sr: int, prop_decrease: float = 0.3) -> np.ndarray:
    """Стаціонарне шумозаглушення для усунення металевого призвуку та фонового шипіння."""
    if len(x) < 64 or prop_decrease <= 0.01:
        return x
    try:
        import noisereduce as nr
        return nr.reduce_noise(
            y=x, sr=sr, prop_decrease=float(min(1.0, max(0.0, prop_decrease))), stationary=True
        ).astype(np.float32)
    except Exception:
        return x


def _fx_noise_gate(
    x: np.ndarray,
    sr: int,
    threshold_db: float = -45.0,
    attack_ms: float = 2.0,
    hold_ms: float = 25.0,
    release_ms: float = 60.0,
    floor_db: float = -80.0,
) -> np.ndarray:
    """Студійний нойз-гейт: плавно прибирає фоновий шум і гул у паузах між словами."""
    if len(x) == 0:
        return x
    from scipy.ndimage import maximum_filter1d, uniform_filter1d

    thresh_lin = float(10.0 ** (threshold_db / 20.0))
    floor_lin = float(10.0 ** (floor_db / 20.0))
    win_len = max(1, int(sr * 0.004))
    env = np.sqrt(np.maximum(0.0, uniform_filter1d(x * x, size=win_len, mode="constant")) + 1e-12).astype(np.float32)

    attack_coef = float(np.exp(-1.0 / (max(1e-4, attack_ms * 0.001) * sr)))
    release_coef = float(np.exp(-1.0 / (max(1e-4, release_ms * 0.001) * sr)))
    hold_samples = max(0, int(hold_ms * 0.001 * sr))

    open_raw = env >= thresh_lin
    if hold_samples > 0:
        w = hold_samples + 1
        is_open = maximum_filter1d(
            open_raw.astype(np.uint8),
            size=w,
            origin=-((w - 1) // 2),
            mode="constant",
            cval=0,
        ).astype(bool)
    else:
        is_open = open_raw

    n = len(env)
    changes = np.flatnonzero(is_open[1:] != is_open[:-1]) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [n]))

    gains = np.empty(n, dtype=np.float32)
    g_prev = floor_lin
    for s, e in zip(starts, ends):
        length = int(e - s)
        if is_open[s]:
            target, coef = 1.0, attack_coef
        else:
            target, coef = floor_lin, release_coef
        exps = np.power(coef, np.arange(1, length + 1, dtype=np.float64))
        seg = target + (g_prev - target) * exps
        gains[s:e] = seg.astype(np.float32)
        g_prev = float(seg[-1])
    return (x * gains).astype(np.float32)


def _time_stretch_ola(x: np.ndarray, sr: int, stretch_rate: float) -> np.ndarray:
    """Змінює темп мовлення (stretch_rate > 1 — швидше, < 1 — повільніше) без зміни висоти голосу (OLA)."""
    if abs(stretch_rate - 1.0) < 0.01 or len(x) < 512:
        return x
    win_size = max(256, int(sr * 0.025))
    if win_size % 2 != 0:
        win_size += 1
    hop_out = win_size // 2
    hop_in = max(1, int(round(hop_out * stretch_rate)))
    window = np.hanning(win_size).astype(np.float32)

    num_frames = max(1, (len(x) - win_size) // hop_in + 1)
    out_len = (num_frames - 1) * hop_out + win_size
    out = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)

    for i in range(num_frames):
        in_start = i * hop_in
        out_start = i * hop_out
        frame = x[in_start : in_start + win_size]
        if len(frame) < win_size:
            frame = np.pad(frame, (0, win_size - len(frame)))
        out[out_start : out_start + win_size] += frame * window
        norm[out_start : out_start + win_size] += window ** 2

    mask = norm > 1e-6
    out[mask] /= norm[mask]
    return out


FFMPEG_ENGINE_ID = "__ffmpeg__"
FFMPEG_DIR = ENGINES_DIR / "ffmpeg"
FFMPEG_EXE = FFMPEG_DIR / "ffmpeg.exe"
FFMPEG_DOWNLOAD_URLS = (
    "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
)


def _find_ffmpeg_exe() -> Optional[str]:
    from core.config import BASE_DIR, BUNDLE_DIR

    for local_cand in (FFMPEG_EXE, BASE_DIR / "ffmpeg.exe", BUNDLE_DIR / "ffmpeg.exe"):
        if local_cand.exists():
            return str(local_cand)
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    for cand in (r"C:\ffmpeg\ffmpeg.exe", r"C:\ffmpeg\bin\ffmpeg.exe"):
        if Path(cand).exists():
            return cand
    try:
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def is_ffmpeg_installed() -> bool:
    """Повертає True, якщо у програмі або системі доступний ffmpeg."""
    return _find_ffmpeg_exe() is not None


def _fx_speed_and_pitch_array(
    x: np.ndarray, sr: int, speed_rate: float = 1.0, pitch_steps: float = 0.0
) -> np.ndarray:
    """Одночасна зміна швидкості мовлення та тону (висоти) голосу у півтонах через ffmpeg atempo (з OLA fallback)."""
    has_speed = abs(speed_rate - 1.0) > 0.01
    has_pitch = abs(pitch_steps) > 0.05
    if not has_speed and not has_pitch or len(x) < 64:
        return x

    pitch_ratio = float(2.0 ** (pitch_steps / 12.0)) if has_pitch else 1.0
    atempo_rate = float(speed_rate) / pitch_ratio

    ffmpeg_exe = _find_ffmpeg_exe()
    if ffmpeg_exe:
        try:
            filters: List[str] = []
            if has_pitch:
                filters.append(f"asetrate={sr * pitch_ratio:.2f}")
                filters.append(f"aresample={sr}")
            r = atempo_rate
            if abs(r - 1.0) > 1e-4:
                while r > 2.0:
                    filters.append("atempo=2.0")
                    r /= 2.0
                while r < 0.5:
                    filters.append("atempo=0.5")
                    r /= 0.5
                filters.append(f"atempo={r:.6f}")

            if filters:
                buf_in = io.BytesIO()
                sf.write(buf_in, x.astype(np.float32), int(sr), format="WAV", subtype="FLOAT")
                cmd = [
                    ffmpeg_exe, "-y", "-loglevel", "error",
                    "-f", "wav", "-i", "pipe:0",
                    "-filter:a", ",".join(filters),
                    "-f", "wav", "-acodec", "pcm_f32le", "pipe:1",
                ]
                extra_kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
                res = subprocess.run(
                    cmd,
                    input=buf_in.getvalue(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                    **extra_kwargs,
                )
                if res.returncode == 0 and res.stdout:
                    out_data, _out_sr = sf.read(io.BytesIO(res.stdout), dtype="float32")
                    if out_data.ndim > 1:
                        out_data = np.mean(out_data, axis=1).astype(np.float32)
                    return out_data.astype(np.float32)
        except Exception:
            pass

    # Fallback якщо ffmpeg недоступний
    stretched = _time_stretch_ola(x, sr, atempo_rate)
    if has_pitch and len(stretched) > 16:
        new_len = max(1, int(round(len(stretched) / pitch_ratio)))
        old_idx = np.linspace(0.0, 1.0, num=len(stretched), dtype=np.float32)
        new_idx = np.linspace(0.0, 1.0, num=new_len, dtype=np.float32)
        stretched = np.interp(new_idx, old_idx, stretched).astype(np.float32)
    return stretched


def _apply_volume_and_limiter(samples: np.ndarray, volume_scale: float) -> np.ndarray:
    """Застосовує підсилення та soft-knee лімітер для уникнення кліпінгу (як у AI-проєкті)."""
    if abs(volume_scale - 1.0) < 1e-4:
        return samples
    if volume_scale <= 0.0:
        return np.zeros_like(samples)

    scaled = samples * float(volume_scale)
    max_peak = float(np.max(np.abs(scaled))) if scaled.size > 0 else 0.0
    if max_peak > 0.80:
        threshold = 0.80
        mask = np.abs(scaled) > threshold
        if np.any(mask):
            excess = np.abs(scaled[mask]) - threshold
            compressed = threshold + 0.19 * np.tanh(excess / 0.30)
            scaled[mask] = np.sign(scaled[mask]) * compressed

    return np.clip(scaled, -1.0, 1.0)


def apply_engine_dsp(samples: np.ndarray, sr: int, engine_id: str, voice_id: str = "") -> Tuple[np.ndarray, int]:
    """Застосовує індивідуальний ланцюжок покращення звуку (DSP) та гучність для вказаного TTS-движка."""
    if samples is None or len(samples) == 0:
        return samples, sr
    dsp = get_engine_dsp_settings(engine_id, voice_id)
    vol_scale = max(0.0, min(2.0, float(dsp.get("volume", 100)) / 100.0))

    x = np.asarray(samples, dtype=np.float32)

    any_fx = (
        bool(dsp.get("denoise_enabled", False))
        or bool(dsp.get("deess_enabled", False))
        or bool(dsp.get("speed_enabled", False))
        or bool(dsp.get("pitch_enabled", False))
        or bool(dsp.get("gate_enabled", False))
    )
    if any_fx:
        if x.ndim > 1:
            x = np.mean(x, axis=1).astype(np.float32)
        if bool(dsp.get("denoise_enabled", False)):
            prop = float(dsp.get("denoise_prop", 0.30))
            x = _fx_denoise(x, sr, prop_decrease=prop)

        x = _fx_highpass(x, sr, cutoff=80.0, order=2)

        if bool(dsp.get("deess_enabled", False)):
            reduc_db = float(dsp.get("deess_db", -6.0))
            x = _fx_deesser(x, sr, reduction_db=reduc_db)

        speed_rate = float(dsp.get("speed_rate", 1.0)) if bool(dsp.get("speed_enabled", False)) else 1.0
        pitch_steps = float(dsp.get("pitch_steps", 0.0)) if bool(dsp.get("pitch_enabled", False)) else 0.0
        if abs(speed_rate - 1.0) > 0.01 or abs(pitch_steps) > 0.05:
            x = _fx_speed_and_pitch_array(x, sr, speed_rate=speed_rate, pitch_steps=pitch_steps)

        if bool(dsp.get("gate_enabled", False)):
            gate_db = float(dsp.get("gate_db", -45.0))
            x = _fx_noise_gate(x, sr, threshold_db=gate_db)

    x = _apply_volume_and_limiter(x, vol_scale)
    return np.clip(x, -1.0, 1.0).astype(np.float32), sr

# ──────────────────────────────────────────────────────────────────────
# 1. Опис локальних TTS-движків
# ──────────────────────────────────────────────────────────────────────

ENGINE_SPECS: Dict[str, Dict[str, Any]] = {
    "piper": {
        "id": "piper",
        "title": "Piper TTS",
        "size_label": "~75–185 МБ",
        "voices": [
            # Українська (uk)
            {"id": "piper_ua_tetiana", "name": "Tetiana (High)", "gender": "female", "lang": "uk", "onnx": "uk_UA-tetiana-high.onnx", "speaker": None},
            {"id": "piper_ua_lada", "name": "Lada (Medium)", "gender": "female", "lang": "uk", "onnx": "uk_UA-ukrainian_tts-medium.onnx", "speaker": 0},
            {"id": "piper_ua_mykyta", "name": "Mykyta (Medium)", "gender": "male", "lang": "uk", "onnx": "uk_UA-ukrainian_tts-medium.onnx", "speaker": 1},
            {"id": "piper_ua_tetiana_m", "name": "Tetiana (Medium)", "gender": "female", "lang": "uk", "onnx": "uk_UA-ukrainian_tts-medium.onnx", "speaker": 2},
            # English (en)
            {"id": "piper_en_amy", "name": "Amy (Medium)", "gender": "female", "lang": "en", "onnx": "en_US-amy-medium.onnx", "speaker": None},
            {"id": "piper_en_ryan", "name": "Ryan (Medium)", "gender": "male", "lang": "en", "onnx": "en_US-ryan-medium.onnx", "speaker": None},
            # Polski (pl)
            {"id": "piper_pl_gosia", "name": "Gosia (Medium)", "gender": "female", "lang": "pl", "onnx": "pl_PL-gosia-medium.onnx", "speaker": None},
            {"id": "piper_pl_darkman", "name": "Darkman (Medium)", "gender": "male", "lang": "pl", "onnx": "pl_PL-darkman-medium.onnx", "speaker": None},
            # Deutsch (de)
            {"id": "piper_de_eva", "name": "Eva (Low)", "gender": "female", "lang": "de", "onnx": "de_DE-eva_k-x_low.onnx", "speaker": None},
            {"id": "piper_de_thorsten", "name": "Thorsten (Medium)", "gender": "male", "lang": "de", "onnx": "de_DE-thorsten-medium.onnx", "speaker": None},
            # Français (fr)
            {"id": "piper_fr_siwis", "name": "Siwis (Medium)", "gender": "female", "lang": "fr", "onnx": "fr_FR-siwis-medium.onnx", "speaker": None},
            {"id": "piper_fr_upmc", "name": "Pierre (Medium)", "gender": "male", "lang": "fr", "onnx": "fr_FR-upmc-medium.onnx", "speaker": 1},
            # Español (es)
            {"id": "piper_es_sharvard_f", "name": "Sharvard (Medium)", "gender": "female", "lang": "es", "onnx": "es_ES-sharvard-medium.onnx", "speaker": 1},
            {"id": "piper_es_davefx", "name": "Davefx (Medium)", "gender": "male", "lang": "es", "onnx": "es_ES-davefx-medium.onnx", "speaker": None},
            # Русский (ru)
            {"id": "piper_ru_irina", "name": "Irina (Medium)", "gender": "female", "lang": "ru", "onnx": "ru_RU-irina-medium.onnx", "speaker": None},
            {"id": "piper_ru_dmitri", "name": "Dmitri (Medium)", "gender": "male", "lang": "ru", "onnx": "ru_RU-dmitri-medium.onnx", "speaker": None},
        ],
        "files": [
            # uk
            ("uk_UA-tetiana-high.onnx", "https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/tetiana/high/uk_UA-tetiana-high.onnx", "uk"),
            ("uk_UA-tetiana-high.onnx.json", "https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/tetiana/high/uk_UA-tetiana-high.onnx.json", "uk"),
            ("uk_UA-ukrainian_tts-medium.onnx", "https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium.onnx", "uk"),
            ("uk_UA-ukrainian_tts-medium.onnx.json", "https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium.onnx.json", "uk"),
            # en
            ("en_US-amy-medium.onnx", "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx", "en"),
            ("en_US-amy-medium.onnx.json", "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json", "en"),
            ("en_US-ryan-medium.onnx", "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx", "en"),
            ("en_US-ryan-medium.onnx.json", "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json", "en"),
            # pl
            ("pl_PL-gosia-medium.onnx", "https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx", "pl"),
            ("pl_PL-gosia-medium.onnx.json", "https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx.json", "pl"),
            ("pl_PL-darkman-medium.onnx", "https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/darkman/medium/pl_PL-darkman-medium.onnx", "pl"),
            ("pl_PL-darkman-medium.onnx.json", "https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/darkman/medium/pl_PL-darkman-medium.onnx.json", "pl"),
            # de
            ("de_DE-eva_k-x_low.onnx", "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/eva_k/x_low/de_DE-eva_k-x_low.onnx", "de"),
            ("de_DE-eva_k-x_low.onnx.json", "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/eva_k/x_low/de_DE-eva_k-x_low.onnx.json", "de"),
            ("de_DE-thorsten-medium.onnx", "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx", "de"),
            ("de_DE-thorsten-medium.onnx.json", "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json", "de"),
            # fr
            ("fr_FR-siwis-medium.onnx", "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx", "fr"),
            ("fr_FR-siwis-medium.onnx.json", "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx.json", "fr"),
            ("fr_FR-upmc-medium.onnx", "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx", "fr"),
            ("fr_FR-upmc-medium.onnx.json", "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx.json", "fr"),
            # es
            ("es_ES-sharvard-medium.onnx", "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx", "es"),
            ("es_ES-sharvard-medium.onnx.json", "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx.json", "es"),
            ("es_ES-davefx-medium.onnx", "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx", "es"),
            ("es_ES-davefx-medium.onnx.json", "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json", "es"),
            # ru
            ("ru_RU-irina-medium.onnx", "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx", "ru"),
            ("ru_RU-irina-medium.onnx.json", "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx.json", "ru"),
            ("ru_RU-dmitri-medium.onnx", "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx", "ru"),
            ("ru_RU-dmitri-medium.onnx.json", "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx.json", "ru"),
        ],
    },
    "silero": {
        "id": "silero",
        "title": "Silero TTS v5",
        "size_label": "~140 МБ",
        "voices": [
            # Українська (uk)
            {"id": "silero_ua_tetiana", "name": "Tetiana (v5)", "gender": "female", "lang": "uk", "model": "v5_cis_ext.pt", "speaker": "ukr_tetiana"},
            {"id": "silero_ua_kateryna", "name": "Kateryna (v5)", "gender": "female", "lang": "uk", "model": "v5_cis_ext.pt", "speaker": "ukr_kateryna"},
            {"id": "silero_ua_lada", "name": "Lada (v5)", "gender": "female", "lang": "uk", "model": "v5_cis_ext.pt", "speaker": "ukr_lada"},
            {"id": "silero_ua_mykyta", "name": "Mykyta (v5)", "gender": "male", "lang": "uk", "model": "v5_cis_ext.pt", "speaker": "ukr_mykyta"},
            {"id": "silero_ua_oleksa", "name": "Oleksa (v5)", "gender": "male", "lang": "uk", "model": "v5_cis_ext.pt", "speaker": "ukr_oleksa"},
            # English (en)
            {"id": "silero_en_0", "name": "Clara (v3)", "gender": "female", "lang": "en", "model": "v3_en.pt", "speaker": "en_0"},
            {"id": "silero_en_1", "name": "Roger (v3)", "gender": "male", "lang": "en", "model": "v3_en.pt", "speaker": "en_1"},
            {"id": "silero_en_2", "name": "Oliver (v3)", "gender": "male", "lang": "en", "model": "v3_en.pt", "speaker": "en_2"},
            # Deutsch (de)
            {"id": "silero_de_eva", "name": "Eva (v3)", "gender": "female", "lang": "de", "model": "v3_de.pt", "speaker": "eva_k"},
            {"id": "silero_de_bernd", "name": "Bernd (v3)", "gender": "male", "lang": "de", "model": "v3_de.pt", "speaker": "bernd_ungerer"},
            # Français (fr)
            {"id": "silero_fr_gilles", "name": "Gilles (v3)", "gender": "male", "lang": "fr", "model": "v3_fr.pt", "speaker": "fr_0"},
            # Español (es)
            {"id": "silero_es_tux", "name": "Carlos (v3)", "gender": "male", "lang": "es", "model": "v3_es.pt", "speaker": "es_0"},
            # Русский (ru)
            {"id": "silero_ru_xenia", "name": "Xenia (v5.6)", "gender": "female", "lang": "ru", "model": "v5_6_ru.pt", "speaker": "xenia"},
            {"id": "silero_ru_kseniya", "name": "Kseniya (v5.6)", "gender": "female", "lang": "ru", "model": "v5_6_ru.pt", "speaker": "kseniya"},
            {"id": "silero_ru_baya", "name": "Baya (v5.6)", "gender": "female", "lang": "ru", "model": "v5_6_ru.pt", "speaker": "baya"},
            {"id": "silero_ru_aidar", "name": "Aidar (v5.6)", "gender": "male", "lang": "ru", "model": "v5_6_ru.pt", "speaker": "aidar"},
            {"id": "silero_ru_eugene", "name": "Eugene (v5.6)", "gender": "male", "lang": "ru", "model": "v5_6_ru.pt", "speaker": "eugene"},
        ],
        "files": [
            ("v5_cis_ext.pt", "https://models.silero.ai/models/tts/ru/v5_cis_ext.pt", "uk"),
            ("v3_en.pt", "https://models.silero.ai/models/tts/en/v3_en.pt", "en"),
            ("v3_de.pt", "https://models.silero.ai/models/tts/de/v3_de.pt", "de"),
            ("v3_fr.pt", "https://models.silero.ai/models/tts/fr/v3_fr.pt", "fr"),
            ("v3_es.pt", "https://models.silero.ai/models/tts/es/v3_es.pt", "es"),
            ("v5_6_ru.pt", "https://models.silero.ai/models/tts/ru/v5_6_ru.pt", "ru"),
        ],
    },
    "holos": {
        "id": "holos",
        "title": "HolosTTS (StyleTTS2)",
        "size_label": "~547 МБ",
        "voices": [
            {"id": "holos_ua_haska", "name": "Haska Shyyan", "gender": "female", "lang": "uk", "speaker": "Гаська Шиян"},
            {"id": "holos_ua_spk0", "name": "Speaker 0", "gender": "male", "lang": "uk", "speaker": "Speaker_0"},
            {"id": "holos_ua_spk1", "name": "Speaker 1", "gender": "male", "lang": "uk", "speaker": "Speaker_1"},
            {"id": "holos_ua_spk3", "name": "Speaker 3", "gender": "female", "lang": "uk", "speaker": "Speaker_3"},
            {"id": "holos_ua_spk11", "name": "Speaker 11", "gender": "male", "lang": "uk", "speaker": "Speaker_11"},
            {"id": "holos_ua_spk12", "name": "Speaker 12", "gender": "female", "lang": "uk", "speaker": "Speaker_12"},
            {"id": "holos_ua_spk16", "name": "Speaker 16", "gender": "male", "lang": "uk", "speaker": "Speaker_16"},
            {"id": "holos_ua_spk17", "name": "Speaker 17", "gender": "female", "lang": "uk", "speaker": "Speaker_17"},
            {"id": "holos_ua_spk18", "name": "Speaker 18", "gender": "male", "lang": "uk", "speaker": "Speaker_18"},
            {"id": "holos_ua_spk21", "name": "Speaker 21", "gender": "male", "lang": "uk", "speaker": "Speaker_21"},
            {"id": "holos_ua_spk24", "name": "Speaker 24", "gender": "male", "lang": "uk", "speaker": "Speaker_24"},
            {"id": "holos_ua_spk26", "name": "Speaker 26", "gender": "female", "lang": "uk", "speaker": "Speaker_26"},
            {"id": "holos_ua_spk29", "name": "Speaker 29", "gender": "male", "lang": "uk", "speaker": "Speaker_29"},
            {"id": "holos_ua_spk30", "name": "Speaker 30", "gender": "female", "lang": "uk", "speaker": "Speaker_30"},
            {"id": "holos_ua_spk34", "name": "Speaker 34", "gender": "male", "lang": "uk", "speaker": "Speaker_34"},
            {"id": "holos_ua_spk35", "name": "Speaker 35", "gender": "male", "lang": "uk", "speaker": "Speaker_35"},
            {"id": "holos_ua_spk36", "name": "Speaker 36", "gender": "male", "lang": "uk", "speaker": "Speaker_36"},
            {"id": "holos_ua_spk37", "name": "Speaker 37", "gender": "female", "lang": "uk", "speaker": "Speaker_37"},
            {"id": "holos_ua_spk42", "name": "Speaker 42", "gender": "male", "lang": "uk", "speaker": "Speaker_42"},
            {"id": "holos_ua_spk43", "name": "Speaker 43", "gender": "female", "lang": "uk", "speaker": "Speaker_43"},
            {"id": "holos_ua_spk53", "name": "Speaker 53", "gender": "male", "lang": "uk", "speaker": "Speaker_53"},
            {"id": "holos_ua_spk56", "name": "Speaker 56", "gender": "male", "lang": "uk", "speaker": "Speaker_56"},
            {"id": "holos_ua_spk59", "name": "Speaker 59", "gender": "female", "lang": "uk", "speaker": "Speaker_59"},
            {"id": "holos_ua_spk67", "name": "Speaker 67", "gender": "female", "lang": "uk", "speaker": "Speaker_67"},
            {"id": "holos_ua_spk69", "name": "Speaker 69", "gender": "male", "lang": "uk", "speaker": "Speaker_69"},
            {"id": "holos_ua_spk78", "name": "Speaker 78", "gender": "female", "lang": "uk", "speaker": "Speaker_78"},
            {"id": "holos_ua_spk84", "name": "Speaker 84", "gender": "male", "lang": "uk", "speaker": "Speaker_84"},
        ],
        "files": [
            ("onnx/encoder.onnx", "https://huggingface.co/patriotyk/HolosTTS/resolve/main/onnx/encoder.onnx", "uk"),
            ("onnx/prosody.onnx", "https://huggingface.co/patriotyk/HolosTTS/resolve/main/onnx/prosody.onnx", "uk"),
            ("onnx/decoder.onnx", "https://huggingface.co/patriotyk/HolosTTS/resolve/main/onnx/decoder.onnx", "uk"),
            ("onnx/holos_onnx.json", "https://huggingface.co/patriotyk/HolosTTS/resolve/main/onnx/holos_onnx.json", "uk"),
            ("onnx/voices.npz", "https://huggingface.co/patriotyk/HolosTTS/resolve/main/onnx/voices.npz", "uk"),
        ],
    },
    "teratts": {
        "id": "teratts",
        "title": "TeraTTS (VITS)",
        "size_label": "~715 МБ",
        "voices": [
            {"id": "teratts_ru_natasha", "name": "Natasha (VITS)", "gender": "female", "lang": "ru", "repo": "TeraTTS/natasha-g2p-vits"},
        ],
        "hf_repo": "TeraTTS/natasha-g2p-vits",
    },
    "kokoro_82m": {
        "id": "kokoro_82m",
        "title": "Kokoro-82M",
        "size_label": "~258–337 МБ",
        "voices": [
            {"id": "kokoro_af_heart", "name": "Heart (US)", "gender": "female", "lang": "en", "speaker": "af_heart"},
            {"id": "kokoro_af_bella", "name": "Bella (US)", "gender": "female", "lang": "en", "speaker": "af_bella"},
            {"id": "kokoro_af_sarah", "name": "Sarah (US)", "gender": "female", "lang": "en", "speaker": "af_sarah"},
            {"id": "kokoro_af_nova", "name": "Nova (US)", "gender": "female", "lang": "en", "speaker": "af_nova"},
            {"id": "kokoro_af_sky", "name": "Sky (US)", "gender": "female", "lang": "en", "speaker": "af_sky"},
            {"id": "kokoro_af_jessica", "name": "Jessica (US)", "gender": "female", "lang": "en", "speaker": "af_jessica"},
            {"id": "kokoro_af_nicole", "name": "Nicole (US)", "gender": "female", "lang": "en", "speaker": "af_nicole"},
            {"id": "kokoro_af_aoede", "name": "Aoede (US)", "gender": "female", "lang": "en", "speaker": "af_aoede"},
            {"id": "kokoro_am_michael", "name": "Michael (US)", "gender": "male", "lang": "en", "speaker": "am_michael"},
            {"id": "kokoro_am_adam", "name": "Adam (US)", "gender": "male", "lang": "en", "speaker": "am_adam"},
            {"id": "kokoro_am_fenrir", "name": "Fenrir (US)", "gender": "male", "lang": "en", "speaker": "am_fenrir"},
            {"id": "kokoro_am_liam", "name": "Liam (US)", "gender": "male", "lang": "en", "speaker": "am_liam"},
            {"id": "kokoro_bf_emma", "name": "Emma (UK)", "gender": "female", "lang": "en", "speaker": "bf_emma"},
            {"id": "kokoro_bf_isabella", "name": "Isabella (UK)", "gender": "female", "lang": "en", "speaker": "bf_isabella"},
            {"id": "kokoro_bm_george", "name": "George (UK)", "gender": "male", "lang": "en", "speaker": "bm_george"},
            {"id": "kokoro_bm_lewis", "name": "Lewis (UK)", "gender": "male", "lang": "en", "speaker": "bm_lewis"},
        ],
    },
    "soprano_1_1": {
        "id": "soprano_1_1",
        "title": "Soprano-1.1-80M",
        "size_label": "~139 МБ",
        "voices": [
            {"id": "soprano_en_default", "name": "Soprano (EN, 32kHz)", "gender": "female", "lang": "en"},
        ],
    },
    "omnivoice": {
        "id": "omnivoice",
        "title": "OmniVoice",
        "size_label": "~1.2 ГБ",
        "voices": [
            {"id": "omnivoice_uk", "name": "OmniVoice (UK / Clone & Design)", "gender": None, "lang": "uk"},
            {"id": "omnivoice_en", "name": "OmniVoice (EN / Clone & Design)", "gender": None, "lang": "en"},
            {"id": "omnivoice_ru", "name": "OmniVoice (RU / Clone & Design)", "gender": None, "lang": "ru"},
            {"id": "omnivoice_pl", "name": "OmniVoice (PL / Clone & Design)", "gender": None, "lang": "pl"},
            {"id": "omnivoice_de", "name": "OmniVoice (DE / Clone & Design)", "gender": None, "lang": "de"},
            {"id": "omnivoice_fr", "name": "OmniVoice (FR / Clone & Design)", "gender": None, "lang": "fr"},
            {"id": "omnivoice_es", "name": "OmniVoice (ES / Clone & Design)", "gender": None, "lang": "es"},
        ],
    },
    "zonos3_mini": {
        "id": "zonos3_mini",
        "title": "ZONOS3-Mini",
        "size_label": "~1.9 ГБ",
        "voices": [
            # English built-in voices
            {"id": "zonos3_en_savannah", "name": "Savannah (US)", "gender": "female", "lang": "en", "speaker": "Savannah_en-American_F_1"},
            {"id": "zonos3_en_imogen", "name": "Imogen (UK)", "gender": "female", "lang": "en", "speaker": "Imogen_en-British_F_1"},
            {"id": "zonos3_en_sakura", "name": "Sakura (Anime)", "gender": "female", "lang": "en", "speaker": "Sakura_en-anime_F_1"},
            {"id": "zonos3_en_willow", "name": "Willow (Soft)", "gender": "female", "lang": "en", "speaker": "Willow_en-asrm_F_1"},
            {"id": "zonos3_en_morgana", "name": "Morgana", "gender": "female", "lang": "en", "speaker": "Morgana_en-witch_F_1"},
            {"id": "zonos3_en_wyatt", "name": "Wyatt (US)", "gender": "male", "lang": "en", "speaker": "Wyatt_en-American_M_1"},
            {"id": "zonos3_en_rupert", "name": "Rupert (UK)", "gender": "male", "lang": "en", "speaker": "Rupert_en-British_M_1"},
            {"id": "zonos3_en_cillian", "name": "Cillian (Irish)", "gender": "male", "lang": "en", "speaker": "Cillian_en-Irish_M_1"},
            {"id": "zonos3_en_jett", "name": "Jett (Energetic)", "gender": "male", "lang": "en", "speaker": "Jett_en-energetic_M_1"},
            {"id": "zonos3_en_thaddeus", "name": "Thaddeus (Announcer)", "gender": "male", "lang": "en", "speaker": "Thaddeus_en-announcervoice_M_1"},
            # German / French / Spanish built-in voices
            {"id": "zonos3_de_liesel", "name": "Liesel (DE)", "gender": "female", "lang": "de", "speaker": "Liesel_de_F_1"},
            {"id": "zonos3_de_wolfgang", "name": "Wolfgang (DE)", "gender": "male", "lang": "de", "speaker": "Wolfgang_de_M_1"},
            {"id": "zonos3_fr_gaspard", "name": "Gaspard (FR)", "gender": "male", "lang": "fr", "speaker": "Gaspard_fr_M_1"},
            {"id": "zonos3_es_ximena", "name": "Ximena (ES)", "gender": "female", "lang": "es", "speaker": "Ximena_es_F_1"},
            {"id": "zonos3_es_rodrigo", "name": "Rodrigo (ES)", "gender": "male", "lang": "es", "speaker": "Rodrigo_es_M_1"},
            # Ukrainian / Russian / Polish (built-in or cloned via ref audio)
            {"id": "zonos3_uk_default", "name": "ZONOS3 (UK / Clone)", "gender": None, "lang": "uk", "speaker": "Savannah_en-American_F_1"},
            {"id": "zonos3_ru_default", "name": "ZONOS3 (RU / Clone)", "gender": None, "lang": "ru", "speaker": "Savannah_en-American_F_1"},
            {"id": "zonos3_pl_default", "name": "ZONOS3 (PL / Clone)", "gender": None, "lang": "pl", "speaker": "Savannah_en-American_F_1"},
        ],
    },
    "qwen3_tts": {
        "id": "qwen3_tts",
        "title": "Qwen3-TTS 1.7B",
        "size_label": "~2.1 ГБ",
        "voices": [
            {"id": "qwen3_en", "name": "Qwen3-TTS (EN / Clone)", "gender": "female", "lang": "en"},
            {"id": "qwen3_ru", "name": "Qwen3-TTS (RU / Clone)", "gender": "female", "lang": "ru"},
            {"id": "qwen3_de", "name": "Qwen3-TTS (DE / Clone)", "gender": "female", "lang": "de"},
            {"id": "qwen3_fr", "name": "Qwen3-TTS (FR / Clone)", "gender": "female", "lang": "fr"},
            {"id": "qwen3_es", "name": "Qwen3-TTS (ES / Clone)", "gender": "female", "lang": "es"},
        ],
    },
}

# ──────────────────────────────────────────────────────────────────────
# 2. Опис онлайн-провайдерів TTS
# ──────────────────────────────────────────────────────────────────────

ONLINE_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "edge": {
        "provider": "edge",
        "title": "Microsoft Edge-TTS",
        "subtitle": "Безкоштовно",
        "voices": [
            # uk
            {"id": "uk-UA-PolinaNeural", "name": "Polina (Neural)", "gender": "female", "lang": "uk"},
            {"id": "uk-UA-PolinaNeural::custom", "name": "Polina DSP Custom (Neural)", "gender": "female", "lang": "uk"},
            {"id": "uk-UA-OstapNeural", "name": "Ostap (Neural)", "gender": "male", "lang": "uk"},
            # en
            {"id": "en-US-AriaNeural", "name": "Aria (Neural)", "gender": "female", "lang": "en"},
            {"id": "en-US-GuyNeural", "name": "Guy (Neural)", "gender": "male", "lang": "en"},
            {"id": "en-US-JennyNeural", "name": "Jenny (Neural)", "gender": "female", "lang": "en"},
            # pl
            {"id": "pl-PL-ZofiaNeural", "name": "Zofia (Neural)", "gender": "female", "lang": "pl"},
            {"id": "pl-PL-MarekNeural", "name": "Marek (Neural)", "gender": "male", "lang": "pl"},
            # de
            {"id": "de-DE-KatjaNeural", "name": "Katja (Neural)", "gender": "female", "lang": "de"},
            {"id": "de-DE-ConradNeural", "name": "Conrad (Neural)", "gender": "male", "lang": "de"},
            # fr
            {"id": "fr-FR-DeniseNeural", "name": "Denise (Neural)", "gender": "female", "lang": "fr"},
            {"id": "fr-FR-HenriNeural", "name": "Henri (Neural)", "gender": "male", "lang": "fr"},
            # es
            {"id": "es-ES-ElviraNeural", "name": "Elvira (Neural)", "gender": "female", "lang": "es"},
            {"id": "es-ES-AlvaroNeural", "name": "Alvaro (Neural)", "gender": "male", "lang": "es"},
            # ru
            {"id": "ru-RU-SvetlanaNeural", "name": "Svetlana (Neural)", "gender": "female", "lang": "ru"},
            {"id": "ru-RU-DariyaNeural", "name": "Dariya (Neural)", "gender": "female", "lang": "ru"},
            {"id": "ru-RU-DmitryNeural", "name": "Dmitry (Neural)", "gender": "male", "lang": "ru"},
        ],
    },
    "google": {
        "provider": "google",
        "title": "Google Cloud TTS API",
        "subtitle": "WaveNet / Neural2",
        "voices": [
            {"id": "uk-UA-Wavenet-A", "name": "Wavenet-A", "gender": "female", "lang": "uk"},
            {"id": "uk-UA-Standard-A", "name": "Standard-A", "gender": "female", "lang": "uk"},
            {"id": "en-US-Wavenet-F", "name": "Wavenet-F", "gender": "female", "lang": "en"},
            {"id": "en-US-Wavenet-D", "name": "Wavenet-D", "gender": "male", "lang": "en"},
            {"id": "pl-PL-Wavenet-A", "name": "Wavenet-A", "gender": "female", "lang": "pl"},
            {"id": "pl-PL-Wavenet-B", "name": "Wavenet-B", "gender": "male", "lang": "pl"},
            {"id": "de-DE-Wavenet-F", "name": "Wavenet-F", "gender": "female", "lang": "de"},
            {"id": "de-DE-Wavenet-B", "name": "Wavenet-B", "gender": "male", "lang": "de"},
            {"id": "fr-FR-Wavenet-A", "name": "Wavenet-A", "gender": "female", "lang": "fr"},
            {"id": "fr-FR-Wavenet-B", "name": "Wavenet-B", "gender": "male", "lang": "fr"},
            {"id": "es-ES-Wavenet-C", "name": "Wavenet-C", "gender": "female", "lang": "es"},
            {"id": "es-ES-Wavenet-B", "name": "Wavenet-B", "gender": "male", "lang": "es"},
            {"id": "ru-RU-Wavenet-A", "name": "Wavenet-A", "gender": "female", "lang": "ru"},
            {"id": "ru-RU-Wavenet-B", "name": "Wavenet-B", "gender": "male", "lang": "ru"},
        ],
    },
    "elevenlabs": {
        "provider": "elevenlabs",
        "title": "ElevenLabs API",
        "subtitle": "Multilingual v2",
        "voices": [
            {"id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel", "gender": "female", "lang": "multi"},
            {"id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella", "gender": "female", "lang": "multi"},
            {"id": "XB0fDUnXU5powFXDhCwa", "name": "Charlotte", "gender": "female", "lang": "multi"},
            {"id": "pNInz6obpgDQGcFmaJgB", "name": "Adam", "gender": "male", "lang": "multi"},
            {"id": "ErXwobaYiN019PkySvjV", "name": "Antoni", "gender": "male", "lang": "multi"},
        ],
    },
    "openai": {
        "provider": "openai",
        "title": "OpenAI TTS API",
        "subtitle": "TTS-1",
        "voices": [
            {"id": "nova", "name": "Nova", "gender": "female", "lang": "multi"},
            {"id": "shimmer", "name": "Shimmer", "gender": "female", "lang": "multi"},
            {"id": "coral", "name": "Coral", "gender": "female", "lang": "multi"},
            {"id": "alloy", "name": "Alloy", "gender": "male", "lang": "multi"},
            {"id": "onyx", "name": "Onyx", "gender": "male", "lang": "multi"},
        ],
    },
    "fish_audio": {
        "provider": "fish_audio",
        "title": "Fish Audio API",
        "subtitle": "Speech S1 (Multilingual)",
        "voices": [
            {"id": "default", "name": "Стандартний голос (M)", "gender": "male", "lang": "multi"},
        ],
    },
    "gemini_tts": {
        "provider": "gemini_tts",
        "title": "Google AI Studio (Gemini TTS)",
        "subtitle": "Gemini Flash TTS",
        "voices": [
            {"id": "Kore", "name": "Kore (Gemini)", "gender": "female", "lang": "multi"},
            {"id": "Aoede", "name": "Aoede (Gemini)", "gender": "female", "lang": "multi"},
            {"id": "Zephyr", "name": "Zephyr (Gemini)", "gender": "female", "lang": "multi"},
            {"id": "Leda", "name": "Leda (Gemini)", "gender": "female", "lang": "multi"},
            {"id": "Erinome", "name": "Erinome (Gemini)", "gender": "female", "lang": "multi"},
            {"id": "Callirrhoe", "name": "Callirrhoe (Gemini)", "gender": "female", "lang": "multi"},
            {"id": "Autonoe", "name": "Autonoe (Gemini)", "gender": "female", "lang": "multi"},
            {"id": "Despina", "name": "Despina (Gemini)", "gender": "female", "lang": "multi"},
            {"id": "Laomedeia", "name": "Laomedeia (Gemini)", "gender": "female", "lang": "multi"},
            {"id": "Pulcherrima", "name": "Pulcherrima (Gemini)", "gender": "female", "lang": "multi"},
            {"id": "Sulafat", "name": "Sulafat (Gemini)", "gender": "female", "lang": "multi"},
            {"id": "Vindemiatrix", "name": "Vindemiatrix (Gemini)", "gender": "female", "lang": "multi"},
            {"id": "Achernar", "name": "Achernar (Gemini)", "gender": "female", "lang": "multi"},
            {"id": "Gacrux", "name": "Gacrux (Gemini)", "gender": "female", "lang": "multi"},
            {"id": "Puck", "name": "Puck (Gemini)", "gender": "male", "lang": "multi"},
            {"id": "Charon", "name": "Charon (Gemini)", "gender": "male", "lang": "multi"},
            {"id": "Fenrir", "name": "Fenrir (Gemini)", "gender": "male", "lang": "multi"},
            {"id": "Orus", "name": "Orus (Gemini)", "gender": "male", "lang": "multi"},
            {"id": "Achird", "name": "Achird (Gemini)", "gender": "male", "lang": "multi"},
            {"id": "Algenib", "name": "Algenib (Gemini)", "gender": "male", "lang": "multi"},
            {"id": "Algieba", "name": "Algieba (Gemini)", "gender": "male", "lang": "multi"},
            {"id": "Alnilam", "name": "Alnilam (Gemini)", "gender": "male", "lang": "multi"},
            {"id": "Enceladus", "name": "Enceladus (Gemini)", "gender": "male", "lang": "multi"},
            {"id": "Iapetus", "name": "Iapetus (Gemini)", "gender": "male", "lang": "multi"},
            {"id": "Rasalgethi", "name": "Rasalgethi (Gemini)", "gender": "male", "lang": "multi"},
            {"id": "Sadachbia", "name": "Sadachbia (Gemini)", "gender": "male", "lang": "multi"},
            {"id": "Sadaltager", "name": "Sadaltager (Gemini)", "gender": "male", "lang": "multi"},
            {"id": "Schedar", "name": "Schedar (Gemini)", "gender": "male", "lang": "multi"},
            {"id": "Umbriel", "name": "Umbriel (Gemini)", "gender": "male", "lang": "multi"},
            {"id": "Zubenelgenubi", "name": "Zubenelgenubi (Gemini)", "gender": "male", "lang": "multi"},
        ],
    },
    "silero_bot": {
        "provider": "silero_bot",
        "title": "Silero Bot (Telegram)",
        "subtitle": "@silero_voice_bot",
        "voices": [
            {"id": "__bot_current__", "name": "Поточний голос бота (/speaker)", "gender": None, "lang": "multi"},
        ],
    },
}

GOOGLE_USAGE_FILE = DATA_DIR / "google_tts_usage.json"
_google_usage_lock = threading.Lock()

# Безпечні місячні ліміти з запасом -50 000 символів (автоматично скидаються 1-го числа)
GOOGLE_FREE_SAFE_LIMITS: Dict[str, int] = {
    "Standard": 3_950_000,    # 4M - 50k запас
    "WaveNet": 3_950_000,     # 4M - 50k запас
    "Neural2": 950_000,       # 1M - 50k запас
    "Polyglot": 950_000,      # 1M - 50k запас
    "Chirp 3: HD": 950_000,   # 1M - 50k запас
    "Studio": 950_000,        # 1M - 50k запас
}


def classify_google_voice_model(voice_id: str) -> Tuple[str, bool]:
    """Повертає (сімейство_моделі, чи_має_безкоштовний_ліміт)."""
    low = str(voice_id or "").lower()
    if "chirp3-hd" in low:
        return "Chirp 3: HD", True
    if "wavenet" in low:
        return "WaveNet", True
    if "standard" in low:
        return "Standard", True
    if "neural2" in low:
        return "Neural2", True
    if "polyglot" in low:
        return "Polyglot", True
    if "studio" in low:
        return "Studio", True
    if "chirp-hd" in low or "-chirp-" in low:
        return "Chirp HD", False
    if "journey" in low:
        return "Journey", False
    if "news" in low:
        return "News", False
    if "casual" in low:
        return "Casual", False
    return "Other", False


def _current_google_month_key() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m")


def load_google_month_usage_map() -> Dict[str, int]:
    """Повертає статистику використаних символів Google Cloud за поточний місяць (скидається 1-го числа)."""
    month_key = _current_google_month_key()
    with _google_usage_lock:
        if not GOOGLE_USAGE_FILE.exists():
            return {}
        try:
            data = json.loads(GOOGLE_USAGE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get(month_key), dict):
                return {str(k): int(v) for k, v in data[month_key].items()}
        except Exception:
            pass
    return {}


def check_google_quota(family: str, incoming_chars: int) -> Tuple[bool, int, int]:
    """Перевіряє, чи не перевищено безпечний безкоштовний місячний ліміт для сімейства Google Cloud TTS."""
    limit = int(GOOGLE_FREE_SAFE_LIMITS.get(family, 0))
    if limit <= 0:
        return False, 0, 0
    used = int(load_google_month_usage_map().get(family, 0))
    if used + max(0, int(incoming_chars)) > limit:
        return False, used, limit
    return True, used, limit


def record_google_usage(family: str, char_count: int) -> None:
    """Записує витрачені символи Google Cloud TTS за поточний місяць у google_tts_usage.json."""
    if char_count <= 0:
        return
    month_key = _current_google_month_key()
    with _google_usage_lock:
        data: Dict[str, Any] = {}
        if GOOGLE_USAGE_FILE.exists():
            try:
                loaded = json.loads(GOOGLE_USAGE_FILE.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = loaded
            except Exception:
                data = {}
        month_map = data.setdefault(month_key, {})
        if not isinstance(month_map, dict):
            month_map = {}
            data[month_key] = month_map
        month_map[family] = int(month_map.get(family, 0)) + int(char_count)
        try:
            GOOGLE_USAGE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass


def translate_error_message(raw_error: str, target_lang: Optional[str] = None) -> str:
    """
    Перекладає текст технічної помилки на обрану мову інтерфейсу:
    1) Google Translate (gtx)
    2) Резерв: MyMemory API (якщо Google дав 429/блокування)
    3) Локальний офлайн-словник для uk/ru на випадок відсутності інтернету
    """
    text = str(raw_error).strip()
    if not text:
        return ""

    lang = (target_lang or str(get_ui_language() or "uk")).strip().lower() or "uk"
    tl = "uk" if lang in ("uk", "ua") else lang

    if text.startswith("<urlopen error ") and text.endswith(">"):
        text = text[len("<urlopen error ") : -1].strip()

    # Якщо текст уже містить кирилицю і ми перекладаємо на uk/ru без латиниці — не чіпаємо
    if tl in ("uk", "ru") and not re.search(r"[A-Za-z]{4,}", text):
        return text
    # Якщо мова інтерфейсу англійська і в тексті немає кирилиці — він уже англійською
    if tl == "en" and not re.search(r"[А-Яа-яІіЇїЄєҐґ]", text):
        return text

    proxies = get_active_api_proxies()
    headers = {"User-Agent": "Mozilla/5.0"}

    # 1. Google Translate (gtx)
    try:
        resp = requests.get(
            "https://translate.googleapis.com/translate_a/single",
            params={"client": "gtx", "sl": "auto", "tl": tl, "dt": "t", "q": text},
            headers=headers,
            proxies=proxies,
            timeout=3.5,
        )
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data and isinstance(data[0], list):
                translated_parts = [seg[0] for seg in data[0] if isinstance(seg, list) and seg and seg[0]]
                translated = "".join(translated_parts).strip()
                if translated:
                    return translated
    except Exception:
        pass

    # 2. Резерв: MyMemory (працює навіть коли Google дає 429)
    try:
        resp = requests.get(
            "https://api.mymemory.translated.net/get",
            params={"q": text[:450], "langpair": f"en|{tl}"},
            headers=headers,
            proxies=proxies,
            timeout=4.0,
        )
        if resp.status_code == 200:
            tr = str((resp.json().get("responseData") or {}).get("translatedText") or "").strip()
            if tr and tr.lower() != text.lower():
                return tr
    except Exception:
        pass

    # 3. Локальний офлайн-переклад найчастіших помилок
    offline_maps: Dict[str, Dict[str, str]] = {
        "uk": {
            "timed out": "Час очікування з'єднання вичерпано (сервер не відповідає або заблокований)",
            "connection refused": "У з'єднанні відмовлено сервером",
            "forcibly closed": "Віддалений сервер примусово розірвав з'єднання",
            "connectionreseterror": "З'єднання було скинуто сервером",
            "getaddrinfo failed": "Не вдалося знайти адресу сервера (перевірте інтернет або увімкніть SOCKS5)",
            "ssl": "Помилка захищеного з'єднання SSL",
            "winerror 1314": "Недостатньо прав Windows (запустіть програму від імені Адміністратора)",
            "no space left on device": "Недостатньо вільного місця на диску",
            "out of memory": "Недостатньо оперативної пам'яті (OOM)",
            "401": "Помилка авторизації (перевірте API ключ)",
            "403": "Доступ заборонено сервером (403 — можливо, потрібне проксі/SOCKS5)",
            "404": "Ресурс або модель не знайдено на сервері (404)",
            "429": "Перевищено ліміт запитів або балансу API (429)",
            "socks": "Помилка підключення через SOCKS5-проксі",
        },
        "en": {
            "timed out": "Connection timed out (server did not respond or is blocked)",
            "connection refused": "Connection refused by server",
            "forcibly closed": "Remote server forcibly closed the connection",
            "connectionreseterror": "Connection was reset by server",
            "getaddrinfo failed": "Could not resolve server address (check internet or enable SOCKS5)",
            "ssl": "SSL secure connection error",
            "winerror 1314": "Insufficient Windows permissions (run as Administrator)",
            "no space left on device": "Not enough free disk space",
            "out of memory": "Out of RAM (OOM)",
            "401": "Authorization failed (check API key)",
            "403": "Access forbidden by server (403 — proxy/SOCKS5 may be needed)",
            "404": "Resource or model not found on server (404)",
            "429": "API rate limit or quota exceeded (429)",
            "socks": "SOCKS5 proxy connection error",
        },
        "ru": {
            "timed out": "Время ожидания соединения истекло (сервер не отвечает или заблокирован)",
            "connection refused": "В соединении отказано сервером",
            "forcibly closed": "Удалённый сервер принудительно разорвал соединение",
            "connectionreseterror": "Соединение было сброшено сервером",
            "getaddrinfo failed": "Не удалось найти адрес сервера (проверьте интернет или включите SOCKS5)",
            "ssl": "Ошибка защищённого соединения SSL",
            "winerror 1314": "Недостаточно прав Windows (запустите программу от имени Администратора)",
            "no space left on device": "Недостаточно свободного места на диске",
            "out of memory": "Недостаточно оперативной памяти (OOM)",
            "401": "Ошибка авторизации (проверьте API ключ)",
            "403": "Доступ запрещён сервером (403 — возможно, нужен прокси/SOCKS5)",
            "404": "Ресурс или модель не найдены на сервере (404)",
            "429": "Превышен лимит запросов или баланса API (429)",
            "socks": "Ошибка подключения через SOCKS5-прокси",
        },
        "pl": {
            "timed out": "Upłynął limit czasu połączenia (serwer nie odpowiada lub jest zablokowany)",
            "connection refused": "Połączenie odrzucone przez serwer",
            "forcibly closed": "Zdalny serwer wymusił zamknięcie połączenia",
            "connectionreseterror": "Połączenie zostało zresetowane przez serwer",
            "getaddrinfo failed": "Nie udało się znaleźć adresu serwera (sprawdź internet lub włącz SOCKS5)",
            "ssl": "Błąd bezpiecznego połączenia SSL",
            "winerror 1314": "Brak uprawnień Windows (uruchom program jako Administrator)",
            "no space left on device": "Brak wolnego miejsca na dysku",
            "out of memory": "Brak pamięci RAM (OOM)",
            "401": "Błąd autoryzacji (sprawdź klucz API)",
            "403": "Dostęp zabroniony przez serwer (403 — może być wymagane proxy/SOCKS5)",
            "404": "Nie znaleziono zasobu lub modelu na serwerze (404)",
            "429": "Przekroczono limit zapytań lub środków API (429)",
            "socks": "Błąd połączenia przez proxy SOCKS5",
        },
        "de": {
            "timed out": "Zeitüberschreitung der Verbindung (Server antwortet nicht oder ist blockiert)",
            "connection refused": "Verbindung vom Server abgelehnt",
            "forcibly closed": "Verbindung wurde vom Remoteserver zwangsweise getrennt",
            "connectionreseterror": "Verbindung wurde vom Server zurückgesetzt",
            "getaddrinfo failed": "Serveradresse konnte nicht aufgelöst werden (Internet prüfen oder SOCKS5 aktivieren)",
            "ssl": "Fehler bei sicherer SSL-Verbindung",
            "winerror 1314": "Unzureichende Windows-Rechte (Programm als Administrator ausführen)",
            "no space left on device": "Nicht genügend freier Speicherplatz auf dem Datenträger",
            "out of memory": "Nicht genügend Arbeitsspeicher (OOM)",
            "401": "Autorisierungsfehler (API-Schlüssel prüfen)",
            "403": "Zugriff vom Server verweigert (403 — ggf. Proxy/SOCKS5 erforderlich)",
            "404": "Ressource oder Modell auf dem Server nicht gefunden (404)",
            "429": "API-Anfragelimit oder Guthaben überschritten (429)",
            "socks": "SOCKS5-Proxy-Verbindungsfehler",
        },
        "fr": {
            "timed out": "Délai de connexion dépassé (le serveur ne répond pas ou est bloqué)",
            "connection refused": "Connexion refusée par le serveur",
            "forcibly closed": "Le serveur distant a fermé la connexion de force",
            "connectionreseterror": "La connexion a été réinitialisée par le serveur",
            "getaddrinfo failed": "Impossible de résoudre l'adresse du serveur (vérifiez Internet ou activez SOCKS5)",
            "ssl": "Erreur de connexion sécurisée SSL",
            "winerror 1314": "Droits Windows insuffisants (exécutez l'application en tant qu'Administrateur)",
            "no space left on device": "Espace disque insuffisant",
            "out of memory": "Mémoire RAM insuffisante (OOM)",
            "401": "Échec d'autorisation (vérifiez la clé API)",
            "403": "Accès refusé par le serveur (403 — un proxy/SOCKS5 peut être nécessaire)",
            "404": "Ressource ou modèle introuvable sur le serveur (404)",
            "429": "Limite de requêtes ou crédit API dépassé (429)",
            "socks": "Erreur de connexion au proxy SOCKS5",
        },
        "es": {
            "timed out": "Tiempo de espera de conexión agotado (el servidor no responde o está bloqueado)",
            "connection refused": "Conexión rechazada por el servidor",
            "forcibly closed": "El servidor remoto cerró la conexión forzosamente",
            "connectionreseterror": "La conexión fue restablecida por el servidor",
            "getaddrinfo failed": "No se pudo resolver la dirección del servidor (revisa internet o activa SOCKS5)",
            "ssl": "Error de conexión segura SSL",
            "winerror 1314": "Permisos de Windows insuficientes (ejecuta el programa como Administrador)",
            "no space left on device": "No hay suficiente espacio libre en el disco",
            "out of memory": "Sin memoria RAM suficiente (OOM)",
            "401": "Error de autorización (revisa la clave API)",
            "403": "Acceso denegado por el servidor (403 — puede requerir proxy/SOCKS5)",
            "404": "Recurso o modelo no encontrado en el servidor (404)",
            "429": "Límite de solicitudes o saldo API superado (429)",
            "socks": "Error de conexión del proxy SOCKS5",
        },
    }
    omap = offline_maps.get(tl)
    if omap:
        low = text.lower()
        for k, v in omap.items():
            if k in low:
                return f"{v} ({text})"

    return text


def format_exception_message(exc: BaseException, context: str = "") -> str:
    """
    Гарантовано формує непорожнє повідомлення про помилку з винятку:
    якщо str(exc) порожній (як буває у ValueError(), AssertionError(), тощо),
    витягує причину з __cause__/__context__, args, repr(exc) та позицію у traceback.
    """
    seen: set = set()
    cur: Optional[BaseException] = exc
    msg = ""
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        cand = str(cur).strip()
        if cand:
            msg = cand
            break
        if getattr(cur, "args", None):
            non_empty_args = [str(a).strip() for a in cur.args if str(a).strip()]
            if non_empty_args:
                msg = "; ".join(non_empty_args)
                break
        cur = cur.__cause__ or cur.__context__

    if not msg:
        exc_name = type(exc).__name__
        loc = ""
        tb = exc.__traceback__
        if tb is not None:
            frames = traceback.extract_tb(tb)
            if frames:
                last = frames[-1]
                fname = Path(last.filename).name if last.filename else "<module>"
                loc = f" ({fname}:{last.lineno} in {last.name})"
        msg = f"{exc_name} (empty error message){loc}"

    if context and context.strip() and context.strip() not in msg:
        return f"{context.strip()}: {msg}"
    return msg


def compute_speech_char_delays(
    text: str,
    samples: np.ndarray,
    sr: int,
    start_offset_ms: float = 85.0,
) -> List[int]:
    """
    Обчислює точні таймінги (у мс від початку відтворення) для кожного символу
    на основі реальної енергії голосу в аудіомасиві, щоб друкарська машинка
    йшла в унісон із мовленням (чекала на тишу, гальмувала на паузах і закінчувала з останнім словом).
    """
    n_chars = len(text)
    if n_chars == 0 or samples is None or len(samples) == 0 or sr <= 0:
        return []

    mono = np.asarray(samples, dtype=np.float32)
    if mono.ndim > 1:
        mono = np.mean(mono, axis=1, dtype=np.float32)

    total_ms = (len(mono) / float(sr)) * 1000.0
    win_size = max(1, min(len(mono), int(sr * 0.025)))
    n_frames = max(1, len(mono) // win_size)
    trimmed = np.abs(mono[: n_frames * win_size]).reshape(n_frames, win_size)
    rms = np.sqrt(np.mean(trimmed * trimmed, axis=1))

    peak = float(np.max(rms)) if len(rms) > 0 else 0.0
    base_offset_ms = max(0.0, float(start_offset_ms))

    if peak < 1e-5:
        step = total_ms / max(1, n_chars)
        return [int(base_offset_ms + i * step) for i in range(n_chars)]

    thresh = peak * 0.08
    active_idx = np.where(rms >= thresh)[0]
    if len(active_idx) > 0:
        first_f = int(active_idx[0])
        last_f = int(active_idx[-1]) + 1
    else:
        first_f, last_f = 0, n_frames

    active_rms = rms[first_f:last_f]
    if len(active_rms) == 0:
        active_rms = rms
        first_f = 0

    mean_act = float(np.mean(active_rms))
    weights_audio = np.maximum(active_rms, mean_act * 0.08)
    cdf_audio = np.cumsum(weights_audio)
    cdf_audio = np.insert(cdf_audio, 0, 0.0)
    cdf_audio /= cdf_audio[-1]

    act_start_ms = (first_f * win_size / float(sr)) * 1000.0 + base_offset_ms
    act_end_ms = (last_f * win_size / float(sr)) * 1000.0 + base_offset_ms
    act_span_ms = max(120.0, act_end_ms - act_start_ms)

    char_w = []
    for ch in text:
        if ch.isspace():
            char_w.append(0.45)
        elif ch in ",;:—–-":
            char_w.append(2.2)
        elif ch in ".!?…":
            char_w.append(3.0)
        else:
            char_w.append(1.0)
    cdf_text = np.cumsum(char_w) / float(sum(char_w))

    frame_times = np.linspace(act_start_ms, act_start_ms + act_span_ms, len(cdf_audio))
    delays = np.interp(cdf_text, cdf_audio, frame_times)
    return [int(round(d)) for d in delays]


def is_network_or_download_error(err_text: str) -> bool:
    """Перевіряє, чи пов'язана помилка із завантаженням, інтернетом або блокуванням сервера провайдером."""
    low = str(err_text).lower()
    keywords = (
        "timed out",
        "timeout",
        "час очікування",
        "з'єднання",
        "connection",
        "connecttimeout",
        "readtimeout",
        "urlerror",
        "urlopen",
        "getaddrinfo",
        "name or service not known",
        "nameresolutionerror",
        "max retries exceeded",
        "встановлення",
        "завантаження",
        "download",
        "network",
        "інтернет",
        "proxy",
        "проксі",
        "403",
        "404",
        "502",
        "503",
        "504",
        "silero.ai",
        "huggingface.co",
        "github.com",
    )
    return any(k in low for k in keywords)


_SAMPLE_FEMALE_DIR_NAMES = {"female", "females", "жіночі", "жіночий", "женские", "женский"}
_SAMPLE_MALE_DIR_NAMES = {"male", "males", "чоловічі", "чоловічий", "мужские", "мужской"}
_SAMPLE_FEMALE_TOKEN_RE = re.compile(
    r"(?:^|[\s_\-\(\)\[\]])(?:ж|жін\w*|жен\w*|female|fem|woman|girl|f)(?=$|[\s_\-\(\)\[\]])",
    re.IGNORECASE,
)
_SAMPLE_MALE_TOKEN_RE = re.compile(
    r"(?:^|[\s_\-\(\)\[\]])(?:ч|чол\w*|муж\w*|м|male|man|boy|m)(?=$|[\s_\-\(\)\[\]])",
    re.IGNORECASE,
)


def _infer_sample_gender(wav_path: Path, raw_stem: str) -> str:
    """Визначає стать диктора з підпапки або маркера в назві файлу (напр. `Ім'я Ж`, `Ім'я Ч`)."""
    for parent in wav_path.parents:
        low_p = parent.name.strip().lower()
        if low_p in _SAMPLE_FEMALE_DIR_NAMES:
            return "female"
        if low_p in _SAMPLE_MALE_DIR_NAMES:
            return "male"
        if low_p in ("samples", "omnivoice", "qwen3_tts", "zonos3_mini"):
            break
    for text_cand in (wav_path.stem, raw_stem):
        if _SAMPLE_FEMALE_TOKEN_RE.search(text_cand):
            return "female"
        if _SAMPLE_MALE_TOKEN_RE.search(text_cand):
            return "male"
    return ""


def _normalize_sample_lang(lang: Optional[str]) -> str:
    code = str(lang or config.get("app_language", "uk") or "uk").strip().lower()
    if code == "ua":
        code = "uk"
    return code if code in ("uk", "en", "ru", "pl", "de", "fr", "es") else "uk"


CLONE_ENGINE_IDS = ("omnivoice", "qwen3_tts", "zonos3_mini")
SHARED_CLONE_SAMPLES_SUBDIR = "clone_voices"


def _sample_storage_key(engine_id: str) -> str:
    return SHARED_CLONE_SAMPLES_SUBDIR if engine_id in CLONE_ENGINE_IDS else engine_id


def _sample_scan_keys(engine_id: str) -> Tuple[str, ...]:
    if engine_id in CLONE_ENGINE_IDS:
        # Для ZONOS3-Mini папка data/samples/zonos3_mini/ зарезервована під власні голоси моделі,
        # щоб не змішувати їх зі спільним каталогом голосів для клонування (clone_voices).
        return (SHARED_CLONE_SAMPLES_SUBDIR, "qwen3_tts")
    return (engine_id,)


def _is_path_under_dir(path_obj: Path, base_dir: Path) -> bool:
    try:
        return path_obj.is_relative_to(base_dir)
    except AttributeError:
        return str(path_obj).lower().startswith(str(base_dir).lower() + os.sep)


def register_custom_clone_sample(engine_id: str, src_path: str, lang: Optional[str] = None) -> str:
    """Реєструє вибраний користувачем файл голосу у спільній папці для всіх рушіїв клонування."""
    clean = str(src_path or "").strip()
    if not clean:
        return ""
    src_p = Path(clean)
    if not src_p.is_file():
        return ""
    try:
        src_resolved = src_p.resolve()
    except OSError:
        src_resolved = src_p

    storage_key = _sample_storage_key(engine_id)
    scan_keys = _sample_scan_keys(engine_id)
    custom_map = dict(config.get("custom_clone_samples", {}) or {})
    eng_entries = [
        dict(e) for e in (custom_map.get(storage_key) or [])
        if isinstance(e, dict) and str(e.get("path", "")).strip()
    ]

    in_engine_samples = any(
        _is_path_under_dir(src_resolved, (SAMPLES_DIR / key).resolve())
        for key in scan_keys
    )

    # Якщо файл вже лежить у спільній папці data/samples/clone_voices/ (або спадковій папці рушія),
    # не копіюємо і не позначаємо його як зовнішній.
    if in_engine_samples:
        return str(src_resolved)

    target_lang = _normalize_sample_lang(lang)
    saved_path = str(src_resolved)
    try:
        dest_dir = SAMPLES_DIR / storage_key / target_lang
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src_p.name
        if dest.exists():
            src_size = src_p.stat().st_size
            if dest.stat().st_size == src_size:
                saved_path = str(dest.resolve())
            else:
                stem, suffix = src_p.stem, src_p.suffix
                idx = 2
                while dest.exists():
                    dest = dest_dir / f"{stem} ({idx}){suffix}"
                    idx += 1
                shutil.copy2(src_p, dest)
                saved_path = str(dest.resolve())
        else:
            shutil.copy2(src_p, dest)
            saved_path = str(dest.resolve())
    except Exception:
        saved_path = str(src_resolved)

    saved_norm = str(Path(saved_path).resolve()).lower()
    if not any(str(Path(str(e.get("path", ""))).resolve()).lower() == saved_norm for e in eng_entries if Path(str(e.get("path", ""))).exists()):
        eng_entries.append({"path": saved_path, "lang": target_lang})
        custom_map[storage_key] = eng_entries
        config.set("custom_clone_samples", custom_map)
    return saved_path


def remove_custom_clone_sample(engine_id: str, sample_path: str) -> None:
    """Видаляє доданий користувачем зразок голосу зі списку (та його копію у data/samples/)."""
    clean = str(sample_path or "").strip()
    if not clean:
        return
    try:
        target_norm = str(Path(clean).resolve()).lower()
    except OSError:
        target_norm = clean.lower()

    scan_keys = _sample_scan_keys(engine_id)
    custom_map = dict(config.get("custom_clone_samples", {}) or {})
    for key in scan_keys:
        eng_entries = custom_map.get(key) or []
        if not eng_entries:
            continue
        kept = []
        for entry in eng_entries:
            if not isinstance(entry, dict):
                continue
            p_str = str(entry.get("path", "")).strip()
            if not p_str:
                continue
            try:
                p_norm = str(Path(p_str).resolve()).lower()
            except OSError:
                p_norm = p_str.lower()
            if p_norm != target_norm:
                kept.append(entry)
        custom_map[key] = kept
    config.set("custom_clone_samples", custom_map)

    try:
        p_obj = Path(clean).resolve()
        is_managed = any(
            _is_path_under_dir(p_obj, (SAMPLES_DIR / key).resolve())
            for key in scan_keys
        )
        if is_managed and p_obj.is_file():
            p_obj.unlink(missing_ok=True)
    except Exception:
        pass

    ref_engines = CLONE_ENGINE_IDS if engine_id in CLONE_ENGINE_IDS else (engine_id,)
    for ref_eng in ref_engines:
        ref_key = f"{ref_eng}_ref_path"
        cur_ref = str(config.get(ref_key, "") or "").strip()
        if cur_ref:
            try:
                cur_norm = str(Path(cur_ref).resolve()).lower()
            except OSError:
                cur_norm = cur_ref.lower()
            if cur_norm == target_norm:
                config.set(ref_key, "")


def get_samples_for_engine(
    engine_id: str,
    lang: Optional[str] = None,
    gender: Optional[str] = None,
    own_only: bool = False,
) -> List[Dict[str, str]]:
    """
    Повертає список доступних прикладів для конкретного локального движка,
    відфільтрований за обраною мовою (app_language) та статтю (voice_gender_filter).
    Шукає файли у:
    - data/samples/<engine_id>/*.{wav,ogg,flac,mp3}
    - data/samples/<engine_id>_*.{wav,ogg,flac,mp3}
    - samples/<engine_id>_*.{wav,ogg,flac,mp3}
    Якщо own_only=True — шукає лише власні голоси моделі (без clone_voices і користувацьких зразків).
    """
    use_clone_catalog = (engine_id in CLONE_ENGINE_IDS) and not own_only
    target_lang = str(lang if lang is not None else (config.get("app_language", "uk") or "uk")).strip().lower()
    if target_lang == "ua":
        target_lang = "uk"
    gender_filter = str(
        gender if gender is not None else (config.get("voice_gender_filter", "both") or "both")
    ).strip().lower()
    known_prefixes = {
        "ua": "uk",
        "uk": "uk",
        "ru": "ru",
        "en": "en",
        "pl": "pl",
        "de": "de",
        "fr": "fr",
        "es": "es",
    }

    voice_meta_map: Dict[str, Dict[str, Any]] = {}
    for v in ENGINE_SPECS.get(engine_id, {}).get("voices", []):
        vid = str(v.get("id", "")).strip()
        if not vid:
            continue
        v_stem = vid[len(engine_id) + 1 :] if vid.startswith(f"{engine_id}_") else vid
        voice_meta_map[v_stem.lower()] = v

    result: List[Dict[str, str]] = []
    seen_paths = set()
    seen_clone_keys = set()

    candidates: List[Tuple[Path, str]] = []
    exts = ("*.wav", "*.ogg", "*.flac", "*.mp3")
    scan_keys: Tuple[str, ...] = (engine_id,) if own_only else _sample_scan_keys(engine_id)
    for key in scan_keys:
        sub_dir = SAMPLES_DIR / key
        if sub_dir.exists():
            # 1. Явні підпапки за мовами (наприклад: data/samples/clone_voices/uk/*.wav)
            for child_dir in sorted((p for p in sub_dir.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
                dir_lang = known_prefixes.get(child_dir.name.strip().lower(), "")
                for ext in exts:
                    for p in sorted(child_dir.rglob(ext)):
                        candidates.append((p, dir_lang))
            # 2. Файли безпосередньо в data/samples/<key>/
            for ext in exts:
                for p in sorted(sub_dir.glob(ext)):
                    candidates.append((p, ""))
        if SAMPLES_DIR.exists():
            for ext in exts:
                for p in sorted(SAMPLES_DIR.glob(f"{key}_{ext[1:]}")):
                    candidates.append((p, ""))
        alt_dir = BASE_DIR / "samples"
        if alt_dir.exists():
            for ext in exts:
                for p in sorted(alt_dir.glob(f"{key}_{ext[1:]}")):
                    candidates.append((p, ""))

    custom_paths_norm = set()
    custom_map = config.get("custom_clone_samples", {}) or {}
    if isinstance(custom_map, dict) and not own_only:
        custom_keys = (SHARED_CLONE_SAMPLES_SUBDIR, *CLONE_ENGINE_IDS) if engine_id in CLONE_ENGINE_IDS else scan_keys
        for key in custom_keys:
            for entry in (custom_map.get(key) or []):
                if not isinstance(entry, dict):
                    continue
                p_str = str(entry.get("path", "")).strip()
                if not p_str:
                    continue
                p_obj = Path(p_str)
                if p_obj.is_file():
                    try:
                        p_norm = str(p_obj.resolve()).lower()
                    except OSError:
                        p_norm = str(p_obj).lower()
                    custom_paths_norm.add(p_norm)
                    c_lang = known_prefixes.get(str(entry.get("lang", "")).strip().lower(), "")
                    candidates.append((p_obj, c_lang))

    for wav_path, folder_lang in candidates:
        norm = str(wav_path.resolve()).lower()
        if norm in seen_paths or not wav_path.exists():
            continue
        seen_paths.add(norm)
        raw_stem = wav_path.stem
        if raw_stem.lower().startswith(f"{engine_id}_"):
            raw_stem = raw_stem[len(engine_id) + 1 :]
        sample_lang = folder_lang
        if not sample_lang and "_" in raw_stem:
            prefix = raw_stem.split("_", 1)[0].lower()
            if prefix in known_prefixes:
                sample_lang = known_prefixes[prefix]
        if sample_lang and target_lang not in ("", "all", "multi") and sample_lang != target_lang:
            continue
        if use_clone_catalog:
            clone_dedup_key = (sample_lang, wav_path.name.lower())
            if clone_dedup_key in seen_clone_keys:
                continue
            seen_clone_keys.add(clone_dedup_key)

        meta = voice_meta_map.get(raw_stem.lower(), {})
        v_gender = str(meta.get("gender") or "").strip().lower()
        if not v_gender:
            v_gender = _infer_sample_gender(wav_path, raw_stem)
        if gender_filter in ("female", "male") and v_gender in ("female", "male") and v_gender != gender_filter:
            continue

        display_name = str(meta.get("name") or "").strip()
        if not display_name:
            clean_stem = raw_stem
            if not folder_lang and "_" in raw_stem and raw_stem.split("_", 1)[0].lower() in known_prefixes:
                clean_stem = raw_stem.split("_", 1)[1]
            display_name = (clean_stem or raw_stem).replace("_", " ").strip().title() or wav_path.name
        result.append({
            "name": display_name,
            "file_name": wav_path.name,
            "path": str(wav_path),
            "gender": v_gender,
            "lang": sample_lang,
            "is_custom": "1" if norm in custom_paths_norm else "",
        })

    if use_clone_catalog:
        result.sort(
            key=lambda item: (
                0 if item.get("gender") == "female" else (1 if item.get("gender") == "male" else 2),
                str(item.get("name", "")).lower(),
            )
        )
    return result


CLONE_DEMO_SAMPLES_SUBDIR = "clone_demos"


def get_clone_demo_samples(
    engine_id: str,
    lang: Optional[str] = None,
    gender: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Повертає список демонстраційних прикладів клонування для конкретного рушія
    з data/samples/clone_demos/<engine_id>/<lang>/*.{wav,ogg,flac,mp3}
    (а також безпосередньо з data/samples/clone_demos/<engine_id>/).
    """
    target_lang = str(lang if lang is not None else (config.get("app_language", "uk") or "uk")).strip().lower()
    if target_lang == "ua":
        target_lang = "uk"
    gender_filter = str(
        gender if gender is not None else (config.get("voice_gender_filter", "both") or "both")
    ).strip().lower()
    known_prefixes = {
        "ua": "uk",
        "uk": "uk",
        "ru": "ru",
        "en": "en",
        "pl": "pl",
        "de": "de",
        "fr": "fr",
        "es": "es",
    }

    exts = ("*.wav", "*.ogg", "*.flac", "*.mp3")
    candidates: List[Tuple[Path, str]] = []
    for base_root in (SAMPLES_DIR, BASE_DIR / "samples"):
        sub_dir = base_root / CLONE_DEMO_SAMPLES_SUBDIR / engine_id
        if not sub_dir.exists():
            continue
        for child_dir in sorted((p for p in sub_dir.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
            dir_lang = known_prefixes.get(child_dir.name.strip().lower(), "")
            for ext in exts:
                for p in sorted(child_dir.rglob(ext)):
                    candidates.append((p, dir_lang))
        for ext in exts:
            for p in sorted(sub_dir.glob(ext)):
                candidates.append((p, ""))

    # Build lookup map of reference clips in clone_voices/<lang>/<stem>.* for automatic pairing
    ref_lookup: Dict[Tuple[str, str], Path] = {}
    ref_lookup_any: Dict[str, Path] = {}
    for base_root in (SAMPLES_DIR, BASE_DIR / "samples"):
        cv_dir = base_root / SHARED_CLONE_SAMPLES_SUBDIR
        if not cv_dir.exists():
            continue
        for child_dir in sorted((p for p in cv_dir.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
            rlang = known_prefixes.get(child_dir.name.strip().lower(), "")
            for ext in exts:
                for rp in sorted(child_dir.rglob(ext)):
                    key_stem = rp.stem.strip().lower()
                    if rlang:
                        ref_lookup.setdefault((rlang, key_stem), rp)
                    ref_lookup_any.setdefault(key_stem, rp)
        for ext in exts:
            for rp in sorted(cv_dir.glob(ext)):
                ref_lookup_any.setdefault(rp.stem.strip().lower(), rp)

    result: List[Dict[str, str]] = []
    seen_paths = set()
    for wav_path, folder_lang in candidates:
        # Skip explicit companion *_ref.* files (they are paired with the main result clip)
        if wav_path.stem.lower().endswith("_ref"):
            continue
        try:
            norm = str(wav_path.resolve()).lower()
        except OSError:
            norm = str(wav_path).lower()
        if norm in seen_paths or not wav_path.exists():
            continue
        seen_paths.add(norm)
        raw_stem = wav_path.stem
        if raw_stem.lower().startswith(f"{engine_id}_"):
            raw_stem = raw_stem[len(engine_id) + 1 :]
        sample_lang = folder_lang
        if not sample_lang and "_" in raw_stem:
            prefix = raw_stem.split("_", 1)[0].lower()
            if prefix in known_prefixes:
                sample_lang = known_prefixes[prefix]
        if sample_lang and target_lang not in ("", "all", "multi") and sample_lang != target_lang:
            continue

        v_gender = _infer_sample_gender(wav_path, raw_stem)
        if gender_filter in ("female", "male") and v_gender in ("female", "male") and v_gender != gender_filter:
            continue

        # Pair with companion <stem>_ref.* in the same folder first, then matching clone_voices/<lang>/<stem>.*
        ref_path_str = ""
        for ext in (".ogg", ".mp3", ".wav", ".flac"):
            local_ref = wav_path.with_name(f"{wav_path.stem}_ref{ext}")
            if local_ref.is_file():
                ref_path_str = str(local_ref)
                break
        if not ref_path_str:
            stem_key = wav_path.stem.strip().lower()
            matched_ref = ref_lookup.get((sample_lang, stem_key)) if sample_lang else None
            if matched_ref is None:
                matched_ref = ref_lookup_any.get(stem_key)
            if matched_ref is not None and matched_ref.is_file():
                ref_path_str = str(matched_ref)

        clean_stem = raw_stem
        if not folder_lang and "_" in raw_stem and raw_stem.split("_", 1)[0].lower() in known_prefixes:
            clean_stem = raw_stem.split("_", 1)[1]
        display_name = (clean_stem or raw_stem).replace("_", " ").strip().title() or wav_path.name
        result.append({
            "name": display_name,
            "file_name": wav_path.name,
            "path": str(wav_path),
            "ref_path": ref_path_str,
            "gender": v_gender,
            "lang": sample_lang,
            "is_custom": "",
        })

    result.sort(
        key=lambda item: (
            0 if item.get("gender") == "female" else (1 if item.get("gender") == "male" else 2),
            str(item.get("name", "")).lower(),
        )
    )
    return result


def split_into_sentences_punct(text: str, min_len: int = 25) -> List[str]:
    """
    Розумно розбиває текст на речення лише за кінцевими знаками пунктуації (., !, ?),
    без розривів по комах, об'єднуючи занадто короткі (< min_len) перші або останні речення.
    """
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return []
    raw = [p.strip() for p in re.split(r"(?<=[.!?])\s+", clean) if p and p.strip()]
    if not raw:
        return [clean]

    merged: List[str] = []
    buf = ""
    for s in raw:
        buf = f"{buf} {s}" if buf else s
        if len(buf) >= min_len:
            merged.append(buf)
            buf = ""
    if buf:
        if merged:
            merged[-1] += " " + buf
        else:
            merged.append(buf)
    return merged


class TTSEngineManager(QObject):
    """
    Головний менеджер TTS:
    - Асинхронна черга озвучення (не блокує інтерфейс і гри)
    - Кешування завантажених у пам'ять моделей
    - Завантаження / видалення локальних моделей в 1 клік
    """

    speech_started = Signal(str, float)              # (text, duration_sec)
    speech_finished = Signal(str)                    # (text)
    phrase_spoken = Signal(str)                      # (повна фраза для дублювання в чат, 1 раз за репліку)
    status_changed = Signal(str)                     # Рядок статусу для UI
    error_occurred = Signal(str)                     # Повідомлення про помилку
    download_progress = Signal(str, int, str)        # (engine_id, percent, status_text)
    download_failed = Signal(str, str)               # (engine_id, error_text)
    download_cancelled = Signal(str)                 # (engine_id)
    model_status_changed = Signal(str, bool)         # (engine_id, is_installed)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._queue: queue.Queue[Optional[Tuple[Any, ...]]] = queue.Queue()
        self._running = True
        self._cancel_batch = False
        self._last_spoken_text: str = ""
        self._last_spoken_audio_file: str = ""
        self._last_spoken_skip_spellcheck: bool = False
        self._last_spoken_mirror_twitch: bool = True
        self._last_corrected_pair: Optional[Tuple[str, str]] = None

        # Розумний двошаровий кеш (L1 RAM + L2 диск із закріпленням частих фраз)
        self._audio_cache = TTSAudioCache(TTS_CACHE_DIR)

        # Кеш ініціалізованих моделей у пам'яті
        self._piper_cache: Dict[str, Any] = {}
        self._silero_cache: Dict[str, Any] = {}
        self._silero_importers: Dict[str, Any] = {}
        self._holos_model: Any = None
        self._holos_stress: Any = None
        self._holos_stanza_stress: Any = None
        self._holos_stanza_started: bool = False
        self._holos_first_inference_logged = False
        self._teratts_model: Any = None
        self._teratts_accent: Any = None
        self._edge_dynamic_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._piper_dynamic_cache: Dict[str, Dict[str, Any]] = {}
        self._piper_dynamic_failed_ts: float = 0.0
        self._model_lock = threading.Lock()
        self._import_lock = _GLOBAL_IMPORT_LOCK
        self._proxy_cache: Dict[str, Tuple[float, List[str]]] = {}
        self._proxy_meta: Dict[str, Dict[str, Any]] = {}
        self._url_probe_meta: Dict[str, Dict[str, Any]] = {}
        self._bad_proxies: Dict[str, float] = {}
        self._proxy_lock = threading.Lock()
        self._warmed_models: set = set()
        self._warmup_lock = threading.Lock()
        self._warmup_exec_lock = threading.Lock()
        self._warmup_timer: Optional[threading.Timer] = None
        self._is_synthesizing: bool = False
        self._current_is_viewer_chat: bool = False
        self._current_chat_user: str = ""
        self._current_chat_msg_id: str = ""
        self._active_downloads: int = 0
        self._download_cancel_flags: Dict[str, threading.Event] = {}
        self._download_lock = threading.RLock()
        self._downloading_engines: set[str] = set()
        self._download_progress_state: Dict[str, Tuple[int, str]] = {}

        # Постійна HTTP/TLS-сесія (Keep-Alive) для швидкого синтезу без повторного handshake
        self._http_session = requests.Session()
        from requests.adapters import HTTPAdapter
        _adapter = HTTPAdapter(pool_connections=8, pool_maxsize=16, max_retries=1)
        self._http_session.mount("https://", _adapter)
        self._http_session.mount("http://", _adapter)
        threading.Thread(target=self._prewarm_http_connections, daemon=True, name="TTSPrewarm").start()

        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="TTSWorkerThread")
        self._worker_thread.start()

    def _prewarm_http_connections(self) -> None:
        """Прогріває TLS Keep-Alive з'єднання для активних онлайн-провайдерів."""
        hosts = {
            "google": "https://texttospeech.googleapis.com",
            "fish_audio": "https://api.fish.audio",
            "elevenlabs": "https://api.elevenlabs.io",
            "openai": "https://api.openai.com",
            "gemini_tts": "https://generativelanguage.googleapis.com",
        }
        try:
            proxies = get_active_api_proxies()
            for item in config.get("online_items", []):
                prov = str(item.get("provider", "")).strip()
                if prov in hosts and str(item.get("api_key", "")).strip():
                    try:
                        self._http_session.head(hosts[prov], proxies=proxies, timeout=4)
                    except Exception:
                        pass
        except Exception:
            pass

    @staticmethod
    def _get_system_resources_for_warmup() -> Tuple[int, bool, float]:
        """
        Повертає (вільна_RAM_МБ, низький_заряд_або_енергозбереження, завантаження_CPU_%).
        Використовує лише Win32 API без важких зовнішніх залежностей.
        """
        avail_mb = 4096
        low_power = False
        cpu_pct = 0.0
        try:
            import ctypes
            from ctypes import wintypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("sullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            class SYSTEM_POWER_STATUS(ctypes.Structure):
                _fields_ = [
                    ("ACLineStatus", ctypes.c_ubyte),
                    ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte),
                    ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_uint32),
                    ("BatteryFullLifeTime", ctypes.c_uint32),
                ]

            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
                avail_mb = int(mem.ullAvailPhys // (1024 * 1024))

            pwr = SYSTEM_POWER_STATUS()
            if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(pwr)):
                # SystemStatusFlag == 1 означає увімкнений режим економії заряду Windows
                battery_low = (pwr.ACLineStatus == 0 and pwr.BatteryLifePercent <= 20 and pwr.BatteryLifePercent != 255)
                low_power = bool(pwr.SystemStatusFlag == 1 or battery_low)

            def _read_times():
                idle, kern, usr = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
                if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(usr)):
                    return None
                to_int = lambda ft: (ft.dwHighDateTime << 32) | ft.dwLowDateTime
                return to_int(idle), to_int(kern), to_int(usr)

            t1 = _read_times()
            if t1 is not None:
                time.sleep(0.08)
                t2 = _read_times()
                if t2 is not None:
                    d_idle = max(0, t2[0] - t1[0])
                    d_total = max(1, (t2[1] - t1[1]) + (t2[2] - t1[2]))
                    cpu_pct = max(0.0, min(100.0, (1.0 - d_idle / float(d_total)) * 100.0))
        except Exception:
            pass
        return avail_mb, low_power, cpu_pct

    def _resolve_local_target_for(self, engine_id: str, voice_id: str) -> Optional[Tuple[str, str, str]]:
        eng = str(engine_id or "").strip()
        if eng not in ("piper", "silero", "holos", "teratts", "kokoro_82m", "soprano_1_1", "omnivoice", "zonos3_mini", "qwen3_tts"):
            return None
        if not self.is_engine_installed(eng):
            return None

        v_id = str(voice_id or "").strip()
        voices = self.get_voices_for_engine(eng)
        valid_ids = [str(v.get("id", "")) for v in voices if v.get("id")]
        if v_id not in valid_ids:
            v_id = valid_ids[0] if valid_ids else ""
        if not v_id:
            return None

        if eng == "piper":
            all_v = list(ENGINE_SPECS["piper"]["voices"])
            v_cfg = next((v for v in all_v if v["id"] == v_id), None)
            if v_cfg is None:
                dyn_v, _ = self._get_dynamic_piper_for_lang(self._current_lang())
                if dyn_v:
                    all_v.extend(dyn_v)
                v_cfg = next((v for v in all_v if v["id"] == v_id), None)
            if not v_cfg:
                return None
            p = ENGINES_DIR / "piper" / v_cfg["onnx"]
            if not p.exists():
                return None
            cache_key = str(p)
        elif eng == "silero":
            v_cfg = next((v for v in ENGINE_SPECS["silero"]["voices"] if v["id"] == v_id), None)
            if not v_cfg:
                return None
            p = ENGINES_DIR / "silero" / v_cfg["model"]
            if not p.exists():
                return None
            cache_key = str(p)
        elif eng == "kokoro_82m":
            from core.llama_tts import resolve_kokoro_device

            cache_key = f"kokoro_82m:{resolve_kokoro_device()}"
        elif eng == "soprano_1_1":
            from core.llama_tts import resolve_soprano_device

            cache_key = f"soprano_1_1:{resolve_soprano_device()}"
        elif eng == "omnivoice":
            from core.omnivoice_tts import get_omnivoice_cache_signature

            cache_key = f"omnivoice:{get_omnivoice_cache_signature()}"
        elif eng == "qwen3_tts":
            ref_p = str(config.get("qwen3_tts_ref_path", "") or "").strip()
            cache_key = f"qwen3_tts:{ref_p}"
        elif eng == "zonos3_mini":
            ref_p = str(config.get("zonos3_mini_ref_path", "") or "").strip()
            cache_key = f"zonos3_mini:{v_id}:{ref_p}"
        else:
            cache_key = eng

        return eng, v_id, cache_key

    def _resolve_local_warmup_target(self) -> Optional[Tuple[str, str, str]]:
        """
        Визначає, чи активний локальний TTS-движок, чи він встановлений,
        та повертає (engine_id, voice_id, cache_key), інакше None.
        """
        engine_id = str(config.get("active_engine", "") or "").strip()
        voice_id = str(config.get("active_voice", "") or "").strip()
        if not engine_id:
            available = self.get_available_active_engines()
            if available:
                engine_id = str(available[0].get("id", "") or "").strip()
                voices = available[0].get("voices") or []
                if not voice_id and voices:
                    voice_id = str(voices[0].get("id", "") or "").strip()
        return self._resolve_local_target_for(engine_id, voice_id)

    def _resolve_local_warmup_targets(self) -> List[Tuple[str, str, str, str]]:
        """
        Повертає пріоритезований список локальних TTS для фонового прогріву:
        (engine_id, voice_id, cache_key, role) для основного, запасного та чат-рушіїв.
        """
        targets: List[Tuple[str, str, str, str]] = []
        seen: set[Tuple[str, str]] = set()

        prim = self._resolve_local_warmup_target()
        if prim is not None:
            targets.append((prim[0], prim[1], prim[2], "primary"))
            seen.add((prim[0], prim[2]))

        fb_eng = str(config.get("fallback_engine", "") or "").strip()
        fb_voice = str(config.get("fallback_voice", "") or "").strip()
        fb = self._resolve_local_target_for(fb_eng, fb_voice)
        if fb is not None and (fb[0], fb[2]) not in seen:
            targets.append((fb[0], fb[1], fb[2], "fallback"))
            seen.add((fb[0], fb[2]))

        if bool(config.get("twitch_read_viewers", False)):
            tw_eng = str(config.get("twitch_chat_tts_engine", "") or "").strip()
            tw_voice = str(config.get("twitch_chat_tts_voice", "") or "").strip()
            tw = self._resolve_local_target_for(tw_eng, tw_voice)
            if tw is not None and (tw[0], tw[2]) not in seen:
                targets.append((tw[0], tw[1], tw[2], "twitch"))
                seen.add((tw[0], tw[2]))

        return targets

    @staticmethod
    def _dsp_needs_prewarm(engine_id: str, voice_id: str) -> bool:
        dsp = get_engine_dsp_settings(engine_id, voice_id)
        any_fx = (
            bool(dsp.get("denoise_enabled", False))
            or bool(dsp.get("deess_enabled", False))
            or bool(dsp.get("speed_enabled", False))
            or bool(dsp.get("pitch_enabled", False))
            or bool(dsp.get("gate_enabled", False))
        )
        if any_fx and ("scipy.signal" not in sys.modules or "scipy.ndimage" not in sys.modules):
            return True
        if bool(dsp.get("denoise_enabled", False)) and "noisereduce" not in sys.modules:
            return True
        return False

    @staticmethod
    def _prewarm_dsp_libs(engine_id: str, voice_id: str) -> None:
        dsp = get_engine_dsp_settings(engine_id, voice_id)
        any_fx = (
            bool(dsp.get("denoise_enabled", False))
            or bool(dsp.get("deess_enabled", False))
            or bool(dsp.get("speed_enabled", False))
            or bool(dsp.get("pitch_enabled", False))
            or bool(dsp.get("gate_enabled", False))
        )
        try:
            with _GLOBAL_IMPORT_LOCK:
                if any_fx:
                    import scipy.ndimage  # noqa: F401
                    import scipy.signal  # noqa: F401
                if bool(dsp.get("denoise_enabled", False)):
                    import noisereduce  # noqa: F401
        except Exception:
            pass

    def _resolve_online_target_for(self, engine_id: str, voice_id: str) -> Optional[Tuple[str, str, str]]:
        eng = str(engine_id or "").strip()
        if not eng.startswith("online:"):
            return None
        item_id = eng.split(":", 1)[1]
        items = list(config.get("online_items", []) or [])
        item = next((x for x in items if str(x.get("id", "")) == item_id), None)
        if not item or not bool(item.get("enabled", True)):
            return None
        provider = str(item.get("provider", "") or "").strip().lower()
        if provider == "edge" and not bool(item.get("installed", False)):
            return None
        if provider == "silero_bot":
            from core.telegram_silero import is_telethon_installed
            if not (bool(item.get("installed", False)) or is_telethon_installed()):
                return None
        return eng, str(voice_id or "").strip(), provider

    def _resolve_online_warmup_targets(self) -> List[Tuple[str, str, str, str]]:
        """
        Повертає пріоритезований список активних онлайн-TTS для легкого фонового прогріву:
        (engine_id, voice_id, provider, role).
        """
        targets: List[Tuple[str, str, str, str]] = []
        seen: set[Tuple[str, str]] = set()

        prim_eng = str(config.get("active_engine", "") or "").strip()
        prim_voice = str(config.get("active_voice", "") or "").strip()
        if not prim_eng:
            available = self.get_available_active_engines()
            if available:
                prim_eng = str(available[0].get("id", "") or "").strip()
                voices = available[0].get("voices") or []
                if not prim_voice and voices:
                    prim_voice = str(voices[0].get("id", "") or "").strip()
        prim = self._resolve_online_target_for(prim_eng, prim_voice)
        if prim is not None:
            targets.append((prim[0], prim[1], prim[2], "primary"))
            seen.add((prim[0], prim[1]))

        fb_eng = str(config.get("fallback_engine", "") or "").strip()
        fb_voice = str(config.get("fallback_voice", "") or "").strip()
        fb = self._resolve_online_target_for(fb_eng, fb_voice)
        if fb is not None and (fb[0], fb[1]) not in seen:
            targets.append((fb[0], fb[1], fb[2], "fallback"))
            seen.add((fb[0], fb[1]))

        if bool(config.get("twitch_read_viewers", False)):
            tw_eng = str(config.get("twitch_chat_tts_engine", "") or "").strip()
            tw_voice = str(config.get("twitch_chat_tts_voice", "") or "").strip()
            tw = self._resolve_online_target_for(tw_eng, tw_voice)
            if tw is not None and (tw[0], tw[1]) not in seen:
                targets.append((tw[0], tw[1], tw[2], "twitch"))
                seen.add((tw[0], tw[1]))

        return targets

    @staticmethod
    def _prewarm_online_provider(provider: str) -> None:
        """
        Прогріває клієнтські модулі, проксі-конектор та DNS для онлайн-провайдера
        без витрати платних символів/лімітів.
        """
        import socket

        try:
            get_active_api_proxies()
        except Exception:
            pass

        host_map = {
            "edge": "speech.platform.bing.com",
            "elevenlabs": "api.elevenlabs.io",
            "openai": "api.openai.com",
            "google": "texttospeech.googleapis.com",
            "gemini_tts": "generativelanguage.googleapis.com",
            "fish_audio": "api.fish.audio",
            "fish": "api.fish.audio",
        }
        if provider == "edge":
            with _GLOBAL_IMPORT_LOCK:
                import edge_tts  # noqa: F401
            try:
                get_active_socks5_url()
            except Exception:
                pass
        elif provider == "silero_bot":
            try:
                with _GLOBAL_IMPORT_LOCK:
                    import telethon  # noqa: F401
            except Exception:
                pass

        host = host_map.get(provider)
        if host:
            try:
                socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            except Exception:
                pass

    def invalidate_warmup(self, engine_id: str) -> None:
        """Скидає позначку прогріву для вказаного движка (наприклад, після зміни GPU/CPU)."""
        self._warmed_models = {item for item in self._warmed_models if item[0] != engine_id}

    def schedule_smart_warmup(self, delay_sec: float = 2.5, reason: str = "startup", _retry: int = 0) -> None:
        """
        Планує розумний фоновий прогрів TTS-рушіїв (локальних та онлайн) і DSP-бібліотек
        з дебаунсом: пропускає вже прогріті компоненти, високе навантаження на ПК, нестачу ОЗП
        або активний синтез/завантаження.
        """
        with self._warmup_lock:
            if self._warmup_timer is not None:
                try:
                    self._warmup_timer.cancel()
                except Exception:
                    pass
                self._warmup_timer = None
            if not self._running:
                return
            t = threading.Timer(max(0.05, float(delay_sec)), self._run_smart_warmup, args=(reason, _retry))
            t.daemon = True
            t.name = "TTSWarmupTimer"
            self._warmup_timer = t
            t.start()

    def _run_smart_warmup(self, reason: str, retry: int) -> None:
        if not self._warmup_exec_lock.acquire(blocking=False):
            if retry < 6 and self._running:
                self.schedule_smart_warmup(delay_sec=3.0, reason=f"{reason}:overlap", _retry=retry + 1)
            return
        try:
            self._run_smart_warmup_locked(reason, retry)
        finally:
            self._warmup_exec_lock.release()

    def _run_smart_warmup_locked(self, reason: str, retry: int) -> None:
        targets = self._resolve_local_warmup_targets()
        online_targets = self._resolve_online_warmup_targets()
        _llama_eids = ("qwen3_tts", "zonos3_mini", "kokoro_82m", "soprano_1_1")
        _kokoro_uses_cpu = False
        try:
            from core.llama_tts import resolve_kokoro_device
            _kokoro_uses_cpu = resolve_kokoro_device() == "cpu"
        except Exception:
            pass
        _active_llama_targets = [
            item for item in targets
            if item[0] in _llama_eids and not (item[0] == "kokoro_82m" and _kokoro_uses_cpu)
        ]
        _primary_uses_llama_server = any(item[3] == "primary" for item in _active_llama_targets)
        if not _active_llama_targets:
            try:
                from core.llama_tts import is_qwen3_server_running, shutdown_llama_tts

                if is_qwen3_server_running():
                    shutdown_llama_tts()
                self._warmed_models = {
                    item for item in self._warmed_models
                    if item[0] not in _llama_eids or (item[0] == "kokoro_82m" and _kokoro_uses_cpu)
                }
            except Exception:
                pass
        if not any(item[0] == "omnivoice" for item in targets):
            try:
                from core.omnivoice_tts import is_omnivoice_server_running, shutdown_omnivoice_tts

                if is_omnivoice_server_running():
                    shutdown_omnivoice_tts()
                self._warmed_models = {item for item in self._warmed_models if item[0] != "omnivoice"}
            except Exception:
                pass
        pending_targets = [
            item for item in targets if (item[0], item[2]) not in self._warmed_models
        ]
        pending_dsp_local = [
            item for item in targets
            if (item[0], item[2]) in self._warmed_models and self._dsp_needs_prewarm(item[0], item[1])
        ]
        pending_online = [
            item for item in online_targets
            if ("online", f"prov:{item[2]}") not in self._warmed_models or self._dsp_needs_prewarm(item[0], item[1])
        ]

        if not pending_targets and not pending_dsp_local and not pending_online:
            return

        if (
            not self._running
            or self._is_synthesizing
            or self._active_downloads > 0
            or audio_router.is_playing
            or not self._queue.empty()
        ):
            if retry < 6:
                self.schedule_smart_warmup(delay_sec=5.0, reason=f"{reason}:busy", _retry=retry + 1)
            return

        avail_mb, low_power, cpu_pct = self._get_system_resources_for_warmup()
        if cpu_pct > 92.0:
            if retry < 6:
                print(f"[TTSWarmup] CPU зайнятий ({cpu_pct:.0f}%), відкладаємо прогрів...", flush=True)
                self.schedule_smart_warmup(delay_sec=5.0, reason=f"{reason}:cpu", _retry=retry + 1)
            else:
                print(f"[TTSWarmup] Пропущено прогрів: високе завантаження CPU ({cpu_pct:.0f}%)", flush=True)
            return

        emitted_status = False
        llama_server_occupied = False
        try:
            for engine_id, voice_id, _cache_key, _role in pending_dsp_local:
                if not self._running or self._is_synthesizing or self._active_downloads > 0 or not self._queue.empty():
                    break
                self._prewarm_dsp_libs(engine_id, voice_id)

            for engine_id, voice_id, provider, role in pending_online:
                if (
                    not self._running
                    or self._is_synthesizing
                    or self._active_downloads > 0
                    or not self._queue.empty()
                ):
                    if retry < 6:
                        self.schedule_smart_warmup(delay_sec=5.0, reason=f"{reason}:busy", _retry=retry + 1)
                    return
                if role != "primary" and low_power:
                    continue
                t0 = time.perf_counter()
                try:
                    prov_key = ("online", f"prov:{provider}")
                    if prov_key not in self._warmed_models:
                        self._prewarm_online_provider(provider)
                        self._warmed_models.add(prov_key)
                    self._prewarm_dsp_libs(engine_id, voice_id)
                    dt_ms = int((time.perf_counter() - t0) * 1000)
                    print(f"[TTSWarmup] Онлайн-рушій {provider} ({voice_id}) [{role}] прогрітий за {dt_ms} мс [{reason}]", flush=True)
                except Exception as exc:
                    print(f"[TTSWarmup] Попередній прогрів {provider} пропущено: {exc}", file=sys.stderr, flush=True)

            for engine_id, voice_id, cache_key, role in pending_targets:
                if (
                    not self._running
                    or self._is_synthesizing
                    or self._active_downloads > 0
                    or not self._queue.empty()
                ):
                    if retry < 6:
                        self.schedule_smart_warmup(delay_sec=5.0, reason=f"{reason}:busy", _retry=retry + 1)
                    return

                uses_llama_server = engine_id in _llama_eids and not (engine_id == "kokoro_82m" and _kokoro_uses_cpu)
                # Єдиний процес llama-server.exe може тримати у VRAM лише одну GGUF-модель:
                # ніколи не витісняємо основний (primary) llama-рушій прогрівом запасного/чат-рушія.
                if uses_llama_server and role != "primary" and (_primary_uses_llama_server or llama_server_occupied):
                    continue

                is_heavy = engine_id in ("silero", "holos", "teratts", "omnivoice", "qwen3_tts", "zonos3_mini")
                min_ram_mb = 2048 if engine_id in ("omnivoice", "qwen3_tts", "zonos3_mini") else (1024 if is_heavy else 350)
                req_ram = min_ram_mb if role == "primary" else (min_ram_mb + 1024)
                if role != "primary" and low_power:
                    continue
                if avail_mb < req_ram:
                    print(f"[TTSWarmup] Пропущено прогрів {engine_id} ({role}): мало вільної ОЗП ({avail_mb} MB < {req_ram} MB)", flush=True)
                    continue

                title = ENGINE_SPECS.get(engine_id, {}).get("title", engine_id)
                if is_heavy and not self._is_synthesizing and self._active_downloads == 0:
                    emitted_status = True
                    self.status_changed.emit(tr("tts.status.warming_up", "⏳ Підготовка голосу ({title})...", title=title))

                t0 = time.perf_counter()
                try:
                    voices = ENGINE_SPECS.get(engine_id, {}).get("voices", [])
                    v_entry = next((v for v in voices if str(v.get("id")) == voice_id), None)
                    v_lang = str((v_entry or {}).get("lang") or self._current_lang() or "uk").strip().lower()
                    warmed_ok = True
                    if engine_id == "qwen3_tts":
                        from core.llama_tts import warmup_qwen3_tts
                        warmed_ok = bool(warmup_qwen3_tts())
                    elif engine_id == "omnivoice":
                        from core.omnivoice_tts import warmup_omnivoice_tts
                        warmed_ok = bool(warmup_omnivoice_tts(lang=v_lang))
                    elif uses_llama_server:
                        from core.llama_tts import warmup_llama_tts
                        spk = str((v_entry or {}).get("speaker") or "")
                        warmed_ok = bool(warmup_llama_tts(engine_id, lang=v_lang, voice=spk))
                    else:
                        probe_text = "a." if engine_id == "kokoro_82m" else ("а." if v_lang in ("uk", "ru") else "a.")
                        self._synthesize_dispatch(engine_id, voice_id, probe_text, 1.0)
                    if not warmed_ok:
                        print(f"[TTSWarmup] Попередній прогрів {engine_id} ({voice_id}) [{role}] не завершився [{reason}]", file=sys.stderr, flush=True)
                        continue
                    if uses_llama_server:
                        llama_server_occupied = True
                    self._prewarm_dsp_libs(engine_id, voice_id)
                    self._warmed_models.add((engine_id, cache_key))
                    dt_ms = int((time.perf_counter() - t0) * 1000)
                    print(f"[TTSWarmup] Модель {engine_id} ({voice_id}) [{role}] прогріта за {dt_ms} мс [{reason}]", flush=True)
                    # Оновлюємо оцінку вільної ОЗП перед вторинним прогрівом
                    avail_mb, _, _ = self._get_system_resources_for_warmup()
                except Exception as exc:
                    print(f"[TTSWarmup] Попередній прогрів {engine_id} пропущено: {exc}", file=sys.stderr, flush=True)
        finally:
            if emitted_status and self._running and not self._is_synthesizing and self._active_downloads == 0:
                self.status_changed.emit(tr("ui_113", "Система готова"))

    @property
    def last_spoken_text(self) -> str:
        return self._last_spoken_text

    # ------------------------------------------------------------------
    # Перевірка статусу встановлення та формування списку активних TTS
    # ------------------------------------------------------------------

    @staticmethod
    def _current_lang() -> str:
        return str(config.get("app_language", "uk")).strip().lower() or "uk"

    def _get_dynamic_piper_for_lang(self, target_lang: str) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
        """
        Сканує каталог rhasspy/piper-voices (voices.json) на Hugging Face для будь-якої мови,
        якої немає в базовому списку ENGINE_SPECS['piper'], і кешує в data/piper_voices_cache.json.
        Повертає (voices_list, files_list).
        """
        lang = str(target_lang or "").strip().lower()
        if not lang:
            return [], []

        if lang in self._piper_dynamic_cache:
            entry = self._piper_dynamic_cache[lang]
            return list(entry.get("voices", [])), list(entry.get("files", []))

        cache_file = DATA_DIR / "piper_voices_cache.json"
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                if isinstance(cached, dict):
                    for k, v in cached.items():
                        if isinstance(v, dict):
                            self._piper_dynamic_cache[str(k).lower()] = v
                    if lang in self._piper_dynamic_cache:
                        entry = self._piper_dynamic_cache[lang]
                        return list(entry.get("voices", [])), [tuple(x) for x in entry.get("files", [])]
            except Exception:
                pass

        if (time.monotonic() - self._piper_dynamic_failed_ts) < 600.0:
            return [], []

        try:
            base_urls = ["https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json"]
            if bool(config.get("use_download_bypass", True)):
                base_urls.append("https://hf-mirror.com/rhasspy/piper-voices/resolve/main/voices.json")
            resp = None
            for url in base_urls:
                try:
                    cand_resp = self._http_session.get(url, proxies=get_active_api_proxies(), timeout=4.5)
                    if cand_resp.status_code == 200:
                        resp = cand_resp
                        break
                except Exception:
                    continue
            if resp is None or resp.status_code != 200:
                self._piper_dynamic_failed_ts = time.monotonic()
                return [], []
            raw_catalog = resp.json()
            if not isinstance(raw_catalog, dict):
                return [], []

            quality_rank = {"high": 0, "medium": 1, "low": 2, "x_low": 3}
            by_lang: Dict[str, List[Tuple[int, str, Dict[str, Any]]]] = {}
            for key, info in raw_catalog.items():
                if not isinstance(info, dict):
                    continue
                fam = str((info.get("language") or {}).get("family", "")).strip().lower()
                if not fam:
                    continue
                q = str(info.get("quality", "medium")).strip().lower()
                by_lang.setdefault(fam, []).append((quality_rank.get(q, 2), str(key), info))

            built_cache: Dict[str, Dict[str, Any]] = {}
            for fam, entries in by_lang.items():
                entries.sort(key=lambda t: (t[0], t[1]))
                picked = entries[:2]  # до 2 найкращих моделей на мову
                v_list: List[Dict[str, Any]] = []
                f_list: List[List[str]] = []
                seen_files = set()
                for _, model_key, info in picked:
                    files_map = info.get("files") or {}
                    onnx_rel = next((p for p in files_map if p.endswith(".onnx")), "")
                    json_rel = next((p for p in files_map if p.endswith(".onnx.json")), "")
                    if not onnx_rel or not json_rel:
                        continue
                    onnx_name = Path(onnx_rel).name
                    json_name = Path(json_rel).name
                    for rel_p, fname in ((onnx_rel, onnx_name), (json_rel, json_name)):
                        if fname not in seen_files:
                            seen_files.add(fname)
                            f_list.append([fname, f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{rel_p}"])

                    pretty_name = str(info.get("name", model_key)).replace("_", " ").title()
                    q_label = str(info.get("quality", "medium")).capitalize()
                    num_spk = int(info.get("num_speakers") or 1)
                    if num_spk > 1:
                        for spk_idx in range(min(num_spk, 4)):
                            v_list.append({
                                "id": f"piper_{fam}_{model_key}_s{spk_idx}",
                                "name": f"{pretty_name} #{spk_idx + 1} ({q_label})",
                                "gender": None,
                                "lang": fam,
                                "onnx": onnx_name,
                                "speaker": spk_idx,
                            })
                    else:
                        v_list.append({
                            "id": f"piper_{fam}_{model_key}",
                            "name": f"{pretty_name} ({q_label})",
                            "gender": None,
                            "lang": fam,
                            "onnx": onnx_name,
                            "speaker": None,
                        })
                if v_list and f_list:
                    built_cache[fam] = {"voices": v_list, "files": f_list}

            if built_cache:
                self._piper_dynamic_cache.update(built_cache)
                try:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(json.dumps(built_cache, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass
                entry = self._piper_dynamic_cache.get(lang, {})
                return list(entry.get("voices", [])), [tuple(x) for x in entry.get("files", [])]
        except Exception:
            self._piper_dynamic_failed_ts = time.monotonic()
        return [], []

    def get_files_for_engine(self, engine_id: str, lang: Optional[str] = None) -> List[Tuple[str, str]]:
        """Повертає список файлів (fname, url) локального движка тільки для обраної мови."""
        spec = ENGINE_SPECS.get(engine_id, {})
        raw_files = spec.get("files", [])
        target_lang = (lang or self._current_lang()).strip().lower()
        filtered: List[Tuple[str, str]] = []
        for entry in raw_files:
            if len(entry) >= 3:
                fname, url, f_lang = entry[0], entry[1], str(entry[2]).lower()
                if f_lang in (target_lang, "multi"):
                    filtered.append((fname, url))
            else:
                filtered.append((entry[0], entry[1]))
        if engine_id == "piper" and not filtered:
            _, dyn_files = self._get_dynamic_piper_for_lang(target_lang)
            return dyn_files
        return filtered

    def _filter_voices_by_lang(self, voices: List[Dict[str, Any]], lang: Optional[str] = None) -> List[Dict[str, Any]]:
        target_lang = (lang or self._current_lang()).strip().lower()
        res: List[Dict[str, Any]] = []
        for v in voices:
            langs_list = [str(x).lower() for x in (v.get("languages") or [])]
            if langs_list:
                if target_lang in langs_list or "multi" in langs_list:
                    res.append(v)
            elif str(v.get("lang", "multi")).lower() in (target_lang, "multi"):
                res.append(v)
        return res

    def _release_silero_cache_locked(self) -> None:
        """Звільняє кеш моделей Silero та закриває дескриптори zip_reader (PyTorchFileReader) на Windows."""
        for imp in self._silero_importers.values():
            try:
                if hasattr(imp, "zip_reader"):
                    imp.zip_reader = None
            except Exception:
                pass
        self._silero_importers.clear()
        self._silero_cache.clear()

    def clear_silero_cache(self) -> None:
        """Потокобезпечно очищає кеш Silero і звільняє відкриті .pt-файли."""
        with self._model_lock:
            self._release_silero_cache_locked()
        gc.collect()

    def cleanup_unused_language_files(self, keep_lang: Optional[str] = None) -> None:
        """Видаляє з папки data/engines файли моделей, які не відносяться до обраної мови."""
        target_lang = (keep_lang or self._current_lang()).strip().lower()
        with self._model_lock:
            self._piper_cache.clear()
            self._release_silero_cache_locked()
        gc.collect()
        for eng_id, spec in ENGINE_SPECS.items():
            eng_dir = ENGINES_DIR / eng_id
            if not eng_dir.exists():
                continue
            raw_files = spec.get("files", [])
            if raw_files:
                for entry in raw_files:
                    if len(entry) >= 3:
                        fname, _, f_lang = entry[0], entry[1], str(entry[2]).lower()
                        if f_lang not in (target_lang, "multi"):
                            fp = eng_dir / fname
                            if fp.exists():
                                try:
                                    fp.unlink()
                                except Exception:
                                    pass
            if eng_id == "holos" and target_lang != "uk":
                for legacy_name in ("holos.pt", "voices.pt", "ready.marker"):
                    try:
                        (eng_dir / legacy_name).unlink(missing_ok=True)
                    except Exception:
                        pass
                shutil.rmtree(eng_dir / "onnx-int8", ignore_errors=True)
            elif eng_id == "teratts" and target_lang != "ru":
                try:
                    shutil.rmtree(eng_dir, ignore_errors=True)
                except Exception:
                    pass

    def is_engine_installed(self, engine_id: str) -> bool:
        """Перевіряє, чи встановлений локальний движок для поточної обраної мови."""
        if engine_id == "qwen3_tts":
            from core.llama_tts import is_qwen3_tts_installed
            return is_qwen3_tts_installed()
        if engine_id == "zonos3_mini":
            from core.llama_tts import is_zonos3_installed
            return is_zonos3_installed()
        if engine_id == "kokoro_82m":
            from core.llama_tts import is_kokoro_installed, resolve_kokoro_device
            if resolve_kokoro_device() == "cpu":
                from core.kokoro_onnx_tts import is_kokoro_onnx_installed
                return is_kokoro_onnx_installed()
            return is_kokoro_installed()
        if engine_id == "soprano_1_1":
            from core.llama_tts import is_soprano_installed
            return is_soprano_installed()
        if engine_id == "omnivoice":
            from core.omnivoice_tts import is_omnivoice_installed
            return is_omnivoice_installed()
        eng_dir = ENGINES_DIR / engine_id
        if not eng_dir.exists():
            return False
        if engine_id in ("piper", "silero", "holos"):
            req_files = self.get_files_for_engine(engine_id)
            if not req_files:
                return False
            return all((eng_dir / fname).exists() for fname, _ in req_files)
        if engine_id == "teratts":
            if self._current_lang() != "ru":
                return False
            return (eng_dir / "ready.marker").exists()
        return False

    def _resolve_engine_voices_for_lang(
        self, eng_id: str, spec: Dict[str, Any], cur_lang: str
    ) -> List[Dict[str, Any]]:
        lang_voices = self._filter_voices_by_lang(spec.get("voices", []), cur_lang)
        if eng_id == "piper" and not lang_voices:
            lang_voices, _ = self._get_dynamic_piper_for_lang(cur_lang)
        return lang_voices

    def get_visible_local_engines(self) -> List[Dict[str, Any]]:
        """
        Повертає список локальних TTS для вкладки «Завантажити на ПК», які підтримують обрану мову.
        """
        cur_lang = self._current_lang()
        items: List[Dict[str, Any]] = []
        for eng_id, spec in ENGINE_SPECS.items():
            if not self._resolve_engine_voices_for_lang(eng_id, spec, cur_lang):
                continue
            items.append(spec)
        return items

    def get_available_active_engines(self) -> List[Dict[str, Any]]:
        """
        Повертає список доступних систем TTS для верхнього блоку «Ваш активний голос»
        (відфільтрований за обраною мовою користувача):
        1) Встановлені локальні движки
        2) Додані користувачем онлайн-сервіси
        """
        available: List[Dict[str, Any]] = []
        cur_lang = self._current_lang()

        # 1. Встановлені локальні TTS
        for eng_id, spec in ENGINE_SPECS.items():
            lang_voices = self._resolve_engine_voices_for_lang(eng_id, spec, cur_lang)
            if not lang_voices:
                continue
            if self.is_engine_installed(eng_id):
                available.append({
                    "id": eng_id,
                    "title": spec["title"],
                    "voices": lang_voices,
                })

        # 2. Додані онлайн-провайдери
        online_items: List[Dict[str, Any]] = config.get("online_items", [])
        for item in online_items:
            item_id = str(item.get("id", "")).strip()
            provider = str(item.get("provider", "")).strip()
            if not item_id or provider not in ONLINE_PROVIDERS:
                continue
            if provider == "edge" and not bool(item.get("installed", False)):
                continue
            prov_spec = ONLINE_PROVIDERS[provider]
            if provider == "silero_bot":
                if cur_lang != "ru":
                    continue
                from core.telegram_silero import SESSION_FILE, is_telethon_installed, telegram_silero
                tg_st = telegram_silero.get_cached_status()
                if not is_telethon_installed() or not (tg_st["connected"] or SESSION_FILE.exists()):
                    continue
                saved_v = list(item.get("saved_voices") or [])
                bot_cur = {
                    "id": "__bot_current__",
                    "name": tr("tg.voice_current_bot", "Поточний голос бота (/speaker)"),
                    "gender": None,
                    "lang": "multi",
                }
                if saved_v:
                    raw_voices = [
                        {
                            "id": str(v.get("id", "")).strip(),
                            "name": (
                                f"{v['id']} ({v['description']})"
                                if str(v.get("description", "")).strip()
                                else str(v.get("id", ""))
                            ),
                            "gender": None,
                            "lang": "multi",
                        }
                        for v in saved_v
                        if str(v.get("id", "")).strip()
                    ] + [bot_cur]
                else:
                    raw_voices = [bot_cur]
                title = str(item.get("title", prov_spec["title"])).strip() or prov_spec["title"]
                available.append({
                    "id": f"online:{item_id}",
                    "title": title,
                    "voices": raw_voices,
                })
                continue
            raw_voices = list(item.get("custom_voices") or [])
            if raw_voices:
                if provider == "fish_audio":
                    ref_custom = str(item.get("reference_id", "")).strip()
                    if ref_custom:
                        raw_voices = [{
                            "id": "default",
                            "name": tr("tts.fish_custom_id", "Власний ID ({id}...)", id=ref_custom[:10]),
                            "gender": None,
                            "lang": "multi",
                        }] + raw_voices
            else:
                if provider == "fish_audio":
                    raw_voices = [{
                        "id": "default",
                        "name": tr("tts.fish_default_voice", "Стандартний голос (M)"),
                        "gender": "male",
                        "lang": "multi",
                    }]
                else:
                    raw_voices = list(prov_spec.get("voices", []))
                if provider == "google":
                    for v in raw_voices:
                        vid = str(v.get("id", ""))
                        if "is_free" not in v:
                            v["is_free"] = ("Wavenet" in vid or "Standard" in vid)
                        if "model" not in v:
                            v["model"] = "WaveNet" if "Wavenet" in vid else ("Standard" if "Standard" in vid else "Other")

            if provider == "google":
                paid_enabled = bool(item.get("paid_enabled", False))
                if not paid_enabled:
                    raw_voices = [v for v in raw_voices if bool(v.get("is_free", True))]
                model_filter = str(item.get("model_filter", "all")).strip()
                if model_filter and model_filter != "all":
                    raw_voices = [v for v in raw_voices if str(v.get("model", "")) == model_filter]

            if provider == "edge":
                by_id = {str(v.get("id", "")): v for v in raw_voices}
                for cv in prov_spec.get("voices", []):
                    cv_id = str(cv.get("id", ""))
                    if cv_id.endswith("::custom"):
                        if cv_id in by_id:
                            by_id[cv_id]["name"] = cv["name"]
                        else:
                            raw_voices.append(dict(cv))

            lang_voices = self._filter_voices_by_lang(raw_voices, cur_lang)
            if provider == "edge" and not lang_voices:
                lang_voices = self._get_dynamic_edge_voices(cur_lang)
            if not lang_voices:
                continue
            title = str(item.get("title", prov_spec["title"])).strip() or prov_spec["title"]
            available.append({
                "id": f"online:{item_id}",
                "title": title,
                "voices": lang_voices,
            })

        return available

    def _get_dynamic_edge_voices(self, target_lang: str) -> List[Dict[str, Any]]:
        """Динамічно підтягує голоси Microsoft Edge-TTS для будь-якої мови (наприклад, ja, it, ko) із кешем на диску."""
        lang = str(target_lang or "").strip().lower()
        if not lang:
            return []
        if lang in self._edge_dynamic_cache:
            return list(self._edge_dynamic_cache[lang])

        cache_file = DATA_DIR / "edge_voices_cache.json"
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                if isinstance(cached, dict):
                    for k, v in cached.items():
                        if isinstance(v, list):
                            self._edge_dynamic_cache[str(k).lower()] = v
                    if lang in self._edge_dynamic_cache:
                        return list(self._edge_dynamic_cache[lang])
            except Exception:
                pass

        try:
            voices_raw = self._fetch_edge_tts_voices_raw()
            grouped = self._group_edge_voices_by_lang(voices_raw)
            if grouped:
                self._edge_dynamic_cache.update(grouped)
                try:
                    cache_file.write_text(json.dumps(grouped, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass
                return list(self._edge_dynamic_cache.get(lang, []))
        except Exception:
            pass
        return []

    def get_voices_for_engine(self, engine_id: str) -> List[Dict[str, Any]]:
        gender_filter = str(config.get("voice_gender_filter", "both")).strip().lower()
        for entry in self.get_available_active_engines():
            if entry["id"] == engine_id:
                voices = list(entry.get("voices", []))
                if gender_filter in ("female", "male"):
                    filtered = [v for v in voices if v.get("gender") in (gender_filter, None)]
                    return filtered
                return voices
        return []

    # ------------------------------------------------------------------
    # Публічні методи озвучки та прослуховування прикладів
    # ------------------------------------------------------------------

    def pending_queue_size(self) -> int:
        """Повертає поточну кількість фраз у черзі озвучення."""
        return int(self._queue.qsize())

    def speak(
        self,
        text: str,
        save_history: bool = True,
        audio_file: str = "",
        skip_spellcheck: bool = False,
        mirror_twitch: bool = True,
        is_viewer_chat: bool = False,
        engine_override: str = "",
        voice_override: str = "",
        volume_override: Optional[int] = None,
        chat_user: str = "",
        chat_msg_id: str = "",
    ) -> None:
        """Додає текст або готовий аудіофайл швидкої фрази у чергу на озвучення."""
        clean = text.strip()
        if not clean and not audio_file:
            return
        # Якщо аудіофайл не передано напряму, перевіряємо, чи є серед швидких фраз записаний файл для цього тексту
        resolved_audio = audio_file.strip()
        if not resolved_audio and clean:
            for qp in config.get("quick_phrases", []):
                if str(qp.get("text", "")).strip() == clean and qp.get("audio_file"):
                    cand = Path(str(qp["audio_file"]))
                    if not cand.is_absolute():
                        cand = BASE_DIR / cand
                    if cand.exists():
                        resolved_audio = str(cand)
                        break

        if not is_viewer_chat:
            self._last_spoken_text = clean or Path(resolved_audio).stem
            self._last_spoken_audio_file = resolved_audio
            self._last_spoken_skip_spellcheck = bool(skip_spellcheck)
            self._last_spoken_mirror_twitch = bool(mirror_twitch)
        if save_history and clean:
            config.add_recent_phrase(clean)
        payload = (
            f"__FILE__{resolved_audio}||{clean}" if resolved_audio else clean,
            False,
            bool(skip_spellcheck),
            bool(mirror_twitch),
            bool(is_viewer_chat),
            str(engine_override or "").strip(),
            str(voice_override or "").strip(),
            int(volume_override) if volume_override is not None else None,
            str(chat_user or "").strip().lower(),
            str(chat_msg_id or "").strip(),
        )
        self._queue.put(payload)

    def speak_quick_phrase(self, qp: Dict[str, Any]) -> None:
        """Озвучує швидку фразу: якщо є збережений аудіофайл — відтворює його миттєво без повторного синтезу."""
        text = str(qp.get("text", "")).strip()
        audio_file = str(qp.get("audio_file", "")).strip()
        if audio_file:
            cand = Path(audio_file)
            if not cand.is_absolute():
                cand = BASE_DIR / cand
            if cand.exists():
                self.speak(
                    text or str(qp.get("title", "")),
                    save_history=bool(text),
                    audio_file=str(cand),
                    skip_spellcheck=True,
                )
                return
        if text:
            self.speak(text, save_history=True, skip_spellcheck=True)

    def speak_quick_text(self, text: str) -> None:
        """Озвучує швидку фразу з гарячої клавіші без AI-перевірки орфографії."""
        clean = str(text or "").strip()
        if clean:
            self.speak(clean, save_history=True, skip_spellcheck=True)

    def synthesize_to_file(self, text: str, dest_path: Path) -> Tuple[bool, str]:
        """Синтезує текст активним голосом (або запасним при збої) та зберігає результат у .wav файл."""
        clean = text.strip()
        if not clean:
            return False, tr("tts.err.enter_text_record", "Введіть текст для запису озвучки.")
        try:
            samples, sr, _ = self._synthesize_with_fallback(clean)
            if samples is None or len(samples) == 0:
                return False, tr("tts.err.gen_audio_failed", "Не вдалося згенерувати звук.")
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(dest_path), samples, sr, subtype="PCM_16")
            return True, str(dest_path)
        except Exception as exc:
            return False, translate_error_message(format_exception_message(exc))

    def clear_audio_cache(self) -> None:
        """Очищає кеш синтезованих аудіофрагментів у пам'яті та на диску."""
        self._audio_cache.clear()

    def flush_audio_cache(self) -> None:
        """Зберігає індекс дискового кешу на диск."""
        self._audio_cache.flush()

    def shutdown(self) -> None:
        """Зупиняє фонові таймери прогріву, чергу синтезу та вивантажує всі локальні моделі/сервери при виході."""
        self._running = False
        self._cancel_batch = True
        with self._warmup_lock:
            if self._warmup_timer is not None:
                try:
                    self._warmup_timer.cancel()
                except Exception:
                    pass
                self._warmup_timer = None
        for evt in list(self._download_cancel_flags.values()):
            try:
                evt.set()
            except Exception:
                pass
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        try:
            from core.llama_tts import shutdown_llama_tts
            shutdown_llama_tts()
        except Exception:
            pass
        try:
            from core.omnivoice_tts import shutdown_omnivoice_tts
            shutdown_omnivoice_tts()
        except Exception:
            pass
        try:
            from core.kokoro_onnx_tts import kokoro_onnx_engine
            kokoro_onnx_engine.unload()
        except Exception:
            pass
        with self._model_lock:
            self._piper_cache.clear()
            self._release_silero_cache_locked()
            self._holos_model = None
            for obj in (
                getattr(self._teratts_accent, "accents", None),
                getattr(getattr(self._teratts_model, "tokenizer", None), "dic", None),
            ):
                if obj is not None and hasattr(obj, "close"):
                    try:
                        obj.close()
                    except Exception:
                        pass
            self._teratts_accent = None
            self._teratts_model = None
            self._warmed_models.clear()

    def _build_audio_cache_key(self, synth_text: str, engine_id: str, voice_id: str) -> Tuple[Any, ...]:
        dsp = get_engine_dsp_settings(engine_id, voice_id)
        dsp_sig = tuple(sorted((str(k), str(v)) for k, v in dsp.items()))
        lang = str(config.get("app_language", "uk")).strip().lower()
        online_sig = ""
        if engine_id.startswith("online:"):
            tag = engine_id.split(":", 1)[1] if ":" in engine_id else "edge"
            if tag != "edge":
                for item in config.get("online_items", []):
                    if isinstance(item, dict) and str(item.get("id")) == tag:
                        safe_item = {
                            str(k): v
                            for k, v in item.items()
                            if str(k).lower() not in {"api_key", "apikey", "token", "secret", "password"}
                        }
                        online_sig = json.dumps(safe_item, sort_keys=True, ensure_ascii=False, default=str)
                        break
        ov_sig = ""
        if engine_id == "omnivoice":
            from core.omnivoice_tts import get_omnivoice_cache_signature

            ov_sig = get_omnivoice_cache_signature()
        elif engine_id in ("qwen3_tts", "zonos3_mini"):
            ref_path_val = str(config.get(f"{engine_id}_ref_path", "") or "").strip()
            ref_stat = ""
            if ref_path_val:
                try:
                    st = Path(ref_path_val).stat()
                    ref_stat = f"{ref_path_val}|{st.st_size}|{st.st_mtime_ns}"
                except OSError:
                    ref_stat = ref_path_val
            ref_txt = str(config.get("qwen3_tts_ref_text", "") or "").strip() if engine_id == "qwen3_tts" else ""
            ov_sig = f"{engine_id}:{ref_stat}:{ref_txt}"
        if engine_id == "kokoro_82m":
            from core.llama_tts import resolve_kokoro_device

            ov_sig = f"{ov_sig}|dev:{resolve_kokoro_device()}"
        elif engine_id == "soprano_1_1":
            from core.llama_tts import resolve_soprano_device

            ov_sig = f"{ov_sig}|dev:{resolve_soprano_device()}"
        return (synth_text, engine_id, voice_id, dsp_sig, lang, online_sig, ov_sig)

    def _cache_lookup(self, key: Tuple[Any, ...]) -> Optional[Tuple[np.ndarray, int, str]]:
        return self._audio_cache.lookup(key)

    def _cache_store(self, key: Tuple[Any, ...], samples: np.ndarray, sr: int, title: str) -> None:
        self._audio_cache.store(key, samples, sr, title)

    def _synthesize_with_fallback(
        self,
        text: str,
        engine_override: str = "",
        voice_override: str = "",
    ) -> Tuple[np.ndarray, int, str]:
        """
        Виконує синтез через Головний TTS + його налаштування DSP (з розумним кешем повтору).
        Якщо Головний TTS дав збій і обрано Запасний TTS — автоматично перемикається на Запасний TTS + його DSP.
        """
        engine_id = str(engine_override or config.get("active_engine", "")).strip()
        voice_id = str(voice_override or ("" if engine_override else config.get("active_voice", ""))).strip()

        available = self.get_available_active_engines()
        if not available:
            raise RuntimeError(tr("tts.err.no_active_tts", "Не вибрано жодного активного TTS. Встановіть голос або додайте онлайн-голос."))

        eng_entry = next((a for a in available if a["id"] == engine_id), available[0])
        primary_id = eng_entry["id"]
        voices = eng_entry.get("voices", [])
        valid_voice_ids = [v["id"] for v in voices]
        primary_voice = voice_id if voice_id in valid_voice_ids else (valid_voice_ids[0] if valid_voice_ids else "")
        primary_title = eng_entry["title"]

        synth_text = (
            normalize_tts_text(text, str(config.get("app_language", "uk")))
            if bool(config.get("normalize_numbers_enabled", True))
            else text
        )
        synth_text = apply_default_stress_fixes(synth_text)

        from core.omnivoice_tags import (
            build_active_reaction_specs,
            expand_omnivoice_tags,
            strip_omnivoice_tags,
        )

        ui_lang_cur = str(config.get("app_language") or config.get("ui_language") or "uk")

        def _for_engine_tags(raw_txt: str, target_eng: str) -> str:
            if target_eng == "omnivoice" and bool(config.get("reaction_omnivoice_enabled", True)):
                specs = build_active_reaction_specs(config.data, engine_id="omnivoice", ui_lang=ui_lang_cur, base_dir=BASE_DIR)
                return expand_omnivoice_tags(raw_txt, specs=specs, ui_lang=ui_lang_cur)
            stripped = strip_omnivoice_tags(raw_txt)
            return stripped if stripped else raw_txt

        primary_synth_text = _for_engine_tags(synth_text, primary_id)
        cache_key = self._build_audio_cache_key(primary_synth_text, primary_id, primary_voice)
        cached = self._cache_lookup(cache_key)
        if cached is not None:
            return cached

        fb_id = str(config.get("fallback_engine", "")).strip()
        fb_voice = str(config.get("fallback_voice", "")).strip()
        fb_entry = next((a for a in available if a["id"] == fb_id), None) if fb_id else None
        fb_voices = fb_entry.get("voices", []) if fb_entry else []
        fb_valid_ids = [v["id"] for v in fb_voices]
        resolved_fb_voice = fb_voice if fb_voice in fb_valid_ids else (fb_valid_ids[0] if fb_valid_ids else "")
        has_fallback = bool(fb_entry and resolved_fb_voice and (fb_id != primary_id or resolved_fb_voice != primary_voice))

        try:
            self.status_changed.emit(tr("tts.status.synth", "Синтез ({title})...", title=primary_title))
            samples, sr = self._synthesize_dispatch(primary_id, primary_voice, primary_synth_text, 1.0)
            samples, sr = apply_engine_dsp(samples, sr, primary_id, primary_voice)
            self._cache_store(cache_key, samples, sr, primary_title)
            return samples, sr, primary_title
        except Exception as primary_exc:
            from core.llama_tts import Qwen3SynthesisCancelledError
            from core.omnivoice_tts import OmniVoiceSynthesisCancelledError
            from core.telegram_silero import SynthesisCancelledError

            if self._cancel_batch or isinstance(primary_exc, (SynthesisCancelledError, Qwen3SynthesisCancelledError, OmniVoiceSynthesisCancelledError)):
                raise
            if has_fallback and fb_entry:
                fb_title = fb_entry["title"]
                print(
                    f"[TTSManager] Збій основного TTS ({primary_title}: {primary_exc}). Перемикання на запасний TTS ({fb_title})...",
                    file=sys.stderr,
                )
                fb_synth_text = _for_engine_tags(synth_text, fb_id)
                fb_cache_key = self._build_audio_cache_key(fb_synth_text, fb_id, resolved_fb_voice)
                fb_cached = self._cache_lookup(fb_cache_key)
                if fb_cached is not None:
                    return fb_cached
                self.status_changed.emit(tr("tts.status.fallback_synth", "⚡ Запасний TTS ({title})...", title=fb_title))
                samples, sr = self._synthesize_dispatch(fb_id, resolved_fb_voice, fb_synth_text, 1.0)
                samples, sr = apply_engine_dsp(samples, sr, fb_id, resolved_fb_voice)
                self._cache_store(fb_cache_key, samples, sr, fb_title)
                return samples, sr, fb_title
            # Retry the selected engine once when no usable backup is configured.
            # This shared path covers local engines and every online provider.
            self.status_changed.emit(
                tr("tts.status.retrying", "Не вдалося озвучити. Повторна спроба ({title})...", title=primary_title)
            )
            print(
                f"[TTSManager] Повторна спроба синтезу через {primary_title} після помилки: {primary_exc}",
                file=sys.stderr,
                flush=True,
            )
            try:
                samples, sr = self._synthesize_dispatch(primary_id, primary_voice, primary_synth_text, 1.0)
                samples, sr = apply_engine_dsp(samples, sr, primary_id, primary_voice)
                self._cache_store(cache_key, samples, sr, primary_title)
                return samples, sr, primary_title
            except Exception as retry_exc:
                if self._cancel_batch or isinstance(retry_exc, (SynthesisCancelledError, Qwen3SynthesisCancelledError, OmniVoiceSynthesisCancelledError)):
                    raise
                print(
                    f"[TTSManager] Повторна спроба через {primary_title} також завершилась помилкою: {retry_exc}",
                    file=sys.stderr,
                    flush=True,
                )
                raise retry_exc

    def repeat_last(self) -> None:
        """Повторює останню озвучену фразу (із кешу, якщо голос/DSP не змінилися)."""
        if self._last_spoken_audio_file and Path(self._last_spoken_audio_file).exists():
            self.speak(
                self._last_spoken_text,
                save_history=False,
                audio_file=self._last_spoken_audio_file,
                skip_spellcheck=True,
                mirror_twitch=self._last_spoken_mirror_twitch,
            )
        elif self._last_spoken_text:
            if (
                bool(config.get("ai_spellcheck_enabled", False))
                and not self._last_spoken_skip_spellcheck
                and self._last_corrected_pair is not None
                and self._last_corrected_pair[0] == self._last_spoken_text
            ):
                self._queue.put((self._last_corrected_pair[1], False, True, self._last_spoken_mirror_twitch))
                return
            self.speak(
                self._last_spoken_text,
                save_history=False,
                skip_spellcheck=self._last_spoken_skip_spellcheck,
                mirror_twitch=self._last_spoken_mirror_twitch,
            )
        else:
            recents = config.get("recent_phrases", [])
            if recents:
                self.speak(recents[0], save_history=False)

    def stop_current(self) -> None:
        """Миттєво зупиняє поточну озвучку, скасовує активні запити (Telegram / Qwen3) та очищує чергу."""
        self._cancel_batch = True
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        try:
            from core.telegram_silero import telegram_silero

            telegram_silero.cancel_active_synthesis()
        except Exception:
            pass
        try:
            from core.llama_tts import cancel_qwen3_synthesis

            cancel_qwen3_synthesis()
        except Exception:
            pass
        try:
            from core.omnivoice_tts import cancel_omnivoice_synthesis

            cancel_omnivoice_synthesis()
        except Exception:
            pass
        audio_router.stop()
        obs_server.broadcast_clear()
        self.status_changed.emit(tr("tts.status.stopped", "Зупинено"))

    def _cancel_active_chat_playback(self, status_text: str) -> None:
        self._cancel_batch = True
        try:
            from core.telegram_silero import telegram_silero

            telegram_silero.cancel_active_synthesis()
        except Exception:
            pass
        try:
            from core.llama_tts import cancel_qwen3_synthesis

            cancel_qwen3_synthesis()
        except Exception:
            pass
        try:
            from core.omnivoice_tts import cancel_omnivoice_synthesis

            cancel_omnivoice_synthesis()
        except Exception:
            pass
        audio_router.stop()
        obs_server.broadcast_clear()
        self.status_changed.emit(status_text)

    def _prune_queue(self, should_drop) -> int:
        removed = 0
        with self._queue.mutex:
            kept: List[Any] = []
            while self._queue.queue:
                queued = self._queue.queue.popleft()
                if queued is not None and should_drop(queued):
                    removed += 1
                    self._queue.unfinished_tasks = max(0, self._queue.unfinished_tasks - 1)
                else:
                    kept.append(queued)
            self._queue.queue.extend(kept)
            if self._queue.unfinished_tasks == 0:
                self._queue.all_tasks_done.notify_all()
        return removed

    def stop_chat_tts(self) -> None:
        """Зупиняє поточну озвучку повідомлення чату та очищає чергу чату (не перериває власну мову)."""
        self._prune_queue(lambda q: bool(q[4]) if isinstance(q, tuple) and len(q) >= 5 else False)
        if self._current_is_viewer_chat:
            self._cancel_active_chat_playback(tr("tts.status.chat_stopped", "Озвучку чату зупинено"))

    def cancel_chat_by_moderation(self, event_type: str, target_user: str = "", target_msg_id: str = "") -> int:
        """Скасовує озвучку в черзі або в ефірі, якщо повідомлення/користувача замутили або видалили."""
        ev = str(event_type or "").strip().lower()
        t_user = str(target_user or "").strip().lower()
        t_id = str(target_msg_id or "").strip()

        def _matches_tuple(is_chat: bool, q_user: str, q_id: str) -> bool:
            if not is_chat:
                return False
            if ev == "clearchat":
                return True if not t_user else (q_user == t_user)
            if ev == "clearmsg":
                if t_id and q_id:
                    return q_id == t_id
                return bool(t_user and q_user == t_user)
            return False

        removed = self._prune_queue(
            lambda q: _matches_tuple(
                bool(q[4]) if isinstance(q, tuple) and len(q) >= 5 else False,
                str(q[8]).lower() if isinstance(q, tuple) and len(q) >= 9 else "",
                str(q[9]) if isinstance(q, tuple) and len(q) >= 10 else "",
            )
        )
        if _matches_tuple(self._current_is_viewer_chat, self._current_chat_user, self._current_chat_msg_id):
            removed += 1
            self._cancel_active_chat_playback(tr("tts.status.chat_moderated", "Повідомлення чату видалено модератором"))
        return removed

    def play_sample_preview(self, sample_filename: str, monitor_only: bool = True) -> None:
        """Відтворює заготовлений .wav-зразок голосу в окремому потоці."""
        p = Path(sample_filename)
        sample_path = p if p.is_absolute() and p.exists() else (SAMPLES_DIR / sample_filename)
        if not sample_path.exists():
            alt = BASE_DIR / sample_filename
            if alt.exists():
                sample_path = alt
        if not sample_path.exists():
            self.error_occurred.emit(tr("tts.err.sample_not_found", "Зразок не знайдено: {name}", name=sample_filename))
            return

        # Новий запит одразу зупиняє поточне прослуховування і скасовує всі попередні,
        # що ще чекають у черзі (раніше кожне натискання ставало в чергу і програвалося по колу).
        gen = audio_router.begin_exclusive_playback()

        def _run():
            if gen != audio_router.current_generation:
                return
            self.status_changed.emit(tr("tts.status.previewing", "Прослуховування: {name}...", name=sample_path.name))
            audio_router.play_wav_file(
                sample_path,
                monitor_only=monitor_only,
                generation=gen,
                trim_leading_silence=True,
            )
            if gen == audio_router.current_generation:
                self.status_changed.emit(tr("ui_113", "Система готова"))

        threading.Thread(target=_run, daemon=True).start()

    # ------------------------------------------------------------------
    # Фоновий потік черги синтезу
    # ------------------------------------------------------------------

    @staticmethod
    def _load_reaction_audio_samples(audio_path: Path) -> Tuple[np.ndarray, int]:
        """Завантажує аудіофайл звукової команди (.wav/.mp3/.ogg/.flac) у float32-масив."""
        if not audio_path.is_file():
            raise RuntimeError(
                tr("tts.err.quick_audio_missing", "Аудіофайл швидкої фрази не знайдено: {name}", name=audio_path.name)
            )
        try:
            data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
            samples = np.asarray(data, dtype=np.float32)
            if samples.ndim == 2 and samples.shape[1] > 2:
                samples = np.mean(samples, axis=1).astype(np.float32)
            return samples, int(sr)
        except Exception:
            # Резервне декодування через ffmpeg (напр. для m4a / нестандартного mp3)
            ff = _find_ffmpeg_exe()
            if not ff:
                raise
            proc = subprocess.run(
                [ff, "-v", "error", "-i", str(audio_path), "-f", "wav", "-acodec", "pcm_f32le", "pipe:1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
            )
            if proc.returncode != 0 or not proc.stdout:
                raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore").strip() or f"Decode failed: {audio_path.name}")
            data, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32", always_2d=False)
            samples = np.asarray(data, dtype=np.float32)
            if samples.ndim == 2 and samples.shape[1] > 2:
                samples = np.mean(samples, axis=1).astype(np.float32)
            return samples, int(sr)

    @staticmethod
    def _play_with_synced_obs(
        samples: np.ndarray,
        sr: int,
        *,
        subtitle_text: str,
        duration: float,
        monitor_only: bool,
        send_primary: bool,
        vban_stream_override: Optional[str],
        extra_gain_pct: int,
    ) -> None:
        """Queue audio first, then broadcast OBS subtitles offset by real PortAudio buffer latency."""
        def _on_queued(lat_ms: float) -> None:
            from core.omnivoice_tags import strip_whisper_tags

            clean_sub = strip_whisper_tags(strip_plus_stress_marks(subtitle_text))
            if not clean_sub:
                return
            delays = compute_speech_char_delays(
                clean_sub,
                samples,
                int(sr),
                start_offset_ms=max(85.0, float(lat_ms)),
            )
            obs_server.broadcast_speech(clean_sub, duration, char_delays=delays)

        audio_router.play_samples(
            samples,
            int(sr),
            monitor_only=monitor_only,
            send_primary=send_primary,
            vban_stream_override=vban_stream_override,
            extra_gain_pct=extra_gain_pct,
            on_playback_queued=_on_queued if subtitle_text else None,
        )

    def _resolve_qwen3_stream_plan(
        self,
        text: str,
        engine_override: str = "",
        voice_override: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Повертає план стримінг-синтезу для Qwen3-TTS, якщо чанк не в кеші та немає важких full-clip DSP."""
        engine_id = str(engine_override or config.get("active_engine", "")).strip()
        voice_id = str(voice_override or ("" if engine_override else config.get("active_voice", ""))).strip()
        available = self.get_available_active_engines()
        if not available:
            return None
        eng_entry = next((a for a in available if a["id"] == engine_id), available[0])
        primary_id = eng_entry["id"]
        if primary_id not in ("qwen3_tts", "zonos3_mini", "omnivoice"):
            return None
        if primary_id == "omnivoice":
            from core.omnivoice_tts import normalize_omnivoice_mode

            own_ref = str(config.get("omnivoice_ref_path", "") or "").strip()
            if normalize_omnivoice_mode() != "clone" or not (own_ref and Path(own_ref).is_file()):
                # Voice Design uses buffered WAV for full peak normalisation (+6..12 dB cleaner level)
                return None
        voices = eng_entry.get("voices", [])
        valid_voice_ids = [v["id"] for v in voices]
        primary_voice = voice_id if voice_id in valid_voice_ids else (valid_voice_ids[0] if valid_voice_ids else "")
        primary_title = eng_entry["title"]

        synth_text = (
            normalize_tts_text(text, str(config.get("app_language", "uk")))
            if bool(config.get("normalize_numbers_enabled", True))
            else text
        )
        synth_text = format_stress_marks(apply_default_stress_fixes(synth_text), "strip")
        from core.omnivoice_tags import (
            build_active_reaction_specs,
            expand_omnivoice_tags,
            strip_omnivoice_tags,
        )

        ui_lang_cur = str(config.get("app_language") or config.get("ui_language") or "uk")
        if primary_id == "omnivoice" and bool(config.get("reaction_omnivoice_enabled", True)):
            specs = build_active_reaction_specs(config.data, engine_id="omnivoice", ui_lang=ui_lang_cur, base_dir=BASE_DIR)
            synth_text = expand_omnivoice_tags(synth_text, specs=specs, ui_lang=ui_lang_cur)
        else:
            stripped = strip_omnivoice_tags(synth_text)
            if stripped:
                synth_text = stripped
        cache_key = self._build_audio_cache_key(synth_text, primary_id, primary_voice)
        if self._cache_lookup(cache_key) is not None:
            return None

        dsp = get_engine_dsp_settings(primary_id, primary_voice)
        any_fx = (
            bool(dsp.get("denoise_enabled", False))
            or bool(dsp.get("deess_enabled", False))
            or bool(dsp.get("speed_enabled", False))
            or bool(dsp.get("pitch_enabled", False))
            or bool(dsp.get("gate_enabled", False))
        )
        if any_fx:
            return None

        voice_entry = next((v for v in voices if str(v.get("id")) == primary_voice), None)
        lang = str((voice_entry or {}).get("lang") or (primary_voice.rsplit("_", 1)[-1] if "_" in primary_voice else "en") or "en")
        speaker = str((voice_entry or {}).get("speaker") or "")
        return {
            "primary_id": primary_id,
            "primary_voice": primary_voice,
            "primary_title": primary_title,
            "synth_text": synth_text,
            "cache_key": cache_key,
            "lang": lang,
            "speaker": speaker,
        }

    def _speak_qwen3_streaming_chunk(
        self,
        chunk_text: str,
        full_phrase_text: str,
        *,
        plan: Dict[str, Any],
        emit_phrase_spoken: bool,
        monitor_only: bool,
        send_primary: bool,
        vban_stream_override: Optional[str],
        extra_gain_pct: int,
        on_ready_for_next: Optional[Callable[[], None]] = None,
    ) -> bool:
        """
        Стримить Qwen3-TTS / ZONOS3-Mini по мірі декодування аудіо-блоків і починає відтворення одразу з 1-го чанка.
        Повертає False, якщо сталася помилка ДО першого звуку (щоб викликач міг відпрацювати fallback/retry).
        """
        from core.llama_tts import Qwen3SynthesisCancelledError, synthesize_llama_tts_stream, synthesize_qwen3_tts_stream
        from core.omnivoice_tts import OmniVoiceSynthesisCancelledError, synthesize_omnivoice_tts_stream
        from core.telegram_silero import SynthesisCancelledError

        primary_id = plan["primary_id"]
        primary_voice = plan["primary_voice"]
        primary_title = plan["primary_title"]
        synth_text = plan["synth_text"]
        cache_key = plan["cache_key"]
        lang = plan["lang"]
        speaker = str(plan.get("speaker") or "")

        self.status_changed.emit(tr("tts.status.synth", "Синтез ({title})...", title=primary_title))
        t0 = time.perf_counter()
        first_chunk_played = False
        first_ms = 0

        def _dsp_chunks():
            if primary_id == "qwen3_tts":
                stream_iter = synthesize_qwen3_tts_stream(synth_text, lang=lang, speed=1.0)
            elif primary_id == "omnivoice":
                stream_iter = synthesize_omnivoice_tts_stream(synth_text, lang=lang, speed=1.0)
            else:
                stream_iter = synthesize_llama_tts_stream(primary_id, synth_text, lang=lang, voice=speaker, speed=1.0)
            for raw_chunk, sr in stream_iter:
                if self._cancel_batch or not self._running:
                    raise Qwen3SynthesisCancelledError("Qwen3-TTS synthesis cancelled")
                if raw_chunk is None or len(raw_chunk) == 0:
                    continue
                proc_chunk, proc_sr = apply_engine_dsp(raw_chunk, sr, primary_id, primary_voice)
                if proc_chunk is not None and len(proc_chunk) > 0:
                    yield proc_chunk, proc_sr

        def _on_first_chunk(first_samples: np.ndarray, sr: int, lat_ms: float) -> None:
            nonlocal first_chunk_played, first_ms
            first_chunk_played = True
            first_ms = int((time.perf_counter() - t0) * 1000)
            self.status_changed.emit(tr("tts.status.speaking_ms", "Озвучення ({ms} мс)...", ms=first_ms))
            if emit_phrase_spoken:
                spoken_disp = strip_plus_stress_marks(full_phrase_text)
                if primary_id == "omnivoice":
                    spoken_disp = strip_omnivoice_tags(spoken_disp)
                self.phrase_spoken.emit(spoken_disp)
            clean_chunk_text = strip_plus_stress_marks(chunk_text)
            if primary_id == "omnivoice":
                clean_chunk_text = strip_omnivoice_tags(clean_chunk_text)
            est_dur = max(len(first_samples) / float(max(1, sr)), len(clean_chunk_text) / 13.0)
            self.speech_started.emit(clean_chunk_text, est_dur)
            if clean_chunk_text:
                n_chars = len(clean_chunk_text)
                base_ms = max(85.0, float(lat_ms))
                step_ms = (est_dur * 1000.0) / float(max(1, n_chars))
                delays = [int(round(base_ms + i * step_ms)) for i in range(n_chars)]
                obs_server.broadcast_speech(clean_chunk_text, est_dur, char_delays=delays)

        def _on_generation_done(full_samples: np.ndarray, sr: int) -> None:
            if not self._cancel_batch and self._running and full_samples is not None and len(full_samples) > 0:
                self._cache_store(cache_key, full_samples, sr, primary_title)
            if on_ready_for_next is not None and not self._cancel_batch and self._running:
                on_ready_for_next()

        try:
            full_samples, _sr = audio_router.play_stream(
                _dsp_chunks(),
                monitor_only=monitor_only,
                send_primary=send_primary,
                vban_stream_override=vban_stream_override,
                extra_gain_pct=extra_gain_pct,
                on_first_chunk=_on_first_chunk,
                on_generation_done=_on_generation_done,
            )
        except (Qwen3SynthesisCancelledError, OmniVoiceSynthesisCancelledError, SynthesisCancelledError):
            if first_chunk_played:
                self.speech_finished.emit(chunk_text)
            raise
        except Exception:
            if not first_chunk_played and not self._cancel_batch:
                return False
            if first_chunk_played:
                self.speech_finished.emit(chunk_text)
            raise

        if not first_chunk_played:
            if self._cancel_batch:
                raise Qwen3SynthesisCancelledError("Qwen3-TTS synthesis cancelled")
            return False

        self.speech_finished.emit(chunk_text)
        if not self._cancel_batch and full_samples is not None and len(full_samples) > 0:
            self.status_changed.emit(tr("tts.status.done_ms", "Готово ({ms} мс)", ms=first_ms))
        return True

    def _worker_loop(self) -> None:
        while self._running:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break

            if len(item) >= 10:
                text, monitor_only, skip_spellcheck, mirror_twitch, is_viewer_chat, eng_ovr, voice_ovr, vol_ovr, chat_user, chat_msg_id = item[:10]
            elif len(item) >= 8:
                text, monitor_only, skip_spellcheck, mirror_twitch, is_viewer_chat, eng_ovr, voice_ovr, vol_ovr = item[:8]
                chat_user, chat_msg_id = "", ""
            elif len(item) >= 5:
                text, monitor_only, skip_spellcheck, mirror_twitch, is_viewer_chat = item[:5]
                eng_ovr, voice_ovr, vol_ovr, chat_user, chat_msg_id = "", "", None, "", ""
            elif len(item) >= 4:
                text, monitor_only, skip_spellcheck, mirror_twitch = item[:4]
                is_viewer_chat = False
                eng_ovr, voice_ovr, vol_ovr, chat_user, chat_msg_id = "", "", None, "", ""
            else:
                text, monitor_only, skip_spellcheck = item[:3]
                mirror_twitch = True
                is_viewer_chat = False
                eng_ovr, voice_ovr, vol_ovr, chat_user, chat_msg_id = "", "", None, "", ""

            send_primary = True
            vban_stream_override: Optional[str] = None
            extra_gain_pct = 100
            if is_viewer_chat:
                send_primary = bool(config.get("twitch_chat_send_vbcable", False))
                send_vban_cfg = config.get("twitch_chat_send_vban", None)
                vban_mode = str(config.get("twitch_chat_vban_mode", "same") or "same").strip().lower()
                send_vban = (vban_mode in ("same", "separate")) if send_vban_cfg is None else bool(send_vban_cfg)
                if not send_vban or vban_mode == "off":
                    vban_stream_override = ""
                elif vban_mode == "separate":
                    vban_stream_override = (
                        str(config.get("twitch_chat_vban_stream", "SilentChat") or "SilentChat").strip()[:16]
                        or "SilentChat"
                    )
                if vol_ovr is not None:
                    extra_gain_pct = max(0, min(200, int(vol_ovr)))
                else:
                    extra_gain_pct = max(0, min(200, int(config.get("twitch_chat_volume", 100))))

            self._cancel_batch = False
            self._is_synthesizing = True
            self._current_is_viewer_chat = bool(is_viewer_chat)
            self._current_chat_user = str(chat_user or "").lower()
            self._current_chat_msg_id = str(chat_msg_id or "")
            try:
                if text.startswith("__FILE__"):
                    rest = text[len("__FILE__") :]
                    file_part, _, caption_part = rest.partition("||")
                    wav_path = Path(file_part)
                    if not wav_path.exists():
                        raise RuntimeError(tr("tts.err.quick_audio_missing", "Аудіофайл швидкої фрази не знайдено: {name}", name=wav_path.name))
                    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
                    samples = np.asarray(data, dtype=np.float32)
                    if samples.ndim == 2 and samples.shape[1] > 2:
                        samples = np.mean(samples, axis=1).astype(np.float32)
                    duration = len(samples) / float(sr) if sr > 0 else 0.0
                    display_txt = caption_part or wav_path.stem
                    self.status_changed.emit(tr("tts.status.playing_rec", "Відтворення запису..."))
                    self.speech_started.emit(display_txt, duration)
                    if caption_part and mirror_twitch:
                        self.phrase_spoken.emit(caption_part)
                    self._play_with_synced_obs(
                        samples,
                        int(sr),
                        subtitle_text=caption_part,
                        duration=duration,
                        monitor_only=monitor_only,
                        send_primary=send_primary,
                        vban_stream_override=vban_stream_override,
                        extra_gain_pct=extra_gain_pct,
                    )
                    self.speech_finished.emit(display_txt)
                    if not self._cancel_batch:
                        self.status_changed.emit(tr("tts.status.done_rec", "Готово (із запису)"))
                    continue

                # Перевіряємо, чи містить фраза активні звукові команди з прикріпленими аудіофайлами
                from core.omnivoice_tags import (
                    build_active_reaction_specs,
                    expand_omnivoice_tags,
                    format_tags_for_display,
                    split_text_for_reaction_playback,
                )

                eff_eng_id = str(eng_ovr or config.get("active_engine", "") or "").strip()
                ui_lang_cur = str(config.get("app_language") or config.get("ui_language") or "uk")
                active_rc_specs = build_active_reaction_specs(
                    config.data,
                    engine_id=eff_eng_id,
                    ui_lang=ui_lang_cur,
                    base_dir=BASE_DIR,
                )
                rc_segments = (
                    split_text_for_reaction_playback(
                        text,
                        active_rc_specs,
                        is_omnivoice_engine=(
                            eff_eng_id == "omnivoice"
                            and bool(config.get("reaction_omnivoice_enabled", True))
                        ),
                        ui_lang=ui_lang_cur,
                    )
                    if active_rc_specs
                    else []
                )
                if any(kind == "audio" for kind, _, _ in rc_segments):
                    display_full = format_tags_for_display(
                        expand_omnivoice_tags(
                            strip_plus_stress_marks(text),
                            specs=active_rc_specs,
                            ui_lang=ui_lang_cur,
                        ),
                        specs=active_rc_specs,
                        ui_lang=ui_lang_cur,
                    )
                    if mirror_twitch and display_full:
                        self.phrase_spoken.emit(display_full)
                    for seg_kind, seg_val, seg_disp in rc_segments:
                        if self._cancel_batch or not self._running:
                            break
                        if seg_kind == "audio":
                            samples, sr = self._load_reaction_audio_samples(Path(seg_val))
                            if samples is not None and len(samples) > 0 and sr > 0:
                                dur = len(samples) / float(sr)
                                self.status_changed.emit(tr("tts.status.playing_rec", "Відтворення запису..."))
                                self.speech_started.emit(seg_disp or Path(seg_val).stem, dur)
                                self._play_with_synced_obs(
                                    samples,
                                    int(sr),
                                    subtitle_text=seg_disp,
                                    duration=dur,
                                    monitor_only=monitor_only,
                                    send_primary=send_primary,
                                    vban_stream_override=vban_stream_override,
                                    extra_gain_pct=extra_gain_pct,
                                )
                                self.speech_finished.emit(seg_disp or Path(seg_val).stem)
                        elif seg_kind == "text" and seg_val.strip():
                            seg_text = seg_val.strip()
                            if bool(config.get("ai_spellcheck_enabled", False)) and not skip_spellcheck:
                                self.status_changed.emit(tr("tts.status.ai_check", "🪄 AI перевірка орфографії..."))
                                seg_text = correct_text_with_ai(seg_text)
                            sub_chunks = (
                                split_into_sentences_punct(seg_text)
                                if bool(config.get("split_sentences_enabled", True))
                                else [seg_text]
                            )
                            if eff_eng_id == "omnivoice" and bool(config.get("reaction_omnivoice_enabled", True)):
                                from core.omnivoice_tags import propagate_whisper_across_chunks

                                sub_chunks = propagate_whisper_across_chunks(sub_chunks)
                            for sc in sub_chunks:
                                if self._cancel_batch or not self._running:
                                    break
                                t0 = time.perf_counter()
                                samples, sr, _ = self._synthesize_with_fallback(sc, eng_ovr, voice_ovr)
                                synth_ms = int((time.perf_counter() - t0) * 1000)
                                if self._cancel_batch or not self._running:
                                    break
                                if samples is not None and len(samples) > 0:
                                    duration = len(samples) / float(sr)
                                    self.status_changed.emit(tr("tts.status.speaking_ms", "Озвучення ({ms} мс)...", ms=synth_ms))
                                    clean_sc = strip_plus_stress_marks(sc)
                                    self.speech_started.emit(clean_sc, duration)
                                    self._play_with_synced_obs(
                                        samples,
                                        sr,
                                        subtitle_text=clean_sc,
                                        duration=duration,
                                        monitor_only=monitor_only,
                                        send_primary=send_primary,
                                        vban_stream_override=vban_stream_override,
                                        extra_gain_pct=extra_gain_pct,
                                    )
                                    self.speech_finished.emit(clean_sc)
                    if not self._cancel_batch:
                        self.status_changed.emit(tr("ui_113", "Система готова"))
                    continue

                if bool(config.get("ai_spellcheck_enabled", False)) and not skip_spellcheck:
                    self.status_changed.emit(tr("tts.status.ai_check", "🪄 AI перевірка орфографії..."))
                    raw_text = text
                    text = correct_text_with_ai(text)
                    self._last_corrected_pair = (raw_text, text)

                if eff_eng_id == "omnivoice" and bool(config.get("reaction_omnivoice_enabled", True)) and active_rc_specs:
                    text = expand_omnivoice_tags(text, specs=active_rc_specs, ui_lang=ui_lang_cur)
                chunks = (
                    split_into_sentences_punct(text)
                    if bool(config.get("split_sentences_enabled", True))
                    else [text]
                )
                if eff_eng_id == "omnivoice" and bool(config.get("reaction_omnivoice_enabled", True)):
                    from core.omnivoice_tags import propagate_whisper_across_chunks

                    chunks = propagate_whisper_across_chunks(chunks)
                if len(chunks) > 1:
                    prefetch_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    try:
                        phrase_emitted = False
                        start_idx = 0
                        next_future = None
                        t_pref = time.perf_counter()

                        first_stream_plan = self._resolve_qwen3_stream_plan(chunks[0], eng_ovr, voice_ovr)
                        if first_stream_plan is not None:
                            def _start_chunk1_prefetch() -> None:
                                nonlocal next_future, t_pref
                                if next_future is None and not self._cancel_batch and self._running:
                                    t_pref = time.perf_counter()
                                    next_future = prefetch_pool.submit(
                                        self._synthesize_with_fallback, chunks[1], eng_ovr, voice_ovr
                                    )

                            streamed_first = self._speak_qwen3_streaming_chunk(
                                chunks[0],
                                text,
                                plan=first_stream_plan,
                                emit_phrase_spoken=mirror_twitch,
                                monitor_only=monitor_only,
                                send_primary=send_primary,
                                vban_stream_override=vban_stream_override,
                                extra_gain_pct=extra_gain_pct,
                                on_ready_for_next=_start_chunk1_prefetch,
                            )
                            if streamed_first:
                                phrase_emitted = True
                                start_idx = 1
                                if next_future is None and not self._cancel_batch and self._running:
                                    t_pref = time.perf_counter()
                                    next_future = prefetch_pool.submit(
                                        self._synthesize_with_fallback, chunks[1], eng_ovr, voice_ovr
                                    )

                        if start_idx == 0:
                            t_pref = time.perf_counter()
                            next_future = prefetch_pool.submit(
                                self._synthesize_with_fallback, chunks[0], eng_ovr, voice_ovr
                            )

                        for idx in range(start_idx, len(chunks)):
                            if self._cancel_batch or not self._running:
                                break
                            samples, sr, _ = next_future.result()
                            synth_ms = int((time.perf_counter() - t_pref) * 1000)
                            if self._cancel_batch or not self._running:
                                break
                            if idx + 1 < len(chunks):
                                t_pref = time.perf_counter()
                                next_future = prefetch_pool.submit(
                                    self._synthesize_with_fallback, chunks[idx + 1], eng_ovr, voice_ovr
                                )
                            if samples is not None and len(samples) > 0:
                                duration = len(samples) / float(sr)
                                self.status_changed.emit(tr("tts.status.speaking_ms", "Озвучення ({ms} мс)...", ms=synth_ms))
                                if not phrase_emitted:
                                    phrase_emitted = True
                                    if mirror_twitch:
                                        self.phrase_spoken.emit(strip_plus_stress_marks(text))
                                clean_chunk = strip_plus_stress_marks(chunks[idx])
                                self.speech_started.emit(clean_chunk, duration)
                                self._play_with_synced_obs(
                                    samples,
                                    sr,
                                    subtitle_text=clean_chunk,
                                    duration=duration,
                                    monitor_only=monitor_only,
                                    send_primary=send_primary,
                                    vban_stream_override=vban_stream_override,
                                    extra_gain_pct=extra_gain_pct,
                                )
                                self.speech_finished.emit(clean_chunk)
                                if not self._cancel_batch:
                                    self.status_changed.emit(tr("tts.status.done_ms", "Готово ({ms} мс)", ms=synth_ms))
                            else:
                                raise RuntimeError(tr("tts.err.empty_audio", "TTS-модель не повернула звук для цього тексту."))
                    finally:
                        prefetch_pool.shutdown(wait=not self._cancel_batch, cancel_futures=True)
                else:
                    for chunk in chunks:
                        if self._cancel_batch or not self._running:
                            break
                        stream_plan = self._resolve_qwen3_stream_plan(chunk, eng_ovr, voice_ovr)
                        if stream_plan is not None:
                            if self._speak_qwen3_streaming_chunk(
                                chunk,
                                text,
                                plan=stream_plan,
                                emit_phrase_spoken=mirror_twitch,
                                monitor_only=monitor_only,
                                send_primary=send_primary,
                                vban_stream_override=vban_stream_override,
                                extra_gain_pct=extra_gain_pct,
                            ):
                                continue
                            if self._cancel_batch or not self._running:
                                break

                        t0 = time.perf_counter()
                        samples, sr, _ = self._synthesize_with_fallback(chunk, eng_ovr, voice_ovr)
                        synth_ms = int((time.perf_counter() - t0) * 1000)

                        if self._cancel_batch or not self._running:
                            break
                        if samples is not None and len(samples) > 0:
                            duration = len(samples) / float(sr)
                            self.status_changed.emit(tr("tts.status.speaking_ms", "Озвучення ({ms} мс)...", ms=synth_ms))
                            if mirror_twitch:
                                self.phrase_spoken.emit(strip_plus_stress_marks(text))
                            clean_chunk = strip_plus_stress_marks(chunk)
                            self.speech_started.emit(clean_chunk, duration)
                            self._play_with_synced_obs(
                                samples,
                                sr,
                                subtitle_text=clean_chunk,
                                duration=duration,
                                monitor_only=monitor_only,
                                send_primary=send_primary,
                                vban_stream_override=vban_stream_override,
                                extra_gain_pct=extra_gain_pct,
                            )
                            self.speech_finished.emit(clean_chunk)
                            if not self._cancel_batch:
                                self.status_changed.emit(tr("tts.status.done_ms", "Готово ({ms} мс)", ms=synth_ms))
                        else:
                            raise RuntimeError(tr("tts.err.empty_audio", "TTS-модель не повернула звук для цього тексту."))
            except Exception as exc:
                from core.llama_tts import Qwen3SynthesisCancelledError
                from core.omnivoice_tts import OmniVoiceSynthesisCancelledError
                from core.telegram_silero import SynthesisCancelledError

                if self._cancel_batch or isinstance(exc, (SynthesisCancelledError, Qwen3SynthesisCancelledError, OmniVoiceSynthesisCancelledError)):
                    self.status_changed.emit(tr("tts.status.stopped", "Зупинено"))
                    continue
                raw_err = format_exception_message(exc)
                translated_err = translate_error_message(raw_err) or raw_err
                print(f"\n[TTSManager] Помилка синтезу: {raw_err}", file=sys.stderr)
                if translated_err != raw_err:
                    print(f"[TTSManager] Переклад ({get_ui_language()}): {translated_err}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                self.error_occurred.emit(translated_err)
                self.status_changed.emit(tr("tts.status.synth_error", "Помилка синтезу"))
            finally:
                self._is_synthesizing = False
                self._current_is_viewer_chat = False
                self._current_chat_user = ""
                self._current_chat_msg_id = ""
                self._queue.task_done()

    # ------------------------------------------------------------------
    # Диспетчер синтезу по кожному движку
    # ------------------------------------------------------------------

    def _synthesize_dispatch(
        self,
        engine_id: str,
        voice_id: str,
        text: str,
        speed: float,
    ) -> Tuple[np.ndarray, int]:
        if engine_id == "piper":
            return self._synth_piper(voice_id, text, speed)
        if engine_id == "silero":
            return self._synth_silero(voice_id, text, speed)
        if engine_id == "holos":
            return self._synth_holos(voice_id, text, speed)
        if engine_id == "teratts":
            return self._synth_teratts(voice_id, text, speed)
        if engine_id == "kokoro_82m":
            from core.llama_tts import resolve_kokoro_device, synthesize_llama_tts

            voices = ENGINE_SPECS.get(engine_id, {}).get("voices", [])
            voice_entry = next((v for v in voices if str(v.get("id")) == voice_id), None)
            speaker = str((voice_entry or {}).get("speaker") or "af_heart")
            if resolve_kokoro_device() == "cpu":
                from core.kokoro_onnx_tts import kokoro_onnx_engine
                return kokoro_onnx_engine.synthesize(
                    format_stress_marks(text, "strip"),
                    voice=speaker,
                    speed=speed,
                )
            lang = str((voice_entry or {}).get("lang") or self._current_lang() or "en")
            return synthesize_llama_tts(
                engine_id,
                format_stress_marks(text, "strip"),
                lang=lang,
                voice=speaker,
                speed=speed,
            )
        if engine_id in ("zonos3_mini", "soprano_1_1"):
            from core.llama_tts import synthesize_llama_tts

            voices = ENGINE_SPECS.get(engine_id, {}).get("voices", [])
            voice_entry = next((v for v in voices if str(v.get("id")) == voice_id), None)
            lang = str((voice_entry or {}).get("lang") or self._current_lang() or "en")
            speaker = str((voice_entry or {}).get("speaker") or "")
            return synthesize_llama_tts(
                engine_id,
                format_stress_marks(text, "strip"),
                lang=lang,
                voice=speaker,
                speed=speed,
            )
        if engine_id == "qwen3_tts":
            from core.llama_tts import synthesize_qwen3_tts

            voices = ENGINE_SPECS.get(engine_id, {}).get("voices", [])
            voice_entry = next((v for v in voices if str(v.get("id")) == voice_id), None)
            lang = str((voice_entry or {}).get("lang") or (voice_id.rsplit("_", 1)[-1] if "_" in voice_id else "en") or "en")
            return synthesize_qwen3_tts(format_stress_marks(text, "strip"), lang=lang, speed=speed)
        if engine_id == "omnivoice":
            from core.omnivoice_tts import synthesize_omnivoice_tts

            voices = ENGINE_SPECS.get(engine_id, {}).get("voices", [])
            voice_entry = next((v for v in voices if str(v.get("id")) == voice_id), None)
            lang = str((voice_entry or {}).get("lang") or self._current_lang() or "uk")
            return synthesize_omnivoice_tts(format_stress_marks(text, "strip"), lang=lang, speed=speed)
        if engine_id.startswith("online:"):
            return self._synth_online_item(engine_id, voice_id, text, speed)
        raise RuntimeError(tr("tts.err.select_active_voice", "Оберіть активний голос у розділі «Голос та Озвучка»."))

    def _synth_piper(self, voice_id: str, text: str, speed: float) -> Tuple[np.ndarray, int]:
        from piper.voice import PiperVoice, SynthesisConfig

        cur_lang = self._current_lang()
        voices = list(ENGINE_SPECS["piper"]["voices"])
        v_cfg = next((v for v in voices if v["id"] == voice_id), None)
        if v_cfg is None:
            lang_fallback = next((v for v in voices if v.get("lang") == cur_lang), None)
            if lang_fallback is None:
                dyn_voices, _ = self._get_dynamic_piper_for_lang(cur_lang)
                if dyn_voices:
                    voices.extend(dyn_voices)
                lang_fallback = next((v for v in voices if v.get("lang") == cur_lang), voices[0])
            v_cfg = next((v for v in voices if v["id"] == voice_id), lang_fallback)
        onnx_path = ENGINES_DIR / "piper" / v_cfg["onnx"]
        if not onnx_path.exists():
            raise RuntimeError(tr("tts.err.model_not_installed", "Модель {name} не встановлена. Натисніть «Встановити» у списку локальних TTS.", name="Piper"))

        with self._model_lock:
            key = str(onnx_path)
            if key not in self._piper_cache:
                self._piper_cache[key] = PiperVoice.load(str(onnx_path), use_cuda=False)
            voice = self._piper_cache[key]

        v_lang = str(v_cfg.get("lang") or cur_lang).strip().lower()

        def _prepare_piper_text(raw: str) -> str:
            if v_lang == "ru":
                return format_stress_marks(raw, "combining")
            if v_lang == "uk":
                def _phonemize_uk_word(word: str) -> List[str]:
                    return [ph for sent in voice.phonemize(word) for ph in sent]

                return apply_piper_uk_stress_phonemes(raw, _phonemize_uk_word)
            return adapt_text_for_local_tts(format_stress_marks(raw, "strip"), v_lang)

        synth_text = _prepare_piper_text(text)

        length_scale = 1.0 / max(0.5, min(2.0, speed))
        syn_cfg = SynthesisConfig(speaker_id=v_cfg.get("speaker"), length_scale=length_scale)
        chunks = list(voice.synthesize(synth_text, syn_config=syn_cfg))
        if not chunks and v_lang in ("uk", "ru"):
            fallback_raw = adapt_text_for_local_tts(text, v_lang)
            fallback_text = _prepare_piper_text(fallback_raw)
            if fallback_text and fallback_text != synth_text:
                chunks = list(voice.synthesize(fallback_text, syn_config=syn_cfg))
        if not chunks:
            raise RuntimeError(
                tr(
                    "tts.err.no_speakable_chars",
                    "Не вдалося озвучити текст «{text}» цією моделлю ({lang}): немає підтримуваних символів.",
                    text=text[:40],
                    lang=v_lang,
                )
            )

        sr = int(chunks[0].sample_rate)
        arrays = [np.asarray(c.audio_float_array, dtype=np.float32) for c in chunks]
        self._warmed_models.add(("piper", str(onnx_path)))
        return np.concatenate(arrays), sr

    def _synth_silero(self, voice_id: str, text: str, speed: float) -> Tuple[np.ndarray, int]:
        voices = ENGINE_SPECS["silero"]["voices"]
        v_cfg = next((v for v in voices if v["id"] == voice_id), voices[0])
        v_lang = str(v_cfg.get("lang") or "uk").strip().lower()
        pt_path = ENGINES_DIR / "silero" / v_cfg["model"]
        if not pt_path.exists():
            raise RuntimeError(tr("tts.err.model_not_installed", "Модель {name} не встановлена. Натисніть «Встановити» у списку локальних TTS.", name="Silero"))

        torch = _get_torch()
        with self._model_lock:
            key = str(pt_path)
            if key not in self._silero_cache:
                import warnings
                with self._import_lock:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=SyntaxWarning)
                        warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
                        importer = torch.package.PackageImporter(key)
                        model = importer.load_pickle("tts_models", "model")
                    model.to(torch.device("cpu"))
                    # Silero v5 під час load_pickle примусово скидає torch.set_num_threads(1),
                    # через що інференс іде на 1 ядрі в ~2.5× повільніше — повертаємо фізичні ядра.
                    try:
                        phys_threads = max(2, min(8, (os.cpu_count() or 4) // 2))
                        torch.set_num_threads(phys_threads)
                        if hasattr(torch, "set_flush_denormal"):
                            torch.set_flush_denormal(True)
                    except Exception:
                        pass
                    self._silero_importers[key] = importer
                    self._silero_cache[key] = model
            model = self._silero_cache[key]

        synth_text = adapt_text_for_local_tts(
            text,
            v_lang,
            uk_to_ru=(v_cfg.get("model") == "v5_6_ru.pt"),
        )
        synth_text = format_stress_marks(synth_text, "plus" if v_lang in ("uk", "ru") else "strip")
        symbols = getattr(model, "symbols", None)
        if symbols:
            speakable = [ch for ch in synth_text if ch.isalnum() and ch.lower() in symbols]
            if not speakable:
                raise RuntimeError(
                    tr(
                        "tts.err.no_speakable_chars",
                        "Не вдалося озвучити текст «{text}» цією моделлю ({lang}): немає підтримуваних символів.",
                        text=text[:40],
                        lang=v_lang,
                    )
                )

        sr = 48000
        try:
            inf_ctx = torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()
            with inf_ctx:
                audio_tensor = model.apply_tts(text=synth_text, speaker=v_cfg["speaker"], sample_rate=sr)
        except ValueError as ve:
            if not str(ve).strip():
                raise RuntimeError(
                    tr(
                        "tts.err.silero_empty_value_error",
                        "Модель Silero ({lang}) не змогла озвучити фразу «{text}» (відхилено символи тексту).",
                        lang=v_lang,
                        text=synth_text[:50],
                    )
                ) from ve
            raise
        samples = audio_tensor.detach().cpu().numpy().astype(np.float32).squeeze()
        if abs(speed - 1.0) > 0.04 and len(samples) > 100:
            from scipy.signal import resample
            new_len = max(1, int(len(samples) / speed))
            samples = resample(samples, new_len).astype(np.float32)
        self._warmed_models.add(("silero", key))
        return samples, sr

    @staticmethod
    def _find_uk_stress_trie_path() -> Optional[Path]:
        """Знаходить data/stress.trie без імпорту пакета ukrainian_word_stress (щоб не тягнути stanza/torch)."""
        import importlib.util

        candidates: List[Path] = [
            BUNDLE_DIR / "ukrainian_word_stress" / "data" / "stress.trie",
            BASE_DIR / "ukrainian_word_stress" / "data" / "stress.trie",
        ]
        try:
            spec = importlib.util.find_spec("ukrainian_word_stress")
            if spec is not None:
                for loc in spec.submodule_search_locations or ():
                    candidates.append(Path(loc) / "data" / "stress.trie")
                if spec.origin:
                    candidates.append(Path(spec.origin).resolve().parent / "data" / "stress.trie")
        except Exception:
            pass
        for cand in candidates:
            if cand.is_file():
                return cand
        return None

    @staticmethod
    def _build_fast_uk_stress():
        """Миттєвий розстановщик наголосів (marisa_trie + ukrainian_accentor) без очікування stanza."""
        import warnings
        import marisa_trie  # type: ignore

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning)
            warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
            import ukrainian_accentor as accentor

        dict_path = TTSEngineManager._find_uk_stress_trie_path()
        if dict_path is not None:
            rec_sep = b"\xff"
        else:
            from importlib import resources as pkg_resources
            from ukrainian_word_stress.tags import TAGS

            dict_path = pkg_resources.files("ukrainian_word_stress").joinpath("data/stress.trie")
            rec_sep = TAGS["Record-separator"]

        trie = marisa_trie.BytesTrie()
        trie.load(str(dict_path))
        word_re = re.compile(r"[А-Яа-яІіЇїЄєҐґ'’\u0301]+")

        def _fast_stress(s: str) -> str:
            def _rep(m: re.Match) -> str:
                w = m.group(0)
                if "\u0301" in w:
                    return w
                for cand in (w, w.lower(), w.title()):
                    if cand in trie:
                        val = trie[cand][0]
                        if rec_sep not in val:
                            res = w
                            for p in sorted((int(b) for b in val), reverse=True):
                                res = res[:p] + "\u0301" + res[p:]
                            return res
                        break
                return accentor.process(w, mode="stress")

            return word_re.sub(_rep, s)

        return _fast_stress

    def _start_holos_stanza_async(self) -> None:
        if self._holos_stanza_started or self._holos_stanza_stress is not None:
            return
        self._holos_stanza_started = True
        import importlib.util

        if importlib.util.find_spec("stanza") is None or not (Path.home() / "stanza_resources" / "uk").is_dir():
            return

        def _bg():
            try:
                from ukrainian_word_stress import Stressifier
                self._holos_stanza_stress = Stressifier()
            except Exception as exc:
                print(f"[Holos] Фонове завантаження stanza пропущено: {exc}", flush=True)

        threading.Thread(target=_bg, daemon=True, name="HolosStanzaBg").start()

    @staticmethod
    def _apply_holos_stress_preserving_manual(text: str, stress_fn: Callable[[str], str]) -> str:
        word_pat = re.compile(r"[А-Яа-яІіЇїЄєҐґ'’\u0301]+")
        parts: List[str] = []
        last = 0
        for m in word_pat.finditer(text):
            w = m.group(0)
            if "\u0301" in w:
                prefix = text[last : m.start()]
                if prefix:
                    parts.append(stress_fn(prefix))
                parts.append(w)
                last = m.end()
        tail = text[last:]
        if tail:
            parts.append(stress_fn(tail))
        return "".join(parts)

    def _synth_holos(self, voice_id: str, text: str, speed: float) -> Tuple[np.ndarray, int]:
        import_started = time.perf_counter()
        with self._import_lock:
            from holos_onnx import HolosTTS
            from ipa_uk import ipa
        import_ms = int((time.perf_counter() - import_started) * 1000)

        voices = ENGINE_SPECS["holos"]["voices"]
        v_cfg = next((v for v in voices if v["id"] == voice_id), voices[0])
        local_dir = ENGINES_DIR / "holos"
        bundle_dir = local_dir / "onnx"
        required_bundle_files = ("encoder.onnx", "prosody.onnx", "decoder.onnx", "holos_onnx.json", "voices.npz")
        if not all((bundle_dir / name).exists() for name in required_bundle_files):
            raise RuntimeError(tr("tts.err.model_not_installed", "Модель {name} не встановлена. Натисніть «Встановити» у списку локальних TTS.", name="HolosTTS"))

        with self._model_lock:
            if self._holos_model is None:
                from concurrent.futures import ThreadPoolExecutor
                init_started = time.perf_counter()

                def _load_stress():
                    return self._build_fast_uk_stress()

                def _load_model():
                    return HolosTTS(
                        str(bundle_dir),
                        providers=["CPUExecutionProvider"],
                    )

                with ThreadPoolExecutor(max_workers=2) as pool:
                    fut_stress = pool.submit(_load_stress)
                    fut_model = pool.submit(_load_model)
                    self._holos_stress = fut_stress.result()
                    self._holos_model = fut_model.result()
                init_ms = int((time.perf_counter() - init_started) * 1000)
                print(f"[Holos] ONNX FP32 ініціалізація: імпорт {import_ms} мс, модель і словник {init_ms} мс", flush=True)
                self._start_holos_stanza_async()

        synth_text = format_stress_marks(adapt_text_for_local_tts(text, "uk"), "combining")
        if not re.search(r"[А-Яа-яІіЇїЄєҐґ0-9]", synth_text):
            raise RuntimeError(
                tr(
                    "tts.err.no_speakable_chars",
                    "Не вдалося озвучити текст «{text}» цією моделлю ({lang}): немає підтримуваних символів.",
                    text=text[:40],
                    lang="uk",
                )
            )
        stress_fn = self._holos_stanza_stress or self._holos_stress
        if "\u0301" in synth_text:
            stressed_text = self._apply_holos_stress_preserving_manual(synth_text, stress_fn)
        else:
            stressed_text = stress_fn(synth_text)
        phonemes = ipa(stressed_text)
        inference_started = time.perf_counter()
        audio = self._holos_model.generate(phonemes, voice=v_cfg["speaker"], speed=float(speed))
        if not self._holos_first_inference_logged:
            self._holos_first_inference_logged = True
            inference_ms = int((time.perf_counter() - inference_started) * 1000)
            print(f"[Holos] Перший синтез ONNX FP32: {inference_ms} мс", flush=True)
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        samples = np.asarray(audio, dtype=np.float32).squeeze()
        sr = int(getattr(self._holos_model, "sample_rate", 24000))
        self._warmed_models.add(("holos", "holos"))
        return samples, sr

    def _import_ruaccent_fast(self):
        """Імпортує RUAccent через легкий Rust-токенізатор `tokenizers` без важкого `transformers`."""
        if "ruaccent" in sys.modules:
            from ruaccent import RUAccent
            return RUAccent

        import tokenizers

        class _FastPreTrainedTokenizer:
            def __init__(
                self,
                pad_token="[pad]",
                unk_token="[unk]",
                bos_token="[bos]",
                eos_token="[eos]",
                do_lower_case=False,
                **kw,
            ):
                self.pad_token = pad_token
                self.unk_token = unk_token
                self.bos_token = bos_token
                self.eos_token = eos_token
                self.bos_token_id = self.vocab.get(bos_token, 2)
                self.eos_token_id = self.vocab.get(eos_token, 3)
                self.pad_token_id = self.vocab.get(pad_token, 0)
                self.unk_token_id = self.vocab.get(unk_token, 1)

            @classmethod
            def from_pretrained(cls, path, **kw):
                cfg_p = os.path.join(path, "tokenizer_config.json")
                cfg = json.load(open(cfg_p, encoding="utf-8")) if os.path.exists(cfg_p) else {}
                allowed = ("pad_token", "unk_token", "bos_token", "eos_token", "do_lower_case")
                return cls(
                    vocab_file=os.path.join(path, "vocab.txt"),
                    **{k: cfg[k] for k in allowed if k in cfg},
                )

            def __call__(self, word, return_tensors="np", **kw):
                toks = self._tokenize(word)
                ids = [self._convert_token_to_id(t) for t in toks]
                seq = self.build_inputs_with_special_tokens(ids)
                return {
                    "input_ids": np.array([seq], dtype=np.int64),
                    "attention_mask": np.ones((1, len(seq)), dtype=np.int64),
                    "token_type_ids": np.zeros((1, len(seq)), dtype=np.int64),
                }

        class _FastAutoTokenizerAdapter:
            def __init__(self, path: str):
                self._lock = threading.Lock()
                self._tokenizer = tokenizers.Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
                cfg_p = os.path.join(path, "tokenizer_config.json")
                cfg = json.load(open(cfg_p, encoding="utf-8")) if os.path.exists(cfg_p) else {}
                pad_tok = cfg.get("pad_token") or "[PAD]"
                unk_tok = cfg.get("unk_token") or "[UNK]"
                if isinstance(pad_tok, dict):
                    pad_tok = pad_tok.get("content", "[PAD]")
                if isinstance(unk_tok, dict):
                    unk_tok = unk_tok.get("content", "[UNK]")
                self.pad_token_id = self._tokenizer.token_to_id(pad_tok) or 0
                self.unk_token_id = self._tokenizer.token_to_id(unk_tok) or 0
                tcls = str(cfg.get("tokenizer_class", ""))
                self.model_input_names = cfg.get("model_input_names") or (
                    ["input_ids", "attention_mask"]
                    if ("Roberta" in tcls or "DistilBert" in tcls)
                    else ["input_ids", "token_type_ids", "attention_mask"]
                )
                self._is_wordpiece = type(self._tokenizer.model).__name__ == "WordPiece"

            @classmethod
            def from_pretrained(cls, path, **kw):
                return cls(str(path))

            def convert_ids_to_tokens(self, idx):
                return self._tokenizer.id_to_token(int(idx))

            def convert_tokens_to_string(self, tokens):
                if self._is_wordpiece:
                    return " ".join(tokens).replace(" ##", "")
                ids = [self._tokenizer.token_to_id(t) for t in tokens]
                return self._tokenizer.decode([i if i is not None else self.unk_token_id for i in ids])

            def __call__(
                self,
                text,
                text_pair=None,
                max_length=None,
                truncation=False,
                padding=False,
                return_tensors=None,
                return_offsets_mapping=False,
                return_special_tokens_mask=False,
                **kw,
            ):
                with self._lock:
                    if truncation and max_length:
                        self._tokenizer.enable_truncation(max_length=int(max_length))
                    else:
                        self._tokenizer.no_truncation()
                    is_batch = isinstance(text, (list, tuple))
                    if padding:
                        self._tokenizer.enable_padding(pad_id=self.pad_token_id, length=None)
                    else:
                        self._tokenizer.no_padding()
                    if is_batch:
                        items = list(zip(text, text_pair)) if text_pair is not None else list(text)
                        encs = self._tokenizer.encode_batch(items)
                    else:
                        encs = [self._tokenizer.encode(text, text_pair)]
                out = {
                    "input_ids": np.array([e.ids for e in encs], dtype=np.int64),
                    "attention_mask": np.array([e.attention_mask for e in encs], dtype=np.int64),
                }
                if "token_type_ids" in self.model_input_names:
                    out["token_type_ids"] = np.array([e.type_ids for e in encs], dtype=np.int64)
                if return_offsets_mapping:
                    out["offset_mapping"] = np.array([e.offsets for e in encs], dtype=np.int64)
                if return_special_tokens_mask:
                    out["special_tokens_mask"] = np.array([e.special_tokens_mask for e in encs], dtype=np.int64)
                return out

        shim = types.ModuleType("transformers")
        shim.AutoTokenizer = _FastAutoTokenizerAdapter
        shim.PreTrainedTokenizer = _FastPreTrainedTokenizer

        _SENTINEL = object()
        prev_transformers = sys.modules.get("transformers", _SENTINEL)
        sys.modules["transformers"] = shim
        try:
            from ruaccent import RUAccent
            return RUAccent
        finally:
            if prev_transformers is _SENTINEL:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = prev_transformers

    @staticmethod
    def _make_sqlite_kv_dict(db_path: str, overlay: Optional[Dict[str, str]] = None):
        """Повертає потокобезпечний dict-адаптер поверх SQLite WITHOUT ROWID із кешем у пам'яті."""

        class _SqliteKVDict:
            def __init__(self, path: str, ov: Optional[Dict[str, str]] = None) -> None:
                self._lock = threading.Lock()
                self._con = sqlite3.connect(path, check_same_thread=False)
                self._cache: Dict[str, str] = dict(ov or {})

            def __contains__(self, key: str) -> bool:
                if key in self._cache:
                    return True
                with self._lock:
                    row = self._con.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
                if row is None:
                    return False
                self._cache[key] = row[0]
                return True

            def __getitem__(self, key: str) -> str:
                if key in self._cache:
                    return self._cache[key]
                with self._lock:
                    row = self._con.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
                if row is None:
                    raise KeyError(key)
                self._cache[key] = row[0]
                return row[0]

            def get(self, key: str, default: Any = None) -> Any:
                if key in self._cache:
                    return self._cache[key]
                with self._lock:
                    row = self._con.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
                if row is None:
                    return default
                self._cache[key] = row[0]
                return row[0]

            def close(self) -> None:
                try:
                    with self._lock:
                        self._con.close()
                except Exception:
                    pass

        return _SqliteKVDict(db_path, overlay)

    @staticmethod
    def _build_sqlite_kv(db_path: Path, pairs_iter) -> None:
        tmp = Path(str(db_path) + ".tmp")
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        con = sqlite3.connect(str(tmp))
        try:
            con.execute("PRAGMA journal_mode=OFF")
            con.execute("PRAGMA synchronous=OFF")
            con.execute("PRAGMA temp_store=MEMORY")
            con.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT) WITHOUT ROWID")
            batch = []
            for k, v in pairs_iter:
                batch.append((k, v))
                if len(batch) >= 50000:
                    con.executemany("INSERT OR REPLACE INTO kv VALUES (?,?)", batch)
                    batch.clear()
            if batch:
                con.executemany("INSERT OR REPLACE INTO kv VALUES (?,?)", batch)
            con.commit()
        finally:
            con.close()
        if db_path.exists():
            db_path.unlink(missing_ok=True)
        os.replace(str(tmp), str(db_path))

    def _ensure_teratts_sqlite_caches(self, local_dir: Path) -> Tuple[Path, Path]:
        ra_dir = local_dir / "ruaccent"
        db_acc = ra_dir / "accents_cache.sqlite"
        acc_gz = ra_dir / "dictionary" / "accents.json.gz"
        if (not db_acc.exists() or db_acc.stat().st_size == 0) and acc_gz.exists():
            with gzip.open(str(acc_gz)) as f:
                data = json.load(f)
            self._build_sqlite_kv(db_acc, data.items())
            del data

        exp_dir = local_dir / "TeraTTS" / "natasha-g2p-vits" / "exported"
        db_tts = exp_dir / "dict_cache.sqlite"
        dict_txt = exp_dir / "dictionary.txt"
        if (not db_tts.exists() or db_tts.stat().st_size == 0) and dict_txt.exists():
            def _iter_tts():
                with open(str(dict_txt), "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        sp = line.split(None, 1)
                        if len(sp) == 2:
                            yield sp[0], " ".join(sp[1].split())

            self._build_sqlite_kv(db_tts, _iter_tts())

        return db_acc, db_tts

    def _load_teratts_fast(self, local_dir: Path) -> Tuple[Any, Any]:
        """Швидке завантаження RUAccent + TeraTTS (~5-8 с) із збереженням повних словників через SQLite KV."""
        from concurrent.futures import ThreadPoolExecutor

        with self._import_lock:
            RUAccent = self._import_ruaccent_fast()
            from TeraTTS import TTS

        ruaccent_dir = local_dir / "ruaccent"
        ruaccent_dir.mkdir(parents=True, exist_ok=True)
        db_acc, db_tts = self._ensure_teratts_sqlite_caches(local_dir)

        dict_dir = ruaccent_dir / "dictionary"
        nn_dir = ruaccent_dir / "nn"
        can_fast_acc = (
            db_acc.exists()
            and db_tts.exists()
            and (dict_dir / "omographs.json.gz").exists()
            and (dict_dir / "yo_words.json.gz").exists()
            and (dict_dir / "yo_homographs.json.gz").exists()
            and (nn_dir / "nn_omograph" / "turbo" / "model.onnx").exists()
        )
        if not can_fast_acc:
            gc_was = gc.isenabled()
            gc.disable()
            try:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    fut_acc = pool.submit(
                        lambda: (lambda a: (a.load(omograph_model_size="turbo", use_dictionary=True, workdir=str(ruaccent_dir)), a)[1])(RUAccent())
                    )
                    fut_tts = pool.submit(
                        lambda: TTS("TeraTTS/natasha-g2p-vits", add_time_to_end=0.8, tokenizer_load_dict=True, save_path=str(local_dir))
                    )
                    return fut_acc.result(), fut_tts.result()
            finally:
                if gc_was:
                    gc.enable()

        acc = RUAccent()
        acc.tiny_mode = False
        acc.workdir = str(ruaccent_dir)
        acc.custom_dict = {}
        acc.accents = self._make_sqlite_kv_dict(str(db_acc), overlay=acc.letters_accent)

        def _load_gz_json(path: Path):
            with gzip.open(str(path)) as f:
                return json.load(f)

        gc_was = gc.isenabled()
        gc.disable()
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                f_omog_dict = pool.submit(_load_gz_json, dict_dir / "omographs.json.gz")
                f_yo_dict = pool.submit(_load_gz_json, dict_dir / "yo_words.json.gz")
                f_yoh_dict = pool.submit(_load_gz_json, dict_dir / "yo_homographs.json.gz")
                f_omo = pool.submit(acc.omograph_model.load, str(nn_dir / "nn_omograph" / "turbo"), "CPU")
                f_acc = pool.submit(acc.accent_model.load, str(nn_dir / "nn_accent"), "CPU")
                f_yoh = pool.submit(acc.yo_homograph_model.load, str(nn_dir / "nn_yo_homograph_resolver"), "CPU")
                f_str = pool.submit(acc.stress_usage_predictor.load, str(nn_dir / "nn_stress_usage_predictor"), "CPU")
                f_tts = pool.submit(
                    lambda: TTS("TeraTTS/natasha-g2p-vits", add_time_to_end=0.8, tokenizer_load_dict=False, save_path=str(local_dir))
                )
                acc.omographs = f_omog_dict.result()
                acc.omographs["коса"] = ["к+оса", "кос+а"]
                acc.yo_words = f_yo_dict.result()
                acc.yo_homographs = f_yoh_dict.result()
                f_omo.result()
                f_acc.result()
                f_yoh.result()
                f_str.result()
                tts = f_tts.result()
        finally:
            if gc_was:
                gc.enable()

        tts.tokenizer.dic = self._make_sqlite_kv_dict(str(db_tts))
        return acc, tts

    def _synth_teratts(self, voice_id: str, text: str, speed: float) -> Tuple[np.ndarray, int]:
        local_dir = ENGINES_DIR / "teratts"
        if not (local_dir / "ready.marker").exists():
            raise RuntimeError(tr("tts.err.model_not_installed", "Модель {name} не встановлена. Натисніть «Встановити» у списку локальних TTS.", name="TeraTTS"))

        with self._model_lock:
            if self._teratts_model is None:
                self._teratts_accent, self._teratts_model = self._load_teratts_fast(local_dir)

        synth_text = format_stress_marks(adapt_text_for_local_tts(text, "ru", uk_to_ru=True), "plus")
        if not re.search(r"[А-Яа-яЁё0-9]", synth_text):
            raise RuntimeError(
                tr(
                    "tts.err.no_speakable_chars",
                    "Не вдалося озвучити текст «{text}» цією моделлю ({lang}): немає підтримуваних символів.",
                    text=text[:40],
                    lang="ru",
                )
            )
        if "+" in synth_text:
            accented = self._teratts_accent.process_all(
                synth_text,
                skip_regex=r"[А-Яа-яЁё]*\+[А-Яа-яЁё]+",
            )
        else:
            accented = self._teratts_accent.process_all(synth_text)
        len_scale = 1.1 / max(0.5, min(2.0, speed))
        audio = self._teratts_model(accented, lenght_scale=len_scale)
        samples = np.asarray(audio, dtype=np.float32).squeeze()
        self._warmed_models.add(("teratts", "teratts"))
        return samples, 22050

    def _synth_online_item(self, engine_id: str, voice_id: str, text: str, speed: float) -> Tuple[np.ndarray, int]:
        tag = engine_id.split(":", 1)[1] if ":" in engine_id else "edge"
        if tag == "edge":
            return self._synth_edge_tts(format_stress_marks(text, "strip"), voice_id, speed)

        online_items: List[Dict[str, Any]] = config.get("online_items", [])
        item = next((x for x in online_items if str(x.get("id")) == tag), None)
        if not item:
            raise RuntimeError(tr("tts.err.online_removed", "Вибраний онлайн-сервіс було видалено зі списку."))

        provider = str(item.get("provider", "")).strip()
        api_key = str(item.get("api_key", "")).strip()

        if provider == "silero_bot":
            return self._synth_silero_bot(format_stress_marks(text, "plus"), voice_id, speed, item)

        clean_text = format_stress_marks(text, "strip")
        if provider == "edge":
            if not bool(item.get("installed", False)):
                raise RuntimeError(tr("tts.err.edge_not_installed_tab", "Бібліотека Edge-TTS ще не встановлена. Натисніть «Встановити» у вкладці «Онлайн-голоси»."))
            return self._synth_edge_tts(clean_text, voice_id, speed)
        if provider == "google":
            return self._synth_google_cloud(clean_text, voice_id, api_key, speed, item)
        if provider == "elevenlabs":
            return self._synth_elevenlabs(clean_text, voice_id, api_key, item)
        if provider == "openai":
            return self._synth_openai(clean_text, voice_id, api_key, speed, item)
        if provider == "fish_audio":
            return self._synth_fish_audio(clean_text, voice_id, api_key, speed, item)
        if provider == "gemini_tts":
            return self._synth_gemini_tts(clean_text, voice_id, api_key, item)
        raise RuntimeError(tr("tts.err.unknown_online_provider", "Невідомий онлайн-провайдер: {provider}", provider=provider))

    def _synth_silero_bot(
        self,
        text: str,
        voice_id: str,
        speed: float,
        item: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, int]:
        from core.telegram_silero import telegram_silero

        timeout_sec = float((item or {}).get("timeout_sec", 60) or 60)
        audio_bytes, _mime = telegram_silero.synthesize(
            text=text,
            voice_code=voice_id,
            timeout_sec=timeout_sec,
        )
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1).astype(np.float32)
        if abs(speed - 1.0) > 0.02:
            data = _fx_speed_and_pitch_array(data, int(sr), speed_rate=speed)
        return data, int(sr)

    def _synth_edge_tts(self, text: str, voice_name: str, speed: float) -> Tuple[np.ndarray, int]:
        try:
            import edge_tts
        except ImportError:
            raise RuntimeError(tr("tts.err.edge_not_installed_near", "Бібліотека edge-tts не встановлена. Натисніть «Встановити» біля Microsoft Edge-TTS у вкладці «Онлайн-голоси»."))

        if not voice_name:
            voice_name = ONLINE_PROVIDERS["edge"]["voices"][0]["id"]
        voice_name = str(voice_name).split("::", 1)[0].strip()

        pct = int(round((speed - 1.0) * 100))
        rate_str = f"{pct:+d}%"

        async def _fetch() -> bytes:
            import aiohttp

            last_exc: Optional[BaseException] = None
            for attempt in range(2):
                buf = bytearray()
                connector, http_proxy = get_edge_tts_proxy_config()
                if attempt > 0 and connector is None and http_proxy is None:
                    connector = aiohttp.TCPConnector(family=socket.AF_INET, force_close=True)
                try:
                    comm = edge_tts.Communicate(
                        text,
                        voice_name,
                        rate=rate_str,
                        connector=connector,
                        proxy=http_proxy,
                    )
                    async for chunk in comm.stream():
                        if chunk["type"] == "audio":
                            buf.extend(chunk["data"])
                    return bytes(buf)
                except (aiohttp.ClientError, OSError, TimeoutError) as exc:
                    last_exc = exc
                    if buf or attempt >= 1:
                        raise
                    await asyncio.sleep(0.3)
            if last_exc is not None:
                raise last_exc
            return b""

        loop = asyncio.new_event_loop()
        loop.set_exception_handler(
            lambda lp, ctx: None if isinstance(ctx.get("exception"), ConnectionResetError) else lp.default_exception_handler(ctx)
        )
        try:
            audio_bytes = loop.run_until_complete(_fetch())
        finally:
            loop.close()

        if not audio_bytes:
            raise RuntimeError(tr("tts.err.edge_empty", "Edge-TTS повернув порожню відповідь. Перевірте інтернет-з'єднання."))
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1).astype(np.float32)
        return data, int(sr)

    def _synth_elevenlabs(self, text: str, voice_id: str, api_key: str, item: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, int]:
        if not api_key:
            raise RuntimeError(tr("tts.err.enter_key_online", "Введіть API ключ у блоці {provider} (вкладка «Онлайн-голоси»).", provider="ElevenLabs API"))
        custom_vid = str((item or {}).get("reference_id", "")).strip()
        if custom_vid:
            voice_id = custom_vid
        if not voice_id:
            voice_id = ONLINE_PROVIDERS["elevenlabs"]["voices"][0]["id"]
        model_id = str((item or {}).get("model", "")).strip() or "eleven_flash_v2_5"
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
        payload = {
            "text": text,
            "model_id": model_id,
        }
        resp = self._http_session.post(url, headers=headers, json=payload, proxies=get_active_api_proxies(), timeout=20)
        if resp.status_code != 200:
            raise RuntimeError(tr("tts.err.provider_api", "{provider} помилка ({code}): {msg}", provider="ElevenLabs API", code=resp.status_code, msg=resp.text[:140]))
        data, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1).astype(np.float32)
        return data, int(sr)

    def _synth_openai(self, text: str, voice: str, api_key: str, speed: float, item: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, int]:
        clean_key = re.sub(r"^bearer\s+", "", str(api_key or "").strip().strip("\"'"), flags=re.IGNORECASE).strip()
        if not clean_key:
            raise RuntimeError(tr("tts.err.enter_key_online", "Введіть API ключ у блоці {provider} (вкладка «Онлайн-голоси»).", provider="OpenAI TTS API"))
        if not voice:
            voice = ONLINE_PROVIDERS["openai"]["voices"][0]["id"]
        model_id = str((item or {}).get("model", "")).strip() or "tts-1"
        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {clean_key}",
        }
        payload = {
            "model": model_id,
            "input": text,
            "voice": voice,
            "speed": max(0.25, min(4.0, speed)),
            "response_format": "wav",
        }
        resp = self._http_session.post(url, headers=headers, json=payload, proxies=get_active_api_proxies(), timeout=20)
        if resp.status_code != 200:
            raise RuntimeError(tr("tts.err.provider_api", "{provider} помилка ({code}): {msg}", provider="OpenAI TTS API", code=resp.status_code, msg=resp.text[:140]))
        data, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1).astype(np.float32)
        return data, int(sr)

    def _synth_google_cloud(
        self,
        text: str,
        voice_name: str,
        api_key: str,
        speed: float,
        item: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, int]:
        if not api_key:
            raise RuntimeError(tr("tts.err.enter_key_online", "Введіть API ключ у блоці {provider} (вкладка «Онлайн-голоси»).", provider="Google Cloud TTS API"))
        if not voice_name:
            voice_name = ONLINE_PROVIDERS["google"]["voices"][0]["id"]

        family, _has_free = classify_google_voice_model(voice_name)
        paid_enabled = bool((item or {}).get("paid_enabled", False))
        if not paid_enabled:
            allowed, used, limit = check_google_quota(family, len(text))
            if not allowed:
                if limit <= 0:
                    raise RuntimeError(
                        tr(
                            "tts.err.google_paid_only",
                            "Модель {family} не входить до безкоштовного ліміту Google Cloud. Увімкніть «💳 Перейти на платно» у вкладці «Онлайн-голоси».",
                            family=family,
                        )
                    )
                raise RuntimeError(
                    tr(
                        "tts.err.google_quota_reached",
                        "Досягнуто безпечний безкоштовний ліміт Google Cloud для {family}: {used} / {limit} симв. за поточний місяць (запас 50 000 до ліміту Google). Ліміт автоматично скинеться 1-го числа, або увімкніть «💳 Перейти на платно».",
                        family=family,
                        used=f"{used:,}",
                        limit=f"{limit:,}",
                    )
                )

        parts = voice_name.split("-")
        lang_code = "-".join(parts[:2]) if len(parts) >= 2 else "uk-UA"
        url = "https://texttospeech.googleapis.com/v1/text:synthesize"
        payload: Dict[str, Any] = {
            "input": {"text": text},
            "voice": {"languageCode": lang_code, "name": voice_name},
            "audioConfig": {"audioEncoding": "LINEAR16", "speakingRate": round(speed, 2)},
        }
        resp = self._http_session.post(url, params={"key": api_key}, json=payload, proxies=get_active_api_proxies(), timeout=20)
        if resp.status_code == 400 and ("speaking" in resp.text.lower() or "rate" in resp.text.lower()):
            payload["audioConfig"] = {"audioEncoding": "LINEAR16"}
            resp = self._http_session.post(url, params={"key": api_key}, json=payload, proxies=get_active_api_proxies(), timeout=20)
        if resp.status_code != 200:
            raise RuntimeError(tr("tts.err.provider_api", "{provider} помилка ({code}): {msg}", provider="Google Cloud TTS", code=resp.status_code, msg=resp.text[:140]))

        record_google_usage(family, len(text))

        b64 = resp.json().get("audioContent", "")
        raw_bytes = base64.b64decode(b64)
        data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1).astype(np.float32)
        return data, int(sr)

    def _synth_fish_audio(
        self,
        text: str,
        voice_id: str,
        api_key: str,
        speed: float,
        item: Dict[str, Any],
    ) -> Tuple[np.ndarray, int]:
        clean_key = re.sub(r"^bearer\s+", "", str(api_key or "").strip().strip("\"'"), flags=re.IGNORECASE).strip()
        if not clean_key:
            raise RuntimeError(tr("tts.err.enter_key_online", "Введіть API ключ у блоці {provider} (вкладка «Онлайн-голоси»).", provider="Fish Audio API"))

        ref_id = str(voice_id or "").strip()
        if not ref_id or ref_id == "default":
            ref_id = str(item.get("reference_id", "")).strip()

        model_ver = str(item.get("model", "s2.1-pro-free")).strip() or "s2.1-pro-free"
        if not bool(item.get("paid_enabled", False)):
            model_ver = "s2.1-pro-free"
        url = "https://api.fish.audio/v1/tts"
        headers = {
            "Authorization": f"Bearer {clean_key}",
            "Content-Type": "application/json",
            "model": model_ver,
        }
        payload: Dict[str, Any] = {
            "text": text,
            "format": "wav",
            "prosody": {
                "speed": round(max(0.5, min(2.0, float(speed))), 2),
            },
        }
        if ref_id:
            payload["reference_id"] = ref_id

        resp = self._http_session.post(url, headers=headers, json=payload, proxies=get_active_api_proxies(), timeout=25)
        if resp.status_code != 200:
            err_detail = resp.text[:160]
            try:
                err_json = resp.json()
                if isinstance(err_json, dict) and err_json.get("message"):
                    err_detail = str(err_json["message"])[:160]
            except Exception:
                pass
            if resp.status_code == 401:
                raise RuntimeError(
                    tr(
                        "tts.err.fish_401",
                        "Fish Audio API (401): Невірний або недійсний API-ключ (Invalid Token). Перевірте ключ у кабінеті fish.audio → API Keys та оновіть його у вкладці «Онлайн-голоси».",
                    )
                )
            raise RuntimeError(tr("tts.err.provider_api", "{provider} помилка ({code}): {msg}", provider="Fish Audio API", code=resp.status_code, msg=err_detail))

        data, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1).astype(np.float32)
        return data, int(sr)

    def _synth_gemini_tts(
        self,
        text: str,
        voice_name: str,
        api_key: str,
        item: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, int]:
        clean_key = str(api_key or "").strip().strip("\"'")
        if not clean_key:
            raise RuntimeError(tr("tts.err.enter_key_online", "Введіть API ключ у блоці {provider} (вкладка «Онлайн-голоси»).", provider="Google AI Studio (Gemini TTS)"))
        if not voice_name:
            voice_name = "Kore"
        avail = list((item or {}).get("available_models") or [])
        fallback_model = str(avail[0]).strip() if avail else "gemini-3.8-flash-lite-tts"
        model_id = str((item or {}).get("model", "")).strip() or fallback_model
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
        headers = {
            "x-goog-api-key": clean_key,
            "Content-Type": "application/json",
        }
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": voice_name,
                        }
                    }
                },
            },
        }
        resp = self._http_session.post(url, headers=headers, json=payload, proxies=get_active_api_proxies(), timeout=25)
        if resp.status_code != 200:
            err_detail = resp.text[:180]
            try:
                err_json = resp.json()
                if isinstance(err_json, dict) and isinstance(err_json.get("error"), dict):
                    err_detail = str(err_json["error"].get("message", err_detail))[:180]
            except Exception:
                pass
            raise RuntimeError(tr("tts.err.provider_api", "{provider} помилка ({code}): {msg}", provider="Gemini TTS API", code=resp.status_code, msg=err_detail))

        r_json = resp.json()
        inline = None
        for cand in r_json.get("candidates", []) or []:
            parts = ((cand.get("content") or {}).get("parts") or [])
            for p in parts:
                if isinstance(p, dict) and isinstance(p.get("inlineData"), dict) and p["inlineData"].get("data"):
                    inline = p["inlineData"]
                    break
            if inline:
                break
        if not inline:
            raise RuntimeError(tr("tts.err.gemini_no_audio", "Gemini TTS повернув відповідь без аудіо."))

        raw_bytes = base64.b64decode(inline["data"])
        if raw_bytes[:4] == b"RIFF":
            data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1).astype(np.float32)
            return data, int(sr)

        mime = str(inline.get("mimeType", ""))
        m_rate = re.search(r"rate=(\d+)", mime)
        sr = int(m_rate.group(1)) if m_rate else 24000
        pcm = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return pcm, sr

    @staticmethod
    def fetch_fish_audio_voices(api_key: str, lang: Optional[str] = None) -> List[Dict[str, Any]]:
        """Завантажує власні та популярні голоси з Fish Audio API для обраної мови."""
        key = re.sub(r"^bearer\s+", "", str(api_key or "").strip().strip("\"'"), flags=re.IGNORECASE).strip()
        if not key:
            raise RuntimeError(tr("tts.err.enter_key_first", "Спочатку введіть API ключ {provider}.", provider="Fish Audio"))
        target_lang = (lang or str(config.get("app_language", "uk"))).strip().lower() or "uk"
        headers = {"Authorization": f"Bearer {key}"}
        voices: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()

        for self_only in (True, False):
            params: Dict[str, Any] = {
                "page_size": 15 if self_only else 30,
                "sort_by": "task_count",
            }
            if self_only:
                params["self_only"] = "true"
            else:
                params["language"] = target_lang
            resp = requests.get("https://api.fish.audio/model", headers=headers, params=params, proxies=get_active_api_proxies(), timeout=15)
            if resp.status_code != 200:
                if self_only:
                    continue
                raise RuntimeError(tr("tts.err.provider_api", "{provider} помилка ({code}): {msg}", provider="Fish Audio API", code=resp.status_code, msg=resp.text[:140]))
            data = resp.json()
            items = data.get("items", []) if isinstance(data, dict) else []
            for m in items:
                if not isinstance(m, dict):
                    continue
                mid = str(m.get("_id") or m.get("id") or "").strip()
                title = str(m.get("title") or "").strip()
                if not mid or not title or mid in seen_ids:
                    continue
                seen_ids.add(mid)
                tags_low = " ".join(str(t).lower() for t in (m.get("tags") or []))
                gender: Optional[str] = None
                if "female" in tags_low or "woman" in tags_low or "girl" in tags_low:
                    gender = "female"
                elif "male" in tags_low or "man" in tags_low or "boy" in tags_low:
                    gender = "male"
                m_langs = [str(l).lower() for l in (m.get("languages") or []) if l]
                v_lang = "multi" if (self_only or not m_langs) else (target_lang if target_lang in m_langs else m_langs[0])
                prefix = "★ " if self_only else ""
                lang_badge = f" [{','.join(m_langs[:2])}]" if m_langs else ""
                is_primary_lang = (not self_only and len(m_langs) > 0 and m_langs[0] == target_lang)
                is_pure_lang = (not self_only and m_langs == [target_lang])
                voices.append({
                    "id": mid,
                    "name": f"{prefix}{title}{lang_badge}",
                    "gender": gender,
                    "lang": v_lang,
                    "languages": m_langs,
                    "_sort_rank": 0 if self_only else (1 if is_pure_lang else (2 if is_primary_lang else 3)),
                })
        voices.sort(key=lambda x: x.get("_sort_rank", 3))
        for v in voices:
            v.pop("_sort_rank", None)
        return voices

    @staticmethod
    def fetch_google_cloud_voices(api_key: str) -> List[Dict[str, Any]]:
        """Завантажує список доступних голосів і моделей з Google Cloud Text-to-Speech API."""
        key = api_key.strip()
        if not key:
            raise RuntimeError(tr("tts.err.enter_key_first", "Спочатку введіть API ключ {provider}.", provider="Google Cloud TTS"))
        url = "https://texttospeech.googleapis.com/v1/voices"
        resp = requests.get(url, params={"key": key}, proxies=get_active_api_proxies(), timeout=15)
        if resp.status_code != 200:
            err_detail = resp.text[:180]
            try:
                err_json = resp.json()
                if isinstance(err_json, dict) and isinstance(err_json.get("error"), dict):
                    err_detail = str(err_json["error"].get("message", err_detail))[:180]
            except Exception:
                pass
            raise RuntimeError(tr("tts.err.provider_api", "{provider} помилка ({code}): {msg}", provider="Google Cloud TTS API", code=resp.status_code, msg=err_detail))

        raw_list = resp.json().get("voices", []) if isinstance(resp.json(), dict) else []
        voices: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()

        def _classify_model(vid: str) -> Tuple[str, bool]:
            low = vid.lower()
            if "chirp3-hd" in low:
                return "Chirp 3: HD", True
            if "chirp-hd" in low or "-chirp-" in low:
                return "Chirp HD", False
            if "neural2" in low:
                return "Neural2", False
            if "studio" in low:
                return "Studio", False
            if "journey" in low:
                return "Journey", False
            if "polyglot" in low:
                return "Polyglot", False
            if "news" in low:
                return "News", False
            if "casual" in low:
                return "Casual", False
            if "wavenet" in low:
                return "WaveNet", True
            if "standard" in low:
                return "Standard", True
            return "Other", False

        for v in raw_list:
            if not isinstance(v, dict):
                continue
            vid = str(v.get("name", "")).strip()
            if not vid or vid in seen_ids:
                continue
            seen_ids.add(vid)

            lang_codes = v.get("languageCodes") or []
            primary_lc = str(lang_codes[0]) if lang_codes else ("-".join(vid.split("-")[:2]) if "-" in vid else "multi")
            short_lang = primary_lc.split("-")[0].lower() if "-" in primary_lc else primary_lc.lower()

            ssml_g = str(v.get("ssmlGender", "")).upper()
            gender: Optional[str] = "female" if ssml_g == "FEMALE" else ("male" if ssml_g == "MALE" else None)

            model_name, is_free = classify_google_voice_model(vid)
            short_label = vid
            prefix = f"{primary_lc}-"
            if vid.startswith(prefix):
                short_label = vid[len(prefix):]
            badge = "" if is_free else " 💎"
            display_name = f"{short_label} ({model_name}, {primary_lc}){badge}"

            voices.append({
                "id": vid,
                "name": display_name,
                "gender": gender,
                "lang": short_lang,
                "lang_code": primary_lc,
                "model": model_name,
                "is_free": is_free,
            })

        voices.sort(key=lambda x: (0 if x.get("is_free") else 1, str(x.get("model", "")), str(x.get("name", ""))))
        return voices

    @staticmethod
    def fetch_elevenlabs_voices(api_key: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Завантажує доступні голоси та TTS-моделі з ElevenLabs API."""
        key = api_key.strip()
        if not key:
            raise RuntimeError(tr("tts.err.enter_key_first", "Спочатку введіть API ключ {provider}.", provider="ElevenLabs"))
        headers = {"xi-api-key": key}
        proxies = get_active_api_proxies()

        resp = requests.get("https://api.elevenlabs.io/v1/voices", headers=headers, proxies=proxies, timeout=15)
        if resp.status_code != 200:
            raise RuntimeError(tr("tts.err.provider_api", "{provider} помилка ({code}): {msg}", provider="ElevenLabs API", code=resp.status_code, msg=resp.text[:160]))
        raw_voices = resp.json().get("voices", []) if isinstance(resp.json(), dict) else []

        voices: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for v in raw_voices:
            if not isinstance(v, dict):
                continue
            vid = str(v.get("voice_id", "")).strip()
            if not vid or vid in seen:
                continue
            seen.add(vid)
            name = str(v.get("name", vid)).strip()
            labels = v.get("labels") if isinstance(v.get("labels"), dict) else {}
            g_raw = str(labels.get("gender", "")).strip().lower()
            gender: Optional[str] = "female" if g_raw == "female" else ("male" if g_raw == "male" else None)
            category = str(v.get("category", "")).strip()
            disp = f"{name} [{category}]" if category else name
            voices.append({
                "id": vid,
                "name": disp,
                "gender": gender,
                "lang": "multi",
            })

        models: List[str] = []
        try:
            m_resp = requests.get("https://api.elevenlabs.io/v1/models", headers=headers, proxies=proxies, timeout=12)
            if m_resp.status_code == 200 and isinstance(m_resp.json(), list):
                for m in m_resp.json():
                    if isinstance(m, dict) and m.get("can_do_text_to_speech", True):
                        mid = str(m.get("model_id", "")).strip()
                        if mid and mid not in models:
                            models.append(mid)
        except Exception:
            pass
        if not models:
            models = ["eleven_flash_v2_5", "eleven_turbo_v2_5", "eleven_multilingual_v2"]
        elif "eleven_flash_v2_5" in models:
            models.remove("eleven_flash_v2_5")
            models.insert(0, "eleven_flash_v2_5")
        return voices, models

    @staticmethod
    def fetch_openai_voices(api_key: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Перевіряє ключ OpenAI та отримує доступні TTS-моделі і голоси."""
        key = re.sub(r"^bearer\s+", "", str(api_key or "").strip().strip("\"'"), flags=re.IGNORECASE).strip()
        if not key:
            raise RuntimeError(tr("tts.err.enter_key_first", "Спочатку введіть API ключ {provider}.", provider="OpenAI"))
        headers = {"Authorization": f"Bearer {key}"}
        resp = requests.get(
            "https://api.openai.com/v1/models",
            headers=headers,
            proxies=get_active_api_proxies(),
            timeout=15,
        )
        if resp.status_code != 200:
            raise RuntimeError(tr("tts.err.provider_api", "{provider} помилка ({code}): {msg}", provider="OpenAI API", code=resp.status_code, msg=resp.text[:160]))

        data = resp.json().get("data", []) if isinstance(resp.json(), dict) else []
        models = [
            str(m.get("id", "")).strip()
            for m in data
            if isinstance(m, dict) and "tts" in str(m.get("id", "")).lower()
        ]
        models = sorted(set(models)) or ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]

        voices: List[Dict[str, Any]] = [
            {"id": "nova", "name": "Nova (OpenAI)", "gender": "female", "lang": "multi"},
            {"id": "shimmer", "name": "Shimmer (OpenAI)", "gender": "female", "lang": "multi"},
            {"id": "coral", "name": "Coral (OpenAI)", "gender": "female", "lang": "multi"},
            {"id": "sage", "name": "Sage (OpenAI)", "gender": "female", "lang": "multi"},
            {"id": "alloy", "name": "Alloy (OpenAI)", "gender": "male", "lang": "multi"},
            {"id": "onyx", "name": "Onyx (OpenAI)", "gender": "male", "lang": "multi"},
            {"id": "echo", "name": "Echo (OpenAI)", "gender": "male", "lang": "multi"},
            {"id": "ash", "name": "Ash (OpenAI)", "gender": "male", "lang": "multi"},
            {"id": "ballad", "name": "Ballad (OpenAI)", "gender": "male", "lang": "multi"},
            {"id": "fable", "name": "Fable (OpenAI)", "gender": None, "lang": "multi"},
            {"id": "verse", "name": "Verse (OpenAI)", "gender": None, "lang": "multi"},
        ]
        return voices, models

    @staticmethod
    def _group_edge_voices_by_lang(voices_raw: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Групує сирий список голосів Edge-TTS за кодом мови (uk, ru, en, ...)."""
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for v in voices_raw or []:
            if not isinstance(v, dict):
                continue
            loc = str(v.get("Locale", "")).strip()
            l_code = loc.split("-")[0].lower() if loc else ""
            if not l_code:
                continue
            short_name = str(v.get("ShortName", "")).strip()
            if not short_name:
                continue
            gender_raw = str(v.get("Gender", "")).strip().lower()
            gender = "female" if gender_raw == "female" else ("male" if gender_raw == "male" else None)
            display = short_name.split("-")[-1].replace("Neural", "").strip() or short_name
            grouped.setdefault(l_code, []).append({
                "id": short_name,
                "name": f"{display} ({loc} Neural)",
                "gender": gender,
                "lang": l_code,
            })
        return grouped

    @staticmethod
    def _fetch_edge_voices_via_requests(proxies: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Резервне завантаження списку голосів Edge-TTS через requests (стійкіше на Windows при WinError 64)."""
        from edge_tts.constants import SEC_MS_GEC_VERSION, VOICE_HEADERS, VOICE_LIST
        from edge_tts.drm import DRM

        for attempt in range(2):
            url = (
                f"{VOICE_LIST}&Sec-MS-GEC={DRM.generate_sec_ms_gec()}"
                f"&Sec-MS-GEC-Version={SEC_MS_GEC_VERSION}"
            )
            resp = requests.get(
                url,
                headers=DRM.headers_with_muid(VOICE_HEADERS),
                proxies=proxies,
                timeout=(8.0, 15.0),
            )
            if resp.status_code == 403 and attempt == 0:
                server_date = resp.headers.get("Date")
                parsed = DRM.parse_rfc2616_date(server_date) if server_date else None
                if parsed is not None:
                    DRM.adj_clock_skew_seconds(parsed - DRM.get_unix_timestamp())
                    continue
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                raise RuntimeError("Invalid Edge-TTS voices response format")
            return data
        raise RuntimeError("Edge-TTS HTTP 403")

    @staticmethod
    def _fetch_edge_tts_voices_raw() -> List[Dict[str, Any]]:
        """Завантажує сирий список голосів Edge-TTS із підтримкою проксі, ретраїв та fallback через requests."""
        try:
            import edge_tts
        except ImportError:
            raise RuntimeError(tr("tts.err.edge_lib_missing", "Бібліотека edge-tts ще не встановлена."))

        import aiohttp

        async def _list() -> List[Dict[str, Any]]:
            last_exc: Optional[BaseException] = None
            for attempt in range(2):
                connector, http_proxy = get_edge_tts_proxy_config()
                if attempt > 0 and connector is None and http_proxy is None:
                    connector = aiohttp.TCPConnector(family=socket.AF_INET, force_close=True)
                try:
                    return await edge_tts.list_voices(connector=connector, proxy=http_proxy)
                except (aiohttp.ClientError, OSError, TimeoutError) as exc:
                    last_exc = exc
                    if attempt >= 1:
                        raise
                    await asyncio.sleep(0.35)
            if last_exc is not None:
                raise last_exc
            return []

        aiohttp_err: Optional[Exception] = None
        loop = asyncio.new_event_loop()
        loop.set_exception_handler(
            lambda lp, ctx: None if isinstance(ctx.get("exception"), ConnectionResetError) else lp.default_exception_handler(ctx)
        )
        try:
            return loop.run_until_complete(_list())
        except Exception as exc:
            aiohttp_err = exc
        finally:
            loop.close()

        proxies = get_active_api_proxies()
        if proxies is None:
            custom_px = str(config.get("custom_proxy_url", "") or "").strip()
            if custom_px:
                proxies = TTSEngineManager._format_proxy_dict(custom_px)

        try:
            return TTSEngineManager._fetch_edge_voices_via_requests(proxies=proxies)
        except Exception as req_exc:
            raw_msg = str(aiohttp_err or req_exc)
            clean_msg = re.sub(r"\s*ssl:<[^>]+>", "", raw_msg).strip()
            raise RuntimeError(clean_msg) from (aiohttp_err or req_exc)

    @staticmethod
    def fetch_edge_tts_voices() -> List[Dict[str, Any]]:
        """Завантажує повний список голосів Microsoft Edge-TTS для всіх мов."""
        raw = TTSEngineManager._fetch_edge_tts_voices_raw()
        try:
            grouped = TTSEngineManager._group_edge_voices_by_lang(raw)
            if grouped:
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                (DATA_DIR / "edge_voices_cache.json").write_text(
                    json.dumps(grouped, ensure_ascii=False), encoding="utf-8"
                )
        except Exception:
            pass

        voices: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for v in raw:
            if not isinstance(v, dict):
                continue
            vid = str(v.get("ShortName", "")).strip()
            if not vid or vid in seen:
                continue
            seen.add(vid)
            locale = str(v.get("Locale", "")).strip()
            short_lang = locale.split("-")[0].lower() if "-" in locale else locale.lower()
            g_raw = str(v.get("Gender", "")).strip().lower()
            gender: Optional[str] = "female" if g_raw == "female" else ("male" if g_raw == "male" else None)
            short_name = vid.split("-")[-1].replace("Neural", "") if "-" in vid else vid
            voices.append({
                "id": vid,
                "name": f"{short_name} ({locale})",
                "gender": gender,
                "lang": short_lang,
                "lang_code": locale,
            })
        voices.sort(key=lambda x: (str(x.get("lang", "")), str(x.get("name", ""))))
        return voices

    @staticmethod
    def fetch_gemini_tts_voices(api_key: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Перевіряє ключ Google AI Studio та знаходить доступні Gemini TTS моделі."""
        key = str(api_key or "").strip().strip("\"'")
        if not key:
            raise RuntimeError(tr("tts.err.enter_key_first", "Спочатку введіть API ключ {provider}.", provider="Google AI Studio"))
        resp = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            headers={"x-goog-api-key": key},
            proxies=get_active_api_proxies(),
            timeout=15,
        )
        if resp.status_code != 200:
            raise RuntimeError(tr("tts.err.provider_api", "{provider} помилка ({code}): {msg}", provider="Google AI Studio API", code=resp.status_code, msg=resp.text[:160]))

        raw_models = resp.json().get("models", []) if isinstance(resp.json(), dict) else []
        models: List[str] = []
        for m in raw_models:
            if not isinstance(m, dict):
                continue
            mid = str(m.get("name", "")).replace("models/", "").strip()
            if mid and "tts" in mid.lower() and mid not in models:
                models.append(mid)
        if not models:
            models = [
                "gemini-3.8-flash-lite-tts",
                "gemini-3.8-flash-tts",
                "gemini-3.1-flash-tts-preview",
                "gemini-2.5-pro-preview-tts",
                "gemini-2.5-flash-preview-tts",
            ]
        else:
            models.reverse()
        return list(ONLINE_PROVIDERS["gemini_tts"]["voices"]), models

    @classmethod
    def fetch_provider_catalog(cls, provider: str, api_key: str = "") -> Dict[str, Any]:
        """Універсальний пошук доступних голосів і моделей для будь-якого онлайн-провайдера."""
        prov = str(provider or "").strip().lower()
        if prov == "google":
            return {"voices": cls.fetch_google_cloud_voices(api_key), "models": []}
        if prov == "elevenlabs":
            voices, models = cls.fetch_elevenlabs_voices(api_key)
            return {"voices": voices, "models": models}
        if prov == "openai":
            voices, models = cls.fetch_openai_voices(api_key)
            return {"voices": voices, "models": models}
        if prov == "gemini_tts":
            voices, models = cls.fetch_gemini_tts_voices(api_key)
            return {"voices": voices, "models": models}
        if prov == "fish_audio":
            return {
                "voices": cls.fetch_fish_audio_voices(api_key),
                "models": ["s2.1-pro-free", "s2.1-pro", "s2-pro", "s1", "speech-1.6", "speech-1.5"],
            }
        if prov == "edge":
            return {"voices": cls.fetch_edge_tts_voices(), "models": []}
        raise RuntimeError(tr("tts.err.provider_no_api_fetch", "Провайдер {provider} не підтримує пошук через API.", provider=prov))

    # ------------------------------------------------------------------
    # Менеджер встановлення та видалення моделей в 1 клік
    # ------------------------------------------------------------------

    def _install_pip_online_library(
        self,
        item_id: str,
        pip_package: Union[str, List[str]],
        import_name: str,
        display_name: str,
        step1_text: str,
        step2_text: str,
        config_flag: str = "installed",
    ) -> None:
        """Спільний фоновий інсталятор Python-бібліотек для онлайн-провайдерів (edge-tts, telethon)."""
        pkgs = [pip_package] if isinstance(pip_package, str) else list(pip_package)

        def _worker() -> None:
            try:
                self.download_progress.emit(item_id, 15, step1_text)
                if not getattr(sys, "frozen", False):
                    extra_kw = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "--upgrade", *pkgs],
                        check=True,
                        capture_output=True,
                        **extra_kw,
                    )
                self.download_progress.emit(item_id, 75, step2_text)
                importlib.import_module(import_name)

                items: List[Dict[str, Any]] = list(config.get("online_items", []))
                for x in items:
                    if str(x.get("id")) == item_id:
                        x[config_flag] = True
                        break
                config.set("online_items", items)

                self.download_progress.emit(item_id, 100, tr("tts.dl.installed", "Встановлено!"))
                self.model_status_changed.emit(f"online:{item_id}", True)
            except Exception as exc:
                raw_err = format_exception_message(exc)
                translated_err = translate_error_message(raw_err) or raw_err
                err_text = tr(
                    "tts.err.pkg_install_failed",
                    "Помилка встановлення {name}: {err}",
                    name=display_name,
                    err=translated_err,
                )
                print(f"\n[TTSManager] Помилка встановлення {display_name}: {raw_err}", file=sys.stderr, flush=True)
                self.download_failed.emit(item_id, tr("tts.dl.error_prefix", "Помилка: {err}", err=translated_err))
                self.error_occurred.emit(err_text)

        threading.Thread(target=_worker, daemon=True, name=f"Install_{import_name}_{item_id}").start()

    def install_edge_tts_library(self, item_id: str) -> None:
        """Встановлює та активує бібліотеку edge-tts для доданого блоку Microsoft Edge-TTS."""
        self._install_pip_online_library(
            item_id=item_id,
            pip_package="edge-tts",
            import_name="edge_tts",
            display_name="Microsoft Edge-TTS",
            step1_text=tr("tts.dl.edge_installing", "Встановлення бібліотеки edge-tts..."),
            step2_text=tr("tts.dl.edge_checking", "Перевірка модуля edge-tts..."),
        )

    def install_telethon_library(self, item_id: str) -> None:
        """Встановлює та активує бібліотеки telethon + qrcode для блоку Silero Bot (Telegram)."""
        self._install_pip_online_library(
            item_id=item_id,
            pip_package=["telethon", "pysocks", "qrcode"],
            import_name="telethon",
            display_name="Telethon (Telegram)",
            step1_text=tr("tg.dl.telethon_installing", "Встановлення бібліотеки Telethon..."),
            step2_text=tr("tg.dl.telethon_checking", "Перевірка модуля Telethon..."),
        )

    class _DownloadCancelledError(Exception):
        """Внутрішній виняток при скасуванні завантаження користувачем."""

    def cancel_download(self, engine_id: str) -> None:
        """Запитує негайну зупинку активного завантаження моделі."""
        ev = self._download_cancel_flags.get(engine_id)
        if ev is not None:
            ev.set()

    def is_downloading(self, engine_id: str) -> bool:
        with self._download_lock:
            return engine_id in self._downloading_engines

    def get_download_progress(self, engine_id: str) -> Optional[Tuple[int, str]]:
        with self._download_lock:
            return self._download_progress_state.get(engine_id)

    def download_engine_model(self, engine_id: str) -> None:
        """Запускає фонове завантаження/встановлення обраного движка з прогрес-баром."""
        with self._download_lock:
            if engine_id in self._downloading_engines:
                return
            self._downloading_engines.add(engine_id)
            ev = threading.Event()
            self._download_cancel_flags[engine_id] = ev
            self._download_progress_state[engine_id] = (1, tr("tts.dl.preparing", "Підготовка до встановлення..."))
        threading.Thread(
            target=self._download_worker,
            args=(engine_id, ev),
            daemon=True,
            name=f"Download_{engine_id}",
        ).start()

    def install_ffmpeg(self) -> None:
        """Завантажує та розпаковує ffmpeg.exe у data/engines/ffmpeg, якщо його немає в системі."""
        with self._download_lock:
            if FFMPEG_ENGINE_ID in self._downloading_engines:
                return
            self._downloading_engines.add(FFMPEG_ENGINE_ID)
            ev = threading.Event()
            self._download_cancel_flags[FFMPEG_ENGINE_ID] = ev
            self._download_progress_state[FFMPEG_ENGINE_ID] = (
                1,
                tr("tts.dl.ffmpeg_preparing", "Підготовка до встановлення FFmpeg..."),
            )
        threading.Thread(
            target=self._ffmpeg_install_worker,
            args=(ev,),
            daemon=True,
            name="Download_FFmpeg",
        ).start()

    def _ffmpeg_install_worker(self, cancel_event: threading.Event) -> None:
        def _check_cancel() -> None:
            if cancel_event.is_set() or not self._running:
                raise self._DownloadCancelledError("Cancelled by user")

        def _emit_progress(pct: int, label: str) -> None:
            with self._download_lock:
                self._download_progress_state[FFMPEG_ENGINE_ID] = (pct, label)
            self.download_progress.emit(FFMPEG_ENGINE_ID, pct, label)

        def _cleanup_parts() -> None:
            if FFMPEG_DIR.exists():
                for part_f in FFMPEG_DIR.glob("*.part"):
                    try:
                        part_f.unlink(missing_ok=True)
                    except OSError:
                        pass
                for tmp_f in FFMPEG_DIR.glob("*.tmp"):
                    try:
                        tmp_f.unlink(missing_ok=True)
                    except OSError:
                        pass

        self._active_downloads += 1
        zip_path = FFMPEG_DIR / "ffmpeg.zip"
        try:
            _check_cancel()
            if is_ffmpeg_installed():
                _emit_progress(100, tr("tts.dl.installed", "Встановлено!"))
                self.model_status_changed.emit(FFMPEG_ENGINE_ID, True)
                return

            FFMPEG_DIR.mkdir(parents=True, exist_ok=True)
            _emit_progress(2, tr("tts.dl.ffmpeg_downloading", "Завантаження FFmpeg..."))

            last_exc: Optional[Exception] = None
            downloaded = False
            for url in FFMPEG_DOWNLOAD_URLS:
                _check_cancel()
                try:
                    self._download_url_with_progress(
                        url,
                        zip_path,
                        FFMPEG_ENGINE_ID,
                        2,
                        84,
                        tr("tts.dl.ffmpeg_downloading", "Завантаження FFmpeg..."),
                        cancel_event=cancel_event,
                    )
                    downloaded = True
                    break
                except self._DownloadCancelledError:
                    raise
                except Exception as exc:
                    last_exc = exc
                    zip_path.unlink(missing_ok=True)
                    _cleanup_parts()

            if not downloaded:
                raise last_exc or RuntimeError(
                    tr("tts.err.file_download_failed", "Не вдалося завантажити файл {name}", name="ffmpeg.zip")
                )

            _check_cancel()
            _emit_progress(90, tr("tts.dl.ffmpeg_extracting", "Розпакування ffmpeg.exe..."))
            tmp_exe = FFMPEG_DIR / "ffmpeg.exe.tmp"
            extracted = False
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    if Path(member.filename).name.lower() == "ffmpeg.exe":
                        with zf.open(member, "r") as src, open(tmp_exe, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        extracted = True
                        break
            zip_path.unlink(missing_ok=True)
            if not extracted or not tmp_exe.is_file() or tmp_exe.stat().st_size < 1_000_000:
                tmp_exe.unlink(missing_ok=True)
                raise RuntimeError("Downloaded archive does not contain a valid ffmpeg.exe")
            os.replace(tmp_exe, FFMPEG_EXE)

            # Clear cached audio so previously OLA-processed phrases re-render cleanly with FFmpeg.
            self._audio_cache.clear()
            _emit_progress(100, tr("tts.dl.installed", "Встановлено!"))
            self.model_status_changed.emit(FFMPEG_ENGINE_ID, True)
        except self._DownloadCancelledError:
            zip_path.unlink(missing_ok=True)
            _cleanup_parts()
            self.download_cancelled.emit(FFMPEG_ENGINE_ID)
            self.status_changed.emit(tr("tts.dl.cancelled", "Завантаження скасовано"))
        except Exception as exc:
            zip_path.unlink(missing_ok=True)
            _cleanup_parts()
            if cancel_event.is_set():
                self.download_cancelled.emit(FFMPEG_ENGINE_ID)
                self.status_changed.emit(tr("tts.dl.cancelled", "Завантаження скасовано"))
                return
            raw_err = format_exception_message(exc)
            translated_err = translate_error_message(raw_err) or raw_err
            err_text = tr(
                "tts.err.model_install_failed",
                "Помилка встановлення {title}: {err}",
                title="FFmpeg",
                err=translated_err,
            )
            print(f"\n[TTSManager] Помилка встановлення FFmpeg: {raw_err}", file=sys.stderr, flush=True)
            self.download_failed.emit(FFMPEG_ENGINE_ID, tr("tts.dl.error_prefix", "Помилка: {err}", err=translated_err))
            self.error_occurred.emit(err_text)
        finally:
            self._active_downloads = max(0, self._active_downloads - 1)
            with self._download_lock:
                self._downloading_engines.discard(FFMPEG_ENGINE_ID)
                self._download_cancel_flags.pop(FFMPEG_ENGINE_ID, None)

    def delete_engine_model(self, engine_id: str) -> None:
        """Видаляє локальні файли обраного движка з папки data/engines/<engine_id>."""
        eng_dir = ENGINES_DIR / engine_id
        try:
            with self._model_lock:
                if engine_id == "piper":
                    self._piper_cache.clear()
                elif engine_id == "silero":
                    self._release_silero_cache_locked()
                elif engine_id == "holos":
                    self._holos_model = None
                elif engine_id == "teratts":
                    for obj in (
                        getattr(self._teratts_accent, "accents", None),
                        getattr(getattr(self._teratts_model, "tokenizer", None), "dic", None),
                    ):
                        if obj is not None and hasattr(obj, "close"):
                            obj.close()
                    self._teratts_accent = None
                    self._teratts_model = None
                elif engine_id == "omnivoice":
                    from core.omnivoice_tts import shutdown_omnivoice_tts
                    shutdown_omnivoice_tts()
                elif engine_id in ("qwen3_tts", "zonos3_mini", "kokoro_82m", "soprano_1_1"):
                    from core.llama_tts import shutdown_llama_tts
                    shutdown_llama_tts()
                    if engine_id == "kokoro_82m":
                        from core.kokoro_onnx_tts import kokoro_onnx_engine
                        kokoro_onnx_engine.unload()
                self._warmed_models = {item for item in self._warmed_models if item[0] != engine_id}

            gc.collect()

            delete_err: Optional[Exception] = None
            if eng_dir.exists():
                for attempt in range(4):
                    try:
                        shutil.rmtree(eng_dir)
                        delete_err = None
                        break
                    except OSError as exc:
                        delete_err = exc
                        gc.collect()
                        if attempt < 3:
                            time.sleep(0.1)
                if eng_dir.exists():
                    shutil.rmtree(eng_dir, ignore_errors=True)
            if engine_id in ("qwen3_tts", "zonos3_mini", "kokoro_82m", "soprano_1_1"):
                from core.llama_tts import LLAMA_TTS_DIR, is_any_llama_tts_installed
                if not is_any_llama_tts_installed() and LLAMA_TTS_DIR.exists():
                    shutil.rmtree(LLAMA_TTS_DIR, ignore_errors=True)
            elif engine_id == "omnivoice":
                from core.omnivoice_tts import OMNIVOICE_RUNTIME_DIR, OMNIVOICE_WHISPER_DIR
                for extra_dir in (OMNIVOICE_RUNTIME_DIR, OMNIVOICE_WHISPER_DIR):
                    if extra_dir.exists():
                        shutil.rmtree(extra_dir, ignore_errors=True)

            if self.is_engine_installed(engine_id):
                self.model_status_changed.emit(engine_id, True)
                err_msg = delete_err or OSError("Model files are still locked and could not be removed.")
                self.error_occurred.emit(
                    tr("tts.err.model_delete_failed", "Не вдалося видалити модель {id}: {err}", id=engine_id, err=err_msg)
                )
                return

            self.model_status_changed.emit(engine_id, False)
            self.status_changed.emit(tr("tts.status.model_deleted", "Модель {title} видалено.", title=ENGINE_SPECS.get(engine_id, {}).get("title", engine_id)))
        except Exception as exc:
            self.error_occurred.emit(tr("tts.err.model_delete_failed", "Не вдалося видалити модель {id}: {err}", id=engine_id, err=exc))

    def _resolve_active_proxies_dict(self) -> Optional[Dict[str, str]]:
        """Повертає словник активного проксі (кастомного або системного)."""
        custom_px = str(config.get("custom_proxy_url", "") or "").strip()
        if custom_px:
            return self._format_proxy_dict(custom_px)
        return get_active_api_proxies()

    def _download_worker(self, engine_id: str, cancel_event: Optional[threading.Event] = None) -> None:
        spec = ENGINE_SPECS.get(engine_id)
        if not spec:
            return
        eng_dir = ENGINES_DIR / engine_id
        eng_dir.mkdir(parents=True, exist_ok=True)
        cancel_evt = cancel_event or self._download_cancel_flags.setdefault(engine_id, threading.Event())

        def _check_cancel() -> None:
            if cancel_evt.is_set() or not self._running:
                raise self._DownloadCancelledError("Cancelled by user")

        def _emit_progress(pct: int, label: str) -> None:
            with self._download_lock:
                self._download_progress_state[engine_id] = (pct, label)
            self.download_progress.emit(engine_id, pct, label)

        def _dl_simple(url: str, dest: Path, eid: str, bpct: int, spct: int, lbl: str) -> None:
            self._download_url_with_progress(
                url,
                dest,
                eid,
                bpct,
                spct,
                lbl,
                cancel_event=cancel_evt,
            )

        install_succeeded = False
        self._active_downloads += 1
        try:
            _check_cancel()
            # A second click or a stale install action should never rewrite live model files.
            ready_marker = eng_dir / "ready.marker"
            if engine_id == "teratts" and ready_marker.exists():
                self.model_status_changed.emit(engine_id, True)
                return
            if engine_id == "holos" and self.is_engine_installed(engine_id):
                self.model_status_changed.emit(engine_id, True)
                return
            _emit_progress(2, tr("tts.dl.preparing", "Підготовка до встановлення..."))

            if "files" in spec:
                files = self.get_files_for_engine(engine_id)
                if not files:
                    raise RuntimeError(tr("tts.err.no_voices_for_lang", "Модель {title} не має голосів для обраної мови.", title=spec["title"]))
                total_files = len(files)
                for idx, (fname, url) in enumerate(files):
                    _check_cancel()
                    dest = eng_dir / fname
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    base_pct = int((idx / total_files) * 100)
                    span_pct = int(100 / total_files)
                    self._download_url_with_progress(
                        url,
                        dest,
                        engine_id,
                        base_pct,
                        span_pct,
                        tr("tts.dl.file_progress", "Завантаження {fname} ({idx}/{total})...", fname=fname, idx=idx + 1, total=total_files),
                        cancel_event=cancel_evt,
                    )
                if engine_id == "holos":
                    required = ("encoder.onnx", "prosody.onnx", "decoder.onnx", "holos_onnx.json", "voices.npz")
                    bundle_dir = eng_dir / "onnx"
                    if not all((bundle_dir / name).is_file() for name in required):
                        raise IOError("HolosTTS ONNX FP32 bundle is incomplete")
                    # Keep the old model files until the replacement bundle is complete.
                    (eng_dir / "ready.marker").write_text("onnx-fp32", encoding="utf-8")
                    shutil.rmtree(eng_dir / "onnx-int8", ignore_errors=True)
                    for legacy_name in ("holos.pt", "voices.pt"):
                        (eng_dir / legacy_name).unlink(missing_ok=True)
            elif engine_id == "teratts":
                ruaccent_dir = eng_dir / "ruaccent"
                ruaccent_dir.mkdir(parents=True, exist_ok=True)

                # Збираємо повний перелік файлів через HF Tree API (із підтримкою дзеркала hf-mirror.com),
                # щоб завантажувати кожен файл із живим прогрес-баром, швидкістю, проксі та миттєвим скасуванням.
                def _hf_list_files(repo_id: str, subpath: str) -> List[str]:
                    _check_cancel()
                    bases = ["https://huggingface.co"]
                    if bool(config.get("use_download_bypass", True)):
                        bases.append("https://hf-mirror.com")
                    custom_px = str(config.get("custom_proxy_url", "") or "").strip()
                    px = self._format_proxy_dict(custom_px) if custom_px else get_active_api_proxies()
                    last_e: Optional[Exception] = None
                    for base_api in bases:
                        _check_cancel()
                        api_url = f"{base_api}/api/models/{repo_id}/tree/main/{subpath}".rstrip("/")
                        try:
                            r = requests.get(api_url, proxies=px, timeout=10.0)
                            if r.status_code == 200:
                                return [
                                    str(item.get("path", ""))
                                    for item in r.json()
                                    if isinstance(item, dict) and item.get("type") == "file" and item.get("path")
                                ]
                        except Exception as exc:
                            last_e = exc
                    if last_e is not None:
                        raise last_e
                    return []

                _emit_progress(4, tr("tts.dl.teratts_dicts", "Завантаження TeraTTS та словників..."))
                download_tasks: List[Tuple[str, Path, str]] = []

                ra_subpaths = (
                    "nn/nn_accent",
                    "nn/nn_stress_usage_predictor",
                    "nn/nn_yo_homograph_resolver",
                    "dictionary",
                    "nn/nn_omograph/turbo",
                )
                for sp in ra_subpaths:
                    for rel_path in _hf_list_files("ruaccent/accentuator", sp):
                        dest_p = ruaccent_dir / rel_path
                        if not dest_p.exists() or dest_p.stat().st_size == 0:
                            url = f"https://huggingface.co/ruaccent/accentuator/resolve/main/{rel_path}"
                            download_tasks.append((url, dest_p, Path(rel_path).name))

                tera_model_dir = eng_dir / "TeraTTS" / "natasha-g2p-vits"
                for rel_path in _hf_list_files("TeraTTS/natasha-g2p-vits", "exported"):
                    if rel_path.endswith((".txt", ".onnx", ".json")):
                        dest_p = tera_model_dir / rel_path
                        if not dest_p.exists() or dest_p.stat().st_size == 0:
                            url = f"https://huggingface.co/TeraTTS/natasha-g2p-vits/resolve/main/{rel_path}"
                            download_tasks.append((url, dest_p, Path(rel_path).name))

                total_files = max(1, len(download_tasks))
                for idx, (url, dest_p, short_name) in enumerate(download_tasks):
                    _check_cancel()
                    dest_p.parent.mkdir(parents=True, exist_ok=True)
                    base_pct = 5 + int((idx / total_files) * 90)
                    span_pct = max(1, int(90 / total_files))
                    self._download_url_with_progress(
                        url,
                        dest_p,
                        engine_id,
                        base_pct,
                        span_pct,
                        tr("tts.dl.file_progress", "Завантаження {fname} ({idx}/{total})...", fname=short_name, idx=idx + 1, total=total_files),
                        cancel_event=cancel_evt,
                    )

                _check_cancel()
                _emit_progress(96, tr("tts.dl.teratts_model", "Індексація словників TeraTTS..."))
                acc_loaded, tts_loaded = self._load_teratts_fast(eng_dir)
                with self._model_lock:
                    self._teratts_accent = acc_loaded
                    self._teratts_model = tts_loaded
                _check_cancel()
                (eng_dir / "ready.marker").write_text("ok", encoding="utf-8")
            elif engine_id == "qwen3_tts":
                from core.llama_tts import (
                    QWEN3_LEGACY_MODEL_FILES,
                    QWEN3_MMPROJ_FILE,
                    QWEN3_MMPROJ_URL,
                    QWEN3_QUANT_SPECS,
                    get_qwen3_model_spec,
                    install_llama_tts_runtime,
                    warmup_qwen3_tts,
                )

                px_dict = self._resolve_active_proxies_dict()
                q_spec = get_qwen3_model_spec()
                model_file = q_spec["file"]
                model_url = q_spec["url"]

                # Видаляємо файли іншого квантування (якщо раніше ставилась інша версія), щоб не займати зайві гігабайти
                other_model_files = {
                    item["file"] for item in QWEN3_QUANT_SPECS.values() if item["file"] != model_file
                } | set(QWEN3_LEGACY_MODEL_FILES)
                for old_name in other_model_files:
                    try:
                        (eng_dir / old_name).unlink(missing_ok=True)
                    except OSError:
                        pass

                install_llama_tts_runtime(
                    engine_id=engine_id,
                    progress_cb=_emit_progress,
                    download_fn=_dl_simple,
                    cancel_check=_check_cancel,
                    proxies=px_dict,
                )
                downloads = [
                    (model_url, eng_dir / model_file, model_file, 25, 45),
                    (QWEN3_MMPROJ_URL, eng_dir / QWEN3_MMPROJ_FILE, QWEN3_MMPROJ_FILE, 70, 29),
                ]
                for idx, (url, dest_f, fname, bpct, spct) in enumerate(downloads, start=1):
                    _check_cancel()
                    if not dest_f.exists() or dest_f.stat().st_size == 0:
                        _dl_simple(
                            url,
                            dest_f,
                            engine_id,
                            bpct,
                            spct,
                            tr(
                                "tts.dl.file_progress",
                                "Завантаження {fname} ({idx}/{total})...",
                                fname=fname,
                                idx=idx,
                                total=len(downloads),
                            ),
                        )
                if str(config.get("active_engine", "") or "").strip() == "qwen3_tts":
                    warmup_qwen3_tts()
            elif engine_id == "kokoro_82m":
                from core.llama_tts import (
                    get_llama_engine_downloads,
                    install_llama_tts_runtime,
                    resolve_kokoro_device,
                    warmup_llama_tts,
                )

                if resolve_kokoro_device() == "cpu":
                    from core.kokoro_onnx_tts import KOKORO_ONNX_DIR, KOKORO_ONNX_DOWNLOAD_FILES

                    KOKORO_ONNX_DIR.mkdir(parents=True, exist_ok=True)
                    total_files = len(KOKORO_ONNX_DOWNLOAD_FILES)
                    for idx, (fname, url, bpct, spct) in enumerate(KOKORO_ONNX_DOWNLOAD_FILES, start=1):
                        _check_cancel()
                        dest_f = KOKORO_ONNX_DIR / fname
                        if not dest_f.exists() or dest_f.stat().st_size == 0:
                            _dl_simple(
                                url,
                                dest_f,
                                engine_id,
                                bpct,
                                spct,
                                tr(
                                    "tts.dl.file_progress",
                                    "Завантаження {fname} ({idx}/{total})...",
                                    fname=fname,
                                    idx=idx,
                                    total=total_files,
                                ),
                            )
                else:
                    px_dict = self._resolve_active_proxies_dict()
                    install_llama_tts_runtime(
                        engine_id=engine_id,
                        progress_cb=_emit_progress,
                        download_fn=_dl_simple,
                        cancel_check=_check_cancel,
                        proxies=px_dict,
                    )
                    downloads = get_llama_engine_downloads(engine_id, lang=self._current_lang())
                    for idx, (url, dest_f, fname, bpct, spct) in enumerate(downloads, start=1):
                        _check_cancel()
                        dest_f.parent.mkdir(parents=True, exist_ok=True)
                        if not dest_f.exists() or dest_f.stat().st_size == 0:
                            _dl_simple(
                                url,
                                dest_f,
                                engine_id,
                                bpct,
                                spct,
                                tr(
                                    "tts.dl.file_progress",
                                    "Завантаження {fname} ({idx}/{total})...",
                                    fname=fname,
                                    idx=idx,
                                    total=len(downloads),
                                ),
                            )
                    if str(config.get("active_engine", "") or "").strip() == engine_id:
                        warmup_llama_tts(engine_id, lang=self._current_lang())
            elif engine_id in ("zonos3_mini", "soprano_1_1"):
                from core.llama_tts import (
                    get_llama_engine_downloads,
                    install_llama_tts_runtime,
                    warmup_llama_tts,
                )

                px_dict = self._resolve_active_proxies_dict()
                install_llama_tts_runtime(
                    engine_id=engine_id,
                    progress_cb=_emit_progress,
                    download_fn=_dl_simple,
                    cancel_check=_check_cancel,
                    proxies=px_dict,
                )
                downloads = get_llama_engine_downloads(engine_id, lang=self._current_lang())
                for idx, (url, dest_f, fname, bpct, spct) in enumerate(downloads, start=1):
                    _check_cancel()
                    dest_f.parent.mkdir(parents=True, exist_ok=True)
                    if not dest_f.exists() or dest_f.stat().st_size == 0:
                        _dl_simple(
                            url,
                            dest_f,
                            engine_id,
                            bpct,
                            spct,
                            tr(
                                "tts.dl.file_progress",
                                "Завантаження {fname} ({idx}/{total})...",
                                fname=fname,
                                idx=idx,
                                total=len(downloads),
                            ),
                        )
                if str(config.get("active_engine", "") or "").strip() == engine_id:
                    warmup_llama_tts(engine_id, lang=self._current_lang())
            elif engine_id == "omnivoice":
                from core.omnivoice_tts import (
                    OMNIVOICE_QUANT_SPECS,
                    get_omnivoice_quant_spec,
                    get_omnivoice_weight_downloads,
                    install_omnivoice_runtime,
                    install_omnivoice_whisper,
                    warmup_omnivoice_tts,
                )

                px_dict = self._resolve_active_proxies_dict()
                q_spec = get_omnivoice_quant_spec()
                keep_files = {q_spec["base_file"], q_spec["codec_file"]}
                for spec_item in OMNIVOICE_QUANT_SPECS.values():
                    for fname_key in ("base_file", "codec_file"):
                        old_f = spec_item[fname_key]
                        if old_f not in keep_files:
                            try:
                                (eng_dir / old_f).unlink(missing_ok=True)
                            except OSError:
                                pass

                install_omnivoice_runtime(
                    engine_id=engine_id,
                    progress_cb=_emit_progress,
                    download_fn=_dl_simple,
                    cancel_check=_check_cancel,
                    proxies=px_dict,
                )
                install_omnivoice_whisper(
                    engine_id=engine_id,
                    progress_cb=_emit_progress,
                    download_fn=_dl_simple,
                    cancel_check=_check_cancel,
                    proxies=px_dict,
                )
                downloads = get_omnivoice_weight_downloads()
                for idx, (url, dest_f, fname, bpct, spct) in enumerate(downloads, start=1):
                    _check_cancel()
                    dest_f.parent.mkdir(parents=True, exist_ok=True)
                    if not dest_f.exists() or dest_f.stat().st_size == 0:
                        _dl_simple(
                            url,
                            dest_f,
                            engine_id,
                            bpct,
                            spct,
                            tr(
                                "tts.dl.file_progress",
                                "Завантаження {fname} ({idx}/{total})...",
                                fname=fname,
                                idx=idx,
                                total=len(downloads),
                            ),
                        )
                if str(config.get("active_engine", "") or "").strip() == "omnivoice":
                    warmup_omnivoice_tts(lang=self._current_lang())

            _emit_progress(100, tr("tts.dl.installed", "Встановлено!"))
            self.model_status_changed.emit(
                engine_id,
                self.is_engine_installed(engine_id) if engine_id == "kokoro_82m" else True,
            )
            install_succeeded = True
        except self._DownloadCancelledError:
            for part_f in list(eng_dir.rglob("*.part")) + list(eng_dir.rglob("*.part.meta")):
                try:
                    part_f.unlink(missing_ok=True)
                except Exception:
                    pass
            print(f"[TTSManager] Завантаження {spec['title']} скасовано користувачем.", flush=True)
            self.download_cancelled.emit(engine_id)
            self.status_changed.emit(tr("tts.dl.cancelled", "Завантаження скасовано"))
        except Exception as exc:
            if cancel_evt.is_set():
                for part_f in list(eng_dir.rglob("*.part")) + list(eng_dir.rglob("*.part.meta")):
                    try:
                        part_f.unlink(missing_ok=True)
                    except Exception:
                        pass
                self.download_cancelled.emit(engine_id)
                self.status_changed.emit(tr("tts.dl.cancelled", "Завантаження скасовано"))
                return
            raw_err = format_exception_message(exc)
            translated_err = translate_error_message(raw_err) or raw_err
            err_text = tr("tts.err.model_install_failed", "Помилка встановлення {title}: {err}", title=spec["title"], err=translated_err)
            print(f"\n[TTSManager] Оригінал помилки ({spec['title']}): {raw_err}", file=sys.stderr)
            print(f"[TTSManager] Переклад ({get_ui_language()}): {err_text}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            self.download_failed.emit(engine_id, tr("tts.dl.error_prefix", "Помилка: {err}", err=translated_err))
            self.error_occurred.emit(err_text)
        finally:
            self._active_downloads = max(0, self._active_downloads - 1)
            with self._download_lock:
                self._downloading_engines.discard(engine_id)
                self._download_cancel_flags.pop(engine_id, None)
            if install_succeeded:
                self.schedule_smart_warmup(0.35, "post_install")

    class _SlowProxyError(IOError):
        """Виняток при занадто повільному або завислому маршруті/проксі (< 300 КБ/с), щоб переключитися на швидший."""

    _PROXY_CACHE_TTL_SEC = 600.0
    _MIN_PROXY_SPEED_BPS = 300 * 1024
    _MIN_NON_RESUME_PROXY_SPEED_BPS = 100 * 1024
    _SLOW_PROXY_GRACE_SEC = 7.0
    _SLOW_PROXY_WINDOW_SEC = 6.0
    _STALL_TIMEOUT_SEC = 10.0
    _MULTISTREAM_MIN_SIZE_BYTES = 2 * 1024 * 1024
    _MULTISTREAM_BLOCK_SIZE = 2 * 1024 * 1024
    _PROXY_SWARM_BLOCK_SIZE = 512 * 1024
    _MULTISTREAM_WORKERS = 5

    @staticmethod
    def _format_proxy_dict(proxy_str: str) -> Optional[Dict[str, str]]:
        p = str(proxy_str).strip()
        if not p:
            return None
        if "://" not in p:
            p = f"http://{p}"
        return {"http": p, "https": p}

    @staticmethod
    def _format_speed_label(speed_bps: float) -> str:
        if speed_bps >= 1024 * 1024:
            return f"{speed_bps / (1024 * 1024):.1f} MB/s"
        if speed_bps >= 1024:
            return f"{int(speed_bps / 1024)} KB/s"
        return f"{int(max(0.0, speed_bps))} B/s"

    @staticmethod
    def _close_response_immediately(resp: Any) -> None:
        """Миттєво закриває HTTP-відповідь і її сокет, щоб розблокувати iter_content() без очікування таймауту."""
        if resp is None:
            return
        try:
            raw = getattr(resp, "raw", None)
            if raw is not None:
                fp = getattr(raw, "_fp", None)
                if fp is not None:
                    sock = getattr(fp, "fp", None)
                    raw_sock = getattr(sock, "raw", None)
                    inner_sock = getattr(raw_sock, "_sock", None) if raw_sock is not None else None
                    if inner_sock is not None:
                        try:
                            inner_sock.close()
                        except Exception:
                            pass
                    try:
                        fp.close()
                    except Exception:
                        pass
                try:
                    raw.close()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            resp.close()
        except Exception:
            pass

    @staticmethod
    def _make_download_session(pool_size: int = 8) -> requests.Session:
        sess = requests.Session()
        try:
            from requests.adapters import HTTPAdapter
            adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size, max_retries=0)
            sess.mount("https://", adapter)
            sess.mount("http://", adapter)
        except Exception:
            pass
        return sess

    def _mark_proxy_bad(self, proxy_url: str, target_url: str) -> None:
        host = urllib.parse.urlparse(target_url).netloc.lower()
        now = time.monotonic()
        with self._proxy_lock:
            self._bad_proxies[proxy_url] = now
            cached = self._proxy_cache.get(host)
            if cached:
                ts, items = cached
                self._proxy_cache[host] = (ts, [p for p in items if p != proxy_url])

    def _get_cached_proxies(self, target_url: str, exclude_proxies: Optional[set] = None) -> List[str]:
        host = urllib.parse.urlparse(target_url).netloc.lower()
        now = time.monotonic()
        excluded = exclude_proxies or set()
        with self._proxy_lock:
            self._bad_proxies = {
                k: v for k, v in self._bad_proxies.items() if now - v < self._PROXY_CACHE_TTL_SEC
            }
            cached = self._proxy_cache.get(host)
            if not cached:
                return []
            ts, items = cached
            if now - ts > self._PROXY_CACHE_TTL_SEC:
                self._proxy_cache.pop(host, None)
                self._proxy_meta.pop(host, None)
                return []
            return [p for p in items if p not in excluded and p not in self._bad_proxies]

    def _update_proxy_rankings(self, target_url: str, speed_map: Dict[str, float], alive_order: List[str]) -> None:
        """Оновлює кеш проксі для хоста на основі реально виміряної швидкості під час завантаження."""
        host = urllib.parse.urlparse(target_url).netloc.lower()
        if not host or not alive_order:
            return
        with self._proxy_lock:
            bad_set = set(self._bad_proxies.keys())
            alive = [p for p in alive_order if p not in bad_set]
            if not alive:
                return
            meta = self._proxy_meta.setdefault(host, {})
            prev_speeds: Dict[str, float] = dict(meta.get("speeds") or {})
            prev_speeds.update(speed_map)
            meta["speeds"] = prev_speeds
            alive.sort(key=lambda p: prev_speeds.get(p, 0.0), reverse=True)
            self._proxy_cache[host] = (time.monotonic(), alive)

    def _tcp_prefilter_proxies(
        self,
        candidates: List[str],
        cancel_event: Optional[threading.Event] = None,
        max_alive: int = 110,
        connect_timeout: float = 1.1,
        deadline_sec: float = 2.5,
    ) -> List[str]:
        """
        Швидкий паралельний TCP-пінг проксі-серверів (~1.5–2.5 с для 350+ IP):
        миттєво відсіює мертві адреси ще до TLS/HTTP-перевірки та сортує живі за мінімальним пінгом.
        """
        if not candidates:
            return []
        alive: List[Tuple[float, str]] = []
        lock = threading.Lock()
        stop_event = threading.Event()
        deadline = time.monotonic() + deadline_sec

        def _ping(px_url: str) -> None:
            if stop_event.is_set() or (cancel_event is not None and cancel_event.is_set()) or time.monotonic() > deadline:
                return
            try:
                parsed = urllib.parse.urlparse(px_url)
                host = parsed.hostname
                port = parsed.port
                if not host or not port:
                    return
                t0 = time.monotonic()
                with socket.create_connection((host, port), timeout=connect_timeout):
                    rtt = time.monotonic() - t0
                with lock:
                    alive.append((rtt, px_url))
                    if len(alive) >= max_alive:
                        stop_event.set()
            except Exception:
                pass

        workers = min(120, max(1, len(candidates)))
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        try:
            futures = [pool.submit(_ping, px) for px in candidates]
            try:
                for fut in concurrent.futures.as_completed(futures, timeout=deadline_sec + 0.5):
                    try:
                        fut.result(timeout=0.02)
                    except Exception:
                        pass
                    if stop_event.is_set() or (cancel_event is not None and cancel_event.is_set()) or time.monotonic() > deadline:
                        stop_event.set()
                        break
            except concurrent.futures.TimeoutError:
                stop_event.set()
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        with lock:
            alive.sort(key=lambda item: item[0])
            return [px for _, px in alive]

    def _find_working_proxies_for_url(
        self,
        target_url: str,
        max_count: int = 12,
        exclude_proxies: Optional[set] = None,
        cancel_event: Optional[threading.Event] = None,
        engine_id: str = "",
        progress_pct: int = 3,
    ) -> List[str]:
        """
        Двоетапний пошук найшвидших проксі:
        1) Паралельно завантажує списки HTTPS/SOCKS5/HTTP проксі й відсіює мертві швидким TCP-пінгом (~2 с).
        2) Паралельно бенчмаркає реальну швидкість на фрагменті до 512 КБ (Range 206), враховуючи
           загальний час (TTFB + передача), і повертає відсортований список від найшвидшого до повільнішого.
        """
        excluded = set(exclude_proxies or ())
        now = time.monotonic()
        with self._proxy_lock:
            bad_set = {
                k for k, v in self._bad_proxies.items() if now - v < self._PROXY_CACHE_TTL_SEC
            }
        excluded.update(bad_set)

        proxy_sources: List[Tuple[str, str]] = [
            ("http", "https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/protocols/https/data.txt"),
            ("socks5h", "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/socks5.txt"),
            ("socks5h", "https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/protocols/socks5/data.txt"),
            ("http", "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=4000&country=all&ssl=yes&anonymity=all"),
            ("socks5h", "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=socks5&timeout=4000&country=all"),
            ("http", "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt"),
            ("socks5h", "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/socks5.txt"),
            ("http", "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt"),
        ]

        def _fetch_source(item: Tuple[str, str]) -> List[str]:
            if cancel_event is not None and cancel_event.is_set():
                return []
            scheme, src = item
            out: List[str] = []
            try:
                r = requests.get(src, timeout=3.5)
                if r.status_code == 200:
                    for line in r.text.splitlines():
                        raw = line.strip()
                        if not raw or ":" not in raw:
                            continue
                        if "://" in raw:
                            px = raw
                            if px.startswith("socks5://"):
                                px = "socks5h://" + px[len("socks5://"):]
                            elif px.startswith("https://"):
                                px = "http://" + px[len("https://"):]
                        else:
                            px = f"{scheme}://{raw}"
                        if px not in excluded:
                            out.append(px)
                    random.shuffle(out)
            except Exception:
                pass
            return out[:80]

        per_source_lists: List[List[str]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(proxy_sources)) as src_pool:
            for res in src_pool.map(_fetch_source, proxy_sources):
                if res:
                    per_source_lists.append(res)

        if cancel_event is not None and cancel_event.is_set():
            return []

        raw_candidates: List[str] = []
        seen = set(excluded)
        idx = 0
        while len(raw_candidates) < 380:
            added_any = False
            for lst in per_source_lists:
                if idx < len(lst):
                    px = lst[idx]
                    if px not in seen:
                        seen.add(px)
                        raw_candidates.append(px)
                        if len(raw_candidates) >= 380:
                            break
                    added_any = True
            if not added_any:
                break
            idx += 1

        if not raw_candidates:
            return []

        # Етап 1: миттєвий TCP-пінг відсіює мертві IP і виводить найближчі проксі на початок
        tcp_alive = self._tcp_prefilter_proxies(
            raw_candidates,
            cancel_event=cancel_event,
            max_alive=110,
            connect_timeout=1.1,
            deadline_sec=2.5,
        )
        candidates = tcp_alive[:90] if tcp_alive else raw_candidates[:120]
        if not candidates or (cancel_event is not None and cancel_event.is_set()):
            return []

        # Етап 2: реальний бенчмарк пропускної здатності до 512 КБ (Range 206)
        # Елемент: (rank_score, score_bps, ttfb, supports_range, total_size, resolved_url, px_url)
        results: List[Tuple[float, float, float, bool, int, str, str]] = []
        lock = threading.Lock()
        stop_event = threading.Event()
        active_probe_resps: set[Any] = set()
        resp_lock = threading.Lock()
        is_pt = urllib.parse.urlparse(target_url).path.lower().endswith(".pt")
        is_zip = urllib.parse.urlparse(target_url).path.lower().endswith(".zip")
        probe_bytes = 524288  # 512 КБ для чесного виміру реальної швидкості без ілюзії буфера
        probe_start = time.monotonic()
        probe_deadline = probe_start + 9.0
        checked_count = 0
        last_ui_emit = 0.0
        total_to_check = len(candidates)

        def _maybe_emit_search_ui(force: bool = False) -> None:
            nonlocal last_ui_emit
            if not engine_id:
                return
            now_t = time.monotonic()
            if not force and (now_t - last_ui_emit) < 0.35:
                return
            last_ui_emit = now_t
            found_n = len(results)
            best_bps = max((item[1] for item in results), default=0.0)
            best_str = self._format_speed_label(best_bps) if best_bps > 0 else "..."
            self.download_progress.emit(
                engine_id,
                max(2, min(98, progress_pct)),
                tr(
                    "tts.dl.proxy_benchmark",
                    "Пошук найшвидших проксі: перевірено {checked}/{total}, робочих {found} ({best})...",
                    checked=checked_count,
                    total=total_to_check,
                    found=found_n,
                    best=best_str,
                ),
            )

        def _probe(px_url: str) -> None:
            nonlocal checked_count
            if stop_event.is_set() or (cancel_event is not None and cancel_event.is_set()) or time.monotonic() > probe_deadline:
                return
            t0 = time.monotonic()
            resp = None
            try:
                resp = requests.get(
                    target_url,
                    headers={
                        "User-Agent": "Mozilla/5.0",
                        "Range": f"bytes=0-{probe_bytes - 1}",
                        "Accept-Encoding": "identity",
                    },
                    proxies={"http": px_url, "https": px_url},
                    stream=True,
                    timeout=(3.0, 3.5),
                )
                with resp_lock:
                    active_probe_resps.add(resp)
                if resp.status_code not in (200, 206):
                    return
                ctype = (resp.headers.get("Content-Type") or "").lower()
                if "text/html" in ctype:
                    return

                supports_range = resp.status_code == 206
                total_sz = (
                    self._parse_content_range_total(resp.headers.get("Content-Range", ""))
                    if supports_range
                    else int(resp.headers.get("Content-Length", 0) or 0)
                )
                resolved_u = str(getattr(resp, "url", "") or target_url)
                t_headers = time.monotonic()
                ttfb = max(0.01, t_headers - t0)

                sample = bytearray()
                t_first_chunk: Optional[float] = None
                first_chunk_len = 0
                body_deadline = min(probe_deadline, t_headers + 3.2)

                for chunk in resp.iter_content(chunk_size=32768):
                    if stop_event.is_set() or (cancel_event is not None and cancel_event.is_set()):
                        return
                    if not chunk:
                        continue
                    need = probe_bytes - len(sample)
                    if need > 0:
                        part = chunk[:need]
                        sample.extend(part)
                        if t_first_chunk is None:
                            t_first_chunk = time.monotonic()
                            first_chunk_len = len(part)
                    if len(sample) >= probe_bytes or time.monotonic() >= body_deadline:
                        break

                min_sample = min(32768, total_sz) if total_sz > 0 else 32768
                if len(sample) < max(512, min_sample):
                    return
                if (is_pt or is_zip) and bytes(sample[:2]) != b"PK":
                    return

                t_end = time.monotonic()
                total_elapsed = max(0.05, t_end - t0)
                effective_bps = len(sample) / total_elapsed
                bytes_after_first = len(sample) - first_chunk_len
                if t_first_chunk is not None and bytes_after_first >= 65536 and (t_end - t_first_chunk) >= 0.12:
                    sustained_bps = bytes_after_first / (t_end - t_first_chunk)
                    score_bps = 0.6 * sustained_bps + 0.4 * effective_bps
                else:
                    score_bps = effective_bps

                rank_score = score_bps * (1.15 if supports_range else 0.85)

                with lock:
                    results.append((rank_score, score_bps, ttfb, supports_range, total_sz, resolved_u, px_url))
                    fast_count = sum(1 for item in results if item[1] >= 300 * 1024 and item[3])
                    decent_count = sum(1 for item in results if item[1] >= 150 * 1024 and item[3])
                    elapsed_all = time.monotonic() - probe_start
                    if fast_count >= 5:
                        stop_event.set()
                    elif elapsed_all >= 3.0 and len(results) >= max(max_count, 8) and decent_count >= 5:
                        stop_event.set()
            except Exception:
                pass
            finally:
                with lock:
                    checked_count += 1
                    _maybe_emit_search_ui()
                if resp is not None:
                    with resp_lock:
                        active_probe_resps.discard(resp)
                    self._close_response_immediately(resp)

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=min(45, len(candidates)))
        try:
            futures = [pool.submit(_probe, px) for px in candidates]
            done_deadline = probe_deadline + 0.5
            try:
                for fut in concurrent.futures.as_completed(futures, timeout=10.0):
                    try:
                        fut.result(timeout=0.05)
                    except Exception:
                        pass
                    if stop_event.is_set() or (cancel_event is not None and cancel_event.is_set()) or time.monotonic() > done_deadline:
                        stop_event.set()
                        break
            except concurrent.futures.TimeoutError:
                stop_event.set()
        finally:
            stop_event.set()
            with resp_lock:
                to_close = list(active_probe_resps)
                active_probe_resps.clear()
            for r_obj in to_close:
                self._close_response_immediately(r_obj)
            pool.shutdown(wait=False, cancel_futures=True)

        with lock:
            results.sort(key=lambda item: (item[0], -item[2]), reverse=True)
            top_items = results[:max_count]
            ranked = [item[6] for item in top_items]

        if ranked:
            host = urllib.parse.urlparse(target_url).netloc.lower()
            speeds_map = {item[6]: item[1] for item in top_items}
            range_items = [item for item in top_items if item[3] and item[4] > 0]
            known_total = range_items[0][4] if range_items else 0
            known_resolved = range_items[0][5] if range_items else ""
            with self._proxy_lock:
                self._proxy_cache[host] = (time.monotonic(), list(ranked))
                self._proxy_meta[host] = {
                    "ts": time.monotonic(),
                    "url": target_url,
                    "total": known_total,
                    "supports_range": bool(range_items),
                    "resolved_url": known_resolved,
                    "speeds": speeds_map,
                }
            top_speed = self._format_speed_label(top_items[0][1])
            print(
                f"[TTSManager] Відібрано {len(ranked)} найшвидших проксі для {host} "
                f"(найкращий: {top_items[0][6]} ~ {top_speed}, Range206={bool(range_items)})",
                file=sys.stderr,
                flush=True,
            )
        return ranked

    @staticmethod
    def _parse_content_range_total(content_range: str) -> int:
        if not content_range or "/" not in content_range:
            return 0
        tail = content_range.rsplit("/", 1)[-1].strip()
        if tail.isdigit():
            return int(tail)
        return 0

    def _build_candidate_urls(self, url: str, bypass_enabled: bool) -> List[str]:
        urls_to_try = [url]
        if url.endswith("/v3_de.pt"):
            for de_m in (
                "https://huggingface.co/maerkeov/silero-tts-de/resolve/main/v3_de.pt",
                "https://hf-mirror.com/maerkeov/silero-tts-de/resolve/main/v3_de.pt",
            ):
                if de_m not in urls_to_try:
                    urls_to_try.append(de_m)
        # Офіційне дзеркало GitLab Generic Packages для бінарників OmniVoice Runtime
        m_ov = re.match(
            r"^https://github\.com/blaite164/SilentVoice-TTS/releases/download/omnivoice-runtime-v?([^/]+)/([^/?#]+)$",
            url,
            re.IGNORECASE,
        )
        if m_ov:
            ov_ver, ov_asset = m_ov.group(1), m_ov.group(2)
            gl_mirror = (
                f"https://gitlab.com/api/v4/projects/blaite164%2Fsilentvoice-tts"
                f"/packages/generic/omnivoice-runtime/{ov_ver}/{ov_asset}"
            )
            if gl_mirror not in urls_to_try:
                urls_to_try.append(gl_mirror)
        if bypass_enabled:
            if "huggingface.co/" in url:
                mirror_url = url.replace("https://huggingface.co/", "https://hf-mirror.com/")
                if mirror_url not in urls_to_try:
                    urls_to_try.append(mirror_url)
            elif "github.com/" in url and "/releases/download/" in url:
                for gh_mirror in (
                    f"https://gh-proxy.com/{url}",
                    f"https://ghfast.top/{url}",
                ):
                    if gh_mirror not in urls_to_try:
                        urls_to_try.append(gh_mirror)
        return urls_to_try

    def _race_candidate_urls(
        self,
        urls: List[str],
        cancel_event: Optional[threading.Event] = None,
    ) -> List[str]:
        """
        Одночасно перевіряє швидкість основного сервера та дзеркал на фрагменті (до 256 КБ / ~1.8 с),
        зберігає розмір файлу й фінальний CDN-URL (без зайвого 302) і ставить найшвидший сервер першим.
        """
        if len(urls) <= 1:
            return list(urls)
        is_pt = urllib.parse.urlparse(urls[0]).path.lower().endswith(".pt")
        scored: List[Tuple[float, int, str]] = []
        failed: List[Tuple[int, str]] = []
        lock = threading.Lock()
        probe_bytes = 262144

        def _probe_one(idx_and_url: Tuple[int, str]) -> None:
            idx, u = idx_and_url
            if cancel_event is not None and cancel_event.is_set():
                return
            t0 = time.monotonic()
            resp = None
            try:
                resp = requests.get(
                    u,
                    headers={
                        "User-Agent": "Mozilla/5.0",
                        "Range": f"bytes=0-{probe_bytes - 1}",
                        "Accept-Encoding": "identity",
                    },
                    stream=True,
                    timeout=(2.5, 2.5),
                )
                if resp.status_code not in (200, 206):
                    with lock:
                        failed.append((idx, u))
                        self._url_probe_meta[u] = {"ok": False, "ts": time.monotonic()}
                    return
                ctype = (resp.headers.get("Content-Type") or "").lower()
                if "text/html" in ctype:
                    with lock:
                        failed.append((idx, u))
                        self._url_probe_meta[u] = {"ok": False, "ts": time.monotonic()}
                    return
                supports_range = resp.status_code == 206
                total_sz = (
                    self._parse_content_range_total(resp.headers.get("Content-Range", ""))
                    if supports_range
                    else 0
                )
                resolved_u = str(getattr(resp, "url", "") or u)
                t_first_chunk: Optional[float] = None
                first_chunk_len = 0
                sample = bytearray()
                for chunk in resp.iter_content(chunk_size=32768):
                    if cancel_event is not None and cancel_event.is_set():
                        return
                    if not chunk:
                        continue
                    need = probe_bytes - len(sample)
                    if need > 0:
                        part = chunk[:need]
                        sample.extend(part)
                        if t_first_chunk is None:
                            t_first_chunk = time.monotonic()
                            first_chunk_len = len(part)
                    if len(sample) >= probe_bytes or (time.monotonic() - t0) > 1.8:
                        break
                if len(sample) < 512 or (is_pt and bytes(sample[:2]) != b"PK"):
                    with lock:
                        failed.append((idx, u))
                        self._url_probe_meta[u] = {"ok": False, "ts": time.monotonic()}
                    return
                t_end = time.monotonic()
                total_elapsed = max(0.05, t_end - t0)
                effective_bps = len(sample) / total_elapsed
                bytes_after_first = len(sample) - first_chunk_len
                if t_first_chunk is not None and bytes_after_first >= 65536 and (t_end - t_first_chunk) >= 0.10:
                    sustained_bps = bytes_after_first / (t_end - t_first_chunk)
                    speed_bps = 0.6 * sustained_bps + 0.4 * effective_bps
                else:
                    speed_bps = effective_bps
                # Якщо основний сервер уже швидкий (>= 500 КБ/с), даємо йому невелику перевагу 15%
                effective_score = speed_bps * 1.15 if (idx == 0 and speed_bps >= 500 * 1024) else speed_bps
                with lock:
                    scored.append((effective_score, -idx, u))
                    self._url_probe_meta[u] = {
                        "ok": True,
                        "ts": time.monotonic(),
                        "total": total_sz if supports_range else 0,
                        "resolved_url": resolved_u,
                        "speed_bps": speed_bps,
                    }
            except Exception:
                with lock:
                    failed.append((idx, u))
                    self._url_probe_meta[u] = {"ok": False, "ts": time.monotonic()}
            finally:
                self._close_response_immediately(resp)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(urls)) as pool:
            list(pool.map(_probe_one, enumerate(urls)))

        if not scored:
            return list(urls)
        scored.sort(reverse=True)
        failed.sort(key=lambda item: item[0])
        ordered = [u for _, _, u in scored] + [u for _, u in failed if u not in {x[2] for x in scored}]
        for u in urls:
            if u not in ordered:
                ordered.append(u)
        return ordered

    def _probe_range_info(
        self,
        target_url: str,
        proxies: Optional[Dict[str, str]],
        connect_timeout: float,
        read_timeout: float,
    ) -> Tuple[int, str]:
        """Перевіряє підтримку HTTP Range (206) і повертає (total_bytes, resolved_final_url)."""
        resp = None
        try:
            resp = requests.get(
                target_url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Range": "bytes=0-0",
                    "Accept-Encoding": "identity",
                },
                proxies=proxies,
                stream=True,
                timeout=(connect_timeout, min(10.0, read_timeout)),
            )
            if resp.status_code != 206:
                return 0, target_url
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "text/html" in ctype:
                return 0, target_url
            total = self._parse_content_range_total(resp.headers.get("Content-Range", ""))
            resolved_url = str(getattr(resp, "url", "") or target_url)
            if total >= self._MULTISTREAM_MIN_SIZE_BYTES:
                return total, resolved_url
            return 0, resolved_url
        except Exception:
            return 0, target_url
        finally:
            self._close_response_immediately(resp)

    def _probe_range_total(
        self,
        target_url: str,
        proxies: Optional[Dict[str, str]],
        connect_timeout: float,
        read_timeout: float,
    ) -> int:
        """Перевіряє, чи підтримує сервер HTTP Range (206 Partial Content), і повертає загальний розмір файлу."""
        total, _ = self._probe_range_info(target_url, proxies, connect_timeout, read_timeout)
        return total

    @staticmethod
    def _read_part_progress(dest: Path) -> Tuple[int, int]:
        """Повертає (done_bytes, total_bytes) зі збережених .part.meta / .part без зміни файлів."""
        tmp_dest = dest.with_suffix(dest.suffix + ".part")
        meta_dest = dest.with_suffix(dest.suffix + ".part.meta")
        if meta_dest.exists() and tmp_dest.exists():
            try:
                meta = json.loads(meta_dest.read_text(encoding="utf-8"))
                total = int(meta.get("total", 0) or 0)
                block_size = int(meta.get("block_size", 0) or 0)
                if total > 0 and block_size > 0:
                    num_blocks = (total + block_size - 1) // block_size
                    done_set = {int(i) for i in (meta.get("done") or []) if 0 <= int(i) < num_blocks}
                    done_bytes = sum(min(block_size, total - i * block_size) for i in done_set)
                    return done_bytes, total
            except Exception:
                pass
        if tmp_dest.exists() and not meta_dest.exists():
            try:
                return int(tmp_dest.stat().st_size), 0
            except OSError:
                pass
        return 0, 0

    @staticmethod
    def _normalize_part_file_for_single_stream(tmp_dest: Path, meta_dest: Path) -> None:
        """
        Якщо попередня спроба була багатопотоковою (.part.meta), обрізає .part до неперервного
        початкового префікса завершених блоків, щоб однопотоковий Range міг безпечно докачати файл.
        """
        if not meta_dest.exists():
            return
        try:
            meta = json.loads(meta_dest.read_text(encoding="utf-8"))
            total = int(meta.get("total", 0) or 0)
            block_size = int(meta.get("block_size", 0) or 0)
            done_set = {int(i) for i in (meta.get("done") or [])}
            contig_blocks = 0
            while contig_blocks in done_set:
                contig_blocks += 1
            contig_bytes = min(total, contig_blocks * block_size) if (total > 0 and block_size > 0) else 0
            if tmp_dest.exists():
                if contig_bytes > 0:
                    with open(tmp_dest, "r+b") as f:
                        f.truncate(contig_bytes)
                else:
                    tmp_dest.unlink(missing_ok=True)
        except Exception:
            tmp_dest.unlink(missing_ok=True)
        finally:
            meta_dest.unlink(missing_ok=True)

    def _multistream_download_attempt(
        self,
        target_url: str,
        dest: Path,
        total: int,
        engine_id: str,
        base_pct: int,
        span_pct: int,
        label: str,
        proxies: Optional[Dict[str, str]] = None,
        connect_timeout: float = 8.0,
        read_timeout: float = 30.0,
        abort_if_slow: bool = False,
        cancel_event: Optional[threading.Event] = None,
        resolved_url: str = "",
        proxy_pool: Optional[List[str]] = None,
    ) -> None:
        """
        Багатопотокове скачування файлу у 5 паралельних HTTP Range-потоків із:
        - збереженням завершених блоків на диск (.part + .part.meta) і докачуванням без втрат,
        - прямим CDN-URL без зайвих 302-редиректів на кожен блок,
        - підтримкою Multi-Proxy Swarm (роздача блоків на пул найшвидших проксі з миттєвим
          перемиканням лише проблемного потоку без зупинки інших),
        - watchdog-контролем зависання (0 байт), швидкості (< 300 КБ/с) і миттєвим скасуванням.
        """
        if cancel_event is not None and cancel_event.is_set():
            raise self._DownloadCancelledError("Cancelled")

        use_swarm = bool(proxy_pool)
        tmp_dest = dest.with_suffix(dest.suffix + ".part")
        meta_dest = dest.with_suffix(dest.suffix + ".part.meta")

        # Адаптивний розмір блока: для великих прямих файлів 4–8 МБ, для рою проксі 1 МБ.
        # Якщо тести зменшили _MULTISTREAM_BLOCK_SIZE (< 2 МБ), зберігаємо його точно.
        configured_bs = max(65536, int(self._MULTISTREAM_BLOCK_SIZE))
        default_bs = 2 * 1024 * 1024
        if configured_bs == default_bs:
            if use_swarm:
                block_size = self._PROXY_SWARM_BLOCK_SIZE
            elif total >= 256 * 1024 * 1024 and proxies is None:
                block_size = 8 * 1024 * 1024
            elif total >= 64 * 1024 * 1024 and proxies is None:
                block_size = 4 * 1024 * 1024
            else:
                block_size = configured_bs
        else:
            block_size = configured_bs

        done_blocks: set[int] = set()
        meta_loaded = False
        if meta_dest.exists() and tmp_dest.exists():
            try:
                meta = json.loads(meta_dest.read_text(encoding="utf-8"))
                meta_total = int(meta.get("total", 0) or 0)
                meta_bs = int(meta.get("block_size", 0) or 0)
                if meta_total == total and 65536 <= meta_bs <= 16 * 1024 * 1024:
                    # Зберігаємо попередній block_size, щоб не втратити жодного вже завантаженого блока
                    block_size = meta_bs
                    num_blocks = (total + block_size - 1) // block_size
                    done_blocks = {
                        int(i) for i in (meta.get("done") or []) if 0 <= int(i) < num_blocks
                    }
                    meta_loaded = True
                else:
                    self._normalize_part_file_for_single_stream(tmp_dest, meta_dest)
            except Exception:
                meta_dest.unlink(missing_ok=True)

        num_blocks = (total + block_size - 1) // block_size

        if not meta_loaded and not done_blocks and tmp_dest.exists():
            try:
                existing_sz = tmp_dest.stat().st_size
                if dest.suffix == ".pt" and existing_sz >= 2:
                    with open(tmp_dest, "rb") as chk_f:
                        if chk_f.read(2) != b"PK":
                            tmp_dest.unlink(missing_ok=True)
                            existing_sz = 0
                if 0 < existing_sz <= total:
                    done_blocks = set(range(existing_sz // block_size))
                elif existing_sz > total:
                    tmp_dest.unlink(missing_ok=True)
            except Exception:
                pass

        pending_indices = [i for i in range(num_blocks) if i not in done_blocks]
        if not pending_indices and tmp_dest.exists() and tmp_dest.stat().st_size == total:
            meta_dest.unlink(missing_ok=True)
            tmp_dest.replace(dest)
            return

        if not tmp_dest.exists():
            with open(tmp_dest, "wb") as init_f:
                init_f.truncate(total)
        else:
            with open(tmp_dest, "r+b") as init_f:
                init_f.truncate(total)

        last_meta_save = 0.0
        blocks_since_flush = 0

        def _save_meta_unlocked(force: bool = False) -> None:
            nonlocal last_meta_save
            now_m = time.monotonic()
            if not force and (now_m - last_meta_save) < 0.5:
                return
            last_meta_save = now_m
            try:
                meta_dest.write_text(
                    json.dumps({"total": total, "block_size": block_size, "done": sorted(done_blocks)}),
                    encoding="utf-8",
                )
            except Exception:
                pass

        _save_meta_unlocked(force=True)

        q_blocks: queue.Queue[int] = queue.Queue()
        for b_idx in pending_indices:
            q_blocks.put(b_idx)

        state_lock = threading.Lock()
        file_lock = threading.Lock()
        resp_lock = threading.Lock()
        pool_lock = threading.Lock()
        abort_event = threading.Event()
        active_responses: set[Any] = set()
        first_error: List[Exception] = []

        completed_bytes = sum(min(block_size, total - i * block_size) for i in done_blocks)
        max_reported_bytes = completed_bytes
        in_flight_bytes: Dict[int, int] = {}
        block_attempts: Dict[int, int] = {}
        consecutive_errors = 0
        speed_window: deque[Tuple[float, int]] = deque()
        window_bytes = 0
        session_start = time.monotonic()
        last_byte_ts = session_start
        last_emit = 0.0

        active_url_box: List[str] = [resolved_url or target_url]
        chunk_read_size = 65536 if (use_swarm or proxies is not None) else 131072
        stall_timeout = max(self._STALL_TIMEOUT_SEC, 8.0 if abort_if_slow else 25.0)

        # Стан пулу проксі для Multi-Proxy Swarm
        available_proxies: List[str] = list(proxy_pool or ())
        dead_proxies: set[str] = set()
        slow_proxies: set[str] = set()
        proxy_failures: Dict[str, int] = {}
        proxy_in_use: Dict[str, int] = {}
        proxy_live_speed: Dict[str, float] = {}
        proxy_orig_rank: Dict[str, int] = {p: idx for idx, p in enumerate(available_proxies)}

        def _acquire_swarm_proxy(prev_px: Optional[str] = None, failed_prev: bool = False, slow_prev: bool = False) -> Optional[str]:
            with pool_lock:
                if prev_px:
                    proxy_in_use[prev_px] = max(0, proxy_in_use.get(prev_px, 0) - 1)
                    if failed_prev:
                        proxy_failures[prev_px] = proxy_failures.get(prev_px, 0) + 1
                        if proxy_failures[prev_px] >= 2:
                            dead_proxies.add(prev_px)
                    if slow_prev:
                        slow_proxies.add(prev_px)

                healthy = [p for p in available_proxies if p not in dead_proxies]
                if not healthy:
                    return None
                preferred = [p for p in healthy if p not in slow_proxies] or healthy
                # Якщо є вільні проксі, роздаємо по 1 потоку на проксі, інакше довантажуємо найшвидші
                preferred.sort(
                    key=lambda p: (
                        proxy_in_use.get(p, 0),
                        -proxy_live_speed.get(p, 0.0),
                        proxy_orig_rank.get(p, 999),
                    )
                )
                chosen = preferred[0]
                proxy_in_use[chosen] = proxy_in_use.get(chosen, 0) + 1
                return chosen

        def _release_swarm_proxy(px_url: Optional[str]) -> None:
            if not px_url:
                return
            with pool_lock:
                proxy_in_use[px_url] = max(0, proxy_in_use.get(px_url, 0) - 1)

        def _has_faster_alternative_proxy(px_url: str) -> bool:
            with pool_lock:
                return any(
                    p != px_url and p not in dead_proxies and p not in slow_proxies
                    for p in available_proxies
                )

        def _current_route_label() -> str:
            if not use_swarm:
                return label
            with pool_lock:
                active_n = sum(1 for c in proxy_in_use.values() if c > 0)
            if active_n > 1:
                return tr("tts.dl.proxy_swarm_tag", "{label} [Проксі ×{count}]", label=label, count=active_n)
            return tr("tts.dl.best_proxy_tag", "{label} [Найкращий проксі]", label=label)

        def _compute_speed_locked(now_t: float) -> float:
            nonlocal window_bytes
            cutoff = now_t - self._SLOW_PROXY_WINDOW_SEC
            while speed_window and speed_window[0][0] < cutoff:
                _, old_len = speed_window.popleft()
                window_bytes -= old_len
            elapsed = max(0.05, now_t - session_start)
            if len(speed_window) > 1:
                span = max(0.5, now_t - speed_window[0][0])
                return window_bytes / span
            return window_bytes / elapsed

        def _close_all_active_responses() -> None:
            with resp_lock:
                resps = list(active_responses)
                active_responses.clear()
            for r_obj in resps:
                self._close_response_immediately(r_obj)

        def _trigger_abort(err: Optional[Exception] = None) -> None:
            with state_lock:
                if err is not None and not first_error:
                    first_error.append(err)
            abort_event.set()
            _close_all_active_responses()

        num_workers = max(1, min(int(self._MULTISTREAM_WORKERS), len(pending_indices)))
        file_closed = False
        out_f = open(tmp_dest, "r+b")
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        futures: List[concurrent.futures.Future] = []

        try:
            def _worker() -> None:
                nonlocal completed_bytes, max_reported_bytes, window_bytes, last_emit, last_byte_ts, consecutive_errors, blocks_since_flush
                sess = self._make_download_session(pool_size=4)
                cur_px_url: Optional[str] = _acquire_swarm_proxy() if use_swarm else None
                try:
                    while not abort_event.is_set():
                        if cancel_event is not None and cancel_event.is_set():
                            abort_event.set()
                            return
                        if use_swarm and not cur_px_url:
                            return
                        try:
                            b_idx = q_blocks.get_nowait()
                        except queue.Empty:
                            return

                        start_b = b_idx * block_size
                        end_b = min(total - 1, start_b + block_size - 1)
                        expected_len = end_b - start_b + 1
                        px_dict = self._format_proxy_dict(cur_px_url) if use_swarm and cur_px_url else proxies
                        req_url = active_url_box[0]
                        buf = bytearray()
                        resp = None
                        block_t0 = time.monotonic()
                        block_slow_switch = False
                        block_failed_proxy = False
                        fatal_err: Optional[Exception] = None

                        try:
                            resp = sess.get(
                                req_url,
                                headers={
                                    "User-Agent": "Mozilla/5.0",
                                    "Range": f"bytes={start_b}-{end_b}",
                                    "Accept-Encoding": "identity",
                                },
                                proxies=px_dict,
                                stream=True,
                                timeout=(connect_timeout, read_timeout),
                            )
                            with resp_lock:
                                if abort_event.is_set():
                                    self._close_response_immediately(resp)
                                    return
                                active_responses.add(resp)

                            if resp.status_code in (401, 403, 404, 410) and req_url != target_url:
                                # Підписаний CDN-URL міг протухнути — повертаємося до основного URL
                                active_url_box[0] = target_url
                                raise IOError(f"CDN URL expired (HTTP {resp.status_code}), refreshing")

                            if resp.status_code != 206:
                                if resp.status_code in (404, 410) and req_url == target_url:
                                    fatal_err = IOError(f"HTTP {resp.status_code}")
                                raise IOError(f"Expected HTTP 206 for Range request, got {resp.status_code}")

                            ctype = (resp.headers.get("Content-Type") or "").lower()
                            if "text/html" in ctype:
                                block_failed_proxy = True
                                raise IOError(tr("tts.err.file_too_small", "Завантажений файл занадто малий (пошкоджена відповідь проксі)"))

                            for chunk in resp.iter_content(chunk_size=chunk_read_size):
                                if cancel_event is not None and cancel_event.is_set():
                                    abort_event.set()
                                    return
                                if abort_event.is_set():
                                    return
                                if not chunk:
                                    continue
                                buf.extend(chunk)
                                chunk_len = len(chunk)
                                now = time.monotonic()

                                with state_lock:
                                    last_byte_ts = now
                                    in_flight_bytes[b_idx] = len(buf)
                                    speed_window.append((now, chunk_len))
                                    window_bytes += chunk_len
                                    cur_speed = _compute_speed_locked(now)
                                    cur_downloaded = min(total, completed_bytes + sum(in_flight_bytes.values()))
                                    if cur_downloaded > max_reported_bytes:
                                        max_reported_bytes = cur_downloaded

                                    if now - last_emit > 0.12:
                                        last_emit = now
                                        route_lbl = _current_route_label()
                                        speed_label = f"{route_lbl} ({self._format_speed_label(cur_speed)})"
                                        shown_bytes = max_reported_bytes
                                        pct = base_pct + int((shown_bytes / total) * span_pct)
                                        mb_done = shown_bytes / (1024 * 1024)
                                        mb_tot = total / (1024 * 1024)
                                        self.download_progress.emit(
                                            engine_id,
                                            min(99, pct),
                                            tr(
                                                "tts.dl.mb_progress",
                                                "{label} ({done:.1f} / {total:.1f} МБ)",
                                                label=speed_label,
                                                done=mb_done,
                                                total=mb_tot,
                                            ),
                                        )

                                # У режимі рою проксі: якщо цей один проксі повзе (< 45 КБ/с), а є інші —
                                # перемикаємо ЛИШЕ цей потік, не чіпаючи решту!
                                if use_swarm and cur_px_url:
                                    b_elapsed = now - block_t0
                                    if b_elapsed >= 5.0 and (len(buf) / b_elapsed) < 45 * 1024 and _has_faster_alternative_proxy(cur_px_url):
                                        block_slow_switch = True
                                        break

                            if abort_event.is_set():
                                return
                            if block_slow_switch:
                                raise IOError("Single swarm proxy too slow, rotating worker")
                            if len(buf) != expected_len:
                                raise IOError(f"Incomplete block {b_idx}: {len(buf)} / {expected_len}")
                            if b_idx == 0 and dest.suffix == ".pt" and bytes(buf[:2]) != b"PK":
                                fatal_err = IOError(tr("tts.err.invalid_pt_archive", "Завантажений файл не є коректним архівом моделі PyTorch (.pt)"))
                                raise fatal_err

                            with file_lock:
                                if file_closed or abort_event.is_set():
                                    return
                                out_f.seek(start_b)
                                out_f.write(buf)
                                blocks_since_flush += 1
                                if blocks_since_flush >= 8:
                                    out_f.flush()
                                    blocks_since_flush = 0
                                done_blocks.add(b_idx)
                                _save_meta_unlocked()

                            block_elapsed = max(0.05, time.monotonic() - block_t0)
                            if use_swarm and cur_px_url:
                                blk_bps = expected_len / block_elapsed
                                with pool_lock:
                                    prev_sp = proxy_live_speed.get(cur_px_url)
                                    proxy_live_speed[cur_px_url] = blk_bps if prev_sp is None else (0.5 * prev_sp + 0.5 * blk_bps)
                                    proxy_failures[cur_px_url] = 0

                            with state_lock:
                                in_flight_bytes.pop(b_idx, None)
                                completed_bytes += expected_len
                                consecutive_errors = 0
                        except Exception as exc:
                            with state_lock:
                                in_flight_bytes.pop(b_idx, None)
                            if abort_event.is_set() or (cancel_event is not None and cancel_event.is_set()):
                                return
                            if fatal_err is not None:
                                _trigger_abort(fatal_err)
                                return

                            with state_lock:
                                attempts_now = block_attempts.get(b_idx, 0) + 1
                                block_attempts[b_idx] = attempts_now
                                if not use_swarm:
                                    consecutive_errors += 1

                            if use_swarm and cur_px_url:
                                failed_px = cur_px_url
                                if not block_slow_switch:
                                    # Помилка мережі / обрив / сміття — позначаємо проксі проблемним
                                    if block_failed_proxy or isinstance(exc, (requests.exceptions.ProxyError, requests.exceptions.ConnectTimeout)):
                                        with pool_lock:
                                            proxy_failures[failed_px] = 2
                                        self._mark_proxy_bad(failed_px, target_url)
                                    cur_px_url = _acquire_swarm_proxy(failed_px, failed_prev=True)
                                    if failed_px in dead_proxies:
                                        self._mark_proxy_bad(failed_px, target_url)
                                else:
                                    cur_px_url = _acquire_swarm_proxy(failed_px, slow_prev=True)

                                if attempts_now <= 4 and cur_px_url is not None:
                                    q_blocks.put(b_idx)
                                elif cur_px_url is None:
                                    with state_lock:
                                        if not first_error:
                                            first_error.append(exc)
                                    # Повертаємо блок у чергу — можливо, інші потоки ще мають живі проксі
                                    if attempts_now <= 4:
                                        q_blocks.put(b_idx)
                                    return
                                else:
                                    _trigger_abort(exc)
                                    return
                            else:
                                # Прямий / одиничний маршрут: допускаємо локальний ретрай окремого блока
                                if attempts_now <= 2 and consecutive_errors <= 5:
                                    q_blocks.put(b_idx)
                                else:
                                    _trigger_abort(exc)
                                    return
                        finally:
                            if resp is not None:
                                with resp_lock:
                                    active_responses.discard(resp)
                                self._close_response_immediately(resp)
                finally:
                    if use_swarm:
                        _release_swarm_proxy(cur_px_url)
                    try:
                        sess.close()
                    except Exception:
                        pass

            futures = [pool.submit(_worker) for _ in range(num_workers)]

            # Watchdog у головному потоці: миттєве скасування, контроль зависання (0 байт) і швидкості (< 300 КБ/с)
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    _trigger_abort(self._DownloadCancelledError("Cancelled by user"))
                    break
                if all(f.done() for f in futures):
                    break

                now = time.monotonic()
                with state_lock:
                    cur_speed = _compute_speed_locked(now)
                    cur_downloaded = min(total, completed_bytes + sum(in_flight_bytes.values()))
                    elapsed = now - session_start
                    stalled_for = now - last_byte_ts

                if (
                    abort_if_slow
                    and elapsed >= self._SLOW_PROXY_GRACE_SEC
                    and cur_speed < self._MIN_PROXY_SPEED_BPS
                    and cur_downloaded < total
                ):
                    _trigger_abort(
                        self._SlowProxyError(
                            f"Маршрут занадто повільний ({self._format_speed_label(cur_speed)} < 300 KB/s), перемикаємось на інший"
                        )
                    )
                    break

                if stalled_for >= stall_timeout and cur_downloaded < total:
                    _trigger_abort(
                        IOError(f"Завантаження зависло без передачі даних ({int(stalled_for)} с)")
                    )
                    break

                time.sleep(0.2)
        finally:
            abort_event.set()
            _close_all_active_responses()
            wait_deadline = time.monotonic() + 1.5
            for fut in futures:
                rem = max(0.0, wait_deadline - time.monotonic())
                if rem <= 0.0:
                    break
                try:
                    fut.result(timeout=rem)
                except Exception:
                    pass
            pool.shutdown(wait=False, cancel_futures=True)
            with file_lock:
                try:
                    out_f.flush()
                except Exception:
                    pass
                try:
                    _save_meta_unlocked(force=True)
                except Exception:
                    pass
                file_closed = True
                try:
                    out_f.close()
                except Exception:
                    pass

        if use_swarm:
            alive_after = [p for p in available_proxies if p not in dead_proxies]
            if alive_after:
                self._update_proxy_rankings(target_url, proxy_live_speed, alive_after)

        if cancel_event is not None and cancel_event.is_set():
            raise self._DownloadCancelledError("Cancelled by user")
        if first_error:
            raise first_error[0]
        if len(done_blocks) < num_blocks:
            raise IOError(tr("tts.err.download_interrupted", "Завантаження обірвалося ({done} із {total} байт)", done=completed_bytes, total=total))

        min_bytes = 5_000 if dest.name == "voices.pt" else 100_000
        if dest.suffix in (".pt", ".onnx") and total < min_bytes:
            tmp_dest.unlink(missing_ok=True)
            meta_dest.unlink(missing_ok=True)
            raise IOError(tr("tts.err.file_too_small", "Завантажений файл занадто малий (пошкоджена відповідь проксі)"))
        if dest.suffix == ".pt":
            with open(tmp_dest, "rb") as chk_f:
                magic = chk_f.read(2)
            if magic != b"PK":
                tmp_dest.unlink(missing_ok=True)
                meta_dest.unlink(missing_ok=True)
                raise IOError(tr("tts.err.invalid_pt_archive", "Завантажений файл не є коректним архівом моделі PyTorch (.pt)"))

        meta_dest.unlink(missing_ok=True)
        tmp_dest.replace(dest)

    def _stream_download_attempt(
        self,
        target_url: str,
        dest: Path,
        engine_id: str,
        base_pct: int,
        span_pct: int,
        label: str,
        proxies: Optional[Dict[str, str]] = None,
        connect_timeout: float = 8.0,
        read_timeout: float = 30.0,
        abort_if_slow: bool = False,
        allow_resume: bool = True,
        cancel_event: Optional[threading.Event] = None,
        allow_multistream: bool = True,
        known_total: int = 0,
        known_resolved_url: str = "",
        proxy_pool: Optional[List[str]] = None,
    ) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise self._DownloadCancelledError("Cancelled")
        tmp_dest = dest.with_suffix(dest.suffix + ".part")
        meta_dest = dest.with_suffix(dest.suffix + ".part.meta")

        if allow_resume and allow_multistream and int(self._MULTISTREAM_WORKERS) > 1:
            if known_total >= self._MULTISTREAM_MIN_SIZE_BYTES:
                total_probe = known_total
                resolved_u = known_resolved_url or target_url
            else:
                probe_px = proxies if proxies is not None else (self._format_proxy_dict(proxy_pool[0]) if proxy_pool else None)
                total_probe, resolved_u = self._probe_range_info(target_url, probe_px, connect_timeout, read_timeout)
            if total_probe >= self._MULTISTREAM_MIN_SIZE_BYTES:
                self._multistream_download_attempt(
                    target_url=target_url,
                    dest=dest,
                    total=total_probe,
                    engine_id=engine_id,
                    base_pct=base_pct,
                    span_pct=span_pct,
                    label=label,
                    proxies=proxies,
                    connect_timeout=connect_timeout,
                    read_timeout=read_timeout,
                    abort_if_slow=abort_if_slow,
                    cancel_event=cancel_event,
                    resolved_url=resolved_u,
                    proxy_pool=proxy_pool,
                )
                return

        self._normalize_part_file_for_single_stream(tmp_dest, meta_dest)
        resume_offset = 0
        if allow_resume and tmp_dest.exists():
            try:
                sz = tmp_dest.stat().st_size
                if sz > 0:
                    if dest.suffix == ".pt" and sz >= 2:
                        with open(tmp_dest, "rb") as chk_f:
                            if chk_f.read(2) == b"PK":
                                resume_offset = sz
                            else:
                                tmp_dest.unlink(missing_ok=True)
                    elif dest.suffix != ".pt":
                        resume_offset = sz
            except Exception:
                resume_offset = 0

        headers = {"User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity"}
        if resume_offset > 0:
            headers["Range"] = f"bytes={resume_offset}-"

        active_proxies = proxies if proxies is not None else (self._format_proxy_dict(proxy_pool[0]) if proxy_pool else None)
        with requests.get(
            target_url,
            headers=headers,
            proxies=active_proxies,
            stream=True,
            timeout=(connect_timeout, read_timeout),
        ) as resp:
            if resp.status_code == 416 and resume_offset > 0:
                tmp_dest.unlink(missing_ok=True)
                resume_offset = 0
                resp.close()
                with requests.get(
                    target_url,
                    headers={"User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity"},
                    proxies=active_proxies,
                    stream=True,
                    timeout=(connect_timeout, read_timeout),
                ) as retry_resp:
                    self._consume_download_response(
                        retry_resp, tmp_dest, dest, engine_id, base_pct, span_pct, label, 0, abort_if_slow,
                        cancel_event=cancel_event, is_proxied=bool(active_proxies),
                    )
                tmp_dest.replace(dest)
                return

            self._consume_download_response(
                resp, tmp_dest, dest, engine_id, base_pct, span_pct, label, resume_offset, abort_if_slow,
                cancel_event=cancel_event, is_proxied=bool(active_proxies),
            )

        tmp_dest.replace(dest)

    def _consume_download_response(
        self,
        resp: requests.Response,
        tmp_dest: Path,
        dest: Path,
        engine_id: str,
        base_pct: int,
        span_pct: int,
        label: str,
        resume_offset: int,
        abort_if_slow: bool,
        cancel_event: Optional[threading.Event] = None,
        is_proxied: bool = False,
    ) -> None:
        resp.raise_for_status()
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "text/html" in ctype:
            raise IOError(tr("tts.err.file_too_small", "Завантажений файл занадто малий (пошкоджена відповідь проксі)"))

        is_partial = resp.status_code == 206 and resume_offset > 0
        accept_ranges = str(resp.headers.get("Accept-Ranges", "") or "").strip().lower()
        server_supports_range = is_partial or accept_ranges == "bytes"
        if is_partial:
            content_range = resp.headers.get("Content-Range", "")
            total = self._parse_content_range_total(content_range)
            if total == 0:
                rem = int(resp.headers.get("Content-Length", 0) or 0)
                total = resume_offset + rem if rem > 0 else 0
            downloaded = resume_offset
            file_mode = "ab"
        else:
            if resume_offset > 0 and is_proxied and server_supports_range:
                # Сервер підтримує Range (Accept-Ranges: bytes), але проксі повернув 200 замість 206:
                # не затираємо вже завантажений .part, а перемикаємося на інший проксі!
                raise IOError("Proxy ignored Range header (HTTP 200 instead of 206); preserving partial file")
            total = int(resp.headers.get("Content-Length", 0) or 0)
            downloaded = 0
            file_mode = "wb"

        session_start = time.monotonic()
        last_emit = 0.0
        speed_window: deque[Tuple[float, int]] = deque()
        window_bytes = 0
        chunk_read_size = 65536 if is_proxied else 131072
        # Якщо сервер НЕ підтримує докачування (наприклад, models.silero.ai повертає 200 без Range),
        # заміна проксі завжди скидає прогрес до 0! Тому для таких серверів:
        # 1) перемикаємо проксі лише якщо він справді дуже повільний (< 100 КБ/с), а не < 300 КБ/с;
        # 2) якщо вже завантажено > 25% або >= 8 МБ і швидкість жива (>= 50 КБ/с), даємо файлу докачатись!
        eff_min_speed = self._MIN_PROXY_SPEED_BPS if server_supports_range else self._MIN_NON_RESUME_PROXY_SPEED_BPS

        with open(tmp_dest, file_mode) as out_f:
            for chunk in resp.iter_content(chunk_size=chunk_read_size):
                if cancel_event is not None and cancel_event.is_set():
                    raise self._DownloadCancelledError("Cancelled by user")
                if not chunk:
                    continue
                out_f.write(chunk)
                chunk_len = len(chunk)
                downloaded += chunk_len
                now = time.monotonic()

                speed_window.append((now, chunk_len))
                window_bytes += chunk_len
                cutoff = now - self._SLOW_PROXY_WINDOW_SEC
                while speed_window and speed_window[0][0] < cutoff:
                    _, old_len = speed_window.popleft()
                    window_bytes -= old_len

                elapsed = max(0.05, now - session_start)
                if len(speed_window) > 1:
                    span = max(0.5, now - speed_window[0][0])
                    cur_speed = window_bytes / span
                else:
                    cur_speed = window_bytes / elapsed

                non_resume_deep_progress = (
                    not server_supports_range
                    and cur_speed >= 50 * 1024
                    and (downloaded >= 8 * 1024 * 1024 or (total > 0 and downloaded >= total * 0.25))
                )
                if (
                    abort_if_slow
                    and not non_resume_deep_progress
                    and elapsed >= self._SLOW_PROXY_GRACE_SEC
                    and cur_speed < eff_min_speed
                    and (total == 0 or downloaded < total)
                ):
                    raise self._SlowProxyError(
                        f"Проксі занадто повільний ({self._format_speed_label(cur_speed)}), перемикаємось на інший"
                    )

                if now - last_emit > 0.12:
                    last_emit = now
                    speed_label = f"{label} ({self._format_speed_label(cur_speed)})"
                    if total > 0:
                        pct = base_pct + int((downloaded / total) * span_pct)
                        mb_done = downloaded / (1024 * 1024)
                        mb_tot = total / (1024 * 1024)
                        self.download_progress.emit(
                            engine_id,
                            min(99, pct),
                            tr("tts.dl.mb_progress", "{label} ({done:.1f} / {total:.1f} МБ)", label=speed_label, done=mb_done, total=mb_tot),
                        )
                    else:
                        self.download_progress.emit(engine_id, base_pct + span_pct // 2, speed_label)

        if total > 0 and downloaded < total:
            raise IOError(tr("tts.err.download_interrupted", "Завантаження обірвалося ({done} із {total} байт)", done=downloaded, total=total))

        min_bytes = 5_000 if dest.name == "voices.pt" else 100_000
        if dest.suffix in (".pt", ".onnx") and downloaded < min_bytes:
            tmp_dest.unlink(missing_ok=True)
            raise IOError(tr("tts.err.file_too_small", "Завантажений файл занадто малий (пошкоджена відповідь проксі)"))
        if dest.suffix == ".pt":
            with open(tmp_dest, "rb") as chk_f:
                magic = chk_f.read(2)
            if magic != b"PK":
                tmp_dest.unlink(missing_ok=True)
                raise IOError(tr("tts.err.invalid_pt_archive", "Завантажений файл не є коректним архівом моделі PyTorch (.pt)"))

    @staticmethod
    def _has_internet_probe() -> bool:
        """Швидка перевірка наявності інтернету без прив'язки до конкретного сайту."""
        for host, port in (("1.1.1.1", 443), ("8.8.8.8", 53), ("huggingface.co", 443)):
            try:
                with socket.create_connection((host, port), timeout=1.5):
                    return True
            except OSError:
                continue
        return False

    def _wait_for_internet_or_cancel(
        self,
        engine_id: str,
        pct: int,
        dest: Path,
        cancel_event: Optional[threading.Event] = None,
        max_wait_sec: float = 240.0,
    ) -> bool:
        """
        Якщо інтернет тимчасово зник, чекає його відновлення (до 4 хв) із миттєвою реакцією на скасування,
        не псуючи файл .part і не перестрибуючи на повільні дзеркала.
        """
        if self._has_internet_probe():
            return True

        part_path = dest.with_suffix(dest.suffix + ".part")
        deadline = time.monotonic() + max_wait_sec
        last_ui = 0.0
        while time.monotonic() < deadline:
            if not self._running or (cancel_event is not None and cancel_event.is_set()):
                raise self._DownloadCancelledError("Cancelled by user")
            now = time.monotonic()
            if now - last_ui >= 1.5:
                last_ui = now
                mb_done = (part_path.stat().st_size / (1024 * 1024)) if part_path.exists() else 0.0
                self.download_progress.emit(
                    engine_id,
                    max(1, min(99, pct)),
                    tr(
                        "tts.dl.waiting_net",
                        "Втрачено інтернет — очікування відновлення ({done:.1f} МБ збережено)...",
                        done=mb_done,
                    ),
                )
            if self._has_internet_probe():
                for _ in range(2):
                    if not self._running or (cancel_event is not None and cancel_event.is_set()):
                        raise self._DownloadCancelledError("Cancelled by user")
                    time.sleep(0.5)
                return True
            time.sleep(0.5)
        return self._has_internet_probe()

    def _download_url_with_progress(
        self,
        url: str,
        dest: Path,
        engine_id: str,
        base_pct: int,
        span_pct: int,
        label: str,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        """
        1) Паралельно тестує офіційний сервер і дзеркала (Auto-Race) та завантажує файл у 5 потоків напряму з CDN.
        2) Якщо прямий доступ заблоковано або надто повільний — пробує власний проксі / локальний SOCKS5.
        3) Якщо і вони недоступні — знаходить пул найшвидших проксі (TCP-пінг + реальний бенчмарк 512 КБ)
           та запускає багатопотокове завантаження через Multi-Proxy Swarm (роздача блоків між найкращими
           проксі одночасно з автоматичною заміною проблемного потоку без зупинки решти).
        """
        if cancel_event is not None and cancel_event.is_set():
            raise self._DownloadCancelledError("Cancelled by user")
        bypass_enabled = bool(config.get("use_download_bypass", True))
        raw_urls = self._build_candidate_urls(url, bypass_enabled)
        raced = len(raw_urls) > 1
        urls_to_try = self._race_candidate_urls(raw_urls, cancel_event=cancel_event) if raced else raw_urls

        # Якщо Auto-Race вже перевірив усі дзеркала і жодне не відповіло — не витрачаємо 18 с на повтор мертвих дзеркал
        if raced:
            now_r = time.monotonic()
            has_fresh_meta = any(
                (now_r - float((self._url_probe_meta.get(u) or {}).get("ts", 0.0))) < 30.0
                for u in urls_to_try
            )
            if has_fresh_meta:
                ok_urls = [
                    u for u in urls_to_try
                    if (self._url_probe_meta.get(u) or {}).get("ok")
                    and (now_r - float((self._url_probe_meta.get(u) or {}).get("ts", 0.0))) < 30.0
                ]
                urls_to_try = ok_urls if ok_urls else [urls_to_try[0]]

        last_exc: Optional[Exception] = None
        # (url, proxies_dict, proxy_pool, label, known_total, known_resolved_url)
        slow_fallback_route: Optional[Tuple[str, Optional[Dict[str, str]], Optional[List[str]], str, int, str]] = None

        part_file = dest.with_suffix(dest.suffix + ".part")
        meta_file = dest.with_suffix(dest.suffix + ".part.meta")
        for candidate_url in urls_to_try:
            is_mirror = candidate_url != url
            tag = tr("tts.dl.mirror_tag", " [Дзеркало]") if is_mirror else ""
            probe_info = self._url_probe_meta.get(candidate_url) or {}
            probe_fresh = (time.monotonic() - float(probe_info.get("ts", 0.0))) < 30.0
            k_total = int(probe_info.get("total", 0) or 0) if (probe_fresh and probe_info.get("ok")) else 0
            k_resolved = str(probe_info.get("resolved_url", "") or "") if (probe_fresh and probe_info.get("ok")) else ""
            same_host_retries = 0
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise self._DownloadCancelledError("Cancelled by user")
                offset_before = (
                    part_file.stat().st_size if (part_file.exists() and not meta_file.exists())
                    else (meta_file.stat().st_mtime_ns if meta_file.exists() else 0)
                )
                try:
                    self._stream_download_attempt(
                        candidate_url,
                        dest,
                        engine_id,
                        base_pct,
                        span_pct,
                        f"{label}{tag}",
                        proxies=None,
                        connect_timeout=4.5 if bypass_enabled else 6.0,
                        read_timeout=22.0,
                        abort_if_slow=True,
                        cancel_event=cancel_event,
                        known_total=k_total,
                        known_resolved_url=k_resolved,
                    )
                    return
                except self._DownloadCancelledError:
                    raise
                except Exception as exc:
                    last_exc = exc
                    if isinstance(exc, self._SlowProxyError) and slow_fallback_route is None:
                        slow_fallback_route = (candidate_url, None, None, f"{label}{tag}", k_total, k_resolved)
                    offset_after = (
                        part_file.stat().st_size if (part_file.exists() and not meta_file.exists())
                        else (meta_file.stat().st_mtime_ns if meta_file.exists() else 0)
                    )
                    made_progress = offset_after != offset_before and offset_after > 0
                    print(
                        f"[TTSManager] Пряме завантаження перервалось ({candidate_url}): {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if not self._has_internet_probe():
                        if self._wait_for_internet_or_cancel(engine_id, base_pct, dest, cancel_event):
                            same_host_retries = 0
                            continue
                    if made_progress and not isinstance(exc, self._SlowProxyError) and same_host_retries < 3:
                        same_host_retries += 1
                        time.sleep(1.0)
                        continue
                    break

        if not bypass_enabled:
            if slow_fallback_route is not None and isinstance(last_exc, self._SlowProxyError):
                fb_url, fb_px, fb_pool, fb_lbl, fb_tot, fb_res = slow_fallback_route
                self._stream_download_attempt(
                    fb_url, dest, engine_id, base_pct, span_pct, fb_lbl,
                    proxies=fb_px, proxy_pool=fb_pool, connect_timeout=8.0, read_timeout=35.0,
                    abort_if_slow=False, cancel_event=cancel_event,
                    known_total=fb_tot, known_resolved_url=fb_res,
                )
                return
            if last_exc is not None:
                raise last_exc
            raise RuntimeError(tr("tts.err.file_download_failed", "Не вдалося завантажити файл {name}", name=dest.name))

        # User-configured proxy is a fallback too; do not route healthy direct downloads through it.
        custom_px = str(config.get("custom_proxy_url", "") or "").strip()
        if custom_px:
            px_dict = self._format_proxy_dict(custom_px)
            if px_dict:
                try:
                    self._stream_download_attempt(
                        url, dest, engine_id, base_pct, span_pct,
                        f"{label} [Власний проксі]", proxies=px_dict,
                        connect_timeout=6.0, read_timeout=22.0,
                        abort_if_slow=True, cancel_event=cancel_event,
                    )
                    return
                except self._DownloadCancelledError:
                    raise
                except Exception as exc:
                    last_exc = exc
                    if isinstance(exc, self._SlowProxyError) and slow_fallback_route is None:
                        slow_fallback_route = (url, px_dict, None, f"{label} [Власний проксі]", 0, "")
                    print(f"[TTSManager] Власний проксі ({custom_px}) не зміг завантажити {url}: {exc}", file=sys.stderr, flush=True)

        # Configured SOCKS5 is also a fallback, after the direct route.
        api_proxies = get_active_api_proxies()
        if api_proxies:
            if cancel_event is not None and cancel_event.is_set():
                raise self._DownloadCancelledError("Cancelled by user")
            try:
                self._stream_download_attempt(
                    url,
                    dest,
                    engine_id,
                    base_pct,
                    span_pct,
                    f"{label} [SOCKS5]",
                    proxies=api_proxies,
                    connect_timeout=6.0,
                    read_timeout=22.0,
                    abort_if_slow=True,
                    cancel_event=cancel_event,
                )
                return
            except self._DownloadCancelledError:
                raise
            except Exception as exc:
                last_exc = exc
                if isinstance(exc, self._SlowProxyError) and slow_fallback_route is None:
                    slow_fallback_route = (url, api_proxies, None, f"{label} [SOCKS5]", 0, "")
                print(
                    f"[TTSManager] Локальний SOCKS5-проксі не зміг завантажити {url}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

        # 2. Двоетапний пошук пулу найшвидших проксі + Multi-Proxy Swarm (паралельно через топ-проксі)
        def _saved_progress_pct() -> int:
            done_b, tot_b = self._read_part_progress(dest)
            if tot_b > 0 and done_b > 0:
                return min(98, max(3, base_pct + int((done_b / tot_b) * span_pct)))
            return max(3, base_pct + 2)

        self.download_progress.emit(
            engine_id,
            _saved_progress_pct(),
            tr("tts.dl.proxy_bypass", "Прямий доступ заблоковано — автоматична спроба через додатковий сервер ({name})...", name=dest.name),
        )
        print(
            f"[TTSManager] Прямий доступ не вдався, пошук найшвидших проксі для {url}...",
            file=sys.stderr,
            flush=True,
        )

        host_key = urllib.parse.urlparse(url).netloc.lower()
        tried_proxies: set = set()
        carry_alive_proxies: List[str] = []
        has_direct_slow_fallback = (
            slow_fallback_route is not None
            and slow_fallback_route[1] is None
            and slow_fallback_route[2] is None
        )
        max_proxy_rounds = 1 if has_direct_slow_fallback else 3
        for search_round in range(max_proxy_rounds):
            if cancel_event is not None and cancel_event.is_set():
                raise self._DownloadCancelledError("Cancelled by user")
            cur_saved_pct = _saved_progress_pct()
            if search_round == 0:
                working_proxies = self._get_cached_proxies(url, exclude_proxies=tried_proxies)
                if len(working_proxies) < 2:
                    fresh = self._find_working_proxies_for_url(
                        url,
                        max_count=12,
                        exclude_proxies=tried_proxies | set(working_proxies),
                        cancel_event=cancel_event,
                        engine_id=engine_id,
                        progress_pct=cur_saved_pct,
                    )
                    working_proxies = working_proxies + [p for p in fresh if p not in working_proxies]
            else:
                fresh = self._find_working_proxies_for_url(
                    url,
                    max_count=12,
                    exclude_proxies=tried_proxies,
                    cancel_event=cancel_event,
                    engine_id=engine_id,
                    progress_pct=cur_saved_pct,
                )
                # Поєднуємо нові швидкі проксі з живими з попереднього раунду, щоб не втрачати стабільні вузли
                working_proxies = fresh + [p for p in carry_alive_proxies if p not in fresh]

            working_proxies = [p for p in working_proxies if p not in tried_proxies or p in carry_alive_proxies]
            if not working_proxies:
                continue

            with self._proxy_lock:
                host_meta = dict(self._proxy_meta.get(host_key) or {})
            same_url_meta = host_meta.get("url") == url
            known_total = int(host_meta.get("total", 0) or 0) if same_url_meta else 0
            known_resolved = str(host_meta.get("resolved_url", "") or "") if same_url_meta else ""

            # Якщо на диску вже є збережений .part.meta з попередніх раундів — беремо його total,
            # щоб гарантовано продовжити багатопотоково без скидання прогресу!
            _, meta_saved_total = self._read_part_progress(dest)
            if known_total < self._MULTISTREAM_MIN_SIZE_BYTES and meta_saved_total >= self._MULTISTREAM_MIN_SIZE_BYTES:
                known_total = meta_saved_total

            if known_total < self._MULTISTREAM_MIN_SIZE_BYTES and int(self._MULTISTREAM_WORKERS) > 1:
                probe_candidates = working_proxies[:3]
                if len(probe_candidates) == 1:
                    probe_px = self._format_proxy_dict(probe_candidates[0])
                    known_total, known_resolved = self._probe_range_info(url, probe_px, 4.5, 8.0)
                else:
                    def _probe_cached_px(px_u: str) -> Tuple[int, str]:
                        return self._probe_range_info(url, self._format_proxy_dict(px_u), 4.0, 6.0)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=len(probe_candidates)) as p_pool:
                        for tot_c, res_c in p_pool.map(_probe_cached_px, probe_candidates):
                            if tot_c >= self._MULTISTREAM_MIN_SIZE_BYTES:
                                known_total, known_resolved = tot_c, res_c
                                break

            has_more_rounds = (search_round + 1) < max_proxy_rounds

            # Якщо файл >= 2 МБ і сервер підтримує Range — запускаємо Multi-Proxy Swarm одразу по всіх топ-проксі!
            if known_total >= self._MULTISTREAM_MIN_SIZE_BYTES and int(self._MULTISTREAM_WORKERS) > 1:
                try:
                    print(
                        f"[TTSManager] Запуск Multi-Proxy Swarm (раунд {search_round + 1}, {len(working_proxies)} проксі, "
                        f"найкращий: {working_proxies[0]})...",
                        file=sys.stderr,
                        flush=True,
                    )
                    self._stream_download_attempt(
                        url,
                        dest,
                        engine_id,
                        base_pct,
                        span_pct,
                        label,
                        proxies=None,
                        proxy_pool=working_proxies,
                        connect_timeout=5.0,
                        read_timeout=14.0 if (has_more_rounds or slow_fallback_route is not None) else 25.0,
                        abort_if_slow=has_more_rounds or (slow_fallback_route is not None),
                        cancel_event=cancel_event,
                        known_total=known_total,
                        known_resolved_url=known_resolved,
                    )
                    return
                except self._DownloadCancelledError:
                    raise
                except self._SlowProxyError as exc:
                    last_exc = exc
                    alive_from_round = [p for p in self._get_cached_proxies(url) if p in working_proxies] or working_proxies[:4]
                    carry_alive_proxies = alive_from_round[:2]
                    if slow_fallback_route is None and alive_from_round:
                        slow_fallback_route = (url, None, alive_from_round, label, known_total, known_resolved)
                    tried_proxies.update(working_proxies)
                    print(
                        f"[TTSManager] Multi-Proxy Swarm занадто повільний ({exc}), шукаємо швидший пул...",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                except Exception as exc:
                    last_exc = exc
                    if not self._has_internet_probe():
                        if self._wait_for_internet_or_cancel(engine_id, _saved_progress_pct(), dest, cancel_event):
                            continue
                    carry_alive_proxies = [p for p in self._get_cached_proxies(url) if p in working_proxies][:2]
                    tried_proxies.update(working_proxies)
                    print(
                        f"[TTSManager] Multi-Proxy Swarm раунд {search_round + 1} завершився помилкою: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue

            # Для невеликих файлів (< 2 МБ) або серверів без HTTP Range — качаємо через топ-відсортовані проксі
            for idx, px_url in enumerate(working_proxies):
                if cancel_event is not None and cancel_event.is_set():
                    raise self._DownloadCancelledError("Cancelled by user")
                if px_url in tried_proxies:
                    continue
                tried_proxies.add(px_url)
                has_fallback = (idx + 1 < len(working_proxies)) or has_more_rounds or (slow_fallback_route is not None)
                px_dict = self._format_proxy_dict(px_url)
                if not px_dict:
                    continue
                route_tag = (
                    tr("tts.dl.best_proxy_tag", "{label} [Найкращий проксі]", label=label)
                    if idx == 0 and search_round == 0
                    else tr("tts.dl.auto_proxy_tag", "{label} [Авто-проксі #{num}]", label=label, num=idx + 1)
                )
                try:
                    print(
                        f"[TTSManager] Завантаження через найшвидший проксі ({px_url})...",
                        file=sys.stderr,
                        flush=True,
                    )
                    self._stream_download_attempt(
                        url,
                        dest,
                        engine_id,
                        base_pct,
                        span_pct,
                        route_tag,
                        proxies=px_dict,
                        connect_timeout=4.5,
                        read_timeout=14.0 if has_fallback else 28.0,
                        abort_if_slow=has_fallback,
                        allow_multistream=False,
                        cancel_event=cancel_event,
                    )
                    return
                except self._DownloadCancelledError:
                    raise
                except self._SlowProxyError as exc:
                    last_exc = exc
                    if slow_fallback_route is None:
                        slow_fallback_route = (url, px_dict, None, route_tag, 0, "")
                    print(
                        f"[TTSManager] Проксі ({px_url}) повільний ({exc}), пробуємо наступний...",
                        file=sys.stderr,
                        flush=True,
                    )
                except Exception as exc:
                    last_exc = exc
                    if not self._has_internet_probe():
                        if self._wait_for_internet_or_cancel(engine_id, base_pct, dest, cancel_event):
                            tried_proxies.discard(px_url)
                            continue
                    self._mark_proxy_bad(px_url, url)
                    print(
                        f"[TTSManager] Проксі ({px_url}) замінено: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )

        if slow_fallback_route is not None:
            fb_url, fb_px, fb_pool, fb_lbl, fb_tot, fb_res = slow_fallback_route
            self._stream_download_attempt(
                fb_url,
                dest,
                engine_id,
                base_pct,
                span_pct,
                fb_lbl,
                proxies=fb_px,
                proxy_pool=fb_pool,
                connect_timeout=8.0,
                read_timeout=35.0,
                abort_if_slow=False,
                cancel_event=cancel_event,
                known_total=fb_tot,
                known_resolved_url=fb_res,
            )
            return
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(tr("tts.err.file_download_failed", "Не вдалося завантажити файл {name}", name=dest.name))


tts_manager = TTSEngineManager()
