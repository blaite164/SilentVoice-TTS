"""
core/audio_router.py — Маршрутизація звуку через sounddevice (PortAudio):
- Одночасний вивід у 2 канали: Віртуальний мікрофон (VB-Cable для Discord/ігор) + Навушники (моніторинг для себе)
- Очищення країв аудіо (DC offset, Hann fade-in/fade-out, анти-кліпінг лімітер)
- Тихий звуковий відгук (тільки в навушники) при активації/відправці набору в грі
- Перевірка наявності VB-Cable та встановлення в 1 клік
"""

from __future__ import annotations

from collections import deque
import ctypes
from ctypes import wintypes
import queue
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import sounddevice as sd
import soundfile as sf

from core.config import DATA_DIR, config
from core.i18n import tr
from core.win_api import shell32
from core.vban import VbanPlayback, shutdown_vban, start_samples as start_vban_samples


def clean_audio_edges(
    samples: np.ndarray,
    sr: int,
    fade_in_ms: float = 6.0,
    fade_out_ms: float = 12.0,
    pad_tail_ms: float = 55.0,
) -> np.ndarray:
    """
    Розумне очищення країв аудіо (умовне зняття DC offset + адаптивний Hann fade-in/fade-out
    лише за потреби + короткий нульовий хвіст для чистого спорожнення буфера PortAudio),
    що прибирає клацання без «з'їдання» перших приголосних.
    """
    if samples is None or len(samples) == 0:
        return np.zeros(0, dtype=np.float32)

    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 2:
        if x.shape[1] == 0:
            return np.zeros(0, dtype=np.float32)
        if x.shape[1] == 1:
            x = x[:, 0].copy()
        elif x.shape[1] == 2:
            x = x.copy()
        else:
            x = np.mean(x, axis=1, dtype=np.float32)
    elif x.ndim > 2:
        x = np.mean(x.reshape(len(x), -1), axis=1, dtype=np.float32)
    else:
        x = x.copy()

    # 1. Умовне зняття DC offset (лише коли постійне зміщення реально відчутне, щоб не викривляти асиметричну мову)
    if x.ndim == 2:
        dc = np.mean(x, axis=0, keepdims=True)
        peak = np.max(np.abs(x), axis=0, keepdims=True)
        dc_mask = (np.abs(dc) > 0.005) & (np.abs(dc) > peak * 0.02)
        x -= np.where(dc_mask, dc, 0.0).astype(np.float32)
    else:
        dc_val = float(np.mean(x))
        peak_val = float(np.max(np.abs(x)))
        if abs(dc_val) > 0.005 and abs(dc_val) > peak_val * 0.02:
            x -= dc_val

    # 2. Адаптивний Hann fade-in: повний fade при різкому стрибку на старті, мікро-fade 2 мс якщо старт уже плавний
    head_slice = x[: min(8, len(x))]
    start_jump = float(np.max(np.abs(head_slice))) > 0.015
    eff_fade_in_ms = float(fade_in_ms) if start_jump else min(float(fade_in_ms), 2.0)
    n_in = min(int(sr * eff_fade_in_ms / 1000.0), len(x) // 4)
    if n_in > 0:
        hann_in = (0.5 * (1.0 - np.cos(np.pi * np.arange(n_in, dtype=np.float32) / float(n_in)))).astype(np.float32)
        if x.ndim == 2:
            x[:n_in] *= hann_in[:, None]
        else:
            x[:n_in] *= hann_in

    # 3. Адаптивний Hann fade-out: повний fade при обриві на кінці, короткий 3 мс якщо кінець уже тихий
    tail_slice = x[max(0, len(x) - 8) :]
    end_jump = float(np.max(np.abs(tail_slice))) > 0.015
    eff_fade_out_ms = float(fade_out_ms) if end_jump else min(float(fade_out_ms), 3.0)
    n_out = min(int(sr * eff_fade_out_ms / 1000.0), len(x) // 4)
    if n_out > 0:
        hann_out = (0.5 * (1.0 + np.cos(np.pi * np.arange(n_out, dtype=np.float32) / float(n_out)))).astype(np.float32)
        if x.ndim == 2:
            x[-n_out:] *= hann_out[:, None]
        else:
            x[-n_out:] *= hann_out

    # 4. Короткий нульовий хвіст для чистого спорожнення буфера PortAudio
    n_pad = int(sr * pad_tail_ms / 1000.0)
    if n_pad > 0:
        pad = np.zeros((n_pad, x.shape[1]), dtype=np.float32) if x.ndim == 2 else np.zeros(n_pad, dtype=np.float32)
        x = np.concatenate([x, pad], axis=0)

    return x


def apply_volume_and_limiter(samples: np.ndarray, volume_percent: int) -> np.ndarray:
    """Масштабує гучність (0..200%) із м'яким tanh-лімітером проти перевантаження."""
    if len(samples) == 0:
        return samples
    scale = max(0.0, float(volume_percent) / 100.0)
    if scale <= 0.0:
        return np.zeros_like(samples)
    if abs(scale - 1.0) < 1e-4:
        return np.clip(samples, -1.0, 1.0)

    scaled = samples * scale
    peak = float(np.max(np.abs(scaled)))
    if peak > 0.82:
        threshold = 0.82
        mask = np.abs(scaled) > threshold
        if np.any(mask):
            excess = np.abs(scaled[mask]) - threshold
            compressed = threshold + 0.17 * np.tanh(excess / 0.30)
            scaled[mask] = np.sign(scaled[mask]) * compressed
    return np.clip(scaled, -1.0, 1.0).astype(np.float32)


def get_output_devices() -> List[Dict[str, object]]:
    """
    Повертає чистий список пристроїв виводу звуку без дублікатів між драйверами.
    Пріоритетно використовуються індекси MME (найстабільніший API у фонових потоках Windows),
    повні необрізані назви підтягуються з WASAPI / DirectSound, а якщо пристрій (наприклад,
    Bluetooth-навушники) доступний лише у WASAPI — він також потрапляє до списку.
    """
    result: List[Dict[str, object]] = []
    seen_mme_raw: set = set()
    seen_full_low: set = set()
    seen_prefix31: set = set()
    try:
        hostapis = sd.query_hostapis()
        devices = sd.query_devices()

        # Карта повних назв із Windows WASAPI та Windows DirectSound (де назви не обрізані до 31 символу)
        exact_full_map: Dict[str, str] = {}
        prefix31_map: Dict[str, str] = {}
        prefix26_map: Dict[str, str] = {}
        wasapi_outputs: List[Tuple[int, str, int]] = []
        for api_target in ("Windows WASAPI", "Windows DirectSound"):
            for idx, dev in enumerate(devices):
                if int(dev.get("max_output_channels", 0)) <= 0:
                    continue
                h_idx = int(dev.get("hostapi", -1))
                h_name = hostapis[h_idx]["name"] if 0 <= h_idx < len(hostapis) else ""
                if h_name != api_target:
                    continue
                full_name = str(dev.get("name", "")).strip()
                low_full = full_name.lower()
                if not full_name or "sound mapper" in low_full or "primary sound" in low_full:
                    continue
                exact_full_map.setdefault(low_full, full_name)
                prefix31_map.setdefault(low_full[:31], full_name)
                prefix26_map.setdefault(low_full[:26], full_name)
                if api_target == "Windows WASAPI":
                    wasapi_outputs.append((idx, full_name, int(dev.get("default_samplerate", 48000) or 48000)))

        # 1. Основний список формуємо за індексами MME (найстабільніший API у робочих потоках Windows)
        for idx, dev in enumerate(devices):
            if int(dev.get("max_output_channels", 0)) <= 0:
                continue
            h_idx = int(dev.get("hostapi", -1))
            h_name = hostapis[h_idx]["name"] if 0 <= h_idx < len(hostapis) else ""
            if h_name != "MME":
                continue
            raw_name = str(dev.get("name", "")).strip()
            low_raw = raw_name.lower()
            if not raw_name or "sound mapper" in low_raw or "primary sound" in low_raw:
                continue
            if low_raw in seen_mme_raw:
                continue
            seen_mme_raw.add(low_raw)
            display_name = (
                exact_full_map.get(low_raw)
                or prefix31_map.get(low_raw[:31])
                or (prefix26_map.get(low_raw[:26]) if len(low_raw) >= 28 else None)
                or raw_name
            )
            seen_full_low.add(display_name.lower())
            seen_prefix31.add(display_name.lower()[:31])
            seen_prefix31.add(low_raw[:31])
            result.append({
                "index": idx,
                "name": display_name,
                "hostapi": h_name,
                "default_samplerate": int(dev.get("default_samplerate", 44100)),
            })

        # 2. Якщо пристрій є у Windows WASAPI (наприклад, щойно підключені Bluetooth/USB-навушники),
        # але його немає в MME — додаємо його, щоб навушники ніколи не зникали зі списку.
        for idx, full_name, sr_val in wasapi_outputs:
            low_full = full_name.lower()
            if low_full in seen_full_low or low_full[:31] in seen_prefix31:
                continue
            seen_full_low.add(low_full)
            seen_prefix31.add(low_full[:31])
            result.append({
                "index": idx,
                "name": full_name,
                "hostapi": "Windows WASAPI",
                "default_samplerate": sr_val,
            })
    except Exception as exc:
        print(f"[AudioRouter] get_output_devices error: {exc}")
    return result


def get_input_devices() -> List[Dict[str, object]]:
    """
    Повертає список фізичних мікрофонів (пристроїв вводу) без дублікатів між драйверами
    та без VB-Cable, щоб уникнути зациклення звуку.
    """
    result: List[Dict[str, object]] = []
    seen_names: set = set()
    try:
        hostapis = sd.query_hostapis()
        devices = sd.query_devices()

        full_names_map: Dict[str, str] = {}
        for api_target in ("Windows WASAPI", "Windows DirectSound"):
            for dev in devices:
                if int(dev.get("max_input_channels", 0)) <= 0:
                    continue
                h_idx = int(dev.get("hostapi", -1))
                h_name = hostapis[h_idx]["name"] if 0 <= h_idx < len(hostapis) else ""
                if h_name != api_target:
                    continue
                full_name = str(dev.get("name", "")).strip()
                low = full_name.lower()
                if full_name and "sound mapper" not in low and "primary sound" not in low:
                    prefix = full_name[:26].lower()
                    if prefix not in full_names_map:
                        full_names_map[prefix] = full_name

        for idx, dev in enumerate(devices):
            if int(dev.get("max_input_channels", 0)) <= 0:
                continue
            h_idx = int(dev.get("hostapi", -1))
            h_name = hostapis[h_idx]["name"] if 0 <= h_idx < len(hostapis) else ""
            if h_name != "MME":
                continue
            raw_name = str(dev.get("name", "")).strip()
            low_raw = raw_name.lower()
            if (
                not raw_name
                or "sound mapper" in low_raw
                or "primary sound" in low_raw
                or "cable" in low_raw
                or "vb-audio" in low_raw
            ):
                continue
            norm_key = raw_name[:26].lower()
            if norm_key in seen_names:
                continue
            seen_names.add(norm_key)
            display_name = full_names_map.get(norm_key, raw_name)
            if "cable" in display_name.lower() or "vb-audio" in display_name.lower():
                continue
            result.append({
                "index": idx,
                "name": display_name,
                "hostapi": h_name,
                "default_samplerate": int(dev.get("default_samplerate", 44100)),
            })
    except Exception as exc:
        print(f"[AudioRouter] get_input_devices error: {exc}")
    return result


def resolve_input_device_index(device_name: str) -> Optional[int]:
    """Знаходить MME-індекс фізичного мікрофона (або стандартного мікрофона Windows без VB-Cable)."""
    target = (device_name or "").strip().lower()
    mme_inputs = get_input_devices()
    if not mme_inputs:
        return None
    if target:
        for dev in mme_inputs:
            d_name = str(dev["name"]).strip().lower()
            if d_name == target or target in d_name or d_name in target:
                return int(dev["index"])  # type: ignore[arg-type]
    # Якщо обрано "Стандартний мікрофон Windows" або пристрій не знайдено — перевіряємо системний за замовчуванням
    try:
        def_in = sd.query_devices(None, "input")
        def_name = str(def_in.get("name", "")).strip().lower()
        if def_name and "cable" not in def_name and "vb-audio" not in def_name:
            for dev in mme_inputs:
                if str(dev["name"]).strip().lower()[:20] in def_name or def_name[:20] in str(dev["name"]).strip().lower():
                    return int(dev["index"])  # type: ignore[arg-type]
    except Exception:
        pass
    return int(mme_inputs[0]["index"])  # type: ignore[arg-type]


def resolve_output_device_index(device_name: str) -> Optional[int]:
    """Знаходить MME-індекс пристрою sounddevice за назвою (спочатку точний збіг, потім частковий)."""
    if not device_name or not device_name.strip() or device_name.strip() == "__off__":
        return None
    target = device_name.strip().lower()
    try:
        # Спочатку шукаємо серед очищеного списку MME-пристроїв із повними іменами
        mme_devices = get_output_devices()
        for dev in mme_devices:
            if str(dev["name"]).strip().lower() == target:
                return int(dev["index"])
        prefix_len = 24
        for dev in mme_devices:
            d_name = str(dev["name"]).strip().lower()
            if target.startswith(d_name[:prefix_len]) or d_name.startswith(target[:prefix_len]):
                return int(dev["index"])
        for dev in mme_devices:
            d_name = str(dev["name"]).strip().lower()
            if "cable input" in target and "cable input" in d_name:
                return int(dev["index"])
    except Exception as exc:
        print(f"[AudioRouter] resolve_output_device_index error: {exc}")
    return None


_VIRTUAL_ENHANCER_TOKENS = (
    "fxsound",
    "voicemeeter",
    "virtual",
    "sonar",
    "krisp",
    "broadcast",
    "wave link",
    "steam streaming",
    "oculus",
    "enhancer",
)


def _is_cable_device_name(name: str) -> bool:
    low = str(name or "").lower()
    return "cable" in low or "vb-audio" in low


def _is_virtual_enhancer_name(name: str) -> bool:
    low = str(name or "").lower()
    return any(tok in low for tok in _VIRTUAL_ENHANCER_TOKENS)


def _get_core_audio_default_output_names() -> Dict[str, str]:
    """Query live Windows Core Audio defaults (eConsole, eMultimedia, eCommunications)."""
    result: Dict[str, str] = {}
    if not hasattr(ctypes, "windll"):
        return result
    try:
        import uuid

        ole32 = ctypes.windll.ole32

        class _GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", ctypes.c_uint32),
                ("Data2", ctypes.c_uint16),
                ("Data3", ctypes.c_uint16),
                ("Data4", ctypes.c_ubyte * 8),
            ]

            def __init__(self, text: str) -> None:
                u = uuid.UUID(text)
                super().__init__(
                    u.time_low,
                    u.time_mid,
                    u.time_hi_version,
                    (ctypes.c_ubyte * 8)(*u.bytes[8:]),
                )

        class _PROPERTYKEY(ctypes.Structure):
            _fields_ = [("fmtid", _GUID), ("pid", ctypes.c_uint32)]

        class _PROPVARIANT(ctypes.Structure):
            _fields_ = [
                ("vt", ctypes.c_ushort),
                ("r1", ctypes.c_ushort),
                ("r2", ctypes.c_ushort),
                ("r3", ctypes.c_ushort),
                ("val", ctypes.c_void_p),
                ("pad", ctypes.c_void_p),
            ]

        def _vtbl(ptr: ctypes.c_void_p, idx: int, restype, *argtypes):
            table = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(table[idx])

        def _release(ptr: Optional[ctypes.c_void_p]) -> None:
            if ptr and ptr.value:
                _vtbl(ptr, 2, ctypes.c_ulong)(ptr)

        hr_init = ole32.CoInitializeEx(None, 2)  # COINIT_APARTMENTTHREADED
        should_uninit = hr_init in (0, 1)
        enum_ptr = ctypes.c_void_p()
        try:
            clsid_enum = _GUID("{BCDE0395-E52F-467C-8E3D-C4579291692E}")
            iid_enum = _GUID("{A95664D2-9614-4F35-A746-DE8DB63617E6}")
            hr = ole32.CoCreateInstance(
                ctypes.byref(clsid_enum),
                None,
                1,
                ctypes.byref(iid_enum),
                ctypes.byref(enum_ptr),
            )
            if hr != 0 or not enum_ptr.value:
                return result

            get_default = _vtbl(
                enum_ptr,
                4,
                ctypes.c_long,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_void_p),
            )
            pk = _PROPERTYKEY(_GUID("{a45c254e-df1c-4efd-8020-67d146a850e0}"), 14)
            for key, role in (("console", 0), ("multimedia", 1), ("communications", 2)):
                dev_ptr = ctypes.c_void_p()
                store_ptr = ctypes.c_void_p()
                pv = _PROPVARIANT()
                try:
                    if get_default(enum_ptr, 0, role, ctypes.byref(dev_ptr)) != 0 or not dev_ptr.value:
                        continue
                    open_store = _vtbl(dev_ptr, 4, ctypes.c_long, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p))
                    if open_store(dev_ptr, 0, ctypes.byref(store_ptr)) != 0 or not store_ptr.value:
                        continue
                    get_val = _vtbl(store_ptr, 5, ctypes.c_long, ctypes.POINTER(_PROPERTYKEY), ctypes.POINTER(_PROPVARIANT))
                    if get_val(store_ptr, ctypes.byref(pk), ctypes.byref(pv)) == 0 and pv.vt == 31 and pv.val:
                        name = str(ctypes.wstring_at(pv.val) or "").strip()
                        if name:
                            result[key] = name
                finally:
                    try:
                        ole32.PropVariantClear(ctypes.byref(pv))
                    except Exception:
                        pass
                    _release(store_ptr)
                    _release(dev_ptr)
        finally:
            _release(enum_ptr)
            if should_uninit:
                ole32.CoUninitialize()
    except Exception:
        pass
    return result


def _get_windows_preferred_output_name() -> str:
    """Повертає актуальну назву пристрою виводу Windows за замовчуванням через WinMM."""
    try:
        winmm = ctypes.windll.winmm
        dev_id = wintypes.UINT(0)
        flags = wintypes.UINT(0)
        # DRVM_MAPPER_PREFERRED_GET = 0x2015
        res = winmm.waveOutMessage(wintypes.HANDLE(-1), 0x2015, ctypes.byref(dev_id), ctypes.byref(flags))
        if res != 0:
            return ""
        num_devs = winmm.waveOutGetNumDevs()
        if dev_id.value >= num_devs:
            return ""

        class _WAVEOUTCAPSW(ctypes.Structure):
            _fields_ = [
                ("wMid", wintypes.WORD),
                ("wPid", wintypes.WORD),
                ("vDriverVersion", wintypes.UINT),
                ("szPname", wintypes.WCHAR * 32),
                ("dwFormats", wintypes.DWORD),
                ("wChannels", wintypes.WORD),
                ("wReserved1", wintypes.WORD),
                ("dwSupport", wintypes.DWORD),
            ]

        caps = _WAVEOUTCAPSW()
        if winmm.waveOutGetDevCapsW(dev_id.value, ctypes.byref(caps), ctypes.sizeof(caps)) == 0:
            return str(caps.szPname or "").strip()
    except Exception:
        pass
    return ""


def _get_sound_mapper_output_index() -> Optional[int]:
    """Повертає MME-індекс 'Microsoft Sound Mapper - Output', який маршрутизує у поточний дефолт Windows."""
    try:
        hostapis = sd.query_hostapis()
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if int(dev.get("max_output_channels", 0)) <= 0:
                continue
            h_idx = int(dev.get("hostapi", -1))
            h_name = hostapis[h_idx]["name"] if 0 <= h_idx < len(hostapis) else ""
            if h_name == "MME" and "sound mapper" in str(dev.get("name", "")).lower():
                return idx
    except Exception:
        pass
    return None


_PA_REFRESH_LOCK = threading.Lock()
_LAST_CORE_AUDIO_SNAPSHOT: Optional[Tuple[str, str, str]] = None
_LAST_ACTIVE_CORE_AUDIO_OUTPUTS: Optional[Tuple[str, ...]] = None


def get_configured_audio_output_format() -> str:
    """Повертає налаштований формат виводу звуку ('auto' або 'int32')."""
    raw = str(config.get("audio_output_format", "auto") or "auto").strip().lower()
    return "int32" if raw in ("int32", "i32") else "auto"


def _describe_device_for_log(device_idx: Optional[int]) -> str:
    if device_idx is None:
        return "None (default)"
    try:
        info = sd.query_devices(device_idx, "output")
        name = str(info.get("name", "") or "").strip()
        return f"{device_idx} ({name!r})" if name else str(device_idx)
    except Exception:
        return str(device_idx)


def get_core_audio_active_device_names(data_flow: int = 0) -> Tuple[str, ...]:
    """
    Повертає актуальний відсортований кортеж назв активних пристроїв Windows Core Audio
    без перезавантаження PortAudio (data_flow: 0 = eRender/вивід, 1 = eCapture/мікрофони).
    """
    if not hasattr(ctypes, "windll"):
        return ()
    names: List[str] = []
    try:
        import uuid

        ole32 = ctypes.windll.ole32

        class _GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", ctypes.c_uint32),
                ("Data2", ctypes.c_uint16),
                ("Data3", ctypes.c_uint16),
                ("Data4", ctypes.c_ubyte * 8),
            ]

            def __init__(self, text: str) -> None:
                u = uuid.UUID(text)
                super().__init__(
                    u.time_low,
                    u.time_mid,
                    u.time_hi_version,
                    (ctypes.c_ubyte * 8)(*u.bytes[8:]),
                )

        class _PROPERTYKEY(ctypes.Structure):
            _fields_ = [("fmtid", _GUID), ("pid", ctypes.c_uint32)]

        class _PROPVARIANT(ctypes.Structure):
            _fields_ = [
                ("vt", ctypes.c_ushort),
                ("r1", ctypes.c_ushort),
                ("r2", ctypes.c_ushort),
                ("r3", ctypes.c_ushort),
                ("val", ctypes.c_void_p),
                ("pad", ctypes.c_void_p),
            ]

        def _vtbl(ptr: ctypes.c_void_p, idx: int, restype, *argtypes):
            table = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(table[idx])

        def _release(ptr: Optional[ctypes.c_void_p]) -> None:
            if ptr and ptr.value:
                _vtbl(ptr, 2, ctypes.c_ulong)(ptr)

        hr_init = ole32.CoInitializeEx(None, 2)
        should_uninit = hr_init in (0, 1)
        enum_ptr = ctypes.c_void_p()
        col_ptr = ctypes.c_void_p()
        try:
            clsid_enum = _GUID("{BCDE0395-E52F-467C-8E3D-C4579291692E}")
            iid_enum = _GUID("{A95664D2-9614-4F35-A746-DE8DB63617E6}")
            hr = ole32.CoCreateInstance(
                ctypes.byref(clsid_enum),
                None,
                1,
                ctypes.byref(iid_enum),
                ctypes.byref(enum_ptr),
            )
            if hr != 0 or not enum_ptr.value:
                return ()
            enum_endpoints = _vtbl(
                enum_ptr, 3, ctypes.c_long, ctypes.c_int, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)
            )
            # DEVICE_STATE_ACTIVE = 1
            if enum_endpoints(enum_ptr, int(data_flow), 1, ctypes.byref(col_ptr)) != 0 or not col_ptr.value:
                return ()
            count = ctypes.c_uint32(0)
            get_count = _vtbl(col_ptr, 3, ctypes.c_long, ctypes.POINTER(ctypes.c_uint32))
            if get_count(col_ptr, ctypes.byref(count)) != 0:
                return ()
            get_item = _vtbl(col_ptr, 4, ctypes.c_long, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p))
            pk = _PROPERTYKEY(_GUID("{a45c254e-df1c-4efd-8020-67d146a850e0}"), 14)
            for i in range(int(count.value)):
                dev_ptr = ctypes.c_void_p()
                store_ptr = ctypes.c_void_p()
                pv = _PROPVARIANT()
                try:
                    if get_item(col_ptr, i, ctypes.byref(dev_ptr)) != 0 or not dev_ptr.value:
                        continue
                    open_store = _vtbl(dev_ptr, 4, ctypes.c_long, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p))
                    if open_store(dev_ptr, 0, ctypes.byref(store_ptr)) != 0 or not store_ptr.value:
                        continue
                    get_val = _vtbl(store_ptr, 5, ctypes.c_long, ctypes.POINTER(_PROPERTYKEY), ctypes.POINTER(_PROPVARIANT))
                    if get_val(store_ptr, ctypes.byref(pk), ctypes.byref(pv)) == 0 and pv.vt == 31 and pv.val:
                        name = str(ctypes.wstring_at(pv.val) or "").strip()
                        if name:
                            names.append(name)
                finally:
                    try:
                        ole32.PropVariantClear(ctypes.byref(pv))
                    except Exception:
                        pass
                    _release(store_ptr)
                    _release(dev_ptr)
        finally:
            _release(col_ptr)
            _release(enum_ptr)
            if should_uninit:
                ole32.CoUninitialize()
    except Exception:
        return ()
    return tuple(sorted(set(names), key=str.lower))


def get_core_audio_active_output_names() -> Tuple[str, ...]:
    """Повертає актуальний список активних пристроїв виводу Windows Core Audio (eRender)."""
    return get_core_audio_active_device_names(0)


def get_core_audio_active_all_device_names() -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Повертає пару (пристрої_виводу, мікрофони) з Windows Core Audio для автооновлення списків."""
    return (
        get_core_audio_active_device_names(0),
        get_core_audio_active_device_names(1),
    )


def ensure_portaudio_synced_with_windows() -> bool:
    """
    Порівнює живий список активних пристроїв Windows Core Audio з останнім відомим станом
    і перезавантажує PortAudio ЛИШЕ тоді, коли пристрої реально змінилися (підключили або
    відключили навушники/Bluetooth). Повертає True, якщо PortAudio було оновлено.
    """
    global _LAST_ACTIVE_CORE_AUDIO_OUTPUTS
    live_outputs = get_core_audio_active_output_names()
    if not live_outputs:
        return False
    prev = _LAST_ACTIVE_CORE_AUDIO_OUTPUTS
    if prev is None:
        _LAST_ACTIVE_CORE_AUDIO_OUTPUTS = live_outputs
        # Якщо вже при першій перевірці в PortAudio бракує реального активного пристрою Windows — синхронізуємось
        missing_any = any(
            not _is_cable_device_name(name) and resolve_output_device_index(name) is None
            for name in live_outputs
        )
        if not missing_any:
            return False
    elif live_outputs == prev:
        return False

    _LAST_ACTIVE_CORE_AUDIO_OUTPUTS = live_outputs
    refreshed = refresh_portaudio_devices()
    if refreshed:
        _log_audio(
            f"[AudioRouter] Змінився список пристроїв Windows ({len(live_outputs)} активних): "
            f"оновлено PortAudio -> {list(live_outputs)}"
        )
    return refreshed


def _log_audio(msg: str) -> None:
    """Безпечний вивід діагностичних логів AudioRouter без UnicodeEncodeError на cp1252-консолях."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        try:
            enc = getattr(sys.stdout, "encoding", None) or "ascii"
            print(msg.encode(enc, errors="replace").decode(enc, errors="replace"), flush=True)
        except Exception:
            pass
    except Exception:
        pass


def _core_audio_snapshot(core_defaults: Optional[Dict[str, str]] = None) -> Tuple[str, str, str]:
    info = core_defaults if core_defaults is not None else _get_core_audio_default_output_names()
    return (
        str(info.get("console", "")).strip().lower(),
        str(info.get("multimedia", "")).strip().lower(),
        str(info.get("communications", "")).strip().lower(),
    )


def _has_unregistered_real_default_output(core_defaults: Optional[Dict[str, str]] = None) -> bool:
    """Повертає True, якщо Windows вказує на реальний пристрій виводу, якого ще немає в кеші PortAudio."""
    info = core_defaults if core_defaults is not None else _get_core_audio_default_output_names()
    for role in ("console", "multimedia", "communications"):
        cand = str(info.get(role, "")).strip()
        if not cand or _is_cable_device_name(cand) or _is_virtual_enhancer_name(cand):
            continue
        if resolve_output_device_index(cand) is None:
            return True
    return False


def _slice_audio_from_seconds(
    samples: Optional[np.ndarray],
    sr: int,
    offset_sec: float,
    fade_in_ms: float = 6.0,
) -> np.ndarray:
    """Вирізає залишок аудіо з позиції offset_sec із коротким fade-in для безшовного перемикання пристрою."""
    if samples is None or len(samples) == 0 or sr <= 0:
        return np.zeros(0, dtype=np.float32)
    start_idx = max(0, min(len(samples), int(round(max(0.0, offset_sec) * float(sr)))))
    if len(samples) - start_idx <= int(0.05 * float(sr)):
        return np.zeros(0, dtype=np.float32)
    chunk = np.array(samples[start_idx:], dtype=np.float32, copy=True)
    fade_n = min(len(chunk), max(1, int(float(sr) * fade_in_ms / 1000.0)))
    if fade_n > 1:
        ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        if chunk.ndim == 2:
            chunk[:fade_n] *= ramp[:, None]
        else:
            chunk[:fade_n] *= ramp
    return chunk


def refresh_portaudio_devices() -> bool:
    """
    Перезавантажує кеш пристроїв PortAudio, щоб виявити щойно підключені
    навушники, USB-гарнітури чи Bluetooth без перезапуску програми.
    """
    router = globals().get("audio_router")
    if router is not None:
        return bool(router.refresh_devices())
    with _PA_REFRESH_LOCK:
        if hasattr(sd, "_terminate") and hasattr(sd, "_initialize"):
            try:
                sd._terminate()
                sd._initialize()
                return True
            except Exception as exc:
                print(f"[AudioRouter] refresh_portaudio_devices error: {exc}")
    return False


def resolve_default_output_device_index(skip_cable: bool = False, _allow_refresh: bool = True) -> Optional[int]:
    """
    Визначає MME-індекс актуального пристрою виводу Windows за замовчуванням.
    Якщо skip_cable=True і Windows перемкнула за замовчуванням на VB-Cable
    чи віртуальний підсилювач, перевіряє також Communications-пристрій,
    за потреби оновлює кеш PortAudio (після підключення навушників) або
    обирає реальні навушники/динаміки.
    """
    global _LAST_CORE_AUDIO_SNAPSHOT

    core_defaults = _get_core_audio_default_output_names()
    snap = _core_audio_snapshot(core_defaults)
    if _LAST_CORE_AUDIO_SNAPSHOT is not None and snap != _LAST_CORE_AUDIO_SNAPSHOT:
        # Windows змінила системний пристрій виводу — закриваємо старі потоки перед вибором нового
        router = globals().get("audio_router")
        if router is not None:
            router.stop_local()
    _LAST_CORE_AUDIO_SNAPSHOT = snap

    winmm_name = _get_windows_preferred_output_name()
    sd_name = ""
    try:
        def_dev = sd.query_devices(None, "output")
        sd_name = str(def_dev.get("name", "")).strip()
    except Exception:
        sd_name = ""

    if skip_cable:
        candidates = (
            core_defaults.get("console", ""),
            core_defaults.get("multimedia", ""),
            winmm_name,
            sd_name,
            core_defaults.get("communications", ""),
        )
        saw_unmatched_real = False
        for cand in candidates:
            if not cand or _is_cable_device_name(cand) or _is_virtual_enhancer_name(cand):
                continue
            idx = resolve_output_device_index(cand)
            if idx is not None:
                return idx
            saw_unmatched_real = True
        if saw_unmatched_real and _allow_refresh and refresh_portaudio_devices():
            return resolve_default_output_device_index(skip_cable=skip_cable, _allow_refresh=False)
        non_cable = resolve_non_cable_speaker_indices()
        if non_cable:
            return non_cable[0]
        return None

    pref_name = core_defaults.get("console", "") or winmm_name or sd_name
    if pref_name:
        idx = resolve_output_device_index(pref_name)
        if idx is not None:
            return idx
        if _allow_refresh and refresh_portaudio_devices():
            return resolve_default_output_device_index(skip_cable=False, _allow_refresh=False)

    mapper_idx = _get_sound_mapper_output_index()
    if mapper_idx is not None:
        return mapper_idx

    try:
        hostapis = sd.query_hostapis()
        for h in hostapis:
            if h.get("name") == "MME":
                def_idx = int(h.get("default_output_device", -1))
                if def_idx >= 0:
                    return def_idx
    except Exception:
        pass
    return None


def resolve_secondary_output_targets(_allow_refresh: bool = True) -> List[Optional[int]]:
    """Повертає список індексів пристроїв для Каналу 2 (моніторингу) з урахуванням пристрою за замовчуванням."""
    secondary_name = str(config.get("secondary_output_device", "")).strip()
    if secondary_name:
        sec_idx = resolve_output_device_index(secondary_name)
        if sec_idx is not None:
            return [sec_idx]
        if _allow_refresh and refresh_portaudio_devices():
            sec_idx = resolve_output_device_index(secondary_name)
            if sec_idx is not None:
                return [sec_idx]
            return [resolve_default_output_device_index(skip_cable=True, _allow_refresh=False)]
    target = resolve_default_output_device_index(skip_cable=True, _allow_refresh=_allow_refresh)
    return [target]


def resolve_playback_device_indices(
    primary_name: str,
    primary_off: bool,
) -> Tuple[Optional[int], List[Optional[int]]]:
    """
    Узгоджено визначає індекси (primary_idx, secondary_targets) з ОДНОГО й того самого знімка PortAudio:
    1) Спочатку синхронізує PortAudio з живим списком Windows Core Audio (якщо підключили/відключили навушники).
    2) Резолвить secondary_targets (це теж може оновити PortAudio при появі нових навушників).
    3) Лише ПІСЛЯ можливого оновлення PortAudio резолвить primary_idx, щоб Канал 1 ніколи не
       зсунувся на чужий індекс (наприклад, на колонки замість VB-Cable).
    4) Якщо primary_name задано явно (наприклад, VB-Cable), але пристрій не знайдено — НЕ
       підміняє його системними колонками за замовчуванням.
    """
    ensure_portaudio_synced_with_windows()
    secondary_targets = resolve_secondary_output_targets(_allow_refresh=True)

    primary_idx: Optional[int] = None
    if not primary_off:
        target_primary = primary_name or "CABLE Input (VB-Audio Virtual Cable)"
        primary_idx = resolve_output_device_index(target_primary)
        if primary_idx is None and refresh_portaudio_devices():
            primary_idx = resolve_output_device_index(target_primary)
            # Повторно резолвимо secondary_targets із того самого оновленого знімка PortAudio
            secondary_targets = resolve_secondary_output_targets(_allow_refresh=False)

    return primary_idx, secondary_targets


def resolve_non_cable_speaker_indices() -> List[int]:
    """
    Повертає список індексів реальних навушників/динаміків (не VB-Cable).
    Потрібно тоді, коли після встановлення VB-Cable Windows автоматично перемкнула
    системний пристрій за замовчуванням на CABLE In 16ch / CABLE Input, через що
    користувач перестає чути звук у своїх реальних колонках/навушниках.
    """
    scored: List[Tuple[int, int, int]] = []
    for pos, dev in enumerate(get_output_devices()):
        low = str(dev["name"]).lower()
        if _is_cable_device_name(low):
            continue
        # Пріоритет:
        # 0: реальні навушники/гарнітура
        # 1: реальні динаміки (Realtek / вбудовані / USB)
        # 2: монітор / HDMI / цифрові
        # 3: сторонні віртуальні аудіопрограми (FxSound, Voicemeeter, Sonar тощо)
        if _is_virtual_enhancer_name(low):
            rank = 3
        elif any(k in low for k in ("headphone", "headset", "earphone", "навуш", "гарнітур", "наушник", "bluetooth", "airpods")):
            rank = 0
        elif any(k in low for k in ("nvidia", "amd", "hdmi", "displayport", "monitor", "digital", "spdif")):
            rank = 2
        else:
            rank = 1
        scored.append((rank, pos, int(dev["index"])))  # type: ignore[arg-type]
    scored.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in scored]


def check_vbcable_installed() -> Tuple[bool, Optional[str]]:
    """Перевіряє, чи встановлено віртуальний мікрофон VB-Audio Virtual Cable (пріоритет CABLE Input)."""
    devices = get_output_devices()
    for dev in devices:
        name = str(dev["name"])
        if "cable input" in name.lower():
            return True, name
    for dev in devices:
        name = str(dev["name"])
        low = name.lower()
        if "cable" in low or "vb-audio" in low:
            return True, name
    return False, None


def download_and_install_vbcable(progress_cb: Optional[Callable[[int, str], None]] = None) -> Tuple[bool, str]:
    """
    Завантажує офіційний пакет VB-Cable та запускає інсталятор з правами адміністратора.
    """
    vbcable_dir = DATA_DIR / "vbcable"
    vbcable_dir.mkdir(parents=True, exist_ok=True)
    zip_path = vbcable_dir / "VBCABLE_Driver_Pack43.zip"
    url = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack43.zip"

    try:
        if progress_cb:
            progress_cb(10, tr("vbcable.downloading", "Завантаження офіційного драйвера VB-Cable..."))

        if not zip_path.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(zip_path, "wb") as out_f:
                total = int(resp.headers.get("Content-Length", 0) or 1_200_000)
                downloaded = 0
                while True:
                    chunk = resp.read(32768)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb and total > 0:
                        pct = min(75, 10 + int(downloaded * 65 / total))
                        progress_cb(pct, tr("vbcable.downloading_kb", "Завантаження: {kb} КБ...", kb=downloaded // 1024))

        if progress_cb:
            progress_cb(80, tr("vbcable.extracting", "Розпакування архіву драйвера..."))

        with zipfile.ZipFile(zip_path, "r") as zf:
            root = vbcable_dir.resolve()
            for member in zf.infolist():
                dest = (vbcable_dir / member.filename).resolve()
                if dest != root and root not in dest.parents:
                    raise RuntimeError("Invalid file path inside VB-Cable archive.")
            zf.extractall(vbcable_dir)

        installer_exe = vbcable_dir / "VBCABLE_Setup_x64.exe"
        if not installer_exe.exists():
            installer_exe = vbcable_dir / "VBCABLE_Setup.exe"

        if not installer_exe.exists():
            return False, tr("vbcable.exe_missing", "Не знайдено VBCABLE_Setup_x64.exe в архіві.")

        if progress_cb:
            progress_cb(95, tr("vbcable.uac_prompt", "Запуск інсталятора VB-Cable (підтвердіть UAC)..."))

        res = shell32.ShellExecuteW(
            None,
            "runas",
            str(installer_exe),
            "",
            str(vbcable_dir),
            1,
        )
        if int(res) <= 32:
            return False, tr("vbcable.uac_cancelled", "Встановлення скасовано користувачем або заблоковано UAC.")

        if progress_cb:
            progress_cb(100, tr("vbcable.launched_progress", "Інсталятор VB-Cable запущено! Натисніть 'Install Driver' у вікні, що відкрилося."))
        return True, tr("vbcable.launched_ok", "Інсталятор VB-Cable запущено. Після натискання 'Install Driver' список пристроїв оновиться автоматично.")
    except Exception as exc:
        return False, tr("vbcable.error", "Помилка завантаження/встановлення VB-Cable: {err}", err=exc)


def _apply_stream_fade_in(samples: np.ndarray, sr: int, fade_in_ms: float = 12.0) -> np.ndarray:
    """М'який Hann fade-in на початку першого чанка стриму без обрізання хвоста."""
    if samples is None or len(samples) == 0:
        return np.zeros(0, dtype=np.float32)
    x = np.asarray(samples, dtype=np.float32).copy()
    n_in = min(int(sr * fade_in_ms / 1000.0), len(x) // 4)
    if n_in > 0:
        hann_in = (0.5 * (1.0 - np.cos(np.pi * np.arange(n_in, dtype=np.float32) / float(n_in)))).astype(np.float32)
        if x.ndim == 2:
            x[:n_in] *= hann_in[:, None]
        else:
            x[:n_in] *= hann_in
    return x


class _DeviceTrack:
    __slots__ = ("audio", "done_event", "cursor", "stopping")

    def __init__(self, audio: np.ndarray, done_event: threading.Event) -> None:
        self.audio = audio
        self.done_event = done_event
        self.cursor = 0
        self.stopping = False


class _StreamingDeviceTrack:
    __slots__ = ("chunks", "done_event", "cursor", "played_frames", "eof", "stopping")

    def __init__(self, done_event: threading.Event) -> None:
        self.chunks: deque[np.ndarray] = deque()
        self.done_event = done_event
        self.cursor = 0
        self.played_frames = 0
        self.eof = False
        self.stopping = False


class _LocalTrackHandle:
    """Керування відтворенням одного треку на одному локальному аудіопристрої."""

    def __init__(self, player: "_PersistentDevicePlayer", track: Union[_DeviceTrack, _StreamingDeviceTrack]) -> None:
        self._player = player
        self._track = track

    def stop(self) -> None:
        self._player.stop_track(self._track)

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._track.done_event.wait(timeout=timeout)

    def is_done(self) -> bool:
        return self._track.done_event.is_set()

    def progress_seconds(self) -> float:
        with self._player._lock:
            sr = max(1, int(self._player.samplerate or 48000))
            if isinstance(self._track, _StreamingDeviceTrack):
                return float(self._track.played_frames) / float(sr)
            return float(self._track.cursor) / float(sr)

    def mark_done(self) -> None:
        self._track.done_event.set()


class _StreamingTrackHandle(_LocalTrackHandle):
    """Потоковий трек, у який можна дописувати PCM-чанки на льоту."""

    def __init__(self, player: "_PersistentDevicePlayer", track: _StreamingDeviceTrack) -> None:
        super().__init__(player, track)
        self._stream_track = track

    def append(self, samples: np.ndarray, sr: int) -> None:
        self._player.append_stream(self._stream_track, samples, sr)

    def finish(self) -> None:
        self._player.finish_stream(self._stream_track)


class _HotSwapContext:
    """Стан поточного відтворення для перемикання динаміків/навушників на ходу."""

    __slots__ = (
        "sr",
        "prim_audio",
        "prim_handle",
        "prim_dev_idx",
        "allow_prim_fallback",
        "sec_audio",
        "sec_handles",
        "sec_dev_indices",
        "monitor_only",
        "primary_name",
        "recovery_attempts",
    )

    def __init__(
        self,
        *,
        sr: int,
        prim_audio: Optional[np.ndarray],
        prim_handle: Optional[_LocalTrackHandle],
        prim_dev_idx: Optional[int],
        allow_prim_fallback: bool,
        sec_audio: Optional[np.ndarray],
        sec_handles: List[_LocalTrackHandle],
        sec_dev_indices: List[Optional[int]],
        monitor_only: bool,
        primary_name: str,
    ) -> None:
        self.sr = sr
        self.prim_audio = prim_audio
        self.prim_handle = prim_handle
        self.prim_dev_idx = prim_dev_idx
        self.allow_prim_fallback = allow_prim_fallback
        self.sec_audio = sec_audio
        self.sec_handles = sec_handles
        self.sec_dev_indices = sec_dev_indices
        self.monitor_only = monitor_only
        self.primary_name = primary_name
        self.recovery_attempts = 0


class _PersistentDevicePlayer:
    """
    Постійний callback-потік для одного аудіопристрою:
    - Не відкриває й не закриває драйвер на кожному реченні (без пауз і клацань).
    - Працює у стерео (якщо пристрій підтримує >= 2 канали) або моно.
    - Автоматично переходить на int32, якщо пристрій відхиляє float32.
    - Закривається сам після idle_timeout_secs тиші.
    """

    def __init__(self, device_idx: Optional[int], preferred_sr: int, output_format: Optional[str] = None) -> None:
        self.device_idx = device_idx
        self.requested_format = output_format or get_configured_audio_output_format()
        self._lock = threading.Lock()
        self._tracks: List[_DeviceTrack] = []
        self._last_active_at = time.monotonic()
        self._last_callback_at = time.monotonic()
        self._broken = False
        self._closed = False
        self._stream: Optional[sd.OutputStream] = None
        self.samplerate = 48000
        self.channels = 2
        self.dtype = "float32"
        self._open(preferred_sr)

    def _open(self, preferred_sr: int) -> None:
        candidate_srs: List[int] = []
        max_out_ch = 2
        dev_name = ""
        try:
            dev_info = sd.query_devices(self.device_idx, "output")
            dev_name = str(dev_info.get("name", "") or "").strip()
            native_sr = int(dev_info.get("default_samplerate", 48000) or 48000)
            if native_sr > 0:
                candidate_srs.append(native_sr)
            max_out_ch = max(1, int(dev_info.get("max_output_channels", 2) or 2))
        except Exception:
            pass
        for cand_sr in (48000, 44100, int(preferred_sr)):
            if cand_sr > 0 and cand_sr not in candidate_srs:
                candidate_srs.append(cand_sr)

        channel_candidates = (2, 1) if max_out_ch >= 2 else (1,)
        dtype_candidates = ("int32", "float32") if self.requested_format == "int32" else ("float32", "int32")
        last_exc: Optional[Exception] = None
        for ch in channel_candidates:
            for target_sr in candidate_srs:
                for dt in dtype_candidates:
                    try:
                        self.samplerate = target_sr
                        self.channels = ch
                        self.dtype = dt
                        stream = sd.OutputStream(
                            samplerate=target_sr,
                            channels=ch,
                            dtype=dt,
                            device=self.device_idx,
                            callback=self._callback,
                        )
                        stream.start()
                        self._stream = stream
                        now = time.monotonic()
                        self._last_active_at = now
                        self._last_callback_at = now
                        _log_audio(
                            f"[AudioRouter] Відкрито потік: idx={self.device_idx} ({dev_name or 'default'}) "
                            f"sr={target_sr} ch={ch} dtype={dt} fmt={self.requested_format}"
                        )
                        return
                    except Exception as exc:
                        last_exc = exc
        _log_audio(
            f"[AudioRouter] Не вдалося відкрити потік idx={self.device_idx} ({dev_name or 'default'}): {last_exc}"
        )
        raise RuntimeError(f"Cannot open audio device {self.device_idx}: {last_exc}")

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        self._last_callback_at = time.monotonic()
        mix = np.zeros((frames, self.channels), dtype=np.float32)
        completed: List[threading.Event] = []
        with self._lock:
            if self._tracks:
                self._last_active_at = time.monotonic()
            remaining_tracks: List[Union[_DeviceTrack, _StreamingDeviceTrack]] = []
            for trk in self._tracks:
                if isinstance(trk, _StreamingDeviceTrack):
                    if trk.stopping:
                        if trk.chunks:
                            head = trk.chunks[0]
                            rem = len(head) - trk.cursor
                            fade_n = min(frames, max(0, rem), 64)
                            if fade_n > 0:
                                chunk = head[trk.cursor : trk.cursor + fade_n]
                                ramp = np.linspace(1.0, 0.0, fade_n, endpoint=False, dtype=np.float32)[:, None]
                                mix[:fade_n] += chunk * ramp
                        trk.chunks.clear()
                        completed.append(trk.done_event)
                        continue
                    written = 0
                    while written < frames and trk.chunks:
                        head = trk.chunks[0]
                        rem = len(head) - trk.cursor
                        if rem <= 0:
                            trk.chunks.popleft()
                            trk.cursor = 0
                            continue
                        take = min(frames - written, rem)
                        mix[written : written + take] += head[trk.cursor : trk.cursor + take]
                        trk.cursor += take
                        trk.played_frames += take
                        written += take
                        if trk.cursor >= len(head):
                            trk.chunks.popleft()
                            trk.cursor = 0
                    if not trk.chunks and trk.eof:
                        completed.append(trk.done_event)
                    else:
                        remaining_tracks.append(trk)
                    continue

                rem = len(trk.audio) - trk.cursor
                if rem <= 0:
                    completed.append(trk.done_event)
                    continue
                if trk.stopping:
                    fade_n = min(frames, rem, 64)
                    if fade_n > 0:
                        chunk = trk.audio[trk.cursor : trk.cursor + fade_n]
                        ramp = np.linspace(1.0, 0.0, fade_n, endpoint=False, dtype=np.float32)[:, None]
                        mix[:fade_n] += chunk * ramp
                    completed.append(trk.done_event)
                    continue
                take = min(frames, rem)
                mix[:take] += trk.audio[trk.cursor : trk.cursor + take]
                trk.cursor += take
                if trk.cursor >= len(trk.audio):
                    completed.append(trk.done_event)
                else:
                    remaining_tracks.append(trk)
            self._tracks = remaining_tracks
        for evt in completed:
            evt.set()
        np.clip(mix, -1.0, 1.0, out=mix)
        if self.dtype == "int32":
            outdata[:] = np.rint(mix * 2147483647.0).astype(np.int32)
        else:
            outdata[:] = mix

    def is_alive(self) -> bool:
        with self._lock:
            if self._closed or self._broken or self._stream is None:
                return False
            stream = self._stream
        try:
            return bool(stream.active)
        except Exception:
            return False

    def is_stalled(self, stall_secs: float = 0.45) -> bool:
        """Повертає True, якщо є незавершений трек, але аудіопотік/колбек зупинився (пристрій від'єднано)."""
        now = time.monotonic()
        with self._lock:
            if self._closed or self._broken or self._stream is None:
                return True
            has_pending = any(not trk.done_event.is_set() for trk in self._tracks)
            if not has_pending:
                return False
            last_cb = self._last_callback_at
            stream = self._stream
        try:
            if not bool(stream.active):
                return True
        except Exception:
            return True
        return (now - last_cb) > stall_secs

    def is_idle_expired(self, idle_secs: float, now: float) -> bool:
        with self._lock:
            if self._closed:
                return True
            if self._tracks:
                return False
            return (now - self._last_active_at) >= idle_secs

    def _prepare_audio(self, samples: np.ndarray, sr: int) -> np.ndarray:
        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        elif arr.ndim == 2 and arr.shape[1] > 2:
            arr = np.mean(arr, axis=1, dtype=np.float32)[:, None]

        if arr.shape[1] == 1 and self.channels == 2:
            arr = np.repeat(arr, 2, axis=1)
        elif arr.shape[1] == 2 and self.channels == 1:
            arr = np.mean(arr, axis=1, dtype=np.float32)[:, None]

        if int(sr) != self.samplerate and len(arr) > 0:
            import math
            from scipy.signal import resample_poly

            g = math.gcd(int(sr), self.samplerate)
            arr = resample_poly(arr, self.samplerate // g, int(sr) // g, axis=0).astype(np.float32, copy=False)
        return np.ascontiguousarray(arr, dtype=np.float32)

    def enqueue(self, samples: np.ndarray, sr: int) -> _LocalTrackHandle:
        prepared = self._prepare_audio(samples, sr)
        done_event = threading.Event()
        trk = _DeviceTrack(prepared, done_event)
        with self._lock:
            if self._closed or self._broken or self._stream is None:
                raise RuntimeError("Device stream is not active")
            self._tracks.append(trk)
            self._last_active_at = time.monotonic()
        return _LocalTrackHandle(self, trk)

    def enqueue_stream(self) -> _StreamingTrackHandle:
        done_event = threading.Event()
        trk = _StreamingDeviceTrack(done_event)
        with self._lock:
            if self._closed or self._broken or self._stream is None:
                raise RuntimeError("Device stream is not active")
            self._tracks.append(trk)
            self._last_active_at = time.monotonic()
        return _StreamingTrackHandle(self, trk)

    def append_stream(self, track: _StreamingDeviceTrack, samples: np.ndarray, sr: int) -> None:
        if samples is None or len(samples) == 0:
            return
        prepared = self._prepare_audio(samples, sr)
        if len(prepared) == 0:
            return
        with self._lock:
            if self._closed or self._broken or self._stream is None or track.stopping or track.eof:
                return
            track.chunks.append(prepared)
            self._last_active_at = time.monotonic()

    def finish_stream(self, track: _StreamingDeviceTrack) -> None:
        mark_done = False
        with self._lock:
            track.eof = True
            if not track.chunks and (self._closed or self._broken or self._stream is None or track not in self._tracks):
                mark_done = True
        if mark_done:
            track.done_event.set()

    def stop_track(self, track: Union[_DeviceTrack, _StreamingDeviceTrack]) -> None:
        with self._lock:
            track.stopping = True
            if isinstance(track, _StreamingDeviceTrack):
                track.eof = True
            self._last_active_at = time.monotonic()
        track.done_event.set()

    @property
    def output_latency_ms(self) -> float:
        try:
            if self._stream is not None:
                lat = float(getattr(self._stream, "latency", 0.0) or 0.0)
                return max(0.0, lat * 1000.0)
        except Exception:
            pass
        return 0.0

    def close(self, abort: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            tracks = self._tracks
            self._tracks = []
            stream = self._stream
            self._stream = None
        for trk in tracks:
            trk.done_event.set()
        if stream is not None:
            # На Windows MME stream.stop() може надовго зависати після витягування навушників,
            # тому спершу негайно скидаємо буфери через stream.abort().
            try:
                if abort and hasattr(stream, "abort"):
                    stream.abort()
                else:
                    stream.stop()
            except Exception:
                try:
                    stream.stop()
                except Exception:
                    pass
            try:
                stream.close()
            except Exception:
                pass


class DualAudioRouter:
    """
    Відтворює аудіо одночасно у два канали:
    1. Основний пристрій (Віртуальний мікрофон VB-Cable -> у гру / Discord)
    2. Додатковий пристрій (Навушники / Динаміки -> собі для контролю)
    """

    _IDLE_STREAM_TIMEOUT = 15.0

    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._play_lock = threading.Lock()
        self._play_lock_owner: Optional[int] = None
        self._is_playing = False
        self._mic_lock = threading.Lock()
        self._mic_stream: Optional[sd.Stream] = None
        self._mic_active_key: Optional[Tuple[int, int]] = None
        self._players_lock = threading.Lock()
        self._players: Dict[Optional[int], _PersistentDevicePlayer] = {}
        self._idle_timer: Optional[threading.Timer] = None
        # «Покоління» ексклюзивних відтворень (прослуховування зразків):
        # кожен новий запит робить усі попередні (і ті, що ще чекають у черзі на lock) застарілими.
        self._gen_lock = threading.Lock()
        self._generation = 0
        self._active_generation: Optional[int] = None

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    def stop(self) -> None:
        """Миттєво зупиняє поточне відтворення на всіх пристроях."""
        self._stop_event.set()

    def begin_exclusive_playback(self) -> int:
        """
        Зупиняє поточне відтворення та скасовує всі раніше поставлені в чергу ексклюзивні відтворення.
        Повертає токен, який треба передати в play_samples/play_wav_file(generation=...).
        """
        with self._gen_lock:
            self._generation += 1
            gen = self._generation
        self._stop_event.set()
        return gen

    @property
    def current_generation(self) -> int:
        return self._generation

    def _should_stop(self) -> bool:
        if self._stop_event.is_set():
            return True
        active = self._active_generation
        return active is not None and active != self._generation

    def _schedule_idle_reap(self) -> None:
        with self._players_lock:
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None
            if not self._players:
                return
            timer = threading.Timer(self._IDLE_STREAM_TIMEOUT, self._reap_idle_players)
            timer.daemon = True
            self._idle_timer = timer
            timer.start()

    def _reap_idle_players(self) -> None:
        now = time.monotonic()
        to_close: List[_PersistentDevicePlayer] = []
        has_more = False
        with self._players_lock:
            self._idle_timer = None
            for dev_idx in list(self._players.keys()):
                player = self._players[dev_idx]
                if not player.is_alive() or player.is_idle_expired(self._IDLE_STREAM_TIMEOUT, now):
                    to_close.append(self._players.pop(dev_idx))
            has_more = bool(self._players)
        for player in to_close:
            player.close()
        if has_more:
            self._schedule_idle_reap()

    def _enqueue_on_device(
        self,
        samples: np.ndarray,
        sr: int,
        device_idx: Optional[int],
    ) -> Optional[_LocalTrackHandle]:
        if len(samples) == 0 or self._should_stop():
            return None
        want_fmt = get_configured_audio_output_format()
        with self._players_lock:
            player = self._players.get(device_idx)
            if player is not None and (not player.is_alive() or player.requested_format != want_fmt):
                self._players.pop(device_idx, None)
                player.close()
                player = None
            if player is None:
                try:
                    player = _PersistentDevicePlayer(device_idx, sr, output_format=want_fmt)
                    self._players[device_idx] = player
                except Exception:
                    return None
        try:
            handle = player.enqueue(samples, sr)
            self._schedule_idle_reap()
            return handle
        except Exception:
            with self._players_lock:
                if self._players.get(device_idx) is player:
                    self._players.pop(device_idx, None)
            player.close()
            try:
                player = _PersistentDevicePlayer(device_idx, sr)
                with self._players_lock:
                    self._players[device_idx] = player
                handle = player.enqueue(samples, sr)
                self._schedule_idle_reap()
                return handle
            except Exception:
                player.close()
                return None

    def _start_on_single_device(
        self,
        samples: np.ndarray,
        sr: int,
        device_idx: Optional[int],
        allow_speaker_fallback: bool = True,
    ) -> Optional[_LocalTrackHandle]:
        if len(samples) == 0 or self._should_stop():
            return None

        handle = self._enqueue_on_device(samples, sr, device_idx)
        if handle is not None:
            return handle

        # Якщо обрано віртуальний кабель / явний Канал 1 — НЕ перекидаємо його у фізичні динаміки,
        # щоб не дублювати звук в навушниках і колонках ноутбука одночасно.
        if not allow_speaker_fallback:
            print(f"[AudioRouter] Канал 1 недоступний (index={device_idx}) — резервний вивід у динаміки вимкнено.")
            return None

        # Якщо стандартний пристрій моніторингу зайнятий/недоступний — пробуємо резервні динаміки/навушники
        for alt_idx in resolve_non_cable_speaker_indices():
            if alt_idx != device_idx:
                handle = self._enqueue_on_device(samples, sr, alt_idx)
                if handle is not None:
                    return handle

        print(f"[AudioRouter] Не вдалося відкрити аудіопристрій (index={device_idx})")
        return None

    def _close_player_for_device(self, device_idx: Optional[int], abort: bool = True) -> None:
        with self._players_lock:
            player = self._players.pop(device_idx, None)
        if player is not None:
            player.close(abort=abort)

    def _enqueue_secondary_remaining(
        self,
        hot_swap: _HotSwapContext,
        targets: List[Optional[int]],
        rem_sec: np.ndarray,
        local_handles: List[_LocalTrackHandle],
    ) -> None:
        new_sec_handles: List[_LocalTrackHandle] = []
        new_sec_devs: List[Optional[int]] = []
        active_prim_dev = (
            hot_swap.prim_dev_idx
            if (hot_swap.prim_handle is not None and not hot_swap.prim_handle.is_done() and not hot_swap.monitor_only)
            else None
        )
        seen_devs = set(new_sec_devs)
        for t_idx in targets:
            if t_idx is not None and t_idx == active_prim_dev:
                continue
            h = self._start_on_single_device(rem_sec, hot_swap.sr, t_idx)
            if h is not None:
                opened_idx = h._player.device_idx
                if opened_idx in seen_devs:
                    h.stop()
                    h.mark_done()
                    continue
                seen_devs.add(opened_idx)
                new_sec_handles.append(h)
                new_sec_devs.append(opened_idx)
                local_handles.append(h)
        hot_swap.sec_handles = new_sec_handles
        hot_swap.sec_dev_indices = new_sec_devs

    def _handle_live_output_change(
        self,
        hot_swap: _HotSwapContext,
        local_handles: List[_LocalTrackHandle],
        core_defaults: Dict[str, str],
        *,
        snap_changed: bool,
        sec_stalled: bool,
        prim_stalled: bool,
    ) -> None:
        global _LAST_CORE_AUDIO_SNAPSHOT

        if prim_stalled and hot_swap.prim_handle is not None and not snap_changed:
            hot_swap.prim_handle.mark_done()
            self._close_player_for_device(hot_swap.prim_dev_idx, abort=True)
            if hot_swap.prim_handle in local_handles:
                local_handles.remove(hot_swap.prim_handle)
            hot_swap.prim_handle = None

        if hot_swap.sec_audio is None or (not snap_changed and not sec_stalled):
            return
        if not any(not h.is_done() for h in hot_swap.sec_handles):
            return

        if sec_stalled:
            hot_swap.recovery_attempts += 1
            if hot_swap.recovery_attempts > 3:
                for old_h in list(hot_swap.sec_handles):
                    old_h.stop()
                    old_h.mark_done()
                    if old_h in local_handles:
                        local_handles.remove(old_h)
                for old_idx in list(hot_swap.sec_dev_indices):
                    if old_idx != hot_swap.prim_dev_idx or prim_stalled:
                        self._close_player_for_device(old_idx, abort=True)
                hot_swap.sec_handles = []
                hot_swap.sec_dev_indices = []
                return

        if hot_swap.prim_handle is not None and not prim_stalled and not hot_swap.prim_handle.is_done():
            sec_offset_sec = hot_swap.prim_handle.progress_seconds()
        else:
            sec_offset_sec = max((h.progress_seconds() for h in hot_swap.sec_handles), default=0.0)

        secondary_name = str(config.get("secondary_output_device", "")).strip()
        if secondary_name:
            need_pa_refresh = resolve_output_device_index(secondary_name) is None
        else:
            need_pa_refresh = _has_unregistered_real_default_output(core_defaults)

        _LAST_CORE_AUDIO_SNAPSHOT = _core_audio_snapshot(core_defaults)

        if not need_pa_refresh:
            candidate_targets = resolve_secondary_output_targets(_allow_refresh=False)
            if not sec_stalled and candidate_targets == hot_swap.sec_dev_indices:
                return

            rem_sec = _slice_audio_from_seconds(hot_swap.sec_audio, hot_swap.sr, sec_offset_sec)
            for old_h in list(hot_swap.sec_handles):
                old_h.stop()
                old_h.mark_done()
                if old_h in local_handles:
                    local_handles.remove(old_h)
            for old_idx in list(hot_swap.sec_dev_indices):
                if old_idx != hot_swap.prim_dev_idx or prim_stalled:
                    self._close_player_for_device(old_idx, abort=True)
            hot_swap.sec_handles = []
            hot_swap.sec_dev_indices = []

            if len(rem_sec) > 0:
                self._enqueue_secondary_remaining(hot_swap, candidate_targets, rem_sec, local_handles)
            return

        # Портативні/нові USB чи Bluetooth навушники ще не в кеші PortAudio:
        # перезавантажуємо PortAudio і підхоплюємо відтворення з точного місця.
        prim_offset_sec: Optional[float] = None
        if hot_swap.prim_handle is not None and hot_swap.prim_audio is not None and not prim_stalled and not hot_swap.prim_handle.is_done():
            prim_offset_sec = hot_swap.prim_handle.progress_seconds()

        for old_h in list(local_handles):
            old_h.stop()
            old_h.mark_done()
        local_handles.clear()
        hot_swap.sec_handles = []
        hot_swap.sec_dev_indices = []
        hot_swap.prim_handle = None

        self.refresh_devices()

        if hot_swap.prim_audio is not None and prim_offset_sec is not None:
            rem_prim = _slice_audio_from_seconds(hot_swap.prim_audio, hot_swap.sr, prim_offset_sec)
            if len(rem_prim) > 0:
                new_prim_idx: Optional[int] = None
                if hot_swap.primary_name:
                    new_prim_idx = resolve_output_device_index(hot_swap.primary_name)
                else:
                    new_prim_idx = resolve_default_output_device_index(skip_cable=False, _allow_refresh=False)
                if new_prim_idx is not None or not hot_swap.primary_name:
                    new_prim_h = self._start_on_single_device(
                        rem_prim,
                        hot_swap.sr,
                        new_prim_idx,
                        hot_swap.allow_prim_fallback,
                    )
                    if new_prim_h is not None:
                        hot_swap.prim_handle = new_prim_h
                        hot_swap.prim_dev_idx = new_prim_h._player.device_idx
                        local_handles.append(new_prim_h)

        rem_sec = _slice_audio_from_seconds(hot_swap.sec_audio, hot_swap.sr, sec_offset_sec)
        if len(rem_sec) > 0:
            candidate_targets = resolve_secondary_output_targets(_allow_refresh=False)
            self._enqueue_secondary_remaining(hot_swap, candidate_targets, rem_sec, local_handles)

    def _wait_handles(
        self,
        local_handles: List[_LocalTrackHandle],
        vban_handle: Optional[VbanPlayback] = None,
        hot_swap: Optional[_HotSwapContext] = None,
    ) -> None:
        stopped_sent = False
        last_dev_poll = time.monotonic()
        last_snap = _core_audio_snapshot() if hot_swap is not None else None

        while any(not h.is_done() for h in local_handles) or (
            vban_handle is not None and not vban_handle.done_event.is_set()
        ):
            if self._should_stop():
                if not stopped_sent:
                    stopped_sent = True
                    for h in local_handles:
                        h.stop()
                    if vban_handle is not None:
                        vban_handle.stop()
                if vban_handle is not None:
                    vban_handle.wait(timeout=0.2)
                for h in local_handles:
                    h.wait(timeout=0.1)
                break

            now = time.monotonic()
            stalled_handles = [h for h in local_handles if not h.is_done() and h._player.is_stalled(0.45)]
            if hot_swap is None:
                if stalled_handles:
                    for h in stalled_handles:
                        h.mark_done()
                        self._close_player_for_device(h._player.device_idx, abort=True)
                time.sleep(0.015)
                continue

            if stalled_handles or (now - last_dev_poll >= 0.10):
                last_dev_poll = now
                core_defaults = _get_core_audio_default_output_names()
                snap = _core_audio_snapshot(core_defaults)
                snap_changed = last_snap is not None and snap != last_snap
                last_snap = snap

                sec_stalled = any(h in stalled_handles for h in hot_swap.sec_handles)
                prim_stalled = hot_swap.prim_handle is not None and hot_swap.prim_handle in stalled_handles
                if snap_changed or sec_stalled or prim_stalled:
                    self._handle_live_output_change(
                        hot_swap,
                        local_handles,
                        core_defaults,
                        snap_changed=snap_changed,
                        sec_stalled=sec_stalled,
                        prim_stalled=prim_stalled,
                    )

            time.sleep(0.015)

    def _notify_playback_queued(
        self,
        local_handles: List[_LocalTrackHandle],
        on_playback_queued: Optional[Callable[[float], None]],
    ) -> None:
        if on_playback_queued is None or self._should_stop():
            return
        lat_ms = max(
            (h._player.output_latency_ms for h in local_handles),
            default=85.0,
        )
        if lat_ms <= 0.0:
            lat_ms = 85.0
        on_playback_queued(lat_ms)

    def play_samples(
        self,
        raw_samples: np.ndarray,
        sr: int,
        monitor_only: bool = False,
        generation: Optional[int] = None,
        *,
        send_primary: bool = True,
        vban_stream_override: Optional[str] = None,
        extra_gain_pct: int = 100,
        on_playback_queued: Optional[Callable[[float], None]] = None,
    ) -> None:
        """
        Синхронно (у робочому потоці) відтворює підготовлений масив аудіо
        одночасно у віртуальний мікрофон та навушники.
        Якщо monitor_only=True (наприклад, прослуховування прикладів голосів у меню),
        звук іде в навушники (або основний вивід, якщо моніторинг вимкнено).
        generation — токен з begin_exclusive_playback(): якщо поки запит чекав своєї черги
        вже прийшов новіший, цей запит просто пропускається (без нескінченних повторів).
        """
        if raw_samples is None or len(raw_samples) == 0:
            return
        if generation is not None and generation != self._generation:
            return

        with self._play_lock:
            if generation is not None and generation != self._generation:
                return
            self._play_lock_owner = threading.get_ident()
            self._stop_event.clear()
            self._active_generation = generation
            self._is_playing = True
            try:
                cleaned = clean_audio_edges(raw_samples, sr)
                gain_pct = max(0, min(200, int(extra_gain_pct)))
                if gain_pct != 100:
                    cleaned = apply_volume_and_limiter(cleaned, gain_pct)

                primary_name = str(config.get("primary_output_device", "")).strip()
                primary_off = primary_name == "__off__"
                primary_vol = int(config.get("primary_volume", 100))
                secondary_enabled = bool(config.get("secondary_enabled", True))
                secondary_vol = int(config.get("secondary_volume", 85))

                primary_idx, secondary_targets = resolve_playback_device_indices(primary_name, primary_off)

                if monitor_only:
                    _log_audio(
                        f"[AudioRouter] Маршрут [preview]: secondary={[ _describe_device_for_log(i) for i in secondary_targets ]}"
                    )
                    mon_samples = apply_volume_and_limiter(cleaned, secondary_vol if secondary_vol > 0 else 90)
                    mon_handles: List[_LocalTrackHandle] = []
                    for t_idx in secondary_targets:
                        h = self._start_on_single_device(mon_samples, sr, t_idx)
                        if h is not None:
                            mon_handles.append(h)
                    self._notify_playback_queued(mon_handles, on_playback_queued)
                    hot_swap = _HotSwapContext(
                        sr=sr,
                        prim_audio=None,
                        prim_handle=None,
                        prim_dev_idx=None,
                        allow_prim_fallback=False,
                        sec_audio=mon_samples,
                        sec_handles=list(mon_handles),
                        sec_dev_indices=[h._player.device_idx for h in mon_handles],
                        monitor_only=True,
                        primary_name=primary_name,
                    )
                    self._wait_handles(mon_handles, hot_swap=hot_swap)
                    return

                # Mirror speech to a VBAN receiver without changing local outputs.
                # vban_stream_override == "" disables VBAN for this utterance; None uses default stream name.
                vban_handle: Optional[VbanPlayback] = None
                if bool(config.get("vban_enabled", False)) and vban_stream_override != "":
                    vban_vol = max(0, min(200, int(config.get("vban_volume", 100))))
                    eff_stream = (
                        vban_stream_override.strip()
                        if vban_stream_override is not None
                        else str(config.get("vban_stream_name", "SilentVoice") or "SilentVoice")
                    )
                    vban_target = (
                        str(config.get("vban_address", "")).strip(),
                        int(config.get("vban_port", 6980) or 6980),
                        eff_stream or "SilentVoice",
                    )
                    if vban_target[0] and vban_vol > 0:
                        try:
                            vban_samples = apply_volume_and_limiter(cleaned, vban_vol)
                            vban_handle = start_vban_samples(
                                vban_samples,
                                sr,
                                *vban_target,
                                kind="speech",
                            )
                        except Exception as exc:
                            print(f"[VBAN] Не вдалося надіслати звук: {exc}")

                # Формуємо унікальний список задач (role, device_idx, samples_array, allow_speaker_fallback),
                # щоб ніколи не відкривати один і той самий пристрій двічі.
                device_jobs: List[Tuple[str, Optional[int], np.ndarray, bool]] = []
                used_indices: set = set()
                prim_samples: Optional[np.ndarray] = None
                sec_samples_for_swap: Optional[np.ndarray] = None
                allow_prim_fallback = False

                # 1. Основний вивід (Віртуальний мікрофон / обраний пристрій; якщо не знайдено — вимкнено)
                if send_primary and not primary_off and primary_vol > 0 and primary_idx is not None:
                    prim_samples = apply_volume_and_limiter(cleaned, primary_vol)
                    device_jobs.append(("primary", primary_idx, prim_samples, allow_prim_fallback))
                    used_indices.add(primary_idx)

                # 2. Додатковий вивід (Навушники / колонки для самоконтролю та OBS)
                eff_sec_vol = secondary_vol if secondary_vol > 0 else (90 if not send_primary else 0)
                if (secondary_enabled or not send_primary) and eff_sec_vol > 0:
                    sec_samples_for_swap = apply_volume_and_limiter(cleaned, eff_sec_vol)
                    for s_idx in secondary_targets:
                        if s_idx not in used_indices:
                            device_jobs.append(("secondary", s_idx, sec_samples_for_swap, True))
                            used_indices.add(s_idx)

                _log_audio(
                    f"[AudioRouter] Маршрут [tts]: primary={_describe_device_for_log(primary_idx) if (send_primary and not primary_off and primary_idx is not None) else 'off'}, "
                    f"secondary={[ _describe_device_for_log(i) for i in secondary_targets ] if (secondary_enabled or not send_primary) else 'off'}"
                )

                local_handles: List[_LocalTrackHandle] = []
                prim_handle: Optional[_LocalTrackHandle] = None
                sec_handles: List[_LocalTrackHandle] = []
                for role, dev_idx, job_samples, allow_fb in device_jobs:
                    h = self._start_on_single_device(job_samples, sr, dev_idx, allow_fb)
                    if h is not None:
                        local_handles.append(h)
                        if role == "primary":
                            prim_handle = h
                        else:
                            sec_handles.append(h)

                hot_swap = _HotSwapContext(
                    sr=sr,
                    prim_audio=prim_samples,
                    prim_handle=prim_handle,
                    prim_dev_idx=prim_handle._player.device_idx if prim_handle is not None else primary_idx,
                    allow_prim_fallback=allow_prim_fallback,
                    sec_audio=sec_samples_for_swap,
                    sec_handles=sec_handles,
                    sec_dev_indices=[h._player.device_idx for h in sec_handles],
                    monitor_only=False,
                    primary_name=primary_name,
                )

                self._notify_playback_queued(local_handles, on_playback_queued)
                self._wait_handles(local_handles, vban_handle, hot_swap=hot_swap)
                if vban_handle is not None and vban_handle.status == "failed" and vban_handle.error:
                    print(f"[VBAN] Не вдалося надіслати звук: {vban_handle.error}")
            finally:
                self._play_lock_owner = None
                self._is_playing = False
                self._active_generation = None

    def _enqueue_stream_on_device(
        self,
        device_idx: Optional[int],
        preferred_sr: int,
    ) -> Optional[_StreamingTrackHandle]:
        if self._should_stop():
            return None
        want_fmt = get_configured_audio_output_format()
        with self._players_lock:
            player = self._players.get(device_idx)
            if player is not None and (not player.is_alive() or player.requested_format != want_fmt):
                self._players.pop(device_idx, None)
                player.close()
                player = None
            if player is None:
                try:
                    player = _PersistentDevicePlayer(device_idx, preferred_sr, output_format=want_fmt)
                    self._players[device_idx] = player
                except Exception:
                    return None
        try:
            handle = player.enqueue_stream()
            self._schedule_idle_reap()
            return handle
        except Exception:
            with self._players_lock:
                if self._players.get(device_idx) is player:
                    self._players.pop(device_idx, None)
            player.close()
            try:
                player = _PersistentDevicePlayer(device_idx, preferred_sr)
                with self._players_lock:
                    self._players[device_idx] = player
                handle = player.enqueue_stream()
                self._schedule_idle_reap()
                return handle
            except Exception:
                player.close()
                return None

    def _start_stream_on_single_device(
        self,
        device_idx: Optional[int],
        preferred_sr: int,
        allow_speaker_fallback: bool = True,
    ) -> Optional[_StreamingTrackHandle]:
        if self._should_stop():
            return None
        handle = self._enqueue_stream_on_device(device_idx, preferred_sr)
        if handle is not None:
            return handle
        if not allow_speaker_fallback:
            print(f"[AudioRouter] Канал 1 недоступний (index={device_idx}) — резервний вивід у динаміки вимкнено.")
            return None
        for alt_idx in resolve_non_cable_speaker_indices():
            if alt_idx != device_idx:
                handle = self._enqueue_stream_on_device(alt_idx, preferred_sr)
                if handle is not None:
                    return handle
        print(f"[AudioRouter] Не вдалося відкрити аудіопристрій (index={device_idx})")
        return None

    def play_stream(
        self,
        chunk_iter: Iterable[Tuple[np.ndarray, int]],
        *,
        monitor_only: bool = False,
        generation: Optional[int] = None,
        send_primary: bool = True,
        vban_stream_override: Optional[str] = None,
        extra_gain_pct: int = 100,
        on_first_chunk: Optional[Callable[[np.ndarray, int, float], None]] = None,
        on_generation_done: Optional[Callable[[np.ndarray, int], None]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Потокове відтворення аудіо по мірі надходження PCM-чанків (наприклад, зі стримінгу Qwen3-TTS).
        Починає грати одразу після приходу першого чанка, без очікування кінця генерації.
        Повертає повний склеєний масив (samples, sr) для кешування.
        """
        if generation is not None and generation != self._generation:
            return np.zeros(0, dtype=np.float32), 24000

        it = iter(chunk_iter)
        first_samples: Optional[np.ndarray] = None
        stream_sr = 24000
        try:
            for raw_chunk, chunk_sr in it:
                if raw_chunk is not None and len(raw_chunk) > 0:
                    first_samples = np.asarray(raw_chunk, dtype=np.float32)
                    stream_sr = max(1, int(chunk_sr))
                    break
        except Exception:
            close_fn = getattr(it, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
            raise

        if first_samples is None or len(first_samples) == 0:
            close_fn = getattr(it, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
            return np.zeros(0, dtype=np.float32), stream_sr

        if generation is not None and generation != self._generation:
            close_fn = getattr(it, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
            return np.zeros(0, dtype=np.float32), stream_sr

        gain_pct = max(0, min(200, int(extra_gain_pct)))
        collected_chunks: List[np.ndarray] = [first_samples]
        producer_error: List[Optional[BaseException]] = [None]
        producer_done = threading.Event()
        producer_stop = threading.Event()

        with self._play_lock:
            if generation is not None and generation != self._generation:
                close_fn = getattr(it, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception:
                        pass
                return np.zeros(0, dtype=np.float32), stream_sr
            self._play_lock_owner = threading.get_ident()
            self._stop_event.clear()
            self._active_generation = generation
            self._is_playing = True
            local_handles: List[_StreamingTrackHandle] = []
            # Each target entry: (handle, vol_pct)
            stream_targets: List[Tuple[_StreamingTrackHandle, int]] = []
            vban_q: Optional["queue.Queue[Optional[Tuple[np.ndarray, int]]]"] = None
            vban_done = threading.Event()
            vban_cur_lock = threading.Lock()
            vban_cur_holder: List[Optional[VbanPlayback]] = [None]
            vban_vol = 0
            vban_thread: Optional[threading.Thread] = None

            try:
                primary_name = str(config.get("primary_output_device", "")).strip()
                primary_off = primary_name == "__off__"
                primary_vol = int(config.get("primary_volume", 100))
                secondary_enabled = bool(config.get("secondary_enabled", True))
                secondary_vol = int(config.get("secondary_volume", 85))

                primary_idx, secondary_targets = resolve_playback_device_indices(primary_name, primary_off)
                _log_audio(
                    f"[AudioRouter] Маршрут [stream/{'preview' if monitor_only else 'tts'}]: "
                    f"primary={_describe_device_for_log(primary_idx) if (not monitor_only and send_primary and not primary_off and primary_idx is not None) else 'off'}, "
                    f"secondary={[ _describe_device_for_log(i) for i in secondary_targets ] if (monitor_only or secondary_enabled or not send_primary) else 'off'}"
                )

                if monitor_only:
                    mon_vol = secondary_vol if secondary_vol > 0 else 90
                    for t_idx in secondary_targets:
                        h = self._start_stream_on_single_device(t_idx, stream_sr, allow_speaker_fallback=True)
                        if h is not None:
                            local_handles.append(h)
                            stream_targets.append((h, mon_vol))
                else:
                    if bool(config.get("vban_enabled", False)) and vban_stream_override != "":
                        vban_vol = max(0, min(200, int(config.get("vban_volume", 100))))
                        eff_stream = (
                            vban_stream_override.strip()
                            if vban_stream_override is not None
                            else str(config.get("vban_stream_name", "SilentVoice") or "SilentVoice")
                        )
                        vban_target = (
                            str(config.get("vban_address", "")).strip(),
                            int(config.get("vban_port", 6980) or 6980),
                            eff_stream or "SilentVoice",
                        )
                        if vban_target[0] and vban_vol > 0:
                            vban_q = queue.Queue()

                            def _vban_stream_worker() -> None:
                                try:
                                    while not producer_stop.is_set() and not self._should_stop():
                                        try:
                                            pkt = vban_q.get(timeout=0.05)  # type: ignore[union-attr]
                                        except queue.Empty:
                                            continue
                                        if pkt is None:
                                            break
                                        vsamp, vsr = pkt
                                        try:
                                            vh = start_vban_samples(vsamp, vsr, *vban_target, kind="speech")
                                        except Exception as exc:
                                            print(f"[VBAN] Не вдалося надіслати звук: {exc}")
                                            continue
                                        with vban_cur_lock:
                                            vban_cur_holder[0] = vh
                                        while not vh.done_event.wait(timeout=0.015):
                                            if producer_stop.is_set() or self._should_stop():
                                                vh.stop()
                                                vh.wait(timeout=0.2)
                                                break
                                        with vban_cur_lock:
                                            vban_cur_holder[0] = None
                                finally:
                                    vban_done.set()

                            vban_thread = threading.Thread(target=_vban_stream_worker, daemon=True, name="VbanStreamBridge")
                            vban_thread.start()
                        else:
                            vban_done.set()
                    else:
                        vban_done.set()

                    used_indices: set = set()
                    allow_prim_fallback = False
                    if send_primary and not primary_off and primary_vol > 0 and primary_idx is not None:
                        h = self._start_stream_on_single_device(primary_idx, stream_sr, allow_prim_fallback)
                        if h is not None:
                            local_handles.append(h)
                            stream_targets.append((h, primary_vol))
                            used_indices.add(primary_idx)
                            used_indices.add(h._player.device_idx)

                    eff_sec_vol = secondary_vol if secondary_vol > 0 else (90 if not send_primary else 0)
                    if (secondary_enabled or not send_primary) and eff_sec_vol > 0:
                        for s_idx in secondary_targets:
                            if s_idx not in used_indices:
                                h = self._start_stream_on_single_device(s_idx, stream_sr, True)
                                if h is not None:
                                    local_handles.append(h)
                                    stream_targets.append((h, eff_sec_vol))
                                    used_indices.add(s_idx)
                                    used_indices.add(h._player.device_idx)

                if monitor_only:
                    vban_done.set()

                def _push_chunk(chunk_arr: np.ndarray, chunk_sr: int, *, is_first: bool) -> None:
                    arr = _apply_stream_fade_in(chunk_arr, chunk_sr) if is_first else np.asarray(chunk_arr, dtype=np.float32)
                    if gain_pct != 100:
                        arr = apply_volume_and_limiter(arr, gain_pct)
                    for h, vol_pct in stream_targets:
                        h.append(apply_volume_and_limiter(arr, vol_pct), chunk_sr)
                    if vban_q is not None and vban_vol > 0:
                        vban_q.put((apply_volume_and_limiter(arr, vban_vol), chunk_sr))

                _push_chunk(first_samples, stream_sr, is_first=True)
                self._schedule_idle_reap()

                if on_first_chunk is not None and not self._should_stop():
                    lat_ms = max((h._player.output_latency_ms for h in local_handles), default=85.0)
                    if lat_ms <= 0.0:
                        lat_ms = 85.0
                    on_first_chunk(first_samples, stream_sr, lat_ms)

                def _producer() -> None:
                    try:
                        for raw_chunk, chunk_sr in it:
                            if producer_stop.is_set() or self._should_stop():
                                break
                            if raw_chunk is None or len(raw_chunk) == 0:
                                continue
                            arr = np.asarray(raw_chunk, dtype=np.float32)
                            csr = max(1, int(chunk_sr))
                            collected_chunks.append(arr)
                            _push_chunk(arr, csr, is_first=False)
                        if not producer_stop.is_set() and not self._should_stop():
                            tail_n = int(stream_sr * 0.055)
                            if tail_n > 0:
                                tail = np.zeros(tail_n, dtype=np.float32)
                                for h, _vol in stream_targets:
                                    h.append(tail, stream_sr)
                            full = (
                                collected_chunks[0]
                                if len(collected_chunks) == 1
                                else np.concatenate(collected_chunks, axis=0)
                            )
                            if on_generation_done is not None:
                                try:
                                    on_generation_done(full, stream_sr)
                                except Exception:
                                    pass
                    except BaseException as exc:
                        producer_error[0] = exc
                    finally:
                        close_fn = getattr(it, "close", None)
                        if callable(close_fn):
                            try:
                                close_fn()
                            except Exception:
                                pass
                        for h in local_handles:
                            h.finish()
                        if vban_q is not None:
                            vban_q.put(None)
                        producer_done.set()

                prod_thread = threading.Thread(target=_producer, daemon=True, name="AudioStreamProducer")
                prod_thread.start()

                stopped_sent = False
                while (
                    not producer_done.is_set()
                    or any(not h.is_done() for h in local_handles)
                    or not vban_done.is_set()
                ):
                    if self._should_stop():
                        if not stopped_sent:
                            stopped_sent = True
                            producer_stop.set()
                            for h in local_handles:
                                h.stop()
                            if vban_q is not None:
                                vban_q.put(None)
                            with vban_cur_lock:
                                cur_v = vban_cur_holder[0]
                            if cur_v is not None:
                                cur_v.stop()
                        for h in local_handles:
                            h.wait(timeout=0.1)
                        vban_done.wait(timeout=0.2)
                        break

                    stalled = [h for h in local_handles if not h.is_done() and h._player.is_stalled(0.45)]
                    if stalled:
                        for h in stalled:
                            h.mark_done()
                            self._close_player_for_device(h._player.device_idx, abort=True)

                    time.sleep(0.015)

                prod_thread.join(timeout=0.5)
                if vban_thread is not None:
                    vban_thread.join(timeout=0.3)

                if producer_error[0] is not None and not self._should_stop():
                    raise producer_error[0]

                full_audio = (
                    collected_chunks[0]
                    if len(collected_chunks) == 1
                    else np.concatenate(collected_chunks, axis=0)
                )
                return full_audio, stream_sr
            finally:
                producer_stop.set()
                for h in local_handles:
                    h.finish()
                self._play_lock_owner = None
                self._is_playing = False
                self._active_generation = None

    def play_wav_file(self, wav_path: str | Path, monitor_only: bool = False, generation: Optional[int] = None) -> float:
        """Зчитує .wav файл та відтворює його (зі збереженням стерео). Повертає тривалість у секундах."""
        p = Path(wav_path)
        if not p.exists():
            return 0.0
        if generation is not None and generation != self._generation:
            return 0.0
        data, sr = sf.read(str(p), dtype="float32")
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] > 2:
            arr = np.mean(arr, axis=1).astype(np.float32)
        dur = len(arr) / float(sr) if sr > 0 else 0.0
        self.play_samples(arr, int(sr), monitor_only=monitor_only, generation=generation)
        return dur

    def play_feedback_beep(self, kind: str = "start") -> None:
        """
        Відтворює дуже м'який короткий звуковий сигнал ТІЛЬКИ в навушники (моніторинг),
        коли режим набору тексту увімкнувся або був скасований.
        """
        if kind not in ("start", "cancel"):
            return
        if self._is_playing:
            return
        # Start and cancel cues are independent toggles sharing a single volume level.
        if kind == "start" and not config.get("sound_feedback", True):
            return
        if kind == "cancel" and not config.get("cancel_sound_enabled", True):
            return
        vol_pct = max(0, min(200, int(config.get("sound_feedback_volume", 100))))
        if vol_pct <= 0:
            return
        vol_scale = float(vol_pct) / 100.0

        def _beep_worker():
            if not self._play_lock.acquire(blocking=False):
                return
            self._play_lock_owner = threading.get_ident()
            try:
                self._stop_event.clear()
                sr = 44100
                if kind == "start":
                    dur = 0.065
                    freq1, freq2 = 520.0, 740.0
                    base_amp = 0.115
                else:
                    # Легкий м'який спадний тон виходу / скасування в стилі сигналу старту
                    dur = 0.072
                    freq1, freq2 = 680.0, 450.0
                    base_amp = 0.105
                amp = base_amp * vol_scale

                n = int(sr * dur)
                freq = np.linspace(freq1, freq2, n, dtype=np.float32)
                phase = 2.0 * np.pi * np.cumsum(freq) / float(sr)
                wave = amp * np.sin(phase)
                # Плавне вікно Ганна без клацань на краях
                env = np.sin(np.pi * np.arange(n, dtype=np.float32) / float(max(1, n))) ** 2
                wave = (wave * env).astype(np.float32)

                sec_targets = resolve_secondary_output_targets()
                sec_idx = sec_targets[0] if sec_targets else None
                h = self._start_on_single_device(wave, sr, sec_idx, allow_speaker_fallback=True)
                if h is not None:
                    self._wait_handles([h])
            except Exception:
                pass
            finally:
                self._play_lock_owner = None
                self._play_lock.release()

        threading.Thread(target=_beep_worker, daemon=True).start()

    def stop_local(self, abort: bool = True) -> None:
        """Закриває всі відкриті локальні аудіопотоки."""
        with self._players_lock:
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None
            players = list(self._players.values())
            self._players.clear()
        for player in players:
            player.close(abort=abort)

    def refresh_devices(self) -> bool:
        """
        Закриває активні потоки й оновлює список пристроїв PortAudio,
        щоб підхопити щойно підключені навушники чи мікрофони без перезапуску.
        """
        tid = threading.get_ident()
        already_holds = self._play_lock_owner == tid
        if not already_holds:
            if not self._play_lock.acquire(blocking=False):
                return False
            self._play_lock_owner = tid
        refreshed = False
        try:
            with _PA_REFRESH_LOCK:
                self.stop_local()
                self.stop_mic_passthrough()
                if hasattr(sd, "_terminate") and hasattr(sd, "_initialize"):
                    try:
                        sd._terminate()
                        sd._initialize()
                        refreshed = True
                    except Exception as exc:
                        print(f"[AudioRouter] refresh_devices error: {exc}")
        finally:
            if not already_holds:
                self._play_lock_owner = None
                self._play_lock.release()
        if refreshed:
            self.sync_mic_passthrough()
        return refreshed

    def stop_vban(self) -> None:
        """Зупиняє фоновий VBAN-потік та закриває UDP-сокет при виході."""
        shutdown_vban()

    def shutdown(self) -> None:
        """Повна зупинка всіх аудіоканалів (локальні потоки, мікрофон, VBAN)."""
        self.stop()
        self.stop_mic_passthrough()
        self.stop_local()
        self.stop_vban()

    def stop_mic_passthrough(self) -> None:
        """Зупиняє фонове мікшування фізичного мікрофона у Канал 1."""
        with self._mic_lock:
            if self._mic_stream is not None:
                try:
                    self._mic_stream.stop()
                    self._mic_stream.close()
                except Exception:
                    pass
                self._mic_stream = None
            self._mic_active_key = None

    def sync_mic_passthrough(self) -> None:
        """Запускає або зупиняє передачу фізичного мікрофона у Канал 1 згідно з налаштуваннями."""
        enabled = bool(config.get("mic_passthrough_enabled", False))
        primary_name = str(config.get("primary_output_device", "")).strip()
        if not enabled or not primary_name or primary_name == "__off__":
            self.stop_mic_passthrough()
            return

        primary_idx = resolve_output_device_index(primary_name)
        mic_name = str(config.get("mic_passthrough_device", "")).strip()
        mic_idx = resolve_input_device_index(mic_name)
        if primary_idx is None or mic_idx is None:
            self.stop_mic_passthrough()
            return

        new_key = (mic_idx, primary_idx)
        with self._mic_lock:
            if self._mic_stream is not None and self._mic_active_key == new_key:
                return
            if self._mic_stream is not None:
                try:
                    self._mic_stream.stop()
                    self._mic_stream.close()
                except Exception:
                    pass
                self._mic_stream = None
                self._mic_active_key = None

            def _mic_callback(indata: np.ndarray, outdata: np.ndarray, frames: int, time_info, status) -> None:
                vol = max(0.0, float(config.get("mic_passthrough_volume", 100)) / 100.0)
                if vol <= 0.0:
                    outdata.fill(0.0)
                else:
                    np.multiply(indata, vol, out=outdata)
                    np.clip(outdata, -1.0, 1.0, out=outdata)

            for sr in (48000, 44100):
                try:
                    stream = sd.Stream(
                        device=(mic_idx, primary_idx),
                        samplerate=sr,
                        channels=1,
                        dtype="float32",
                        blocksize=1024,
                        latency="low",
                        callback=_mic_callback,
                    )
                    stream.start()
                    self._mic_stream = stream
                    self._mic_active_key = new_key
                    return
                except Exception:
                    continue
            print(f"[AudioRouter] Не вдалося запустити мікшування мікрофона ({mic_idx} -> {primary_idx})")


audio_router = DualAudioRouter()
