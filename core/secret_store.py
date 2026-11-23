"""
core/secret_store.py — Захист локальних секретів через Windows DPAPI (CryptProtectData).

Усі чутливі поля конфігурації (API-ключі TTS/AI, Twitch OAuth-токен, паролі проксі,
Telegram API Hash та Telethon-сесія) зберігаються на диску в зашифрованому вигляді
з прив'язкою до поточного облікового запису Windows (`dpapi:<base64>`).
У пам'яті додатку дані доступні в розшифрованому вигляді, а старі відкриті значення
автоматично мігруються на DPAPI при завантаженні.
"""

from __future__ import annotations

import base64
import copy
import ctypes
from ctypes import wintypes
import os
from typing import Any, Dict, Tuple

DPAPI_PREFIX = "dpapi:"
_CRYPTPROTECT_UI_FORBIDDEN = 0x01

# Ключі верхнього рівня в config.json, які містять секрети або можуть містити логін/пароль
TOP_LEVEL_SECRET_KEYS = frozenset({
    "twitch_token",
    "custom_proxy_url",
    "socks5_proxy_url",
})

# Імена вкладених полів у словниках (online_items, ai_items тощо)
NESTED_SECRET_KEYS = frozenset({
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "authorization",
    "tg_api_hash",
    "tg_phone",
    "proxy_url",
})


class _DATA_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def is_dpapi_available() -> bool:
    """Повертає True, якщо поточна ОС підтримує Windows DPAPI (crypt32.CryptProtectData)."""
    if os.name != "nt":
        return False
    try:
        return hasattr(ctypes.windll, "crypt32") and hasattr(ctypes.windll.crypt32, "CryptProtectData")
    except Exception:
        return False


def protect_bytes(raw: bytes) -> bytes:
    """Шифрує байти через Windows DPAPI (поточний користувач Windows)."""
    if not raw or not is_dpapi_available():
        return raw
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(_DATA_BLOB),
        wintypes.LPCWSTR,
        ctypes.POINTER(_DATA_BLOB),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DATA_BLOB),
    ]
    crypt32.CryptProtectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p

    buf = ctypes.create_string_buffer(raw, len(raw))
    in_blob = _DATA_BLOB(len(raw), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    out_blob = _DATA_BLOB()
    ok = crypt32.CryptProtectData(
        ctypes.byref(in_blob),
        "SilentVoiceSecret",
        None,
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(out_blob),
    )
    if not ok:
        raise ctypes.WinError()
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        if out_blob.pbData:
            kernel32.LocalFree(out_blob.pbData)


def unprotect_bytes(blob: bytes) -> bytes:
    """Розшифровує байти, зашифровані Windows DPAPI."""
    if not blob or not is_dpapi_available():
        return blob
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DATA_BLOB),
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(_DATA_BLOB),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DATA_BLOB),
    ]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p

    buf = ctypes.create_string_buffer(blob, len(blob))
    in_blob = _DATA_BLOB(len(blob), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    out_blob = _DATA_BLOB()
    ok = crypt32.CryptUnprotectData(
        ctypes.byref(in_blob),
        None,
        None,
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(out_blob),
    )
    if not ok:
        raise ctypes.WinError()
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        if out_blob.pbData:
            kernel32.LocalFree(out_blob.pbData)


def protect_secret(plaintext: str) -> str:
    """Повертає рядок у форматі 'dpapi:<base64>' на Windows (або оригінал, якщо порожньо / не Windows)."""
    text = str(plaintext or "")
    if not text or text.startswith(DPAPI_PREFIX):
        return text
    if not is_dpapi_available():
        return text
    try:
        encrypted = protect_bytes(text.encode("utf-8"))
        return DPAPI_PREFIX + base64.b64encode(encrypted).decode("ascii")
    except Exception as exc:
        print(f"[SecretStore] DPAPI protect failed: {exc}")
        return text


def unprotect_secret(value: str) -> str:
    """Розшифровує рядок 'dpapi:<base64>'. Відкриті (старі) рядки повертає як є."""
    text = str(value or "")
    if not text or not text.startswith(DPAPI_PREFIX):
        return text
    if not is_dpapi_available():
        return ""
    try:
        raw_blob = base64.b64decode(text[len(DPAPI_PREFIX):].encode("ascii"))
        return unprotect_bytes(raw_blob).decode("utf-8")
    except Exception as exc:
        print(f"[SecretStore] DPAPI unprotect failed (можливо, файл перенесено з іншого ПК/користувача): {exc}")
        return ""


def _should_protect_proxy(url_val: str) -> bool:
    """Проксі-URL шифруємо завжди, якщо він не порожній (може містити user:pass@host)."""
    return bool(str(url_val or "").strip())


def _walk_protect(obj: Any, *, is_root: bool = False) -> Any:
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            key_low = str(k).strip().lower()
            if isinstance(v, str) and v:
                if is_root and key_low in TOP_LEVEL_SECRET_KEYS:
                    if key_low in ("custom_proxy_url", "socks5_proxy_url"):
                        out[k] = protect_secret(v) if _should_protect_proxy(v) else v
                    else:
                        out[k] = protect_secret(v)
                    continue
                if not is_root and key_low in NESTED_SECRET_KEYS:
                    out[k] = protect_secret(v)
                    continue
            out[k] = _walk_protect(v, is_root=False)
        return out
    if isinstance(obj, list):
        return [_walk_protect(item, is_root=False) for item in obj]
    return copy.deepcopy(obj)


def _walk_unprotect(obj: Any, *, is_root: bool = False) -> Tuple[Any, bool]:
    """Повертає (розшифрований_об'єкт, знайдено_незашифровані_секрети)."""
    needs_migration = False
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            key_low = str(k).strip().lower()
            if isinstance(v, str) and v:
                is_secret_field = (
                    (is_root and key_low in TOP_LEVEL_SECRET_KEYS)
                    or (not is_root and key_low in NESTED_SECRET_KEYS)
                )
                if is_secret_field:
                    if v.startswith(DPAPI_PREFIX):
                        out[k] = unprotect_secret(v)
                    else:
                        if is_dpapi_available():
                            needs_migration = True
                        out[k] = v
                    continue
            child, child_mig = _walk_unprotect(v, is_root=False)
            if child_mig:
                needs_migration = True
            out[k] = child
        return out, needs_migration
    if isinstance(obj, list):
        res_list = []
        for item in obj:
            child, child_mig = _walk_unprotect(item, is_root=False)
            if child_mig:
                needs_migration = True
            res_list.append(child)
        return res_list, needs_migration
    return copy.deepcopy(obj), False


def protect_config_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Повертає копію конфігурації із зашифрованими через DPAPI секретами для запису на диск."""
    return _walk_protect(data, is_root=True)


def unprotect_config_dict(data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Розшифровує секрети після читання config.json і сигналізує, чи потрібна міграція на DPAPI."""
    return _walk_unprotect(data, is_root=True)
