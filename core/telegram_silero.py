"""
core/telegram_silero.py — Клієнт Telegram (Telethon MTProto) для взаємодії з ботом @silero_voice_bot.
Реалізує підключення за номером телефону + API ID / API Hash (2FA підтримується),
керування голосом (/speaker), отримання лімітів (/limits) та синтез мовлення з суворою
кореляцією відповідей (аналогічно архітектурі TTSBard).
"""

from __future__ import annotations

import asyncio
import datetime
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import zlib
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import unquote, urlparse

from core.config import DATA_DIR, config
from core.i18n import tr
from core.proxy_helper import format_socks5_url, get_active_socks5_url
from core.secret_store import DPAPI_PREFIX, is_dpapi_available, protect_secret, unprotect_secret

BOT_USERNAME = "silero_voice_bot"
SESSION_PATH_BASE = DATA_DIR / "telegram_silero"
SESSION_FILE = SESSION_PATH_BASE.with_suffix(".session")
SESSION_DPAPI_FILE = DATA_DIR / "telegram_silero.session.dpapi"
APP_KEYS_FILE = DATA_DIR / "tg_app_keys.json"

_TG_MASK = (115, 196, 43, 143, 22, 229, 73, 162)
_TG_DATA = (
    11, 30, 46, 78, 23, 244, 73, 130, 123, 199, 235, 197, 5, 107, 4, 206,
    80, 78, 214, 172, 238, 10, 135, 50, 114, 47, 46, 168, 46, 158, 137, 203,
    148, 206, 174, 199, 160, 247, 251, 192, 156, 203, 236, 233, 28, 232,
)


def _get_bundled_tg_credentials() -> Tuple[int, str]:
    try:
        raw = bytes(b ^ _TG_MASK[i % len(_TG_MASK)] for i, b in enumerate(_TG_DATA))
        decoded = zlib.decompress(raw).decode("utf-8")
        aid_s, ahash = decoded.split(":", 1)
        if aid_s.isdigit() and ahash:
            return int(aid_s), ahash
    except Exception:
        pass
    return 0, ""


DEFAULT_TG_API_ID: int = 0
DEFAULT_TG_API_HASH: str = ""

MAX_DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_RETRY_DELAY_SEC = 1.0


class SynthesisCancelledError(Exception):
    """Синтез через Telegram скасовано користувачем (Зупинити голос)."""


def is_telethon_installed() -> bool:
    """Перевіряє, чи встановлена бібліотека telethon."""
    return importlib.util.find_spec("telethon") is not None


def is_qrcode_installed() -> bool:
    """Перевіряє, чи встановлена бібліотека qrcode."""
    return importlib.util.find_spec("qrcode") is not None


def ensure_qrcode_installed() -> bool:
    """Гарантує наявність чисто-Python бібліотеки qrcode (встановлює за потреби)."""
    if is_qrcode_installed():
        return True
    if getattr(sys, "frozen", False):
        return False
    extra_kw = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "qrcode"],
        check=True,
        capture_output=True,
        **extra_kw,
    )
    importlib.invalidate_caches()
    return is_qrcode_installed()


def parse_telegram_app_credentials(raw_text: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Розумний парсер: витягує api_id та 32-символьний hex api_hash з довільного фрагмента тексту,
    скопійованого зі сторінки my.telegram.org/apps.
    """
    txt = str(raw_text or "").strip()
    if not txt:
        return None, None

    found_hash: Optional[str] = None
    m_hash = re.search(r"\b([0-9a-fA-F]{32})\b", txt)
    if m_hash:
        found_hash = m_hash.group(1).lower()

    found_id: Optional[int] = None
    m_labeled_id = re.search(r"(?i)api[_\s-]*id\D{0,10}(\d{5,12})", txt)
    if m_labeled_id:
        found_id = int(m_labeled_id.group(1))
    else:
        for m_num in re.finditer(r"\b(\d{5,12})\b", txt):
            cand = m_num.group(1)
            if not found_hash or cand not in found_hash:
                found_id = int(cand)
                break

    return found_id, found_hash


def load_default_api_credentials() -> Tuple[int, str, str]:
    """
    Повертає стандартні (не прив'язані до конкретної картки) API-ключі додатку:
    (api_id, api_hash, source), де source — 'app_keys_file' | 'env' | 'bundled' | ''.
    """
    if APP_KEYS_FILE.exists():
        try:
            raw = json.loads(APP_KEYS_FILE.read_text(encoding="utf-8"))
            aid_s = str(raw.get("api_id", "") or "").strip()
            raw_hash = str(raw.get("api_hash", "") or "").strip()
            ahash = unprotect_secret(raw_hash) if raw_hash.startswith(DPAPI_PREFIX) else raw_hash
            if aid_s.isdigit() and int(aid_s) > 0 and ahash:
                if is_dpapi_available() and not raw_hash.startswith(DPAPI_PREFIX):
                    try:
                        save_default_api_credentials(int(aid_s), ahash)
                    except Exception:
                        pass
                return int(aid_s), ahash, "app_keys_file"
        except Exception:
            pass

    env_id = str(os.environ.get("SILENTVOICE_TG_API_ID", "") or "").strip()
    env_hash = str(os.environ.get("SILENTVOICE_TG_API_HASH", "") or "").strip()
    if env_id.isdigit() and int(env_id) > 0 and env_hash:
        return int(env_id), env_hash, "env"

    if int(DEFAULT_TG_API_ID or 0) > 0 and str(DEFAULT_TG_API_HASH or "").strip():
        return int(DEFAULT_TG_API_ID), str(DEFAULT_TG_API_HASH).strip(), "bundled"

    b_id, b_hash = _get_bundled_tg_credentials()
    if b_id > 0 and b_hash:
        return b_id, b_hash, "bundled"

    return 0, "", ""


def save_default_api_credentials(api_id: int, api_hash: str) -> None:
    """Локально зберігає API ID / API Hash (зашифрований через Windows DPAPI) у data/tg_app_keys.json."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    protected_hash = protect_secret(str(api_hash).strip())
    APP_KEYS_FILE.write_text(
        json.dumps({"api_id": int(api_id), "api_hash": protected_hash}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def clear_default_api_credentials() -> None:
    """Видаляє локально збережені стандартні API-ключі (data/tg_app_keys.json)."""
    if APP_KEYS_FILE.exists():
        try:
            APP_KEYS_FILE.unlink()
        except Exception:
            pass


def _parse_proxy_for_telethon(proxy_mode: str = "auto", custom_proxy_url: str = "") -> Optional[Tuple[Any, ...]]:
    """
    Повертає tuple проксі для Telethon: (socks.SOCKS5, host, port, True, user, password)
    або None, якщо проксі вимкнено.
    """
    mode = str(proxy_mode or "auto").strip().lower()
    if mode == "none":
        return None

    raw_url = ""
    if mode == "socks5":
        raw_url = format_socks5_url(custom_proxy_url or str(config.get("socks5_proxy_url", "")), remote_dns=True)
    else:
        # auto: використовуємо глобальне налаштування SOCKS5 додатку, якщо увімкнено
        raw_url = get_active_socks5_url()

    if not raw_url:
        return None

    try:
        import socks  # pysocks
    except ImportError:
        return None

    parsed = urlparse(raw_url if "://" in raw_url else f"socks5h://{raw_url}")
    host = parsed.hostname
    port = parsed.port
    if not host or not port:
        return None

    username = unquote(parsed.username) if parsed.username else None
    password = unquote(parsed.password) if parsed.password else None
    rdns = parsed.scheme in ("socks5h", "socks4a", "socks5", "")
    if username or password:
        return (socks.SOCKS5, host, int(port), rdns, username, password)
    return (socks.SOCKS5, host, int(port), rdns)


def classify_silero_rejection(text: str) -> str:
    """
    Класифікує текстову відмову від @silero_voice_bot:
    - 'limit_exceeded': вичерпано добовий/символьний ліміт
    - 'text_too_long': перевищено довжину одного запиту
    - 'generic': інша текстова відповідь-відмова
    """
    lower = (text or "").lower()

    if re.search(r"(?:лимит|ліміт).*озвучк.*(?:исчерпан|вичерп)", lower, re.DOTALL):
        return "limit_exceeded"
    if re.search(r"(?:превысит|превыша|превышен|перевищ).*(?:лимит|ліміт).*озвучк", lower, re.DOTALL):
        return "limit_exceeded"
    if re.search(r"(?:длина текста|довжина тексту).*(?:превыша|перевищ).*символ", lower, re.DOTALL):
        return "text_too_long"

    return "generic"


def parse_voice_info(text: str) -> Optional[Dict[str, str]]:
    """
    Парсить відповідь на команду /speaker:
    "Выбранный голос: /speaker hamster_clerk\\nНаходится в паке: Хомяки"
    """
    voice_id = ""
    pack_name = ""
    for raw in (text or "").splitlines():
        line = raw.strip()
        low = line.lower()
        if "выбранный голос:" in low or "выбраний голос:" in low or "обраний голос:" in low or "selected voice:" in low:
            parts = line.split()
            for i, part in enumerate(parts):
                if part == "/speaker" and i + 1 < len(parts):
                    voice_id = parts[i + 1].strip()
                    break
        if (
            "находится в паке:" in low
            or "знаходиться в паке:" in low
            or "located in pack:" in low
        ):
            if ":" in line:
                pack_name = line.split(":", 1)[1].strip()

    if voice_id:
        return {"id": voice_id, "name": pack_name or voice_id}
    return None


def parse_set_speaker_response(text: str) -> Optional[bool]:
    """
    Парсить відповідь бота на /speaker <code>.
    Повертає True при успіху, False при невірному коді голосу, або None якщо формат невідомий.
    """
    s = text or ""
    success_markers = (
        "Успешно выбран спикер",
        "Успішно обрано спікера",
        "Successfully selected speaker",
        "Успешно выбран тот же самый спикер",
        "Успішно обрано того самого спікера",
    )
    if any(m in s for m in success_markers):
        return True

    invalid_markers = (
        "Указан неверный голос",
        "Вказано невірний голос",
        "Invalid voice",
    )
    if any(m in s for m in invalid_markers):
        return False
    return None


def parse_speaker_selected_from_text(text: str) -> Optional[str]:
    """
    Витягує код голосу, якщо бот @silero_voice_bot надіслав/оновив повідомлення
    про вибір голосу (наприклад «Успешно выбран спикер: shorthair» або відповідь /speaker).
    """
    s = str(text or "").strip()
    if not s:
        return None
    m = re.search(
        r"(?:Успешно выбран спикер|Успішно обрано спікера|Successfully selected speaker)[:\s]+([A-Za-z0-9_.-]+)",
        s,
    )
    if m:
        return m.group(1).strip().lower()
    info = parse_voice_info(s)
    if info and info.get("id"):
        return str(info["id"]).strip().lower()
    return None


def _try_parse_slash_counter(line: str) -> Optional[str]:
    if "/" not in line:
        return None
    before, after = line.split("/", 1)
    before = before.strip()
    after_clean = after.strip()
    m_after = re.match(r"^(\d+)", after_clean)
    if not before.isdigit() or not m_after:
        return None
    return f"{before} / {m_after.group(1)}"


def _parse_reset_timestamp(text: str) -> Optional[str]:
    m_full = re.search(
        r"(?i)(?:обновится|оновлюється)\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+UTC[+-]\d{1,2})",
        text or "",
    )
    if m_full:
        return m_full.group(1)
    m_leg = re.search(
        r"(?i)(?:обновится|оновлюється)\s+(\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+UTC[+-]\d{1,2})",
        text or "",
    )
    if m_leg:
        return m_leg.group(1)
    return None


def parse_limits_info(text: str) -> Optional[Dict[str, str]]:
    """
    Парсить відповідь на команду /limits (підтримує однорядковий та багаторядковий формати).
    """
    voices: Optional[str] = None
    gifs: Optional[str] = None
    pending: Optional[str] = None

    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue

        if pending is not None:
            counter = _try_parse_slash_counter(line)
            if counter:
                if pending == "voices":
                    voices = counter
                elif pending == "gifs":
                    gifs = counter
                pending = None
                continue
            pending = None

        cleaned = line.strip("*").strip()
        if "Открытые голоса:" in cleaned or "Відкриті голоси:" in cleaned:
            after = cleaned.split(":", 1)[1].strip() if ":" in cleaned else ""
            counter = _try_parse_slash_counter(after) if after else None
            if counter:
                voices = counter
            else:
                pending = "voices"
        elif "Кружки/гифки:" in cleaned or "Кружки/гіфки:" in cleaned or "Гифки:" in cleaned:
            after = cleaned.split(":", 1)[1].strip() if ":" in cleaned else ""
            counter = _try_parse_slash_counter(after) if after else None
            if counter:
                gifs = counter
            else:
                pending = "gifs"
        elif cleaned.startswith("🎙") and any(
            x in cleaned for x in ("Переозвучка:", "Переозвучення:", "Только текст:", "Тільки текст:")
        ):
            pending = "other"
        else:
            pending = None

    if voices and gifs:
        res: Dict[str, str] = {"voices": voices, "gifs": gifs}
        reset_ts = _parse_reset_timestamp(text or "")
        if reset_ts:
            res["reset_timestamp"] = reset_ts
        return res
    return None


def _extract_reply_to_msg_id(msg: Any) -> Optional[int]:
    reply_to = getattr(msg, "reply_to", None)
    if reply_to is not None:
        rid = getattr(reply_to, "reply_to_msg_id", None)
        if rid is not None:
            return int(rid)
    rid_direct = getattr(msg, "reply_to_msg_id", None)
    return int(rid_direct) if rid_direct is not None else None


def _get_message_mime(msg: Any) -> str:
    media = getattr(msg, "media", None)
    doc = getattr(media, "document", None) if media is not None else None
    mime = str(getattr(doc, "mime_type", "") or "").lower()
    if mime:
        return mime
    file_obj = getattr(msg, "file", None)
    if file_obj is not None:
        return str(getattr(file_obj, "mime_type", "") or "").lower()
    return ""


class TelegramSileroManager:
    """
    Потокобезпечний менеджер Telegram-клієнта для роботи з @silero_voice_bot.
    Працює поверх виділеного фонового asyncio event loop.
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._init_lock = threading.Lock()

        # Усі поля нижче належать _loop потоку
        self._client: Any = None
        self._phone: str = ""
        self._api_id: int = 0
        self._api_hash: str = ""
        self._phone_code_hash: str = ""
        self._proxy_mode: str = "auto"
        self._proxy_url: str = ""

        self._bot_entity: Any = None
        self._bot_user_id: int = 0
        self._confirmed_speaker: Optional[str] = None
        self._user_info: Optional[Dict[str, Any]] = None
        self._connected: bool = False
        self._qr_login: Any = None
        self._qr_wait_task: Optional[asyncio.Task[Any]] = None
        self._active_synth_task: Optional[asyncio.Task[Any]] = None
        self._speaker_changed_cb: Optional[Callable[[str], None]] = None
        self._listener_client_id: int = 0

        self._speaker_lock: Optional[asyncio.Lock] = None
        self._limits_lock: Optional[asyncio.Lock] = None
        self._connect_lock: Optional[asyncio.Lock] = None

    def set_speaker_changed_callback(self, cb: Optional[Callable[[str], None]]) -> None:
        """Реєструє callback(voice_id) для автоматичного підхоплення голосу, обраного в @silero_voice_bot."""
        self._speaker_changed_cb = cb

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._init_lock:
            if self._loop is not None and self._loop.is_running():
                return self._loop
            ready = threading.Event()

            def _runner() -> None:
                lp = asyncio.new_event_loop()
                asyncio.set_event_loop(lp)
                self._loop = lp
                self._speaker_lock = asyncio.Lock()
                self._limits_lock = asyncio.Lock()
                self._connect_lock = asyncio.Lock()
                ready.set()
                lp.run_forever()

            t = threading.Thread(target=_runner, daemon=True, name="TelegramSileroLoop")
            self._loop_thread = t
            t.start()
            ready.wait(timeout=5.0)
            if self._loop is None:
                raise RuntimeError("Failed to start Telegram background event loop")
            return self._loop

    def _run_coro(self, coro: Any, timeout: float = 90.0) -> Any:
        loop = self._ensure_loop()
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        return fut.result(timeout=timeout)

    def _get_saved_credentials(self) -> Dict[str, Any]:
        for item in config.get("online_items", []):
            if str(item.get("provider", "")).strip() == "silero_bot":
                return {
                    "api_id": str(item.get("tg_api_id", "") or "").strip(),
                    "api_hash": str(item.get("tg_api_hash", "") or "").strip(),
                    "phone": str(item.get("tg_phone", "") or "").strip(),
                    "proxy_mode": str(item.get("proxy_mode", "auto") or "auto").strip().lower(),
                    "proxy_url": str(item.get("proxy_url", "") or "").strip(),
                    "timeout_sec": float(item.get("timeout_sec", 60) or 60),
                }
        return {}

    def _resolve_api_credentials(self, api_id: int = 0, api_hash: str = "") -> Tuple[int, str]:
        """Повертає (api_id, api_hash) з явних аргументів, конфігу картки або дефолтного джерела."""
        use_id = int(api_id or 0)
        use_hash = str(api_hash or "").strip()
        if use_id > 0 and use_hash:
            return use_id, use_hash

        creds = self._get_saved_credentials()
        item_id_s = str(creds.get("api_id", "") or "").strip()
        item_hash = str(creds.get("api_hash", "") or "").strip()
        if not use_id and item_id_s.isdigit():
            use_id = int(item_id_s)
        if not use_hash:
            use_hash = item_hash
        if use_id > 0 and use_hash:
            return use_id, use_hash

        def_id, def_hash, _src = load_default_api_credentials()
        if def_id > 0 and def_hash:
            return def_id, def_hash
        return 0, ""

    def has_effective_credentials(self, api_id: int = 0, api_hash: str = "") -> bool:
        use_id, use_hash = self._resolve_api_credentials(api_id, api_hash)
        return use_id > 0 and bool(use_hash)

    def get_cached_status(self) -> Dict[str, Any]:
        return {
            "installed": is_telethon_installed(),
            "connected": self._connected,
            "user": dict(self._user_info) if self._user_info else None,
            "confirmed_speaker": self._confirmed_speaker,
            "session_exists": self._has_saved_session(),
        }

    # ------------------------------------------------------------------
    # Публічні синхронні методи
    # ------------------------------------------------------------------

    def start_auth(
        self,
        phone: str,
        api_id: int = 0,
        api_hash: str = "",
        proxy_mode: str = "auto",
        proxy_url: str = "",
    ) -> Dict[str, Any]:
        use_id, use_hash = self._resolve_api_credentials(api_id, api_hash)
        if not use_id or not use_hash:
            raise RuntimeError(
                tr("tg.err.api_creds_missing", "Не вказано API ID та API Hash (отримайте на my.telegram.org/apps).")
            )
        return self._run_coro(
            self._async_start_auth(phone, use_id, use_hash, proxy_mode, proxy_url),
            timeout=45.0,
        )

    def start_qr_login(
        self,
        api_id: int = 0,
        api_hash: str = "",
        proxy_mode: str = "auto",
        proxy_url: str = "",
    ) -> Dict[str, Any]:
        use_id, use_hash = self._resolve_api_credentials(api_id, api_hash)
        if not use_id or not use_hash:
            raise RuntimeError(
                tr("tg.err.api_creds_missing", "Не вказано API ID та API Hash (отримайте на my.telegram.org/apps).")
            )
        return self._run_coro(
            self._async_start_qr_login(use_id, use_hash, proxy_mode, proxy_url),
            timeout=45.0,
        )

    def recreate_qr_login(self) -> Dict[str, Any]:
        return self._run_coro(self._async_recreate_qr_login(), timeout=30.0)

    def wait_qr_login(self, timeout_sec: float = 35.0) -> Dict[str, Any]:
        t = max(5.0, float(timeout_sec or 35.0))
        return self._run_coro(self._async_wait_qr_login(t), timeout=t + 15.0)

    def cancel_qr_login(self) -> None:
        loop = self._loop
        if loop is None or not loop.is_running():
            return

        def _cancel_on_loop() -> None:
            task = self._qr_wait_task
            if task is not None and not task.done():
                task.cancel()
            self._qr_login = None

        loop.call_soon_threadsafe(_cancel_on_loop)

    def submit_code(self, code: str) -> Dict[str, Any]:
        return self._run_coro(self._async_submit_code(code), timeout=45.0)

    def submit_password(self, password: str) -> Dict[str, Any]:
        return self._run_coro(self._async_submit_password(password), timeout=45.0)

    def connect_existing(
        self,
        api_id: int = 0,
        api_hash: str = "",
        proxy_mode: str = "",
        proxy_url: str = "",
    ) -> Dict[str, Any]:
        creds = self._get_saved_credentials()
        use_id, use_hash = self._resolve_api_credentials(api_id, api_hash)
        use_pmode = str(proxy_mode or creds.get("proxy_mode") or "auto").strip().lower()
        use_purl = str(proxy_url or creds.get("proxy_url") or "").strip()
        if not use_id or not use_hash:
            return {"state": "auth_required", "connected": False, "user": None}
        return self._run_coro(
            self._async_connect_existing(use_id, use_hash, use_pmode, use_purl),
            timeout=50.0,
        )

    def disconnect(self) -> None:
        if self._loop is None:
            return
        self._run_coro(self._async_disconnect(delete_session=False), timeout=15.0)

    def sign_out(self) -> None:
        if self._loop is not None:
            self._run_coro(self._async_disconnect(delete_session=True), timeout=20.0)
        else:
            self._remove_session_files()

    def get_current_voice(self) -> Optional[Dict[str, str]]:
        return self._run_coro(self._async_get_current_voice(), timeout=65.0)

    def set_speaker(self, voice_code: str, force: bool = False) -> Tuple[bool, str]:
        return self._run_coro(self._async_set_speaker(voice_code, force=force), timeout=45.0)

    def get_limits(self) -> Optional[Dict[str, str]]:
        return self._run_coro(self._async_get_limits(), timeout=65.0)

    def synthesize(self, text: str, voice_code: str = "", timeout_sec: float = 60.0) -> Tuple[bytes, str]:
        return self._run_coro(
            self._async_synthesize_cancellable(text, voice_code=voice_code, timeout_sec=timeout_sec),
            timeout=max(45.0, float(timeout_sec) + 35.0),
        )

    def cancel_active_synthesis(self) -> None:
        """Миттєво скасовує активний запит синтезу до @silero_voice_bot при натисканні «Зупинити»."""
        loop = self._loop
        if loop is None or not loop.is_running():
            return

        def _cancel() -> None:
            t = self._active_synth_task
            if t is not None and not t.done():
                t.cancel()

        loop.call_soon_threadsafe(_cancel)

    async def _async_synthesize_cancellable(
        self,
        text: str,
        voice_code: str = "",
        timeout_sec: float = 60.0,
    ) -> Tuple[bytes, str]:
        task = asyncio.create_task(self._async_synthesize(text, voice_code=voice_code, timeout_sec=timeout_sec))
        self._active_synth_task = task
        try:
            return await task
        except asyncio.CancelledError as exc:
            raise SynthesisCancelledError("Telegram synthesis cancelled") from exc
        finally:
            if self._active_synth_task is task:
                self._active_synth_task = None

    # ------------------------------------------------------------------
    # Асинхронна реалізація поверх Telethon
    # ------------------------------------------------------------------

    @staticmethod
    def _has_saved_session() -> bool:
        return SESSION_DPAPI_FILE.exists() or SESSION_FILE.exists()

    @staticmethod
    def _remove_legacy_sqlite_session_files() -> None:
        for suffix in (".session", ".session-journal", ".session-wal", ".session-shm"):
            p = SESSION_PATH_BASE.with_suffix(suffix)
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    @classmethod
    def _remove_session_files(cls) -> None:
        if SESSION_DPAPI_FILE.exists():
            try:
                SESSION_DPAPI_FILE.unlink()
            except Exception:
                pass
        cls._remove_legacy_sqlite_session_files()

    @classmethod
    def _save_protected_session_string(cls, session_str: str) -> None:
        clean = str(session_str or "").strip()
        if not clean:
            return
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        protected = protect_secret(clean)
        tmp_file = SESSION_DPAPI_FILE.with_suffix(".dpapi.tmp")
        try:
            tmp_file.write_text(protected, encoding="utf-8")
            os.replace(tmp_file, SESSION_DPAPI_FILE)
        finally:
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except Exception:
                    pass

    @classmethod
    def _load_telethon_session(cls) -> Any:
        """
        Завантажує Telethon StringSession, зашифровану через Windows DPAPI.
        Якщо знайдено старий відкритий SQLite-файл `.session` — автоматично
        мігрує його у DPAPI-файл та видаляє незашифрований `.session` з диска.
        """
        from telethon.sessions import StringSession

        if SESSION_DPAPI_FILE.exists():
            try:
                raw = SESSION_DPAPI_FILE.read_text(encoding="utf-8").strip()
                dec = unprotect_secret(raw).strip() if raw else ""
                if dec:
                    return StringSession(dec)
            except Exception as exc:
                print(f"[TelegramSilero] Не вдалося прочитати DPAPI-сесію: {exc}")

        if SESSION_FILE.exists():
            try:
                from telethon.sessions import SQLiteSession

                legacy = SQLiteSession(str(SESSION_PATH_BASE))
                try:
                    migrated = StringSession.save(legacy)
                finally:
                    try:
                        legacy.close()
                    except Exception:
                        pass
                if migrated:
                    cls._save_protected_session_string(migrated)
                    cls._remove_legacy_sqlite_session_files()
                    return StringSession(migrated)
            except Exception as exc:
                print(f"[TelegramSilero] Помилка міграції старої SQLite-сесії в DPAPI: {exc}")
                return str(SESSION_PATH_BASE)

        return StringSession()

    @classmethod
    def _persist_client_session(cls, client: Any) -> None:
        if client is None:
            return
        try:
            sess = getattr(client, "session", None)
            if sess is not None and hasattr(sess, "save"):
                saved = sess.save()
                if isinstance(saved, str) and saved.strip():
                    cls._save_protected_session_string(saved)
        except Exception as exc:
            print(f"[TelegramSilero] Не вдалося зберегти DPAPI-сесію: {exc}")

    async def _async_disconnect(self, delete_session: bool = False) -> None:
        if self._active_synth_task is not None and not self._active_synth_task.done():
            self._active_synth_task.cancel()
        self._active_synth_task = None
        if self._qr_wait_task is not None and not self._qr_wait_task.done():
            self._qr_wait_task.cancel()
        self._qr_wait_task = None
        self._qr_login = None
        self._bot_entity = None
        self._bot_user_id = 0
        self._confirmed_speaker = None
        self._phone_code_hash = ""
        self._connected = False
        self._user_info = None
        self._listener_client_id = 0
        client = self._client
        self._client = None
        if client is not None:
            try:
                if delete_session:
                    try:
                        await asyncio.wait_for(client.log_out(), timeout=6.0)
                    except Exception:
                        pass
                else:
                    self._persist_client_session(client)
                await asyncio.wait_for(client.disconnect(), timeout=6.0)
            except Exception:
                pass
        if delete_session:
            self._remove_session_files()

    async def _create_client(self, api_id: int, api_hash: str, proxy_mode: str, proxy_url: str) -> Any:
        if not is_telethon_installed():
            raise RuntimeError(
                tr("tg.err.telethon_missing", "Бібліотека Telethon не встановлена. Натисніть «Встановити Telethon».")
            )
        from telethon import TelegramClient

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        proxy_tuple = _parse_proxy_for_telethon(proxy_mode, proxy_url)
        kwargs: Dict[str, Any] = {
            "timeout": 12,
            "connection_retries": 4,
            "retry_delay": 1,
            "request_retries": 3,
        }
        if proxy_tuple is not None:
            kwargs["proxy"] = proxy_tuple

        session_obj = self._load_telethon_session()
        client = TelegramClient(session_obj, int(api_id), api_hash, **kwargs)
        await asyncio.wait_for(client.connect(), timeout=35.0)
        return client

    async def _fetch_me(self, client: Any) -> Dict[str, Any]:
        me = await asyncio.wait_for(client.get_me(), timeout=15.0)
        if me is None:
            return {}
        self._persist_client_session(client)
        info = {
            "id": int(getattr(me, "id", 0) or 0),
            "first_name": str(getattr(me, "first_name", "") or ""),
            "last_name": str(getattr(me, "last_name", "") or ""),
            "username": str(getattr(me, "username", "") or ""),
            "phone": str(getattr(me, "phone", "") or self._phone or ""),
        }
        self._user_info = info
        try:
            await self._attach_bot_listener(client)
        except Exception:
            pass
        return info

    async def _attach_bot_listener(self, client: Any) -> None:
        """Підписується на повідомлення/оновлення від @silero_voice_bot для автопідхоплення обраного голосу."""
        if self._listener_client_id == id(client):
            return
        from telethon import events

        if self._bot_entity is None or not self._bot_user_id:
            bot = await asyncio.wait_for(client.get_entity(BOT_USERNAME), timeout=15.0)
            self._bot_entity = bot
            self._bot_user_id = int(getattr(bot, "id", 0) or 0)

        target_bot = self._bot_entity or BOT_USERNAME

        async def _on_bot_event(event: Any) -> None:
            msg = getattr(event, "message", None)
            if msg is None or getattr(msg, "out", False):
                return
            txt = str(getattr(msg, "raw_text", "") or getattr(msg, "message", "") or "")
            vid = parse_speaker_selected_from_text(txt)
            if not vid:
                return
            self._confirmed_speaker = vid
            cb = self._speaker_changed_cb
            if cb is not None:
                try:
                    cb(vid)
                except Exception:
                    pass

        client.add_event_handler(_on_bot_event, events.NewMessage(chats=target_bot))
        client.add_event_handler(_on_bot_event, events.MessageEdited(chats=target_bot))
        self._listener_client_id = id(client)

    async def _async_start_auth(
        self,
        phone: str,
        api_id: int,
        api_hash: str,
        proxy_mode: str,
        proxy_url: str,
    ) -> Dict[str, Any]:
        await self._async_disconnect(delete_session=False)
        self._phone = phone.strip()
        self._api_id = int(api_id)
        self._api_hash = api_hash.strip()
        self._proxy_mode = proxy_mode
        self._proxy_url = proxy_url

        client = await self._create_client(self._api_id, self._api_hash, proxy_mode, proxy_url)
        self._client = client

        if await asyncio.wait_for(client.is_user_authorized(), timeout=15.0):
            self._connected = True
            user = await self._fetch_me(client)
            return {"state": "connected", "connected": True, "user": user}

        sent = await asyncio.wait_for(client.send_code_request(self._phone), timeout=20.0)
        self._phone_code_hash = str(getattr(sent, "phone_code_hash", "") or "")
        return {"state": "code_required", "connected": False}

    @staticmethod
    def _qr_expires_seconds(qr_obj: Any) -> int:
        exp = getattr(qr_obj, "expires", None)
        if exp is None:
            return 30
        try:
            now = datetime.datetime.now(tz=exp.tzinfo or datetime.timezone.utc)
            rem = int((exp - now).total_seconds())
            return max(10, min(120, rem))
        except Exception:
            return 30

    async def _async_start_qr_login(
        self,
        api_id: int,
        api_hash: str,
        proxy_mode: str,
        proxy_url: str,
    ) -> Dict[str, Any]:
        await self._async_disconnect(delete_session=False)
        self._phone = ""
        self._api_id = int(api_id)
        self._api_hash = api_hash.strip()
        self._proxy_mode = proxy_mode
        self._proxy_url = proxy_url

        client = await self._create_client(self._api_id, self._api_hash, proxy_mode, proxy_url)
        self._client = client

        if await asyncio.wait_for(client.is_user_authorized(), timeout=15.0):
            self._connected = True
            user = await self._fetch_me(client)
            return {"state": "connected", "connected": True, "user": user}

        qr = await asyncio.wait_for(client.qr_login(), timeout=20.0)
        self._qr_login = qr
        return {
            "state": "qr_ready",
            "connected": False,
            "url": qr.url,
            "expires_in": self._qr_expires_seconds(qr),
        }

    async def _async_recreate_qr_login(self) -> Dict[str, Any]:
        if self._client is None or self._qr_login is None:
            raise RuntimeError(tr("tg.err.not_initialized", "Telegram-клієнт не ініціалізовано."))
        await asyncio.wait_for(self._qr_login.recreate(), timeout=20.0)
        return {
            "state": "qr_ready",
            "connected": False,
            "url": self._qr_login.url,
            "expires_in": self._qr_expires_seconds(self._qr_login),
        }

    async def _async_wait_qr_login(self, timeout_sec: float) -> Dict[str, Any]:
        if self._client is None or self._qr_login is None:
            raise RuntimeError(tr("tg.err.not_initialized", "Telegram-клієнт не ініціалізовано."))
        from telethon.errors import SessionPasswordNeededError

        task = asyncio.create_task(self._qr_login.wait(timeout=timeout_sec))
        self._qr_wait_task = task
        try:
            await task
            self._connected = True
            self._qr_login = None
            user = await self._fetch_me(self._client)
            return {"state": "connected", "connected": True, "user": user}
        except asyncio.CancelledError:
            return {"state": "cancelled", "connected": False}
        except asyncio.TimeoutError:
            return {"state": "qr_expired", "connected": False}
        except SessionPasswordNeededError:
            self._qr_login = None
            return {"state": "password_required", "connected": False}
        finally:
            if self._qr_wait_task is task:
                self._qr_wait_task = None

    async def _async_submit_code(self, code: str) -> Dict[str, Any]:
        if self._client is None:
            raise RuntimeError(tr("tg.err.not_initialized", "Telegram-клієнт не ініціалізовано."))
        from telethon.errors import PhoneCodeExpiredError, PhoneCodeInvalidError, SessionPasswordNeededError

        clean_code = re.sub(r"\s+", "", str(code or ""))
        try:
            await asyncio.wait_for(
                self._client.sign_in(
                    phone=self._phone,
                    code=clean_code,
                    phone_code_hash=self._phone_code_hash or None,
                ),
                timeout=30.0,
            )
            self._connected = True
            user = await self._fetch_me(self._client)
            return {"state": "connected", "connected": True, "user": user}
        except SessionPasswordNeededError:
            return {"state": "password_required", "connected": False}
        except PhoneCodeInvalidError:
            raise RuntimeError(tr("tg.err.invalid_code", "Невірний код підтвердження Telegram."))
        except PhoneCodeExpiredError:
            raise RuntimeError(tr("tg.err.expired_code", "Термін дії коду вичерпано. Запросіть код повторно."))

    async def _async_submit_password(self, password: str) -> Dict[str, Any]:
        if self._client is None:
            raise RuntimeError(tr("tg.err.not_initialized", "Telegram-клієнт не ініціалізовано."))
        from telethon.errors import PasswordHashInvalidError

        try:
            await asyncio.wait_for(self._client.sign_in(password=password), timeout=30.0)
            self._connected = True
            user = await self._fetch_me(self._client)
            return {"state": "connected", "connected": True, "user": user}
        except PasswordHashInvalidError:
            raise RuntimeError(tr("tg.err.invalid_2fa", "Невірний пароль двофакторної аутентифікації (2FA)."))

    async def _async_connect_existing(
        self,
        api_id: int,
        api_hash: str,
        proxy_mode: str,
        proxy_url: str,
    ) -> Dict[str, Any]:
        if self._connect_lock is None:
            self._connect_lock = asyncio.Lock()
        async with self._connect_lock:
            return await self._async_connect_existing_locked(api_id, api_hash, proxy_mode, proxy_url)

    async def _async_connect_existing_locked(
        self,
        api_id: int,
        api_hash: str,
        proxy_mode: str,
        proxy_url: str,
    ) -> Dict[str, Any]:
        if not self._has_saved_session():
            self._connected = False
            self._user_info = None
            return {"state": "auth_required", "connected": False, "user": None}

        if (
            self._client is not None
            and self._client.is_connected()
            and self._api_id == int(api_id)
            and self._api_hash == api_hash
            and self._proxy_mode == proxy_mode
            and self._proxy_url == proxy_url
            and self._connected
        ):
            return {"state": "connected", "connected": True, "user": self._user_info}

        await self._async_disconnect(delete_session=False)
        self._api_id = int(api_id)
        self._api_hash = api_hash
        self._proxy_mode = proxy_mode
        self._proxy_url = proxy_url

        client = await self._create_client(self._api_id, self._api_hash, proxy_mode, proxy_url)
        self._client = client

        authorized = await asyncio.wait_for(client.is_user_authorized(), timeout=15.0)
        if not authorized:
            await self._async_disconnect(delete_session=False)
            return {"state": "auth_required", "connected": False, "user": None}

        self._connected = True
        user = await self._fetch_me(client)
        return {"state": "connected", "connected": True, "user": user}

    async def _ensure_ready(self) -> Tuple[Any, Any, int]:
        if self._client is not None and self._client.is_connected() and self._connected:
            if self._bot_entity is None or not self._bot_user_id:
                bot = await asyncio.wait_for(self._client.get_entity(BOT_USERNAME), timeout=15.0)
                self._bot_entity = bot
                self._bot_user_id = int(getattr(bot, "id", 0) or 0)
            return self._client, self._bot_entity, self._bot_user_id

        creds = self._get_saved_credentials()
        api_id, api_hash = self._resolve_api_credentials()
        proxy_mode = str(creds.get("proxy_mode") or self._proxy_mode or "auto").strip().lower()
        proxy_url = str(creds.get("proxy_url") or self._proxy_url or "").strip()

        if api_id and api_hash and self._has_saved_session():
            res = await self._async_connect_existing(api_id, api_hash, proxy_mode, proxy_url)
            if res.get("connected") and self._client is not None:
                bot = await asyncio.wait_for(self._client.get_entity(BOT_USERNAME), timeout=15.0)
                self._bot_entity = bot
                self._bot_user_id = int(getattr(bot, "id", 0) or 0)
                return self._client, self._bot_entity, self._bot_user_id

        raise RuntimeError(
            tr(
                "tg.err.not_connected",
                "Telegram не підключено. Відкрийте вкладку «Онлайн-голоси» та підключіть акаунт у блоці Silero Bot.",
            )
        )

    async def _wait_bot_reply(
        self,
        client: Any,
        bot_entity: Any,
        bot_user_id: int,
        command_text: str,
        matcher: Any,
        timeout_sec: float,
        listen_edits: bool = False,
    ) -> Any:
        from telethon import events

        q: asyncio.Queue[Any] = asyncio.Queue()

        async def _on_event(ev: Any) -> None:
            await q.put(ev.message)

        ev_new = events.NewMessage(chats=bot_entity, incoming=True)
        ev_edit = events.MessageEdited(chats=bot_entity, incoming=True) if listen_edits else None
        client.add_event_handler(_on_event, ev_new)
        if ev_edit is not None:
            client.add_event_handler(_on_event, ev_edit)

        try:
            sent_msg = await asyncio.wait_for(client.send_message(bot_entity, command_text), timeout=20.0)
            sent_msg_id = int(getattr(sent_msg, "id", 0) or 0)

            deadline = time.monotonic() + timeout_sec
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timeout waiting for @silero_voice_bot response")
                msg = await asyncio.wait_for(q.get(), timeout=remaining)
                if getattr(msg, "out", False):
                    continue
                if int(getattr(msg, "id", 0) or 0) <= sent_msg_id:
                    continue
                res = matcher(msg, sent_msg_id, bot_user_id)
                if res is not None:
                    return res
        finally:
            client.remove_event_handler(_on_event, ev_new)
            if ev_edit is not None:
                client.remove_event_handler(_on_event, ev_edit)

    async def _async_get_current_voice(self) -> Optional[Dict[str, str]]:
        client, bot_entity, bot_user_id = await self._ensure_ready()
        assert self._speaker_lock is not None
        async with self._speaker_lock:
            def _match(msg: Any, sent_id: int, _bot_id: int) -> Optional[Dict[str, str]]:
                if _extract_reply_to_msg_id(msg) != sent_id:
                    return None
                if getattr(msg, "media", None) is not None and not getattr(msg, "raw_text", ""):
                    return None
                parsed = parse_voice_info(getattr(msg, "raw_text", "") or getattr(msg, "message", "") or "")
                return parsed

            try:
                info = await self._wait_bot_reply(
                    client, bot_entity, bot_user_id, "/speaker", _match, timeout_sec=45.0, listen_edits=True
                )
            except TimeoutError:
                return None

            if info and info.get("id"):
                self._confirmed_speaker = info["id"]
            return info

    async def _async_set_speaker(self, voice_code: str, force: bool = False) -> Tuple[bool, str]:
        clean = str(voice_code or "").strip()
        if clean.lower().startswith("/speaker"):
            clean = clean[len("/speaker"):].strip()
        clean = clean.lstrip("/").strip().lower()
        if not clean or any(ch.isspace() for ch in clean):
            return False, tr("tg.err.invalid_voice_code", "Невірний код голосу.")

        client, bot_entity, bot_user_id = await self._ensure_ready()
        assert self._speaker_lock is not None
        async with self._speaker_lock:
            if not force and self._confirmed_speaker == clean:
                return True, clean

            def _match(msg: Any, sent_id: int, _bot_id: int) -> Optional[bool]:
                if _extract_reply_to_msg_id(msg) != sent_id:
                    return None
                txt = str(getattr(msg, "raw_text", "") or getattr(msg, "message", "") or "").strip()
                if not txt:
                    return None
                parsed = parse_set_speaker_response(txt)
                if parsed is None:
                    return False
                return parsed

            try:
                ok = await self._wait_bot_reply(
                    client,
                    bot_entity,
                    bot_user_id,
                    f"/speaker {clean}",
                    _match,
                    timeout_sec=30.0,
                    listen_edits=True,
                )
            except TimeoutError:
                self._confirmed_speaker = None
                raise RuntimeError(tr("tg.err.speaker_timeout", "Бот @silero_voice_bot не відповів на команду /speaker."))

            if ok:
                self._confirmed_speaker = clean
                return True, clean
            self._confirmed_speaker = None
            return False, tr("tg.err.invalid_voice_code_detail", "Бот відхилив код голосу «{code}».", code=clean)

    async def _async_get_limits(self) -> Optional[Dict[str, str]]:
        client, bot_entity, bot_user_id = await self._ensure_ready()
        assert self._limits_lock is not None
        async with self._limits_lock:
            def _match(msg: Any, sent_id: int, _bot_id: int) -> Optional[Dict[str, str]]:
                if getattr(msg, "media", None) is not None and not getattr(msg, "web_preview", None):
                    return None
                txt = str(getattr(msg, "raw_text", "") or getattr(msg, "message", "") or "").strip()
                if not txt:
                    return None
                return parse_limits_info(txt)

            try:
                return await self._wait_bot_reply(
                    client, bot_entity, bot_user_id, "/limits", _match, timeout_sec=45.0, listen_edits=True
                )
            except TimeoutError:
                return None

    async def _async_synthesize(
        self,
        text: str,
        voice_code: str = "",
        timeout_sec: float = 60.0,
    ) -> Tuple[bytes, str]:
        clean_text = (text or "").strip()
        if not clean_text:
            raise RuntimeError(tr("tg.err.empty_text", "Текст для озвучення порожній."))
        if len(clean_text) > 4000:
            raise RuntimeError(tr("tg.err.text_too_long_4000", "Текст задовгий (максимум 4000 символів)."))

        client, bot_entity, bot_user_id = await self._ensure_ready()

        v_code = str(voice_code or "").strip()
        if v_code and v_code != "__bot_current__":
            ok, msg_err = await self._async_set_speaker(v_code)
            if not ok:
                raise RuntimeError(msg_err)

        def _match(msg: Any, sent_id: int, _bot_id: int) -> Optional[Tuple[str, Any]]:
            reply_id = _extract_reply_to_msg_id(msg)
            mime = _get_message_mime(msg)
            has_audio = bool(
                getattr(msg, "voice", None) is not None
                or getattr(msg, "audio", None) is not None
                or mime in ("audio/ogg", "audio/mpeg")
                or mime.startswith("audio/")
            )
            if has_audio and reply_id == sent_id:
                return ("audio", (msg, mime or "audio/ogg"))

            txt = str(getattr(msg, "raw_text", "") or getattr(msg, "message", "") or "").strip()
            if txt:
                if reply_id == sent_id:
                    return ("reject", (classify_silero_rejection(txt), txt))
                if reply_id is None:
                    kind = classify_silero_rejection(txt)
                    if kind in ("limit_exceeded", "text_too_long"):
                        return ("reject", (kind, txt))
            return None

        try:
            kind_res, payload = await self._wait_bot_reply(
                client,
                bot_entity,
                bot_user_id,
                clean_text,
                _match,
                timeout_sec=max(5.0, float(timeout_sec)),
                listen_edits=False,
            )
        except TimeoutError:
            raise RuntimeError(
                tr("tg.err.synth_timeout", "Перевищено час очікування аудіо-відповіді від @silero_voice_bot.")
            )

        if kind_res == "reject":
            rej_kind, raw_txt = payload
            if rej_kind == "limit_exceeded":
                raise RuntimeError(tr("tg.err.limit_exceeded", "Вичерпано ліміт озвучки @silero_voice_bot."))
            if rej_kind == "text_too_long":
                raise RuntimeError(tr("tg.err.bot_text_too_long", "Довжина тексту перевищує допустимий ліміт @silero_voice_bot."))
            raise RuntimeError(tr("tg.err.bot_rejected", "Бот @silero_voice_bot відхилив запит: {msg}", msg=raw_txt[:180]))

        voice_msg, mime = payload
        for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
            buf = io.BytesIO()
            await asyncio.wait_for(client.download_media(voice_msg, file=buf), timeout=30.0)
            audio_bytes = buf.getvalue()
            if audio_bytes:
                return audio_bytes, mime
            if attempt < MAX_DOWNLOAD_ATTEMPTS:
                await asyncio.sleep(DOWNLOAD_RETRY_DELAY_SEC)

        raise RuntimeError(
            tr("tg.err.empty_download", "Не вдалося завантажити голосове повідомлення від @silero_voice_bot (порожній файл).")
        )


telegram_silero = TelegramSileroManager()
