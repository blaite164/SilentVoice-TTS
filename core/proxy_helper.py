"""
core/proxy_helper.py — Управління SOCKS5-проксі для додаткових мережевих підключень для онлайн-API (TTS та AI).
Підтримує власний SOCKS5-сервер або автоматичне підхоплення локального SOCKS5-порту
від уже запущених на ПК програм (V2Ray, Nekoray, Nekobox, Clash, Shadowsocks, Tor).
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Any, Dict, Optional, Tuple

import requests

from core.config import config
from core.i18n import tr

_CACHE_LOCK = threading.Lock()
_CACHED_SOCKS5_URL: str = ""
_CACHED_AT: float = 0.0
_CACHE_TTL_SEC = 300.0  # 5 хвилин

# Популярні локальні SOCKS5-порти (V2Ray/Nekoray, Nekobox, Clash, Shadowsocks, Tor)
_LOCAL_SOCKS5_CANDIDATES: Tuple[str, ...] = (
    "127.0.0.1:10808",
    "127.0.0.1:2080",
    "127.0.0.1:1080",
    "127.0.0.1:7890",
    "127.0.0.1:7891",
    "127.0.0.1:9050",
)

_PROBE_URLS: Tuple[str, ...] = (
    "https://www.gstatic.com/generate_204",
    "https://api.ipify.org?format=json",
)


def format_socks5_url(raw: str, remote_dns: bool = True) -> str:
    """Нормалізує рядок проксі до формату socks5h://... або socks5://..."""
    p = str(raw or "").strip()
    if not p:
        return ""
    scheme = "socks5h" if remote_dns else "socks5"
    if "://" not in p:
        return f"{scheme}://{p}"
    if remote_dns and p.startswith("socks5://"):
        return "socks5h://" + p[len("socks5://"):]
    if not remote_dns and p.startswith("socks5h://"):
        return "socks5://" + p[len("socks5h://"):]
    return p


def _is_local_port_open(host_port: str, timeout: float = 0.25) -> bool:
    """Швидко перевіряє за TCP, чи слухає локальний порт на ПК."""
    try:
        host, port_str = host_port.rsplit(":", 1)
        with socket.create_connection((host, int(port_str)), timeout=timeout):
            return True
    except Exception:
        return False


def _probe_socks5(proxy_url: str, timeout: Tuple[float, float] = (3.5, 4.5)) -> bool:
    """Перевіряє, чи пропускає SOCKS5-проксі HTTPS-трафік."""
    proxies = {"http": proxy_url, "https": proxy_url}
    for test_url in _PROBE_URLS:
        try:
            resp = requests.get(
                test_url,
                headers={"User-Agent": "Mozilla/5.0"},
                proxies=proxies,
                timeout=timeout,
            )
            ok = resp.status_code in (200, 204)
            resp.close()
            if ok:
                return True
        except Exception:
            continue
    return False


def find_local_socks5_proxy() -> str:
    """Шукає запущений локальний SOCKS5-порт на ПК паралельно та повертає найшвидший робочий."""
    open_ports = [hp for hp in _LOCAL_SOCKS5_CANDIDATES if _is_local_port_open(hp)]
    if not open_ports:
        return ""
    if len(open_ports) == 1:
        url = format_socks5_url(open_ports[0], remote_dns=True)
        return url if _probe_socks5(url, timeout=(2.5, 3.5)) else ""

    import concurrent.futures

    best: list[tuple[float, str]] = []
    lock = threading.Lock()

    def _check(host_port: str) -> None:
        t0 = time.monotonic()
        url = format_socks5_url(host_port, remote_dns=True)
        if _probe_socks5(url, timeout=(2.5, 3.5)):
            with lock:
                best.append((time.monotonic() - t0, url))

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(open_ports)) as pool:
        list(pool.map(_check, open_ports))
    if not best:
        return ""
    best.sort(key=lambda item: item[0])
    return best[0][1]


def invalidate_socks5_cache() -> None:
    """Скидає кеш знайденого локального SOCKS5-проксі."""
    global _CACHED_SOCKS5_URL, _CACHED_AT
    with _CACHE_LOCK:
        _CACHED_SOCKS5_URL = ""
        _CACHED_AT = 0.0


def get_active_socks5_url(force_refresh: bool = False) -> str:
    """
    Повертає активний SOCKS5 URL (socks5h://...), якщо SOCKS5 увімкнено у налаштуваннях.
    Якщо користувач не вказав власний сервер — шукає локальний SOCKS5 на ПК.
    """
    global _CACHED_SOCKS5_URL, _CACHED_AT
    if not bool(config.get("use_socks5_proxy", False)):
        return ""

    custom = str(config.get("socks5_proxy_url", "")).strip()
    if custom:
        return format_socks5_url(custom, remote_dns=True)

    now = time.monotonic()
    with _CACHE_LOCK:
        if not force_refresh and _CACHED_SOCKS5_URL and (now - _CACHED_AT) < _CACHE_TTL_SEC:
            return _CACHED_SOCKS5_URL

    found = find_local_socks5_proxy()
    with _CACHE_LOCK:
        _CACHED_SOCKS5_URL = found
        _CACHED_AT = time.monotonic()
        return _CACHED_SOCKS5_URL


def get_active_api_proxies() -> Optional[Dict[str, str]]:
    """Повертає словник proxies для requests, якщо увімкнено SOCKS5, інакше None."""
    url = get_active_socks5_url()
    if not url:
        return None
    return {"http": url, "https": url}


def get_edge_tts_proxy_config() -> Tuple[Optional[Any], Optional[str]]:
    """
    Повертає (aiohttp_connector, http_proxy_url) для Edge-TTS з урахуванням SOCKS5
    та custom_proxy_url. Повинен викликатись усередині активного asyncio-циклу.
    """
    socks_url = get_active_socks5_url()
    if socks_url:
        try:
            from aiohttp_socks import ProxyConnector
            return ProxyConnector.from_url(format_socks5_url(socks_url, remote_dns=False), rdns=True), None
        except Exception:
            return None, None

    custom_px = str(config.get("custom_proxy_url", "") or "").strip()
    if custom_px:
        low = custom_px.lower()
        if low.startswith(("socks5://", "socks5h://", "socks4://")):
            try:
                from aiohttp_socks import ProxyConnector
                norm = format_socks5_url(custom_px, remote_dns=False) if low.startswith("socks5") else custom_px
                return ProxyConnector.from_url(norm, rdns=True), None
            except Exception:
                return None, None
        http_px = custom_px if "://" in custom_px else f"http://{custom_px}"
        return None, http_px

    return None, None


def get_edge_tts_connector() -> Optional[Any]:
    """Повертає aiohttp_socks.ProxyConnector для Edge-TTS, якщо увімкнено SOCKS5 або SOCKS-проксі."""
    connector, _ = get_edge_tts_proxy_config()
    return connector


def ensure_socks5_proxy_ready() -> Tuple[bool, str]:
    """
    Перевіряє вказаний SOCKS5-проксі або знаходить локальний порт на ПК.
    Повертає (ok, зрозумілий_статус_для_UI).
    """
    if not bool(config.get("use_socks5_proxy", False)):
        return False, tr("proxy.disabled", "Вимкнено")

    custom = str(config.get("socks5_proxy_url", "")).strip()
    if custom:
        url = format_socks5_url(custom, remote_dns=True)
        if _probe_socks5(url, timeout=(4.0, 5.0)):
            return True, tr("proxy.enabled_addr", "Увімкнено проксі ({addr})", addr=custom)
        return False, tr("proxy.no_connection", "Немає з'єднання з {addr} — перевірте адресу або проксі-сервер", addr=custom)

    url = get_active_socks5_url(force_refresh=True)
    if url:
        host = url.split("://", 1)[-1]
        return True, tr("proxy.enabled_addr", "Увімкнено проксі ({addr})", addr=host)
    return False, tr("proxy.not_found", "Локальний SOCKS5 не знайдено — вкажіть свій ip:port або запустіть мережеву програму")
