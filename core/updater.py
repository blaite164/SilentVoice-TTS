"""
core/updater.py — Перевірка оновлень та автооновлення SilentVoice TTS через GitHub Releases і GitLab Releases.

Як увімкнути автооновлення після створення репозиторію:
1. Впишіть свій GitHub у DEFAULT_GITHUB_REPO (наприклад: "Blaite/SilentVoice" або "https://github.com/Blaite/SilentVoice")
   та/або GitLab у DEFAULT_GITLAB_REPO (наприклад: "https://gitlab.com/Blaite/SilentVoice").
2. При випуску нової версії змініть APP_VERSION (та AppVersion у installer/SilentVoice.iss),
   зберіть інсталятор через build_installer.ps1 і прикріпіть SilentVoice_Setup_<версія>.exe до Release на GitHub/GitLab.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Callable, List, Optional, Tuple
from urllib.parse import quote, urlsplit

import requests

from core.config import DATA_DIR, config
from core.i18n import get_ui_language, tr
from core.proxy_helper import get_active_api_proxies

APP_VERSION = "1.0.3"
_INNO_SUPPORTED_LANGS = {"uk", "en", "pl", "de", "fr", "es", "ru"}
_SECTION_HEADING_RE = re.compile(r"(?m)^##\s+(.+)$")
_LANG_SECTION_MARKERS = {
    "uk": ("🇺🇦", "що нового", "українськ"),
    "en": ("🇬🇧", "🇺🇸", "what's new", "whats new", "changelog", "release notes"),
    "ru": ("🇷🇺", "что нового", "русск"),
    "pl": ("🇵🇱", "co nowego", "polsk"),
    "de": ("🇩🇪", "was ist neu", "neuheiten", "deutsch"),
    "fr": ("🇫🇷", "nouveautés", "quoi de neuf", "français"),
    "es": ("🇪🇸", "novedades", "qué hay de nuevo", "español"),
}

# Впишіть сюди адресу свого репозиторію GitHub та/або GitLab (або "логін/репозиторій"):
DEFAULT_GITHUB_REPO = "https://github.com/blaite164/SilentVoice-TTS"
DEFAULT_GITLAB_REPO = "https://gitlab.com/blaite164/silentvoice-tts"

UPDATES_DIR = DATA_DIR / "updates"
_SHA256_HEX_RE = re.compile(r"\b([a-fA-F0-9]{64})\b")


class UpdateRepoNotConfiguredError(RuntimeError):
    """Джерело оновлень (GitHub / GitLab репозиторій) ще не вказано."""


@dataclass(frozen=True)
class UpdateReleaseInfo:
    version: str
    tag_name: str
    title: str
    notes: str
    installer_url: str
    installer_name: str
    installer_size: int = 0
    sha256: str = ""
    html_url: str = ""
    source: str = "GitHub"


def parse_version_tuple(version_str: str) -> Tuple[int, ...]:
    """Перетворює рядок версії ('v1.2.3', '1.0.1-beta') на кортеж чисел (1, 2, 3)."""
    cleaned = str(version_str or "").strip().lstrip("vV").strip()
    if not cleaned:
        return (0, 0, 0)
    main_part = re.split(r"[-+_\s]", cleaned, maxsplit=1)[0]
    nums = [int(part) for part in re.findall(r"\d+", main_part)]
    if not nums:
        return (0, 0, 0)
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums)


def is_newer_version(remote_version: str, current_version: str = APP_VERSION) -> bool:
    """Повертає True, якщо remote_version новіша за current_version."""
    rem = list(parse_version_tuple(remote_version))
    cur = list(parse_version_tuple(current_version))
    target_len = max(len(rem), len(cur), 3)
    while len(rem) < target_len:
        rem.append(0)
    while len(cur) < target_len:
        cur.append(0)
    return tuple(rem) > tuple(cur)


def parse_repo_source(raw: str) -> Optional[Tuple[str, str, str]]:
    """
    Розпізнає рядок репозиторію і повертає (provider, host, slug):
    - 'owner/repo' -> ('github', 'github.com', 'owner/repo')
    - 'https://github.com/owner/repo' -> ('github', 'github.com', 'owner/repo')
    - 'https://gitlab.com/owner/repo' -> ('gitlab', 'gitlab.com', 'owner/repo')
    """
    text = str(raw or "").strip()
    if not text:
        return None

    if "://" not in text and not text.startswith(("github.com/", "gitlab.com/")):
        slug = text.strip("/").strip()
        if re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", slug):
            if slug.lower().endswith(".git"):
                slug = slug[:-4]
            return ("github", "github.com", slug)
        return None

    if not text.startswith(("http://", "https://")):
        text = "https://" + text

    parsed = urlsplit(text)
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return None

    parts = [p for p in (parsed.path or "").split("/") if p]
    # Відрізаємо службові суфікси (/releases, /-/releases тощо)
    cleaned_parts: List[str] = []
    for p in parts:
        if p in ("-", "releases", "tags", "tree", "blob"):
            break
        cleaned_parts.append(p)

    if len(cleaned_parts) < 2:
        return None

    if cleaned_parts[-1].lower().endswith(".git"):
        cleaned_parts[-1] = cleaned_parts[-1][:-4]

    if host in ("github.com", "www.github.com"):
        slug = f"{cleaned_parts[0]}/{cleaned_parts[1]}"
        return ("github", "github.com", slug)

    if "gitlab" in host:
        slug = "/".join(cleaned_parts)
        return ("gitlab", host, slug)

    return None


def get_configured_update_sources(custom_source: str = "") -> List[Tuple[str, str, str]]:
    """Повертає список унікальних джерел оновлень (provider, host, slug)."""
    raw_candidates = [
        custom_source,
        str(config.get("update_repo_url", "") or ""),
        DEFAULT_GITHUB_REPO,
        DEFAULT_GITLAB_REPO,
    ]
    sources: List[Tuple[str, str, str]] = []
    seen = set()
    for raw in raw_candidates:
        for piece in re.split(r"[,\n;]+", str(raw or "")):
            parsed = parse_repo_source(piece)
            if parsed is not None and parsed not in seen:
                seen.add(parsed)
                sources.append(parsed)
    return sources


def has_configured_update_sources() -> bool:
    return bool(get_configured_update_sources())


def _extract_sha256_for_asset(
    asset_name: str,
    digest_field: str,
    release_notes: str,
    checksums_text: str = "",
) -> str:
    """Шукає SHA-256 для конкретного файлу інсталятора у полі digest, SHA256SUMS.txt або описі релізу."""
    if digest_field:
        low_d = digest_field.strip().lower()
        if low_d.startswith("sha256:"):
            cand = low_d.split("sha256:", 1)[1].strip()
            if _SHA256_HEX_RE.fullmatch(cand):
                return cand
        elif _SHA256_HEX_RE.fullmatch(low_d):
            return low_d

    target_low = asset_name.strip().lower()
    for blob in (checksums_text, release_notes):
        if not blob:
            continue
        for line in blob.splitlines():
            if target_low and target_low in line.lower():
                m = _SHA256_HEX_RE.search(line)
                if m:
                    return m.group(1).lower()
        # Якщо у файлі SHA256SUMS лише один хеш
        matches = _SHA256_HEX_RE.findall(blob)
        if blob is checksums_text and len(matches) == 1:
            return matches[0].lower()
    return ""


def _fetch_text_quiet(url: str, proxies: Optional[dict], timeout: float = 6.0) -> str:
    if not url:
        return ""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": f"SilentVoice-TTS/{APP_VERSION}"},
            proxies=proxies,
            timeout=timeout,
        )
        if resp.status_code == 200 and len(resp.text) < 64_000:
            return resp.text
    except Exception:
        pass
    return ""


def _check_github_release(slug: str, proxies: Optional[dict], timeout: float) -> UpdateReleaseInfo:
    api_url = f"https://api.github.com/repos/{slug}/releases/latest"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": f"SilentVoice-TTS/{APP_VERSION}",
    }
    resp = requests.get(api_url, headers=headers, proxies=proxies, timeout=timeout)
    if resp.status_code == 404:
        raise RuntimeError(
            tr(
                "update.err.no_releases",
                "У репозиторії {repo} ще немає опублікованих релізів (Releases).",
                repo=slug,
            )
        )
    if resp.status_code != 200:
        raise RuntimeError(f"GitHub API HTTP {resp.status_code}")

    data = resp.json()
    tag_name = str(data.get("tag_name", "") or "").strip()
    version = tag_name.lstrip("vV").strip() or str(data.get("name", "") or "").strip()
    title = str(data.get("name", "") or tag_name).strip()
    notes = str(data.get("body", "") or "").strip()
    html_url = str(data.get("html_url", "") or f"https://github.com/{slug}/releases").strip()

    assets = data.get("assets") or []
    exe_asset = None
    sums_url = ""
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name", "") or "").strip()
        dl_url = str(asset.get("browser_download_url", "") or "").strip()
        low_name = name.lower()
        if not dl_url:
            continue
        if low_name.endswith(".exe"):
            if exe_asset is None or ("setup" in low_name and "silentvoice" in low_name):
                exe_asset = asset
        elif "sha256" in low_name or low_name.endswith((".sha256", "checksums.txt", "sha256sums.txt")):
            sums_url = dl_url

    installer_url = ""
    installer_name = ""
    installer_size = 0
    digest_field = ""
    if isinstance(exe_asset, dict):
        installer_url = str(exe_asset.get("browser_download_url", "") or "").strip()
        installer_name = str(exe_asset.get("name", "") or "").strip()
        installer_size = int(exe_asset.get("size", 0) or 0)
        digest_field = str(exe_asset.get("digest", "") or "").strip()

    checksums_text = _fetch_text_quiet(sums_url, proxies) if sums_url else ""
    sha256 = _extract_sha256_for_asset(installer_name, digest_field, notes, checksums_text)

    return UpdateReleaseInfo(
        version=version,
        tag_name=tag_name or f"v{version}",
        title=title,
        notes=notes,
        installer_url=installer_url,
        installer_name=installer_name or f"SilentVoice_Setup_{version}.exe",
        installer_size=installer_size,
        sha256=sha256,
        html_url=html_url,
        source="GitHub",
    )


def _check_gitlab_release(host: str, slug: str, proxies: Optional[dict], timeout: float) -> UpdateReleaseInfo:
    encoded_slug = quote(slug, safe="")
    api_url = f"https://{host}/api/v4/projects/{encoded_slug}/releases"
    headers = {"User-Agent": f"SilentVoice-TTS/{APP_VERSION}"}
    resp = requests.get(api_url, headers=headers, proxies=proxies, timeout=timeout)
    if resp.status_code == 404:
        raise RuntimeError(
            tr(
                "update.err.no_releases",
                "У репозиторії {repo} ще немає опублікованих релізів (Releases).",
                repo=slug,
            )
        )
    if resp.status_code != 200:
        raise RuntimeError(f"GitLab API HTTP {resp.status_code}")

    releases = resp.json()
    if not isinstance(releases, list) or not releases:
        raise RuntimeError(
            tr(
                "update.err.no_releases",
                "У репозиторії {repo} ще немає опублікованих релізів (Releases).",
                repo=slug,
            )
        )

    data = releases[0]
    tag_name = str(data.get("tag_name", "") or "").strip()
    version = tag_name.lstrip("vV").strip() or str(data.get("name", "") or "").strip()
    title = str(data.get("name", "") or tag_name).strip()
    notes = str(data.get("description", "") or "").strip()
    html_url = f"https://{host}/{slug}/-/releases/{tag_name}" if tag_name else f"https://{host}/{slug}/-/releases"

    links = ((data.get("assets") or {}).get("links") or []) if isinstance(data.get("assets"), dict) else []
    installer_url = ""
    installer_name = ""
    sums_url = ""
    for link in links:
        if not isinstance(link, dict):
            continue
        name = str(link.get("name", "") or "").strip()
        url = str(link.get("direct_asset_url") or link.get("url") or "").strip()
        if not url:
            continue
        low_check = f"{name} {url}".lower()
        if ".exe" in low_check:
            if not installer_url or ("setup" in low_check and "silentvoice" in low_check):
                installer_url = url
                installer_name = name if name.lower().endswith(".exe") else Path(urlsplit(url).path).name
        elif "sha256" in low_check or "checksum" in low_check:
            sums_url = url

    checksums_text = _fetch_text_quiet(sums_url, proxies) if sums_url else ""
    sha256 = _extract_sha256_for_asset(installer_name, "", notes, checksums_text)

    return UpdateReleaseInfo(
        version=version,
        tag_name=tag_name or f"v{version}",
        title=title,
        notes=notes,
        installer_url=installer_url,
        installer_name=installer_name or f"SilentVoice_Setup_{version}.exe",
        installer_size=0,
        sha256=sha256,
        html_url=html_url,
        source="GitLab",
    )


def check_for_updates(
    custom_source: str = "",
    timeout: float = 10.0,
) -> Tuple[UpdateReleaseInfo, bool]:
    """
    Перевіряє останній реліз на GitHub та/або GitLab.
    Повертає (release_info, is_newer).
    """
    sources = get_configured_update_sources(custom_source)
    if not sources:
        raise UpdateRepoNotConfiguredError(
            tr(
                "update.err.not_configured",
                "Адресу репозиторію GitHub або GitLab ще не вказано.",
            )
        )

    proxies = get_active_api_proxies()
    errors: List[str] = []
    for provider, host, slug in sources:
        try:
            if provider == "github":
                info = _check_github_release(slug, proxies, timeout)
            else:
                info = _check_gitlab_release(host, slug, proxies, timeout)
            return info, is_newer_version(info.version, APP_VERSION)
        except Exception as exc:
            errors.append(f"{provider.title()} ({slug}): {exc}")

    raise RuntimeError("; ".join(errors))


def download_update_installer(
    release: UpdateReleaseInfo,
    progress_cb: Optional[Callable[[int, float, float], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    timeout: float = 30.0,
) -> Path:
    """
    Завантажує .exe інсталятор оновлення у папку %APPDATA%/SilentVoice/updates/,
    перевіряє заголовок PE ('MZ') та SHA-256 (якщо хеш вказано у релізі).
    """
    if not release.installer_url:
        raise RuntimeError(
            tr(
                "update.err.no_installer_asset",
                "У релізі {tag} не прикріплено файл інсталятора (.exe). Відкрийте сторінку релізу в браузері.",
                tag=release.tag_name,
            )
        )

    UPDATES_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(release.installer_name).name)
    if not safe_name.lower().endswith(".exe"):
        safe_name = f"SilentVoice_Setup_{release.version}.exe"

    dest_path = UPDATES_DIR / safe_name
    part_path = UPDATES_DIR / f"{safe_name}.part"

    # Якщо інсталятор уже завантажено раніше (наприклад, користувач скасував UAC)
    # і його SHA-256 збігається з релізом — не перекачуємо ~200 МБ повторно.
    if dest_path.is_file() and release.sha256:
        try:
            existing_hash = hashlib.sha256()
            with open(dest_path, "rb") as existing_f:
                head = existing_f.read(2)
                if head == b"MZ":
                    existing_hash.update(head)
                    for block in iter(lambda: existing_f.read(65536), b""):
                        existing_hash.update(block)
                    if existing_hash.hexdigest().lower() == release.sha256.strip().lower():
                        size_mb = dest_path.stat().st_size / (1024.0 * 1024.0)
                        if progress_cb is not None:
                            progress_cb(100, size_mb, size_mb)
                        return dest_path
        except OSError:
            pass

    proxies = get_active_api_proxies()
    headers = {"User-Agent": f"SilentVoice-TTS/{APP_VERSION}"}

    hasher = hashlib.sha256()
    downloaded = 0
    first_bytes = b""

    try:
        with requests.get(
            release.installer_url,
            headers=headers,
            proxies=proxies,
            stream=True,
            timeout=timeout,
        ) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}")
            total = int(resp.headers.get("Content-Length", 0) or release.installer_size or 0)
            with open(part_path, "wb") as out_f:
                last_cb_ts = 0.0
                last_cb_pct = -1
                for chunk in resp.iter_content(chunk_size=65536):
                    if cancel_event is not None and cancel_event.is_set():
                        raise RuntimeError(tr("update.cancelled", "Завантаження оновлення скасовано."))
                    if not chunk:
                        continue
                    if len(first_bytes) < 2:
                        first_bytes += chunk[:2]
                    out_f.write(chunk)
                    hasher.update(chunk)
                    downloaded += len(chunk)
                    if progress_cb is not None:
                        pct = min(99, int(downloaded * 100 / total)) if total > 0 else 50
                        now = time.monotonic()
                        if pct != last_cb_pct or (now - last_cb_ts) >= 0.05:
                            last_cb_ts = now
                            last_cb_pct = pct
                            dl_mb = downloaded / (1024.0 * 1024.0)
                            tot_mb = total / (1024.0 * 1024.0) if total > 0 else 0.0
                            progress_cb(pct, dl_mb, tot_mb)

        if len(first_bytes) < 2 or first_bytes[:2] != b"MZ":
            raise RuntimeError(
                tr(
                    "update.err.invalid_exe",
                    "Завантажений файл не є виконуваним інсталятором Windows (.exe).",
                )
            )

        if release.sha256:
            actual_hash = hasher.hexdigest().lower()
            expected_hash = release.sha256.strip().lower()
            if actual_hash != expected_hash:
                raise RuntimeError(
                    tr(
                        "update.err.sha256_mismatch",
                        "Контрольна сума SHA-256 інсталятора не збігається! Файл міг пошкодитися під час завантаження.",
                    )
                )

        os.replace(part_path, dest_path)
        if progress_cb is not None:
            dl_mb = downloaded / (1024.0 * 1024.0)
            progress_cb(100, dl_mb, dl_mb)
        return dest_path
    except Exception:
        try:
            part_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def build_installer_args(silent: bool = True, ui_lang: Optional[str] = None) -> List[str]:
    """
    Формує аргументи запуску інсталятора Inno Setup.
    Використовує /SILENT (замість /VERYSILENT), щоб оновлення відбувалося без кліків,
    але користувач бачив вікно прогресу встановлення до автоматичного перезапуску програми.
    """
    lang_code = str(ui_lang or get_ui_language() or "en").strip().lower()
    if lang_code not in _INNO_SUPPORTED_LANGS:
        lang_code = "en"
    if silent:
        return [
            "/SILENT",
            "/SP-",
            "/SUPPRESSMSGBOXES",
            "/NORESTART",
            "/CLOSEAPPLICATIONS",
            "/AUTORELAUNCH=1",
            f"/LANG={lang_code}",
        ]
    return ["/AUTORELAUNCH=1", f"/LANG={lang_code}"]


def launch_update_installer(
    installer_path: Path,
    silent: bool = True,
    ui_lang: Optional[str] = None,
) -> bool:
    """
    Запускає завантажений інсталятор Inno Setup.
    У режимі автооновлення передає /SILENT /SP- /SUPPRESSMSGBOXES /NORESTART
    /CLOSEAPPLICATIONS /AUTORELAUNCH=1 /LANG=<lang>: показує вікно прогресу
    встановлення без зайвих запитань і сам перезапускає SilentVoice після завершення.
    """
    exe = Path(installer_path).resolve()
    if not exe.is_file():
        raise FileNotFoundError(str(exe))

    args_list = build_installer_args(silent=silent, ui_lang=ui_lang)
    args_str = " ".join(args_list)

    if sys.platform == "win32":
        from core.win_api import shell32

        res = shell32.ShellExecuteW(
            None,
            "runas",
            str(exe),
            args_str,
            str(exe.parent),
            1,
        )
        if int(res) > 32:
            return True
        return False

    subprocess.Popen([str(exe)] + args_list, cwd=str(exe.parent))
    return True


def _detect_heading_lang(heading_text: str) -> Optional[str]:
    low = str(heading_text or "").strip().lower()
    for lang, markers in _LANG_SECTION_MARKERS.items():
        if any(m in low for m in markers):
            return lang
    return None


def extract_release_notes_section(notes: str, target_lang: str) -> Tuple[str, bool]:
    """
    Якщо опис релізу містить кілька мовних секцій (`## 🇺🇦 ...`, `## 🇬🇧 ...`),
    повертає `(текст_секції, already_in_target_lang)`.
    - Якщо секція потрібною мовою вже є — повертає її напряму (`already_in_target_lang=True`).
    - Якщо є кілька секцій, але потрібної немає — бере англійську (або першу) як найкраще джерело для перекладу.
    - Якщо секцій по мовах немає — повертає весь текст (`already_in_target_lang=False`).
    """
    raw = str(notes or "").strip()
    if not raw:
        return "", True
    tl = "uk" if str(target_lang or "").strip().lower() in ("uk", "ua") else str(target_lang or "en").strip().lower()

    matches = list(_SECTION_HEADING_RE.finditer(raw))
    if not matches:
        return raw, False

    sections: List[Tuple[Optional[str], str]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        block = raw[start:end].strip()
        # Прибираємо розділювач `---` в кінці секції
        block = re.sub(r"\n+---+\s*$", "", block).strip()
        sections.append((_detect_heading_lang(match.group(1)), block))

    lang_sections = {lang: block for lang, block in sections if lang is not None}
    if not lang_sections:
        return raw, False

    if tl in lang_sections:
        return lang_sections[tl], True

    # Для перекладу на іншу мову краще брати англійську секцію, щоб не перекладати дубль двома мовами
    if "en" in lang_sections:
        return lang_sections["en"], False
    return next(iter(lang_sections.values())), False


def _split_markdown_chunks(text: str, max_chars: int = 1200) -> List[str]:
    chunks: List[str] = []
    cur_lines: List[str] = []
    cur_len = 0
    for line in text.splitlines():
        add_len = len(line) + 1
        if cur_lines and cur_len + add_len > max_chars:
            chunks.append("\n".join(cur_lines))
            cur_lines = [line]
            cur_len = add_len
        else:
            cur_lines.append(line)
            cur_len += add_len
    if cur_lines:
        chunks.append("\n".join(cur_lines))
    return chunks


def localize_release_notes(
    notes: str,
    target_lang: Optional[str] = None,
    timeout: float = 4.5,
) -> str:
    """
    Повертає опис оновлення мовою інтерфейсу:
    1) Якщо в описі вже є секція мовою користувача — повертає саме її.
    2) Інакше автоматично перекладає текст через стандартний перекладач (`Google Translate gtx`),
       зберігаючи структуру рядків; при відсутності інтернету або помилці повертає оригінал.
    """
    raw = str(notes or "").strip()
    if not raw:
        return ""
    tl = "uk" if str(target_lang or get_ui_language() or "en").strip().lower() in ("uk", "ua") else str(target_lang or get_ui_language() or "en").strip().lower()

    selected, already_localized = extract_release_notes_section(raw, tl)
    if already_localized or not selected:
        return selected

    from core.i18n import _standard_translate_text

    translated_chunks: List[str] = []
    for chunk in _split_markdown_chunks(selected, max_chars=420):
        # Не перекладаємо рядки, які складаються лише з хешів/посилань/роздільників
        stripped = chunk.strip()
        if not stripped or re.fullmatch(r"[-=*_`#\s]+", stripped):
            translated_chunks.append(chunk)
            continue
        tr_chunk = _standard_translate_text(chunk, target_lang=tl, source_lang="auto", timeout=timeout)
        if tr_chunk:
            tr_chunk = re.sub(r"(?m)^[—–]\s+", "- ", tr_chunk)
        translated_chunks.append(tr_chunk or chunk)
    return "\n".join(translated_chunks).strip() or selected
