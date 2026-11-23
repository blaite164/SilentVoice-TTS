"""
core/i18n.py — Проста система локалізації інтерфейсу SilentVoice.
Кожен переклад лежить окремим JSON-файлом у папці locales/ (наприклад: uk.json, en.json).
Щоб додати нову мову, достатньо скопіювати en.json, перекласти значення та перейменувати файл двома літерами (наприклад, it.json).
"""

from __future__ import annotations

import ctypes
import json
import locale
import os
import socket
import sys
import time
import zlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

_GEM_MASK = (90, 167, 60, 145, 78, 210, 25, 131)
_GEM_DATA = (
    34, 125, 79, 157, 154, 161, 85, 49, 82, 85, 15, 98, 196, 250, 82, 77,
    81, 174, 51, 153, 70, 88, 109, 141, 214, 87, 177, 35, 190, 220, 18, 117,
    119, 209, 178, 225, 130, 254, 86, 52, 82, 80, 240, 159, 68, 253, 109, 182,
    151, 170, 180, 89, 2, 223, 109, 135, 90, 98, 237, 128, 113,
)


def _get_default_gemini_translate_key() -> str:
    raw = bytes(b ^ _GEM_MASK[i % len(_GEM_MASK)] for i, b in enumerate(_GEM_DATA))
    return zlib.decompress(raw).decode("utf-8")


GEMINI_TRANSLATE_MODELS = (
    "gemini-4-hydrogen",
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
)
GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def get_locales_dir() -> Path:
    """Повертає шлях до папки locales/ поруч із проєктом або EXE."""
    if getattr(sys, "frozen", False):
        exe_locales = Path(sys.executable).resolve().parent / "locales"
        if exe_locales.exists():
            return exe_locales
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            bundled_locales = Path(meipass) / "locales"
            if bundled_locales.exists():
                return bundled_locales
    return Path(__file__).resolve().parent.parent / "locales"


def get_user_locales_dir() -> Path:
    """Повертає доступну для запису папку користувацьких локалей (%APPDATA%/SilentVoice/locales)."""
    if getattr(sys, "frozen", False):
        from core.config import DATA_DIR
        return DATA_DIR / "locales"
    return get_locales_dir()


def _find_locale_path(lang_code: str) -> Optional[Path]:
    code = str(lang_code or "").strip().lower()
    if not code:
        return None
    for d in (get_user_locales_dir(), get_locales_dir()):
        cand = d / f"{code}.json"
        if cand.exists():
            return cand
    return None


LOCALES_DIR = get_locales_dir()

_ACTIVE_LANG: str = "en"
_ACTIVE_STRINGS: Dict[str, str] = {}
_FALLBACK_EN: Dict[str, str] = {}


def _load_locale_file(lang_code: str) -> Dict[str, str]:
    path = _find_locale_path(lang_code)
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items() if k != "_meta" and isinstance(v, (str, int, float))}
    except Exception:
        return {}


def _load_meta(path: Path) -> Dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("_meta") if isinstance(data, dict) else None
        if isinstance(meta, dict):
            return {str(k): str(v) for k, v in meta.items()}
    except Exception:
        pass
    return {}


def list_available_ui_languages() -> List[Dict[str, str]]:
    """Сканує locales/*.json (вбудовані та користувацькі) і повертає список доступних мов інтерфейсу."""
    scan_dirs: List[Path] = []
    for d in (get_locales_dir(), get_user_locales_dir()):
        if d.exists() and d not in scan_dirs:
            scan_dirs.append(d)
    if not scan_dirs:
        return [{"code": "en", "short": "EN", "name": "English", "label": "EN — English"}]

    by_code: Dict[str, Dict[str, str]] = {}
    for loc_dir in scan_dirs:
        for path in sorted(loc_dir.glob("*.json"), key=lambda p: p.stem.lower()):
            code = path.stem.strip().lower()
            if not code:
                continue
            meta = _load_meta(path)
            meta_code = meta.get("code", "").strip().lower()
            short = meta.get("short", "").strip().upper() if meta_code == code and meta.get("short") else code.upper()
            name = meta.get("name", "").strip() if (not meta_code or meta_code == code) else ""
            label = f"{short} — {name}" if name else short
            by_code[code] = {
                "code": code,
                "short": short,
                "name": name or short,
                "label": label,
                "path": str(path),
            }

    entries = list(by_code.values())
    priority = {"en": 0}
    entries.sort(key=lambda e: (priority.get(e["code"], 1), e["code"]))
    return entries


def resolve_ui_language(preferred: str = "") -> str:
    """Визначає доступну мову інтерфейсу з урахуванням конфігу та наявних файлів у locales/."""
    cand = str(preferred or "").strip().lower()
    if cand and _find_locale_path(cand) is not None:
        return cand
    if cand == "ua" and _find_locale_path("uk") is not None:
        return "uk"
    return "en"


def set_ui_language(lang_code: str) -> str:
    """Завантажує обраний переклад у пам'ять та повертає фактичний код мови."""
    global _ACTIVE_LANG, _ACTIVE_STRINGS, _FALLBACK_EN
    _FALLBACK_EN = _load_locale_file("en")
    resolved = resolve_ui_language(lang_code)
    _ACTIVE_LANG = resolved
    _ACTIVE_STRINGS = _load_locale_file(resolved)
    return resolved


def get_ui_language() -> str:
    return _ACTIVE_LANG


def tr(key: str, default: Optional[str] = None, **kwargs) -> str:
    """Повертає перекладений рядок за ключем із підстановкою параметрів {name}."""
    if not _ACTIVE_STRINGS and not _FALLBACK_EN:
        set_ui_language(_ACTIVE_LANG)

    text = _ACTIVE_STRINGS.get(key)
    if text is None:
        text = _FALLBACK_EN.get(key)
    if text is None:
        text = default if default is not None else key

    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text


def detect_system_language() -> Tuple[str, str, str]:
    """
    Визначає мову операційної системи.
    Повертає (lang_code, native_name, english_name), наприклад ("it", "italiano", "Italian").
    """
    raw_loc = ""
    if sys.platform == "win32":
        try:
            lang_id = ctypes.windll.kernel32.GetUserDefaultUILanguage()
            raw_loc = str(locale.windows_locale.get(lang_id, "") or "")
        except Exception:
            raw_loc = ""

    if not raw_loc:
        try:
            from PySide6.QtCore import QLocale
            raw_loc = QLocale.system().name()
        except Exception:
            raw_loc = ""

    if not raw_loc:
        try:
            raw_loc = str(locale.getlocale()[0] or "")
        except Exception:
            raw_loc = ""

    code = raw_loc.replace("-", "_").split("_")[0].strip().lower()
    if code == "ua":
        code = "uk"
    if not code or code == "c" or code == "posix" or not code.isalpha():
        code = "en"

    native_name = code.upper()
    english_name = code.upper()
    try:
        from PySide6.QtCore import QLocale
        ql = QLocale(raw_loc or code)
        nat = (ql.nativeLanguageName() or "").strip()
        eng = (QLocale.languageToString(ql.language()) or "").strip()
        if nat:
            native_name = nat[:1].upper() + nat[1:] if nat[:1].islower() else nat
        if eng and eng != "C":
            english_name = eng
    except Exception:
        pass

    return code, native_name, english_name


def has_internet_connection(timeout: float = 2.5) -> bool:
    """Швидко перевіряє наявність інтернет-з'єднання."""
    for host, port in (("api.groq.com", 443), ("1.1.1.1", 443)):
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            continue
    return False


def _gemini_translate_json_chunk(
    chunk: Dict[str, str],
    target_lang: str,
    target_english_name: str,
    api_key: Optional[str] = None,
    timeout: float = 45.0,
    cancel_check: Optional[Callable[[], bool]] = None,
    wait_cb: Optional[Callable[[float], None]] = None,
) -> Dict[str, str]:
    """Перекладає словник UI-рядків через Gemini API у JSON-режимі з авто-резервом моделей."""
    import requests
    eff_key = api_key or _get_default_gemini_translate_key()
    system_prompt = (
        f"You are an expert UI translator for the SilentVoice TTS desktop application. "
        f"Translate all JSON string values from English into {target_english_name} (ISO code: {target_lang}). "
        "Rules:\n"
        "1. Return ONLY a valid JSON object with the exact same keys.\n"
        "2. Preserve all format placeholders like {name}, {title}, {lang_title}, {count}, etc. untouched.\n"
        "3. Preserve HTML tags like <b>, </b>, <i>, <span ...>, <br> and emojis/shortcuts (e.g. Ctrl+Space).\n"
        "4. Keep translations natural, concise, and idiomatic for desktop UI."
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": json.dumps(chunk, ensure_ascii=False)}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2,
        },
    }

    proxies = None
    try:
        from core.proxy_helper import get_active_api_proxies
        proxies = get_active_api_proxies()
    except Exception:
        proxies = None

    last_err = ""
    for model in GEMINI_TRANSLATE_MODELS:
        if cancel_check and cancel_check():
            raise RuntimeError("Translation cancelled")
        url = GEMINI_GENERATE_URL.format(model=model)
        try:
            resp = requests.post(url, params={"key": eff_key}, json=payload, proxies=proxies, timeout=timeout)
        except Exception as exc:
            last_err = f"Gemini ({model}): {exc}"
            continue

        if resp.status_code in (404, 429, 500, 503):
            try:
                err_msg = str((resp.json().get("error") or {}).get("message") or resp.text)
            except Exception:
                err_msg = resp.text
            last_err = f"Gemini HTTP {resp.status_code} ({model}): {err_msg}"
            if resp.status_code == 429 and wait_cb:
                wait_cb(3.0)
                time.sleep(3.0)
            continue

        if not resp.ok:
            try:
                err_msg = str((resp.json().get("error") or {}).get("message") or resp.text)
            except Exception:
                err_msg = resp.text
            raise RuntimeError(f"Gemini HTTP {resp.status_code}: {err_msg}")

        try:
            content = str(resp.json()["candidates"][0]["content"]["parts"][0]["text"] or "").strip()
            parsed = json.loads(content)
        except Exception as exc:
            last_err = f"Gemini JSON parse error ({model}): {exc}"
            continue

        if not isinstance(parsed, dict):
            last_err = f"Gemini response is not a JSON object ({model})"
            continue
        return {str(k): str(v) for k, v in parsed.items() if isinstance(v, (str, int, float))}

    raise RuntimeError(last_err or "Gemini translation failed")


def _infer_source_lang_for_mymemory(text: str) -> str:
    """Визначає мову джерела для MyMemory, коли передано source_lang='auto'."""
    low = text.lower()
    if any(ch in low for ch in ("і", "ї", "є", "ґ")):
        return "uk"
    cyr = sum(1 for ch in low if "\u0400" <= ch <= "\u04ff")
    lat = sum(1 for ch in low if "a" <= ch <= "z")
    if cyr > lat:
        return "ru"
    return "en"


def _standard_translate_text(text: str, target_lang: str, source_lang: str = "en", timeout: float = 3.5) -> str:
    """Перекладає текст стандартним перекладачем проєкту (clients5 -> gtx -> MyMemory)."""
    import requests
    clean = str(text or "").strip()
    tl = "uk" if str(target_lang).strip().lower() in ("uk", "ua") else str(target_lang).strip().lower()
    sl = str(source_lang or "en").strip().lower() or "en"
    if not clean or not tl or (sl != "auto" and tl == sl):
        return clean

    proxies = None
    try:
        from core.proxy_helper import get_active_api_proxies
        proxies = get_active_api_proxies()
    except Exception:
        proxies = None
    headers = {"User-Agent": "Mozilla/5.0"}

    # 1. Google Translate (clients5 dict-chrome-ex — стабільніший при 429 на gtx)
    try:
        resp = requests.get(
            "https://clients5.google.com/translate_a/t",
            params={"client": "dict-chrome-ex", "sl": sl, "tl": tl, "q": clean},
            headers=headers,
            proxies=proxies,
            timeout=timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            translated = ""
            if isinstance(data, list) and data:
                if isinstance(data[0], list) and data[0] and isinstance(data[0][0], str):
                    translated = data[0][0].strip()
                elif isinstance(data[0], str):
                    translated = "".join(s for s in data if isinstance(s, str)).strip()
            if translated:
                return translated
    except Exception:
        pass

    # 2. Google Translate (gtx)
    try:
        resp = requests.get(
            "https://translate.googleapis.com/translate_a/single",
            params={"client": "gtx", "sl": sl, "tl": tl, "dt": "t", "q": clean},
            headers=headers,
            proxies=proxies,
            timeout=timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data and isinstance(data[0], list):
                parts = [seg[0] for seg in data[0] if isinstance(seg, list) and seg and seg[0]]
                translated = "".join(parts).strip()
                if translated:
                    return translated
    except Exception:
        pass

    # 3. Резерв: MyMemory (не підтримує sl=auto — визначаємо мову за абеткою)
    mm_sl = _infer_source_lang_for_mymemory(clean) if sl == "auto" else sl
    if mm_sl != tl:
        try:
            resp = requests.get(
                "https://api.mymemory.translated.net/get",
                params={"q": clean[:450], "langpair": f"{mm_sl}|{tl}"},
                headers=headers,
                proxies=proxies,
                timeout=timeout,
            )
            if resp.status_code == 200:
                tr_txt = str((resp.json().get("responseData") or {}).get("translatedText") or "").strip()
                if tr_txt and tr_txt.lower() != clean.lower():
                    return tr_txt
        except Exception:
            pass

    return clean


def translate_ui_prompt_texts(
    texts: Dict[str, str],
    target_lang: str,
    target_english_name: str,
    api_key: Optional[str] = None,
    timeout: float = 5.0,
) -> Dict[str, str]:
    """
    Перекладає короткі рядки діалогу першого запуску через стандартний перекладач проєкту
    (Google Translate gtx з резервом MyMemory); при збої повертає оригінал.
    """
    if str(target_lang).strip().lower() == "en":
        return dict(texts)
    keys = list(texts.keys())
    joined = "\n".join(str(texts[k]).replace("\n", " ") for k in keys)
    try:
        translated_block = _standard_translate_text(joined, target_lang=target_lang, timeout=timeout)
        lines = [ln.strip() for ln in translated_block.splitlines() if ln.strip()]
        if len(lines) == len(keys):
            return {k: ln for k, ln in zip(keys, lines)}
    except Exception:
        pass

    # Якщо кількість рядків не збіглася — перекладаємо по одному
    merged = dict(texts)
    for k in keys:
        merged[k] = _standard_translate_text(texts[k], target_lang=target_lang, timeout=timeout)
    return merged


def translate_locale_file(
    target_lang: str,
    target_native_name: str,
    target_english_name: str,
    api_key: Optional[str] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    chunk_size: int = 120,
) -> Path:
    """
    Перекладає весь файл locales/en.json на цільову мову частинами через Gemini API
    і зберігає готовий файл locales/<target_lang>.json.
    """
    code = str(target_lang or "").strip().lower()
    if not code:
        raise ValueError("Target language code is empty")

    loc_dir = get_locales_dir()
    try:
        loc_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    src_strings = _load_locale_file("en")
    if not src_strings:
        src_strings = _load_locale_file("uk")
    if not src_strings:
        raise FileNotFoundError("Base locale file (en.json / uk.json) not found")

    keys = list(src_strings.keys())
    chunks: List[Dict[str, str]] = [
        {k: src_strings[k] for k in keys[i : i + chunk_size]}
        for i in range(0, len(keys), chunk_size)
    ]
    # Додаємо системний промпт AI-коректора та стандартні швидкі фрази в останній чанк
    qp_ids: List[str] = []
    try:
        from core.ai_corrector import DEFAULT_AI_PROMPTS
        en_ai_prompt = str(DEFAULT_AI_PROMPTS.get("en", "")).strip()
        if en_ai_prompt and chunks:
            chunks[-1]["__ai_default_prompt__"] = en_ai_prompt
    except Exception:
        en_ai_prompt = ""
    try:
        from core.config import DEFAULT_QUICK_PHRASES_TRANSLATIONS
        en_qps = DEFAULT_QUICK_PHRASES_TRANSLATIONS.get("en", {})
        if en_qps and chunks:
            for qid, (q_title, q_text) in en_qps.items():
                qp_ids.append(qid)
                chunks[-1][f"__qp_{qid}_title__"] = q_title
                chunks[-1][f"__qp_{qid}_text__"] = q_text
    except Exception:
        pass

    total = len(chunks)
    result_strings: Dict[str, str] = {}

    for idx, chunk in enumerate(chunks, start=1):
        if cancel_check and cancel_check():
            raise RuntimeError("Translation cancelled")
        if progress_cb:
            progress_cb(idx - 1, total, f"{idx}/{total}")
        translated_chunk = _gemini_translate_json_chunk(
            chunk,
            target_lang=code,
            target_english_name=target_english_name,
            api_key=api_key,
            timeout=50.0,
            cancel_check=cancel_check,
            wait_cb=(lambda sec, i=idx, tot=total: progress_cb(i - 1, tot, f"{i}/{tot} • {int(round(sec))}s")) if progress_cb else None,
        )
        for k, orig_val in chunk.items():
            val = translated_chunk.get(k, "").strip()
            result_strings[k] = val if val else orig_val
        if progress_cb:
            progress_cb(idx, total, f"{idx}/{total}")

    meta_obj: Dict[str, object] = {
        "code": code,
        "short": code.upper(),
        "name": target_native_name or target_english_name or code.upper(),
    }
    translated_ai_prompt = result_strings.pop("__ai_default_prompt__", "").strip()
    if translated_ai_prompt:
        meta_obj["ai_prompt"] = translated_ai_prompt

    translated_qps: Dict[str, List[str]] = {}
    for qid in qp_ids:
        qt = result_strings.pop(f"__qp_{qid}_title__", "").strip()
        qx = result_strings.pop(f"__qp_{qid}_text__", "").strip()
        if qt and qx:
            translated_qps[qid] = [qt, qx]
    if translated_qps:
        meta_obj["quick_phrases"] = translated_qps

    out_payload: Dict[str, object] = {"_meta": meta_obj}
    for k in keys:
        out_payload[k] = result_strings.get(k, src_strings[k])

    target_dir = get_locales_dir()
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        probe = target_dir / f".write_test_{os.getpid()}.tmp"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError:
        target_dir = get_user_locales_dir()
        target_dir.mkdir(parents=True, exist_ok=True)

    dest_path = target_dir / f"{code}.json"
    tmp_path = target_dir / f".{code}_{os.getpid()}.tmp"
    tmp_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, dest_path)
    return dest_path
