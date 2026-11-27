"""
core/ai_corrector.py — AI-коректор орфографії, пунктуації та помилок розкладки перед TTS та OBS.
Підтримує: Groq, Google Gemini, OpenAI, OpenRouter, Ollama (локально), з головним та запасним сервісом.
"""

from __future__ import annotations

import json
import re
import sys
import threading
from typing import Any, Dict, List, Optional

import requests

from core.config import config
from core.i18n import tr
from core.proxy_helper import get_active_api_proxies

AI_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "groq": {
        "provider": "groq",
        "title": "Groq API",
        "default_model": "openai/gpt-oss-20b",
        "needs_key": True,
        "guide_title": "⚡ Як отримати API ключ для Groq (безкоштовно і дуже швидко)",
        "guide_html": (
            "1. Перейдіть на <b>console.groq.com/keys</b> та увійдіть у свій акаунт.\n"
            "2. Натисніть <b>Create API Key</b>, задайте будь-яку назву та скопіюйте ключ (починається на <code>gsk_...</code>).\n"
            "3. Вставте ключ у поле <b>API ключ</b> у блоці Groq API.\n"
            "4. <i>Порада:</i> Groq працює найшвидше (зазвичай 150–300 мс) і має щедрий безкоштовний ліміт."
        ),
        "guide_btn": "Відкрити Groq Console",
        "guide_url": "https://console.groq.com/keys",
    },
    "gemini": {
        "provider": "gemini",
        "title": "Google Gemini API",
        "default_model": "gemini-flash-lite-latest",
        "needs_key": True,
        "guide_title": "✨ Як отримати API ключ для Google Gemini",
        "guide_html": (
            "1. Відкрийте <b>aistudio.google.com/apikey</b> та увійдіть у Google-акаунт.\n"
            "2. Натисніть <b>Create API key</b> та скопіюйте ключ (починається на <code>AIza...</code>).\n"
            "3. Вставте скопійований ключ у поле <b>API ключ</b> у блоці Google Gemini API."
        ),
        "guide_btn": "Відкрити Google AI Studio",
        "guide_url": "https://aistudio.google.com/apikey",
    },
    "openai": {
        "provider": "openai",
        "title": "OpenAI API",
        "default_model": "gpt-4o-mini",
        "needs_key": True,
        "guide_title": "🤖 Як отримати API ключ для OpenAI",
        "guide_html": (
            "1. Відкрийте <b>platform.openai.com/api-keys</b>.\n"
            "2. Натисніть <b>+ Create new secret key</b> та скопіюйте ключ (<code>sk-...</code>).\n"
            "3. Вставте ключ у поле <b>API ключ</b> у блоці OpenAI API."
        ),
        "guide_btn": "Відкрити OpenAI API Keys",
        "guide_url": "https://platform.openai.com/api-keys",
    },
    "openrouter": {
        "provider": "openrouter",
        "title": "OpenRouter API",
        "default_model": "openai/gpt-4o-mini",
        "needs_key": True,
        "guide_title": "🌐 Як отримати API ключ для OpenRouter",
        "guide_html": (
            "1. Перейдіть на <b>openrouter.ai/keys</b> та увійдіть в акаунт.\n"
            "2. Натисніть <b>Create Key</b> та скопіюйте ключ (<code>sk-or-...</code>).\n"
            "3. Вставте ключ у поле <b>API ключ</b> у блоці OpenRouter API."
        ),
        "guide_btn": "Відкрити OpenRouter Keys",
        "guide_url": "https://openrouter.ai/keys",
    },
    "deepseek": {
        "provider": "deepseek",
        "title": "DeepSeek API",
        "default_model": "deepseek-chat",
        "needs_key": True,
        "guide_title": "🐳 Як отримати API ключ для DeepSeek",
        "guide_html": (
            "1. Перейдіть на <b>platform.deepseek.com/api_keys</b> та увійдіть в акаунт.\n"
            "2. Натисніть <b>Create new API key</b> та скопіюйте ключ (починається на <code>sk-...</code>).\n"
            "3. Вставте ключ у поле <b>API ключ</b> у блоці DeepSeek API."
        ),
        "guide_btn": "Відкрити DeepSeek Platform",
        "guide_url": "https://platform.deepseek.com/api_keys",
    },
    "zai": {
        "provider": "zai",
        "title": "Z.ai API (GLM)",
        "default_model": "glm-4.5-flash",
        "needs_key": True,
        "guide_title": "⚡ Як отримати API ключ для Z.ai (GLM)",
        "guide_html": (
            "1. Перейдіть на <b>z.ai/manage-apikey/apikey-list</b> (або <b>chat.z.ai</b> → API) та увійдіть в акаунт.\n"
            "2. Створіть новий API ключ і скопіюйте його.\n"
            "3. Вставте ключ у поле <b>API ключ</b> у блоці Z.ai API (GLM)."
        ),
        "guide_btn": "Відкрити Z.ai API Keys",
        "guide_url": "https://z.ai/manage-apikey/apikey-list",
    },
    "ollama": {
        "provider": "ollama",
        "title": "Ollama (та інші локальні моделі)",
        "default_model": "gemma3:4b",
        "needs_key": False,
        "guide_title": "🖥️ Ollama: коротка інструкція",
        "guide_html": (
            "1. Якщо Ollama ще не встановлена, натисніть <b>Встановити Ollama</b>.\n"
            "2. Щоб знайти іншу модель, відкрийте бібліотеку за кнопкою нижче. На сторінці моделі скопіюйте її назву з тегом, наприклад <code>gemma3:4b</code>, і вставте в поле моделі.\n"
            "3. Натисніть <b>Завантажити модель</b> і стежте за перебігом у картці. Позначка «встановлено» з’явиться після підтвердження завершення завантаження.\n"
            "Для початку обирайте невелику модель: вона потребує менше пам’яті. Більша може бути розумнішою, але працювати повільніше. Модель виправляє текст, а голос озвучення не змінює.\n"
            "Модель із іншої програми підключається через кнопку <b>⋯</b> праворуч від списку. Спершу відкрийте її в LM Studio, llama.cpp чи іншій програмі та увімкніть там сервер.\n"
            "Якщо Ollama встановлена, SilentVoice запускає її автоматично, коли ви вмикаєте AI-коректор. LM Studio, llama.cpp та інші програми потрібно відкрити окремо й увімкнути в них сервер. Команди вводити не потрібно."
        ),
        "guide_btn": "Відкрити сайт Ollama",
        "guide_url": "https://ollama.com/download",
        "guide_models_btn": "Переглянути моделі",
        "guide_models_url": "https://ollama.com/library",
    },
    "local_api": {
        "provider": "local_api",
        "title": "Інший локальний сервер",
        "default_model": "",
        "default_base_url": "http://localhost:1234/v1",
        "needs_key": False,
        "guide_title": "🖥️ Як підключити іншу локальну модель",
        "guide_html": (
            "Цей спосіб підходить для Ollama, LM Studio, llama.cpp та інших програм, якщо вони надають OpenAI-сумісний API. Для звичайної Ollama простіше користуватися карткою Ollama вище.\n"
            "Файли <b>GGUF</b>, <b>BF16</b> та інші — це способи зберігання моделей. Сам файл на диску ще не можна вибрати тут.\n"
            "1. Відкрийте файл у програмі, яка підтримує його формат, наприклад <b>LM Studio</b> або <b>llama.cpp</b> для GGUF.\n"
            "2. Увімкніть у цій програмі локальний сервер. Він має підтримувати OpenAI-сумісний API — стандартний спосіб передавати запити між програмами.\n"
            "3. Вставте адресу сервера в це поле та натисніть <b>Оновити моделі</b>. Наприклад, адреса LM Studio зазвичай <code>http://localhost:1234/v1</code>, а llama.cpp — <code>http://localhost:8080/v1</code>.\n"
            "У списку з’являться моделі, які сервер зараз надає. Кнопка <b>⋯</b> біля списку відкриває це вікно для пошуку моделей в іншій програмі."
            "Коли модель з’явиться, оберіть її. Формат має підтримувати програма-сервер; вводити команди в SilentVoice не потрібно."
        ),
        "guide_btn": "Документація LM Studio",
        "guide_url": "https://lmstudio.ai/docs/developer/core/server",
        "guide_models_btn": "Документація llama.cpp",
        "guide_models_url": "https://github.com/ggml-org/llama.cpp/tree/master/tools/server",
    },
}


def normalize_ollama_model_name(name: str) -> str:
    """Normalize Ollama's implicit :latest tag without merging model variants."""
    value = str(name or "").strip().casefold()
    if not value:
        return ""
    last_part = value.rsplit("/", 1)[-1]
    if ":" not in last_part:
        value += ":latest"
    return value


def ollama_model_family_key(name: str) -> str:
    """Group base and instruct tags for the installed-family count only."""
    value = normalize_ollama_model_name(name)
    if not value:
        return ""
    parts = value.rsplit(":", 1)
    model_part, tag = parts if len(parts) == 2 else (value, "latest")
    model_part = re.sub(r"(?i)[-_.]instruct(?=$|[-_.])", "", model_part)
    tag = re.sub(r"(?i)[-_.]instruct(?=$|[-_.])", "", tag)
    return f"{model_part}:{tag}"


def get_ai_item_title(item: Dict[str, Any]) -> str:
    """Return the current-language display name for a saved AI service."""
    provider = str(item.get("provider", "")).strip()
    spec = AI_PROVIDERS.get(provider, {})
    saved_title = str(item.get("title", "")).strip()
    if provider not in ("ollama", "local_api"):
        return saved_title or str(spec.get("title", provider))

    localized_title = tr(f"ai.prov.{provider}.title", str(spec.get("title", provider)))
    if provider == "ollama":
        suffix = ""
        if " #" in saved_title:
            maybe_suffix = saved_title.rsplit(" #", 1)[-1]
            if maybe_suffix.isdigit():
                suffix = f" #{maybe_suffix}"
        return f"{localized_title}{suffix}"

    # Managed backends are selected and started by SilentVoice. Naming them
    # after the actual app avoids suggesting that requests go to a second server.
    managed_backend = str(item.get("managed_backend", "")).strip().lower()
    if managed_backend == "lmstudio":
        return "LM Studio"
    if managed_backend == "llama_cpp":
        return "llama.cpp"

    # Manually configured OpenAI-compatible endpoints keep their saved label.
    for separator in (" — ", " – ", " - "):
        if separator in saved_title:
            backend_name = saved_title.split(separator, 1)[1].strip()
            if backend_name:
                return f"{localized_title} — {backend_name}"
    return localized_title

DEFAULT_AI_PROMPTS: Dict[str, str] = {
    "uk": (
        "Делікатний коректор українського тексту. Правила:\n"
        "1. Виправляй ЛИШЕ реальні орфографічні помилки, описки та явну пунктуацію. НЕ перефразовуй, не підбирай синоніми й не змінюй час/форму слів (наприклад, «говоритиме» залишай як є).\n"
        "2. Виправляй помилки розкладки (ghbdsn -> привіт) і прибирай лише зайві пробіли.\n"
        "3. НЕ видаляй і не змінюй числа, окремі цифри та символи (1, 100, 50%).\n"
        "Якщо текст правильний — поверни його без змін. Виведи ТІЛЬКИ текст, без лапок і пояснень."
    ),
    "ru": (
        "Деликатный корректор русского текста. Правила:\n"
        "1. Исправляй ТОЛЬКО реальные орфографические ошибки, опечатки и явную пунктуацию. НЕ перефразируй, не меняй слова на синонимы и не меняй время/форму глаголов.\n"
        "2. Исправляй ошибки раскладки (ghbdtn -> привет) и убирай только лишние пробелы.\n"
        "3. НЕ удаляй и не изменяй числа, отдельные цифры и символы (1, 100, 50%).\n"
        "Если текст правильный — верни его без изменений. Выведи ТОЛЬКО текст, без кавычек и пояснений."
    ),
    "en": (
        "Minimal English proofreader. Rules:\n"
        "1. Fix ONLY actual spelling errors, typos, and clear punctuation mistakes. Do NOT paraphrase, swap synonyms, or change verb tenses or valid phrasing.\n"
        "2. Fix wrong-layout typos and remove only redundant spaces.\n"
        "3. Keep ALL numbers, single digits, and symbols (1, 100, 50%, $10) intact—do NOT delete or rewrite them.\n"
        "If the text is already valid, return it unchanged. Output ONLY the text, without quotes or comments."
    ),
    "pl": (
        "Delikatny korektor tekstu polskiego. Zasady:\n"
        "1. Poprawiaj TYLKO realne błędy ortograficzne, literówki i wyraźną interpunkcję. NIE parafrazuj, nie zamieniaj słów na synonimy i nie zmieniaj czasu ani formy czasowników.\n"
        "2. Poprawiaj błędy układu klawiatury i usuwaj wyłącznie zbędne spacje.\n"
        "3. NIE usuwaj ani nie zmieniaj liczb, pojedynczych cyfr i symboli (1, 100, 50%).\n"
        "Jeśli tekst jest poprawny, zwróć go bez zmian. Wypisz TYLKO tekst, bez cudzysłowów i komentarzy."
    ),
    "de": (
        "Behutsamer deutscher Textkorrektor. Regeln:\n"
        "1. Korrigiere NUR echte Rechtschreibfehler, Tippfehler und klare Zeichensetzung. Paraphrasiere NICHT, tausche keine Wörter aus und ändere keine Zeitformen.\n"
        "2. Korrigiere Tastaturlayoutfehler und entferne nur überflüssige Leerzeichen.\n"
        "3. Behalte ALLE Zahlen, einzelnen Ziffern und Symbole (1, 100, 50%, 10€) unverändert bei.\n"
        "Wenn der Text bereits korrekt ist, gib ihn unverändert zurück. Gib NUR den Text aus, ohne Anführungszeichen oder Kommentare."
    ),
    "fr": (
        "Correcteur discret de texte français. Règles :\n"
        "1. Corrige UNIQUEMENT les vraies fautes d'orthographe, coquilles et ponctuations évidentes. Ne reformule PAS, ne remplace pas les mots et ne change pas les temps verbaux.\n"
        "2. Corrige les erreurs de disposition clavier et supprime uniquement les espaces superflus.\n"
        "3. Conserve TOUS les nombres, chiffres isolés et symboles (1, 100, 50%, 10€) tels quels sans les supprimer.\n"
        "Si le texte est déjà correct, renvoie-le tel quel. Affiche UNIQUEMENT le texte, sans guillemets ni commentaires."
    ),
    "es": (
        "Corrector sutil de texto en español. Reglas:\n"
        "1. Corrige SOLO faltas reales de ortografía, erratas y puntuación clara. NO parafrasees, no cambies palabras por sinónimos ni alteres los tiempos verbales.\n"
        "2. Corrige errores de distribución de teclado y elimina solo espacios sobrantes.\n"
        "3. Conserva TODOS los números, dígitos sueltos y símbolos (1, 100, 50%, $10) intactos sin eliminarlos.\n"
        "Si el texto ya es correcto, devuélvelo sin cambios. Devuelve SOLO el texto, sin comillas ni explicaciones."
    ),
}

DEFAULT_AI_PROMPT = DEFAULT_AI_PROMPTS["uk"]


def get_default_ai_prompt(lang: Optional[str] = None) -> str:
    code = (lang or str(config.get("app_language", "en"))).strip().lower() or "en"
    if code in DEFAULT_AI_PROMPTS:
        return DEFAULT_AI_PROMPTS[code]
    try:
        from core.i18n import _find_locale_path
        loc_path = _find_locale_path(code)
        if loc_path is not None and loc_path.exists():
            data = json.loads(loc_path.read_text(encoding="utf-8"))
            meta = data.get("_meta") if isinstance(data, dict) else None
            if isinstance(meta, dict):
                tr_prompt = str(meta.get("ai_prompt", "")).strip()
                if tr_prompt:
                    return tr_prompt
    except Exception:
        pass
    return DEFAULT_AI_PROMPTS["en"]


def get_effective_ai_prompt(lang: Optional[str] = None) -> str:
    code = (lang or str(config.get("app_language", "en"))).strip().lower() or "en"
    prompts_map = config.get("ai_spellcheck_prompts", {})
    if isinstance(prompts_map, dict):
        custom = str(prompts_map.get(code, "")).strip()
        if custom:
            return custom
    legacy = str(config.get("ai_spellcheck_prompt", "")).strip()
    if legacy and code == "uk":
        return legacy
    return get_default_ai_prompt(code)


_AI_HTTP_LOCAL = threading.local()
_LAST_AI_EDIT_LOCK = threading.Lock()
_LAST_AI_EDIT: Dict[str, Any] = {}


def get_last_ai_edit_result() -> Dict[str, Any]:
    """Return the latest spellcheck input/output captured from the normal editor path."""
    with _LAST_AI_EDIT_LOCK:
        return dict(_LAST_AI_EDIT)


def _get_ai_http_session() -> requests.Session:
    sess = getattr(_AI_HTTP_LOCAL, "session", None)
    if sess is None:
        sess = requests.Session()
        _AI_HTTP_LOCAL.session = sess
    return sess


def _clean_ai_output(text: str) -> str:
    import re as _re
    cleaned = _re.sub(r"<think>.*?</think>", "", str(text or ""), flags=_re.DOTALL | _re.IGNORECASE)
    return cleaned.strip()


def _openai_compatible_base_url(base_url: str, default: str = "") -> str:
    base = str(base_url or default).strip().rstrip("/")
    if not base:
        raise RuntimeError(tr("ai.err.local_url", "Вкажіть адресу локального сервера."))
    return base if base.lower().endswith("/v1") else f"{base}/v1"


def _friendly_ai_api_error(title: str, response: requests.Response) -> str:
    """Convert raw provider errors into actionable text without echoing secrets."""
    code = int(response.status_code)
    try:
        data = response.json()
        error = data.get("error", data) if isinstance(data, dict) else data
        if isinstance(error, dict):
            hint = " ".join(
                str(error.get(key, ""))
                for key in ("message", "type", "code", "status")
            ).lower()
        else:
            hint = str(error).lower()
    except Exception:
        hint = str(getattr(response, "text", "")).lower()

    if code == 401 or any(token in hint for token in (
        "authentication", "unauthorized", "invalid api key", "api key is invalid",
        "incorrect api key", "invalid_api_key", "authentication_error",
    )):
        return tr(
            "ai.err.authentication",
            "{title}: API-ключ неправильний або неактивний. Перевірте його в налаштуваннях сервісу.",
            title=title,
        )
    if code in (402, 429) or any(token in hint for token in (
        "quota", "rate limit", "too many requests", "billing", "insufficient credit",
    )):
        return tr(
            "ai.err.quota",
            "{title}: вичерпано ліміт запитів або коштів. Перевірте ліміти акаунта.",
            title=title,
        )
    if code == 403:
        return tr(
            "ai.err.access",
            "{title}: акаунт не має доступу до цієї моделі або функції.",
            title=title,
        )
    if code == 404 and "model" in hint:
        return tr(
            "ai.err.model",
            "{title}: вибраної моделі не знайдено. Оновіть список моделей і виберіть доступну.",
            title=title,
        )
    if code >= 500:
        return tr(
            "ai.err.temporary",
            "{title}: сервіс тимчасово недоступний. Спробуйте пізніше.",
            title=title,
        )
    if code == 400:
        return tr(
            "ai.err.bad_request",
            "{title}: сервіс не прийняв запит. Перевірте модель і її налаштування.",
            title=title,
        )
    return tr(
        "ai.err.generic",
        "{title}: запит не виконано (код {code}). Перевірте налаштування сервісу.",
        title=title,
        code=code,
    )


def _call_ai_item(item: Dict[str, Any], text: str, system_prompt: str, timeout: float = 8.0) -> str:
    provider = str(item.get("provider", "")).strip()
    api_key = str(item.get("api_key", "")).strip()
    spec = AI_PROVIDERS.get(provider)
    if not spec:
        raise RuntimeError(tr("ai.err.unknown_provider", "Невідомий AI-провайдер: {provider}", provider=provider))

    model = str(item.get("model") or spec["default_model"]).strip() or spec["default_model"]
    if spec.get("needs_key", True) and not api_key:
        raise RuntimeError(tr("ai.err.no_api_key", "Не вказано API ключ для {title}", title=spec["title"]))

    proxies = get_active_api_proxies()
    sess = _get_ai_http_session()
    # Corrected text is usually close to the input length. Bound generated
    # output for providers where the request format exposes a token limit.
    output_token_limit = min(1024, max(256, len(str(text)) // 2 + 64))

    if provider == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {"temperature": 0.1},
        }
        resp = sess.post(url, params={"key": api_key}, json=payload, proxies=proxies, timeout=timeout)
        if resp.status_code != 200:
            raise RuntimeError(_friendly_ai_api_error(spec["title"], resp))
        data = resp.json()
        candidates = data.get("candidates") or []
        parts = (((candidates[0] or {}).get("content") or {}).get("parts") or []) if candidates else []
        out = _clean_ai_output("".join(str(p.get("text", "")) for p in parts))
        if not out:
            raise RuntimeError(tr("ai.err.gemini_empty", "Gemini повернув порожню відповідь"))
        return out

    if provider == "ollama":
        base_url = str(item.get("base_url") or "http://localhost:11434").strip().rstrip("/")
        url = f"{base_url}/api/chat"
        payload = {
            "model": model,
            "stream": False,
            "options": {"temperature": 0.1},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        }
        # Qwen3.5 accepts an explicit thinking switch. Qwen2.5 and Gemma 3
        # reject the field, so omit it for those models.
        if model.lower().startswith("qwen3.5"):
            payload["think"] = False
        resp = sess.post(url, json=payload, timeout=timeout)
        if resp.status_code != 200:
            raise RuntimeError(_friendly_ai_api_error(spec["title"], resp))
        data = resp.json()
        message = data.get("message") or {}
        out = _clean_ai_output(message.get("content", ""))
        if not out:
            raise RuntimeError(tr("ai.err.ollama_empty", "Ollama повернула порожню відповідь"))
        return out

    if provider == "local_api":
        base_url = _openai_compatible_base_url(
            str(item.get("base_url") or ""), str(spec.get("default_base_url", ""))
        )
        url = f"{base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "temperature": 0.1,
            "max_tokens": output_token_limit,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        }
        resp = sess.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code != 200:
            raise RuntimeError(_friendly_ai_api_error(spec["title"], resp))
        choices = resp.json().get("choices") or []
        out = _clean_ai_output(((choices[0] or {}).get("message") or {}).get("content", "")) if choices else ""
        if not out:
            raise RuntimeError(tr("ai.err.provider_empty", "{title} повернув порожню відповідь", title=spec["title"]))
        return out

    # OpenAI-сумісні API: groq, openai, openrouter
    endpoints = {
        "groq": "https://api.groq.com/openai/v1/chat/completions",
        "openai": "https://api.openai.com/v1/chat/completions",
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
        "deepseek": "https://api.deepseek.com/chat/completions",
        "zai": "https://api.z.ai/api/paas/v4/chat/completions",
    }
    url = endpoints[provider]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
    }
    if provider == "groq" and "gpt-oss-120b" in model.lower():
        payload["reasoning_effort"] = "low"
    resp = sess.post(url, headers=headers, json=payload, proxies=proxies, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(_friendly_ai_api_error(spec["title"], resp))
    choices = resp.json().get("choices") or []
    out = _clean_ai_output(((choices[0] or {}).get("message") or {}).get("content", "")) if choices else ""
    if not out:
        raise RuntimeError(tr("ai.err.provider_empty", "{title} повернув порожню відповідь", title=spec["title"]))
    return out


def fetch_ai_provider_models(provider: str, api_key: str = "", base_url: str = "") -> List[str]:
    """Отримує список доступних LLM-моделей через API обраного провайдера."""
    prov = str(provider or "").strip().lower()
    spec = AI_PROVIDERS.get(prov)
    if not spec:
        raise RuntimeError(tr("ai.err.unknown_provider", "Невідомий AI-провайдер: {provider}", provider=prov))

    import re as _re
    clean_key = _re.sub(r"^bearer\s+", "", str(api_key or "").strip().strip("\"'"), flags=_re.IGNORECASE).strip()
    if spec.get("needs_key", True) and not clean_key:
        raise RuntimeError(tr("ai.err.enter_api_key", "Спочатку введіть API ключ для {title}.", title=spec["title"]))

    proxies = get_active_api_proxies()
    models: List[str] = []

    if prov == "ollama":
        base = (base_url or "http://localhost:11434").strip().rstrip("/")
        resp = requests.get(f"{base}/api/tags", timeout=8)
        if resp.status_code != 200:
            raise RuntimeError(_friendly_ai_api_error(spec["title"], resp))
        seen_models: set[str] = set()
        for m in resp.json().get("models", []) or []:
            if isinstance(m, dict) and m.get("name"):
                mname = str(m["name"]).strip()
                low = mname.lower()
                if any(b in low for b in ("embed", "bge-", "all-minilm")):
                    continue
                model_key = normalize_ollama_model_name(mname)
                if not model_key or model_key in seen_models:
                    continue
                seen_models.add(model_key)
                models.append(mname)
    elif prov == "local_api":
        base = _openai_compatible_base_url(base_url, str(spec.get("default_base_url", "")))
        headers = {"Authorization": f"Bearer {clean_key}"} if clean_key else {}
        resp = requests.get(f"{base}/models", headers=headers, timeout=12)
        if resp.status_code != 200:
            raise RuntimeError(_friendly_ai_api_error(spec["title"], resp))
        raw_list = resp.json().get("data", []) if isinstance(resp.json(), dict) else []
        for model_info in raw_list:
            if isinstance(model_info, dict):
                model_id = str(model_info.get("id", "")).strip()
                if model_id:
                    models.append(model_id)
    elif prov == "gemini":
        resp = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            headers={"x-goog-api-key": clean_key},
            proxies=proxies,
            timeout=15,
        )
        if resp.status_code != 200:
            raise RuntimeError(_friendly_ai_api_error(spec["title"], resp))
        for m in resp.json().get("models", []) or []:
            if not isinstance(m, dict):
                continue
            methods = m.get("supportedGenerationMethods") or []
            if "generateContent" not in methods:
                continue
            mid = str(m.get("name", "")).replace("models/", "").strip()
            low = mid.lower()
            if not mid or any(
                bad in low
                for bad in (
                    "embedding",
                    "imagen",
                    "veo",
                    "-tts",
                    "aqa",
                    "deep-research",
                    "computer-use",
                    "robotics",
                    "lyria",
                    "transcribe",
                    "-image",
                    "nano-banana",
                    "customtools",
                    "antigravity",
                )
            ):
                continue
            models.append(mid)
        models.reverse()
        _gem_priority = {
            "gemini-flash-lite-latest": 0,
            "gemini-flash-latest": 1,
            "gemini-pro-latest": 2,
        }
        models = sorted(
            dict.fromkeys(models),
            key=lambda x: (0 if x in _gem_priority else 1, _gem_priority.get(x, 99)),
        )
    else:
        endpoints = {
            "groq": "https://api.groq.com/openai/v1/models",
            "openai": "https://api.openai.com/v1/models",
            "openrouter": "https://openrouter.ai/api/v1/models",
            "deepseek": "https://api.deepseek.com/models",
            "zai": "https://api.z.ai/api/paas/v4/models",
        }
        url = endpoints[prov]
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {clean_key}"},
            proxies=proxies,
            timeout=15,
        )
        if resp.status_code != 200:
            raise RuntimeError(_friendly_ai_api_error(spec["title"], resp))
        raw_list = resp.json().get("data", []) if isinstance(resp.json(), dict) else []
        for m in raw_list:
            if not isinstance(m, dict):
                continue
            mid = str(m.get("id", "")).strip()
            low = mid.lower()
            if not mid:
                continue
            if prov == "groq" and any(b in low for b in ("whisper", "tts", "guard")):
                continue
            if prov == "openai":
                if not any(low.startswith(p) for p in ("gpt-", "o1", "o3", "o4", "chatgpt")):
                    continue
                if any(
                    b in low
                    for b in (
                        "tts",
                        "whisper",
                        "dall-e",
                        "embedding",
                        "audio",
                        "realtime",
                        "transcribe",
                        "image",
                        "sora",
                        "moderation",
                        "search",
                        "codex",
                        "deep-research",
                        "computer-use",
                    )
                ):
                    continue
            if prov == "openrouter" and any(
                b in low
                for b in (
                    "image",
                    "embedding",
                    "whisper",
                    "tts",
                    "dall-e",
                    "flux",
                    "stable-diffusion",
                    "lyria",
                    "deep-research",
                    "guard",
                )
            ):
                continue
            if prov == "zai" and any(b in low for b in ("cogview", "cogvideo", "embedding", "image", "tts", "asr")):
                continue
            models.append(mid)

        if prov == "openrouter":
            _or_priority = {
                "openai/gpt-4o-mini": 0,
                "google/gemini-2.5-flash-lite": 1,
                "google/gemini-2.5-flash": 2,
                "deepseek/deepseek-chat": 3,
            }
            models = sorted(
                set(models),
                key=lambda x: (
                    0 if x.endswith(":free") else (1 if x in _or_priority else 2),
                    _or_priority.get(x, 99),
                    x,
                ),
            )
        elif prov == "openai":
            _oai_priority = {
                "gpt-4o-mini": 0,
                "gpt-4.1-mini": 1,
                "gpt-4.1-nano": 2,
                "gpt-4o": 3,
                "gpt-4.1": 4,
            }
            models = sorted(
                set(models),
                key=lambda x: (0 if x in _oai_priority else 1, _oai_priority.get(x, 99), x),
                reverse=False,
            )
        elif prov == "deepseek":
            models = sorted(set(models), key=lambda x: (0 if x == "deepseek-chat" else 1, x))
        elif prov == "zai":
            models = sorted(
                set(models),
                key=lambda x: (0 if "flash" in x.lower() else 1, x),
            )
        elif prov == "groq":
            _groq_priority = {
                "openai/gpt-oss-20b": 0,
                "openai/gpt-oss-120b": 1,
                "llama-3.3-70b-versatile": 2,
                "llama-3.1-8b-instant": 3,
            }
            models = sorted(
                set(models),
                key=lambda x: (
                    0 if x in _groq_priority else (2 if x.lower().startswith("qwen/") else 1),
                    _groq_priority.get(x, 99),
                    x,
                ),
            )
        else:
            models = sorted(set(models), reverse=True)

    deduped = list(dict.fromkeys(models))
    default_m = str(spec.get("default_model", "")).strip()
    if default_m and default_m in deduped:
        deduped.remove(default_m)
        deduped.insert(0, default_m)
    return deduped


def _extract_protected_reaction_spans(text: str) -> tuple[List[str], List[str]]:
    """
    Розділяє текст на чергування звичайних фрагментів (`spans`) та захищених токенів
    звукових команд / реакцій (`tokens`), щоб AI-коректор гарантовано не спотворював
    `[laughter]`, `[хаха]` або `/хаха` / `\хаха` / `|хаха`.
    """
    from core.omnivoice_tags import (
        OMNIVOICE_ALL_OFFICIAL_TAGS,
        OMNIVOICE_WHISPER_TAG,
        build_all_configured_reaction_specs,
        expand_omnivoice_tags,
    )

    specs = build_all_configured_reaction_specs(config.data)
    expanded = expand_omnivoice_tags(text, specs=specs)
    known_tokens = list(OMNIVOICE_ALL_OFFICIAL_TAGS) + [OMNIVOICE_WHISPER_TAG] + [s.tag for s in specs if s.tag]
    if not known_tokens or "[" not in expanded:
        return [text], []

    pat = re.compile(
        "|".join(re.escape(tok) for tok in sorted(set(known_tokens), key=len, reverse=True)),
        re.IGNORECASE,
    )
    spans: List[str] = []
    tokens: List[str] = []
    pos = 0
    for m in pat.finditer(expanded):
        spans.append(expanded[pos : m.start()])
        tokens.append(m.group(0))
        pos = m.end()
    spans.append(expanded[pos:])
    return spans, tokens


def _run_ai_single_call(
    clean: str,
    prompt: str,
    prim_item: Optional[Dict[str, Any]],
    fb_item: Optional[Dict[str, Any]],
) -> tuple[Optional[str], str, str, str]:
    """Виконує один виклик AI (основний -> запасний). Повертає `(corrected_or_None, provider, route, error)`."""
    primary_error = ""
    if prim_item is not None:
        try:
            corrected = _call_ai_item(prim_item, clean, prompt)
            return corrected, get_ai_item_title(prim_item), "primary", ""
        except Exception as exc:
            primary_error = str(exc)
            print(f"[AICorrector] Помилка основного AI ({prim_item.get('title', prim_item.get('provider'))}): {exc}", file=sys.stderr)

    if fb_item is not None and (prim_item is None or str(fb_item.get("id")) != str(prim_item.get("id"))):
        try:
            corrected = _call_ai_item(fb_item, clean, prompt)
            return corrected, get_ai_item_title(fb_item), "fallback", primary_error
        except Exception as exc:
            print(f"[AICorrector] Помилка запасного AI ({fb_item.get('title', fb_item.get('provider'))}): {exc}", file=sys.stderr)

    return None, "", "failed", primary_error or "AI provider failed"


def correct_text_with_ai(text: str) -> str:
    """
    Виправляє орфографію/пунктуацію/розкладку через Головний AI-сервіс (або Запасний при збої).
    Якщо AI вимкнений, не налаштований або дав збій — безпечно повертає вихідний текст.
    Гарантовано зберігає токени звукових команд і реакцій (`[laughter]`, `[хаха]` тощо).
    """
    clean = (text or "").strip()
    if not clean or not bool(config.get("ai_spellcheck_enabled", False)):
        return text

    items: List[Dict[str, Any]] = list(config.get("ai_items", []))
    if not items:
        return text

    prim_id = str(config.get("ai_primary_id", "")).strip()
    fb_id = str(config.get("ai_fallback_id", "")).strip()
    prompt = get_effective_ai_prompt()

    prim_item = next((x for x in items if str(x.get("id")) == prim_id), items[0])
    fb_item = next((x for x in items if str(x.get("id")) == fb_id), None) if fb_id else None

    spans, tokens = _extract_protected_reaction_spans(clean)
    if not tokens:
        corrected, prov, route, err = _run_ai_single_call(clean, prompt, prim_item, fb_item)
        if corrected is not None:
            with _LAST_AI_EDIT_LOCK:
                _LAST_AI_EDIT.update({"input": text, "output": corrected, "provider": prov, "route": route, "error": err})
            return corrected
        with _LAST_AI_EDIT_LOCK:
            _LAST_AI_EDIT.update({"input": text, "output": text, "provider": "", "route": "failed", "error": err})
        return text

    # Якщо у фразі лише звукові команди без жодних слів — не витрачаємо запит до AI
    if not any(re.search(r"\w", sp) for sp in spans):
        return "".join(
            spans[i] + (tokens[i] if i < len(tokens) else "")
            for i in range(len(spans))
        ).strip()

    # Один запит до AI з індексованими маркерами ⟪T0⟫..⟪Tn⟫ + сувора перевірка цілісності
    placeholders = [f"⟪T{i}⟫" for i in range(len(tokens))]
    masked = "".join(
        spans[i] + (placeholders[i] if i < len(tokens) else "")
        for i in range(len(spans))
    )
    protected_prompt = (
        prompt
        + "\nIMPORTANT: Keep every placeholder token like ⟪T0⟫, ⟪T1⟫ completely intact, in exact order, without altering or translating it."
    )
    corrected_masked, prov, route, err = _run_ai_single_call(masked, protected_prompt, prim_item, fb_item)
    if corrected_masked is not None:
        # Перевіряємо, що всі маркери збереглися рівно по одному разу та в правильному порядку
        positions: List[int] = []
        intact = True
        for ph in placeholders:
            if corrected_masked.count(ph) != 1:
                intact = False
                break
            positions.append(corrected_masked.index(ph))
        if intact and positions == sorted(positions):
            restored = corrected_masked
            for ph, tok in zip(placeholders, tokens):
                restored = restored.replace(ph, tok, 1)
            with _LAST_AI_EDIT_LOCK:
                _LAST_AI_EDIT.update({"input": text, "output": restored, "provider": prov, "route": route, "error": err})
            return restored

        # Резервний режим (якщо мала локальна модель зіпсувала маркер):
        # виправляємо окремі текстові фрагменти й гарантовано склеюємо оригінальні токени
        rebuilt: List[str] = []
        any_ok = False
        for i, sp in enumerate(spans):
            if re.search(r"\w", sp):
                lead = sp[: len(sp) - len(sp.lstrip())]
                trail = sp[len(sp.rstrip()) :]
                part_corr, prov_p, route_p, err_p = _run_ai_single_call(sp.strip(), prompt, prim_item, fb_item)
                if part_corr is not None:
                    any_ok = True
                    prov, route, err = prov_p, route_p, err_p
                    rebuilt.append(f"{lead}{part_corr.strip()}{trail}")
                else:
                    rebuilt.append(sp)
            else:
                rebuilt.append(sp)
            if i < len(tokens):
                rebuilt.append(tokens[i])
        final_txt = re.sub(r"\s{2,}", " ", "".join(rebuilt)).strip()
        with _LAST_AI_EDIT_LOCK:
            _LAST_AI_EDIT.update({"input": text, "output": final_txt, "provider": prov if any_ok else "", "route": route if any_ok else "failed", "error": err})
        return final_txt

    with _LAST_AI_EDIT_LOCK:
        _LAST_AI_EDIT.update({"input": text, "output": text, "provider": "", "route": "failed", "error": err})
    return text
