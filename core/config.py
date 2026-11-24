"""
core/config.py — Менеджер конфігурації додатку (JSON-збереження, дефолтні налаштування).
"""

from __future__ import annotations

import atexit
import copy
import json
import os
import threading
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.secret_store import protect_config_dict, unprotect_config_dict

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
    BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", BASE_DIR))
    DATA_DIR = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")) / "SilentVoice"
else:
    BASE_DIR = Path(__file__).resolve().parent.parent
    BUNDLE_DIR = BASE_DIR
    DATA_DIR = BASE_DIR / "data"

LOCALES_DIR = (BASE_DIR / "locales") if (BASE_DIR / "locales").exists() else (BUNDLE_DIR / "locales")
SAMPLES_DIR = (DATA_DIR / "samples") if (DATA_DIR / "samples").exists() else (BUNDLE_DIR / "data" / "samples")
ENGINES_DIR = DATA_DIR / "engines"
PHRASES_AUDIO_DIR = DATA_DIR / "phrases_audio"
TTS_CACHE_DIR = DATA_DIR / "tts_cache"
OBS_TEMPLATES_DIR = DATA_DIR / "obs_templates"
CONFIG_FILE = DATA_DIR / "config.json"

DEFAULT_QUICK_PHRASES: List[Dict[str, str]] = [
    {"id": "qp_1", "title": "Привіт", "text": "Привіт всім!", "hotkey": "Alt+1", "emoji": "👋"},
    {"id": "qp_2", "title": "Секунду", "text": "Секунду, зараз напишу.", "hotkey": "Alt+2", "emoji": "✍️"},
    {"id": "qp_3", "title": "Так / Згодна", "text": "Так, повністю згодна!", "hotkey": "Alt+3", "emoji": "👍"},
    {"id": "qp_4", "title": "Ні", "text": "Ні, дякую, не треба.", "hotkey": "Alt+4", "emoji": "🙅‍♀️"},
    {"id": "qp_5", "title": "Дякую", "text": "Дякую велике!", "hotkey": "Alt+5", "emoji": "❤️"},
    {"id": "qp_6", "title": "Допомога", "text": "Допоможіть мені, будь ласка!", "hotkey": "Alt+6", "emoji": "🆘"},
    {"id": "qp_7", "title": "Відійду", "text": "Я зараз відійду на хвилинку, скоро буду.", "hotkey": "Alt+7", "emoji": "☕"},
    {"id": "qp_8", "title": "Повернулась", "text": "Я вже тут, можемо продовжувати!", "hotkey": "Alt+8", "emoji": "✨"},
    {"id": "qp_9", "title": "Гарна гра", "text": "Гарна гра, всім дякую!", "hotkey": "Alt+9", "emoji": "🎮"},
    {"id": "qp_10", "title": "Йдемо вперед", "text": "Пішли вперед, я поруч!", "hotkey": "", "emoji": "🛡️"},
    {"id": "qp_11", "title": "Про програму", "text": "Я спілкуюся через програму озвучення тексту.", "hotkey": "", "emoji": "🔊"},
    {"id": "qp_12", "title": "Прощання", "text": "Бувайте, всім гарного вечора!", "hotkey": "", "emoji": "🌙"},
]

DEFAULT_QUICK_PHRASES_TRANSLATIONS: Dict[str, Dict[str, tuple]] = {
    "uk": {
        "qp_1": ("Привіт", "Привіт всім!"),
        "qp_2": ("Секунду", "Секунду, зараз напишу."),
        "qp_3": ("Так / Згодна", "Так, повністю згодна!"),
        "qp_4": ("Ні", "Ні, дякую, не треба."),
        "qp_5": ("Дякую", "Дякую велике!"),
        "qp_6": ("Допомога", "Допоможіть мені, будь ласка!"),
        "qp_7": ("Відійду", "Я зараз відійду на хвилинку, скоро буду."),
        "qp_8": ("Повернулась", "Я вже тут, можемо продовжувати!"),
        "qp_9": ("Гарна гра", "Гарна гра, всім дякую!"),
        "qp_10": ("Йдемо вперед", "Пішли вперед, я поруч!"),
        "qp_11": ("Про програму", "Я спілкуюся через програму озвучення тексту."),
        "qp_12": ("Прощання", "Бувайте, всім гарного вечора!"),
    },
    "en": {
        "qp_1": ("Hello", "Hello everyone!"),
        "qp_2": ("One sec", "One second, I'll type it now."),
        "qp_3": ("Yes / Agree", "Yes, I completely agree!"),
        "qp_4": ("No", "No, thank you, no need."),
        "qp_5": ("Thank you", "Thank you very much!"),
        "qp_6": ("Help", "Help me, please!"),
        "qp_7": ("BRB", "I'll step away for a minute, be right back."),
        "qp_8": ("I'm back", "I'm back now, we can continue!"),
        "qp_9": ("Good game", "Good game, thanks everyone!"),
        "qp_10": ("Let's go", "Let's move forward, I'm right here!"),
        "qp_11": ("About app", "I communicate using a text-to-speech program."),
        "qp_12": ("Goodbye", "Goodbye, have a great evening everyone!"),
    },
    "ru": {
        "qp_1": ("Привет", "Привет всем!"),
        "qp_2": ("Секунду", "Секунду, сейчас напишу."),
        "qp_3": ("Да / Согласна", "Да, полностью согласна!"),
        "qp_4": ("Нет", "Нет, спасибо, не нужно."),
        "qp_5": ("Спасибо", "Большое спасибо!"),
        "qp_6": ("Помощь", "Помогите мне, пожалуйста!"),
        "qp_7": ("Отойду", "Я сейчас отойду на минутку, скоро буду."),
        "qp_8": ("Вернулась", "Я уже тут, можем продолжать!"),
        "qp_9": ("Хорошая игра", "Хорошая игра, всем спасибо!"),
        "qp_10": ("Идём вперёд", "Пошли вперёд, я рядом!"),
        "qp_11": ("О программе", "Я общаюсь через программу озвучивания текста."),
        "qp_12": ("Прощание", "Пока, всем хорошего вечера!"),
    },
    "pl": {
        "qp_1": ("Cześć", "Cześć wszystkim!"),
        "qp_2": ("Sekundę", "Sekundę, zaraz napiszę."),
        "qp_3": ("Tak / Zgoda", "Tak, całkowicie się zgadzam!"),
        "qp_4": ("Nie", "Nie, dziękuję, nie trzeba."),
        "qp_5": ("Dziękuję", "Dziękuję bardzo!"),
        "qp_6": ("Pomoc", "Pomóżcie mi, proszę!"),
        "qp_7": ("Zaraz wracam", "Odejdę na minutkę, zaraz wrócę."),
        "qp_8": ("Wróciłem", "Już jestem, możemy kontynuować!"),
        "qp_9": ("Dobra gra", "Dobra gra, dzięki wszystkim!"),
        "qp_10": ("Naprzód", "Ruszajmy naprzód, jestem tuż obok!"),
        "qp_11": ("O programie", "Rozmawiam przez program do zamiany tekstu na mowę."),
        "qp_12": ("Pożegnanie", "Cześć, miłego wieczoru wszystkim!"),
    },
    "de": {
        "qp_1": ("Hallo", "Hallo zusammen!"),
        "qp_2": ("Sekunde", "Eine Sekunde, ich schreibe gleich."),
        "qp_3": ("Ja / Einverstanden", "Ja, da stimme ich voll zu!"),
        "qp_4": ("Nein", "Nein, danke, nicht nötig."),
        "qp_5": ("Danke", "Vielen Dank!"),
        "qp_6": ("Hilfe", "Helft mir bitte!"),
        "qp_7": ("Bin kurz weg", "Ich bin kurz weg, bin gleich zurück."),
        "qp_8": ("Wieder da", "Ich bin wieder da, wir können weitermachen!"),
        "qp_9": ("Gutes Spiel", "Gutes Spiel, danke an alle!"),
        "qp_10": ("Vorwärts", "Gehen wir vor, ich bin gleich neben euch!"),
        "qp_11": ("Über die App", "Ich kommuniziere über ein Text-zu-Sprache-Programm."),
        "qp_12": ("Tschüss", "Tschüss, euch allen einen schönen Abend!"),
    },
    "fr": {
        "qp_1": ("Salut", "Salut tout le monde !"),
        "qp_2": ("Une seconde", "Une seconde, j'écris tout de suite."),
        "qp_3": ("Oui / D'accord", "Oui, je suis tout à fait d'accord !"),
        "qp_4": ("Non", "Non, merci, ce n'est pas nécessaire."),
        "qp_5": ("Merci", "Merci beaucoup !"),
        "qp_6": ("Aide", "Aidez-moi, s'il vous plaît !"),
        "qp_7": ("Je reviens", "Je m'absente une minute, je reviens vite."),
        "qp_8": ("De retour", "Me revoilà, on peut continuer !"),
        "qp_9": ("Bonne partie", "Bonne partie, merci à tous !"),
        "qp_10": ("En avant", "Allons-y, je suis juste à côté !"),
        "qp_11": ("À propos", "Je communique via un programme de synthèse vocale."),
        "qp_12": ("Au revoir", "Au revoir, bonne soirée à tous !"),
    },
    "es": {
        "qp_1": ("Hola", "¡Hola a todos!"),
        "qp_2": ("Un segundo", "Un segundo, ahora escribo."),
        "qp_3": ("Sí / De acuerdo", "¡Sí, totalmente de acuerdo!"),
        "qp_4": ("No", "No, gracias, no hace falta."),
        "qp_5": ("Gracias", "¡Muchas gracias!"),
        "qp_6": ("Ayuda", "¡Ayúdenme, por favor!"),
        "qp_7": ("Ya vuelvo", "Me ausento un minuto, enseguida vuelvo."),
        "qp_8": ("Volví", "¡Ya estoy aquí, podemos continuar!"),
        "qp_9": ("Buena partida", "¡Buena partida, gracias a todos!"),
        "qp_10": ("Adelante", "¡Vamos adelante, estoy cerca!"),
        "qp_11": ("Sobre la app", "Me comunico mediante un programa de texto a voz."),
        "qp_12": ("Adiós", "¡Adiós, que tengan una buena noche!"),
    },
}


def _load_locale_quick_phrases(lang: str) -> Dict[str, tuple]:
    code = str(lang or "").strip().lower()
    if not code:
        return {}
    if code in DEFAULT_QUICK_PHRASES_TRANSLATIONS:
        return DEFAULT_QUICK_PHRASES_TRANSLATIONS[code]
    loc_file = DATA_DIR / "locales" / f"{code}.json"
    if not loc_file.exists():
        loc_file = LOCALES_DIR / f"{code}.json"
    if loc_file.exists():
        try:
            data = json.loads(loc_file.read_text(encoding="utf-8"))
            meta = data.get("_meta") if isinstance(data, dict) else None
            qp_map = meta.get("quick_phrases") if isinstance(meta, dict) else None
            if isinstance(qp_map, dict):
                parsed: Dict[str, tuple] = {}
                for qp_id, pair in qp_map.items():
                    if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                        parsed[str(qp_id)] = (str(pair[0]), str(pair[1]))
                if parsed:
                    return parsed
        except Exception:
            pass
    return {}


def get_default_quick_phrases_for_lang(lang: str) -> List[Dict[str, str]]:
    """Повертає стандартний набір швидких фраз, перекладений на вказану мову (або EN за замовчуванням)."""
    tr_map = _load_locale_quick_phrases(lang) or DEFAULT_QUICK_PHRASES_TRANSLATIONS["en"]
    res: List[Dict[str, str]] = []
    for base in DEFAULT_QUICK_PHRASES:
        item = copy.deepcopy(base)
        qp_id = item["id"]
        if qp_id in tr_map:
            item["title"], item["text"] = tr_map[qp_id]
        res.append(item)
    return res

DEFAULT_CONFIG: Dict[str, Any] = {
    # 0. Мова інтерфейсу та перекладу помилок ("en", "uk", "pl", "de" тощо) + підтвердження при першому встановленні
    "app_language": "en",
    "ui_language": "en",
    "language_confirmed": False,
    "first_run_lang_checked": False,
    "first_run_help_shown": False,
    "voice_gender_filter": "both",
    "main_window_width": 0,
    "main_window_height": 0,
    "main_window_x": None,
    "main_window_y": None,

    # Обхід блокувань (вбудований авто-проксі та дзеркала для України та РФ)
    "use_download_bypass": True,
    "custom_proxy_url": "",
    "use_socks5_proxy": False,
    "socks5_proxy_url": "",

    # 1. Активний (Головний) та Запасний TTS-движок і голос + індивідуальні налаштування DSP
    "active_engine": "",
    "active_voice": "",
    "fallback_engine": "",
    "fallback_voice": "",
    "speech_speed": 1.0,
    "split_sentences_enabled": True,
    "normalize_numbers_enabled": True,
    "ai_spellcheck_enabled": False,
    "ai_items": [],
    "ai_primary_id": "",
    "ai_fallback_id": "",
    "ai_spellcheck_prompt": "",
    "ai_spellcheck_prompts": {},
    "engine_dsp_settings": {},
    "openvoice_enabled": False,
    "openvoice_ref_path": "",
    "openvoice_device": "gpu",
    "qwen3_tts_ref_path": "",
    "qwen3_tts_ref_text": "",
    "qwen3_tts_quant": "q8",
    "zonos3_mini_ref_path": "",
    "custom_clone_samples": {},

    # Онлайн-голоси (Edge за замовчуванням + список доданих користувачем API)
    "edge_tts_enabled": True,
    "online_items": [],  # Список словників: {"id": "...", "provider": "google"|"elevenlabs"|"openai", "api_key": ""}

    # 2. Екранний текст
    "overlay_enabled": True,
    "overlay_style": "text_only",
    "overlay_draggable": False,
    "overlay_bg_opacity": 78,
    "overlay_elements_opacity": 30,
    "overlay_text_opacity": 100,
    "overlay_font_family": "Segoe UI",
    "overlay_font_size": 18,
    "overlay_outline_width": 3,
    "overlay_shadow_blur": 10,
    "overlay_text_color": "#FFFFFF",
    "overlay_bg_color": "#13151F",
    "overlay_border_color": "#7C5CFF",
    "overlay_x_offset": 0,
    "overlay_y_offset": 85,
    "overlay_screen_name": "",
    "overlay_max_width": 1120,
    "overlay_input_width": 700,
    "overlay_input_height": 110,
    "overlay_auto_hide_sec": 3.0,
    "hide_from_capture": False,

    # Twitch Chat integration (off by default)
    "twitch_enabled": False,
    "twitch_intercept_own": False,
    "twitch_read_viewers": False,
    "twitch_chat_read_all": False,
    "twitch_chat_command_prefix": "!!!",
    "twitch_chat_tts_engine": "",
    "twitch_chat_tts_voice": "",
    "twitch_chat_volume": 100,
    "twitch_chat_voice_rules": [],
    "twitch_viewer_filter": "all",
    "twitch_allowed_users": "",
    "twitch_ignored_users": "nightbot, streamelements, streamlabs, moobot, fossabot, wizebot",
    "twitch_read_format": "user_says",
    "twitch_ignore_commands": True,
    "twitch_ignore_links": True,
    "twitch_max_chars": 220,
    "twitch_user_cooldown_sec": 0,
    "twitch_max_queue": 5,
    "twitch_save_viewers_history": False,
    "twitch_chat_send_vbcable": False,
    "twitch_chat_send_vban": False,
    "twitch_chat_vban_mode": "same",
    "twitch_chat_vban_stream": "SilentChat",
    "twitch_username": "",
    "twitch_token": "",
    "twitch_channel": "",

    # 3. Субтитри OBS
    "obs_enabled": True,
    "obs_port": 8765,
    "obs_template_mode": "builtin",
    "obs_style": "text_only",
    "obs_font_family": "Montserrat",
    "obs_font_size": 24,
    "obs_text_opacity": 100,
    "obs_text_color": "#FFFFFF",
    "obs_outline_width": 3,
    "obs_shadow_blur": 10,
    "obs_shadow_enabled": True,
    "obs_bg_enabled": False,
    "obs_bg_color": "#13151F",
    "obs_border_color": "#7C5CFF",
    "obs_bg_opacity": 78,
    "obs_border_radius": 14,
    "obs_anim_style": "typewriter",
    "obs_animate_block": True,
    "obs_animate_motion_block": False,
    "obs_motion_effect": "none",
    "obs_anim_params": {},
    "obs_motion_params": {},
    "obs_show_live_typing": False,
    "obs_live_typing_mode": "typing_only",
    "obs_fade_sec": 3.5,
    "obs_show_cursor": False,
    "obs_cursor_char": "▌",
    "obs_cursor_color": "#9B82FF",
    "obs_top_fade": True,

    # 4. Хоткеї та Ввід
    "input_mode": "hook",
    "continuous_chat_mode": False,
    "hotkey_toggle_input": "Ctrl+Space",
    "hotkey_conversation_mode": "Ctrl+Shift+Space",
    "hotkey_send": "Enter",
    "hotkey_cancel": "Esc",
    "hotkey_stop_tts": "Ctrl+Shift+S",
    "hotkey_repeat_last": "Ctrl+Shift+R",
    "hotkey_toggle_game_mode": "Ctrl+Shift+G",
    "hotkey_stop_chat_tts": "",
    "hotkey_toggle_chat_tts": "",
    "sound_feedback": True,
    "cancel_sound_enabled": True,
    "sound_feedback_volume": 100,
    "save_draft_on_close": True,
    "draft_keep_ttl_sec": 10,
    "game_mode_enabled": True,
    "game_emergency_cancel": True,
    "game_mouse_sensitivity": "low",
    "game_save_draft": True,
    "game_draft_ttl_sec": 20,

    # 5. Аудіо та Система
    "primary_output_device": "CABLE Input (VB-Audio Virtual Cable)",
    "primary_volume": 100,
    "secondary_enabled": True,
    "secondary_output_device": "",
    "secondary_volume": 85,
    "mic_passthrough_enabled": False,
    "mic_passthrough_device": "",
    "mic_passthrough_volume": 100,
    "audio_output_format": "auto",
    # VBAN network audio output (off by default)
    "vban_enabled": False,
    "vban_address": "",
    "vban_port": 6980,
    "vban_stream_name": "SilentVoice",
    "vban_volume": 100,
    "start_minimized": False,
    "minimize_to_tray_on_close": True,
    "autostart_windows": False,
    # За замовчуванням увімкнено через Windows UIPI: без прав адміністратора глобальний
    # перехват клавіатури (WH_KEYBOARD_LL) не працює поверх ігор/античітів, запущених
    # з підвищеними правами. Користувач може вимкнути цю опцію в налаштуваннях.
    "run_as_admin": True,
    "auto_check_updates": True,
    "update_repo_url": "",
    "skipped_update_version": "",

    # 6. Швидкі фрази та історія
    "quick_phrases": copy.deepcopy(DEFAULT_QUICK_PHRASES),
    "recent_phrases": [],
}


class ConfigManager:
    """Потокобезпечний менеджер конфігурації з автозбереженням у JSON."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._io_lock = threading.Lock()
        self._save_timer: Optional[threading.Timer] = None
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        ENGINES_DIR.mkdir(parents=True, exist_ok=True)
        PHRASES_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)
        self.load()

    def load(self) -> None:
        with self._lock:
            if not CONFIG_FILE.exists():
                self.save()
                return
            try:
                raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    raw_plain, needs_dpapi_migration = unprotect_config_dict(raw)
                    merged = copy.deepcopy(DEFAULT_CONFIG)
                    for k, v in raw_plain.items():
                        merged[k] = v
                    if "first_run_lang_checked" not in raw_plain and (
                        bool(raw_plain.get("language_confirmed", False))
                        or bool(str(raw_plain.get("ui_language", "")).strip())
                        or bool(str(raw_plain.get("app_language", "")).strip())
                    ):
                        merged["first_run_lang_checked"] = True
                    self._data = merged
                    if needs_dpapi_migration:
                        self.save()
            except Exception as exc:
                print(f"[Config] Помилка читання config.json: {exc}")

    def save(self) -> None:
        with self._lock:
            if self._save_timer is not None:
                self._save_timer.cancel()
                self._save_timer = None
            disk_data = protect_config_dict(self._data)
            payload = json.dumps(disk_data, ensure_ascii=False, indent=2)
        with self._io_lock:
            tmp_file = CONFIG_FILE.with_name(f"config_{os.getpid()}_{threading.get_ident()}.tmp")
            try:
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                tmp_file.write_text(payload, encoding="utf-8")
                replaced = False
                for _ in range(3):
                    try:
                        os.replace(tmp_file, CONFIG_FILE)
                        replaced = True
                        break
                    except PermissionError:
                        time.sleep(0.02)
                if not replaced:
                    CONFIG_FILE.write_text(payload, encoding="utf-8")
            except Exception as exc:
                try:
                    CONFIG_FILE.write_text(payload, encoding="utf-8")
                except Exception:
                    print(f"[Config] Помилка збереження config.json: {exc}")
            finally:
                if tmp_file.exists():
                    try:
                        tmp_file.unlink()
                    except Exception:
                        pass

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if default is None and key in DEFAULT_CONFIG:
                default = DEFAULT_CONFIG[key]
            return self._data.get(key, default)

    def set(self, key: str, value: Any, auto_save: bool = True) -> None:
        with self._lock:
            self._data[key] = value
            if auto_save:
                self.save()

    def update(self, values: Dict[str, Any], auto_save: bool = True) -> None:
        with self._lock:
            self._data.update(values)
            if auto_save:
                self.save()

    def schedule_save(self, delay_sec: float = 0.22) -> None:
        """Відкладає запис JSON на диск (для плавного перетягування повзунків і вікон без фризів)."""
        with self._lock:
            if self._save_timer is not None:
                self._save_timer.cancel()
            timer = threading.Timer(max(0.05, float(delay_sec)), self.save)
            timer.daemon = True
            self._save_timer = timer
            timer.start()

    def flush_pending_save(self) -> None:
        with self._lock:
            if self._save_timer is not None:
                self._save_timer.cancel()
                self._save_timer = None
                self.save()

    def add_recent_phrase(self, text: str) -> List[str]:
        clean = text.strip()
        if not clean:
            return self.get("recent_phrases", [])
        with self._lock:
            recents: List[str] = list(self._data.get("recent_phrases", []))
            if clean in recents:
                recents.remove(clean)
            recents.insert(0, clean)
            recents = recents[:20]
            self._data["recent_phrases"] = recents
            self.save()
            return recents

    @property
    def data(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data)


def sync_default_quick_phrases_for_lang(lang: str) -> bool:
    """
    Оновлює незмінені користувачем стандартні швидкі фрази (qp_1..qp_12) під обрану мову.
    Власні або відредаговані фрази користувача залишаються недоторканими.
    """
    code = str(lang or "").strip().lower() or "en"
    target_map = _load_locale_quick_phrases(code) or DEFAULT_QUICK_PHRASES_TRANSLATIONS["en"]

    # Збираємо всі відомі стандартні (title, text) для кожного qp_id (включно з локалями у locales/)
    known_by_id: Dict[str, set] = {b["id"]: {(b["title"], b["text"])} for b in DEFAULT_QUICK_PHRASES}
    for lang_map in DEFAULT_QUICK_PHRASES_TRANSLATIONS.values():
        for qid, pair in lang_map.items():
            known_by_id.setdefault(qid, set()).add(pair)
    for loc_dir in (LOCALES_DIR, DATA_DIR / "locales"):
        if loc_dir.exists():
            for lp in loc_dir.glob("*.json"):
                for qid, pair in _load_locale_quick_phrases(lp.stem).items():
                    known_by_id.setdefault(qid, set()).add(pair)

    phrases = list(config.get("quick_phrases", []))
    changed = False
    for p in phrases:
        qid = str(p.get("id", ""))
        if qid not in known_by_id or qid not in target_map:
            continue
        cur_pair = (str(p.get("title", "")), str(p.get("text", "")))
        if cur_pair in known_by_id[qid]:
            new_title, new_text = target_map[qid]
            if cur_pair != (new_title, new_text):
                p["title"] = new_title
                p["text"] = new_text
                changed = True
    if changed:
        config.set("quick_phrases", phrases)
    return changed


config = ConfigManager()
atexit.register(config.flush_pending_save)
