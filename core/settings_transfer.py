"""Portable, category-based import/export for user configuration."""

from __future__ import annotations

import copy
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping
from uuid import uuid4

from core.config import DEFAULT_CONFIG, PHRASES_AUDIO_DIR, config
from core.i18n import tr


FORMAT_NAME = "SilentVoiceSettings"
FORMAT_VERSION = 1
MAX_IMPORT_BYTES = 10 * 1024 * 1024
MAX_PROFILE_BYTES = 512 * 1024 * 1024
MAX_PROFILE_CONTENT_BYTES = 250 * 1024 * 1024

# Device names, local paths, recent history, Telegram sessions, and proxy
# credentials are deliberately excluded from portable profiles.
CATEGORY_KEYS: Dict[str, tuple[str, ...]] = {
    "general": (
        "ui_language", "app_language", "voice_gender_filter",
        "start_minimized", "minimize_to_tray_on_close", "auto_check_updates",
    ),
    "appearance": tuple(k for k in DEFAULT_CONFIG if k.startswith("overlay_")) + (
        "hide_from_capture", "main_window_width", "main_window_height", "main_window_x", "main_window_y",
    ),
    "voices": (
        "active_engine", "active_voice", "fallback_engine", "fallback_voice",
        "speech_speed", "split_sentences_enabled", "normalize_numbers_enabled",
        "engine_dsp_settings",
        "omnivoice_quant", "omnivoice_steps", "omnivoice_mode",
        "omnivoice_denoise",
        "omnivoice_design_gender", "omnivoice_design_age",
        "omnivoice_design_pitch", "omnivoice_design_accent",
        "omnivoice_design_seed",
    ),
    "input": (
        "input_mode",
        "continuous_chat_mode",
        "sound_feedback",
        "cancel_sound_enabled",
        "sound_feedback_volume",
        "save_draft_on_close",
        "draft_keep_ttl_sec",
        "game_mode_enabled",
        "game_emergency_cancel",
        "game_mouse_sensitivity",
        "game_save_draft",
        "game_draft_ttl_sec",
    ),
    "hotkeys": tuple(k for k in DEFAULT_CONFIG if k.startswith("hotkey_")),
    "obs": tuple(k for k in DEFAULT_CONFIG if k.startswith("obs_")),
    "quick_phrases": ("quick_phrases",),
    "online_services": ("edge_tts_enabled", "online_items"),
    "ai_services": (
        "ai_spellcheck_enabled", "ai_items", "ai_primary_id", "ai_fallback_id",
        "ai_spellcheck_prompt", "ai_spellcheck_prompts",
    ),
    # Physical device identifiers are machine-specific, so only levels and
    # routing toggles travel with a profile.
    "audio": (
        "primary_volume", "secondary_enabled", "secondary_volume",
        "mic_passthrough_enabled", "mic_passthrough_volume",
        "vban_volume",
    ),
}


_SECRET_FIELD_NAMES = frozenset({
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "authorization",
    "tg_api_id",
    "tg_api_hash",
    "tg_phone",
})


def _without_secrets(value: Any) -> Any:
    """Deep-copy JSON-like values while dropping credential-shaped fields."""
    if isinstance(value, dict):
        return {
            str(key): _without_secrets(item)
            for key, item in value.items()
            if str(key).strip().lower() not in _SECRET_FIELD_NAMES
        }
    if isinstance(value, list):
        return [_without_secrets(item) for item in value]
    return copy.deepcopy(value)


def _category_data(source: Mapping[str, Any], category: str, include_api_keys: bool) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for key in CATEGORY_KEYS[category]:
        if key not in source:
            continue
        value = copy.deepcopy(source[key])
        if not include_api_keys:
            value = _without_secrets(value)
        values[key] = value
    return values


def build_export(source: Mapping[str, Any], categories: Iterable[str], include_api_keys: bool) -> Dict[str, Any]:
    requested = set(categories)
    chosen = [name for name in CATEGORY_KEYS if name in requested]
    if include_api_keys:
        # Credentials live inside service records; include those records as well.
        for required in ("online_services", "ai_services"):
            if required not in chosen:
                chosen.append(required)
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "contains_api_keys": bool(include_api_keys),
        "categories": {
            name: _category_data(source, name, include_api_keys)
            for name in chosen
        },
    }


def write_export_file(path: str | Path, payload: Mapping[str, Any]) -> None:
    file_path = Path(path)
    temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    try:
        profile = copy.deepcopy(dict(payload))
        categories = profile.get("categories", {})
        audio_files: Dict[str, Path] = {}
        phrase_data = categories.get("quick_phrases", {}) if isinstance(categories, dict) else {}
        phrases = phrase_data.get("quick_phrases", []) if isinstance(phrase_data, dict) else []
        if isinstance(phrases, list):
            for index, phrase in enumerate(phrases):
                if not isinstance(phrase, dict):
                    continue
                source_path = Path(str(phrase.get("audio_file", "") or ""))
                if not source_path.is_file():
                    phrase.pop("audio_file", None)
                    continue
                if source_path.stat().st_size > 200 * 1024 * 1024:
                    raise ValueError(tr("transfer.error.audio_too_large", "Audio file '{name}' exceeds the 200 MB limit.", name=source_path.name))
                safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", source_path.name).strip("._") or "audio"
                archive_name = f"audio/phrase_{index + 1}_{safe_name}"
                phrase["audio_file"] = archive_name
                audio_files[archive_name] = source_path
        if sum(source.stat().st_size for source in audio_files.values()) > MAX_PROFILE_CONTENT_BYTES:
            raise ValueError(tr("transfer.error.audio_total_too_large", "The total size of quick phrase audio files exceeds the 250 MB limit."))
        profile["contains_audio"] = bool(audio_files)
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr("settings.json", json.dumps(profile, ensure_ascii=False, indent=2).encode("utf-8"))
            for archive_name, source_path in audio_files.items():
                archive.write(source_path, archive_name)
        temp_path.replace(file_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def read_import_file(path: str | Path) -> Dict[str, Any]:
    file_path = Path(path)
    try:
        file_size = file_path.stat().st_size
        if zipfile.is_zipfile(file_path):
            if file_size > MAX_PROFILE_BYTES:
                raise ValueError(tr("transfer.error.profile_too_large", "The profile archive is too large (512 MB limit)."))
            with zipfile.ZipFile(file_path, "r") as archive:
                infos = archive.infolist()
                total_size = sum(info.file_size for info in infos)
                if total_size > MAX_PROFILE_CONTENT_BYTES:
                    raise ValueError(tr("transfer.error.unpacked_too_large", "The unpacked profile is too large (250 MB limit)."))
                settings_info = next((info for info in infos if info.filename == "settings.json"), None)
                if settings_info is None or settings_info.file_size > MAX_IMPORT_BYTES:
                    raise ValueError(tr("transfer.error.missing_settings", "The profile does not contain a valid settings.json file."))
                payload = json.loads(archive.read(settings_info).decode("utf-8-sig"))
                audio_files: Dict[str, bytes] = {}
                for info in infos:
                    name = info.filename
                    if not name.startswith("audio/") or info.is_dir():
                        continue
                    parts = Path(name).parts
                    if len(parts) != 2 or parts[0] != "audio" or ".." in parts:
                        raise ValueError(tr("transfer.error.bad_audio_path", "The archive contains an invalid audio file path."))
                    if info.file_size > 200 * 1024 * 1024:
                        raise ValueError(tr("transfer.error.audio_too_large_generic", "An audio file exceeds the 200 MB limit."))
                    audio_files[name] = archive.read(info)
            if isinstance(payload, dict):
                payload["_audio_files"] = audio_files
        else:
            if file_size > MAX_IMPORT_BYTES:
                raise ValueError(tr("transfer.error.file_too_large", "The settings file is too large (10 MB limit)."))
            payload = json.loads(file_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise ValueError(tr("transfer.error.read_failed", "Could not read the settings file: {error}", error=exc)) from exc
    if not isinstance(payload, dict) or payload.get("format") != FORMAT_NAME:
        raise ValueError(tr("transfer.error.not_profile", "This is not a SilentVoice settings profile."))
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(tr("transfer.error.unsupported_version", "This app version does not support the profile format."))
    categories = payload.get("categories")
    if not isinstance(categories, dict):
        raise ValueError(tr("transfer.error.bad_categories", "The file does not contain a valid list of sections."))
    # Ignore unknown categories and keys; imports only ever write the allowlist above.
    return payload


def _is_loopback_base_url(url_str: str) -> bool:
    from urllib.parse import urlsplit

    raw = str(url_str or "").strip()
    if not raw:
        return True
    parsed = urlsplit(raw)
    host = (parsed.hostname or "").strip().lower()
    return host in ("127.0.0.1", "localhost", "::1")


def _merge_service_items(incoming: Any, current: Any, allow_imported_keys: bool) -> list[Any]:
    if not isinstance(incoming, list):
        return copy.deepcopy(current) if isinstance(current, list) else []
    current_by_id: Dict[str, Dict[str, Any]] = {}
    current_by_provider: Dict[str, Dict[str, Any]] = {}
    if isinstance(current, list):
        for cur_item in current:
            if not isinstance(cur_item, dict):
                continue
            cid = str(cur_item.get("id", "")).strip()
            if cid:
                current_by_id[cid] = cur_item
            cprov = str(cur_item.get("provider", "")).strip()
            if cprov and cprov != "local_api" and cprov not in current_by_provider:
                current_by_provider[cprov] = cur_item
    merged = []
    for raw_item in incoming:
        if not isinstance(raw_item, dict):
            continue
        item = copy.deepcopy(raw_item)
        prov = str(item.get("provider", "")).strip()
        old_item = current_by_id.get(str(item.get("id", "")).strip())
        # Prevent cross-provider credential transplanting if an imported profile
        # reuses an existing item ID with a different provider (e.g., local_api).
        if isinstance(old_item, dict) and str(old_item.get("provider", "")).strip() != prov:
            old_item = None
        if old_item is None and prov and prov != "local_api":
            old_item = current_by_provider.get(prov)
        old_dict = old_item if isinstance(old_item, dict) else {}
        existing_key = str(old_dict.get("api_key", "") or "")
        # Never attach a local API key onto a newly imported non-loopback custom base_url.
        if prov in ("local_api", "ollama") and existing_key:
            inc_url = str(item.get("base_url", "") or "").strip()
            old_url = str(old_dict.get("base_url", "") or "").strip()
            if inc_url != old_url and not _is_loopback_base_url(inc_url):
                existing_key = ""
        if allow_imported_keys:
            if not str(item.get("api_key", "") or ""):
                item["api_key"] = existing_key
            for tg_key in ("tg_api_id", "tg_api_hash", "tg_phone"):
                if not item.get(tg_key) and old_dict.get(tg_key):
                    item[tg_key] = old_dict[tg_key]
        else:
            item["api_key"] = existing_key
            for tg_key in ("tg_api_id", "tg_api_hash", "tg_phone"):
                if old_dict.get(tg_key):
                    item[tg_key] = old_dict[tg_key]
                else:
                    item.pop(tg_key, None)
        merged.append(item)
    return merged


def _compatible_value(key: str, value: Any) -> bool:
    expected = DEFAULT_CONFIG.get(key)
    if isinstance(expected, bool):
        return type(value) is bool
    if isinstance(expected, int):
        return type(value) is int
    if isinstance(expected, float):
        return type(value) in (int, float)
    if isinstance(expected, str):
        return isinstance(value, str) and len(value) <= 100_000
    if isinstance(expected, list):
        return isinstance(value, list)
    if isinstance(expected, dict):
        return isinstance(value, dict)
    return False


def apply_import(payload: Mapping[str, Any], selected: Iterable[str], include_api_keys: bool) -> set[str]:
    raw_categories = payload.get("categories", {})
    if not isinstance(raw_categories, dict):
        raise ValueError(tr("transfer.error.bad_categories", "The file does not contain a valid list of sections."))
    selected_names = set(selected)
    if include_api_keys:
        selected_names.update(("online_services", "ai_services"))
    updates: Dict[str, Any] = {}
    applied: set[str] = set()
    current = config.data
    for category in CATEGORY_KEYS:
        if category not in selected_names or category not in raw_categories:
            continue
        incoming = raw_categories[category]
        if not isinstance(incoming, dict):
            continue
        valid_keys = CATEGORY_KEYS[category]
        for key in valid_keys:
            if key not in incoming or not _compatible_value(key, incoming[key]):
                continue
            value = incoming[key]
            if key in ("online_items", "ai_items"):
                value = _merge_service_items(value, current.get(key, []), include_api_keys)
            elif key == "quick_phrases":
                value = _import_quick_phrases(value, payload.get("_audio_files", {}))
            updates[key] = copy.deepcopy(value)
        applied.add(category)
    if updates:
        config.update(updates)
    return applied


def _import_quick_phrases(phrases: list[Any], audio_files: Any) -> list[Dict[str, Any]]:
    if not isinstance(audio_files, dict):
        audio_files = {}
    imported: list[Dict[str, Any]] = []
    for phrase in phrases:
        if not isinstance(phrase, dict):
            continue
        item = {
            str(name): str(field)
            for name, field in phrase.items()
            if name in {"id", "title", "text", "hotkey", "emoji"}
            and isinstance(field, (str, int, float, bool))
        }
        archive_name = phrase.get("audio_file")
        audio_bytes = audio_files.get(archive_name) if isinstance(archive_name, str) else None
        if isinstance(audio_bytes, bytes):
            suffix = Path(archive_name).suffix.lower()
            if suffix not in {".wav", ".mp3", ".ogg", ".flac"}:
                suffix = ".audio"
            PHRASES_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            destination = PHRASES_AUDIO_DIR / f"imported_{uuid4().hex}{suffix}"
            destination.write_bytes(audio_bytes)
            item["audio_file"] = str(destination)
        imported.append(item)
    return imported
