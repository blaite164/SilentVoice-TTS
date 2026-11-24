"""Comprehensive unit and regression tests for fixed bugs and edge cases."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from core.ai_corrector import correct_text_with_ai
from core.config import config
from core.obs_animations import (
    ENTRY_ANIMATIONS,
    MOTION_ANIMATIONS,
    get_entry_spec,
    get_motion_spec,
    get_variant_defaults,
    get_variant_key,
    resolve_anim_params,
)
from core.settings_transfer import (
    _merge_service_items,
    _without_secrets,
    apply_import,
    build_export,
    read_import_file,
    write_export_file,
)
from core.text_normalizer import normalize_tts_text
from core.tts_manager import split_into_sentences_punct, tts_manager
from core.win_api import format_hotkey, parse_hotkey_string, user32


class WinApiSignaturesAndHotkeyTests(unittest.TestCase):
    def test_win32_ctypes_64bit_signatures_defined(self) -> None:
        if hasattr(user32, "GetWindowLongPtrW"):
            self.assertEqual(user32.GetWindowLongPtrW.argtypes, [wintypes.HWND, ctypes.c_int])
            self.assertEqual(user32.GetWindowLongPtrW.restype, ctypes.c_ssize_t)
        if hasattr(user32, "SetWindowLongPtrW"):
            self.assertEqual(
                user32.SetWindowLongPtrW.argtypes,
                [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t],
            )
            self.assertEqual(user32.SetWindowLongPtrW.restype, ctypes.c_ssize_t)
        self.assertEqual(
            user32.SetWindowPos.argtypes,
            [
                wintypes.HWND,
                wintypes.HWND,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                wintypes.UINT,
            ],
        )
        self.assertEqual(user32.SetForegroundWindow.argtypes, [wintypes.HWND])
        self.assertEqual(user32.SetFocus.argtypes, [wintypes.HWND])

    def test_hotkey_parse_and_format_roundtrip(self) -> None:
        spec = parse_hotkey_string("Ctrl+Shift+Space")
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.mods, frozenset({"ctrl", "shift"}))
        self.assertFalse(spec.is_double)
        self.assertEqual(format_hotkey(spec.mods, spec.vk, spec.is_double), "Ctrl+Shift+Space")

        dbl = parse_hotkey_string("2x Ctrl")
        self.assertIsNotNone(dbl)
        assert dbl is not None
        self.assertTrue(dbl.is_double)
        self.assertEqual(dbl.mods, frozenset())
        self.assertEqual(format_hotkey(set(dbl.mods), dbl.vk, dbl.is_double), "2x Ctrl")


class AiCorrectorFallbackTests(unittest.TestCase):
    def test_stale_primary_id_falls_back_to_first_item_and_then_fallback_item(self) -> None:
        items = [
            {"id": "ai_groq_new", "provider": "groq", "title": "Groq", "api_key": "k1", "model": "m1"},
            {"id": "ai_openai_fb", "provider": "openai", "title": "OpenAI", "api_key": "k2", "model": "m2"},
        ]
        state = {
            "ai_spellcheck_enabled": True,
            "ai_items": items,
            "ai_primary_id": "ai_groq_deleted_old_id",
            "ai_fallback_id": "ai_openai_fb",
            "app_language": "uk",
        }
        called_ids: list[str] = []

        def _fake_call(item, text, prompt):
            called_ids.append(str(item.get("id")))
            if item.get("id") == "ai_groq_new":
                return "Виправлений текст"
            return "Запасний текст"

        with (
            patch.object(config, "get", side_effect=lambda k, d=None: state.get(k, d)),
            patch("core.ai_corrector._call_ai_item", side_effect=_fake_call),
        ):
            out = correct_text_with_ai("текст")
            self.assertEqual(out, "Виправлений текст")
            self.assertEqual(called_ids, ["ai_groq_new"])

        # When primary fails, fallback item is used
        called_ids.clear()

        def _fail_primary(item, text, prompt):
            called_ids.append(str(item.get("id")))
            if item.get("id") == "ai_groq_new":
                raise RuntimeError("Primary rate limit")
            return "Запасний текст"

        with (
            patch.object(config, "get", side_effect=lambda k, d=None: state.get(k, d)),
            patch("core.ai_corrector._call_ai_item", side_effect=_fail_primary),
        ):
            out2 = correct_text_with_ai("текст")
            self.assertEqual(out2, "Запасний текст")
            self.assertEqual(called_ids, ["ai_groq_new", "ai_openai_fb"])


class SettingsTransferSecretsAndMergeTests(unittest.TestCase):
    def test_without_secrets_strips_api_keys_and_telegram_credentials(self) -> None:
        src = {
            "online_items": [
                {
                    "id": "silero_bot_1",
                    "provider": "silero_bot",
                    "api_key": "secret_key",
                    "tg_api_id": "123456",
                    "tg_api_hash": "abcdef123456",
                    "tg_phone": "+380991112233",
                    "timeout_sec": 60,
                }
            ],
            "ai_items": [
                {
                    "id": "ai_groq_1_1111",
                    "provider": "groq",
                    "api_key": "gsk_secret",
                    "model": "llama-3.3-70b-versatile",
                }
            ],
        }
        cleaned = _without_secrets(src)
        tg_item = cleaned["online_items"][0]
        self.assertNotIn("api_key", tg_item)
        self.assertNotIn("tg_api_id", tg_item)
        self.assertNotIn("tg_api_hash", tg_item)
        self.assertNotIn("tg_phone", tg_item)
        self.assertEqual(tg_item["timeout_sec"], 60)
        self.assertNotIn("api_key", cleaned["ai_items"][0])

        exported = build_export(src, ["online_services", "ai_services"], include_api_keys=False)
        exp_tg = exported["categories"]["online_services"]["online_items"][0]
        self.assertNotIn("tg_api_hash", exp_tg)
        self.assertNotIn("tg_phone", exp_tg)

    def test_merge_service_items_matches_by_provider_when_id_differs(self) -> None:
        current = [
            {
                "id": "ai_groq_1_1001",
                "provider": "groq",
                "api_key": "local_groq_key",
                "model": "old-model",
            },
            {
                "id": "silero_bot_1_2002",
                "provider": "silero_bot",
                "api_key": "",
                "tg_api_id": "777",
                "tg_api_hash": "local_hash",
                "tg_phone": "+38000",
            },
        ]
        incoming = [
            {
                "id": "ai_groq_1_9999",
                "provider": "groq",
                "model": "new-model",
            },
            {
                "id": "silero_bot_1_8888",
                "provider": "silero_bot",
                "api_key": "",
                "tg_api_id": "999",
                "tg_api_hash": "imported_hash",
                "tg_phone": "+48999",
            },
        ]
        merged_no_keys = _merge_service_items(incoming, current, allow_imported_keys=False)
        self.assertEqual(merged_no_keys[0]["api_key"], "local_groq_key")
        self.assertEqual(merged_no_keys[0]["model"], "new-model")
        self.assertEqual(merged_no_keys[1]["tg_api_hash"], "local_hash")
        self.assertEqual(merged_no_keys[1]["tg_phone"], "+38000")


class TtsManagerQueueAndFallbackCacheTests(unittest.TestCase):
    def test_prune_queue_is_atomic_and_preserves_unfinished_tasks(self) -> None:
        tts_manager.stop_chat_tts()
        tts_manager._queue.put(("keep_1", False, True, False, False, "", "", None, "", ""))
        tts_manager._queue.put(("drop_chat", False, True, False, True, "", "", None, "user1", "m1"))
        tts_manager._queue.put(("keep_2", False, True, False, False, "", "", None, "", ""))
        try:
            removed = tts_manager._prune_queue(lambda q: bool(q[4]))
            self.assertEqual(removed, 1)
            self.assertEqual(tts_manager.pending_queue_size(), 2)
            out = []
            while not tts_manager._queue.empty():
                out.append(tts_manager._queue.get_nowait()[0])
                tts_manager._queue.task_done()
            self.assertEqual(out, ["keep_1", "keep_2"])
            self.assertEqual(tts_manager._queue.unfinished_tasks, 0)
        finally:
            tts_manager.stop_chat_tts()

    def test_fallback_synthesis_stores_and_reuses_audio_cache(self) -> None:
        avail = [
            {"id": "online:prim", "title": "Primary Online", "voices": [{"id": "v_p", "name": "VP"}]},
            {"id": "piper", "title": "Piper Local", "voices": [{"id": "v_fb", "name": "VFB"}]},
        ]
        state = {
            "active_engine": "online:prim",
            "active_voice": "v_p",
            "fallback_engine": "piper",
            "fallback_voice": "v_fb",
            "normalize_numbers_enabled": False,
            "app_language": "uk",
            "openvoice_enabled": False,
        }
        dispatch_calls: list[tuple[str, str]] = []
        fake_pcm = np.full(1200, 0.2, dtype=np.float32)

        def _fake_dispatch(eng_id, voice_id, txt, spd):
            dispatch_calls.append((eng_id, voice_id))
            if eng_id == "online:prim":
                raise RuntimeError("Primary offline")
            return fake_pcm.copy(), 24000

        with (
            patch.object(config, "get", side_effect=lambda k, d=None: state.get(k, d)),
            patch.object(tts_manager, "get_available_active_engines", return_value=avail),
            patch.object(tts_manager, "_synthesize_dispatch", side_effect=_fake_dispatch),
            patch("core.tts_manager.apply_engine_dsp", side_effect=lambda s, sr, *_: (s, sr)),
        ):
            tts_manager.clear_audio_cache()
            try:
                s1, sr1, t1 = tts_manager._synthesize_with_fallback("Тест кешу запасного TTS")
                self.assertEqual(t1, "Piper Local")
                self.assertEqual(sr1, 24000)
                self.assertEqual(dispatch_calls, [("online:prim", "v_p"), ("piper", "v_fb")])

                # Second call when primary fails again should hit fallback cache without re-running piper dispatch
                dispatch_calls.clear()
                s2, sr2, t2 = tts_manager._synthesize_with_fallback("Тест кешу запасного TTS")
                self.assertEqual(t2, "Piper Local")
                self.assertEqual(sr2, 24000)
                self.assertEqual(dispatch_calls, [("online:prim", "v_p")])
                self.assertTrue(np.allclose(s1, s2))
            finally:
                tts_manager.clear_audio_cache()


class TextNormalizerAndUiRegressionTests(unittest.TestCase):
    def test_text_normalizer_and_sentence_splitter(self) -> None:
        long_text = (
            "Це перше достатньо довге речення для перевірки автоматичного розбиття тексту! "
            "А це вже друге речення, яке також має достатню довжину для окремого фрагмента? "
            "І нарешті третє речення завершує наш тестовий абзац."
        )
        chunks = split_into_sentences_punct(long_text)
        self.assertEqual(len(chunks), 3)
        norm_uk = normalize_tts_text("Знижка 50%", "uk")
        self.assertNotIn("50%", norm_uk)

    def test_unified_stress_marks_and_organov_default_fix(self) -> None:
        from core.text_normalizer import (
            adapt_text_for_local_tts,
            apply_default_stress_fixes,
            apply_piper_uk_stress_phonemes,
            format_stress_marks,
            strip_plus_stress_marks,
        )

        # Точковий дефолтний наголос для «органов» зі збереженням регістру
        self.assertEqual(apply_default_stress_fixes("внутренних органов."), "внутренних +органов.")
        self.assertEqual(apply_default_stress_fixes("Органов дыхания"), "+Органов дыхания")
        self.assertEqual(apply_default_stress_fixes("ОРГАНОВ"), "+ОРГАНОВ")
        # Не перезаписує явний наголос користувача і не чіпає інші слова
        self.assertEqual(apply_default_stress_fixes("+органов"), "+органов")
        self.assertEqual(apply_default_stress_fixes("орг+анов"), "орг+анов")
        self.assertEqual(apply_default_stress_fixes("орга\u0301нов"), "орга\u0301нов")
        self.assertEqual(apply_default_stress_fixes("о\u0301рганов"), "о\u0301рганов")
        self.assertEqual(apply_default_stress_fixes("органовый зал"), "органовый зал")

        # Конвертація єдиної позначки наголосу для різних движків
        self.assertEqual(format_stress_marks("з+амок и +органов", "plus"), "з+амок и +органов")
        self.assertEqual(format_stress_marks("за\u0301мок и о\u0301рганов", "plus"), "з+амок и +органов")
        self.assertEqual(format_stress_marks("з+амок и +органов", "combining"), "за\u0301мок и о\u0301рганов")
        self.assertEqual(format_stress_marks("з+амок и +органов", "strip"), "замок и органов")

        # Не ламає математику, цифри та C++
        self.assertEqual(strip_plus_stress_marks("2+2, +5, C++, a+b, без +органов"), "2+2, +5, C++, a+b, без органов")
        self.assertEqual(format_stress_marks("C++ и +5 и з+амок", "combining"), "C++ и +5 и за\u0301мок")
        self.assertEqual(normalize_tts_text("без +органов 50%", "ru"), "без +органов пятьдесят процентов")

        # Українська літера +ї при адаптації під RU-моделі (Silero v5_6_ru)
        self.assertEqual(adapt_text_for_local_tts("+їжак", "ru", uk_to_ru=True), "й+ижак")

        # Український Piper: перенесення маркера 'ˈ' у [[...]] перед цільовою голосною фонемою
        fake_uk_phonemes = {
            "органов": ["o", "r", "h", "ˈ", "a", "n", "o", "β"],
            "замок": ["z", "a", "m", "ˈ", "o", "k"],
        }
        res_uk = apply_piper_uk_stress_phonemes(
            "Без «+органов», це з+амок і зам+ок.",
            lambda w: list(fake_uk_phonemes[w]),
        )
        self.assertEqual(res_uk, "Без [[ˈorhanoβ,]] це [[zˈamok]] і [[zamˈok.]]")

    def test_holos_preserves_manual_stress_marks(self) -> None:
        from core.tts_manager import TTSEngineManager

        called: list[str] = []

        def _fake_stress(chunk: str) -> str:
            called.append(chunk)
            return chunk.replace("це", "це\u0301")

        out = TTSEngineManager._apply_holos_stress_preserving_manual(
            "це за\u0301мок і о\u0301рганов тут",
            _fake_stress,
        )
        self.assertEqual(out, "це\u0301 за\u0301мок і о\u0301рганов тут")
        self.assertEqual(called, ["це ", " і ", " тут"])

    def test_gemini_tts_models_list_is_reversed_newest_first(self) -> None:
        from core.tts_manager import TTSEngineManager

        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {
            "models": [
                {"name": "models/gemini-2.5-flash-preview-tts"},
                {"name": "models/gemini-2.5-pro-preview-tts"},
                {"name": "models/gemini-3.1-flash-tts-preview"},
                {"name": "models/gemini-3.8-flash-tts"},
                {"name": "models/gemini-3.8-flash-lite-tts"},
                {"name": "models/gemini-3.8-flash"},
            ]
        }
        with (
            patch("core.tts_manager.requests.get", return_value=fake_resp),
            patch("core.tts_manager.get_active_api_proxies", return_value=None),
        ):
            _voices, models = TTSEngineManager.fetch_gemini_tts_voices("AIza-fake-key")
        self.assertEqual(
            models,
            [
                "gemini-3.8-flash-lite-tts",
                "gemini-3.8-flash-tts",
                "gemini-3.1-flash-tts-preview",
                "gemini-2.5-pro-preview-tts",
                "gemini-2.5-flash-preview-tts",
            ],
        )

    def test_first_run_language_dialog_saves_chosen_ui_language(self) -> None:
        from PySide6.QtWidgets import QApplication
        from ui.dialogs import FirstRunLanguageDialog

        app = QApplication.instance() or QApplication([])
        saved_updates: dict = {}

        with (
            patch("ui.dialogs.translate_ui_prompt_texts", side_effect=lambda base, **_: dict(base)),
            patch("ui.dialogs.set_ui_language") as mock_set_ui,
            patch("ui.dialogs.config.update", side_effect=saved_updates.update),
            patch("core.config.sync_default_quick_phrases_for_lang"),
        ):
            dlg = FirstRunLanguageDialog("it", "Italiano", "Italian")
            de_idx = dlg.lang_combo.findData("de")
            self.assertGreaterEqual(de_idx, 0)
            dlg.lang_combo.setCurrentIndex(de_idx)
            dlg._on_use_selected()
            mock_set_ui.assert_called_with("de")
            self.assertEqual(saved_updates.get("ui_language"), "de")
            self.assertEqual(saved_updates.get("app_language"), "de")
            self.assertTrue(saved_updates.get("first_run_lang_checked"))
        _ = app

    def test_reconnect_signal_replaces_old_slot_without_duplicates(self) -> None:
        from PySide6.QtCore import QObject, Signal
        from ui.main_window import MainWindow

        class _DummyEmitter(QObject):
            sig = Signal(str)

        emitter = _DummyEmitter()
        calls_1: list[str] = []
        calls_2: list[str] = []

        MainWindow._reconnect_signal(emitter, emitter.sig, "_slot_test", calls_1.append)
        emitter.sig.emit("first")
        self.assertEqual(calls_1, ["first"])

        MainWindow._reconnect_signal(emitter, emitter.sig, "_slot_test", calls_2.append)
        emitter.sig.emit("second")
        self.assertEqual(calls_1, ["first"])
        self.assertEqual(calls_2, ["second"])

    def test_settings_transfer_zip_roundtrip_with_quick_phrase_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav_file = root / "hello.wav"
            wav_file.write_bytes(b"RIFF____WAVEfmt ")
            profile_path = root / "backup.svprofile"

            source_cfg = {
                "ui_language": "uk",
                "app_language": "uk",
                "quick_phrases": [
                    {"title": "Привіт", "text": "Привіт усім!", "emoji": "👋", "hotkey": "Alt+1", "audio_file": str(wav_file)}
                ],
                "ai_items": [
                    {"id": "ai_groq_1_9999", "provider": "groq", "api_key": "exported_secret", "model": "m1"}
                ],
            }
            payload = build_export(source_cfg, ["general", "quick_phrases", "ai_services"], include_api_keys=False)
            write_export_file(profile_path, payload)
            loaded = read_import_file(profile_path)
            self.assertEqual(loaded["format"], "SilentVoiceSettings")
            self.assertTrue(loaded.get("contains_audio"))

            dest_audio_dir = root / "imported_phrases"
            current_data = {
                "ai_items": [
                    {"id": "ai_groq_1_1111", "provider": "groq", "api_key": "keep_local_secret", "model": "old"}
                ]
            }
            applied_updates: dict = {}
            with (
                patch("core.settings_transfer.PHRASES_AUDIO_DIR", dest_audio_dir),
                patch.object(config, "_data", current_data),
                patch.object(config, "update", side_effect=applied_updates.update),
            ):
                applied = apply_import(loaded, ["general", "quick_phrases", "ai_services"], include_api_keys=False)
                self.assertEqual(applied, {"general", "quick_phrases", "ai_services"})
                self.assertEqual(applied_updates["ai_items"][0]["api_key"], "keep_local_secret")
                imported_qp = applied_updates["quick_phrases"][0]
                self.assertTrue(Path(imported_qp["audio_file"]).is_file())
                self.assertEqual(Path(imported_qp["audio_file"]).read_bytes(), b"RIFF____WAVEfmt ")

    def test_obs_animations_resolution_and_variants(self) -> None:
        self.assertGreater(len(ENTRY_ANIMATIONS), 5)
        self.assertGreater(len(MOTION_ANIMATIONS), 3)
        for spec in ENTRY_ANIMATIONS:
            aid = str(spec["id"])
            s = get_entry_spec(aid)
            params = resolve_anim_params("entry", aid, {})
            vkey = get_variant_key("entry", s, params)
            vdefs = get_variant_defaults("entry", s, params)
            self.assertIsInstance(vkey, str)
            self.assertIsInstance(vdefs, dict)
        for spec in MOTION_ANIMATIONS:
            mid = str(spec["id"])
            s = get_motion_spec(mid)
            params = resolve_anim_params("motion", mid, {})
            vkey = get_variant_key("motion", s, params)
            vdefs = get_variant_defaults("motion", s, params)
            self.assertIsInstance(vkey, str)
            self.assertIsInstance(vdefs, dict)


class SecurityHardeningTests(unittest.TestCase):
    def test_obs_server_blocks_cross_origin_csrf_and_cswsh_while_allowing_local_and_streamdeck(self) -> None:
        import asyncio
        from aiohttp.test_utils import make_mocked_request
        from core.obs_server import OBSSubtitleServer

        srv = OBSSubtitleServer()
        spoken: list[str] = []
        srv.external_speak_requested.connect(spoken.append)

        # 1. Malicious website cross-origin requests must be rejected (403)
        evil_origins = [
            {"Origin": "https://evil.example"},
            {"Origin": "http://attacker.local:8080"},
            {"Origin": "null"},
            {"Sec-Fetch-Site": "cross-site"},
            {"Referer": "https://evil.example/exploit.html"},
        ]
        for headers in evil_origins:
            req_speak = make_mocked_request("GET", "/api/speak?text=hacked", headers=headers)
            resp_speak = asyncio.run(srv._handle_api_speak(req_speak))
            self.assertEqual(resp_speak.status, 403, f"Expected 403 for headers={headers}")

            req_ws = make_mocked_request("GET", "/ws?preview=1", headers=headers)
            resp_ws = asyncio.run(srv._handle_ws(req_ws))
            self.assertEqual(resp_ws.status, 403, f"Expected 403 on /ws for headers={headers}")

        self.assertEqual(spoken, [])

        # 2. Local OBS Browser Source / Preview and headerless Stream Deck / CLI requests must succeed
        valid_headers_list = [
            {},  # Stream Deck / curl / local Python script
            {"Origin": "http://127.0.0.1:8765", "Sec-Fetch-Site": "same-origin"},
            {"Origin": "http://localhost:8765", "Referer": "http://localhost:8765/?preview=1"},
        ]
        for headers in valid_headers_list:
            req_ok = make_mocked_request("GET", "/api/speak?text=Hello", headers=headers)
            resp_ok = asyncio.run(srv._handle_api_speak(req_ok))
            self.assertEqual(resp_ok.status, 200, f"Expected 200 for headers={headers}")

        self.assertEqual(spoken, ["Hello", "Hello", "Hello"])

    def test_extract_zip_bytes_blocks_zip_slip_path_traversal(self) -> None:
        import zipfile
        from core.voice_cloner import extract_zip_bytes

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_dir = root / "unpacked"
            outside_file = root / "escaped.txt"

            # Safe archive works
            safe_zip = root / "safe.zip"
            with zipfile.ZipFile(safe_zip, "w") as zf:
                zf.writestr("subdir/ok.txt", "safe-content")
            extract_zip_bytes(safe_zip, target_dir)
            self.assertEqual((target_dir / "subdir" / "ok.txt").read_text(encoding="utf-8"), "safe-content")

            # Malicious Zip Slip archive is blocked
            evil_zip = root / "evil.zip"
            with zipfile.ZipFile(evil_zip, "w") as zf:
                zf.writestr("../escaped.txt", "pwned")
            with self.assertRaises(RuntimeError):
                extract_zip_bytes(evil_zip, target_dir)
            self.assertFalse(outside_file.exists())

    def test_settings_transfer_prevents_cross_provider_or_remote_base_url_key_exfiltration(self) -> None:
        current = [
            {"id": "ai_openai_1", "provider": "openai", "api_key": "sk-victim-openai-secret"},
            {
                "id": "ai_local_1",
                "provider": "local_api",
                "base_url": "http://127.0.0.1:1234/v1",
                "api_key": "local-secret-token",
            },
        ]
        # Crafted incoming items attempting to steal sk-victim-openai-secret or local-secret-token
        malicious_incoming = [
            {
                "id": "ai_openai_1",
                "provider": "local_api",
                "base_url": "https://attacker.example/v1",
                "api_key": "",
            },
            {
                "id": "ai_local_1",
                "provider": "local_api",
                "base_url": "https://attacker.example/v1",
                "api_key": "",
            },
        ]
        merged = _merge_service_items(malicious_incoming, current, allow_imported_keys=False)
        self.assertEqual(merged[0]["api_key"], "")
        self.assertEqual(merged[1]["api_key"], "")

        # Legitimate local_api import keeping loopback base_url preserves the local key
        legit_incoming = [
            {
                "id": "ai_local_1",
                "provider": "local_api",
                "base_url": "http://127.0.0.1:1234/v1",
                "api_key": "",
            }
        ]
        merged_legit = _merge_service_items(legit_incoming, current, allow_imported_keys=False)
        self.assertEqual(merged_legit[0]["api_key"], "local-secret-token")

    def test_user_locales_dir_ai_prompt_and_quick_phrases_sync(self) -> None:
        import copy
        import json
        from core.ai_corrector import get_default_ai_prompt
        from core.config import DEFAULT_QUICK_PHRASES, sync_default_quick_phrases_for_lang

        with tempfile.TemporaryDirectory() as tmp:
            fake_data_dir = Path(tmp)
            user_locales = fake_data_dir / "locales"
            user_locales.mkdir(parents=True, exist_ok=True)
            it_locale = user_locales / "it.json"
            it_locale.write_text(
                json.dumps(
                    {
                        "_meta": {
                            "code": "it",
                            "short": "IT",
                            "name": "Italiano",
                            "ai_prompt": "Correttore delicato italiano.",
                            "quick_phrases": {"qp_1": ["Ciao", "Ciao a tutti!"]},
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with (
                patch("core.i18n.get_user_locales_dir", return_value=user_locales),
                patch("core.config.DATA_DIR", fake_data_dir),
            ):
                self.assertEqual(get_default_ai_prompt("it"), "Correttore delicato italiano.")

                phrases = copy.deepcopy(DEFAULT_QUICK_PHRASES)
                with patch.object(config, "_data", {"quick_phrases": phrases}), patch.object(config, "save"):
                    self.assertTrue(sync_default_quick_phrases_for_lang("it"))
                    self.assertEqual(config.get("quick_phrases")[0]["title"], "Ciao")
                    # Switching back from user-translated locale to English recognizes "Ciao" as default
                    self.assertTrue(sync_default_quick_phrases_for_lang("en"))
                    self.assertEqual(config.get("quick_phrases")[0]["title"], "Hello")

    def test_updater_version_parsing_github_gitlab_and_sha256_verification(self) -> None:
        import hashlib
        from core.updater import (
            UpdateReleaseInfo,
            check_for_updates,
            download_update_installer,
            is_newer_version,
            parse_repo_source,
            parse_version_tuple,
        )

        self.assertEqual(parse_version_tuple("v1.0.1"), (1, 0, 1))
        self.assertTrue(is_newer_version("v1.0.1", "1.0.0"))
        self.assertTrue(is_newer_version("1.2.0", "1.1.9"))
        self.assertFalse(is_newer_version("1.0.0", "1.0.0"))
        self.assertFalse(is_newer_version("0.9.9", "1.0.0"))

        self.assertEqual(parse_repo_source("Blaite/SilentVoice"), ("github", "github.com", "Blaite/SilentVoice"))
        self.assertEqual(
            parse_repo_source("https://github.com/Blaite/SilentVoice/releases"),
            ("github", "github.com", "Blaite/SilentVoice"),
        )
        self.assertEqual(
            parse_repo_source("https://gitlab.com/Blaite/SilentVoice/-/releases"),
            ("gitlab", "gitlab.com", "Blaite/SilentVoice"),
        )

        fake_exe_bytes = b"MZ" + b"\x90" * 2048
        fake_sha256 = hashlib.sha256(fake_exe_bytes).hexdigest()

        class _FakeResp:
            def __init__(self, status_code: int, json_data=None, text_data: str = "", content_bytes: bytes = b"") -> None:
                self.status_code = status_code
                self._json = json_data
                self.text = text_data
                self._bytes = content_bytes
                self.headers = {"Content-Length": str(len(content_bytes))}

            def json(self):
                return self._json

            def iter_content(self, chunk_size: int = 65536):
                yield self._bytes

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        # Test GitHub check + GitLab fallback
        def _fake_get(url: str, **kwargs):
            if "api.github.com" in url:
                return _FakeResp(
                    200,
                    json_data={
                        "tag_name": "v1.1.0",
                        "name": "SilentVoice 1.1.0",
                        "body": "New features!",
                        "html_url": "https://github.com/Blaite/SilentVoice/releases/tag/v1.1.0",
                        "assets": [
                            {
                                "name": "SilentVoice_Setup_1.1.0.exe",
                                "browser_download_url": "https://github.com/Blaite/SilentVoice/releases/download/v1.1.0/SilentVoice_Setup_1.1.0.exe",
                                "size": len(fake_exe_bytes),
                                "digest": f"sha256:{fake_sha256}",
                            }
                        ],
                    },
                )
            if "SilentVoice_Setup_1.1.0.exe" in url:
                return _FakeResp(200, content_bytes=fake_exe_bytes)
            return _FakeResp(404)

        with patch("core.updater.requests.get", side_effect=_fake_get):
            rel, is_new = check_for_updates(custom_source="https://github.com/Blaite/SilentVoice")
            self.assertTrue(is_new)
            self.assertEqual(rel.version, "1.1.0")
            self.assertEqual(rel.sha256, fake_sha256)
            self.assertEqual(rel.source, "GitHub")

            with tempfile.TemporaryDirectory() as tmp:
                with patch("core.updater.UPDATES_DIR", Path(tmp)):
                    dl_path = download_update_installer(rel)
                    self.assertTrue(dl_path.is_file())
                    self.assertEqual(dl_path.read_bytes(), fake_exe_bytes)

                    # Bad SHA-256 must raise RuntimeError and clean up .part file
                    bad_rel = UpdateReleaseInfo(
                        version="1.1.0",
                        tag_name="v1.1.0",
                        title="t",
                        notes="",
                        installer_url=rel.installer_url,
                        installer_name="Bad_Setup.exe",
                        sha256="0" * 64,
                    )
                    with self.assertRaises(RuntimeError):
                        download_update_installer(bad_rel)
                    self.assertFalse((Path(tmp) / "Bad_Setup.exe").exists())


class WarmupAndDspOptimizationTests(unittest.TestCase):
    def test_vectorized_noise_gate_attenuates_silence_and_preserves_speech(self) -> None:
        from core.tts_manager import _fx_noise_gate

        for sr in (48000, 48040):  # covers both even (1200) and odd (1201) hold_samples
            t = np.linspace(0.0, 0.5, sr // 2, endpoint=False, dtype=np.float32)
            speech = 0.25 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
            silence = 0.0005 * np.sin(2.0 * np.pi * 60.0 * t).astype(np.float32)
            combined = np.concatenate([silence, speech, silence])

            out = _fx_noise_gate(combined, sr)
            self.assertEqual(out.shape, combined.shape)
            self.assertEqual(out.dtype, np.float32)
            # Leading silence is attenuated; middle active speech stays strong
            self.assertLess(float(np.max(np.abs(out[: sr // 4]))), 1e-4)
            self.assertGreater(float(np.sqrt(np.mean(out[sr // 2 : sr] ** 2))), 0.12)

    def test_holos_fast_stress_trie_lookup_avoids_stanza_import(self) -> None:
        import sys
        from core.tts_manager import TTSEngineManager

        trie_path = TTSEngineManager._find_uk_stress_trie_path()
        self.assertIsNotNone(trie_path)
        assert trie_path is not None
        self.assertTrue(trie_path.is_file())
        # Finding the trie file must not have triggered importing stanza
        if "stanza" not in sys.modules:
            _ = TTSEngineManager._find_uk_stress_trie_path()
            self.assertNotIn("stanza", sys.modules)

    def test_smart_warmup_warms_fallback_and_openvoice_when_primary_is_online(self) -> None:
        state = {
            "active_engine": "online:edge_1",
            "active_voice": "uk-UA-PolinaNeural",
            "fallback_engine": "silero",
            "fallback_voice": "silero_ua_tetiana",
            "openvoice_enabled": True,
            "openvoice_ref_path": "",
            "openvoice_device": "cpu",
            "app_language": "uk",
            "twitch_read_viewers": False,
        }
        orig_warmed = set(tts_manager._warmed_models)
        orig_queue_flag = tts_manager._is_synthesizing
        synth_calls: list[tuple[str, str]] = []
        ov_warm_calls: list[str] = []

        def _fake_installed(eng_id: str) -> bool:
            return eng_id == "silero"

        def _fake_synth(eng_id: str, v_id: str, txt: str, spd: float):
            synth_calls.append((eng_id, v_id))
            return np.zeros(100, dtype=np.float32), 24000

        try:
            tts_manager._warmed_models.clear()
            tts_manager._is_synthesizing = False
            with tempfile.TemporaryDirectory() as tmp:
                eng_dir = Path(tmp)
                (eng_dir / "silero").mkdir(parents=True)
                (eng_dir / "silero" / "v5_cis_ext.pt").write_bytes(b"pt")
                with (
                    patch("core.tts_manager.ENGINES_DIR", eng_dir),
                    patch.object(config, "get", side_effect=lambda k, d=None: state.get(k, d)),
                    patch.object(tts_manager, "is_engine_installed", side_effect=_fake_installed),
                    patch.object(tts_manager, "_get_system_resources_for_warmup", return_value=(8192, False, 5.0)),
                    patch.object(tts_manager, "_synthesize_dispatch", side_effect=_fake_synth),
                    patch("core.voice_cloner.voice_cloner.models_present", return_value=True),
                    patch("core.voice_cloner.voice_cloner._warmed_key", None),
                    patch("core.voice_cloner.voice_cloner.warmup", side_effect=lambda: ov_warm_calls.append("ok")),
                ):
                    targets = tts_manager._resolve_local_warmup_targets()
                    self.assertEqual([(t[0], t[1], t[3]) for t in targets], [("silero", "silero_ua_tetiana", "fallback")])
                    tts_manager._run_smart_warmup("test", 0)
                    self.assertEqual(synth_calls, [("silero", "silero_ua_tetiana")])
                    self.assertEqual(ov_warm_calls, ["ok"])
        finally:
            tts_manager._warmed_models = orig_warmed
            tts_manager._is_synthesizing = orig_queue_flag

    def test_online_warmup_prewarms_provider_once_and_dsp_on_custom_voice_switch(self) -> None:
        from core.tts_manager import tts_manager

        orig_warmed = set(tts_manager._warmed_models)
        orig_queue_flag = tts_manager._is_synthesizing
        provider_calls: list[str] = []
        dsp_calls: list[tuple[str, str]] = []
        synth_calls: list[str] = []
        state = {
            "active_engine": "online:edge_1",
            "active_voice": "uk-UA-PolinaNeural",
            "fallback_engine": "",
            "fallback_voice": "",
            "openvoice_enabled": False,
            "twitch_read_viewers": False,
            "online_items": [{"id": "edge_1", "provider": "edge", "enabled": True, "installed": True}],
        }
        try:
            tts_manager._warmed_models.clear()
            tts_manager._is_synthesizing = False
            with (
                patch.object(config, "get", side_effect=lambda k, d=None: state.get(k, d)),
                patch.object(tts_manager, "_get_system_resources_for_warmup", return_value=(8192, False, 5.0)),
                patch.object(tts_manager, "_prewarm_online_provider", side_effect=lambda prov: provider_calls.append(prov)),
                patch.object(tts_manager, "_prewarm_dsp_libs", side_effect=lambda e, v: dsp_calls.append((e, v))),
                patch.object(tts_manager, "_dsp_needs_prewarm", side_effect=lambda e, v: v.endswith("::custom")),
                patch.object(tts_manager, "_synthesize_dispatch", side_effect=lambda *a: synth_calls.append("synth")),
                patch("core.voice_cloner.voice_cloner.models_present", return_value=False),
            ):
                # 1. First warmup warms Edge provider once without running synthesis
                tts_manager._run_smart_warmup("startup", 0)
                self.assertEqual(provider_calls, ["edge"])
                self.assertEqual(dsp_calls, [("online:edge_1", "uk-UA-PolinaNeural")])
                self.assertEqual(synth_calls, [])

                # 2. Switching to Polina Custom does NOT re-warm Edge provider, only prewarms DSP libs
                state["active_voice"] = "uk-UA-PolinaNeural::custom"
                tts_manager._run_smart_warmup("voice_changed", 0)
                self.assertEqual(provider_calls, ["edge"])
                self.assertEqual(
                    dsp_calls,
                    [("online:edge_1", "uk-UA-PolinaNeural"), ("online:edge_1", "uk-UA-PolinaNeural::custom")],
                )
        finally:
            tts_manager._warmed_models = orig_warmed
            tts_manager._is_synthesizing = orig_queue_flag

    def test_pyinstaller_spec_preserves_svg_support_and_icons_render(self) -> None:
        from PySide6.QtGui import QIcon
        from PySide6.QtWidgets import QApplication

        spec_path = Path(__file__).resolve().parents[1] / "SilentVoice.spec"
        spec_src = spec_path.read_text(encoding="utf-8")
        seg = spec_src.split("_ALLOWED_PYSIDE6_DLLS = ")[1].split("\na.binaries =")[0]
        ns: dict = {}
        exec("_ALLOWED_PYSIDE6_DLLS = " + seg, ns)
        is_unwanted = ns["_is_unwanted_entry"]

        self.assertIn('"PySide6.QtSvg"', spec_src.split("excludes = [")[0])
        self.assertNotIn('"PySide6.QtSvg"', spec_src.split("excludes = [")[1].split("a = Analysis(")[0])
        for rel in (
            r"PySide6\Qt6Svg.dll",
            r"PySide6\plugins\imageformats\qsvg.dll",
            r"PySide6\plugins\iconengines\qsvgicon.dll",
        ):
            self.assertFalse(is_unwanted(rel, rel), f"{rel} must not be stripped from bundle")

        _app = QApplication.instance() or QApplication([])
        assets_dir = Path(__file__).resolve().parents[1] / "assets"
        for icon_name in ("edit.svg", "delete.svg"):
            pm = QIcon(str(assets_dir / icon_name)).pixmap(16, 16)
            self.assertFalse(pm.isNull(), f"{icon_name} failed to render")

    def test_history_list_renders_save_to_quick_phrases_button_and_adds_phrase(self) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication, QDialog, QPushButton
        from ui.main_window import MainWindow

        _app = QApplication.instance() or QApplication([])
        win = MainWindow.__new__(MainWindow)
        win.keyboard_hook = MagicMock()
        cfg_state = {
            "quick_phrases": [],
            "recent_phrases": ["Привіт усім на стрімі!"],
        }

        def _fake_get(k, d=None):
            return cfg_state.get(k, d)

        def _fake_set(k, v):
            cfg_state[k] = v

        saved_dlg_result = {
            "title": "Привіт усім на стрімі!",
            "text": "Привіт усім на стрімі!",
            "hotkey": "",
            "emoji": "⭐",
            "audio_file": "",
        }
        fake_dlg = MagicMock()
        fake_dlg.exec.return_value = QDialog.DialogCode.Accepted
        fake_dlg.get_result.return_value = saved_dlg_result

        with (
            patch("ui.main_window.config.get", side_effect=_fake_get),
            patch("ui.main_window.config.set", side_effect=_fake_set),
            patch("core.config.ConfigManager.data", new_callable=unittest.mock.PropertyMock, return_value=cfg_state),
            patch("ui.main_window.QuickPhraseDialog", return_value=fake_dlg) as dlg_cls,
        ):
            dlg_cls.DialogCode = QDialog.DialogCode
            page = win._build_page_phrases()
            self.assertEqual(win.history_list.count(), 1)
            item = win.history_list.item(0)
            self.assertEqual(item.data(Qt.ItemDataRole.UserRole), "Привіт усім на стрімі!")
            row_w = win.history_list.itemWidget(item)
            self.assertIsNotNone(row_w)
            btns = row_w.findChildren(QPushButton)
            self.assertEqual(len(btns), 1)
            btns[0].click()

            dlg_cls.assert_called_once()
            passed_data = dlg_cls.call_args[0][1]
            self.assertEqual(passed_data["text"], "Привіт усім на стрімі!")
            self.assertEqual(len(cfg_state["quick_phrases"]), 1)
            btns_after = win.history_list.itemWidget(win.history_list.item(0)).findChildren(QPushButton)
            self.assertIn("★", btns_after[0].text())

    def test_hide_from_capture_default_false_and_overlay_ready_indicator_is_oval(self) -> None:
        from PySide6.QtGui import QPainter
        from PySide6.QtWidgets import QApplication
        from core.config import DEFAULT_CONFIG
        from ui.overlay_window import ScreenOverlayWindow

        self.assertFalse(DEFAULT_CONFIG["hide_from_capture"])

        _app = QApplication.instance() or QApplication([])
        ov = ScreenOverlayWindow()
        drawn_rects: list[tuple[float, float]] = []

        class _MockPainter:
            def setPen(self, *_args): pass
            def setBrush(self, *_args): pass
            def drawEllipse(self, rect):
                drawn_rects.append((float(rect.width()), float(rect.height())))

        ov._paint_subtle_ready_indicator(_MockPainter(), 100, "#7C5CFF")  # type: ignore[arg-type]
        self.assertTrue(drawn_rects)
        self.assertTrue(any(w > h * 1.5 for w, h in drawn_rects))

    def test_hotkeys_page_feedback_volume_slider_updates_config_without_arg_error(self) -> None:
        from PySide6.QtWidgets import QApplication
        from ui.main_window import MainWindow

        _app = QApplication.instance() or QApplication([])
        win = MainWindow.__new__(MainWindow)
        win.keyboard_hook = MagicMock()
        win.fallback_window = MagicMock()
        saved: dict[str, object] = {"sound_feedback": True, "cancel_sound_enabled": True, "sound_feedback_volume": 100}

        with (
            patch("ui.main_window.config.get", side_effect=lambda k, d=None: saved.get(k, d)),
            patch("ui.main_window.config.set", side_effect=lambda k, v: saved.__setitem__(k, v)),
        ):
            page = win._build_page_hotkeys()
            win.sld_feedback_vol.setValue(65)
            self.assertEqual(saved["sound_feedback_volume"], 65)
            page.deleteLater()

    def test_game_mode_detection_mouse_guard_draft_ttl_and_ui_subpanel(self) -> None:
        import ctypes
        from PySide6.QtWidgets import QApplication
        from core.win_api import (
            KeyboardHookManager,
            _GameMouseGuard,
            _RAWMOUSE,
            classify_foreground_game,
            classify_foreground_game_detail,
        )
        from ui.main_window import MainWindow

        self.assertEqual(ctypes.sizeof(_RAWMOUSE), 24)

        # 1. Game classifier: detects D3D fullscreen, Steam library, and borderless locked cursor; ignores browsers/launchers/self
        self.assertTrue(classify_foreground_game(
            is_self=False, exe_name="game.exe", exe_path=r"C:\MyApp\game.exe",
            quns_state=3, is_fullscreen=True, cursor_hidden=False, cursor_clipped=False,
        ))
        self.assertTrue(classify_foreground_game(
            is_self=False, exe_name="cs2.exe", exe_path=r"C:\Program Files (x86)\Steam\steamapps\common\CS2\cs2.exe",
            quns_state=5, is_fullscreen=False, cursor_hidden=False, cursor_clipped=False,
        ))
        self.assertTrue(classify_foreground_game(
            is_self=False, exe_name="valorant.exe", exe_path=r"D:\Val\valorant.exe",
            quns_state=2, is_fullscreen=True, cursor_hidden=True, cursor_clipped=False,
        ))
        self.assertFalse(classify_foreground_game(
            is_self=False, exe_name="chrome.exe", exe_path=r"C:\Program Files\Google\Chrome\chrome.exe",
            quns_state=3, is_fullscreen=True, cursor_hidden=True, cursor_clipped=True,
        ))
        for player_exe in ("mpv.exe", "mpvnet.exe"):
            self.assertFalse(classify_foreground_game(
                is_self=False, exe_name=player_exe, exe_path=rf"C:\Tools\{player_exe}",
                quns_state=3, is_fullscreen=True, cursor_hidden=True, cursor_clipped=False,
            ))
        self.assertFalse(classify_foreground_game(
            is_self=False, exe_name="steam.exe", exe_path=r"C:\Program Files (x86)\Steam\steam.exe",
            quns_state=5, is_fullscreen=True, cursor_hidden=False, cursor_clipped=False,
        ))
        self.assertFalse(classify_foreground_game(
            is_self=True, exe_name="python.exe", exe_path=r"C:\Python\python.exe",
            quns_state=3, is_fullscreen=True, cursor_hidden=True, cursor_clipped=True,
        ))

        # 2. Mouse guard: click or flick triggers; micro-jitter does not; click_only ignores motion
        guard = _GameMouseGuard("medium")
        guard.reset(now=10.0)
        self.assertFalse(guard.feed(10, 5, 0, now=10.06))
        self.assertTrue(guard.feed(220, 0, 0, now=10.09))
        self.assertTrue(guard.feed(0, 0, 0x0001, now=10.10))  # LMB down
        guard.set_sensitivity("click_only")
        guard.reset(now=11.0)
        self.assertFalse(guard.feed(500, 500, 0, now=11.08))
        self.assertTrue(guard.feed(0, 0, 0x0004, now=11.09))   # RMB down

        # 3. Auto-detect gating + emergency cancel draft save & TTL expiration
        mgr = KeyboardHookManager()
        mgr.update_from_config({
            "game_mode_enabled": True,
            "game_auto_detect_only": True,
            "game_emergency_cancel": True,
            "game_mouse_sensitivity": "medium",
            "game_save_draft": True,
            "game_draft_ttl_sec": 30,
        })
        with patch("core.win_api.probe_foreground_game_state", return_value=(False, "")):
            mgr.set_typing_mode(True)
            self.assertFalse(mgr._should_arm_game_guard())  # outside game -> normal mode
            mgr.set_typing_mode(False)

        with patch("core.win_api.probe_foreground_game_state", return_value=(True, "strong")):
            mgr.set_typing_mode(True)
            self.assertTrue(mgr._should_arm_game_guard())
            self.assertFalse(mgr._game_guard_weak_mode)
            mgr._buffer = "привіт тіммейт"
            mgr._emergency_cancel_typing()
            self.assertFalse(mgr.is_typing)
            self.assertEqual(mgr.current_text, "")
            # Re-open immediately -> draft restored
            mgr.set_typing_mode(True)
            self.assertEqual(mgr.current_text, "привіт тіммейт")
            mgr.set_typing_mode(False)

            # Re-test TTL expiry
            mgr.set_typing_mode(True)
            mgr._buffer = "старий текст"
            mgr._emergency_cancel_typing()
            mgr._saved_draft_expires_at = 0.0
            mgr.set_typing_mode(True)
            self.assertEqual(mgr.current_text, "")
            mgr.set_typing_mode(False)

        # 3b. Windows GameConfigStore strong signal & cursor-reappear protection on weak_hidden apps
        self.assertEqual(
            classify_foreground_game_detail(
                is_self=False, exe_name="javaw.exe", exe_path=r"C:\Users\u\Downloads\game\javaw.exe",
                quns_state=5, is_fullscreen=False, cursor_hidden=False, cursor_clipped=False, in_game_store=True,
            ),
            (True, "strong"),
        )
        self.assertEqual(
            classify_foreground_game_detail(
                is_self=False, exe_name="unknown_player.exe", exe_path=r"C:\Apps\unknown_player.exe",
                quns_state=5, is_fullscreen=True, cursor_hidden=True, cursor_clipped=False, in_game_store=False,
            ),
            (True, "weak_hidden"),
        )
        with patch("core.win_api.probe_foreground_game_state", return_value=(True, "weak_hidden")):
            mgr.set_typing_mode(True)
            self.assertTrue(mgr._game_guard_armed)
            self.assertTrue(mgr._game_guard_weak_mode)
            mgr._buffer = "дивлюсь фільм"
            # Cursor reappeared on mouse movement -> video player, keep typing and disarm guard
            with patch("core.win_api._is_cursor_still_confined_or_hidden", return_value=False):
                mgr._verify_weak_game_trigger()
            self.assertTrue(mgr.is_typing)
            self.assertFalse(mgr._game_guard_armed)
            self.assertEqual(mgr.current_text, "дивлюсь фільм")
            mgr.set_typing_mode(False)

            # Cursor stays hidden -> real fullscreen game -> emergency cancel fires
            mgr.set_typing_mode(True)
            mgr._buffer = "у бою"
            with patch("core.win_api._is_cursor_still_confined_or_hidden", return_value=True):
                mgr._verify_weak_game_trigger()
            self.assertFalse(mgr.is_typing)
            mgr.set_typing_mode(False)

        # 4. Defaults enabled + TTL=20 + UI toggle collapses/reveals subpanel without redundant auto-detect checkbox
        from core.config import DEFAULT_CONFIG
        self.assertTrue(DEFAULT_CONFIG["game_mode_enabled"])
        self.assertEqual(DEFAULT_CONFIG["game_mouse_sensitivity"], "low")
        self.assertEqual(DEFAULT_CONFIG["game_draft_ttl_sec"], 20)
        self.assertEqual(DEFAULT_CONFIG["hotkey_toggle_game_mode"], "Ctrl+Shift+G")

        _app = QApplication.instance() or QApplication([])
        win = MainWindow.__new__(MainWindow)
        win.keyboard_hook = MagicMock()
        win.fallback_window = MagicMock()
        cfg_state: dict[str, object] = {}
        win._reconnect_signal = MagicMock()
        with (
            patch("ui.main_window.config.get", side_effect=lambda k, d=None: cfg_state.get(k, d)),
            patch("ui.main_window.config.set", side_effect=lambda k, v: cfg_state.__setitem__(k, v)),
            patch("ui.main_window.config.update", side_effect=lambda d, **_kw: cfg_state.update(d)),
            patch("ui.main_window.config.schedule_save"),
            patch("core.config.ConfigManager.data", new_callable=unittest.mock.PropertyMock, return_value=cfg_state),
        ):
            page = win._build_page_hotkeys()
            self.assertFalse(hasattr(win, "chk_game_auto_detect"))
            self.assertFalse(hasattr(win, "chk_game_emergency_cancel"))
            self.assertTrue(hasattr(win, "lbl_game_emergency_cancel"))
            self.assertTrue(win.chk_game_mode.isChecked())
            self.assertFalse(win.game_mode_subpanel.isHidden())
            self.assertEqual(win.game_mouse_sens_combo.currentData(), "low")
            self.assertEqual(win.game_draft_ttl_spin.value(), 20)
            self.assertEqual(win._hk_edits["hotkey_toggle_game_mode"].text(), "Ctrl+Shift+G")
            win.chk_game_mode.setChecked(False)
            self.assertTrue(win.game_mode_subpanel.isHidden())
            self.assertFalse(cfg_state["game_mode_enabled"])
            win.chk_game_mode.setChecked(True)
            self.assertFalse(win.game_mode_subpanel.isHidden())
            self.assertTrue(cfg_state["game_mode_enabled"])
            with patch("ui.main_window.audio_router.play_feedback_beep"):
                win._on_toggle_game_mode_hotkey()
                self.assertFalse(cfg_state["game_mode_enabled"])
                self.assertFalse(win.chk_game_mode.isChecked())
                win._on_toggle_game_mode_hotkey()
                self.assertTrue(cfg_state["game_mode_enabled"])
                self.assertTrue(win.chk_game_mode.isChecked())
            win.game_draft_ttl_spin.setValue(45)
            self.assertEqual(cfg_state["game_draft_ttl_sec"], 45)
            page.deleteLater()

        # 5. General accidental-close draft retention (default True, 10s) outside games & overlay UI
        self.assertTrue(DEFAULT_CONFIG["save_draft_on_close"])
        self.assertEqual(DEFAULT_CONFIG["draft_keep_ttl_sec"], 10)
        self.assertFalse(DEFAULT_CONFIG["overlay_draggable"])

        mgr.update_from_config({
            "game_mode_enabled": False,
            "save_draft_on_close": True,
            "draft_keep_ttl_sec": 10,
        })
        mgr.set_typing_mode(True)
        mgr._buffer = "недописане речення"
        mgr.set_typing_mode(False)
        mgr.set_typing_mode(True)
        self.assertEqual(mgr.current_text, "недописане речення")
        mgr._buffer = "протерміноване"
        mgr.set_typing_mode(False)
        mgr._saved_draft_expires_at = 0.0
        mgr.set_typing_mode(True)
        self.assertEqual(mgr.current_text, "")
        mgr.set_typing_mode(False)

        win._reconnect_signal = MagicMock()
        with (
            patch("ui.main_window.config.get", side_effect=lambda k, d=None: cfg_state.get(k, d)),
            patch("ui.main_window.config.set", side_effect=lambda k, v: cfg_state.__setitem__(k, v)),
            patch("ui.main_window.config.update", side_effect=lambda d, **_kw: cfg_state.update(d)),
            patch("ui.main_window.config.schedule_save"),
            patch("core.config.ConfigManager.data", new_callable=unittest.mock.PropertyMock, return_value=cfg_state),
        ):
            ov_page = win._build_page_overlay()
            self.assertIsNone(win.ov_style_combo.findData("interactive_box") if win.ov_style_combo.findData("interactive_box") >= 0 else None)
            self.assertFalse(win.chk_overlay_draggable.isChecked())
            self.assertTrue(win.ov_elements_op_lbl.isHidden())
            self.assertTrue(win.ov_elements_op_slider.parentWidget().isHidden())
            self.assertEqual(win.ov_elements_op_slider.value(), 30)
            from PySide6.QtWidgets import QGridLayout
            grid_layout = win.ov_elements_op_lbl.parentWidget().findChild(QGridLayout)
            self.assertIsNotNone(grid_layout)
            mw_row = grid_layout.getItemPosition(grid_layout.indexOf(win.ov_max_width_slider.parentWidget()))[0]
            el_row = grid_layout.getItemPosition(grid_layout.indexOf(win.ov_elements_op_lbl))[0]
            self.assertEqual(el_row, grid_layout.rowCount() - 1)
            self.assertGreater(el_row, mw_row)
            self.assertTrue(win.chk_save_draft_on_close.isChecked())
            self.assertEqual(win.spin_draft_keep_ttl.value(), 10)
            self.assertTrue(win.spin_draft_keep_ttl.isEnabled())
            win.spin_draft_keep_ttl.setValue(12)
            self.assertEqual(cfg_state["draft_keep_ttl_sec"], 12)
            win.chk_save_draft_on_close.setChecked(False)
            self.assertFalse(win.spin_draft_keep_ttl.isEnabled())
            self.assertFalse(cfg_state["save_draft_on_close"])
            win.overlay = MagicMock()
            idx_iw = win.ov_style_combo.findData("interactive_window")
            idx_txt = win.ov_style_combo.findData("text_only")
            idx_box = win.ov_style_combo.findData("boxed")
            self.assertGreaterEqual(idx_iw, 0)
            # Was False before interactive_window -> turns True there, then restores False on leave
            win.ov_style_combo.setCurrentIndex(idx_iw)
            self.assertFalse(win.ov_elements_op_lbl.isHidden())
            self.assertFalse(win.ov_elements_op_slider.parentWidget().isHidden())
            win.fallback_window.isVisible.return_value = True
            win.ov_elements_op_slider.setValue(60)
            win.fallback_window.open_and_focus.assert_not_called()
            win.fallback_window.reposition_only.assert_called()
            self.assertFalse(cfg_state.get("continuous_chat_mode", False))
            self.assertFalse(win.chk_continuous_chat.isChecked())
            self.assertTrue(cfg_state["overlay_draggable"])
            self.assertTrue(win.chk_overlay_draggable.isChecked())
            win.fallback_window.sync_draggable_cursor.assert_called()
            win.ov_style_combo.setCurrentIndex(idx_txt)
            self.assertTrue(win.ov_elements_op_lbl.isHidden())
            self.assertTrue(win.ov_elements_op_slider.parentWidget().isHidden())
            self.assertFalse(cfg_state["overlay_draggable"])
            self.assertFalse(win.chk_overlay_draggable.isChecked())
            self.assertFalse(cfg_state.get("continuous_chat_mode", False))
            self.assertFalse(win.chk_continuous_chat.isChecked())
            # Was True before interactive_window -> stays True after leaving
            win.chk_overlay_draggable.setChecked(True)
            win.chk_continuous_chat.setChecked(True)
            win.ov_style_combo.setCurrentIndex(idx_iw)
            win.ov_style_combo.setCurrentIndex(idx_box)
            self.assertTrue(cfg_state["overlay_draggable"])
            self.assertTrue(win.chk_overlay_draggable.isChecked())
            self.assertTrue(cfg_state["continuous_chat_mode"])
            self.assertTrue(win.chk_continuous_chat.isChecked())
            ov_page.deleteLater()

        from PySide6.QtWidgets import QPlainTextEdit
        from ui.overlay_window import FallbackInputWindow
        fb_cfg: dict[str, object] = {}
        with (
            patch("ui.overlay_window.config.get", side_effect=lambda k, d=None: fb_cfg.get(k, d)),
            patch("ui.overlay_window.config.update", side_effect=lambda d: fb_cfg.update(d)),
        ):
            fb = FallbackInputWindow()
            self.assertIsInstance(fb.input_edit, QPlainTextEdit)
            self.assertFalse(hasattr(fb, "chat_mode_btn"))
            self.assertEqual(fb.top_lbl.text(), "")
            self.assertEqual(fb.input_edit.placeholderText(), "")
            self.assertEqual(fb.send_btn.text(), "🔊")
            self.assertEqual(fb.repeat_btn.text(), "🔁")
            self.assertEqual(fb.stop_btn.text(), "🛑")
            self.assertEqual(fb.close_btn.text(), "✕")
            for btn in (fb.send_btn, fb.repeat_btn, fb.stop_btn, fb.close_btn):
                self.assertEqual(btn.size().width(), 24)
                self.assertEqual(btn.size().height(), 24)
                self.assertTrue(bool(btn.toolTip().strip()))
            fb.resize(520, 180)
            fb._persist_window_size()
            self.assertEqual(fb_cfg["overlay_input_width"], 520)
            self.assertEqual(fb_cfg["overlay_input_height"], 180)
            fb.input_edit.setText("чернетка")
            fb._on_close_clicked()
            self.assertEqual(fb._saved_draft_text, "чернетка")
            fb.open_and_focus()
            self.assertEqual(fb.input_edit.text(), "чернетка")
            fb.retranslate_ui()
            self.assertEqual(fb.top_lbl.text(), "")
            self.assertEqual(fb.input_edit.placeholderText(), "")
            fb.deleteLater()

        # 6. Real Win32 raw message window & Raw Input registration smoke test
        if sys.platform == "win32":
            raw_mgr = KeyboardHookManager()
            hwnd = raw_mgr._create_raw_message_window()
            try:
                self.assertGreater(hwnd, 0)
                self.assertTrue(raw_mgr._set_raw_mouse_registered(hwnd, True))
                self.assertTrue(raw_mgr._set_raw_mouse_registered(hwnd, False))
            finally:
                raw_mgr._raw_hwnd = hwnd
                raw_mgr._destroy_raw_message_window()

    def test_multi_monitor_overlay_and_input_window_remember_target_screen(self) -> None:
        from PySide6.QtCore import QRect
        from PySide6.QtWidgets import QApplication
        from ui.overlay_window import (
            FallbackInputWindow,
            ScreenOverlayWindow,
            calc_screen_relative_offsets,
            resolve_screen_by_name,
        )

        app = QApplication.instance() or QApplication([])
        self.assertIsNotNone(app)

        mon1 = MagicMock()
        mon1.name.return_value = r"\\.\DISPLAY1"
        mon1.geometry.return_value = QRect(0, 0, 1920, 1080)

        mon2 = MagicMock()
        mon2.name.return_value = r"\\.\DISPLAY2"
        mon2.geometry.return_value = QRect(1920, 0, 2560, 1440)

        cfg_data: dict[str, object] = {}

        def fake_screen_at(pt):
            for s in (mon1, mon2):
                if s.geometry().contains(pt):
                    return s
            return None

        with (
            patch("ui.overlay_window.QGuiApplication.screens", return_value=[mon1, mon2]),
            patch("ui.overlay_window.QGuiApplication.primaryScreen", return_value=mon1),
            patch("ui.overlay_window.QGuiApplication.screenAt", side_effect=fake_screen_at),
            patch("ui.overlay_window.config.get", side_effect=lambda k, d=None: cfg_data.get(k, d)),
            patch("ui.overlay_window.config.update", side_effect=lambda d, **_kw: cfg_data.update(d)),
        ):
            # Dragged onto center-bottom of monitor 2
            dragged_rect = QRect(1920 + (2560 - 700) // 2 + 120, 1440 - 110 - 95, 700, 110)
            sname, x_off, y_off = calc_screen_relative_offsets(dragged_rect)
            self.assertEqual(sname, r"\\.\DISPLAY2")
            self.assertEqual(x_off, 120)
            self.assertEqual(y_off, 95)

            cfg_data.update({
                "overlay_screen_name": sname,
                "overlay_x_offset": x_off,
                "overlay_y_offset": y_off,
                "overlay_input_width": 700,
                "overlay_input_height": 110,
            })

            fb = FallbackInputWindow()
            fb.reposition_only()
            self.assertEqual(fb.x(), 1920 + (2560 - 700) // 2 + 120)
            self.assertEqual(fb.y(), 1440 - 110 - 95)
            fb.deleteLater()

            ov = ScreenOverlayWindow()
            ov.reposition()
            self.assertGreaterEqual(ov.x(), 1920)
            self.assertLess(ov.x(), 1920 + 2560)
            ov.deleteLater()

        # When monitor 2 is disconnected, fallback cleanly selects primary monitor 1
        with (
            patch("ui.overlay_window.QGuiApplication.screens", return_value=[mon1]),
            patch("ui.overlay_window.QGuiApplication.primaryScreen", return_value=mon1),
        ):
            self.assertIs(resolve_screen_by_name(r"\\.\DISPLAY2"), mon1)

    def test_edge_tts_proxy_config_and_resilient_voice_fetch(self) -> None:
        import asyncio
        import core.proxy_helper as ph
        from core.tts_manager import TTSEngineManager

        # 1. Proxy configuration resolution for Edge-TTS (SOCKS5 and HTTP/SOCKS custom_proxy_url)
        async def _probe_proxy_configs():
            with (
                patch("core.proxy_helper.get_active_socks5_url", return_value=""),
                patch("core.proxy_helper.config.get", return_value=""),
            ):
                self.assertEqual(ph.get_edge_tts_proxy_config(), (None, None))

            with (
                patch("core.proxy_helper.get_active_socks5_url", return_value=""),
                patch("core.proxy_helper.config.get", return_value="127.0.0.1:8080"),
            ):
                conn, http_px = ph.get_edge_tts_proxy_config()
                self.assertIsNone(conn)
                self.assertEqual(http_px, "http://127.0.0.1:8080")

            with (
                patch("core.proxy_helper.get_active_socks5_url", return_value="socks5h://127.0.0.1:10808"),
                patch("core.proxy_helper.config.get", return_value=""),
            ):
                conn, http_px = ph.get_edge_tts_proxy_config()
                self.assertIsNotNone(conn)
                self.assertIsNone(http_px)
                self.assertTrue(getattr(conn, "_rdns", False))
                await conn.close()

        asyncio.run(_probe_proxy_configs())

        # 2. Retry on transient WinError 64 in edge_tts.list_voices() using IPv4 connector
        sample_raw = [
            {"ShortName": "uk-UA-PolinaNeural", "Locale": "uk-UA", "Gender": "Female"},
            {"ShortName": "en-US-EmmaNeural", "Locale": "en-US", "Gender": "Female"},
        ]
        calls = []

        async def _flaky_list_voices(*, connector=None, proxy=None):
            calls.append((connector, proxy))
            if connector is not None:
                await connector.close()
            if len(calls) == 1:
                raise OSError(64, "Указанное сетевое имя более недоступно")
            return sample_raw

        with (
            patch("edge_tts.list_voices", side_effect=_flaky_list_voices),
            patch("core.proxy_helper.get_active_socks5_url", return_value=""),
            patch("core.proxy_helper.config.get", return_value=""),
            patch("core.tts_manager.DATA_DIR", Path(tempfile.gettempdir()) / "sv_edge_test_1"),
        ):
            voices = TTSEngineManager.fetch_edge_tts_voices()
            self.assertEqual(len(calls), 2)
            self.assertIsNone(calls[0][0])
            self.assertIsNotNone(calls[1][0])  # IPv4 TCPConnector on retry
            self.assertEqual({v["id"] for v in voices}, {"uk-UA-PolinaNeural", "en-US-EmmaNeural"})

        # 3. Fallback to synchronous requests when aiohttp fails both attempts
        async def _always_fail_list_voices(*, connector=None, proxy=None):
            if connector is not None:
                await connector.close()
            raise OSError(64, "Cannot connect to host speech.platform.bing.com:443 ssl:<ssl.SSLContext object at 0x1234> [Указанное сетевое имя более недоступно]")

        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = sample_raw
        fake_resp.raise_for_status.return_value = None

        with (
            patch("edge_tts.list_voices", side_effect=_always_fail_list_voices),
            patch("core.proxy_helper.get_active_socks5_url", return_value=""),
            patch("core.proxy_helper.config.get", return_value=""),
            patch("core.tts_manager.config.get", return_value=""),
            patch("core.tts_manager.requests.get", return_value=fake_resp) as mock_req,
            patch("core.tts_manager.DATA_DIR", Path(tempfile.gettempdir()) / "sv_edge_test_2"),
        ):
            voices = TTSEngineManager.fetch_edge_tts_voices()
            self.assertEqual(mock_req.call_count, 1)
            self.assertEqual(len(voices), 2)

        # 4. When both aiohttp and requests fail, clean SSLContext noise from error message
        with (
            patch("edge_tts.list_voices", side_effect=_always_fail_list_voices),
            patch("core.proxy_helper.get_active_socks5_url", return_value=""),
            patch("core.proxy_helper.config.get", return_value=""),
            patch("core.tts_manager.config.get", return_value=""),
            patch("core.tts_manager.requests.get", side_effect=ConnectionError("net down")),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                TTSEngineManager.fetch_edge_tts_voices()
            self.assertNotIn("<ssl.SSLContext", str(ctx.exception))
            self.assertIn("speech.platform.bing.com:443", str(ctx.exception))

    def test_socks5_status_label_elides_and_sets_tooltip(self) -> None:
        from PySide6.QtWidgets import QApplication
        from ui.dialogs import ElidedLabel
        from ui.main_window import MainWindow

        _app = QApplication.instance() or QApplication([])
        dummy = MainWindow.__new__(MainWindow)
        dummy._socks5_checkboxes = []
        dummy._socks5_url_edits = []
        dummy._socks5_refresh_btns = []
        dummy._socks5_status_labels = []

        bar = dummy._create_socks5_bar()
        self.assertEqual(len(dummy._socks5_status_labels), 1)
        lbl = dummy._socks5_status_labels[0]
        self.assertIsInstance(lbl, ElidedLabel)
        self.assertEqual(lbl.minimumSizeHint().width(), 0)

        long_status = "🔴 Локальний SOCKS5 не знайдено — вкажіть свій ip:port або запустіть мережеву програму"
        dummy._set_socks5_status_ui(long_status, "#F87171")
        self.assertEqual(lbl.text(), long_status)
        self.assertEqual(lbl.toolTip(), long_status)
        self.assertLessEqual(bar.minimumSizeHint().width(), 420)

        dummy._set_socks5_status_ui("", "#94A3B8")
        self.assertEqual(lbl.toolTip(), "")
        bar.deleteLater()

    def test_telegram_auth_dialog_uses_scroll_area_and_bounded_minimum_size(self) -> None:
        from PySide6.QtGui import QGuiApplication
        from PySide6.QtWidgets import QApplication
        from ui.dialogs import TelegramAuthDialog

        app = QApplication.instance() or QApplication([])
        with patch("ui.dialogs.QTimer.singleShot"):
            dlg = TelegramAuthDialog("tg_test")
            dlg.show()
            app.processEvents()
            self.assertLessEqual(dlg.minimumSizeHint().height(), 420)
            self.assertLessEqual(dlg.minimumHeight(), 360)

            dlg._on_toggle_custom_keys(True)
            dlg._set_error("Помилка підключення до Telegram API: " + ("текст " * 20))
            dlg._set_busy(True, "Перевірка підключення...")
            app.processEvents()
            app.processEvents()

            self.assertLessEqual(dlg.minimumSizeHint().height(), 420)
            screen = dlg.screen() or QGuiApplication.primaryScreen()
            if screen is not None:
                self.assertLessEqual(dlg.height(), screen.availableGeometry().height())
            dlg.close()
            app.processEvents()
            dlg.deleteLater()

    def test_start_model_download_reads_dont_show_again_property_without_type_error(self) -> None:
        from PySide6.QtWidgets import QApplication, QDialog
        from ui.dialogs import LocalModelDownloadNoticeDialog
        from ui.main_window import MainWindow

        _app = QApplication.instance() or QApplication([])

        class _AcceptedNotice(LocalModelDownloadNoticeDialog):
            def __init__(self, engine_name: str, _parent=None) -> None:
                super().__init__(engine_name, None)

            def exec(self) -> int:
                self.dont_show_cb.setChecked(True)
                return QDialog.DialogCode.Accepted

        dummy = MainWindow.__new__(MainWindow)
        dummy._ensure_language_confirmed = lambda: True
        dummy._engine_cards_ui = {}
        cfg_store: dict[str, object] = {"hide_local_model_first_run_notice": False}

        with (
            patch("ui.main_window.LocalModelDownloadNoticeDialog", _AcceptedNotice),
            patch("ui.main_window.tts_manager.is_downloading", return_value=False),
            patch("ui.main_window.tts_manager.get_files_for_engine", return_value=[("f.pt", "https://example.com/f.pt")]),
            patch("ui.main_window.tts_manager.download_engine_model") as mock_dl,
            patch("ui.main_window.config.get", side_effect=lambda k, d=None: cfg_store.get(k, d)),
            patch("ui.main_window.config.set", side_effect=lambda k, v: cfg_store.__setitem__(k, v)),
        ):
            dummy._start_model_download("silero")
            self.assertTrue(cfg_store["hide_local_model_first_run_notice"])
            mock_dl.assert_called_once_with("silero")

    def test_ollama_pull_worker_falls_back_to_http_stream_when_cli_fails(self) -> None:
        from ui.main_window import MainWindow

        dummy = MainWindow.__new__(MainWindow)
        progress_events: list[tuple] = []
        finished_events: list[tuple] = []

        class _Sig:
            def __init__(self, sink):
                self._sink = sink

            def emit(self, *args):
                self._sink.append(args)

        dummy.ollama_pull_progress = _Sig(progress_events)
        dummy.ollama_pull_finished = _Sig(finished_events)
        dummy._ollama_executable = lambda: r"C:\Program Files\Ollama\ollama.exe"

        fake_proc = MagicMock()
        fake_proc.wait.return_value = 1

        # Case A: CLI fails with code 1, built-in /api/pull succeeds
        ok_resp = MagicMock()
        ok_resp.__enter__.return_value = ok_resp
        ok_resp.__exit__.return_value = False
        ok_resp.raise_for_status.return_value = None
        ok_resp.iter_lines.return_value = [
            '{"status": "downloading", "completed": 50, "total": 100}',
            '{"status": "success"}',
        ]

        with (
            patch("ui.main_window.subprocess.Popen", return_value=fake_proc),
            patch("ui.main_window.requests.post", return_value=ok_resp) as mock_post,
        ):
            dummy._ollama_pull_worker("ollama_1", "gemma3:4b", "http://localhost:11434")
            mock_post.assert_called_once()
            self.assertEqual(finished_events[-1], ("ollama_1", "gemma3:4b", ""))
            self.assertTrue(any(ev[2] == "downloading" and ev[3] == 50 for ev in progress_events))

        # Case B: CLI fails with code 1, built-in /api/pull returns exact server error
        err_resp = MagicMock()
        err_resp.__enter__.return_value = err_resp
        err_resp.__exit__.return_value = False
        err_resp.raise_for_status.return_value = None
        err_resp.iter_lines.return_value = ['{"error": "pull model manifest: file does not exist"}']

        with (
            patch("ui.main_window.subprocess.Popen", return_value=fake_proc),
            patch("ui.main_window.requests.post", return_value=err_resp),
        ):
            dummy._ollama_pull_worker("ollama_1", "gemma3:4b", "http://localhost:11434")
            self.assertEqual(
                finished_events[-1],
                ("ollama_1", "gemma3:4b", "pull model manifest: file does not exist"),
            )

    def test_online_add_combo_labels_include_free_and_ru_availability(self) -> None:
        from PySide6.QtWidgets import QApplication, QComboBox
        from core.i18n import get_ui_language, set_ui_language
        from ui.main_window import MainWindow

        _app = QApplication.instance() or QApplication([])
        dummy = MainWindow.__new__(MainWindow)
        dummy.online_add_combo = QComboBox()

        prev_ui_lang = get_ui_language()
        try:
            set_ui_language("ru")
            with patch("ui.main_window.config.get", side_effect=lambda k, d=None: "uk" if k == "app_language" else ([] if k == "online_items" else d)):
                dummy._populate_online_add_combo()
                by_key = {dummy.online_add_combo.itemData(i): dummy.online_add_combo.itemText(i) for i in range(dummy.online_add_combo.count())}
                self.assertEqual(by_key["google"], "Google Cloud TTS API (бесплатно)")
                self.assertEqual(by_key["fish_audio"], "Fish Audio API (бесплатно) (клонирование)")
                self.assertEqual(by_key["elevenlabs"], "ElevenLabs API (клонирование)")
                self.assertEqual(by_key["openai"], "OpenAI TTS API (только платно)")

            with patch("ui.main_window.config.get", side_effect=lambda k, d=None: "ru" if k == "app_language" else ([] if k == "online_items" else d)):
                dummy._populate_online_add_combo()
                by_key = {dummy.online_add_combo.itemData(i): dummy.online_add_combo.itemText(i) for i in range(dummy.online_add_combo.count())}
                self.assertEqual(by_key["google"], "Google Cloud TTS API (бесплатно)")
                self.assertEqual(by_key["fish_audio"], "Fish Audio API (бесплатно) (клонирование) (доступно в РФ)")
                self.assertEqual(by_key["elevenlabs"], "ElevenLabs API (клонирование)")
                self.assertEqual(by_key["openai"], "OpenAI TTS API (только платно)")

            set_ui_language("uk")
            with patch("ui.main_window.config.get", side_effect=lambda k, d=None: "uk" if k == "app_language" else ([] if k == "online_items" else d)):
                dummy._populate_online_add_combo()
                by_key = {dummy.online_add_combo.itemData(i): dummy.online_add_combo.itemText(i) for i in range(dummy.online_add_combo.count())}
                self.assertEqual(by_key["fish_audio"], "Fish Audio API (безкоштовно) (клонування)")
                self.assertEqual(by_key["elevenlabs"], "ElevenLabs API (клонування)")
        finally:
            set_ui_language(prev_ui_lang)
            dummy.online_add_combo.deleteLater()

    def test_probe_foreground_game_state_skips_com_on_self_and_input_sync(self) -> None:
        import os
        import core.win_api as win_api_mod

        if sys.platform != "win32":
            return

        # 1. When foreground window belongs to current process (is_self=True),
        # SHQueryUserNotificationState must NOT be called (avoids RPC_E_CANTCALLOUT_ININPUTSYNCCALL 0x8001010d).
        def _set_self_pid(_hwnd, pid_ptr):
            ctypes.cast(pid_ptr, ctypes.POINTER(wintypes.DWORD)).contents.value = os.getpid()
            return 1

        with (
            patch.object(win_api_mod.user32, "GetForegroundWindow", return_value=12345),
            patch.object(win_api_mod.user32, "IsIconic", return_value=False),
            patch.object(win_api_mod.user32, "GetWindowThreadProcessId", side_effect=_set_self_pid),
            patch.object(win_api_mod.shell32, "SHQueryUserNotificationState") as mock_quns,
        ):
            self.assertEqual(win_api_mod.probe_foreground_game_state(), (False, ""))
            mock_quns.assert_not_called()

        # 2. When InSendMessageEx != 0 (input-synchronous call), SHQueryUserNotificationState is skipped safely
        def _set_other_pid(_hwnd, pid_ptr):
            ctypes.cast(pid_ptr, ctypes.POINTER(wintypes.DWORD)).contents.value = os.getpid() + 99999
            return 1

        with (
            patch.object(win_api_mod.user32, "GetForegroundWindow", return_value=23456),
            patch.object(win_api_mod.user32, "IsIconic", return_value=False),
            patch.object(win_api_mod.user32, "GetWindowThreadProcessId", side_effect=_set_other_pid),
            patch.object(win_api_mod.kernel32, "OpenProcess", return_value=0),
            patch("core.win_api._get_windows_game_store_paths", return_value=frozenset()),
            patch.object(win_api_mod.user32, "MonitorFromWindow", return_value=1),
            patch.object(win_api_mod.user32, "GetMonitorInfoW", return_value=True),
            patch.object(win_api_mod.user32, "GetWindowRect", return_value=True),
            patch.object(win_api_mod.user32, "GetClipCursor", return_value=False),
            patch.object(win_api_mod.user32, "GetCursorInfo", return_value=False),
            patch.object(win_api_mod.user32, "InSendMessageEx", return_value=1),
            patch.object(win_api_mod.shell32, "SHQueryUserNotificationState") as mock_quns_sync,
        ):
            win_api_mod.probe_foreground_game_state()
            mock_quns_sync.assert_not_called()

    def test_online_prewarm_host_map_and_engine_cards_ui_partitioning(self) -> None:
        from core.tts_manager import ENGINE_SPECS, TTSEngineManager
        from core.updater import is_newer_version, parse_version_tuple

        # 1. Version comparison with 1-part, 3-part, and 4-part version strings
        self.assertEqual(parse_version_tuple(""), (0, 0, 0))
        self.assertEqual(parse_version_tuple("2"), (2, 0, 0))
        self.assertFalse(is_newer_version("1.0.0.0", "1.0.0"))
        self.assertFalse(is_newer_version("2.0", "2"))
        self.assertTrue(is_newer_version("1.0.0.1", "1.0.0"))
        self.assertTrue(is_newer_version("2", "1.0.0"))

        # 2. _prewarm_online_provider resolves DNS for fish_audio and gemini_tts
        queried_hosts: list[str] = []
        with (
            patch("core.tts_manager.get_active_api_proxies", return_value=None),
            patch("socket.getaddrinfo", side_effect=lambda h, *_a, **_kw: queried_hosts.append(h)),
        ):
            TTSEngineManager._prewarm_online_provider("fish_audio")
            TTSEngineManager._prewarm_online_provider("gemini_tts")
        self.assertEqual(queried_hosts, ["api.fish.audio", "generativelanguage.googleapis.com"])

        # 3. _engine_cards_ui key partitioning between local engines and online items (e.g. edge_default)
        self.assertNotIn("edge_default", ENGINE_SPECS)
        self.assertIn("piper", ENGINE_SPECS)
        self.assertIn("openvoice", ENGINE_SPECS)
        self.assertIn("qwen3_tts", ENGINE_SPECS)

    def test_multistream_range_download_mirror_race_and_300kb_slow_switch(self) -> None:
        from core.tts_manager import tts_manager

        # 1. Slow speed threshold is 300 KB/s
        self.assertEqual(tts_manager._MIN_PROXY_SPEED_BPS, 300 * 1024)
        self.assertEqual(tts_manager._MULTISTREAM_WORKERS, 5)

        # 2. Candidate URLs include HuggingFace mirror and GitHub release mirrors when bypass is enabled
        hf_urls = tts_manager._build_candidate_urls(
            "https://huggingface.co/org/model/resolve/main/weights.gguf",
            bypass_enabled=True,
        )
        self.assertIn("https://hf-mirror.com/org/model/resolve/main/weights.gguf", hf_urls)

        gh_urls = tts_manager._build_candidate_urls(
            "https://github.com/ggml-org/llama.cpp/releases/download/b12608/llama-b12608-bin-win-vulkan-x64.zip",
            bypass_enabled=True,
        )
        self.assertGreaterEqual(len(gh_urls), 3)

        # 3. Multi-stream Range download reassembles chunks in parallel and cleans up .part / .part.meta
        payload = bytes((i % 251 for i in range(450_000)))

        class _FakeRangeResponse:
            def __init__(self, start: int, end: int, total: int):
                self.status_code = 206
                self.headers = {
                    "Content-Type": "application/octet-stream",
                    "Content-Range": f"bytes {start}-{end}/{total}",
                    "Content-Length": str(end - start + 1),
                }
                self._slice = payload[start : end + 1]

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def close(self):
                pass

            def iter_content(self, chunk_size: int = 65536):
                for off in range(0, len(self._slice), chunk_size):
                    yield self._slice[off : off + chunk_size]

        range_calls: list[tuple[int, int]] = []

        def _fake_session_get(url, headers=None, **kwargs):
            rng = str((headers or {}).get("Range", "bytes=0-0"))
            b_part = rng.split("=", 1)[1]
            s_str, e_str = b_part.split("-", 1)
            s_i, e_i = int(s_str), int(e_str)
            range_calls.append((s_i, e_i))
            return _FakeRangeResponse(s_i, e_i, len(payload))

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "test_model.gguf"
            with (
                patch.object(tts_manager, "_MULTISTREAM_MIN_SIZE_BYTES", 100_000),
                patch.object(tts_manager, "_MULTISTREAM_BLOCK_SIZE", 100_000),
                patch("requests.get", side_effect=_fake_session_get),
                patch("requests.Session.get", side_effect=_fake_session_get),
            ):
                tts_manager._stream_download_attempt(
                    "https://example.invalid/test_model.gguf",
                    dest,
                    "qwen3_tts",
                    0,
                    100,
                    "Downloading",
                )
            self.assertTrue(dest.is_file())
            self.assertEqual(dest.read_bytes(), payload)
            self.assertFalse(dest.with_suffix(".gguf.part").exists())
            self.assertFalse(dest.with_suffix(".gguf.part.meta").exists())
            # 1 range probe (0-0) + 5 block requests
            self.assertEqual(len(range_calls), 6)

    def test_multi_proxy_swarm_failover_cdn_reuse_and_proxy_benchmark_ranking(self) -> None:
        import threading
        import time
        import requests
        from core.tts_manager import tts_manager

        payload = b"PK" + bytes(((i * 37 + 13) % 251 for i in range(520_000 - 2)))
        dead_px = "http://10.0.0.1:8080"
        fast_px = "socks5h://10.0.0.2:1080"
        backup_px = "http://10.0.0.3:8080"

        class _SwarmRangeResp:
            def __init__(self, start: int, end: int, total: int, final_url: str, status_code: int = 206):
                self.status_code = status_code
                self.url = final_url
                self.headers = {
                    "Content-Type": "application/octet-stream",
                    "Content-Range": f"bytes {start}-{end}/{total}",
                    "Content-Length": str(end - start + 1),
                }
                self._slice = payload[start : end + 1]

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def close(self):
                pass

            def iter_content(self, chunk_size: int = 65536):
                for off in range(0, len(self._slice), chunk_size):
                    yield self._slice[off : off + chunk_size]

        used_proxies: list[str] = []
        used_urls: list[str] = []
        cdn_expired_once = {"done": False}
        call_lock = threading.Lock()

        def _fake_swarm_get(url, headers=None, proxies=None, **kwargs):
            px_val = str((proxies or {}).get("https") or "")
            rng = str((headers or {}).get("Range", "bytes=0-0"))
            b_part = rng.split("=", 1)[1]
            s_str, e_str = b_part.split("-", 1)
            s_i, e_i = int(s_str), int(e_str)
            with call_lock:
                used_proxies.append(px_val)
                used_urls.append(url)
            if px_val == dead_px:
                raise requests.exceptions.ProxyError("Dead proxy")
            # Simulate an expired signed CDN URL on one request -> workers should fall back to origin URL
            if "cdn-signed.example.invalid" in url and not cdn_expired_once["done"]:
                with call_lock:
                    if not cdn_expired_once["done"]:
                        cdn_expired_once["done"] = True
                        return _SwarmRangeResp(0, 0, len(payload), url, status_code=403)
            return _SwarmRangeResp(s_i, e_i, len(payload), url, status_code=206)

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "silero_test.pt"
            prev_cache = dict(tts_manager._proxy_cache)
            prev_bad = dict(tts_manager._bad_proxies)
            prev_meta = dict(tts_manager._proxy_meta)
            try:
                tts_manager._proxy_cache.clear()
                tts_manager._bad_proxies.clear()
                tts_manager._proxy_meta.clear()
                with (
                    patch.object(tts_manager, "_MULTISTREAM_MIN_SIZE_BYTES", 100_000),
                    patch.object(tts_manager, "_MULTISTREAM_BLOCK_SIZE", 100_000),
                    patch("requests.Session.get", side_effect=_fake_swarm_get),
                ):
                    tts_manager._multistream_download_attempt(
                        target_url="https://models.silero.ai/models/tts/ru/v5_cis_ext.pt",
                        dest=dest,
                        total=len(payload),
                        engine_id="silero",
                        base_pct=0,
                        span_pct=100,
                        label="Silero",
                        proxies=None,
                        proxy_pool=[dead_px, fast_px, backup_px],
                        resolved_url="https://cdn-signed.example.invalid/v5_cis_ext.pt",
                    )

                # Byte-exact reassembly even though dead_px failed and signed CDN URL expired once
                self.assertTrue(dest.is_file())
                self.assertEqual(dest.read_bytes(), payload)
                self.assertFalse(dest.with_suffix(".pt.part").exists())
                self.assertFalse(dest.with_suffix(".pt.part.meta").exists())
                # Dead proxy was marked bad and removed from cache; healthy proxies stayed cached
                self.assertIn(dead_px, tts_manager._bad_proxies)
                cached = tts_manager._get_cached_proxies("https://models.silero.ai/models/tts/ru/v5_cis_ext.pt")
                self.assertNotIn(dead_px, cached)
                self.assertTrue(any(p in cached for p in (fast_px, backup_px)))
                # Origin URL was used after signed CDN URL expired with 403
                self.assertIn("https://models.silero.ai/models/tts/ru/v5_cis_ext.pt", used_urls)

                # Verify _find_working_proxies_for_url ranks fast proxy above slow/high-TTFB proxy
                slow_bench_px = "http://10.1.0.1:8080"
                fast_bench_px = "socks5h://10.1.0.2:1080"

                class _ListResp:
                    status_code = 200
                    text = "10.1.0.1:8080\n10.1.0.2:1080\n"

                def _fake_bench_get(url, proxies=None, **kwargs):
                    if not proxies:
                        return _ListResp()
                    px_val = str(proxies.get("https") or "")
                    if "10.1.0.1" in px_val:
                        time.sleep(0.28)
                        return _SwarmRangeResp(0, 65535, len(payload), url, status_code=206)
                    time.sleep(0.02)
                    return _SwarmRangeResp(0, 524287 if len(payload) > 524288 else len(payload) - 1, len(payload), url, status_code=206)

                tts_manager._proxy_cache.clear()
                tts_manager._bad_proxies.clear()
                tts_manager._proxy_meta.clear()
                with (
                    patch.object(tts_manager, "_tcp_prefilter_proxies", side_effect=lambda cands, **kw: [slow_bench_px, fast_bench_px]),
                    patch("requests.get", side_effect=_fake_bench_get),
                ):
                    ranked = tts_manager._find_working_proxies_for_url(
                        "https://models.silero.ai/models/tts/ru/v5_cis_ext.pt",
                        max_count=5,
                        engine_id="silero",
                    )
                self.assertEqual(ranked[0], fast_bench_px)
                self.assertIn(slow_bench_px, ranked)
                meta = tts_manager._proxy_meta.get("models.silero.ai") or {}
                self.assertTrue(meta.get("supports_range"))
                self.assertEqual(meta.get("total"), len(payload))
            finally:
                tts_manager._proxy_cache.clear()
                tts_manager._proxy_cache.update(prev_cache)
                tts_manager._bad_proxies.clear()
                tts_manager._bad_proxies.update(prev_bad)
                tts_manager._proxy_meta.clear()
                tts_manager._proxy_meta.update(prev_meta)

    def test_proxy_switch_preserves_partial_progress_and_never_wipes_part_file(self) -> None:
        import json
        from core.tts_manager import tts_manager

        payload = bytes(((i * 19 + 7) % 251 for i in range(400_000)))

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "model.onnx"
            part = dest.with_suffix(".onnx.part")
            meta = dest.with_suffix(".onnx.part.meta")

            # 1. Pre-populate blocks 1 and 2 (block 0 missing!) to verify non-contiguous blocks survive proxy switch
            buf = bytearray(len(payload))
            buf[100_000:300_000] = payload[100_000:300_000]
            part.write_bytes(buf)
            meta.write_text(json.dumps({"total": len(payload), "block_size": 100_000, "done": [1, 2]}), encoding="utf-8")

            done_b, tot_b = tts_manager._read_part_progress(dest)
            self.assertEqual(done_b, 200_000)
            self.assertEqual(tot_b, 400_000)

            # 2. If a Range-capable server's proxy ignores Range and returns 200 OK while resume_offset > 0, .part MUST NOT be wiped
            part2 = Path(tmp) / "single.onnx.part"
            dest2 = Path(tmp) / "single.onnx"
            part2.write_bytes(payload[:150_000])

            class _Resp200AcceptRanges:
                status_code = 200
                headers = {
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(len(payload)),
                    "Accept-Ranges": "bytes",
                }
                def raise_for_status(self): pass
                def iter_content(self, chunk_size=65536):
                    yield payload

            with self.assertRaises(IOError):
                tts_manager._consume_download_response(
                    _Resp200AcceptRanges(), part2, dest2, "piper", 0, 100, "Piper", 150_000, False, is_proxied=True
                )
            self.assertEqual(part2.stat().st_size, 150_000)

            # 2b. Non-Range server (like models.silero.ai without Accept-Ranges) allows clean restart on 200 OK
            class _Resp200NoRange:
                status_code = 200
                headers = {
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(len(payload)),
                }
                def raise_for_status(self): pass
                def iter_content(self, chunk_size=65536):
                    yield payload

            tts_manager._consume_download_response(
                _Resp200NoRange(), part2, dest2, "silero", 0, 100, "Silero", 150_000, False, is_proxied=True
            )
            self.assertEqual(part2.read_bytes(), payload)

            # 3. Pre-allocated empty .part with meta done=[] must NOT treat zero-filled file as already downloaded
            dest3 = Path(tmp) / "empty_meta.onnx"
            part3 = dest3.with_suffix(".onnx.part")
            meta3 = dest3.with_suffix(".onnx.part.meta")
            part3.write_bytes(bytes(len(payload)))
            meta3.write_text(json.dumps({"total": len(payload), "block_size": 100_000, "done": []}), encoding="utf-8")

            class _FullRangeResp:
                def __init__(self, start, end):
                    self.status_code = 206
                    self.headers = {
                        "Content-Type": "application/octet-stream",
                        "Content-Range": f"bytes {start}-{end}/{len(payload)}",
                    }
                    self._slice = payload[start : end + 1]
                def close(self): pass
                def iter_content(self, chunk_size=65536):
                    yield self._slice

            fetched_ranges: list[tuple[int, int]] = []

            def _fake_get(url, headers=None, **kw):
                rng = str((headers or {}).get("Range", "bytes=0-0")).split("=", 1)[1]
                s_i, e_i = [int(x) for x in rng.split("-", 1)]
                fetched_ranges.append((s_i, e_i))
                return _FullRangeResp(s_i, e_i)

            with (
                patch.object(tts_manager, "_MULTISTREAM_MIN_SIZE_BYTES", 100_000),
                patch.object(tts_manager, "_MULTISTREAM_BLOCK_SIZE", 100_000),
                patch("requests.Session.get", side_effect=_fake_get),
            ):
                tts_manager._multistream_download_attempt(
                    "https://example.invalid/empty_meta.onnx",
                    dest3,
                    len(payload),
                    "piper",
                    0,
                    100,
                    "Piper",
                )
            self.assertEqual(dest3.read_bytes(), payload)
            self.assertEqual(len(fetched_ranges), 4)

    def test_delete_silero_releases_package_importer_file_handle(self) -> None:
        import core.tts_manager as tm_mod

        class _FakeZipReader:
            def __init__(self, path: Path) -> None:
                self._fh = open(path, "rb")

            def __del__(self) -> None:
                try:
                    self._fh.close()
                except Exception:
                    pass

        class _FakeImporter:
            def __init__(self, path: Path) -> None:
                self.zip_reader = _FakeZipReader(path)

        with tempfile.TemporaryDirectory() as tmp:
            engines_root = Path(tmp) / "engines"
            silero_dir = engines_root / "silero"
            silero_dir.mkdir(parents=True)
            pt_file = silero_dir / "v5_6_ru.pt"
            pt_file.write_bytes(b"fake-silero-model")

            # Keep a strong external reference to the importer (mimicking torch.package
            # module references) so only nulling `importer.zip_reader` can release the handle.
            importer = _FakeImporter(pt_file)
            status_events: list[tuple[str, bool]] = []
            error_events: list[str] = []

            def _on_status(eid: str, inst: bool) -> None:
                status_events.append((eid, inst))

            def _on_error(msg: str) -> None:
                error_events.append(msg)

            prev_cache = dict(tts_manager._silero_cache)
            prev_importers = dict(tts_manager._silero_importers)
            try:
                tts_manager._silero_cache[str(pt_file)] = object()
                tts_manager._silero_importers[str(pt_file)] = importer
                tts_manager.model_status_changed.connect(_on_status)
                tts_manager.error_occurred.connect(_on_error)

                with (
                    patch.object(tm_mod, "ENGINES_DIR", engines_root),
                    patch.object(tts_manager, "_current_lang", return_value="ru"),
                ):
                    self.assertTrue(tts_manager.is_engine_installed("silero"))
                    tts_manager.delete_engine_model("silero")
                    self.assertIsNone(importer.zip_reader)
                    self.assertFalse(pt_file.exists())
                    self.assertFalse(silero_dir.exists())
                    self.assertFalse(tts_manager.is_engine_installed("silero"))
                    self.assertIn(("silero", False), status_events)
                    self.assertEqual(error_events, [])
            finally:
                try:
                    tts_manager.model_status_changed.disconnect(_on_status)
                except Exception:
                    pass
                try:
                    tts_manager.error_occurred.disconnect(_on_error)
                except Exception:
                    pass
                tts_manager._silero_cache.clear()
                tts_manager._silero_cache.update(prev_cache)
                tts_manager._silero_importers.clear()
                tts_manager._silero_importers.update(prev_importers)
                if getattr(importer, "zip_reader", None) is not None:
                    importer.zip_reader = None

    def test_delete_engine_model_keeps_installed_state_if_files_locked(self) -> None:
        import core.tts_manager as tm_mod

        with tempfile.TemporaryDirectory() as tmp:
            engines_root = Path(tmp) / "engines"
            silero_dir = engines_root / "silero"
            silero_dir.mkdir(parents=True)
            pt_file = silero_dir / "v5_6_ru.pt"
            pt_file.write_bytes(b"locked-silero-model")

            status_events: list[tuple[str, bool]] = []
            error_events: list[str] = []

            def _on_status(eid: str, inst: bool) -> None:
                status_events.append((eid, inst))

            def _on_error(msg: str) -> None:
                error_events.append(msg)

            def _blocked_rmtree(path, ignore_errors=False, *args, **kwargs):
                if not ignore_errors:
                    raise PermissionError("WinError 32: file in use")

            tts_manager.model_status_changed.connect(_on_status)
            tts_manager.error_occurred.connect(_on_error)
            try:
                with (
                    patch.object(tm_mod, "ENGINES_DIR", engines_root),
                    patch.object(tts_manager, "_current_lang", return_value="ru"),
                    patch("core.tts_manager.shutil.rmtree", side_effect=_blocked_rmtree),
                    patch("core.tts_manager.time.sleep", return_value=None),
                ):
                    tts_manager.delete_engine_model("silero")
                    self.assertTrue(tts_manager.is_engine_installed("silero"))
                    self.assertIn(("silero", True), status_events)
                    self.assertNotIn(("silero", False), status_events)
                    self.assertTrue(error_events)
            finally:
                try:
                    tts_manager.model_status_changed.disconnect(_on_status)
                except Exception:
                    pass
                try:
                    tts_manager.error_occurred.disconnect(_on_error)
                except Exception:
                    pass


class KeyboardHookKeyupSwallowTests(unittest.TestCase):
    def test_swallowed_space_also_swallows_matching_keyup_without_blocking_preheld_keys(self) -> None:
        import ctypes
        import core.win_api as wa

        mgr = wa.KeyboardHookManager()
        mgr._input_mode = "hook"
        mgr._is_typing = True
        mgr._buffer = "hello"

        def _send(wparam: int, vk: int, scan: int = 0) -> int:
            kb = wa.KBDLLHOOKSTRUCT(vkCode=vk, scanCode=scan, flags=0, time=0, dwExtraInfo=0)
            return int(mgr._low_level_keyboard_proc(0, wparam, ctypes.addressof(kb)))

        with (
            patch.object(wa.user32, "CallNextHookEx", return_value=0) as mock_next,
            patch.object(wa.user32, "GetAsyncKeyState", return_value=0),
            patch.object(mgr, "_get_pressed_modifiers", return_value=set()),
            patch.object(mgr, "_vk_to_unicode", side_effect=lambda vk, scan: " " if vk == wa.VK_SPACE else ""),
            patch.object(mgr, "get_current_lang_code", return_value="EN"),
        ):
            # 1. Space pressed while typing -> BOTH keydown AND keyup MUST be swallowed (return 1)
            self.assertEqual(_send(wa.WM_KEYDOWN, wa.VK_SPACE), 1)
            self.assertEqual(mgr._buffer, "hello ")
            self.assertEqual(_send(wa.WM_KEYUP, wa.VK_SPACE), 1)
            self.assertEqual(mock_next.call_count, 0)

            # 2. Key held BEFORE typing mode (never swallowed on keydown) -> keyup MUST pass through
            self.assertEqual(_send(wa.WM_KEYUP, ord("W")), 0)
            self.assertEqual(mock_next.call_count, 1)


if __name__ == "__main__":
    unittest.main()
