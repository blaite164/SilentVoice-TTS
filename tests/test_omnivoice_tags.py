from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.omnivoice_tags import (
    OMNIVOICE_ALL_OFFICIAL_TAGS,
    OMNIVOICE_REACTION_TAGS,
    backspace_trailing_omnivoice_tag,
    build_active_reaction_specs,
    complete_slash_command,
    expand_omnivoice_tags,
    extract_active_slash_query,
    format_tags_for_display,
    get_default_reaction_commands,
    get_spec_display_label,
    get_tag_description,
    is_reaction_item_playable,
    match_reaction_tags,
    replace_active_slash_with_tag,
    sanitize_reaction_command_name,
    split_text_for_reaction_playback,
    strip_omnivoice_tags,
)


class OmniVoiceTagsLogicTests(unittest.TestCase):
    def test_extract_active_slash_backslash_pipe_queries(self) -> None:
        self.assertEqual(extract_active_slash_query("/"), ("", ""))
        self.assertEqual(extract_active_slash_query("\\"), ("", ""))
        self.assertEqual(extract_active_slash_query("|"), ("", ""))
        self.assertEqual(extract_active_slash_query("/с"), ("", "с"))
        self.assertEqual(extract_active_slash_query("Привіт \\зі"), ("Привіт ", "зі"))
        self.assertEqual(extract_active_slash_query("Привіт |вау"), ("Привіт ", "вау"))
        self.assertIsNone(extract_active_slash_query("Привіт /сміх "))
        self.assertIsNone(extract_active_slash_query("Привіт \\сміх "))
        self.assertIsNone(extract_active_slash_query("1/2"))
        self.assertIsNone(extract_active_slash_query(r"C:\Users\test"))
        self.assertIsNone(extract_active_slash_query("foo|bar"))
        self.assertIsNone(extract_active_slash_query("https://example.com/path"))
        self.assertIsNone(extract_active_slash_query("Привіт"))

    def test_match_reaction_tags_multilingual_and_descriptions(self) -> None:
        all_tags = match_reaction_tags("", "uk")
        self.assertEqual(len(all_tags), len(OMNIVOICE_REACTION_TAGS))
        for spec in all_tags:
            for lang in ("uk", "en", "ru", "de", "es", "fr", "pl"):
                self.assertTrue(get_tag_description(spec, lang))
        self.assertEqual(match_reaction_tags("с", "uk")[0].tag, "[laughter]")
        self.assertEqual(match_reaction_tags("взд", "ru")[0].tag, "[sigh]")
        self.assertEqual(match_reaction_tags("lol", "uk")[0].tag, "[laughter]")
        self.assertEqual(match_reaction_tags("wow", "en")[0].tag, "[surprise-wa]")
        self.assertEqual(match_reaction_tags("хмф", "uk")[0].tag, "[dissatisfaction-hnn]")
        self.assertEqual(match_reaction_tags("zzzz_none", "uk"), [])

    def test_complete_and_replace_slash_commands(self) -> None:
        self.assertEqual(complete_slash_command("Ого /с", "uk"), "Ого [laughter] ")
        self.assertEqual(complete_slash_command("Ого \\с", "uk"), "Ого [laughter] ")
        self.assertEqual(complete_slash_command("Ого |с", "uk"), "Ого [laughter] ")
        self.assertEqual(complete_slash_command("\\", "uk", selected_index=1), "[sigh] ")
        self.assertEqual(
            complete_slash_command("Привіт \\сміх", "uk", exact_only=True),
            "Привіт [laughter] ",
        )
        self.assertEqual(
            replace_active_slash_with_tag("Привіт |зі", "[sigh]"),
            "Привіт [sigh] ",
        )
        self.assertIsNone(complete_slash_command("Привіт /с", "uk", exact_only=True))
        self.assertIsNone(complete_slash_command("Привіт |", "uk", exact_only=True))

    def test_backspace_trailing_tag_atomic(self) -> None:
        self.assertEqual(backspace_trailing_omnivoice_tag("Привіт [laughter]"), "Привіт ")
        self.assertEqual(backspace_trailing_omnivoice_tag("[dissatisfaction-hnn]"), "")
        self.assertIsNone(backspace_trailing_omnivoice_tag("Привіт [laughter] "))
        self.assertIsNone(backspace_trailing_omnivoice_tag("Привіт [custom]"))

    def test_expand_and_strip_and_display(self) -> None:
        self.assertEqual(
            expand_omnivoice_tags(r"Привіт /сміх, \зітх та |вау! 1/2 C:\сміх a|вау https://site.ua/сміх"),
            r"Привіт [laughter], [sigh] та [surprise-wa]! 1/2 C:\сміх a|вау https://site.ua/сміх",
        )
        for tag in OMNIVOICE_ALL_OFFICIAL_TAGS:
            self.assertEqual(strip_omnivoice_tags(f"Привіт {tag}, тест"), "Привіт, тест")
        self.assertEqual(
            format_tags_for_display("Привіт [laughter] ну добре [sigh]"),
            "Привіт 😂 ну добре 😮‍💨",
        )

    def test_custom_sound_commands_playability_display_and_splitting(self) -> None:
        self.assertEqual(sanitize_reaction_command_name(r" \ХаХа! "), "хаха")

        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "haha.wav"
            wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

            item_no_file = {
                "id": "rc_custom_1",
                "command": "хаха",
                "aliases": ["хаха"],
                "display": "",
                "description": "Мій сміх",
                "audio_file": "",
                "omnivoice_tag": "",
                "enabled": True,
            }
            item_with_file = dict(item_no_file, audio_file=str(wav))
            item_ov_builtin = get_default_reaction_commands("uk")[0]

            # Правило галочки: без файлу і без OmniVoice — не можна увімкнути
            self.assertFalse(is_reaction_item_playable(item_no_file, omnivoice_enabled=True))
            self.assertFalse(is_reaction_item_playable(item_no_file, omnivoice_enabled=False))
            self.assertTrue(is_reaction_item_playable(item_with_file, omnivoice_enabled=False))
            self.assertTrue(is_reaction_item_playable(item_ov_builtin, omnivoice_enabled=True))
            self.assertFalse(is_reaction_item_playable(item_ov_builtin, omnivoice_enabled=False))

            cfg = {
                "app_language": "uk",
                "active_engine": "edge_tts",
                "reaction_omnivoice_enabled": True,
                "reaction_commands": [item_no_file, item_with_file, item_ov_builtin],
            }
            # На Edge TTS активна лише кастомна команда з файлом (без вбудованих OmniVoice)
            edge_specs = build_active_reaction_specs(cfg, engine_id="edge_tts")
            self.assertEqual([s.tag for s in edge_specs], ["[хаха]"])
            self.assertEqual(get_spec_display_label(edge_specs[0], "uk"), "[хаха]")

            # Автодоповнення \хаха + Пробіл -> [хаха] , атомарне видалення Backspace
            self.assertEqual(
                complete_slash_command("Привіт \\хаха", "uk", exact_only=True, specs=edge_specs),
                "Привіт [хаха] ",
            )
            self.assertEqual(
                format_tags_for_display("Привіт [хаха] друже", specs=edge_specs, ui_lang="uk"),
                "Привіт [хаха] друже",
            )
            self.assertEqual(
                backspace_trailing_omnivoice_tag("Привіт [хаха]", specs=edge_specs),
                "Привіт ",
            )

            # Розбиття речення на текст + прикріплений звук для будь-якого TTS
            segs_edge = split_text_for_reaction_playback(
                "Привіт [хаха] як справи [sigh]",
                edge_specs,
                is_omnivoice_engine=False,
                ui_lang="uk",
            )
            self.assertEqual(
                segs_edge,
                [
                    ("text", "Привіт", ""),
                    ("audio", str(wav), "[хаха]"),
                    ("text", "як справи", ""),
                ],
            )

            # На OmniVoice активні і власна команда з файлом, і вбудований OmniVoice-тег
            ov_specs = build_active_reaction_specs(cfg, engine_id="omnivoice")
            self.assertEqual([s.tag for s in ov_specs], ["[хаха]", "[laughter]"])
            segs_ov = split_text_for_reaction_playback(
                "Привіт \\хаха [laughter] ну ок",
                ov_specs,
                is_omnivoice_engine=True,
                ui_lang="uk",
            )
            self.assertEqual(
                segs_ov,
                [
                    ("text", "Привіт", ""),
                    ("audio", str(wav), "[хаха]"),
                    ("text", "[laughter] ну ок", ""),
                ],
            )

            # Якщо галочка вимкнена — команда приховується навіть при наявному файлі
            cfg_off = dict(cfg, reaction_commands=[dict(item_with_file, enabled=False)])
            self.assertEqual(build_active_reaction_specs(cfg_off, engine_id="edge_tts"), [])

    def test_localized_defaults_multi_aliases_fast_prefix_and_lang_sync(self) -> None:
        from core.omnivoice_tags import (
            parse_reaction_command_aliases,
            sync_default_reaction_commands_for_lang,
        )

        # Кожна мова має власну природну першу команду (укр/рос -> хаха, en/de/fr/pl -> haha, es -> jaja)
        self.assertEqual(get_default_reaction_commands("uk")[0]["command"], "хаха")
        self.assertEqual(get_default_reaction_commands("ru")[0]["command"], "хаха")
        self.assertEqual(get_default_reaction_commands("en")[0]["command"], "haha")
        self.assertEqual(get_default_reaction_commands("es")[0]["command"], "jaja")
        self.assertIn("ахаха", get_default_reaction_commands("uk")[0]["aliases"])
        self.assertIn("хах", get_default_reaction_commands("uk")[0]["aliases"])

        # Введення кількох команд через кому / крапку з комою / пробіл
        self.assertEqual(
            parse_reaction_command_aliases(r" \хаха, ахаха; /ахах  хах "),
            ["хаха", "ахаха", "ахах", "хах"],
        )

        # Міграція старих англійських/сміх налаштувань під українську зі збереженням файлу, бінду й власних синонімів
        old_cfg = {
            "reaction_commands": [
                {
                    "id": "rc_laughter",
                    "command": "laugh",
                    "aliases": ["laugh", "mycustomlol"],
                    "display": "😂",
                    "description": "Natural laughter",
                    "audio_file": "my.mp3",
                    "omnivoice_tag": "[laughter]",
                    "hotkey": "Alt+1",
                    "enabled": True,
                },
            ]
        }
        synced = sync_default_reaction_commands_for_lang(old_cfg, "uk")
        self.assertIsNotNone(synced)
        assert synced is not None
        self.assertEqual(synced[0]["command"], "хаха")
        self.assertEqual(synced[0]["description"], "Живий сміх")
        self.assertEqual(synced[0]["audio_file"], "my.mp3")
        self.assertEqual(synced[0]["hotkey"], "Alt+1")
        self.assertIn("mycustomlol", synced[0]["aliases"])
        self.assertIn("ахаха", synced[0]["aliases"])

        # Швидкий друк: однозначний префікс 2+ символи завершується вже по Пробілу
        self.assertEqual(complete_slash_command("Привіт \\зіт", "uk", exact_only=True), "Привіт [sigh] ")
        self.assertEqual(complete_slash_command("Привіт \\хах", "uk", exact_only=True), "Привіт [laughter] ")
        self.assertEqual(complete_slash_command("Привіт \\аха", "uk", exact_only=True), "Привіт [laughter] ")
        # Але 1 символ не завершується випадково
        self.assertIsNone(complete_slash_command("Привіт \\с", "uk", exact_only=True))

    def test_ai_spellcheck_preserves_reaction_tokens(self) -> None:
        from core.ai_corrector import correct_text_with_ai
        from core.config import config

        fake_cfg = {
            "ai_spellcheck_enabled": True,
            "ai_primary_id": "p1",
            "ai_fallback_id": "",
            "ai_items": [{"id": "p1", "provider": "groq", "title": "MockAI"}],
            "ui_language": "uk",
            "reaction_commands": [
                {
                    "id": "rc_custom_haha",
                    "command": "хаха",
                    "aliases": ["хаха", "ахах"],
                    "display": "",
                    "description": "",
                    "audio_file": "",
                    "omnivoice_tag": "",
                    "hotkey": "",
                    "enabled": True,
                }
            ],
        }

        def _get(k: str, d=None):
            return fake_cfg.get(k, d)

        with patch.object(config, "_data", fake_cfg), patch.object(config, "get", side_effect=_get):
            # 1. Один виклик AI, модель зберігає ⟪T0⟫ -> відновлення оригінального тегу
            with patch("core.ai_corrector._call_ai_item", return_value="Привіт ⟪T0⟫ як справи?") as mock_ai:
                out = correct_text_with_ai("привыт [laughter] як справы")
                self.assertEqual(out, "Привіт [laughter] як справи?")
                self.assertEqual(mock_ai.call_count, 1)

            # 2. Якщо модель зіпсувала маркер -> автоматичний пофрагментний фолбек зберігає всі теги
            calls: list[str] = []

            def _bad_then_part(_item, text_arg, _prompt):
                calls.append(text_arg)
                if "⟪T0⟫" in text_arg:
                    return "Привіт як справи"  # модель видалила маркер!
                return text_arg.replace("привыт", "Привіт").replace("справы", "справи")

            with patch("core.ai_corrector._call_ai_item", side_effect=_bad_then_part):
                out2 = correct_text_with_ai("привыт \\хаха як справы")
                self.assertEqual(out2, "Привіт [хаха] як справи")

            # 3. Якщо у фразі ТІЛЬКИ команда без слів -> AI взагалі не викликається
            with patch("core.ai_corrector._call_ai_item") as mock_skip:
                out3 = correct_text_with_ai("[laughter]")
                self.assertEqual(out3, "[laughter]")
                mock_skip.assert_not_called()

    def test_inline_whisper_command_omnivoice_only_visibility_and_segments(self) -> None:
        from PySide6.QtWidgets import QApplication
        from core.config import config
        from core.omnivoice_tags import (
            OMNIVOICE_WHISPER_ID,
            OMNIVOICE_WHISPER_TAG,
            build_default_whisper_command,
            is_omnivoice_whisper_command_visible,
            propagate_whisper_across_chunks,
            split_omnivoice_whisper_segments,
            strip_whisper_tags,
        )
        from ui.reaction_commands import ReactionCommandsCard

        # Локалізація шепоту для всіх мов
        self.assertEqual(build_default_whisper_command("uk")["command"], "шепіт")
        self.assertEqual(build_default_whisper_command("ru")["command"], "шёпот")
        self.assertEqual(build_default_whisper_command("en")["command"], "whisper")
        self.assertEqual(build_default_whisper_command("de")["command"], "flüster")
        self.assertEqual(build_default_whisper_command("es")["command"], "susurro")
        self.assertEqual(build_default_whisper_command("fr")["command"], "chuchot")
        self.assertEqual(build_default_whisper_command("pl")["command"], "szept")

        # Розбиття на сегменти шепоту (увімк до кінця / увімк+вимк посередині / перенесення через речення)
        self.assertEqual(
            split_omnivoice_whisper_segments("Привіт [whisper] це секрет"),
            [("Привіт", False), ("це секрет", True)],
        )
        self.assertEqual(
            split_omnivoice_whisper_segments("Привіт [whisper] тихо, там ворог [whisper] пішли!"),
            [("Привіт", False), ("тихо, там ворог", True), ("пішли!", False)],
        )
        self.assertEqual(
            propagate_whisper_across_chunks(["Привіт!", "[whisper] Тихо.", "Тут ворог.", "[whisper] Пішли!"]),
            ["Привіт!", "[whisper] Тихо.", "[whisper] Тут ворог.", "Пішли!"],
        )
        self.assertEqual(strip_whisper_tags("Привіт [whisper] тихо"), "Привіт тихо")
        self.assertEqual(format_tags_for_display("Привіт [whisper] тихо"), "Привіт 🤫 тихо")
        self.assertEqual(backspace_trailing_omnivoice_tag("Привіт [whisper]"), "Привіт ")

        # Видимість у списку та підказці ТІЛЬКИ коли увімкнено OmniVoice
        base_cmds = get_default_reaction_commands("uk")
        cfg_ov_on = {
            "ui_language": "uk",
            "active_engine": "omnivoice",
            "reaction_omnivoice_enabled": True,
            "reaction_commands": base_cmds,
            "reaction_whisper_seeded": True,
        }
        cfg_edge = dict(cfg_ov_on, active_engine="edge_tts")
        cfg_ov_off = dict(cfg_ov_on, reaction_omnivoice_enabled=False)

        self.assertTrue(is_omnivoice_whisper_command_visible(cfg_ov_on))
        self.assertFalse(is_omnivoice_whisper_command_visible(cfg_edge))
        self.assertFalse(is_omnivoice_whisper_command_visible(cfg_ov_off))

        self.assertIn(OMNIVOICE_WHISPER_TAG, [s.tag for s in build_active_reaction_specs(cfg_ov_on)])
        self.assertNotIn(OMNIVOICE_WHISPER_TAG, [s.tag for s in build_active_reaction_specs(cfg_edge)])
        self.assertNotIn(OMNIVOICE_WHISPER_TAG, [s.tag for s in build_active_reaction_specs(cfg_ov_off)])

        app = QApplication.instance() or QApplication([])
        self.assertIsNotNone(app)
        cur_cfg = dict(cfg_edge)
        with patch.object(config, "_data", cur_cfg), patch.object(
            config, "get", side_effect=lambda k, d=None: cur_cfg.get(k, d)
        ), patch.object(config, "set", side_effect=lambda k, v, auto_save=True: cur_cfg.__setitem__(k, v)):
            card = ReactionCommandsCard()
            try:
                ids_edge = [card.cmd_list.item(i).data(0x0100) for i in range(card.cmd_list.count())]
                self.assertNotIn(OMNIVOICE_WHISPER_ID, ids_edge)

                cur_cfg["active_engine"] = "omnivoice"
                card.rebuild_list()
                ids_ov = [card.cmd_list.item(i).data(0x0100) for i in range(card.cmd_list.count())]
                self.assertIn(OMNIVOICE_WHISPER_ID, ids_ov)

                cur_cfg["reaction_omnivoice_enabled"] = False
                card.rebuild_list()
                ids_off = [card.cmd_list.item(i).data(0x0100) for i in range(card.cmd_list.count())]
                self.assertNotIn(OMNIVOICE_WHISPER_ID, ids_off)

                # Мова дефолтного списку команд береться з фільтра мов (app_language), а не ui_language
                cur_cfg["app_language"] = "pl"
                self.assertEqual(card._ui_lang(), "pl")
            finally:
                card.close()
                card.deleteLater()

        # У діалозі редагування шепоту не можна прикріпити аудіофайл
        from ui.reaction_commands import ReactionCommandDialog

        w_item = build_default_whisper_command("uk")
        w_dlg = ReactionCommandDialog(w_item, omnivoice_enabled=True)
        try:
            self.assertTrue(w_dlg.btn_pick_file.isHidden())
            self.assertTrue(w_dlg.btn_preview_file.isHidden())
            self.assertTrue(w_dlg.btn_clear_file.isHidden())
            w_dlg._data["audio_file"] = "ignored.wav"
            w_dlg._on_save()
            self.assertEqual(w_dlg.get_result().get("audio_file"), "")
        finally:
            w_dlg.close()
            w_dlg.deleteLater()

        # Перехресна мова та `ё`/`е` + `шопот`: має працювати навіть коли фільтр мови `uk`, а друкують `\шёпот` або `\шопот`
        specs_uk = build_active_reaction_specs(cfg_ov_on, engine_id="omnivoice", ui_lang="uk")
        for variant in (r"\шёпот", r"\шепот", r"\шопот", r"\шепіт", r"\whisper"):
            self.assertEqual(
                complete_slash_command(f"пиривет меня звыть {variant}", "uk", exact_only=True, specs=specs_uk),
                "пиривет меня звыть [whisper] ",
                msg=f"complete_slash_command failed for {variant}",
            )
            self.assertEqual(
                expand_omnivoice_tags(f"пиривет меня звыть {variant} Дима", specs=specs_uk, ui_lang="uk"),
                "пиривет меня звыть [whisper] Дима",
                msg=f"expand_omnivoice_tags failed for {variant}",
            )


class TypingUndoRedoTests(unittest.TestCase):
    def test_keyboard_hook_undo_redo_ctrl_z_and_ctrl_y(self) -> None:
        from core.win_api import KeyboardHookManager

        hook = KeyboardHookManager()
        self.assertFalse(hasattr(hook, "test_tag_hint_mode_toggled"))
        hook._is_typing = True
        hook._buffer = ""
        hook._reset_undo_history()

        # Введення слова + вставка тегу + очищення + Undo/Redo
        hook._push_undo_snapshot("type", force=True)
        hook._buffer = "Привіт "
        hook._push_undo_snapshot("tag", force=True)
        hook._buffer = "Привіт [хаха] "
        hook._push_undo_snapshot("clear", force=True)
        hook._buffer = ""

        self.assertTrue(hook.undo_typing())
        self.assertEqual(hook._buffer, "Привіт [хаха] ")
        self.assertTrue(hook.undo_typing())
        self.assertEqual(hook._buffer, "Привіт ")
        self.assertTrue(hook.undo_typing())
        self.assertEqual(hook._buffer, "")
        self.assertFalse(hook.undo_typing())

        self.assertTrue(hook.redo_typing())
        self.assertEqual(hook._buffer, "Привіт ")
        self.assertTrue(hook.redo_typing())
        self.assertEqual(hook._buffer, "Привіт [хаха] ")
        self.assertTrue(hook.redo_typing())
        self.assertEqual(hook._buffer, "")
        self.assertFalse(hook.redo_typing())

    def test_reaction_hotkey_binds_in_hook_and_fallback_window(self) -> None:
        from PySide6.QtWidgets import QApplication
        from core.win_api import KeyboardHookManager
        from ui.overlay_window import FallbackInputWindow

        app = QApplication.instance() or QApplication([])
        self.assertIsNotNone(app)

        hook = KeyboardHookManager()
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "laugh.wav"
            wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
            cfg = {
                "app_language": "uk",
                "active_engine": "edge_tts",
                "input_mode": "hook",
                "overlay_style": "text_only",
                "reaction_omnivoice_enabled": False,
                "reaction_commands": [
                    {
                        "id": "rc_bound",
                        "command": "хаха",
                        "aliases": ["хаха"],
                        "display": "",
                        "description": "Сміх",
                        "audio_file": str(wav),
                        "omnivoice_tag": "",
                        "hotkey": "Alt+1",
                        "enabled": True,
                    },
                    {
                        "id": "rc_disabled_bind",
                        "command": "бум",
                        "aliases": ["бум"],
                        "display": "💥",
                        "description": "",
                        "audio_file": "",
                        "omnivoice_tag": "",
                        "hotkey": "Alt+2",
                        "enabled": True,
                    },
                ],
            }
            hook.update_from_config(cfg)
            # Лише команда з файлом отримує активний бінд
            self.assertEqual(len(hook._reaction_hotkeys), 1)
            self.assertEqual(hook._reaction_hotkeys[0][1], "[хаха]")

            # Поза режимом набору бінд не спрацьовує
            hook._is_typing = False
            self.assertFalse(hook.trigger_reaction_hotkey("[хаха]"))

            # У режимі набору (оверлей) вставляється прямо в буфер із підтримкою Ctrl+Z
            hook._is_typing = True
            hook._buffer = "Привіт"
            hook._reset_undo_history()
            self.assertTrue(hook.trigger_reaction_hotkey("[хаха]"))
            self.assertEqual(hook._buffer, "Привіт [хаха] ")
            self.assertTrue(hook.undo_typing())
            self.assertEqual(hook._buffer, "Привіт")

            # У режимі вікна вводу (fallback) шле сигнал, і FallbackInputWindow вставляє в курсор
            received: list[str] = []
            hook.reaction_hotkey_triggered.connect(received.append)
            hook._input_mode = "fallback"
            self.assertTrue(hook.trigger_reaction_hotkey("[хаха]"))
            self.assertEqual(received, ["[хаха]"])

            fw = FallbackInputWindow()
            try:
                fw.input_edit.setPlainText("Ого")
                fw.input_edit.end()
                fw.insert_reaction_tag("[хаха]")
                self.assertEqual(fw.input_edit.toPlainText(), "Ого [хаха] ")
            finally:
                fw.close()
                fw.deleteLater()


class OverlayTagHintPaintTests(unittest.TestCase):
    def test_overlay_cycles_all_five_modes_and_renders_picker(self) -> None:
        from PySide6.QtCore import QPoint
        from PySide6.QtGui import QImage
        from PySide6.QtWidgets import QApplication

        from core.config import config
        from ui.overlay_window import ScreenOverlayWindow, TAG_HINT_MODES

        app = QApplication.instance() or QApplication([])
        self.assertIsNotNone(app)
        ov = ScreenOverlayWindow()
        try:
            seen_modes = [ov._tag_hint_style]
            for _ in range(len(TAG_HINT_MODES)):
                seen_modes.append(ov.toggle_tag_hint_style())
            self.assertEqual(len(set(seen_modes[:-1])), len(TAG_HINT_MODES))
            self.assertEqual(seen_modes[-1], seen_modes[0])

            with patch.object(
                config,
                "get",
                side_effect=lambda k, d=None: {
                    "active_engine": "omnivoice",
                    "app_language": "uk",
                    "overlay_style": "text_only",
                    "overlay_font_size": 18,
                    "overlay_outline_width": 3,
                    "overlay_shadow_blur": 10,
                    "overlay_text_opacity": 100,
                    "overlay_bg_opacity": 78,
                    "overlay_enabled": True,
                    "reaction_omnivoice_enabled": True,
                }.get(k, d),
            ):
                ov._mode = "typing"
                ov._text = "Привіт [laughter] \\"
                matches = match_reaction_tags("", "uk")
                for mode in (
                    "menu_mini",
                    "menu_micro",
                    "menu_v_compact",
                    "emoji_grid",
                    "emoji_dock",
                ):
                    ov._tag_picker.sync_state(
                        style=mode,
                        matches=matches,
                        ui_lang="uk",
                        selected_idx=0,
                        anchor_top_center=QPoint(500, 600),
                        anchor_bottom_y=650,
                        toast_text=f"Test {mode}",
                    )
                    ov._tag_picker._step_scroll(1)
                    pimg = QImage(ov._tag_picker.size(), QImage.Format.Format_ARGB32_Premultiplied)
                    pimg.fill(0)
                    ov._tag_picker.render(pimg)

                ov.preview_tag_picker_style("menu_mini", duration_ms=1500)
                self.assertTrue(ov._picker_preview_active)
                ov._end_picker_preview()
                ov._tag_picker.hide_picker()
        finally:
            ov.close()
            ov.deleteLater()

    def test_reaction_commands_card_locked_checkbox_and_reorder(self) -> None:
        from PySide6.QtWidgets import QApplication, QCheckBox

        from core.config import config
        from ui.reaction_commands import ReactionCommandsCard

        app = QApplication.instance() or QApplication([])
        self.assertIsNotNone(app)

        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "clip.wav"
            wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
            fake_cfg: dict = {
                "app_language": "uk",
                "active_engine": "edge_tts",
                "reaction_picker_style": "emoji_dock",
                "reaction_omnivoice_enabled": False,
                "reaction_commands": [
                    {
                        "id": "rc_a",
                        "command": "безфайлу",
                        "aliases": ["безфайлу"],
                        "display": "",
                        "description": "",
                        "audio_file": "",
                        "omnivoice_tag": "",
                        "enabled": True,
                    },
                    {
                        "id": "rc_b",
                        "command": "хаха",
                        "aliases": ["хаха"],
                        "display": "",
                        "description": "Сміх",
                        "audio_file": str(wav),
                        "omnivoice_tag": "",
                        "enabled": True,
                    },
                ],
            }

            def _get(k: str, d=None):
                return fake_cfg.get(k, d)

            def _set(k: str, v, auto_save: bool = True):
                fake_cfg[k] = v

            with patch.object(config, "_data", fake_cfg), patch.object(config, "get", side_effect=_get), patch.object(
                config, "set", side_effect=_set
            ):
                card = ReactionCommandsCard()
                try:
                    # Згорнуто за замовчуванням, як у блоці VBAN
                    self.assertTrue(card.body.isHidden())
                    self.assertEqual(card.cmd_list.count(), 2)

                    # Перший рядок без файлу -> галочка заблокована й знята
                    w0 = card.cmd_list.itemWidget(card.cmd_list.item(0))
                    chk0 = w0.findChild(QCheckBox)
                    self.assertIsNotNone(chk0)
                    assert chk0 is not None
                    self.assertFalse(chk0.isEnabled())
                    self.assertFalse(chk0.isChecked())

                    # Другий рядок із файлом -> галочка активна й увімкнена
                    w1 = card.cmd_list.itemWidget(card.cmd_list.item(1))
                    chk1 = w1.findChild(QCheckBox)
                    self.assertIsNotNone(chk1)
                    assert chk1 is not None
                    self.assertTrue(chk1.isEnabled())
                    self.assertTrue(chk1.isChecked())

                    # Перетягування змінює порядок у конфігу (і зберігає прихований rc_whisper)
                    card._on_order_changed(["rc_b", "rc_a"])
                    self.assertEqual([x["id"] for x in fake_cfg["reaction_commands"]], ["rc_b", "rc_a", "rc_whisper"])
                finally:
                    card.close()
                    card.deleteLater()

    def test_collapsible_animation_zero_jump_for_widgets_above(self) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication, QFrame, QLabel, QScrollArea, QToolButton, QVBoxLayout, QWidget

        from ui.dialogs import toggle_collapsible_widget

        app = QApplication.instance() or QApplication([])
        self.assertIsNotNone(app)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        content = QWidget()
        c_lay = QVBoxLayout(content)
        c_lay.setSpacing(12)

        above = QFrame()
        above.setMinimumHeight(260)
        al = QVBoxLayout(above)
        al.addWidget(QLabel("Above"))
        c_lay.addWidget(above)

        card = QFrame()
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(16, 10, 16, 10)
        card_lay.setSpacing(0)
        btn = QToolButton()
        btn.setCheckable(True)
        card_lay.addWidget(btn)
        body = QWidget()
        b_lay = QVBoxLayout(body)
        b_lay.setContentsMargins(0, 8, 0, 0)
        inner = QFrame()
        inner.setFixedHeight(240)
        b_lay.addWidget(inner)
        body.setVisible(False)
        card_lay.addWidget(body)
        c_lay.addWidget(card)

        below = QFrame()
        below.setMinimumHeight(260)
        c_lay.addWidget(below)
        c_lay.addStretch(1)

        scroll.setWidget(content)
        scroll.resize(700, 520)
        scroll.show()
        app.processEvents()
        try:
            ref_above = above.geometry().getRect()
            ref_card_y = card.geometry().y()
            toggle_collapsible_widget(btn, body, True, duration_ms=200)
            app.processEvents()
            self.assertEqual(above.geometry().getRect(), ref_above)
            self.assertEqual(card.geometry().y(), ref_card_y)

            anim = getattr(body, "_collapse_anim", None)
            self.assertIsNotNone(anim)
            assert anim is not None
            for t_ms in (20, 100, 180, 220):
                anim.setCurrentTime(t_ms)
                app.processEvents()
                self.assertEqual(above.geometry().getRect(), ref_above)
                self.assertEqual(card.geometry().y(), ref_card_y)
        finally:
            scroll.close()
            scroll.deleteLater()

    def test_hook_typing_coalescing_and_enter_slash_selection(self) -> None:
        import threading
        from PySide6.QtWidgets import QApplication
        from core.win_api import KeyboardHookManager

        app = QApplication.instance() or QApplication([])
        self.assertIsNotNone(app)

        hook = KeyboardHookManager()
        hook.update_from_config(
            {
                "app_language": "ru",
                "active_engine": "omnivoice",
                "reaction_omnivoice_enabled": True,
                "reaction_commands": get_default_reaction_commands("ru"),
            }
        )
        received: list[str] = []
        hook.typing_text_changed.connect(lambda txt, _lang: received.append(txt))

        # 1. Крос-потоковий потік автоповтору "оооо..." зливається в 1 оновлення з найсвіжішим текстом
        def _bg_spam() -> None:
            for n in range(1, 41):
                hook._emit_typing_text("о" * n, "RU")

        th = threading.Thread(target=_bg_spam)
        th.start()
        th.join()
        app.processEvents()
        self.assertEqual(received, ["о" * 40])

        # 2. Enter при голому `\` або після навігації в меню вставляє обрану команду замість відправки голого `\`
        hook._is_typing = True
        hook._buffer = "шопот \\"
        hook._slash_selected_idx = 0
        hook._slash_navigated = False
        self.assertTrue(hook._try_complete_slash_on_enter())
        self.assertEqual(hook._buffer, "шопот [laughter] ")

        # Навігація стрілками/мишкою на конкретний пункт + Enter теж вставляє обраний тег
        hook._buffer = "привет \\ш"
        hook.set_slash_selected_index(0)
        self.assertTrue(hook._try_complete_slash_on_enter())
        self.assertEqual(hook._buffer, "привет [whisper] ")

        # Якщо користувач сам додрукував команду без навігації — Enter відправляє одразу (повертає False)
        hook._buffer = "привет \\хаха"
        hook._slash_navigated = False
        self.assertFalse(hook._try_complete_slash_on_enter())

    def test_quick_phrase_dialog_remove_audio_button_visibility_and_action(self) -> None:
        from PySide6.QtWidgets import QApplication
        from core.win_api import KeyboardHookManager
        from ui.dialogs import QuickPhraseDialog

        app = QApplication.instance() or QApplication([])
        self.assertIsNotNone(app)
        hook = KeyboardHookManager()

        # 1. Без прикріпленого файлу кнопки «Прослухати» та «Видалити» приховані
        dlg_empty = QuickPhraseDialog(hook)
        try:
            self.assertTrue(dlg_empty.btn_preview_audio.isHidden())
            self.assertTrue(dlg_empty.btn_remove_audio.isHidden())
        finally:
            dlg_empty.close()
            dlg_empty.deleteLater()

        # 2. З прикріпленим файлом кнопка видалення показана справа від тексту й під «Прослухати»,
        # і при натисканні очищує `audio_file` (не видаляючи початковий файл до збереження)
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "phrase.wav"
            wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
            dlg = QuickPhraseDialog(
                hook,
                {"title": "Тест", "text": "Привіт", "hotkey": "", "emoji": "💬", "audio_file": str(wav)},
            )
            dlg.show()
            app.processEvents()
            try:
                self.assertFalse(dlg.btn_preview_audio.isHidden())
                self.assertFalse(dlg.btn_remove_audio.isHidden())
                # Кнопка видалення має бути справа від тексту статусу та нижче за кнопку «Прослухати»
                self.assertGreater(dlg.btn_remove_audio.geometry().left(), dlg.audio_status_lbl.geometry().left())
                self.assertGreater(dlg.btn_remove_audio.geometry().top(), dlg.btn_preview_audio.geometry().top())

                dlg.btn_remove_audio.click()
                self.assertEqual(dlg.get_result().get("audio_file"), "")
                self.assertTrue(dlg.btn_preview_audio.isHidden())
                self.assertTrue(dlg.btn_remove_audio.isHidden())
                # Оригінальний файл ще на диску на випадок «Скасувати»
                self.assertTrue(wav.exists())

                # А тимчасовий файл, доданий у цій сесії, при видаленні прибирається і з диска
                tmp_wav = Path(td) / "session_rec.wav"
                tmp_wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
                dlg._data["audio_file"] = str(tmp_wav)
                dlg._session_audio_files.add(str(tmp_wav))
                dlg._refresh_audio_status()
                self.assertFalse(dlg.btn_remove_audio.isHidden())
                dlg.btn_remove_audio.click()
                self.assertEqual(dlg.get_result().get("audio_file"), "")
                self.assertFalse(tmp_wav.exists())
            finally:
                dlg.close()
                dlg.deleteLater()


if __name__ == "__main__":
    unittest.main()
