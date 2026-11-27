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

                    # Перетягування змінює порядок у конфігу
                    card._on_order_changed(["rc_b", "rc_a"])
                    self.assertEqual([x["id"] for x in fake_cfg["reaction_commands"]], ["rc_b", "rc_a"])
                finally:
                    card.close()
                    card.deleteLater()


if __name__ == "__main__":
    unittest.main()
