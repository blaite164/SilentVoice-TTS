import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import core.updater as updater


class UpdaterUxTests(unittest.TestCase):
    def test_build_installer_args_uses_visible_silent_mode_and_language(self) -> None:
        args = updater.build_installer_args(silent=True, ui_lang="uk")
        self.assertIn("/SILENT", args)
        self.assertNotIn("/VERYSILENT", args)
        self.assertIn("/SP-", args)
        self.assertIn("/AUTORELAUNCH=1", args)
        self.assertIn("/LANG=uk", args)

        fallback_args = updater.build_installer_args(silent=True, ui_lang="jp")
        self.assertIn("/LANG=en", fallback_args)

    def test_download_update_installer_reuses_verified_cached_exe(self) -> None:
        payload = b"MZ" + b"\x00" * 1024
        sha = hashlib.sha256(payload).hexdigest()
        release = updater.UpdateReleaseInfo(
            version="1.0.2",
            tag_name="v1.0.2",
            title="v1.0.2",
            notes="",
            installer_url="https://example.invalid/SilentVoice_Setup_1.0.2.exe",
            installer_name="SilentVoice_Setup_1.0.2.exe",
            installer_size=len(payload),
            sha256=sha,
        )
        with tempfile.TemporaryDirectory() as tmp:
            updates_dir = Path(tmp)
            cached = updates_dir / "SilentVoice_Setup_1.0.2.exe"
            cached.write_bytes(payload)

            progress_calls: list[tuple[int, float, float]] = []
            with (
                mock.patch.object(updater, "UPDATES_DIR", updates_dir),
                mock.patch.object(updater.requests, "get", side_effect=AssertionError("Should not re-download")),
            ):
                result = updater.download_update_installer(
                    release,
                    progress_cb=lambda p, dl, tot: progress_calls.append((p, dl, tot)),
                )

            self.assertEqual(result, cached)
            self.assertTrue(progress_calls)
            self.assertEqual(progress_calls[-1][0], 100)

    def test_extract_and_localize_bilingual_release_notes(self) -> None:
        bilingual = (
            "## 🇺🇦 Що нового у v1.0.2\n"
            "- Виправлено курсор OBS\n\n"
            "---\n\n"
            "## 🇬🇧 What's New in v1.0.2\n"
            "- Fixed OBS cursor\n"
        )
        uk_txt, uk_done = updater.extract_release_notes_section(bilingual, "uk")
        self.assertTrue(uk_done)
        self.assertIn("Виправлено курсор OBS", uk_txt)
        self.assertNotIn("Fixed OBS cursor", uk_txt)

        en_txt, en_done = updater.extract_release_notes_section(bilingual, "en")
        self.assertTrue(en_done)
        self.assertIn("Fixed OBS cursor", en_txt)
        self.assertNotIn("Виправлено курсор OBS", en_txt)

        de_src, de_done = updater.extract_release_notes_section(bilingual, "de")
        self.assertFalse(de_done)
        self.assertIn("Fixed OBS cursor", de_src)

        with mock.patch("core.i18n._standard_translate_text", side_effect=AssertionError("Should not translate")):
            self.assertEqual(updater.localize_release_notes(bilingual, target_lang="uk"), uk_txt)

        with mock.patch("core.i18n._standard_translate_text", return_value="## 🇩🇪 Was ist neu\n- OBS-Cursor behoben") as m_tr:
            de_out = updater.localize_release_notes(bilingual, target_lang="de")
            self.assertIn("OBS-Cursor behoben", de_out)
            m_tr.assert_called_once()
            self.assertEqual(m_tr.call_args.kwargs["target_lang"], "de")
            self.assertEqual(m_tr.call_args.kwargs["source_lang"], "auto")

    def test_update_dialog_opens_wide_and_toggles_original_notes(self) -> None:
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from core.i18n import set_ui_language
        from ui.dialogs import AppUpdateDialog

        _app = QApplication.instance() or QApplication([])
        set_ui_language("uk")
        bilingual = "## 🇺🇦 Що нового\n- Тест\n\n## 🇬🇧 What's New\n- Test"
        rel = updater.UpdateReleaseInfo(
            version="9.9.9",
            tag_name="v9.9.9",
            title="v9.9.9",
            notes=bilingual,
            installer_url="https://example.invalid/setup.exe",
            installer_name="setup.exe",
        )
        dlg = AppUpdateDialog(initial_release=rel)
        try:
            self.assertGreaterEqual(dlg.minimumWidth(), 720)
            self.assertGreaterEqual(dlg.width(), 720)
            self.assertFalse(dlg.toggle_notes_lang_btn.isHidden())
            self.assertIn("Показати оригінал", dlg.toggle_notes_lang_btn.text())
            self.assertNotIn("What's New", dlg.notes_edit.toPlainText())

            dlg._toggle_notes_original()
            self.assertIn("Перекласти", dlg.toggle_notes_lang_btn.text())
            self.assertIn("What's New", dlg.notes_edit.toPlainText())

            dlg._toggle_notes_original()
            self.assertIn("Показати оригінал", dlg.toggle_notes_lang_btn.text())
            self.assertNotIn("What's New", dlg.notes_edit.toPlainText())
        finally:
            dlg.close()


if __name__ == "__main__":
    unittest.main()
