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


if __name__ == "__main__":
    unittest.main()
