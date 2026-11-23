import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.secret_store import (
    DPAPI_PREFIX,
    is_dpapi_available,
    protect_config_dict,
    protect_secret,
    unprotect_config_dict,
    unprotect_secret,
)


class SecretStoreTests(unittest.TestCase):
    def test_dpapi_roundtrip_and_plaintext_fallback(self) -> None:
        raw = "sk-test-secret-12345-україна"
        protected = protect_secret(raw)
        if is_dpapi_available():
            self.assertTrue(protected.startswith(DPAPI_PREFIX))
            self.assertNotIn(raw, protected)
            # Idempotent: protecting an already protected value does not double-wrap
            self.assertEqual(protect_secret(protected), protected)
        else:
            self.assertEqual(protected, raw)
        self.assertEqual(unprotect_secret(protected), raw)
        # Legacy plaintext passes through cleanly
        self.assertEqual(unprotect_secret("plain-legacy-token"), "plain-legacy-token")

    def test_protect_and_unprotect_config_dict_migrates_secrets(self) -> None:
        sample = {
            "ui_language": "uk",
            "twitch_token": "oauth:supersecret123",
            "socks5_proxy_url": "socks5h://user:pass@127.0.0.1:1080",
            "online_items": [
                {"id": "el_1", "provider": "elevenlabs", "api_key": "el_key_secret"},
                {"id": "tg_1", "provider": "telegram_silero", "tg_api_hash": "0123456789abcdef0123456789abcdef", "tg_phone": "+380501234567"},
            ],
            "ai_items": [
                {"id": "oa_1", "provider": "openai", "api_key": "sk-openai-secret"},
            ],
        }
        # Plaintext config triggers migration flag on Windows
        loaded_plain, needs_mig = unprotect_config_dict(sample)
        self.assertEqual(loaded_plain["twitch_token"], "oauth:supersecret123")
        self.assertEqual(needs_mig, is_dpapi_available())

        # Protected copy hides all secrets on disk and decrypts back identically
        protected = protect_config_dict(sample)
        if is_dpapi_available():
            self.assertTrue(protected["twitch_token"].startswith(DPAPI_PREFIX))
            self.assertTrue(protected["socks5_proxy_url"].startswith(DPAPI_PREFIX))
            self.assertTrue(protected["online_items"][0]["api_key"].startswith(DPAPI_PREFIX))
            self.assertTrue(protected["online_items"][1]["tg_api_hash"].startswith(DPAPI_PREFIX))
            self.assertTrue(protected["online_items"][1]["tg_phone"].startswith(DPAPI_PREFIX))
            self.assertTrue(protected["ai_items"][0]["api_key"].startswith(DPAPI_PREFIX))
            self.assertEqual(protected["ui_language"], "uk")

        restored, needs_mig_after = unprotect_config_dict(protected)
        self.assertFalse(needs_mig_after)
        self.assertEqual(restored, sample)

    def test_telegram_app_keys_and_session_dpapi_storage(self) -> None:
        from core import telegram_silero as ts

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            keys_file = tmp_dir / "tg_app_keys.json"
            sess_base = tmp_dir / "telegram_silero"
            sess_file = sess_base.with_suffix(".session")
            sess_dpapi = tmp_dir / "telegram_silero.session.dpapi"

            with (
                patch.object(ts, "DATA_DIR", tmp_dir),
                patch.object(ts, "APP_KEYS_FILE", keys_file),
                patch.object(ts, "SESSION_PATH_BASE", sess_base),
                patch.object(ts, "SESSION_FILE", sess_file),
                patch.object(ts, "SESSION_DPAPI_FILE", sess_dpapi),
            ):
                ts.save_default_api_credentials(123456, "abcdef0123456789abcdef0123456789")
                raw_json = json.loads(keys_file.read_text(encoding="utf-8"))
                if is_dpapi_available():
                    self.assertTrue(str(raw_json["api_hash"]).startswith(DPAPI_PREFIX))
                    self.assertNotIn("abcdef0123456789", keys_file.read_text(encoding="utf-8"))
                aid, ahash, src = ts.load_default_api_credentials()
                self.assertEqual((aid, ahash, src), (123456, "abcdef0123456789abcdef0123456789", "app_keys_file"))

                # Session persistence via DPAPI
                mgr = ts.TelegramSileroManager
                self.assertFalse(mgr._has_saved_session())
                mgr._save_protected_session_string("1BVtsOHoDUMMY_SESSION_STRING")
                self.assertTrue(mgr._has_saved_session())
                disk_text = sess_dpapi.read_text(encoding="utf-8")
                if is_dpapi_available():
                    self.assertTrue(disk_text.startswith(DPAPI_PREFIX))
                    self.assertNotIn("DUMMY_SESSION_STRING", disk_text)
                self.assertEqual(unprotect_secret(disk_text), "1BVtsOHoDUMMY_SESSION_STRING")
                mgr._remove_session_files()
                self.assertFalse(mgr._has_saved_session())


if __name__ == "__main__":
    unittest.main()
