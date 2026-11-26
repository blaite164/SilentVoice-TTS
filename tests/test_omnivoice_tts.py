import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from core import omnivoice_tts
from core.config import config
from core.tts_manager import CLONE_ENGINE_IDS, ENGINE_SPECS, tts_manager
from installer.patch_omnivoice_server import patch_omnivoice_sources


class OmniVoiceHelpersTests(unittest.TestCase):
    def test_build_omnivoice_instruct_valid_and_filtered(self) -> None:
        s = omnivoice_tts.build_omnivoice_instruct(
            gender="female",
            age="young adult",
            pitch="low pitch",
            whisper=True,
            accent="british accent",
        )
        self.assertEqual(s, "female, young adult, low pitch, whisper, british accent")

        # Unknown / free-form values must be safely ignored so omnivoice.cpp never rejects the request
        s_bad = omnivoice_tts.build_omnivoice_instruct(
            gender="pirate",
            age="ancient",
            pitch="screaming",
            whisper=False,
            accent="martian",
        )
        self.assertEqual(s_bad, "")

    def test_normalizers(self) -> None:
        self.assertEqual(omnivoice_tts.normalize_omnivoice_quant("bf16"), "BF16")
        self.assertEqual(omnivoice_tts.normalize_omnivoice_quant("bad"), "Q8_0")
        self.assertEqual(omnivoice_tts.normalize_omnivoice_steps(16), 16)
        self.assertEqual(omnivoice_tts.normalize_omnivoice_steps(32), 32)
        self.assertEqual(omnivoice_tts.normalize_omnivoice_steps(1), 8)
        self.assertEqual(omnivoice_tts.normalize_omnivoice_steps(999), 64)
        self.assertEqual(omnivoice_tts.normalize_omnivoice_mode("design"), "design")
        self.assertEqual(omnivoice_tts.normalize_omnivoice_mode("unknown"), "clone")
        self.assertEqual(omnivoice_tts.normalize_omnivoice_lang_label("uk"), "Ukrainian")
        self.assertEqual(omnivoice_tts.normalize_omnivoice_lang_label("ua"), "Ukrainian")
        self.assertEqual(omnivoice_tts.normalize_omnivoice_lang_label("en"), "English")
        self.assertEqual(omnivoice_tts.normalize_omnivoice_lang_label("auto"), "None")

    def test_prepare_reference_wav_smart_pause_trim(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            sr = 24000
            # Build 14s signal with loud tone everywhere EXCEPT a clear quiet pause at 8.5s..9.0s
            t = np.linspace(0.0, 14.0, sr * 14, dtype=np.float32)
            audio = 0.4 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
            audio[int(8.5 * sr) : int(9.0 * sr)] = 0.0
            src_wav = tmp / "long_voice.wav"
            sf.write(str(src_wav), audio, sr, subtype="PCM_16")

            with mock.patch.object(omnivoice_tts, "OMNIVOICE_REF_CACHE_DIR", tmp / "cache"):
                out_wav, digest = omnivoice_tts.prepare_omnivoice_reference_wav(src_wav, max_sec=10.0)
                self.assertTrue(out_wav.is_file())
                self.assertEqual(len(digest), 16)
                out_data, out_sr = sf.read(str(out_wav), dtype="float32")
                self.assertEqual(out_sr, 24000)
                dur_s = len(out_data) / float(out_sr)
                # Cut should land inside the quiet pause (~8.75s) + 0.10s tail silence => ~8.5..9.2s
                self.assertGreater(dur_s, 8.4)
                self.assertLess(dur_s, 9.3)
                # Last 80 ms must be clean silence
                self.assertLess(float(np.max(np.abs(out_data[-int(0.08 * out_sr) :]))), 1e-3)

    def test_resolve_ref_text_priority_and_whisper_cache(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            sr = 24000
            t = np.linspace(0.0, 2.0, sr * 2, dtype=np.float32)
            audio = 0.3 * np.sin(2.0 * np.pi * 300.0 * t).astype(np.float32)
            ref = tmp / "sample.wav"
            sf.write(str(ref), audio, sr, subtype="PCM_16")

            old_vals = {
                k: config.get(k)
                for k in ("omnivoice_ref_path", "omnivoice_ref_text", "omnivoice_ref_text_path")
            }
            try:
                config.set("omnivoice_ref_path", str(ref))
                config.set("omnivoice_ref_text", "")
                config.set("omnivoice_ref_text_path", "")
                with mock.patch.object(omnivoice_tts, "OMNIVOICE_REF_CACHE_DIR", tmp / "cache"):
                    with mock.patch.object(
                        omnivoice_tts,
                        "transcribe_wav_with_whisper_cpp",
                        return_value="Привіт світе!",
                    ) as mock_whisper:
                        _w1, dig1, txt1 = omnivoice_tts.resolve_omnivoice_ref_text(ref)
                        self.assertEqual(txt1, "Привіт світе!")
                        self.assertEqual(mock_whisper.call_count, 1)

                        # Clear config text: second lookup must hit disk cache without invoking Whisper again
                        config.set("omnivoice_ref_text", "")
                        config.set("omnivoice_ref_text_path", "")
                        _w2, dig2, txt2 = omnivoice_tts.resolve_omnivoice_ref_text(ref)
                        self.assertEqual(dig1, dig2)
                        self.assertEqual(txt2, "Привіт світе!")
                        self.assertEqual(mock_whisper.call_count, 1)
            finally:
                for k, v in old_vals.items():
                    config.set(k, v)


class OmniVoiceServerPayloadTests(unittest.TestCase):
    def test_payload_includes_steps_denoise_and_design_fields(self) -> None:
        mgr = omnivoice_tts._OmniVoiceServerManager()
        mgr._supports_steps = True
        payload = mgr._build_speech_payload(
            text="Привіт [laughter]",
            lang_label="Ukrainian",
            voice_name="",
            instruct="female, young adult, whisper",
            seed=12345,
            steps=16,
            denoise=True,
            response_format="wav",
        )
        self.assertEqual(payload["input"], "Привіт [laughter]")
        self.assertEqual(payload["language"], "Ukrainian")
        self.assertEqual(payload["instructions"], "female, young adult, whisper")
        self.assertEqual(payload["seed"], 12345)
        self.assertEqual(payload["steps"], 16)
        self.assertTrue(payload["denoise"])
        self.assertNotIn("voice", payload)

        # Unpatched server omits steps/denoise
        mgr._supports_steps = False
        payload_legacy = mgr._build_speech_payload(
            text="Hello",
            lang_label="English",
            voice_name="clone_abc",
            instruct="",
            seed=None,
            steps=16,
            denoise=True,
            response_format="pcm",
        )
        self.assertEqual(payload_legacy["voice"], "clone_abc")
        self.assertNotIn("steps", payload_legacy)
        self.assertNotIn("denoise", payload_legacy)

    def test_patch_script_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "src").mkdir()
            (repo / "tools").mkdir()
            (repo / "src" / "tts-server.h").write_text(
                "struct tts_request {\n    int64_t seed;\n};\n"
                "struct server_config {\n    int         port = 8080;\n};\n"
                "    req.seed = seed ? yyjson_get_sint(seed) : -1;\n"
                "    // permissive CORS so a browser client can call the API directly.\n"
                "    svr.set_default_headers({\n"
                '        { "Access-Control-Allow-Origin", "*" }\n'
                "    });\n"
                '    svr.Options("/.*", [](const httplib::Request &, httplib::Response & res) {\n'
                '        res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");\n'
                '        res.set_header("Access-Control-Allow-Headers", "Content-Type");\n'
                "    });\n",
                encoding="utf-8",
            )
            (repo / "tools" / "tts-server.cpp").write_text(
                '            "  --clamp-fp16            Clamp hidden states to FP16 range\\n",\n'
                '        } else if (!std::strcmp(arg, "--clamp-fp16")) {\n'
                "            clamp_fp16 = true;\n"
                "        p.mg_seed = (req.seed < 0) ? (uint64_t) std::random_device{}() : (uint64_t) req.seed;\n",
                encoding="utf-8",
            )
            patch_omnivoice_sources(repo)
            first_h = (repo / "src" / "tts-server.h").read_text(encoding="utf-8")
            first_c = (repo / "tools" / "tts-server.cpp").read_text(encoding="utf-8")
            patch_omnivoice_sources(repo)
            self.assertEqual(first_h, (repo / "src" / "tts-server.h").read_text(encoding="utf-8"))
            self.assertEqual(first_c, (repo / "tools" / "tts-server.cpp").read_text(encoding="utf-8"))
            self.assertIn("set_pre_routing_handler", first_h)
            self.assertIn("p.mg_num_step = req.steps;", first_c)


class OmniVoiceIntegrationTests(unittest.TestCase):
    def test_registered_in_engine_specs_and_clone_ids(self) -> None:
        self.assertIn("omnivoice", ENGINE_SPECS)
        self.assertIn("omnivoice", CLONE_ENGINE_IDS)
        langs = {v["lang"] for v in ENGINE_SPECS["omnivoice"]["voices"]}
        self.assertEqual(langs, {"uk", "en", "ru", "pl", "de", "fr", "es"})

    def test_audio_cache_key_tracks_omnivoice_settings(self) -> None:
        old_steps = config.get("omnivoice_steps", 16)
        old_whisper = config.get("omnivoice_whisper", False)
        old_mode = config.get("omnivoice_mode", "clone")
        try:
            config.set("omnivoice_mode", "design")
            config.set("omnivoice_steps", 16)
            config.set("omnivoice_whisper", False)
            k1 = tts_manager._build_audio_cache_key("Привіт", "omnivoice", "omnivoice_uk")

            config.set("omnivoice_steps", 32)
            k2 = tts_manager._build_audio_cache_key("Привіт", "omnivoice", "omnivoice_uk")
            self.assertNotEqual(k1, k2)

            config.set("omnivoice_whisper", True)
            k3 = tts_manager._build_audio_cache_key("Привіт", "omnivoice", "omnivoice_uk")
            self.assertNotEqual(k2, k3)
        finally:
            config.set("omnivoice_steps", old_steps)
            config.set("omnivoice_whisper", old_whisper)
            config.set("omnivoice_mode", old_mode)

    def test_omnivoice_settings_dialog_roundtrip(self) -> None:
        from PySide6.QtWidgets import QApplication
        from ui.dialogs import OmniVoiceSettingsDialog

        app = QApplication.instance() or QApplication([])
        _ = app
        keys = (
            "omnivoice_mode",
            "omnivoice_steps",
            "omnivoice_whisper",
            "omnivoice_denoise",
            "omnivoice_design_gender",
            "omnivoice_design_age",
            "omnivoice_design_pitch",
            "omnivoice_design_accent",
            "omnivoice_design_seed",
        )
        old_vals = {k: config.get(k) for k in keys}
        try:
            dlg = OmniVoiceSettingsDialog()
            dlg.radio_design.setChecked(True)
            dlg.whisper_chk.setChecked(True)
            dlg.denoise_chk.setChecked(False)
            dlg.seed_spin.setValue(777)
            idx32 = dlg.steps_combo.findData(32)
            self.assertGreaterEqual(idx32, 0)
            dlg.steps_combo.setCurrentIndex(idx32)
            dlg._save_and_accept()

            self.assertEqual(config.get("omnivoice_mode"), "design")
            self.assertEqual(config.get("omnivoice_steps"), 32)
            self.assertTrue(config.get("omnivoice_whisper"))
            self.assertFalse(config.get("omnivoice_denoise"))
            self.assertEqual(config.get("omnivoice_design_seed"), 777)
        finally:
            for k, v in old_vals.items():
                config.set(k, v)

    def test_omnivoice_whisper_hotkey_only_active_with_omnivoice(self) -> None:
        from core.settings_transfer import CATEGORY_KEYS
        from core.win_api import KeyboardHookManager

        self.assertIn("hotkey_toggle_omnivoice_whisper", CATEGORY_KEYS["hotkeys"])

        hook = KeyboardHookManager()
        hook.update_from_config({
            "active_engine": "piper",
            "hotkey_toggle_omnivoice_whisper": "Ctrl+Shift+W",
        })
        self.assertFalse(hook._omnivoice_active)
        self.assertIsNotNone(hook._hk_toggle_omnivoice_whisper)

        hook.update_from_config({
            "active_engine": "omnivoice",
            "hotkey_toggle_omnivoice_whisper": "Ctrl+Shift+W",
        })
        self.assertTrue(hook._omnivoice_active)
        self.assertIsNotNone(hook._hk_toggle_omnivoice_whisper)

    def test_omnivoice_runtime_candidate_urls_include_gitlab_mirror(self) -> None:
        from core.tts_manager import tts_manager

        gh = (
            "https://github.com/blaite164/SilentVoice-TTS/releases/download/"
            "omnivoice-runtime-v1/omnivoice-runtime-win-cuda-x64.zip"
        )
        gl = (
            "https://gitlab.com/api/v4/projects/blaite164%2Fsilentvoice-tts"
            "/packages/generic/omnivoice-runtime/1/omnivoice-runtime-win-cuda-x64.zip"
        )
        cands_no_bypass = tts_manager._build_candidate_urls(gh, bypass_enabled=False)
        self.assertIn(gl, cands_no_bypass)
        cands_bypass = tts_manager._build_candidate_urls(gh, bypass_enabled=True)
        self.assertIn(gl, cands_bypass)
        self.assertLess(cands_bypass.index(gl), cands_bypass.index(f"https://gh-proxy.com/{gh}"))

    def test_slow_direct_preferred_over_slow_proxies(self) -> None:
        from core.tts_manager import tts_manager

        url = "https://example.com/model.bin"
        calls: list[dict] = []

        def fake_stream(
            target_url: str,
            dest: Path,
            engine_id: str,
            base_pct: int,
            span_pct: int,
            label: str,
            **kwargs,
        ) -> None:
            calls.append({
                "url": target_url,
                "proxies": kwargs.get("proxies"),
                "proxy_pool": kwargs.get("proxy_pool"),
                "abort_if_slow": kwargs.get("abort_if_slow"),
            })
            if kwargs.get("abort_if_slow"):
                raise tts_manager._SlowProxyError("too slow")
            dest.write_bytes(b"ok")

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "model.bin"
            with (
                mock.patch.object(tts_manager, "_build_candidate_urls", return_value=[url]),
                mock.patch.object(tts_manager, "_stream_download_attempt", side_effect=fake_stream),
                mock.patch.object(tts_manager, "_has_internet_probe", return_value=True),
                mock.patch.object(tts_manager, "_running", True),
                mock.patch.object(tts_manager, "_get_cached_proxies", return_value=["http://1.2.3.4:8080", "http://5.6.7.8:8080"]),
                mock.patch.object(tts_manager, "_find_working_proxies_for_url", return_value=["http://9.9.9.9:8080"]),
                mock.patch.object(tts_manager, "_probe_range_info", return_value=(10 * 1024 * 1024, url)),
                mock.patch("core.tts_manager.get_active_api_proxies", return_value=None),
                mock.patch("core.tts_manager.config.get", side_effect=lambda k, d=None: True if k == "use_download_bypass" else ("" if k == "custom_proxy_url" else d)),
            ):
                tts_manager._download_url_with_progress(url, dest, "omnivoice", 0, 50, "Test")

            self.assertTrue(dest.is_file())
            self.assertGreaterEqual(len(calls), 2)
            # Final winning fallback call must be the direct route (no proxies / no proxy_pool)
            last_call = calls[-1]
            self.assertIsNone(last_call["proxies"])
            self.assertIsNone(last_call["proxy_pool"])
            self.assertFalse(last_call["abort_if_slow"])

    def test_bundled_clone_sample_auto_guide_text(self) -> None:
        from core.omnivoice_tts import (
            CLONE_GUIDE_SAMPLE_TEXTS,
            detect_bundled_clone_sample_lang,
            get_bundled_clone_sample_guide_text,
        )

        self.assertEqual(
            detect_bundled_clone_sample_lang(r"C:\Mia\123\TTS\data\samples\clone_voices\uk\Alexandra Ж.mp3"),
            "uk",
        )
        self.assertEqual(
            detect_bundled_clone_sample_lang(r"C:\Mia\123\TTS\data\samples\clone_voices\ru\Dmitry Ч.mp3"),
            "ru",
        )
        self.assertEqual(
            detect_bundled_clone_sample_lang(r"C:\Mia\123\TTS\data\samples\clone_voices\en\Selene.mp3"),
            "en",
        )
        self.assertIsNone(detect_bundled_clone_sample_lang(r"C:\Users\test\Desktop\my_voice.wav"))

        self.assertEqual(
            get_bundled_clone_sample_guide_text(r"C:\Mia\123\TTS\data\samples\clone_voices\uk\Alexandra Ж.mp3"),
            CLONE_GUIDE_SAMPLE_TEXTS["uk"],
        )
        self.assertEqual(
            get_bundled_clone_sample_guide_text(r"C:\Mia\123\TTS\data\samples\clone_voices\ru\Dmitry Ч.mp3"),
            CLONE_GUIDE_SAMPLE_TEXTS["ru"],
        )
        self.assertEqual(get_bundled_clone_sample_guide_text(r"C:\Users\test\Desktop\my_voice.wav"), "")

    def test_ref_cache_prune_keeps_latest_entries(self) -> None:
        import os
        import core.omnivoice_tts as ov

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            for i in range(6):
                wav_p = cache_dir / f"ovref_{i:04d}.wav"
                js_p = cache_dir / f"ovref_{i:04d}.json"
                wav_p.write_bytes(b"RIFF")
                js_p.write_text('{"text":"hi"}', encoding="utf-8")
                os.utime(wav_p, (1000 + i * 10, 1000 + i * 10))
            with mock.patch.object(ov, "OMNIVOICE_REF_CACHE_DIR", cache_dir):
                ov._prune_ref_cache(max_entries=3)
            remaining = sorted(p.name for p in cache_dir.glob("ovref_*.wav"))
            self.assertEqual(remaining, ["ovref_0003.wav", "ovref_0004.wav", "ovref_0005.wav"])
            self.assertFalse((cache_dir / "ovref_0000.json").exists())

    def test_cold_start_synthesize_does_not_self_cancel(self) -> None:
        import io
        import core.omnivoice_tts as ov

        buf = io.BytesIO()
        sr = 24000
        sf.write(buf, np.zeros(sr // 10, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        class _FakeResp:
            status_code = 200
            content = wav_bytes
            text = ""

            def iter_content(self, chunk_size: int = 32768):
                yield wav_bytes

            def close(self) -> None:
                pass

        mgr = ov._OmniVoiceServerManager()
        with (
            mock.patch.object(ov, "is_omnivoice_runtime_installed", return_value=True),
            mock.patch.object(
                ov,
                "find_installed_omnivoice_weights",
                return_value=(Path("base.gguf"), Path("codec.gguf"), "Q8_0"),
            ),
            mock.patch.object(mgr, "_spawn_and_wait", return_value=18999),
            mock.patch.object(mgr._session, "post", return_value=_FakeResp()),
        ):
            audio, out_sr = mgr.synthesize(text="Привіт!", lang_label="Ukrainian")
        self.assertEqual(out_sr, sr)
        self.assertGreater(len(audio), 0)


if __name__ == "__main__":
    unittest.main()
