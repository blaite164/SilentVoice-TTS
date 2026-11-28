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
        old_mode = config.get("omnivoice_mode", "clone")
        old_seed = config.get("omnivoice_design_seed", 42)
        try:
            config.set("omnivoice_mode", "design")
            config.set("omnivoice_steps", 16)
            config.set("omnivoice_design_seed", 42)
            k1 = tts_manager._build_audio_cache_key("Привіт", "omnivoice", "omnivoice_uk")

            config.set("omnivoice_steps", 32)
            k2 = tts_manager._build_audio_cache_key("Привіт", "omnivoice", "omnivoice_uk")
            self.assertNotEqual(k1, k2)

            k3 = tts_manager._build_audio_cache_key("[whisper] Привіт", "omnivoice", "omnivoice_uk")
            self.assertNotEqual(k2, k3)
        finally:
            config.set("omnivoice_steps", old_steps)
            config.set("omnivoice_mode", old_mode)
            config.set("omnivoice_design_seed", old_seed)

    def test_omnivoice_settings_dialog_roundtrip(self) -> None:
        from PySide6.QtWidgets import QApplication
        from ui.dialogs import OmniVoiceSettingsDialog

        app = QApplication.instance() or QApplication([])
        _ = app
        keys = (
            "omnivoice_mode",
            "omnivoice_steps",
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
            self.assertFalse(hasattr(dlg, "whisper_chk"))
            dlg.radio_design.setChecked(True)
            dlg.denoise_chk.setChecked(False)
            dlg.seed_spin.setValue(777)
            idx32 = dlg.steps_combo.findData(32)
            self.assertGreaterEqual(idx32, 0)
            dlg.steps_combo.setCurrentIndex(idx32)
            dlg._save_and_accept()

            self.assertEqual(config.get("omnivoice_mode"), "design")
            self.assertEqual(config.get("omnivoice_steps"), 32)
            self.assertFalse(config.get("omnivoice_denoise"))
            self.assertEqual(config.get("omnivoice_design_seed"), 777)
        finally:
            for k, v in old_vals.items():
                config.set(k, v)

    def test_omnivoice_global_whisper_hotkey_removed_and_inline_whisper_synthesizes(self) -> None:
        import numpy as np
        from core.omnivoice_tts import _omnivoice_server, synthesize_omnivoice_tts
        from core.settings_transfer import CATEGORY_KEYS
        from core.win_api import KeyboardHookManager

        self.assertNotIn("hotkey_toggle_omnivoice_whisper", CATEGORY_KEYS["hotkeys"])
        self.assertNotIn("omnivoice_whisper", CATEGORY_KEYS["voices"])

        hook = KeyboardHookManager()
        self.assertFalse(hasattr(hook, "toggle_omnivoice_whisper_triggered"))
        self.assertFalse(hasattr(hook, "_hk_toggle_omnivoice_whisper"))

        calls: list[tuple[str, str]] = []

        def _fake_synth(*, text: str, instruct: str = "", **_kw):
            calls.append((text, instruct))
            return np.ones(240, dtype=np.float32) * 0.1, 24000

        with mock.patch("core.omnivoice_tts.is_omnivoice_installed", return_value=True), mock.patch.object(
            _omnivoice_server, "synthesize", side_effect=_fake_synth
        ), mock.patch("core.omnivoice_tts.normalize_omnivoice_mode", return_value="design"), mock.patch(
            "core.omnivoice_tts.get_active_omnivoice_design_instruct",
            side_effect=lambda *, whisper=False: "female, whisper" if whisper else "female",
        ):
            audio, sr = synthesize_omnivoice_tts("Привіт [whisper] це секрет [whisper] бувай", lang="uk")
            self.assertEqual(sr, 24000)
            self.assertGreater(len(audio), 720)
            self.assertEqual(
                calls,
                [
                    ("Привіт", "female"),
                    ("це секрет", "female, whisper"),
                    ("бувай", "female"),
                ],
            )
            # У змішаній фразі шепітний сегмент має бути м'якшим за звичайний (компенсація server peak-norm)
            mid_idx = 240 + int(24000 * 0.045) + 10
            self.assertLess(float(audio[mid_idx]), float(audio[10]))

        # У Voice Design `get_active_omnivoice_design_instruct(whisper=True)` опускає `pitch`, щоб уникнути конфлікту F0
        from core.config import config
        from core.omnivoice_tts import get_active_omnivoice_design_instruct

        old_g = config.get("omnivoice_design_gender")
        old_a = config.get("omnivoice_design_age")
        old_p = config.get("omnivoice_design_pitch")
        old_acc = config.get("omnivoice_design_accent")
        try:
            config.set("omnivoice_design_gender", "male")
            config.set("omnivoice_design_age", "young adult")
            config.set("omnivoice_design_pitch", "moderate pitch")
            config.set("omnivoice_design_accent", "")
            self.assertEqual(get_active_omnivoice_design_instruct(whisper=False), "male, young adult, moderate pitch")
            self.assertEqual(get_active_omnivoice_design_instruct(whisper=True), "male, young adult, whisper")
        finally:
            config.set("omnivoice_design_gender", old_g)
            config.set("omnivoice_design_age", old_a)
            config.set("omnivoice_design_pitch", old_p)
            config.set("omnivoice_design_accent", old_acc)

    def test_fast_preview_resample_and_leading_silence_trim(self) -> None:
        import sys
        import numpy as np
        from core.audio_router import _resample_audio_fast, _trim_leading_silence

        # 1. Чистий NumPy ресемплер працює швидко й точно навіть без імпорту scipy.signal
        t = np.linspace(0.0, 0.25, 6000, endpoint=False, dtype=np.float32)
        src = np.stack([np.sin(2.0 * np.pi * 440.0 * t), np.sin(2.0 * np.pi * 440.0 * t)], axis=1).astype(np.float32)
        saved_scipy = sys.modules.pop("scipy.signal", None)
        try:
            out = _resample_audio_fast(src, 24000, 44100)
            self.assertEqual(out.shape[1], 2)
            self.assertAlmostEqual(len(out), round(6000 * 44100 / 24000), delta=2)
            self.assertFalse(np.isnan(out).any())
            self.assertGreater(float(np.max(np.abs(out))), 0.9)
        finally:
            if saved_scipy is not None:
                sys.modules["scipy.signal"] = saved_scipy

        # 2. Обрізання початкової тиші у прикладах зберігає ~15 мс запасу перед звуком
        sr = 24000
        silence = np.zeros(int(sr * 0.6), dtype=np.float32)
        tone = np.ones(int(sr * 0.2), dtype=np.float32) * 0.5
        trimmed = _trim_leading_silence(np.concatenate([silence, tone]), sr)
        nz = np.flatnonzero(np.abs(trimmed) > 0.008)
        self.assertGreater(len(nz), 0)
        lead_ms = nz[0] / sr * 1000.0
        self.assertLessEqual(lead_ms, 20.0)
        self.assertGreaterEqual(lead_ms, 10.0)

    def test_voice_samples_dialog_default_tab_by_install_state(self) -> None:
        from PySide6.QtWidgets import QApplication
        from ui.dialogs import VoiceSamplesDialog

        app = QApplication.instance() or QApplication([])
        self.assertIsNotNone(app)

        fake_clone = [{"name": "Voice A", "path": "a.ogg", "gender": "female"}]
        fake_demo = [{"name": "Demo A", "path": "d.ogg", "ref_path": "r.ogg", "gender": "female"}]
        fake_model = [{"name": "Willow", "path": "zonos3_en_willow.ogg", "gender": "female"}]

        with mock.patch("core.audio_router.audio_router.prewarm_monitor_device"), mock.patch(
            "ui.dialogs.get_clone_demo_samples", return_value=fake_demo
        ):
            # 1. OmniVoice НЕ встановлено -> «Приклад клонування»
            with mock.patch("ui.dialogs.get_samples_for_engine", return_value=fake_clone):
                d1 = VoiceSamplesDialog("omnivoice", "OmniVoice", allow_install=False)
                try:
                    self.assertEqual(d1._active_tab_mode, "clone_demo")
                    assert d1._mode_tabs is not None
                    self.assertEqual(d1._mode_tabs.tabData(d1._mode_tabs.currentIndex()), "clone_demo")
                finally:
                    d1.close()
                    d1.deleteLater()

                # 2. OmniVoice ВСТАНОВЛЕНО -> «Голоси для клонування»
                d2 = VoiceSamplesDialog("omnivoice", "OmniVoice", allow_install=True)
                try:
                    self.assertEqual(d2._active_tab_mode, "clone")
                    assert d2._mode_tabs is not None
                    self.assertEqual(d2._mode_tabs.tabData(d2._mode_tabs.currentIndex()), "clone")
                finally:
                    d2.close()
                    d2.deleteLater()

            # 3. ZONOS3-Mini НЕ встановлено (є голоси моделі) -> «Голоси моделі»
            def _zonos_samples(eid, lang=None, gender="both", own_only=False):
                return fake_model if own_only else fake_clone

            with mock.patch("ui.dialogs.get_samples_for_engine", side_effect=_zonos_samples):
                d3 = VoiceSamplesDialog("zonos3_mini", "ZONOS3-Mini", allow_install=False)
                try:
                    self.assertEqual(d3._active_tab_mode, "model")
                    assert d3._mode_tabs is not None
                    self.assertEqual(d3._mode_tabs.tabData(d3._mode_tabs.currentIndex()), "model")
                finally:
                    d3.close()
                    d3.deleteLater()

                # 4. ZONOS3-Mini ВСТАНОВЛЕНО -> завжди «Голоси для клонування»
                d4 = VoiceSamplesDialog("zonos3_mini", "ZONOS3-Mini", allow_install=True)
                try:
                    self.assertEqual(d4._active_tab_mode, "clone")
                    assert d4._mode_tabs is not None
                    self.assertEqual(d4._mode_tabs.tabData(d4._mode_tabs.currentIndex()), "clone")
                finally:
                    d4.close()
                    d4.deleteLater()

            # 5. ZONOS3-Mini НЕ встановлено (немає голосів моделі) -> «Приклад клонування»
            with mock.patch("ui.dialogs.get_samples_for_engine", side_effect=lambda eid, lang=None, gender="both", own_only=False: [] if own_only else fake_clone):
                d5 = VoiceSamplesDialog("zonos3_mini", "ZONOS3-Mini", allow_install=False)
                try:
                    self.assertEqual(d5._active_tab_mode, "clone_demo")
                    assert d5._mode_tabs is not None
                    self.assertEqual(d5._mode_tabs.tabData(d5._mode_tabs.currentIndex()), "clone_demo")
                finally:
                    d5.close()
                    d5.deleteLater()

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

    def test_keepalive_endheaders_registers_socket_and_idle_timeout_respects_active_engine(self) -> None:
        import core.omnivoice_tts as ov

        mgr = ov._OmniVoiceServerManager()
        mgr._req_local.active = True
        mgr._req_local.is_warmup = False
        fake_sock = object()

        class _DummyConn(ov._TrackableHTTPConnection):
            _manager = mgr

            def __init__(self) -> None:
                self.sock = fake_sock
                self.sent = False

            def endheaders(self, message_body=None, *, encode_chunked=False) -> None:
                # Replicate _TrackableHTTPConnection.endheaders without real socket I/O
                m = self._manager
                if m is not None and self.sock is not None:
                    m._register_thread_socket(self.sock)
                self.sent = True

        # Verify _TrackableHTTPConnection itself defines endheaders
        self.assertIn("endheaders", ov._TrackableHTTPConnection.__dict__)
        c = _DummyConn()
        c.endheaders()
        self.assertTrue(c.sent)
        self.assertEqual(len(mgr._active_sockets), 1)

        # Idle timeout must NOT stop resident server while OmniVoice is active engine
        with (
            mock.patch.object(ov.config, "get", side_effect=lambda k, d=None: "omnivoice" if k == "active_engine" else d),
            mock.patch.object(mgr, "stop") as mock_stop,
        ):
            mgr._on_idle_timeout()
            mock_stop.assert_not_called()

        # Idle timeout must NOT stop server while user request is active
        mgr._active_user_requests = 1
        with (
            mock.patch.object(ov.config, "get", side_effect=lambda k, d=None: "piper" if k == "active_engine" else d),
            mock.patch.object(mgr, "stop") as mock_stop,
            mock.patch.object(mgr, "_reset_idle_timer") as mock_reset,
        ):
            mgr._on_idle_timeout()
            mock_stop.assert_not_called()
            mock_reset.assert_called_once()

    def test_warmup_uses_single_step_and_guards_failures_and_llama_eviction(self) -> None:
        import core.omnivoice_tts as ov

        # 1) warmup_omnivoice_tts must pass steps=1 and return False on failure
        seen_steps: list[int] = []

        def fake_synth(**kwargs):
            seen_steps.append(int(kwargs.get("steps", -1)))
            return np.zeros(100, dtype=np.float32), 24000

        with (
            mock.patch.object(ov, "is_omnivoice_installed", return_value=True),
            mock.patch.object(
                ov,
                "_resolve_active_synth_params",
                return_value={
                    "lang_label": "Ukrainian",
                    "ref_wav": None,
                    "ref_digest": "",
                    "ref_text": "",
                    "instruct": "female, young adult",
                    "seed": 42,
                    "steps": 32,
                    "denoise": True,
                },
            ),
            mock.patch.object(ov._omnivoice_server, "synthesize", side_effect=fake_synth),
        ):
            self.assertTrue(ov.warmup_omnivoice_tts(lang="uk"))
        self.assertEqual(seen_steps, [1])

        with (
            mock.patch.object(ov, "is_omnivoice_installed", return_value=True),
            mock.patch.object(ov, "_resolve_active_synth_params", side_effect=RuntimeError("boom")),
        ):
            self.assertFalse(ov.warmup_omnivoice_tts(lang="uk"))

        # 2) _run_smart_warmup must NOT mark failed warmup as warmed, and must NOT evict primary llama server with fallback llama server
        old_warmed = set(tts_manager._warmed_models)
        old_running = tts_manager._running
        try:
            tts_manager._warmed_models.clear()
            tts_manager._running = True
            warmed_calls: list[str] = []
            with (
                mock.patch.object(
                    tts_manager,
                    "_resolve_local_warmup_targets",
                    return_value=[
                        ("qwen3_tts", "qwen3_en", "qwen3_tts:", "primary"),
                        ("zonos3_mini", "zonos3_en_savannah", "zonos3_mini:zonos3_en_savannah:", "fallback"),
                    ],
                ),
                mock.patch.object(tts_manager, "_resolve_online_warmup_targets", return_value=[]),
                mock.patch.object(tts_manager, "_get_system_resources_for_warmup", return_value=(16000, False, 10.0)),
                mock.patch("core.llama_tts.resolve_kokoro_device", return_value="gpu"),
                mock.patch("core.llama_tts.warmup_qwen3_tts", side_effect=lambda: (warmed_calls.append("qwen3_tts") or True)),
                mock.patch("core.llama_tts.warmup_llama_tts", side_effect=lambda eid, **k: (warmed_calls.append(eid) or True)),
                mock.patch("core.omnivoice_tts.is_omnivoice_server_running", return_value=False),
            ):
                tts_manager._run_smart_warmup("test", 0)
            self.assertEqual(warmed_calls, ["qwen3_tts"])
            self.assertIn(("qwen3_tts", "qwen3_tts:"), tts_manager._warmed_models)
            self.assertNotIn(("zonos3_mini", "zonos3_mini:zonos3_en_savannah:"), tts_manager._warmed_models)
        finally:
            tts_manager._warmed_models = old_warmed
            tts_manager._running = old_running

    def test_clone_mode_whisper_switches_to_gender_matched_voice_design_and_1d_resample(self) -> None:
        import sys
        import numpy as np
        import soundfile as sf
        from core.audio_router import _resample_audio_fast
        from core.config import config
        from core import omnivoice_tts as ov

        # 1. Регресія 1D моно-ресемплера без scipy.signal (не повинен падати з MemoryError)
        saved_scipy = sys.modules.pop("scipy.signal", None)
        try:
            mono_src = np.sin(2.0 * np.pi * 220.0 * np.linspace(0.0, 0.2, 4800, endpoint=False)).astype(np.float32)
            mono_out = _resample_audio_fast(mono_src, 24000, 48000)
            self.assertEqual(mono_out.ndim, 1)
            self.assertEqual(len(mono_out), 9600)
        finally:
            if saved_scipy is not None:
                sys.modules["scipy.signal"] = saved_scipy

        # 2. Визначення статі референсу (за F0 та іменем) + шепіт у clone-режимі без ref_wav
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            t = np.linspace(0.0, 1.0, 24000, endpoint=False, dtype=np.float32)
            male_wav = td_p / "speaker_a.wav"
            fem_wav = td_p / "speaker_b.wav"
            sf.write(str(male_wav), (0.3 * np.sin(2.0 * np.pi * 115.0 * t)).astype(np.float32), 24000)
            sf.write(str(fem_wav), (0.3 * np.sin(2.0 * np.pi * 225.0 * t)).astype(np.float32), 24000)

            ov._REF_GENDER_CACHE.clear()
            self.assertEqual(ov._detect_ref_gender(male_wav, ref_digest="dig_male_01"), "male")
            self.assertEqual(ov._detect_ref_gender(fem_wav, ref_digest="dig_fem_01"), "female")
            self.assertEqual(ov._detect_ref_gender(male_wav, ref_digest="dig_hint_f", orig_path="Mita Ж.mp3"), "female")

            old_mode = config.get("omnivoice_mode")
            old_ref = config.get("omnivoice_ref_path")
            try:
                config.set("omnivoice_mode", "clone")
                config.set("omnivoice_ref_path", str(male_wav))
                with mock.patch(
                    "core.omnivoice_tts.resolve_omnivoice_ref_text",
                    return_value=(male_wav, "abcdef1234567890", "Привет."),
                ):
                    p_norm = ov._resolve_active_synth_params("ru", whisper=False)
                    p_whisp = ov._resolve_active_synth_params("ru", whisper=True)

                self.assertEqual(p_norm["ref_wav"], male_wav)
                self.assertEqual(p_norm["instruct"], "")
                # Для шепоту ref_wav прибирається (бо omnivoice.cpp ігнорує whisper при ref_audio)
                self.assertIsNone(p_whisp["ref_wav"])
                self.assertEqual(p_whisp["instruct"], "male, young adult, whisper")
                self.assertIsInstance(p_whisp["seed"], int)
            finally:
                config.set("omnivoice_mode", old_mode)
                config.set("omnivoice_ref_path", old_ref)

        # 3. Власний файл користувача, скопійований у data/samples/clone_voices/ru/ і записаний у custom_clone_samples,
        # НЕ повинен отримувати фальшивий вбудований текст-підказку (інакше Whisper не запуститься і клон зламається)
        with tempfile.TemporaryDirectory() as td:
            custom_in_samples = Path(td) / "samples" / "clone_voices" / "ru" / "my_custom_voice.wav"
            custom_in_samples.parent.mkdir(parents=True, exist_ok=True)
            custom_in_samples.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

            old_custom = config.get("custom_clone_samples")
            try:
                config.set("custom_clone_samples", {})
                self.assertEqual(ov.detect_bundled_clone_sample_lang(custom_in_samples), "ru")
                self.assertTrue(bool(ov.get_default_omnivoice_ref_text(custom_in_samples)))

                config.set(
                    "custom_clone_samples",
                    {"clone_voices": [{"path": str(custom_in_samples), "lang": "ru"}]},
                )
                self.assertIsNone(ov.detect_bundled_clone_sample_lang(custom_in_samples))
                self.assertEqual(ov.get_default_omnivoice_ref_text(custom_in_samples), "")
            finally:
                config.set("custom_clone_samples", old_custom)

    def test_quality_assessment_recovery_and_safe_cancel(self) -> None:
        """Collapsed whisper/noise outputs fail quality check, retry with alternate lang/seed, and cancel closes sockets only."""
        from core import omnivoice_tts as ov

        sr = 24000
        # 1. Whisper ASR rejects non-speech/garbage markers and single-burst crackle while accepting real speech
        n = sr  # 1.0s
        burst = np.zeros(n, dtype=np.float32)
        burst[:800] = 0.5
        ok_burst, _, reason_burst = ov._assess_omnivoice_segment_quality(
            burst, sr, "меня зовут дима", lang="ru", is_whisper=True, ref_wav=None, verify_asr=False
        )
        self.assertFalse(ok_burst)
        self.assertEqual(reason_burst, "single_burst")

        # Broadband fricative noise waveform simulates whisper zcr >= 0.15
        rng = np.random.default_rng(7)
        whisp_like = (rng.uniform(-0.2, 0.2, n) * np.sign(np.sin(np.arange(n) * 1.1))).astype(np.float32)
        with (
            mock.patch("core.omnivoice_tts.is_omnivoice_whisper_installed", return_value=True),
            mock.patch("core.omnivoice_tts.transcribe_wav_with_whisper_cpp", side_effect=["*Слышен*", "[музыка]", "Меня зовут Тима."]),
        ):
            ok_ns, _, r1 = ov._assess_omnivoice_segment_quality(whisp_like, sr, "меня зовут дима", lang="ru", is_whisper=True, ref_wav=None)
            self.assertFalse(ok_ns)
            self.assertIn("asr_nonspeech", r1)

            ok_mu, _, r2 = ov._assess_omnivoice_segment_quality(whisp_like, sr, "[confirmation-en] Привет", lang="ru", is_whisper=False, ref_wav=None)
            self.assertFalse(ok_mu)

            ok_good, score_good, _ = ov._assess_omnivoice_segment_quality(whisp_like, sr, "меня зовут дима", lang="ru", is_whisper=True, ref_wav=None)
            self.assertTrue(ok_good)
            self.assertGreaterEqual(score_good, 100.0)

        # 2. Recovery retries with lang_label="None" when initial Russian whisper segment fails
        calls = []

        def _fake_retry_synth(*, text, lang_label, seed, steps, **_kw):
            calls.append((lang_label, seed, steps))
            if len(calls) == 1:
                return burst, sr
            return whisp_like, sr

        params = {
            "lang_label": "Russian",
            "ref_wav": None,
            "ref_digest": "",
            "ref_text": "",
            "instruct": "female, young adult, whisper",
            "seed": 42,
            "steps": 16,
            "denoise": True,
        }
        with (
            mock.patch.object(ov._omnivoice_server, "is_running", return_value=True),
            mock.patch.object(ov._omnivoice_server, "synthesize", side_effect=_fake_retry_synth),
            mock.patch("core.omnivoice_tts.is_omnivoice_whisper_installed", return_value=True),
            mock.patch("core.omnivoice_tts.transcribe_wav_with_whisper_cpp", return_value="Меня зовут Дима."),
        ):
            out_s, out_sr = ov._synthesize_segment_with_recovery("меня зовут дима", lang="ru", params=params, is_whisper=True)
        self.assertEqual(out_sr, sr)
        self.assertEqual(len(out_s), n)
        self.assertEqual(calls[0][0], "Russian")
        self.assertEqual(calls[1][0], "None")

        # 3. cancel_active closes sockets via _force_close_socket and NEVER invokes resp.close() on foreign thread
        mgr = ov._OmniVoiceServerManager()
        sock_a = mock.MagicMock()
        sock_resp = mock.MagicMock()
        resp_mock = mock.MagicMock()
        resp_mock.raw._fp.fp.raw._sock = sock_resp
        with mgr._req_lock:
            mgr._active_sockets[1] = (sock_a, False)
            mgr._active_responses[1] = (resp_mock, False)
        mgr.cancel_active(only_warmup=False)
        sock_a.shutdown.assert_called_once()
        sock_a.close.assert_called_once()
        sock_resp.shutdown.assert_called_once()
        sock_resp.close.assert_called_once()
        resp_mock.close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
