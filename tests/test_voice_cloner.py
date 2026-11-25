import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from core.config import config
from core.tts_manager import ENGINE_SPECS, get_samples_for_engine, tts_manager
from core.win_api import is_real_gpu_adapter
from core.voice_cloner import (
    CONVERTER_FILE,
    OpenVoiceCloner,
    REF_ENCODER_FILE,
    compute_linear_spec,
    extract_voiced_speech,
    to_22k_mono,
    trim_audio_edges,
)


class VoiceClonerTests(unittest.TestCase):
    def test_audio_preprocessing_shapes_and_edge_trimming(self) -> None:
        sr = 48000
        # 0.5s silence + 0.5s tone + 0.5s silence (stereo)
        sil = np.zeros(sr // 2, dtype=np.float32)
        t = np.linspace(0, 0.5, sr // 2, endpoint=False, dtype=np.float32)
        tone = 0.25 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
        mono = np.concatenate([sil, tone, sil])
        stereo = np.stack([mono, mono], axis=1)

        trimmed = trim_audio_edges(mono, sr)
        self.assertLess(len(trimmed), len(mono))
        self.assertGreater(len(trimmed), len(tone))

        voiced = extract_voiced_speech(to_22k_mono(mono, sr), 22050, target_peak=0.75)
        self.assertLess(len(voiced), len(to_22k_mono(mono, sr)))
        self.assertAlmostEqual(float(np.max(np.abs(voiced))), 0.75, delta=0.05)

        y22 = to_22k_mono(stereo, sr)
        self.assertEqual(y22.ndim, 1)
        self.assertAlmostEqual(len(y22) / 22050.0, 1.5, delta=0.05)

        spec = compute_linear_spec(y22)
        self.assertEqual(spec.ndim, 2)
        self.assertEqual(spec.shape[1], 513)
        self.assertEqual(spec.dtype, np.float32)
        self.assertTrue(np.isfinite(spec).all())

    def test_installed_flags_and_cache_key_sensitivity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(OpenVoiceCloner.models_present(root))
            self.assertFalse(OpenVoiceCloner.gpu_runtime_present(root))
            self.assertFalse(OpenVoiceCloner.is_installed(root))

            (root / REF_ENCODER_FILE).write_bytes(b"enc")
            (root / CONVERTER_FILE).write_bytes(b"conv")
            self.assertTrue(OpenVoiceCloner.models_present(root))
            self.assertFalse(OpenVoiceCloner.is_installed(root))

            rt = root / "gpu_runtime"
            (rt / "python_embed").mkdir(parents=True)
            (rt / "python_embed" / "python.exe").write_bytes(b"py")
            (rt / "gpu_worker.py").write_bytes(b"w")
            (rt / "gpu_ready.marker").write_bytes(b"ok")
            self.assertTrue(OpenVoiceCloner.gpu_runtime_present(root))
            self.assertTrue(OpenVoiceCloner.is_installed(root))

            ref1 = root / "v1.wav"
            ref2 = root / "v2.wav"
            sf.write(str(ref1), np.zeros(22050, dtype=np.float32), 22050)
            sf.write(str(ref2), np.ones(22050, dtype=np.float32) * 0.1, 22050)

            with patch("core.tts_manager.config.get") as mock_get:
                state = {"openvoice_enabled": False, "openvoice_ref_path": "", "openvoice_device": "gpu", "app_language": "uk"}
                mock_get.side_effect = lambda k, d=None: state.get(k, d)

                with (
                    patch("core.voice_cloner.OpenVoiceCloner.models_present", return_value=True),
                    patch("core.voice_cloner.OpenVoiceCloner._use_gpu", return_value=True),
                ):
                    k_off = tts_manager._build_audio_cache_key("привіт", "piper", "lada")
                    state["openvoice_enabled"] = True
                    state["openvoice_ref_path"] = str(ref1)
                    k_ref1 = tts_manager._build_audio_cache_key("привіт", "piper", "lada")
                    state["openvoice_ref_path"] = str(ref2)
                    k_ref2 = tts_manager._build_audio_cache_key("привіт", "piper", "lada")
                with (
                    patch("core.voice_cloner.OpenVoiceCloner.models_present", return_value=True),
                    patch("core.voice_cloner.OpenVoiceCloner._use_gpu", return_value=False),
                ):
                    k_cpu = tts_manager._build_audio_cache_key("привіт", "piper", "lada")

                    self.assertNotEqual(k_off, k_ref1)
                    self.assertNotEqual(k_ref1, k_ref2)
                    self.assertNotEqual(k_ref2, k_cpu)

    def test_openvoice_engine_spec_registered_without_standalone_voices(self) -> None:
        self.assertIn("openvoice", ENGINE_SPECS)
        self.assertEqual(ENGINE_SPECS["openvoice"]["voices"], [])

    def test_upgrade_audio_uses_cloner_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "voice.wav"
            sf.write(str(ref), np.zeros(22050, dtype=np.float32), 22050)
            src = np.ones(4800, dtype=np.float32) * 0.2
            fake_out = (np.ones(2205, dtype=np.float32) * 0.3, 22050)

            with patch.object(config, "get", side_effect=lambda k, d=None: True if k == "openvoice_enabled" else (str(ref) if k == "openvoice_ref_path" else d)):
                with patch("core.voice_cloner.OpenVoiceCloner.models_present", return_value=True):
                    with patch("core.voice_cloner.voice_cloner.upgrade_audio", return_value=fake_out) as mock_upg:
                        out, out_sr = tts_manager._maybe_upgrade_openvoice(src, 48000)
                        mock_upg.assert_called_once()
                        self.assertEqual(out_sr, 22050)
                        self.assertEqual(len(out), 2205)

    def test_gpu_adapter_detection_and_openvoice_order_and_samples(self) -> None:
        self.assertFalse(is_real_gpu_adapter("Microsoft Basic Display Adapter", "ROOT\\DISPLAY\\0000"))
        self.assertFalse(is_real_gpu_adapter("Microsoft Basic Render Driver", "PCI\\VEN_1414&DEV_008C", flags=2))
        self.assertFalse(is_real_gpu_adapter("Virtual Display Device", ""))
        self.assertFalse(is_real_gpu_adapter("AMD Radeon(TM) Graphics", "PCI\\VEN_1002&DEV_164C"))
        self.assertFalse(is_real_gpu_adapter("Intel(R) UHD Graphics 770", "PCI\\VEN_8086&DEV_4680"))
        self.assertFalse(is_real_gpu_adapter("Intel(R) Iris(R) Xe Graphics", "PCI\\VEN_8086&DEV_9A49"))
        self.assertFalse(is_real_gpu_adapter("Intel(R) Arc(TM) Graphics", "PCI\\VEN_8086&DEV_7D55"))
        self.assertTrue(is_real_gpu_adapter("NVIDIA GeForce RTX 3060", "PCI\\VEN_10DE&DEV_2504"))
        self.assertTrue(is_real_gpu_adapter("AMD Radeon RX 6600", "PCI\\VEN_1002&DEV_73FF"))
        self.assertTrue(is_real_gpu_adapter("Intel(R) Arc(TM) A750 Graphics", "PCI\\VEN_8086&DEV_56A1"))

        keys = list(ENGINE_SPECS.keys())
        self.assertLess(keys.index("openvoice"), keys.index("qwen3_tts"))

        with tempfile.TemporaryDirectory() as tmp:
            samples_root = Path(tmp) / "samples"
            ov_dir = samples_root / "openvoice"
            (ov_dir / "uk").mkdir(parents=True)
            (ov_dir / "en").mkdir(parents=True)
            (ov_dir / "en" / "alex.wav").write_bytes(b"w")
            (ov_dir / "uk" / "olena.mp3").write_bytes(b"m")
            (ov_dir / "ru_ivan.ogg").write_bytes(b"r")
            (ov_dir / "custom_voice.ogg").write_bytes(b"o")

            with patch("core.tts_manager.SAMPLES_DIR", samples_root):
                all_items = get_samples_for_engine("openvoice", lang="all")
                langs_by_file = {item["file_name"]: item["lang"] for item in all_items}
                self.assertEqual(
                    langs_by_file,
                    {
                        "alex.wav": "en",
                        "olena.mp3": "uk",
                        "ru_ivan.ogg": "ru",
                        "custom_voice.ogg": "",
                    },
                )
                uk_items = {item["file_name"] for item in get_samples_for_engine("openvoice", lang="uk")}
                self.assertEqual(uk_items, {"olena.mp3", "custom_voice.ogg"})
                en_items = {item["file_name"] for item in get_samples_for_engine("openvoice", lang="en")}
                self.assertEqual(en_items, {"alex.wav", "custom_voice.ogg"})

        from ui.dialogs import OPENVOICE_SAMPLE_TEXTS

        for lang in ("uk", "en", "ru", "pl", "de", "fr", "es"):
            self.assertIn(lang, OPENVOICE_SAMPLE_TEXTS)
            self.assertGreater(len(OPENVOICE_SAMPLE_TEXTS[lang]), 100)

    def test_clone_sample_gender_inference_filtering_and_ordering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            samples_root = Path(tmp) / "samples"
            for eng in ("openvoice", "qwen3_tts"):
                uk_dir = samples_root / eng / "uk"
                uk_dir.mkdir(parents=True)
                (uk_dir / "Dmitry Ч.mp3").write_bytes(b"m1")
                (uk_dir / "Alexandra Ж.mp3").write_bytes(b"f1")
                (uk_dir / "Andrey Ч.wav").write_bytes(b"m2")
                (uk_dir / "NeutralClip.ogg").write_bytes(b"u1")

                with patch("core.tts_manager.SAMPLES_DIR", samples_root):
                    both = get_samples_for_engine(eng, lang="uk", gender="both")
                    self.assertEqual(
                        [(s["file_name"], s["gender"]) for s in both],
                        [
                            ("Alexandra Ж.mp3", "female"),
                            ("Andrey Ч.wav", "male"),
                            ("Dmitry Ч.mp3", "male"),
                            ("NeutralClip.ogg", ""),
                        ],
                    )
                    fem = [s["file_name"] for s in get_samples_for_engine(eng, lang="uk", gender="female")]
                    self.assertEqual(fem, ["Alexandra Ж.mp3", "NeutralClip.ogg"])
                    mal = [s["file_name"] for s in get_samples_for_engine(eng, lang="uk", gender="male")]
                    self.assertEqual(mal, ["Andrey Ч.wav", "Dmitry Ч.mp3", "NeutralClip.ogg"])

    def test_voice_samples_dialog_stops_playback_on_close_and_filters(self) -> None:
        from PySide6.QtWidgets import QApplication
        from ui.dialogs import VoiceSamplesDialog

        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            samples_root = Path(tmp) / "samples"
            cb_uk = samples_root / "qwen3_tts" / "uk"
            cb_uk.mkdir(parents=True)
            (cb_uk / "Olena Ж.mp3").write_bytes(b"f")
            (cb_uk / "Taras Ч.mp3").write_bytes(b"m")

            chosen = []
            with (
                patch("core.tts_manager.SAMPLES_DIR", samples_root),
                patch("core.tts_manager.config.get", side_effect=lambda k, d=None: "uk" if k == "app_language" else ("both" if k == "voice_gender_filter" else d)),
                patch("ui.dialogs.tts_manager.stop_current") as mock_stop,
                patch("core.audio_router.audio_router.begin_exclusive_playback") as mock_excl,
            ):
                dlg = VoiceSamplesDialog(
                    "qwen3_tts",
                    "Qwen3-TTS 1.7B Base (GGUF)",
                    on_select_sample=chosen.append,
                    allow_install=True,
                )
                self.assertIsNotNone(dlg._gender_combo)
                self.assertIsNone(dlg._lang_combo)
                self.assertIsNotNone(dlg._guide_btn)
                self.assertEqual(len(dlg._install_buttons), 2)

                fem_idx = dlg._gender_combo.findData("female")
                dlg._gender_combo.setCurrentIndex(fem_idx)
                self.assertEqual(len(dlg._install_buttons), 1)

                dlg._choose_sample(str(cb_uk / "Olena Ж.mp3"))
                self.assertEqual(chosen, [str(cb_uk / "Olena Ж.mp3")])

                dlg.accept()
                self.assertGreaterEqual(mock_stop.call_count, 1)
                self.assertGreaterEqual(mock_excl.call_count, 1)

                mock_stop.reset_mock()
                dlg2 = VoiceSamplesDialog("qwen3_tts", "Qwen3-TTS 1.7B Base (GGUF)")
                dlg2.reject()
                mock_stop.assert_called_once()
        _ = app

    def test_default_window_width_proportional_to_screen(self) -> None:
        from ui.main_window import calculate_default_window_width

        self.assertEqual(calculate_default_window_width(1920), 1247)
        self.assertEqual(calculate_default_window_width(1475), 958)
        self.assertEqual(calculate_default_window_width(2560), 1663)
        self.assertEqual(calculate_default_window_width(1366), 887)
        self.assertEqual(calculate_default_window_width(600), 540)

    def test_custom_clone_sample_register_delete_and_empty_lang_message(self) -> None:
        from PySide6.QtWidgets import QApplication, QLabel, QPushButton
        from core.tts_manager import register_custom_clone_sample, remove_custom_clone_sample
        from ui.dialogs import VoiceSamplesDialog

        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples_root = root / "samples"
            (samples_root / "qwen3_tts" / "uk").mkdir(parents=True)
            builtin_path = samples_root / "qwen3_tts" / "uk" / "Builtin Ж.mp3"
            builtin_path.write_bytes(b"builtin")

            ext_dir = root / "external"
            ext_dir.mkdir(parents=True)
            ext_voice = ext_dir / "MyCustom Ч.mp3"
            ext_voice.write_bytes(b"custom-audio")

            cfg_state = {"app_language": "uk", "voice_gender_filter": "both", "custom_clone_samples": {}, "qwen3_tts_ref_path": ""}

            def fake_get(k, d=None):
                return cfg_state.get(k, d)

            def fake_set(k, v):
                cfg_state[k] = v

            with (
                patch("core.tts_manager.SAMPLES_DIR", samples_root),
                patch("core.tts_manager.config.get", side_effect=fake_get),
                patch("core.tts_manager.config.set", side_effect=fake_set),
                patch("ui.dialogs.config.get", side_effect=fake_get),
            ):
                # Selecting built-in sample does not copy or mark it as custom
                kept_builtin = register_custom_clone_sample("qwen3_tts", str(builtin_path), lang="uk")
                self.assertEqual(Path(kept_builtin).resolve(), builtin_path.resolve())
                self.assertEqual(cfg_state["custom_clone_samples"], {})

                # Adding external voice copies it under shared clone_voices/<lang>/ and shares it across clone engines
                copied_path = register_custom_clone_sample("qwen3_tts", str(ext_voice), lang="uk")
                self.assertTrue(Path(copied_path).is_file())
                self.assertEqual(Path(copied_path).parent, (samples_root / "clone_voices" / "uk").resolve())
                cfg_state["qwen3_tts_ref_path"] = copied_path

                uk_samples = get_samples_for_engine("qwen3_tts", lang="uk", gender="both")
                self.assertEqual(
                    [s["file_name"] for s in get_samples_for_engine("openvoice", lang="uk", gender="both")],
                    [s["file_name"] for s in uk_samples],
                )
                by_name = {s["file_name"]: s for s in uk_samples}
                self.assertEqual(by_name["Builtin Ж.mp3"]["is_custom"], "")
                self.assertEqual(by_name["MyCustom Ч.mp3"]["is_custom"], "1")

                deleted_calls = []
                # Empty language (pl) shows localized "for your language" message
                cfg_state["app_language"] = "pl"
                dlg_pl = VoiceSamplesDialog(
                    "qwen3_tts",
                    "Qwen3-TTS 1.7B",
                    on_select_sample=lambda p: None,
                    allow_install=True,
                )
                self.assertIsNone(dlg_pl._lang_combo)
                empty_labels = [w.text() for w in dlg_pl._container.findChildren(QLabel) if not w.isHidden()]
                self.assertTrue(any(t for t in empty_labels))
                dlg_pl.close()
                dlg_pl.deleteLater()

                # Back to uk -> only 1 delete button (for custom sample, not builtin)
                cfg_state["app_language"] = "uk"
                dlg = VoiceSamplesDialog(
                    "qwen3_tts",
                    "Qwen3-TTS 1.7B",
                    on_select_sample=lambda p: None,
                    allow_install=True,
                    selected_sample_path=copied_path,
                    on_delete_sample=deleted_calls.append,
                )
                del_btns = [b for b in dlg._container.findChildren(QPushButton) if not b.icon().isNull()]
                self.assertEqual(len(del_btns), 1)

                dlg._delete_custom_sample(copied_path)
                self.assertFalse(Path(copied_path).exists())
                self.assertTrue(ext_voice.exists())
                self.assertEqual(cfg_state["qwen3_tts_ref_path"], "")
                self.assertEqual(len(deleted_calls), 1)
                self.assertEqual(len(dlg._install_buttons), 1)

                # Safe idempotent delete
                remove_custom_clone_sample("qwen3_tts", copied_path)
                dlg.reject()
        _ = app

    def test_openvoice_warmup_falls_back_to_cpu_and_deduplicates(self) -> None:
        from unittest.mock import MagicMock

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / "ref.wav"
            t = np.linspace(0, 1.0, 22050, endpoint=False, dtype=np.float32)
            sf.write(str(ref), 0.3 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32), 22050)

            cloner = OpenVoiceCloner()
            fake_enc = MagicMock()
            fake_enc.get_inputs.return_value = [MagicMock(name="spec")]
            fake_enc.run.return_value = [np.zeros((1, 256), dtype=np.float32)]
            fake_conv = MagicMock()
            fake_conv.get_inputs.return_value = [MagicMock(name=f"i{k}") for k in range(4)]

            cfg_state = {
                "openvoice_enabled": True,
                "openvoice_ref_path": str(ref),
                "openvoice_device": "gpu",
            }
            with (
                patch("core.voice_cloner.config.get", side_effect=lambda k, d=None: cfg_state.get(k, d)),
                patch.object(cloner, "models_present", return_value=True),
                patch.object(cloner, "_use_gpu", return_value=True),
                patch.object(cloner._gpu_worker, "ensure_started", return_value=False) as mock_gpu_start,
                patch.object(cloner, "_ensure_cpu_sessions", return_value=(fake_enc, fake_conv)) as mock_cpu,
            ):
                cloner.warmup()
                mock_gpu_start.assert_called_once()
                count_after_first = mock_cpu.call_count
                self.assertGreaterEqual(count_after_first, 1)
                self.assertIsNotNone(cloner._ref_cache_emb)

                # Repeating warmup with unchanged backend & ref is a fast no-op
                cloner.warmup()
                mock_gpu_start.assert_called_once()
                self.assertEqual(mock_cpu.call_count, count_after_first)


if __name__ == "__main__":
    unittest.main()
