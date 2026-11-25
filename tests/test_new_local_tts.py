import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

import core.kokoro_onnx_tts as kokoro_onnx_tts
import core.llama_tts as llama_tts
from core.tts_manager import ENGINE_SPECS, tts_manager


class NewLocalTtsModelsTests(unittest.TestCase):
    def test_visible_local_engines_include_new_models_for_supported_languages(self) -> None:
        expected_by_lang = {
            "en": {"piper", "silero", "kokoro_82m", "soprano_1_1", "zonos3_mini", "qwen3_tts"},
            "uk": {"piper", "silero", "holos", "zonos3_mini"},
            "ru": {"piper", "silero", "teratts", "zonos3_mini", "qwen3_tts"},
            "de": {"piper", "silero", "zonos3_mini", "qwen3_tts"},
            "es": {"piper", "silero", "zonos3_mini", "qwen3_tts"},
            "fr": {"piper", "silero", "zonos3_mini", "qwen3_tts"},
            "pl": {"piper", "zonos3_mini"},
        }
        for lang, expected in expected_by_lang.items():
            with patch.object(tts_manager, "_current_lang", return_value=lang):
                ids = {item["id"] for item in tts_manager.get_visible_local_engines()}
            self.assertEqual(ids, expected, f"Language {lang}: got {ids}, expected {expected}")
            self.assertNotIn("pocket_tts", ids)
            self.assertNotIn("supertonic3", ids)

    def test_llama_engine_install_checks_and_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rt = root / "llama_tts"
            rt.mkdir()
            z_dir = root / "zonos3_mini"
            k_dir = root / "kokoro_82m"
            s_dir = root / "soprano_1_1"
            q_dir = root / "qwen3_tts"
            for d in (z_dir, k_dir, s_dir, q_dir):
                d.mkdir()

            with (
                patch.object(llama_tts, "LLAMA_TTS_EXE", rt / "llama-tts.exe"),
                patch.object(llama_tts, "LLAMA_TTS_SERVER_EXE", rt / "llama-server.exe"),
                patch.object(llama_tts, "LLAMA_TTS_READY_MARKER", rt / "ready.marker"),
                patch.object(llama_tts, "ZONOS3_DIR", z_dir),
                patch.object(llama_tts, "KOKORO_DIR", k_dir),
                patch.object(llama_tts, "SOPRANO_DIR", s_dir),
                patch.object(llama_tts, "QWEN3_TTS_DIR", q_dir),
            ):
                (rt / "llama-tts.exe").write_bytes(b"exe")
                (rt / "llama-server.exe").write_bytes(b"srv")
                (rt / "ready.marker").write_text("vulkan-ok", encoding="utf-8")

                self.assertFalse(llama_tts.is_any_llama_tts_installed())

                (z_dir / llama_tts.ZONOS3_MODEL_FILE).write_bytes(b"z")
                (z_dir / llama_tts.ZONOS3_MMPROJ_FILE).write_bytes(b"zm")
                self.assertTrue(llama_tts.is_zonos3_installed())
                self.assertTrue(llama_tts.is_any_llama_tts_installed())

                (k_dir / llama_tts.KOKORO_MODEL_FILE).write_bytes(b"k")
                (k_dir / llama_tts.KOKORO_MMPROJ_FILE).write_bytes(b"km")
                self.assertTrue(llama_tts.is_kokoro_installed())

                self.assertFalse(llama_tts.is_soprano_installed())
                (s_dir / llama_tts.SOPRANO_MODEL_FILE).write_bytes(b"s")
                (s_dir / llama_tts.SOPRANO_MMPROJ_FILE).write_bytes(b"sm")
                self.assertTrue(llama_tts.is_soprano_installed())

                dl_z = llama_tts.get_llama_engine_downloads("zonos3_mini")
                dl_k = llama_tts.get_llama_engine_downloads("kokoro_82m")
                dl_s = llama_tts.get_llama_engine_downloads("soprano_1_1")
                self.assertEqual(len(dl_z), 2)
                self.assertEqual(len(dl_k), 2)
                self.assertEqual(len(dl_s), 2)
                self.assertIn("Soprano-1.1-80M-Q8_0.gguf", dl_s[0][0])

    def test_synthesize_llama_tts_for_kokoro_zonos3_and_soprano(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            pcm_44k = np.linspace(-0.2, 0.2, 4410, dtype=np.float32)
            fake_resp_44k = MagicMock()
            fake_resp_44k.status_code = 200
            fake_resp_44k.headers = {"Content-Type": "audio/pcm;rate=44100;encoding=float;bits=32"}
            fake_resp_44k.content = pcm_44k.astype("<f4").tobytes()

            pcm_24k = np.linspace(-0.2, 0.2, 2400, dtype=np.float32)
            fake_resp_24k = MagicMock()
            fake_resp_24k.status_code = 200
            fake_resp_24k.headers = {"Content-Type": "audio/pcm;rate=24000;encoding=float;bits=32"}
            fake_resp_24k.content = pcm_24k.astype("<f4").tobytes()

            pcm_32k = np.linspace(-0.2, 0.2, 3200, dtype=np.float32)
            fake_resp_32k = MagicMock()
            fake_resp_32k.status_code = 200
            fake_resp_32k.headers = {"Content-Type": "audio/pcm;rate=32000;encoding=float;bits=32"}
            fake_resp_32k.content = pcm_32k.astype("<f4").tobytes()

            # 1) ZONOS3 built-in voice (no ref) + native speed
            cfg_no_ref = {"zonos3_mini_ref_path": ""}
            with (
                patch.object(llama_tts, "_REF_WAV_CACHE_DIR", root / "ref_cache"),
                patch.object(llama_tts, "has_hardware_gpu", return_value=True),
                patch.object(llama_tts._qwen3_server, "ensure_started", return_value=54330),
                patch.object(llama_tts.config, "get", side_effect=lambda k, d=None: cfg_no_ref.get(k, d)),
                patch.object(llama_tts._qwen3_server._session, "post", return_value=fake_resp_44k) as mock_post,
            ):
                audio, sr = llama_tts.synthesize_llama_tts(
                    "zonos3_mini", "Hello world", lang="en", voice="Savannah_en-American_F_1", speed=1.15
                )
            self.assertEqual(sr, 44100)
            self.assertEqual(len(audio), len(pcm_44k))
            self.assertEqual(mock_post.call_args.kwargs["json"]["voice"], "Savannah_en-American_F_1")
            self.assertEqual(mock_post.call_args.kwargs["json"]["lang"], "en")
            self.assertAlmostEqual(mock_post.call_args.kwargs["json"]["speed"], 1.15, places=2)

            # 1b) ZONOS3 cloning: keeps 24kHz ref WAV and preserves lang="uk" (no EN number normalization)
            ref_44k = root / "ref_44k.wav"
            sf.write(str(ref_44k), np.zeros(44100, dtype=np.float32), 44100)
            cfg_with_ref = {"zonos3_mini_ref_path": str(ref_44k)}
            with (
                patch.object(llama_tts, "_REF_WAV_CACHE_DIR", root / "ref_cache"),
                patch.object(llama_tts, "has_hardware_gpu", return_value=True),
                patch.object(llama_tts._qwen3_server, "ensure_started", return_value=54330),
                patch.object(llama_tts.config, "get", side_effect=lambda k, d=None: cfg_with_ref.get(k, d)),
                patch.object(llama_tts._qwen3_server._session, "post", return_value=fake_resp_44k) as mock_post,
            ):
                audio_u, sr_u = llama_tts.synthesize_llama_tts("zonos3_mini", "У 2026 році", lang="uk")
            self.assertEqual(sr_u, 44100)
            self.assertEqual(len(audio_u), len(pcm_44k))
            self.assertEqual(mock_post.call_args.kwargs["data"]["lang"], "uk")
            self.assertNotIn("speed", mock_post.call_args.kwargs["data"])
            sent_ref_bytes = mock_post.call_args.kwargs["files"]["speaker_ref"][1]
            import io
            _, sent_sr = sf.read(io.BytesIO(sent_ref_bytes), dtype="float32")
            self.assertEqual(sent_sr, 24000)

            # 2) Kokoro British voice -> en-gb, no n_predict, GPU-only, native speed
            with (
                patch.object(llama_tts, "has_hardware_gpu", return_value=False),
                patch.object(llama_tts._qwen3_server, "ensure_started", return_value=54331) as mock_start_k,
                patch.object(llama_tts._qwen3_server._session, "post", return_value=fake_resp_24k) as mock_post,
            ):
                audio_k, sr_k = llama_tts.synthesize_llama_tts(
                    "kokoro_82m", "Hello there", lang="en", voice="bf_emma", speed=1.2
                )
            self.assertEqual(sr_k, 24000)
            self.assertEqual(len(audio_k), len(pcm_24k))
            self.assertEqual(mock_start_k.call_args.args[0], "gpu")
            self.assertEqual(mock_post.call_args.kwargs["json"]["voice"], "bf_emma")
            self.assertEqual(mock_post.call_args.kwargs["json"]["lang"], "en-gb")
            self.assertAlmostEqual(mock_post.call_args.kwargs["json"]["speed"], 1.2, places=2)
            self.assertNotIn("n_predict", mock_post.call_args.kwargs["json"])

            # 3) Soprano-1.1-80M -> 32kHz, automatic GPU when hardware GPU present, CPU otherwise
            with (
                patch.object(llama_tts, "has_hardware_gpu", return_value=False),
                patch.object(llama_tts._qwen3_server, "ensure_started", return_value=54332) as mock_start_s_cpu,
                patch.object(llama_tts._qwen3_server._session, "post", return_value=fake_resp_32k) as mock_post,
            ):
                audio_s, sr_s = llama_tts.synthesize_llama_tts("soprano_1_1", "Hello world!", lang="en")
            self.assertEqual(sr_s, 32000)
            self.assertEqual(len(audio_s), len(pcm_32k))
            self.assertEqual(mock_start_s_cpu.call_args.args[0], "cpu")
            self.assertEqual(mock_post.call_args.kwargs["json"]["input"], "Hello world!")
            self.assertAlmostEqual(mock_post.call_args.kwargs["json"]["temperature"], 0.3, places=2)
            self.assertNotIn("lang", mock_post.call_args.kwargs["json"])
            self.assertNotIn("voice", mock_post.call_args.kwargs["json"])
            self.assertNotIn("n_predict", mock_post.call_args.kwargs["json"])

            with (
                patch.object(llama_tts, "has_hardware_gpu", return_value=True),
                patch.object(llama_tts._qwen3_server, "ensure_started", return_value=54333) as mock_start_s_gpu,
                patch.object(llama_tts._qwen3_server._session, "post", return_value=fake_resp_32k),
            ):
                llama_tts.synthesize_llama_tts("soprano_1_1", "Hello GPU!", lang="en")
            self.assertEqual(mock_start_s_gpu.call_args.args[0], "gpu")

            with patch.object(llama_tts, "has_hardware_gpu", return_value=False):
                self.assertEqual(llama_tts.resolve_soprano_device(), "cpu")

    def test_zonos3_stream_plan_resolves_speaker(self) -> None:
        with (
            patch.object(tts_manager, "is_engine_installed", side_effect=lambda eid: eid == "zonos3_mini"),
            patch.object(tts_manager, "_current_lang", return_value="en"),
        ):
            plan = tts_manager._resolve_qwen3_stream_plan(
                "Hello world", engine_override="zonos3_mini", voice_override="zonos3_en_imogen"
            )
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan["primary_id"], "zonos3_mini")
        self.assertEqual(plan["speaker"], "Imogen_en-British_F_1")

    def test_engine_specs_contain_new_gguf_models_only(self) -> None:
        for eid in ("kokoro_82m", "soprano_1_1", "zonos3_mini"):
            self.assertIn(eid, ENGINE_SPECS)
            self.assertGreater(len(ENGINE_SPECS[eid]["voices"]), 0)
        self.assertNotIn("pocket_tts", ENGINE_SPECS)
        self.assertNotIn("supertonic3", ENGINE_SPECS)

    def test_kokoro_gpu_vs_cpu_onnx_selection_and_synthesis(self) -> None:
        self.assertEqual(kokoro_onnx_tts.normalize_espeak_ipa_for_kokoro("tʃdʒeɪoʊ"), "ʧʤAO")
        chunks = kokoro_onnx_tts.split_kokoro_text("Sentence one. " * 30, max_chars=80)
        self.assertGreater(len(chunks), 2)
        long_no_punct = " ".join(["word"] * 120)
        soft_chunks = kokoro_onnx_tts.split_kokoro_text(long_no_punct, max_chars=80)
        self.assertGreater(len(soft_chunks), 3)
        self.assertTrue(all(len(c) <= 80 for c in soft_chunks))

        # Verify real piper EspeakPhonemizer accepts both US ("en-us") and British ("en") voices
        from piper.phonemize_espeak import ESPEAK_DATA_DIR, EspeakPhonemizer

        real_ph = EspeakPhonemizer(ESPEAK_DATA_DIR)
        self.assertTrue(real_ph.phonemize("en-us", "Hello world"))
        self.assertTrue(real_ph.phonemize("en", "Hello world"))

        with tempfile.TemporaryDirectory() as tmp:
            onnx_dir = Path(tmp) / "kokoro_onnx"
            onnx_dir.mkdir(parents=True)
            with patch.object(kokoro_onnx_tts, "KOKORO_ONNX_DIR", onnx_dir):
                self.assertFalse(kokoro_onnx_tts.is_kokoro_onnx_installed())

                (onnx_dir / kokoro_onnx_tts.KOKORO_ONNX_MODEL_FILE).write_bytes(b"onnx")
                (onnx_dir / kokoro_onnx_tts.KOKORO_ONNX_CONFIG_FILE).write_text(
                    json.dumps({"vocab": {"h": 1, "ə": 2, "l": 3, "O": 4, "!": 5}}),
                    encoding="utf-8",
                )
                np.savez(
                    str(onnx_dir / kokoro_onnx_tts.KOKORO_ONNX_VOICES_FILE),
                    af_heart=np.ones((16, 1, 256), dtype=np.float32),
                    bf_emma=np.ones((16, 1, 256), dtype=np.float32),
                )
                # np.savez appends .npz if suffix isn't .npz; rename to exact voices-v1.0.bin
                npz_written = onnx_dir / f"{kokoro_onnx_tts.KOKORO_ONNX_VOICES_FILE}.npz"
                if npz_written.is_file():
                    npz_written.replace(onnx_dir / kokoro_onnx_tts.KOKORO_ONNX_VOICES_FILE)
                self.assertTrue(kokoro_onnx_tts.is_kokoro_onnx_installed())

                # Automatic device resolution: GPU when hardware GPU present, CPU otherwise
                with patch.object(llama_tts, "has_hardware_gpu", return_value=False):
                    self.assertEqual(llama_tts.resolve_kokoro_device(), "cpu")
                    self.assertTrue(tts_manager.is_engine_installed("kokoro_82m"))
                with patch.object(llama_tts, "has_hardware_gpu", return_value=True):
                    self.assertEqual(llama_tts.resolve_kokoro_device(), "gpu")

                # Warmup target and audio cache key distinguish GPU vs CPU automatically
                with (
                    patch.object(llama_tts, "has_hardware_gpu", return_value=False),
                    patch.object(tts_manager, "_current_lang", return_value="en"),
                ):
                    cpu_target = tts_manager._resolve_local_target_for("kokoro_82m", "kokoro_af_heart")
                    cpu_cache_key = tts_manager._build_audio_cache_key("Hello", "kokoro_82m", "kokoro_af_heart")
                with patch.object(llama_tts, "has_hardware_gpu", return_value=True):
                    gpu_cache_key = tts_manager._build_audio_cache_key("Hello", "kokoro_82m", "kokoro_af_heart")
                self.assertIsNotNone(cpu_target)
                assert cpu_target is not None
                self.assertEqual(cpu_target[2], "kokoro_82m:cpu")
                self.assertNotEqual(cpu_cache_key, gpu_cache_key)

                # Mock ORT + EspeakPhonemizer synthesis
                fake_inp = MagicMock()
                fake_inp.name = "tokens"
                fake_sess = MagicMock()
                fake_sess.get_inputs.return_value = [fake_inp]
                fake_sess.run.return_value = [np.linspace(-0.2, 0.2, 2400, dtype=np.float32)]

                fake_ort = MagicMock()
                fake_ort.SessionOptions.return_value = MagicMock()
                fake_ort.InferenceSession.return_value = fake_sess

                fake_ph = MagicMock()
                fake_ph.phonemize.return_value = [["h", "ə", "l", "o", "ʊ", "!"]]
                fake_pe_mod = MagicMock()
                fake_pe_mod.ESPEAK_DATA_DIR = Path(tmp)
                fake_pe_mod.EspeakPhonemizer.return_value = fake_ph

                eng = kokoro_onnx_tts.KokoroOnnxEngine()
                with patch.dict("sys.modules", {"onnxruntime": fake_ort, "piper.phonemize_espeak": fake_pe_mod}):
                    wav, sr = eng.synthesize("Hello!", voice="af_heart", speed=1.15)
                    self.assertEqual(fake_ph.phonemize.call_args.args[0], "en-us")
                    # British voice must use espeak "en" (not invalid "en-gb")
                    wav_uk, sr_uk = eng.synthesize("Hello!", voice="bf_emma", speed=1.0)
                    self.assertEqual(fake_ph.phonemize.call_args.args[0], "en")
                    self.assertEqual(sr_uk, 24000)
                    self.assertEqual(len(wav_uk), 2400)
                    # Unspeakable text (0 phonemes) must raise RuntimeError cleanly
                    fake_ph.phonemize.return_value = [[]]
                    with self.assertRaises(RuntimeError):
                        eng.synthesize("...", voice="af_heart")
                self.assertEqual(sr, 24000)
                self.assertEqual(len(wav), 2400)
                feeds = fake_sess.run.call_args.args[1]
                self.assertIn("tokens", feeds)
                eng.unload()

    def test_all_internal_imports_resolve(self) -> None:
        """Статично перевіряє, що кожен `from core.* / ui.* import <sym>` реально існує."""
        import ast

        root = Path(__file__).resolve().parent.parent
        py_files = [root / "main.py"] + sorted((root / "core").glob("*.py")) + sorted((root / "ui").glob("*.py"))
        exports: dict[str, set[str]] = {}
        for f in py_files:
            mod_name = ".".join(f.relative_to(root).with_suffix("").parts)
            tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
            names: set[str] = set()
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(node.name)
                elif isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            names.add(t.id)
                        elif isinstance(t, (ast.Tuple, ast.List)):
                            for elt in t.elts:
                                if isinstance(elt, ast.Name):
                                    names.add(elt.id)
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    names.add(node.target.id)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    for a in node.names:
                        names.add(a.asname or a.name.split(".")[0])
                elif isinstance(node, (ast.If, ast.Try)):
                    for sub in ast.walk(node):
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            names.add(sub.name)
                        elif isinstance(sub, ast.Assign):
                            for t in sub.targets:
                                if isinstance(t, ast.Name):
                                    names.add(t.id)
                        elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                            names.add(sub.target.id)
            exports[mod_name] = names

        broken: list[str] = []
        for f in py_files:
            tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module in exports:
                    for a in node.names:
                        if a.name != "*" and a.name not in exports[node.module]:
                            broken.append(f"{f.name}:{node.lineno}: from {node.module} import {a.name}")
        self.assertEqual(broken, [], f"Broken internal imports: {broken}")

    def test_local_engine_cards_listen_clone_buttons_and_device_dropdown_rules(self) -> None:
        from PySide6.QtWidgets import QApplication
        from core.config import config
        from ui.dialogs import VoiceSamplesDialog
        from ui.main_window import MainWindow

        app = QApplication.instance() or QApplication([])
        orig_data = dict(config._data)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                samples_root = Path(tmp) / "samples"
                (samples_root / "clone_voices" / "en").mkdir(parents=True)
                (samples_root / "clone_voices" / "en" / "CloneVoice.mp3").write_bytes(b"c")
                (samples_root / "zonos3_mini" / "en").mkdir(parents=True)
                (samples_root / "zonos3_mini" / "en" / "zonos3_en_savannah.ogg").write_bytes(b"z")

                with (
                    patch("core.tts_manager.SAMPLES_DIR", samples_root),
                    patch("core.tts_manager.config.get", side_effect=lambda k, d=None: "en" if k == "app_language" else ("both" if k == "voice_gender_filter" else d)),
                    patch("ui.dialogs.config.get", side_effect=lambda k, d=None: "en" if k == "app_language" else ("both" if k == "voice_gender_filter" else d)),
                ):
                    for eid, title in (("openvoice", "OpenVoice v2"), ("qwen3_tts", "Qwen3-TTS 1.7B")):
                        dlg_clone = VoiceSamplesDialog(
                            eid,
                            title,
                            on_select_sample=lambda _p: None,
                            allow_install=False,
                        )
                        self.assertTrue(dlg_clone._is_clone_engine)
                        self.assertIsNone(dlg_clone._lang_combo)
                        self.assertIsNone(dlg_clone._mode_tabs)
                        self.assertIsNotNone(dlg_clone._guide_btn)
                        dlg_clone.close()
                        dlg_clone.deleteLater()

                    # ZONOS3-Mini with built-in model samples opens on tab 0 ("model") first, then switches to "clone"
                    dlg_z = VoiceSamplesDialog(
                        "zonos3_mini",
                        "ZONOS3-Mini",
                        on_select_sample=lambda _p: None,
                        allow_install=False,
                    )
                    self.assertIsNone(dlg_z._lang_combo)
                    self.assertIsNotNone(dlg_z._mode_tabs)
                    assert dlg_z._mode_tabs is not None
                    self.assertEqual(dlg_z._mode_tabs.count(), 2)
                    self.assertEqual(dlg_z._mode_tabs.currentIndex(), 0)
                    self.assertEqual(dlg_z._mode_tabs.tabData(0), "model")
                    self.assertEqual(dlg_z._mode_tabs.tabData(1), "clone")
                    self.assertEqual(dlg_z._active_tab_mode, "model")
                    self.assertEqual(len(dlg_z._install_buttons), 0)
                    title_model = dlg_z.windowTitle()
                    hdr_model = dlg_z._hdr_label.text()
                    dlg_z._mode_tabs.setCurrentIndex(1)
                    self.assertEqual(dlg_z._active_tab_mode, "clone")
                    self.assertNotEqual(dlg_z.windowTitle(), title_model)
                    self.assertNotEqual(dlg_z._hdr_label.text(), hdr_model)
                    dlg_z.close()
                    dlg_z.deleteLater()

            # 1) PC WITHOUT GPU: ZONOS3 & Qwen3 stay visible (disabled 'No GPU', hidden listen btn)
            with (
                patch.object(config, "save", lambda *a, **k: None),
                patch("ui.main_window.has_hardware_gpu", return_value=False),
                patch("ui.main_window.get_discrete_gpu_vram_gb", return_value=None),
                patch("core.llama_tts.has_hardware_gpu", return_value=False),
                patch.object(tts_manager, "is_engine_installed", side_effect=lambda eid: eid in ("kokoro_82m", "soprano_1_1")),
            ):
                config._data["app_language"] = "en"
                win = MainWindow()
                try:
                    for eid in ("zonos3_mini", "qwen3_tts"):
                        self.assertIn(eid, win._engine_cards_ui)
                        ui = win._engine_cards_ui[eid]
                        self.assertTrue(ui["listen_btn"].isHidden())
                        self.assertFalse(ui["action_btn"].isEnabled())
                        self.assertNotIn("recommended_badge", ui)
                    for eid in ("kokoro_82m", "soprano_1_1"):
                        ui = win._engine_cards_ui[eid]
                        self.assertFalse(ui["listen_btn"].isHidden())
                        self.assertNotIn("device_combo", ui)
                    self.assertNotIn("device_combo", win._engine_cards_ui["openvoice"])
                finally:
                    win.close()
                    win.deleteLater()

            # 2) PC WITH GPU < 3 GB VRAM: ZONOS3 & Qwen3 are hidden from local cards list
            with (
                patch.object(config, "save", lambda *a, **k: None),
                patch("ui.main_window.has_hardware_gpu", return_value=True),
                patch("ui.main_window.get_discrete_gpu_vram_gb", return_value=2.0),
                patch("core.llama_tts.has_hardware_gpu", return_value=True),
                patch.object(tts_manager, "is_engine_installed", return_value=False),
            ):
                config._data["app_language"] = "en"
                win = MainWindow()
                try:
                    self.assertNotIn("zonos3_mini", win._engine_cards_ui)
                    self.assertNotIn("qwen3_tts", win._engine_cards_ui)
                finally:
                    win.close()
                    win.deleteLater()

            # 3) PC WITH GPU >= 3 GB VRAM (8 GB & 16 GB): both shown with listen btn, no Recommended badges
            for vram in (8.0, 16.0):
                with (
                    patch.object(config, "save", lambda *a, **k: None),
                    patch("ui.main_window.has_hardware_gpu", return_value=True),
                    patch("ui.main_window.get_discrete_gpu_vram_gb", return_value=vram),
                    patch("core.llama_tts.has_hardware_gpu", return_value=True),
                    patch.object(tts_manager, "is_engine_installed", return_value=False),
                ):
                    config._data["app_language"] = "en"
                    win = MainWindow()
                    try:
                        for eid in ("zonos3_mini", "qwen3_tts", "kokoro_82m", "soprano_1_1", "openvoice"):
                            ui = win._engine_cards_ui[eid]
                            self.assertFalse(ui["listen_btn"].isHidden())
                            self.assertNotIn("recommended_badge", ui)
                            self.assertNotIn("device_combo", ui)
                    finally:
                        win.close()
                        win.deleteLater()
        finally:
            config._data.clear()
            config._data.update(orig_data)
            tts_manager.shutdown()
        _ = app

    def test_blaite_easter_egg_dialog_and_credit_link(self) -> None:
        import threading
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
        from ui.easter_egg import (
            BlaiteEasterEggDialog,
            HEIGHT,
            WIDTH,
            _EasterEggAudioPlayer,
            get_easter_egg_audio_path,
        )
        from ui.main_window import MainWindow

        app = QApplication.instance() or QApplication([])
        assert app is not None

        # 0) Easter egg audio asset exists and streams chunks cleanly with mocked OutputStream
        audio_path = get_easter_egg_audio_path()
        self.assertIsNotNone(audio_path)
        assert audio_path is not None
        self.assertTrue(audio_path.is_file())

        wrote_ev = threading.Event()
        aborted = {"called": False}

        class _FakeStream:
            def start(self) -> None:
                pass

            def write(self, data) -> None:
                if data is not None and len(data) > 0:
                    wrote_ev.set()

            def abort(self) -> None:
                aborted["called"] = True

            def stop(self) -> None:
                pass

            def close(self) -> None:
                pass

        player = _EasterEggAudioPlayer(audio_path)
        with (
            patch("core.audio_router.resolve_secondary_output_targets", return_value=[8]),
            patch("core.audio_router.resolve_default_output_device_index", return_value=8),
            patch("core.audio_router.resolve_non_cable_speaker_indices", return_value=[8, 10]),
        ):
            self.assertEqual(player._pick_local_output_devices(), [8, 10])

        with patch.object(player, "_open_stream", return_value=_FakeStream()):
            self.assertTrue(player.start())
            self.assertTrue(wrote_ev.wait(timeout=2.0))
            player.stop()
            self.assertTrue(aborted["called"])

        # 1) Dialog renders all phases (eye, zoom+stars, 3D heart) cleanly and stops timer on close
        dlg = BlaiteEasterEggDialog(enable_audio=False)
        try:
            self.assertEqual(dlg.width(), WIDTH)
            self.assertEqual(dlg.height(), HEIGHT)
            self.assertTrue(dlg._timer.isActive())
            for sample_t in (0.0, 5.0, 18.0, 32.0, 50.5):
                dlg._canvas.render_frame(sample_t)
            self.assertEqual(dlg._canvas._frame_buf.shape, (HEIGHT, WIDTH, 4))
            self.assertTrue(bool((dlg._canvas._frame_buf[..., 3] == 255).all()))
        finally:
            dlg.reject()
            self.assertFalse(dlg._timer.isActive())
            dlg.deleteLater()

        # 2) MainWindow creator credit label has clickable Blaite link & pointing-hand cursor on hover
        win = MainWindow()
        try:
            lbl = win._creator_credit_lbl
            self.assertIn('href="blaite_easter_egg"', lbl.text())
            self.assertIn("Blaite", lbl.text())
            lbl.linkHovered.emit("blaite_easter_egg")
            self.assertEqual(lbl.cursor().shape(), Qt.CursorShape.PointingHandCursor)
            lbl.linkHovered.emit("")
            self.assertEqual(lbl.cursor().shape(), Qt.CursorShape.ArrowCursor)

            with patch.object(win, "_open_blaite_easter_egg") as mock_open:
                lbl.linkActivated.emit("blaite_easter_egg")
                mock_open.assert_called_once()
        finally:
            win.close()
            win.deleteLater()


if __name__ == "__main__":
    unittest.main()
