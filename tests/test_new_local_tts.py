import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

import core.llama_tts as llama_tts
from core.tts_manager import ENGINE_SPECS, tts_manager


class NewLocalTtsModelsTests(unittest.TestCase):
    def test_visible_local_engines_include_new_models_for_supported_languages(self) -> None:
        with patch.object(tts_manager, "_current_lang", return_value="en"):
            en_ids = [item["id"] for item in tts_manager.get_visible_local_engines()]
        self.assertIn("kokoro_82m", en_ids)
        self.assertIn("soprano_1_1", en_ids)
        self.assertIn("zonos3_mini", en_ids)
        self.assertIn("qwen3_tts", en_ids)
        self.assertNotIn("pocket_tts", en_ids)
        self.assertNotIn("supertonic3", en_ids)

        with patch.object(tts_manager, "_current_lang", return_value="uk"):
            uk_ids = [item["id"] for item in tts_manager.get_visible_local_engines()]
        self.assertIn("zonos3_mini", uk_ids)
        self.assertNotIn("kokoro_82m", uk_ids)
        self.assertNotIn("soprano_1_1", uk_ids)
        self.assertNotIn("pocket_tts", uk_ids)
        self.assertNotIn("supertonic3", uk_ids)

        with patch.object(tts_manager, "_current_lang", return_value="de"):
            de_ids = [item["id"] for item in tts_manager.get_visible_local_engines()]
        self.assertIn("zonos3_mini", de_ids)
        self.assertNotIn("kokoro_82m", de_ids)
        self.assertNotIn("soprano_1_1", de_ids)
        self.assertNotIn("pocket_tts", de_ids)
        self.assertNotIn("supertonic3", de_ids)

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

            # 2) Kokoro British voice -> en-gb, no n_predict, GPU-only
            with (
                patch.object(llama_tts, "has_hardware_gpu", return_value=False),
                patch.object(llama_tts._qwen3_server, "ensure_started", return_value=54331) as mock_start_k,
                patch.object(llama_tts._qwen3_server._session, "post", return_value=fake_resp_24k) as mock_post,
            ):
                audio_k, sr_k = llama_tts.synthesize_llama_tts(
                    "kokoro_82m", "Hello there", lang="en", voice="bf_emma"
                )
            self.assertEqual(sr_k, 24000)
            self.assertEqual(len(audio_k), len(pcm_24k))
            self.assertEqual(mock_start_k.call_args.args[0], "gpu")
            self.assertEqual(mock_post.call_args.kwargs["json"]["voice"], "bf_emma")
            self.assertEqual(mock_post.call_args.kwargs["json"]["lang"], "en-gb")
            self.assertNotIn("n_predict", mock_post.call_args.kwargs["json"])

            # 3) Soprano-1.1-80M -> 32kHz, respects user GPU/CPU choice (and falls back to CPU without GPU)
            cfg_s_cpu = {"soprano_1_1_device": "cpu"}
            with (
                patch.object(llama_tts, "has_hardware_gpu", return_value=True),
                patch.object(llama_tts.config, "get", side_effect=lambda k, d=None: cfg_s_cpu.get(k, d)),
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

            cfg_s_gpu = {"soprano_1_1_device": "gpu"}
            with (
                patch.object(llama_tts, "has_hardware_gpu", return_value=True),
                patch.object(llama_tts.config, "get", side_effect=lambda k, d=None: cfg_s_gpu.get(k, d)),
                patch.object(llama_tts._qwen3_server, "ensure_started", return_value=54333) as mock_start_s_gpu,
                patch.object(llama_tts._qwen3_server._session, "post", return_value=fake_resp_32k),
            ):
                llama_tts.synthesize_llama_tts("soprano_1_1", "Hello GPU!", lang="en")
            self.assertEqual(mock_start_s_gpu.call_args.args[0], "gpu")

            with (
                patch.object(llama_tts, "has_hardware_gpu", return_value=False),
                patch.object(llama_tts.config, "get", side_effect=lambda k, d=None: cfg_s_gpu.get(k, d)),
            ):
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


if __name__ == "__main__":
    unittest.main()
