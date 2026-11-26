import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

import core.llama_tts as llama_tts
from core.tts_manager import ENGINE_SPECS, tts_manager


def _make_wav_bytes(sr: int = 24000, dur_sec: float = 0.25) -> bytes:
    t = np.linspace(0.0, dur_sec, int(sr * dur_sec), endpoint=False, dtype=np.float32)
    tone = 0.2 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, tone, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class LlamaTtsTests(unittest.TestCase):
    def test_installed_checks_require_runtime_and_both_gguf_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rt = root / "llama_tts"
            rt.mkdir()
            qw = root / "qwen3_tts"
            qw.mkdir()

            with (
                patch.object(llama_tts, "LLAMA_TTS_EXE", rt / "llama-tts.exe"),
                patch.object(llama_tts, "LLAMA_TTS_SERVER_EXE", rt / "llama-server.exe"),
                patch.object(llama_tts, "LLAMA_TTS_READY_MARKER", rt / "ready.marker"),
                patch.object(llama_tts, "QWEN3_TTS_DIR", qw),
            ):
                self.assertFalse(llama_tts.is_qwen3_tts_installed())

                (rt / "llama-tts.exe").write_bytes(b"exe")
                (rt / "llama-server.exe").write_bytes(b"srv")
                (rt / "ready.marker").write_bytes(b"ok")
                self.assertFalse(llama_tts.is_qwen3_tts_installed())

                (qw / llama_tts.QWEN3_MODEL_FILE).write_bytes(b"m")
                self.assertFalse(llama_tts.is_qwen3_tts_installed())
                (qw / llama_tts.QWEN3_MMPROJ_FILE).write_bytes(b"mm")
                self.assertTrue(llama_tts.is_qwen3_tts_installed())

    def test_prepare_reference_wav_converts_stereo_clips_12s_resamples_24k_and_caches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "long_stereo.wav"
            sr = 48000
            samples = np.ones((sr * 23, 2), dtype=np.float32) * 1.2
            # Add a quiet pause around 10.5s so pause-aware trim cuts cleanly <= 12s
            samples[int(sr * 10.4) : int(sr * 10.8), :] = 0.0
            sf.write(str(src), samples, sr)

            with patch.object(llama_tts, "_REF_WAV_CACHE_DIR", root / "cache"):
                out1 = llama_tts.prepare_reference_wav(src)
                out2 = llama_tts.prepare_reference_wav(src)
                self.assertEqual(out1, out2)
                self.assertTrue(out1.is_file())
                arr, out_sr = sf.read(str(out1), dtype="float32", always_2d=False)
                self.assertEqual(out_sr, llama_tts.QWEN3_NATIVE_SR)
                self.assertEqual(arr.ndim, 1)
                self.assertLessEqual(len(arr), llama_tts.QWEN3_NATIVE_SR * 12)
                self.assertGreaterEqual(len(arr), int(llama_tts.QWEN3_NATIVE_SR * 10.0))
                self.assertLessEqual(float(np.max(np.abs(arr))), 0.96)

    def test_synthesize_qwen3_tts_sends_multipart_pcm_n_predict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / "qwen_ref.wav"
            sf.write(str(ref), np.zeros(16000, dtype=np.float32), 16000)

            pcm_arr = (0.2 * np.sin(np.linspace(0.0, 20.0, 6000, dtype=np.float32))).astype(np.float32)
            fake_resp = MagicMock()
            fake_resp.status_code = 200
            fake_resp.headers = {"Content-Type": "audio/pcm;rate=24000;encoding=float;bits=32"}
            fake_resp.content = pcm_arr.astype("<f4").tobytes()

            cfg = {
                "qwen3_tts_ref_path": str(ref),
                "qwen3_tts_ref_text": "Hello cloned voice",
                "qwen3_tts_device": "gpu",
            }
            with (
                patch.object(llama_tts, "_REF_WAV_CACHE_DIR", root / "ref_cache"),
                patch.object(llama_tts, "is_qwen3_tts_installed", return_value=True),
                patch.object(llama_tts._qwen3_server, "ensure_started", return_value=54321),
                patch.object(llama_tts.config, "get", side_effect=lambda k, d=None: cfg.get(k, d)),
                patch.object(llama_tts._qwen3_server._session, "post", return_value=fake_resp) as mock_post,
            ):
                audio, out_sr = llama_tts.synthesize_qwen3_tts("Test sentence", lang="en")

            self.assertEqual(out_sr, 24000)
            self.assertEqual(len(audio), len(pcm_arr))
            np.testing.assert_allclose(audio, pcm_arr, atol=1e-6)
            call_kw = mock_post.call_args.kwargs
            self.assertNotIn("json", call_kw)
            form_data = call_kw["data"]
            self.assertEqual(form_data["input"], "Test sentence")
            self.assertEqual(form_data["lang"], "en")
            self.assertEqual(form_data["response_format"], "pcm")
            self.assertEqual(form_data["stream"], "false")
            self.assertEqual(form_data["speaker_ref_text"], "Hello cloned voice")
            self.assertEqual(
                int(form_data["n_predict"]),
                llama_tts.estimate_qwen3_n_predict("Test sentence", ref_text="Hello cloned voice"),
            )
            ref_tuple = call_kw["files"]["speaker_ref"]
            self.assertEqual(ref_tuple[0], "ref.wav")
            self.assertGreater(len(ref_tuple[1]), 44)
            self.assertEqual(ref_tuple[2], "audio/wav")

    def test_single_qwen3_entry_uses_streaming_plan_without_fast_duplicate(self) -> None:
        with patch.object(tts_manager, "_current_lang", return_value="en"):
            visible_ids = [item["id"] for item in tts_manager.get_visible_local_engines()]
        self.assertIn("qwen3_tts", visible_ids)
        self.assertNotIn("qwen3_tts_fast", visible_ids)

        for unsupported_lang in ("uk", "pl"):
            with patch.object(tts_manager, "_current_lang", return_value=unsupported_lang):
                unsup_ids = [item["id"] for item in tts_manager.get_visible_local_engines()]
            self.assertNotIn("qwen3_tts", unsup_ids)

        with (
            patch.object(tts_manager, "_current_lang", return_value="en"),
            patch.object(llama_tts, "is_qwen3_tts_installed", return_value=True),
            patch.object(tts_manager, "is_engine_installed", side_effect=lambda eid: eid == "qwen3_tts"),
        ):
            avail = tts_manager.get_available_active_engines()
            avail_ids = [a["id"] for a in avail]
            self.assertIn("qwen3_tts", avail_ids)
            self.assertNotIn("qwen3_tts_fast", avail_ids)
            plan = tts_manager._resolve_qwen3_stream_plan("Hello world", engine_override="qwen3_tts", voice_override="qwen3_en")
            self.assertIsNotNone(plan)
            self.assertEqual(plan["primary_id"], "qwen3_tts")

    def test_estimate_qwen3_n_predict_bounds(self) -> None:
        short_n = llama_tts.estimate_qwen3_n_predict("Привіт!")
        med_n = llama_tts.estimate_qwen3_n_predict("Як справи, чим займаєшся сьогодні?")
        long_n = llama_tts.estimate_qwen3_n_predict("word " * 80)
        huge_n = llama_tts.estimate_qwen3_n_predict("a" * 1500, ref_text="reference transcript")
        self.assertEqual(short_n, 96)
        self.assertGreater(med_n, short_n)
        self.assertLess(med_n, 300)
        self.assertGreater(long_n, 512)
        self.assertLessEqual(long_n, 1200)
        self.assertGreaterEqual(huge_n, 96)
        self.assertLess(huge_n, 2048 - 1500)

    def test_synthesize_qwen3_tts_stream_handles_unaligned_pcm_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / "qwen_ref.wav"
            sf.write(str(ref), np.zeros(24000, dtype=np.float32), 24000)

            full_arr = np.linspace(-0.5, 0.5, 2500, dtype=np.float32)
            raw_bytes = full_arr.astype("<f4").tobytes()
            # Intentionally split mid-float32 (non-multiple of 4 bytes)
            split_parts = [raw_bytes[:1001], raw_bytes[1001:4003], raw_bytes[4003:]]

            fake_resp = MagicMock()
            fake_resp.status_code = 200
            fake_resp.headers = {"Content-Type": "audio/pcm;rate=24000;encoding=float;bits=32"}
            fake_resp.iter_content.return_value = iter(split_parts)

            cfg = {"qwen3_tts_ref_path": str(ref), "qwen3_tts_ref_text": ""}
            with (
                patch.object(llama_tts, "_REF_WAV_CACHE_DIR", root / "ref_cache"),
                patch.object(llama_tts, "is_qwen3_tts_installed", return_value=True),
                patch.object(llama_tts._qwen3_server, "ensure_started", return_value=54322),
                patch.object(llama_tts.config, "get", side_effect=lambda k, d=None: cfg.get(k, d)),
                patch.object(llama_tts._qwen3_server._session, "post", return_value=fake_resp) as mock_post,
            ):
                chunks = list(llama_tts.synthesize_qwen3_tts_stream("Stream test sentence", lang="en"))

            self.assertEqual(len(chunks), 3)
            self.assertTrue(all(sr == 24000 for _, sr in chunks))
            merged = np.concatenate([c for c, _ in chunks])
            np.testing.assert_allclose(merged, full_arr, atol=1e-6)
            self.assertEqual(mock_post.call_args.kwargs["data"]["stream"], "true")
            self.assertTrue(mock_post.call_args.kwargs["stream"])

    def test_warmup_runs_silent_2_frame_inference_once_per_port(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / "voice.wav"
            sf.write(str(ref), np.zeros(24000, dtype=np.float32), 24000)

            fake_resp = MagicMock()
            fake_resp.status_code = 200
            fake_resp.headers = {"Content-Type": "audio/pcm;rate=24000;encoding=float;bits=32"}
            fake_resp.content = np.zeros(128, dtype="<f4").tobytes()

            cfg = {"qwen3_tts_ref_path": str(ref), "app_language": "uk", "active_voice": "qwen3_en"}
            mgr = llama_tts._Qwen3ServerManager()
            with (
                patch.object(llama_tts, "_REF_WAV_CACHE_DIR", root / "ref_cache"),
                patch.object(llama_tts, "_qwen3_server", mgr),
                patch.object(llama_tts, "is_qwen3_tts_installed", return_value=True),
                patch.object(llama_tts.config, "get", side_effect=lambda k, d=None: cfg.get(k, d)),
                patch.object(mgr, "ensure_started", return_value=56789),
                patch.object(mgr._session, "post", return_value=fake_resp) as mock_post,
            ):
                llama_tts.warmup_qwen3_tts()
                llama_tts.warmup_qwen3_tts()

            self.assertEqual(mock_post.call_count, 1)
            self.assertEqual(mock_post.call_args.kwargs["data"]["n_predict"], "2")
            self.assertEqual(mock_post.call_args.kwargs["data"]["lang"], "en")

    def test_cancel_qwen3_synthesis_aborts_blocked_http_request_immediately(self) -> None:
        import socket
        import threading
        import time

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        port = srv.getsockname()[1]
        srv.listen(1)
        accepted = threading.Event()
        release_srv = threading.Event()

        def _hanging_server() -> None:
            try:
                conn, _ = srv.accept()
                conn.recv(4096)
                accepted.set()
                release_srv.wait(timeout=5.0)
                conn.close()
            except OSError:
                pass

        t_srv = threading.Thread(target=_hanging_server, daemon=True)
        t_srv.start()

        mgr = llama_tts._Qwen3ServerManager()

        def _trigger_cancel() -> None:
            accepted.wait(timeout=2.0)
            time.sleep(0.05)
            mgr.cancel_active(only_warmup=False)

        threading.Thread(target=_trigger_cancel, daemon=True).start()

        t0 = time.perf_counter()
        try:
            with patch.object(mgr, "ensure_started", return_value=port):
                with self.assertRaises(llama_tts.Qwen3SynthesisCancelledError):
                    mgr.synthesize("Hanging phrase", "en", None, "", "gpu")
            elapsed = time.perf_counter() - t0
            self.assertLess(elapsed, 1.5)
        finally:
            release_srv.set()
            srv.close()

    def test_cold_start_stop_inside_ensure_started_does_not_cancel_first_synthesis(self) -> None:
        mgr = llama_tts._Qwen3ServerManager()
        pcm_arr = np.full(400, 0.15, dtype=np.float32)
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.headers = {"Content-Type": "audio/pcm;rate=24000;encoding=float;bits=32"}
        fake_resp.content = pcm_arr.astype("<f4").tobytes()

        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        fake_proc.stderr = None

        with (
            patch.object(llama_tts, "is_qwen3_tts_installed", return_value=True),
            patch.object(llama_tts, "_pick_free_port", return_value=57777),
            patch.object(llama_tts, "assign_process_to_kill_on_close_job"),
            patch.object(llama_tts.subprocess, "Popen", return_value=fake_proc),
            patch.object(mgr, "_wait_ready"),
            patch.object(mgr._session, "post", return_value=fake_resp),
        ):
            audio, sr = mgr.synthesize("First cold phrase", "en", None, "", "gpu")
        self.assertEqual(sr, 24000)
        self.assertEqual(len(audio), 400)
        mgr.stop()

    def test_tts_manager_stop_current_cancels_qwen3_and_streaming_caches_full_audio(self) -> None:
        with patch("core.llama_tts.cancel_qwen3_synthesis") as mock_cancel:
            tts_manager.stop_current()
            mock_cancel.assert_called_once()

        c1 = np.full(600, 0.2, dtype=np.float32)
        c2 = np.full(600, 0.3, dtype=np.float32)
        stored: list[tuple] = []
        ready_called: list[bool] = []

        def _fake_play_stream(chunk_iter, **kw):
            chunks = list(chunk_iter)
            first_s, sr = chunks[0]
            if kw.get("on_first_chunk"):
                kw["on_first_chunk"](first_s, sr, 50.0)
            full = np.concatenate([x for x, _ in chunks])
            if kw.get("on_generation_done"):
                kw["on_generation_done"](full, sr)
            return full, sr

        plan = {
            "primary_id": "qwen3_tts",
            "primary_voice": "qwen3_en",
            "primary_title": "Qwen3-TTS 1.7B Base (GGUF)",
            "synth_text": "Hello stream",
            "cache_key": ("Hello stream", "qwen3_tts", "qwen3_en"),
            "lang": "en",
        }
        tts_manager._cancel_batch = False
        tts_manager._running = True
        with (
            patch("core.llama_tts.synthesize_qwen3_tts_stream", return_value=iter([(c1, 24000), (c2, 24000)])),
            patch("core.tts_manager.audio_router.play_stream", side_effect=_fake_play_stream),
            patch("core.tts_manager.obs_server.broadcast_speech"),
            patch.object(tts_manager, "_cache_store", side_effect=lambda *a: stored.append(a)),
        ):
            ok = tts_manager._speak_qwen3_streaming_chunk(
                "Hello stream",
                "Hello stream",
                plan=plan,
                emit_phrase_spoken=False,
                monitor_only=True,
                send_primary=False,
                vban_stream_override="",
                extra_gain_pct=100,
                on_ready_for_next=lambda: ready_called.append(True),
            )
        self.assertTrue(ok)
        self.assertEqual(ready_called, [True])
        self.assertEqual(len(stored), 1)
        self.assertEqual(len(stored[0][1]), 1200)

    def test_engine_specs_and_cache_key_sensitivity(self) -> None:
        self.assertNotIn("chatterbox", ENGINE_SPECS)
        self.assertNotIn("qwen3_tts_06b", ENGINE_SPECS)
        self.assertIn("qwen3_tts", ENGINE_SPECS)
        with tempfile.TemporaryDirectory() as tmp:
            ref1 = Path(tmp) / "a.wav"
            ref2 = Path(tmp) / "b.wav"
            sf.write(str(ref1), np.zeros(8000, dtype=np.float32), 16000)
            sf.write(str(ref2), np.ones(8000, dtype=np.float32) * 0.1, 16000)

            state = {
                "app_language": "uk",
                "qwen3_tts_ref_path": str(ref1),
                "qwen3_tts_ref_text": "hello",
            }
            with patch("core.tts_manager.config.get", side_effect=lambda k, d=None: state.get(k, d)):
                k1 = tts_manager._build_audio_cache_key("hi", "qwen3_tts", "qwen3_en")
                state["qwen3_tts_ref_text"] = "world"
                k2 = tts_manager._build_audio_cache_key("hi", "qwen3_tts", "qwen3_en")
                state["qwen3_tts_ref_path"] = str(ref2)
                k3 = tts_manager._build_audio_cache_key("hi", "qwen3_tts", "qwen3_en")

                self.assertNotEqual(k1, k2)
                self.assertNotEqual(k2, k3)

    def test_qwen3_never_falls_back_to_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / "voice.wav"
            sf.write(str(ref), np.zeros(8000, dtype=np.float32), 16000)

            wav_bytes = _make_wav_bytes(24000, 0.2)
            cfg = {"qwen3_tts_ref_path": str(ref), "app_language": "en"}
            modes_started = []

            def _fake_ensure(mode, engine_id=None):
                modes_started.append(mode)
                return 55555

            fake_resp = MagicMock()
            fake_resp.status_code = 200
            fake_resp.content = wav_bytes

            with (
                patch.object(llama_tts, "_REF_WAV_CACHE_DIR", root / "ref_cache"),
                patch.object(llama_tts, "is_qwen3_tts_installed", return_value=True),
                patch.object(llama_tts.config, "get", side_effect=lambda k, d=None: cfg.get(k, d)),
                patch.object(llama_tts._qwen3_server, "ensure_started", side_effect=_fake_ensure),
                patch.object(llama_tts._qwen3_server._session, "post", return_value=fake_resp),
            ):
                llama_tts.warmup_qwen3_tts()
                llama_tts.synthesize_qwen3_tts("GPU only test", lang="en")
            self.assertEqual(modes_started, ["gpu", "gpu"])

    def test_engine_switch_or_idle_unloads_qwen3(self) -> None:
        import time

        # 1. Idle timer unloads Qwen3 only when NOT selected as active/fallback (30s default)
        self.assertEqual(llama_tts._Qwen3ServerManager.IDLE_UNLOAD_SEC, 30.0)
        mgr = llama_tts._Qwen3ServerManager()
        mgr._idle_timeout_sec = 0.05
        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        mgr._proc = fake_proc
        with patch.object(llama_tts.config, "get", side_effect=lambda k, d=None: "qwen3_tts" if k == "active_engine" else d):
            mgr._reset_idle_timer()
            time.sleep(0.12)
            fake_proc.terminate.assert_not_called()
            self.assertIsNotNone(mgr._proc)

        with patch.object(llama_tts.config, "get", side_effect=lambda k, d=None: "silero" if k == "active_engine" else ""):
            mgr._reset_idle_timer()
            time.sleep(0.15)
            fake_proc.terminate.assert_called_once()
            self.assertIsNone(mgr._proc)

        # 2. Switching away from qwen3_tts in _run_smart_warmup unloads running Qwen3 server
        with (
            patch.object(tts_manager, "_resolve_local_warmup_targets", return_value=[]),
            patch.object(tts_manager, "_resolve_online_warmup_targets", return_value=[]),
            patch("core.llama_tts.is_qwen3_server_running", return_value=True),
            patch("core.llama_tts.shutdown_llama_tts") as mock_shut,
        ):
            tts_manager._warmed_models.add(("qwen3_tts", "qwen3_tts"))
            tts_manager._run_smart_warmup("engine_changed", 0)
            mock_shut.assert_called_once()
            self.assertFalse(any(item[0] == "qwen3_tts" for item in tts_manager._warmed_models))

    def test_install_runtime_downloads_cuda_on_nvidia_and_vulkan_on_non_nvidia(self) -> None:
        import zipfile
        from core.win_api import is_nvidia_gpu_adapter

        self.assertTrue(is_nvidia_gpu_adapter("NVIDIA GeForce RTX 4070", "PCI\\VEN_10DE&DEV_2786"))
        self.assertFalse(is_nvidia_gpu_adapter("AMD Radeon RX 7800 XT", "PCI\\VEN_1002&DEV_747E"))
        self.assertFalse(is_nvidia_gpu_adapter("Intel(R) Arc(TM) A770 Graphics", "PCI\\VEN_8086&DEV_56A0"))
        self.assertFalse(is_nvidia_gpu_adapter("Microsoft Basic Render Driver", "PCI\\VEN_1414&DEV_008C", flags=2))

        def _write_fake_zip(dest: Path, names: tuple[str, ...]) -> None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(dest, "w") as zf:
                for n in names:
                    zf.writestr(n, b"binary")

        # 1. Non-NVIDIA -> downloads Vulkan zip and writes "vulkan-ok"
        with tempfile.TemporaryDirectory() as tmp:
            rt = Path(tmp) / "llama_tts"
            downloaded_urls: list[str] = []

            def _fake_dl_vulkan(url: str, dest: Path, eid: str, bpct: int, spct: int, lbl: str) -> None:
                downloaded_urls.append(url)
                _write_fake_zip(dest, ("llama-tts.exe", "llama-server.exe", "ggml-vulkan.dll"))

            with (
                patch.object(llama_tts, "LLAMA_TTS_DIR", rt),
                patch.object(llama_tts, "LLAMA_TTS_EXE", rt / "llama-tts.exe"),
                patch.object(llama_tts, "LLAMA_TTS_SERVER_EXE", rt / "llama-server.exe"),
                patch.object(llama_tts, "LLAMA_TTS_READY_MARKER", rt / "ready.marker"),
                patch("core.llama_tts.has_nvidia_gpu", return_value=False),
                patch(
                    "core.llama_tts.resolve_llama_cpp_win_asset_url",
                    return_value="https://example.invalid/llama-win-vulkan-x64.zip",
                ),
            ):
                llama_tts.install_llama_tts_runtime("qwen3_tts", lambda p, l: None, _fake_dl_vulkan, lambda: None)
                self.assertEqual(downloaded_urls, ["https://example.invalid/llama-win-vulkan-x64.zip"])
                self.assertEqual((rt / "ready.marker").read_text(encoding="utf-8"), "vulkan-ok")
                self.assertTrue((rt / "ggml-vulkan.dll").is_file())

            # 2. Same folder upgraded on NVIDIA -> downloads CUDA + cudart zips and writes "cuda-ok"
            downloaded_cuda: list[str] = []

            def _fake_dl_cuda(url: str, dest: Path, eid: str, bpct: int, spct: int, lbl: str) -> None:
                downloaded_cuda.append(url)
                if "cudart" in url:
                    _write_fake_zip(dest, ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"))
                else:
                    _write_fake_zip(dest, ("llama-tts.exe", "llama-server.exe", "ggml-cuda.dll"))

            with (
                patch.object(llama_tts, "LLAMA_TTS_DIR", rt),
                patch.object(llama_tts, "LLAMA_TTS_EXE", rt / "llama-tts.exe"),
                patch.object(llama_tts, "LLAMA_TTS_SERVER_EXE", rt / "llama-server.exe"),
                patch.object(llama_tts, "LLAMA_TTS_READY_MARKER", rt / "ready.marker"),
                patch("core.llama_tts.has_nvidia_gpu", return_value=True),
                patch(
                    "core.llama_tts.resolve_llama_cpp_cuda_asset_urls",
                    return_value=(
                        "https://example.invalid/llama-win-cuda-12.4-x64.zip",
                        "https://example.invalid/cudart-llama-win-cuda-12.4-x64.zip",
                    ),
                ),
            ):
                llama_tts.install_llama_tts_runtime("qwen3_tts", lambda p, l: None, _fake_dl_cuda, lambda: None)
                self.assertEqual(
                    downloaded_cuda,
                    [
                        "https://example.invalid/llama-win-cuda-12.4-x64.zip",
                        "https://example.invalid/cudart-llama-win-cuda-12.4-x64.zip",
                    ],
                )
                self.assertEqual((rt / "ready.marker").read_text(encoding="utf-8"), "cuda-ok")
                self.assertTrue((rt / "ggml-cuda.dll").is_file())
                self.assertTrue((rt / "cublas64_12.dll").is_file())

    def test_qwen3_vram_threshold_and_quant_selection_dialog_behavior(self) -> None:
        from PySide6.QtWidgets import QApplication, QDialog
        from ui.dialogs import Qwen3QuantChoiceDialog
        from ui.main_window import MainWindow

        _app = QApplication.instance() or QApplication([])

        # 1. VRAM threshold: >= 9 GB skips dialog -> auto Q8; < 9 GB (4/6/8 GB) or None shows dialog
        self.assertFalse(llama_tts.should_prompt_qwen3_quant_for_vram(16.0))
        self.assertFalse(llama_tts.should_prompt_qwen3_quant_for_vram(12.0))
        self.assertFalse(llama_tts.should_prompt_qwen3_quant_for_vram(10.0))
        self.assertFalse(llama_tts.should_prompt_qwen3_quant_for_vram(9.0))
        self.assertTrue(llama_tts.should_prompt_qwen3_quant_for_vram(8.0))
        self.assertTrue(llama_tts.should_prompt_qwen3_quant_for_vram(6.0))
        self.assertTrue(llama_tts.should_prompt_qwen3_quant_for_vram(None))

        # 1b. SILENTVOICE_TEST_VRAM_GB env override unlocks GPU button & reports simulated VRAM
        import os
        from core.win_api import get_discrete_gpu_vram_gb, has_hardware_gpu

        with patch.dict(os.environ, {"SILENTVOICE_TEST_VRAM_GB": "8"}):
            self.assertTrue(has_hardware_gpu())
            self.assertEqual(get_discrete_gpu_vram_gb(), 8.0)

        # 2. Q5_K_M is removed; only Q8_0 is offered, while legacy Q5_K_M/Q4_K_M files are still detected
        self.assertNotIn("q5", llama_tts.QWEN3_QUANT_SPECS)
        self.assertEqual(llama_tts.normalize_qwen3_quant("q5"), "q8")
        self.assertEqual(llama_tts.get_qwen3_model_spec("q8")["file"], "Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf")
        self.assertEqual(llama_tts.get_qwen3_model_spec("q5")["file"], "Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rt = root / "llama_tts"
            rt.mkdir()
            (rt / "llama-tts.exe").write_bytes(b"exe")
            (rt / "llama-server.exe").write_bytes(b"srv")
            (rt / "ready.marker").write_bytes(b"ok")
            qw = root / "qwen3_tts"
            qw.mkdir()
            (qw / llama_tts.QWEN3_MMPROJ_FILE).write_bytes(b"mm")
            (qw / llama_tts.QWEN3_MODEL_FILE).write_bytes(b"q8")

            with (
                patch.object(llama_tts, "LLAMA_TTS_EXE", rt / "llama-tts.exe"),
                patch.object(llama_tts, "LLAMA_TTS_SERVER_EXE", rt / "llama-server.exe"),
                patch.object(llama_tts, "LLAMA_TTS_READY_MARKER", rt / "ready.marker"),
                patch.object(llama_tts, "QWEN3_TTS_DIR", qw),
            ):
                self.assertTrue(llama_tts.is_qwen3_tts_installed())
                self.assertEqual(
                    llama_tts.find_installed_qwen3_model_file().name,
                    "Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
                )

        # 3. Dialog defaults to ZONOS3-Mini when VRAM < 9 GB, and can switch to Qwen3 Q8
        dlg_real = Qwen3QuantChoiceDialog(8.0)
        self.assertEqual(dlg_real.selected_engine, "zonos3_mini")
        dlg_real.radio_q8.setChecked(True)
        self.assertEqual(dlg_real.selected_engine, "qwen3_tts")

        # 4. MainWindow._start_model_download("qwen3_tts"):
        #    - >= 9 GB VRAM -> does NOT open dialog, sets q8, starts qwen3_tts download
        #    - < 9 GB VRAM & picks ZONOS3-Mini -> starts zonos3_mini download
        #    - < 9 GB VRAM & picks Qwen3 Q8 -> sets q8, starts qwen3_tts download
        #    - < 9 GB VRAM & cancelled -> does NOT start download
        dummy = MainWindow.__new__(MainWindow)
        dummy._ensure_language_confirmed = lambda: True
        dummy._engine_cards_ui = {}
        saved_cfg: dict = {"hide_local_model_first_run_notice": True}

        def _cfg_get(k, d=None):
            return saved_cfg.get(k, d)

        def _cfg_set(k, v):
            saved_cfg[k] = v

        with (
            patch("ui.main_window.config.get", side_effect=_cfg_get),
            patch("ui.main_window.config.set", side_effect=_cfg_set),
            patch("ui.main_window.tts_manager.is_downloading", return_value=False),
            patch("ui.main_window.get_discrete_gpu_vram_gb", return_value=16.0),
            patch("ui.main_window.Qwen3QuantChoiceDialog") as mock_dlg_cls,
            patch("ui.main_window.tts_manager.download_engine_model") as mock_dl,
        ):
            dummy._start_model_download("qwen3_tts")
            mock_dlg_cls.assert_not_called()
            self.assertEqual(saved_cfg.get("qwen3_tts_quant"), "q8")
            mock_dl.assert_called_once_with("qwen3_tts")

        class _PickZonosDlg:
            def __init__(self, vram_gb, parent=None):
                self.vram_gb = vram_gb
                self.selected_engine = "zonos3_mini"

            def exec(self):
                return QDialog.DialogCode.Accepted

        with (
            patch("ui.main_window.config.get", side_effect=_cfg_get),
            patch("ui.main_window.config.set", side_effect=_cfg_set),
            patch("ui.main_window.tts_manager.is_downloading", return_value=False),
            patch("ui.main_window.tts_manager.is_engine_installed", return_value=False),
            patch("ui.main_window.get_discrete_gpu_vram_gb", return_value=8.0),
            patch("ui.main_window.Qwen3QuantChoiceDialog", _PickZonosDlg),
            patch("ui.main_window.tts_manager.download_engine_model") as mock_dl,
        ):
            dummy._start_model_download("qwen3_tts")
            mock_dl.assert_called_once_with("zonos3_mini")

        class _PickQwenDlg:
            def __init__(self, vram_gb, parent=None):
                self.vram_gb = vram_gb
                self.selected_engine = "qwen3_tts"

            def exec(self):
                return QDialog.DialogCode.Accepted

        with (
            patch("ui.main_window.config.get", side_effect=_cfg_get),
            patch("ui.main_window.config.set", side_effect=_cfg_set),
            patch("ui.main_window.tts_manager.is_downloading", return_value=False),
            patch("ui.main_window.get_discrete_gpu_vram_gb", return_value=8.0),
            patch("ui.main_window.Qwen3QuantChoiceDialog", _PickQwenDlg),
            patch("ui.main_window.tts_manager.download_engine_model") as mock_dl,
        ):
            dummy._start_model_download("qwen3_tts")
            self.assertEqual(saved_cfg.get("qwen3_tts_quant"), "q8")
            mock_dl.assert_called_once_with("qwen3_tts")

        class _CancelDlg:
            def __init__(self, vram_gb, parent=None):
                self.selected_engine = "zonos3_mini"

            def exec(self):
                return QDialog.DialogCode.Rejected

        with (
            patch("ui.main_window.config.get", side_effect=_cfg_get),
            patch("ui.main_window.config.set", side_effect=_cfg_set),
            patch("ui.main_window.tts_manager.is_downloading", return_value=False),
            patch("ui.main_window.get_discrete_gpu_vram_gb", return_value=6.0),
            patch("ui.main_window.Qwen3QuantChoiceDialog", _CancelDlg),
            patch("ui.main_window.tts_manager.download_engine_model") as mock_dl,
        ):
            dummy._start_model_download("qwen3_tts")
            mock_dl.assert_not_called()


if __name__ == "__main__":
    unittest.main()
