"""Unit tests for persistent local audio streams, stereo preservation, i32 fallback, and speed/pitch DSP."""

from __future__ import annotations

import time
import unittest
from unittest.mock import patch

import numpy as np

import core.audio_router as audio_router_mod
from core.audio_router import (
    DualAudioRouter,
    _PersistentDevicePlayer,
    apply_volume_and_limiter,
    clean_audio_edges,
)
from core.config import config
from core.tts_manager import _fx_speed_and_pitch_array, compute_speech_char_delays


class _FakeOutputStream:
    def __init__(self, *, samplerate, channels, dtype, device, callback) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.device = device
        self.callback = callback
        self.active = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.active = True

    def stop(self) -> None:
        self.active = False
        self.stopped = True

    def close(self) -> None:
        self.active = False
        self.closed = True

    def pull(self, frames: int) -> np.ndarray:
        buf = np.zeros((frames, self.channels), dtype=self.dtype)
        self.callback(buf, frames, None, None)
        return buf


class AudioProcessingTests(unittest.TestCase):
    def test_clean_audio_edges_preserves_stereo_and_mono(self) -> None:
        mono = np.full(2400, 0.5, dtype=np.float32) + 0.1
        out_mono = clean_audio_edges(mono, 24000, fade_in_ms=10, fade_out_ms=10, pad_tail_ms=20)
        self.assertEqual(out_mono.ndim, 1)
        self.assertEqual(len(out_mono), 2400 + int(24000 * 0.02))
        self.assertAlmostEqual(float(out_mono[-1]), 0.0, places=5)

        left = np.full(2400, 0.4, dtype=np.float32)
        right = np.full(2400, -0.2, dtype=np.float32)
        stereo = np.stack([left, right], axis=1)
        out_stereo = clean_audio_edges(stereo, 24000, fade_in_ms=10, fade_out_ms=10, pad_tail_ms=20)
        self.assertEqual(out_stereo.shape, (2400 + int(24000 * 0.02), 2))
        self.assertAlmostEqual(float(out_stereo[0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(out_stereo[-1, 1]), 0.0, places=5)

    def test_volume_limiter_preserves_stereo_channels(self) -> None:
        stereo = np.array([[0.8, -0.4], [0.9, -0.2]], dtype=np.float32)
        out = apply_volume_and_limiter(stereo, 150)
        self.assertEqual(out.shape, (2, 2))
        self.assertGreater(float(out[0, 0]), 0.8)
        self.assertLess(float(out[0, 1]), -0.4)
        self.assertLessEqual(float(np.max(np.abs(out))), 1.0)

    def test_speed_and_pitch_and_stereo_char_delays(self) -> None:
        sr = 24000
        t = np.arange(sr, dtype=np.float32) / sr
        mono = 0.25 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
        stretched = _fx_speed_and_pitch_array(mono, sr, speed_rate=1.25, pitch_steps=2.0)
        self.assertEqual(stretched.ndim, 1)
        self.assertAlmostEqual(len(stretched), int(sr / 1.25), delta=512)

        # Verify OLA fallback works cleanly even when ffmpeg is absent
        with patch("core.tts_manager._find_ffmpeg_exe", return_value=None):
            ola_stretched = _fx_speed_and_pitch_array(mono, sr, speed_rate=1.25, pitch_steps=2.0)
        self.assertEqual(ola_stretched.ndim, 1)
        self.assertAlmostEqual(len(ola_stretched), int(sr / 1.25), delta=512)

        stereo = np.stack([mono, mono * 0.5], axis=1)
        delays = compute_speech_char_delays("Привіт", stereo, sr)
        self.assertEqual(len(delays), 6)

    def test_install_ffmpeg_extracts_binary_and_reports_installed(self) -> None:
        import io
        import tempfile
        import zipfile
        from pathlib import Path
        import core.tts_manager as tm_mod

        with tempfile.TemporaryDirectory() as tmp:
            ff_dir = Path(tmp) / "ffmpeg"
            ff_exe = ff_dir / "ffmpeg.exe"
            fake_payload = b"MZ" + (b"\x00" * 1_100_000)

            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("ffmpeg-build/bin/ffmpeg.exe", fake_payload)
            zip_bytes = buf.getvalue()

            def _fake_download(url, dest, eid, bp, sp, lbl, cancel_event=None):
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zip_bytes)

            mgr = tm_mod.tts_manager
            done_events: list[tuple[str, bool]] = []
            conn = mgr.model_status_changed.connect(lambda eid, ok: done_events.append((eid, ok)))
            try:
                with (
                    patch.object(tm_mod, "FFMPEG_DIR", ff_dir),
                    patch.object(tm_mod, "FFMPEG_EXE", ff_exe),
                    patch.object(tm_mod, "_find_ffmpeg_exe", side_effect=lambda: str(ff_exe) if ff_exe.is_file() else None),
                    patch.object(mgr, "_download_url_with_progress", side_effect=_fake_download),
                ):
                    self.assertFalse(tm_mod.is_ffmpeg_installed())
                    mgr._ffmpeg_install_worker(__import__("threading").Event())
                    self.assertTrue(tm_mod.is_ffmpeg_installed())
                    self.assertTrue(ff_exe.is_file())
                    self.assertIn((tm_mod.FFMPEG_ENGINE_ID, True), done_events)
            finally:
                mgr.model_status_changed.disconnect(conn)

    def test_ffmpeg_install_dialog_and_dsp_custom_title(self) -> None:
        import os
        from PySide6.QtWidgets import QApplication
        import ui.dialogs as dlg_mod
        import ui.main_window as mw_mod
        import core.tts_manager as tm_mod

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        _app = QApplication.instance() or QApplication([])

        with patch.object(dlg_mod, "is_ffmpeg_installed", return_value=False):
            dlg = dlg_mod.FFmpegInstallDialog(voice_is_custom=True)
            try:
                self.assertIn("Custom", dlg.findChildren(dlg_mod.QLabel)[1].text())
                with patch.object(tm_mod.tts_manager, "install_ffmpeg") as mock_install:
                    dlg.install_btn.click()
                    mock_install.assert_called_once()
                dlg.reject()
            finally:
                dlg.close()

        # Verify _maybe_prompt_ffmpeg_install triggers dialog for ::custom voice when ffmpeg missing
        prompted: list[bool] = []

        class _FakeDlg:
            class DialogCode:
                Accepted = 1

            def __init__(self, parent=None, *, voice_is_custom=False):
                prompted.append(voice_is_custom)

            def exec(self):
                return 0

        dummy = type("DummyMW", (), {
            "_ffmpeg_prompt_open": False,
            "_ffmpeg_last_declined_voice": None,
        })()
        dummy._maybe_prompt_ffmpeg_install = mw_mod.MainWindow._maybe_prompt_ffmpeg_install.__get__(dummy)
        with (
            patch.object(mw_mod, "is_ffmpeg_installed", return_value=False),
            patch.object(mw_mod.tts_manager, "is_downloading", return_value=False),
            patch.object(mw_mod, "FFmpegInstallDialog", _FakeDlg),
        ):
            dummy._maybe_prompt_ffmpeg_install("online:edge_-1_1_6000", "uk-UA-PolinaNeural::custom", from_voice_switch=True)
        self.assertEqual(prompted, [True])
        edge_custom = next(
            v for v in tm_mod.ONLINE_PROVIDERS["edge"]["voices"] if v["id"] == "uk-UA-PolinaNeural::custom"
        )
        self.assertIn("DSP Custom", edge_custom["name"])


class PersistentStreamTests(unittest.TestCase):
    def test_persistent_player_stereo_and_i32_fallback(self) -> None:
        created_streams: list[_FakeOutputStream] = []

        def _fake_stream(*, samplerate, channels, dtype, device, callback):
            if dtype == "float32":
                raise RuntimeError("float32 unsupported by test DAC")
            st = _FakeOutputStream(
                samplerate=samplerate,
                channels=channels,
                dtype=dtype,
                device=device,
                callback=callback,
            )
            created_streams.append(st)
            return st

        with (
            patch.object(audio_router_mod.sd, "query_devices", return_value={"default_samplerate": 48000, "max_output_channels": 2}),
            patch.object(audio_router_mod.sd, "OutputStream", side_effect=_fake_stream),
        ):
            player = _PersistentDevicePlayer(device_idx=3, preferred_sr=48000)
            try:
                self.assertEqual(player.channels, 2)
                self.assertEqual(player.dtype, "int32")
                self.assertEqual(len(created_streams), 1)
                stream = created_streams[0]

                # Stereo phrase preserves distinct L/R channels and converts to i32.
                stereo = np.column_stack([
                    np.full(128, 0.5, dtype=np.float32),
                    np.full(128, -0.25, dtype=np.float32),
                ])
                h1 = player.enqueue(stereo, 48000)
                block = stream.pull(256)
                self.assertEqual(block.dtype, np.int32)
                self.assertGreater(int(block[0, 0]), 1_000_000_000)
                self.assertLess(int(block[0, 1]), -400_000_000)
                self.assertTrue(h1.is_done())

                # Second phrase (mono) reuses the exact same open stream and duplicates L/R.
                mono = np.full(64, 0.3, dtype=np.float32)
                h2 = player.enqueue(mono, 48000)
                block2 = stream.pull(64)
                self.assertEqual(len(created_streams), 1)
                self.assertEqual(int(block2[0, 0]), int(block2[0, 1]))
                self.assertTrue(h2.is_done())
            finally:
                player.close()

    def test_persistent_player_stop_fades_out_promptly(self) -> None:
        created_streams: list[_FakeOutputStream] = []

        def _fake_stream(*, samplerate, channels, dtype, device, callback):
            st = _FakeOutputStream(
                samplerate=samplerate,
                channels=channels,
                dtype=dtype,
                device=device,
                callback=callback,
            )
            created_streams.append(st)
            return st

        with (
            patch.object(audio_router_mod.sd, "query_devices", return_value={"default_samplerate": 48000, "max_output_channels": 2}),
            patch.object(audio_router_mod.sd, "OutputStream", side_effect=_fake_stream),
        ):
            player = _PersistentDevicePlayer(device_idx=1, preferred_sr=48000)
            try:
                audio = np.full((1024, 2), 0.6, dtype=np.float32)
                handle = player.enqueue(audio, 48000)
                handle.stop()
                self.assertTrue(handle.is_done())
                block = created_streams[0].pull(256)
                # After fade-out (<= 64 frames), remainder of the block is zero.
                self.assertTrue(np.allclose(block[64:], 0.0))
            finally:
                player.close()

    def test_router_reuses_persistent_stream_across_sentences(self) -> None:
        open_count = 0

        class _AutoDrainStream(_FakeOutputStream):
            def start(self) -> None:
                super().start()
                import threading

                def _pump() -> None:
                    while self.active:
                        self.pull(256)
                        time.sleep(0.005)

                threading.Thread(target=_pump, daemon=True).start()

        def _fake_stream(*, samplerate, channels, dtype, device, callback):
            nonlocal open_count
            open_count += 1
            return _AutoDrainStream(
                samplerate=samplerate,
                channels=channels,
                dtype=dtype,
                device=device,
                callback=callback,
            )

        router = DualAudioRouter()
        overrides = {
            "primary_output_device": "FakeCable",
            "primary_volume": 100,
            "secondary_enabled": False,
            "vban_enabled": False,
        }

        try:
            with (
                patch.object(config, "get", side_effect=lambda k, d=None: overrides.get(k, d)),
                patch.object(audio_router_mod, "resolve_output_device_index", return_value=7),
                patch.object(audio_router_mod, "resolve_secondary_output_targets", return_value=[]),
                patch.object(audio_router_mod.sd, "query_devices", return_value={"default_samplerate": 48000, "max_output_channels": 2}),
                patch.object(audio_router_mod.sd, "OutputStream", side_effect=_fake_stream),
            ):
                s1 = np.full(1024, 0.2, dtype=np.float32)
                s2 = np.full((1024, 2), 0.3, dtype=np.float32)
                router.play_samples(s1, 48000)
                router.play_samples(s2, 48000)
                self.assertEqual(open_count, 1)

                # Verify on_playback_queued receives real stream latency and delays shift accordingly
                player = router._players[7]
                player._stream.latency = 0.18
                queued_lats: list[float] = []
                router.play_samples(s1, 48000, on_playback_queued=queued_lats.append)
                self.assertEqual(len(queued_lats), 1)
                self.assertAlmostEqual(queued_lats[0], 180.0, delta=0.5)
                delays_shifted = compute_speech_char_delays("Привіт", s1, 48000, start_offset_ms=queued_lats[0])
                self.assertGreaterEqual(delays_shifted[0], 180)
        finally:
            router.shutdown()

    def test_default_secondary_prefers_real_speakers_over_virtual_enhancers(self) -> None:
        fake_devices = [
            {"index": 5, "name": "CABLE In 16ch (VB-Audio Virtual Cable)", "hostapi": "MME", "default_samplerate": 44100},
            {"index": 6, "name": "CABLE Input (VB-Audio Virtual Cable)", "hostapi": "MME", "default_samplerate": 44100},
            {"index": 7, "name": "FxSound Speakers (FxSound Audio Enhancer)", "hostapi": "MME", "default_samplerate": 44100},
            {"index": 8, "name": "Speakers (Realtek(R) Audio)", "hostapi": "MME", "default_samplerate": 44100},
        ]
        with (
            patch.object(audio_router_mod, "get_output_devices", return_value=fake_devices),
            patch.object(audio_router_mod, "_get_core_audio_default_output_names", return_value={"console": "CABLE In 16ch (VB-Audio Virtual Cable)", "communications": "Speakers (Realtek(R) Audio)"}),
            patch.object(audio_router_mod, "_get_windows_preferred_output_name", return_value="CABLE In 16ch (VB-Audio Virtual"),
            patch.object(config, "get", side_effect=lambda k, d=None: "" if k == "secondary_output_device" else d),
        ):
            self.assertEqual(audio_router_mod.resolve_secondary_output_targets(), [8])

    def test_hotplug_headphones_refreshes_portaudio_before_speaker_fallback(self) -> None:
        stale_devices = [
            {"index": 5, "name": "CABLE In 16ch (VB-Audio Virtual Cable)", "hostapi": "MME", "default_samplerate": 44100},
            {"index": 10, "name": "Speakers (Realtek(R) Audio)", "hostapi": "MME", "default_samplerate": 44100},
        ]
        fresh_devices = [
            {"index": 5, "name": "CABLE In 16ch (VB-Audio Virtual Cable)", "hostapi": "MME", "default_samplerate": 44100},
            {"index": 9, "name": "Headphones (Realtek(R) Audio)", "hostapi": "MME", "default_samplerate": 44100},
            {"index": 10, "name": "Speakers (Realtek(R) Audio)", "hostapi": "MME", "default_samplerate": 44100},
        ]
        state = {"refreshed": 0}

        def _fake_refresh() -> bool:
            state["refreshed"] += 1
            return True

        with (
            patch.object(audio_router_mod, "get_output_devices", side_effect=lambda: fresh_devices if state["refreshed"] else stale_devices),
            patch.object(
                audio_router_mod,
                "_get_core_audio_default_output_names",
                return_value={
                    "console": "CABLE In 16ch (VB-Audio Virtual Cable)",
                    "communications": "Headphones (Realtek(R) Audio)",
                },
            ),
            patch.object(audio_router_mod, "_get_windows_preferred_output_name", return_value="CABLE In 16ch (VB-Audio Virtual"),
            patch.object(audio_router_mod, "refresh_portaudio_devices", side_effect=_fake_refresh),
            patch.object(config, "get", side_effect=lambda k, d=None: "" if k == "secondary_output_device" else d),
        ):
            self.assertEqual(audio_router_mod.resolve_secondary_output_targets(), [9])
            self.assertEqual(state["refreshed"], 1)

    def test_primary_index_resolved_after_hotplug_refresh_and_missing_cable_never_leaks_to_speakers(self) -> None:
        # Scenario 1: Headphones plugged in after startup -> PortAudio refresh shifts CABLE Input from index 7 to index 6,
        # while old index 7 becomes Speakers! Primary MUST resolve to index 6 (CABLE Input), NEVER old index 7 (Speakers).
        stale_devices = [
            {"index": 6, "name": "CABLE In 16ch (VB-Audio Virtual Cable)", "hostapi": "MME", "default_samplerate": 44100},
            {"index": 7, "name": "CABLE Input (VB-Audio Virtual Cable)", "hostapi": "MME", "default_samplerate": 44100},
            {"index": 8, "name": "Speakers (Realtek(R) Audio)", "hostapi": "MME", "default_samplerate": 44100},
        ]
        fresh_devices = [
            {"index": 6, "name": "CABLE Input (VB-Audio Virtual Cable)", "hostapi": "MME", "default_samplerate": 44100},
            {"index": 7, "name": "Speakers (Realtek(R) Audio)", "hostapi": "MME", "default_samplerate": 44100},
            {"index": 8, "name": "Headphones (Realtek(R) Audio)", "hostapi": "MME", "default_samplerate": 44100},
        ]
        state = {"refreshed": 0}

        def _fake_refresh() -> bool:
            state["refreshed"] += 1
            return True

        with (
            patch.object(audio_router_mod, "get_output_devices", side_effect=lambda: fresh_devices if state["refreshed"] else stale_devices),
            patch.object(
                audio_router_mod,
                "_get_core_audio_default_output_names",
                return_value={
                    "console": "CABLE In 16ch (VB-Audio Virtual Cable)",
                    "communications": "Headphones (Realtek(R) Audio)",
                },
            ),
            patch.object(audio_router_mod, "_get_windows_preferred_output_name", return_value="CABLE In 16ch"),
            patch.object(audio_router_mod, "refresh_portaudio_devices", side_effect=_fake_refresh),
            patch.object(config, "get", side_effect=lambda k, d=None: "" if k == "secondary_output_device" else d),
        ):
            prim_idx, sec_targets = audio_router_mod.resolve_playback_device_indices(
                "CABLE Input (VB-Audio Virtual Cable)",
                primary_off=False,
            )
            self.assertEqual(state["refreshed"], 1)
            self.assertEqual(prim_idx, 6)  # CABLE Input in fresh table, NOT shifted 7 (Speakers)!
            self.assertEqual(sec_targets, [8])  # Headphones

        # Scenario 2: Explicit primary_output_device (VB-Cable) is missing -> MUST return None, NEVER fall back to Speakers!
        no_cable_devices = [
            {"index": 8, "name": "Headphones (Realtek(R) Audio)", "hostapi": "MME", "default_samplerate": 44100},
            {"index": 10, "name": "Speakers (Realtek(R) Audio)", "hostapi": "MME", "default_samplerate": 44100},
        ]
        with (
            patch.object(audio_router_mod, "get_output_devices", return_value=no_cable_devices),
            patch.object(
                audio_router_mod,
                "_get_core_audio_default_output_names",
                return_value={"console": "Speakers (Realtek(R) Audio)"},
            ),
            patch.object(audio_router_mod, "refresh_portaudio_devices", return_value=False),
            patch.object(config, "get", side_effect=lambda k, d=None: "" if k == "secondary_output_device" else d),
        ):
            prim_idx2, sec_targets2 = audio_router_mod.resolve_playback_device_indices(
                "CABLE Input (VB-Audio Virtual Cable)",
                primary_off=False,
            )
            self.assertIsNone(prim_idx2)
            self.assertEqual(sec_targets2, [10])

    def test_get_output_devices_keeps_distinct_mme_prefixes_and_includes_wasapi_only_headphones(self) -> None:
        fake_hostapis = [{"name": "MME"}, {"name": "Windows WASAPI"}]
        fake_devs = [
            {"name": "Headphones (Realtek USB2.0 A", "hostapi": 0, "max_output_channels": 2, "default_samplerate": 44100},
            {"name": "Headphones (Realtek USB2.0 B", "hostapi": 0, "max_output_channels": 2, "default_samplerate": 44100},
            {"name": "Headphones (Realtek USB2.0 A Full)", "hostapi": 1, "max_output_channels": 2, "default_samplerate": 48000},
            {"name": "Headphones (Realtek USB2.0 B Full)", "hostapi": 1, "max_output_channels": 2, "default_samplerate": 48000},
            {"name": "Headphones (Canyon TWS-5)", "hostapi": 1, "max_output_channels": 2, "default_samplerate": 48000},
        ]
        with (
            patch.object(audio_router_mod.sd, "query_hostapis", return_value=fake_hostapis),
            patch.object(audio_router_mod.sd, "query_devices", return_value=fake_devs),
        ):
            out = audio_router_mod.get_output_devices()
            names = [str(d["name"]) for d in out]
            self.assertIn("Headphones (Realtek USB2.0 A Full)", names)
            self.assertIn("Headphones (Realtek USB2.0 B Full)", names)
            self.assertIn("Headphones (Canyon TWS-5)", names)

    def test_output_format_int32_preference_and_live_core_audio_sync(self) -> None:
        # 1. Setting audio_output_format="int32" opens int32 FIRST even when float32 is supported
        opened_dtypes: list[str] = []

        def _fake_stream(*, samplerate, channels, dtype, device, callback):
            opened_dtypes.append(dtype)
            return _FakeOutputStream(
                samplerate=samplerate, channels=channels, dtype=dtype, device=device, callback=callback
            )

        with (
            patch.object(config, "get", side_effect=lambda k, d=None: "int32" if k == "audio_output_format" else d),
            patch.object(audio_router_mod.sd, "query_devices", return_value={"name": "USB DAC", "default_samplerate": 48000, "max_output_channels": 2}),
            patch.object(audio_router_mod.sd, "OutputStream", side_effect=_fake_stream),
        ):
            player = _PersistentDevicePlayer(device_idx=4, preferred_sr=48000)
            try:
                self.assertEqual(player.dtype, "int32")
                self.assertEqual(opened_dtypes, ["int32"])
            finally:
                player.close()

        # 2. Smart conditional clean_audio_edges preserves smooth start (only 2ms ramp) vs sharp jump (10ms ramp)
        sr = 24000
        smooth = np.zeros(2400, dtype=np.float32)
        t_ac = np.arange(2400 - 48, dtype=np.float32) / float(sr)
        smooth[48:] = 0.4 * np.sin(2.0 * np.pi * 400.0 * t_ac).astype(np.float32)
        out_smooth = clean_audio_edges(smooth, sr, fade_in_ms=10.0, fade_out_ms=10.0, pad_tail_ms=0.0)
        # Sample 63 is 2.6ms into the buffer (past 2ms micro-fade, well inside 10ms fade) -> preserved unattenuated!
        self.assertAlmostEqual(float(out_smooth[63]), float(smooth[63]), places=3)

        abrupt = np.sin(np.linspace(0.5, 40.0, 2400, dtype=np.float32)) * 0.5  # starts with sharp jump (~0.24)
        out_abrupt = clean_audio_edges(abrupt, sr, fade_in_ms=10.0, fade_out_ms=10.0, pad_tail_ms=0.0)
        self.assertAlmostEqual(float(out_abrupt[0]), 0.0, places=5)
        self.assertLess(abs(float(out_abrupt[10])), abs(float(abrupt[10])))

        # 3. ensure_portaudio_synced_with_windows refreshes ONLY when Core Audio device list changes
        prev_cache = audio_router_mod._LAST_ACTIVE_CORE_AUDIO_OUTPUTS
        refreshes: list[int] = []
        try:
            audio_router_mod._LAST_ACTIVE_CORE_AUDIO_OUTPUTS = ("Speakers (Realtek(R) Audio)",)
            with (
                patch.object(audio_router_mod, "get_core_audio_active_output_names", return_value=("Speakers (Realtek(R) Audio)",)),
                patch.object(audio_router_mod, "refresh_portaudio_devices", side_effect=lambda: refreshes.append(1) or True),
            ):
                self.assertFalse(audio_router_mod.ensure_portaudio_synced_with_windows())
                self.assertEqual(len(refreshes), 0)

            with (
                patch.object(
                    audio_router_mod,
                    "get_core_audio_active_output_names",
                    return_value=("Headphones (Realtek(R) Audio)", "Speakers (Realtek(R) Audio)"),
                ),
                patch.object(audio_router_mod, "refresh_portaudio_devices", side_effect=lambda: refreshes.append(1) or True),
            ):
                self.assertTrue(audio_router_mod.ensure_portaudio_synced_with_windows())
                self.assertEqual(len(refreshes), 1)
        finally:
            audio_router_mod._LAST_ACTIVE_CORE_AUDIO_OUTPUTS = prev_cache

    def test_resolve_output_device_index_prefers_exact_over_prefix(self) -> None:
        fake_devices = [
            {"index": 1, "name": "Headphones (Realtek High Definition Audio) Pro", "hostapi": "MME", "default_samplerate": 48000},
            {"index": 2, "name": "Headphones (Realtek High Definition Audio)", "hostapi": "MME", "default_samplerate": 48000},
        ]
        with patch.object(audio_router_mod, "get_output_devices", return_value=fake_devices):
            self.assertEqual(audio_router_mod.resolve_output_device_index("Headphones (Realtek High Definition Audio)"), 2)

    def test_refresh_devices_closes_players_and_skips_when_play_lock_busy(self) -> None:
        router = audio_router_mod.DualAudioRouter()
        events: list[str] = []

        class _FakePlayer:
            def close(self, abort: bool = True) -> None:
                events.append("close_player")

        router._players[10] = _FakePlayer()  # type: ignore[assignment]
        with (
            patch.object(audio_router_mod.sd, "_terminate", side_effect=lambda: events.append("terminate"), create=True),
            patch.object(audio_router_mod.sd, "_initialize", side_effect=lambda: events.append("initialize"), create=True),
            patch.object(router, "sync_mic_passthrough", side_effect=lambda: events.append("sync_mic")),
        ):
            self.assertTrue(router.refresh_devices())
            self.assertEqual(events, ["close_player", "terminate", "initialize", "sync_mic"])
            self.assertEqual(router._players, {})

            events.clear()
            router._play_lock.acquire()
            try:
                self.assertFalse(router.refresh_devices())
                self.assertEqual(events, [])
            finally:
                router._play_lock.release()

    def test_stalled_headphones_unblocks_and_hotswaps_to_speakers_mid_playback(self) -> None:
        import threading

        class _DrainControllableStream(_FakeOutputStream):
            def __init__(self, *, stall_after_pulls: int | None = None, **kwargs) -> None:
                super().__init__(**kwargs)
                self.stall_after_pulls = stall_after_pulls
                self.pulls = 0
                self.aborted = False
                self.received_frames = 0

            def abort(self) -> None:
                self.active = False
                self.aborted = True

            def start(self) -> None:
                super().start()

                def _pump() -> None:
                    while self.active:
                        if self.stall_after_pulls is not None and self.pulls >= self.stall_after_pulls:
                            break
                        block = self.pull(512)
                        self.pulls += 1
                        if float(np.max(np.abs(block))) > 0.0:
                            self.received_frames += len(block)
                        time.sleep(0.005)

                threading.Thread(target=_pump, daemon=True).start()

        streams_by_dev: dict[int, list[_DrainControllableStream]] = {}
        dev_state = {"default": "Headphones (Test)"}
        fake_devices = [
            {"index": 8, "name": "Headphones (Test)", "hostapi": "MME", "default_samplerate": 48000},
            {"index": 10, "name": "Speakers (Test)", "hostapi": "MME", "default_samplerate": 48000},
        ]

        def _fake_stream(*, samplerate, channels, dtype, device, callback):
            stall = 2 if device == 8 else None
            if device == 8 and len(streams_by_dev.get(8, [])) == 0:
                # Simulate unplug right after headphones start draining
                dev_state["default"] = "Speakers (Test)"
            st = _DrainControllableStream(
                stall_after_pulls=stall,
                samplerate=samplerate,
                channels=channels,
                dtype=dtype,
                device=device,
                callback=callback,
            )
            streams_by_dev.setdefault(int(device), []).append(st)
            return st

        router = DualAudioRouter()
        overrides = {
            "primary_output_device": "__off__",
            "secondary_output_device": "",
            "secondary_enabled": True,
            "secondary_volume": 100,
            "vban_enabled": False,
        }
        audio_router_mod._LAST_CORE_AUDIO_SNAPSHOT = None
        try:
            with (
                patch.object(config, "get", side_effect=lambda k, d=None: overrides.get(k, d)),
                patch.object(audio_router_mod, "get_output_devices", return_value=fake_devices),
                patch.object(
                    audio_router_mod,
                    "_get_core_audio_default_output_names",
                    side_effect=lambda: {"console": dev_state["default"], "communications": dev_state["default"]},
                ),
                patch.object(audio_router_mod, "_get_windows_preferred_output_name", side_effect=lambda: dev_state["default"]),
                patch.object(audio_router_mod.sd, "query_devices", return_value={"default_samplerate": 48000, "max_output_channels": 2}),
                patch.object(audio_router_mod.sd, "OutputStream", side_effect=_fake_stream),
            ):
                t0 = time.monotonic()
                t_arr = np.arange(24000, dtype=np.float32) / 48000.0
                samples = (0.3 * np.sin(2.0 * np.pi * 440.0 * t_arr)).astype(np.float32)
                router.play_samples(samples, 48000, monitor_only=True)
                elapsed = time.monotonic() - t0
                self.assertLess(elapsed, 3.0)
                self.assertIn(8, streams_by_dev)
                self.assertTrue(streams_by_dev[8][0].aborted)
                self.assertTrue(streams_by_dev[8][0].closed)
                self.assertIn(10, streams_by_dev)
                self.assertGreater(streams_by_dev[10][0].received_frames, 0)
        finally:
            router.shutdown()

    def test_live_switch_to_headphones_mid_playback_keeps_primary_intact(self) -> None:
        import threading

        class _LiveSwitchStream(_FakeOutputStream):
            def __init__(self, *, on_pull=None, **kwargs) -> None:
                super().__init__(**kwargs)
                self.on_pull = on_pull
                self.received_frames = 0

            def start(self) -> None:
                super().start()

                def _pump() -> None:
                    while self.active:
                        block = self.pull(512)
                        if float(np.max(np.abs(block))) > 0.0:
                            self.received_frames += len(block)
                        if self.on_pull is not None:
                            self.on_pull(self)
                        time.sleep(0.005)

                threading.Thread(target=_pump, daemon=True).start()

        streams_by_dev: dict[int, list[_LiveSwitchStream]] = {}
        dev_state = {"sec": "Speakers (Test)"}
        fake_devices = [
            {"index": 5, "name": "CABLE Input (VB-Audio Virtual Cable)", "hostapi": "MME", "default_samplerate": 48000},
            {"index": 8, "name": "Headphones (Test)", "hostapi": "MME", "default_samplerate": 48000},
            {"index": 10, "name": "Speakers (Test)", "hostapi": "MME", "default_samplerate": 48000},
        ]

        def _fake_stream(*, samplerate, channels, dtype, device, callback):
            def _on_pull(st_obj: _LiveSwitchStream) -> None:
                if device == 5 and st_obj.received_frames >= 4096:
                    dev_state["sec"] = "Headphones (Test)"

            st = _LiveSwitchStream(
                on_pull=_on_pull,
                samplerate=samplerate,
                channels=channels,
                dtype=dtype,
                device=device,
                callback=callback,
            )
            streams_by_dev.setdefault(int(device), []).append(st)
            return st

        router = DualAudioRouter()
        overrides = {
            "primary_output_device": "CABLE Input (VB-Audio Virtual Cable)",
            "primary_volume": 100,
            "secondary_output_device": "",
            "secondary_enabled": True,
            "secondary_volume": 100,
            "vban_enabled": False,
        }
        audio_router_mod._LAST_CORE_AUDIO_SNAPSHOT = None
        try:
            with (
                patch.object(config, "get", side_effect=lambda k, d=None: overrides.get(k, d)),
                patch.object(audio_router_mod, "get_output_devices", return_value=fake_devices),
                patch.object(
                    audio_router_mod,
                    "_get_core_audio_default_output_names",
                    side_effect=lambda: {"console": "CABLE Input (VB-Audio Virtual Cable)", "communications": dev_state["sec"]},
                ),
                patch.object(audio_router_mod, "_get_windows_preferred_output_name", return_value="CABLE Input (VB-Audio Virtual Cable)"),
                patch.object(audio_router_mod.sd, "query_devices", return_value={"default_samplerate": 48000, "max_output_channels": 2}),
                patch.object(audio_router_mod.sd, "OutputStream", side_effect=_fake_stream),
            ):
                t_arr = np.arange(48000, dtype=np.float32) / 48000.0
                samples = (0.3 * np.sin(2.0 * np.pi * 440.0 * t_arr)).astype(np.float32)  # 1.0s at 48kHz
                router.play_samples(samples, 48000)
                self.assertEqual(len(streams_by_dev.get(5, [])), 1)
                self.assertIn(10, streams_by_dev)
                self.assertIn(8, streams_by_dev)
                self.assertGreater(streams_by_dev[8][0].received_frames, 0)
                self.assertLess(streams_by_dev[8][0].received_frames, 48000)
        finally:
            router.shutdown()

    def test_stalled_device_exits_promptly_without_hanging(self) -> None:
        # _FakeOutputStream has no pump thread, so callbacks never fire -> must exit via stall guard
        router = DualAudioRouter()
        overrides = {
            "primary_output_device": "__off__",
            "secondary_output_device": "Dead Device",
            "secondary_enabled": True,
            "secondary_volume": 100,
            "vban_enabled": False,
        }
        fake_devices = [{"index": 4, "name": "Dead Device", "hostapi": "MME", "default_samplerate": 48000}]
        try:
            with (
                patch.object(config, "get", side_effect=lambda k, d=None: overrides.get(k, d)),
                patch.object(audio_router_mod, "get_output_devices", return_value=fake_devices),
                patch.object(audio_router_mod, "_get_core_audio_default_output_names", return_value={"console": "Dead Device"}),
                patch.object(audio_router_mod, "_get_windows_preferred_output_name", return_value="Dead Device"),
                patch.object(audio_router_mod, "refresh_portaudio_devices", return_value=False),
                patch.object(audio_router_mod.sd, "query_devices", return_value={"default_samplerate": 48000, "max_output_channels": 2}),
                patch.object(
                    audio_router_mod.sd,
                    "OutputStream",
                    side_effect=lambda **kw: _FakeOutputStream(**kw),
                ),
            ):
                t0 = time.monotonic()
                router.play_samples(np.full(24000, 0.2, dtype=np.float32), 48000, monitor_only=True)
                self.assertLess(time.monotonic() - t0, 3.0)
        finally:
            router.shutdown()

    def test_feedback_beep_volume_scales_both_start_and_cancel_cues(self) -> None:
        captured: list[tuple[str, float]] = []
        router = DualAudioRouter()
        state = {"sound_feedback": True, "cancel_sound_enabled": True, "sound_feedback_volume": 50}

        def _fake_start(wave, sr, dev_idx, allow_speaker_fallback=True):
            captured.append(("played", float(np.max(np.abs(wave)))))
            return None

        with (
            patch.object(config, "get", side_effect=lambda k, d=None: state.get(k, d)),
            patch.object(audio_router_mod, "resolve_secondary_output_targets", return_value=[1]),
            patch.object(router, "_start_on_single_device", side_effect=_fake_start),
        ):
            router.play_feedback_beep("start")
            time.sleep(0.08)
            router.play_feedback_beep("cancel")
            time.sleep(0.08)
            self.assertEqual(len(captured), 2)
            self.assertAlmostEqual(captured[0][1], 0.115 * 0.5, delta=0.005)
            self.assertAlmostEqual(captured[1][1], 0.105 * 0.5, delta=0.005)

            captured.clear()
            state["sound_feedback_volume"] = 0
            router.play_feedback_beep("start")
            router.play_feedback_beep("cancel")
            time.sleep(0.05)
            self.assertEqual(captured, [])

    def test_play_stream_plays_chunks_incrementally_and_supports_instant_stop(self) -> None:
        import threading

        streams: list[_FakeOutputStream] = []

        def _make_stream(**kw):
            st = _FakeOutputStream(**kw)
            streams.append(st)
            return st

        router = DualAudioRouter()
        overrides = {
            "primary_output_device": "__off__",
            "secondary_output_device": "Headphones",
            "secondary_enabled": True,
            "secondary_volume": 100,
            "vban_enabled": False,
        }
        fake_devices = [{"index": 7, "name": "Headphones", "hostapi": "MME", "default_samplerate": 24000}]
        allow_second = threading.Event()
        first_seen = threading.Event()
        gen_done_sizes: list[int] = []

        c1 = np.full(1200, 0.25, dtype=np.float32)
        c2 = np.full(1200, -0.25, dtype=np.float32)

        def _chunk_gen():
            yield c1, 24000
            allow_second.wait(timeout=2.0)
            yield c2, 24000

        pulled_blocks: list[np.ndarray] = []
        stop_pump = threading.Event()

        def _pump():
            while not stop_pump.is_set():
                if streams and streams[-1].active:
                    pulled_blocks.append(streams[-1].pull(400))
                time.sleep(0.005)

        pump_thread = threading.Thread(target=_pump, daemon=True)
        pump_thread.start()
        try:
            with (
                patch.object(config, "get", side_effect=lambda k, d=None: overrides.get(k, d)),
                patch.object(audio_router_mod, "get_output_devices", return_value=fake_devices),
                patch.object(audio_router_mod.sd, "query_devices", return_value={"default_samplerate": 24000, "max_output_channels": 2}),
                patch.object(audio_router_mod.sd, "OutputStream", side_effect=_make_stream),
            ):
                def _on_first(samples, sr, lat_ms):
                    self.assertEqual(sr, 24000)
                    self.assertEqual(len(samples), 1200)
                    first_seen.set()
                    allow_second.set()

                full, sr = router.play_stream(
                    _chunk_gen(),
                    monitor_only=True,
                    on_first_chunk=_on_first,
                    on_generation_done=lambda arr, s: gen_done_sizes.append(len(arr)),
                )
                self.assertTrue(first_seen.is_set())
                self.assertEqual(sr, 24000)
                self.assertEqual(len(full), 2400)
                self.assertEqual(gen_done_sizes, [2400])
                played = np.concatenate(pulled_blocks, axis=0)
                self.assertGreater(float(np.max(played[:, 0])), 0.2)
                self.assertLess(float(np.min(played[:, 0])), -0.2)

                # Instant stop during waiting for second chunk exits promptly
                hang_release = threading.Event()

                def _slow_gen():
                    yield c1, 24000
                    hang_release.wait(timeout=5.0)
                    yield c2, 24000

                def _stop_soon(samples, sr, lat_ms):
                    threading.Timer(0.03, router.stop).start()

                t0 = time.monotonic()
                router.play_stream(_slow_gen(), monitor_only=True, on_first_chunk=_stop_soon)
                self.assertLess(time.monotonic() - t0, 1.5)
                hang_release.set()
        finally:
            stop_pump.set()
            pump_thread.join(timeout=1.0)
            router.shutdown()

    def test_empty_primary_defaults_to_cable_input_or_off_never_windows_default(self) -> None:
        with_cable = [
            {"index": 6, "name": "CABLE Input (VB-Audio Virtual Cable)", "hostapi": "MME", "default_samplerate": 44100},
            {"index": 10, "name": "Speakers (Realtek(R) Audio)", "hostapi": "MME", "default_samplerate": 44100},
        ]
        without_cable = [
            {"index": 10, "name": "Speakers (Realtek(R) Audio)", "hostapi": "MME", "default_samplerate": 44100},
        ]
        with (
            patch.object(audio_router_mod, "ensure_portaudio_synced_with_windows", return_value=False),
            patch.object(audio_router_mod, "get_output_devices", return_value=with_cable),
            patch.object(audio_router_mod, "resolve_secondary_output_targets", return_value=[10]),
            patch.object(audio_router_mod, "refresh_portaudio_devices", return_value=False),
        ):
            p_idx, s_idxs = audio_router_mod.resolve_playback_device_indices("", primary_off=False)
            self.assertEqual(p_idx, 6)
            self.assertEqual(s_idxs, [10])

        with (
            patch.object(audio_router_mod, "ensure_portaudio_synced_with_windows", return_value=False),
            patch.object(audio_router_mod, "get_output_devices", return_value=without_cable),
            patch.object(audio_router_mod, "resolve_secondary_output_targets", return_value=[10]),
            patch.object(audio_router_mod, "refresh_portaudio_devices", return_value=False),
        ):
            p_idx2, s_idxs2 = audio_router_mod.resolve_playback_device_indices("", primary_off=False)
            self.assertIsNone(p_idx2)
            self.assertEqual(s_idxs2, [10])

    def test_primary_dev_combo_omits_windows_default_and_falls_back_to_off(self) -> None:
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QLabel, QPushButton, QSlider
        import ui.main_window as mw_mod

        _app = QApplication.instance() or QApplication([])

        def _make_dummy():
            d = type("DummyMW", (), {})()
            d.install_vb_btn = QPushButton()
            d.vbcable_status_lbl = QLabel()
            d.primary_dev_combo = QComboBox()
            d.secondary_dev_combo = QComboBox()
            d.mic_dev_combo = QComboBox()
            d.prim_vol_slider = QSlider()
            d.chk_mic_passthrough = QCheckBox()
            d.mic_vol_slider = QSlider()
            return d

        devs_with_cable = [
            {"index": 6, "name": "CABLE Input (VB-Audio Virtual Cable)", "hostapi": "MME", "default_samplerate": 44100},
            {"index": 10, "name": "Speakers (Realtek(R) Audio)", "hostapi": "MME", "default_samplerate": 44100},
        ]
        devs_no_cable = [
            {"index": 10, "name": "Speakers (Realtek(R) Audio)", "hostapi": "MME", "default_samplerate": 44100},
        ]

        # 1. With VB-Cable present and default config -> selects CABLE Input, no "" item in primary_dev_combo
        cfg_store = {"primary_output_device": "CABLE Input (VB-Audio Virtual Cable)"}
        dummy1 = _make_dummy()
        with (
            patch.object(mw_mod, "get_core_audio_active_all_device_names", return_value=((), ())),
            patch.object(mw_mod, "check_vbcable_installed", return_value=(True, "CABLE Input (VB-Audio Virtual Cable)")),
            patch.object(mw_mod, "get_output_devices", return_value=devs_with_cable),
            patch.object(mw_mod, "get_input_devices", return_value=[]),
            patch.object(mw_mod.config, "get", side_effect=lambda k, d=None: cfg_store.get(k, d)),
            patch.object(mw_mod.config, "set", side_effect=lambda k, v: cfg_store.__setitem__(k, v)),
            patch.object(mw_mod.audio_router, "sync_mic_passthrough"),
        ):
            mw_mod.MainWindow._populate_audio_devices(dummy1, refresh_backend=False)
            all_data = [dummy1.primary_dev_combo.itemData(i) for i in range(dummy1.primary_dev_combo.count())]
            self.assertNotIn("", all_data)
            self.assertEqual(dummy1.primary_dev_combo.currentData(), "CABLE Input (VB-Audio Virtual Cable)")

        # 2. Without VB-Cable -> falls back to "__off__", NOT Windows default
        cfg_store2 = {"primary_output_device": "CABLE Input (VB-Audio Virtual Cable)"}
        dummy2 = _make_dummy()
        with (
            patch.object(mw_mod, "get_core_audio_active_all_device_names", return_value=((), ())),
            patch.object(mw_mod, "check_vbcable_installed", return_value=(False, None)),
            patch.object(mw_mod, "get_output_devices", return_value=devs_no_cable),
            patch.object(mw_mod, "get_input_devices", return_value=[]),
            patch.object(mw_mod.config, "get", side_effect=lambda k, d=None: cfg_store2.get(k, d)),
            patch.object(mw_mod.config, "set", side_effect=lambda k, v: cfg_store2.__setitem__(k, v)),
            patch.object(mw_mod.audio_router, "sync_mic_passthrough"),
        ):
            mw_mod.MainWindow._populate_audio_devices(dummy2, refresh_backend=False)
            self.assertEqual(dummy2.primary_dev_combo.currentData(), "__off__")
            self.assertEqual(cfg_store2["primary_output_device"], "__off__")


if __name__ == "__main__":
    unittest.main()
