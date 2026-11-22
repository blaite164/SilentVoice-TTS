"""Unit and UDP loopback integration tests for SilentVoice VBAN subsystem."""

from __future__ import annotations

import socket
import struct
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from core.audio_router import DualAudioRouter
from core.config import config
from core.vban import (
    CHANNELS,
    FRAMES_PER_PACKET,
    HEADER_SIZE,
    PING0_PACKET_SIZE,
    RATE,
    SAMPLE_RATE_INDEX,
    VbanHealthMonitor,
    VbanOutput,
    decode_header,
    decode_ping0_reply,
    encode_header,
    encode_ping0_request,
    ping_receiver,
    prepare_stereo_48k,
)


def _build_receptor_ping0_reply(request_id_echo: int = 0) -> bytes:
    pkt = bytearray(PING0_PACKET_SIZE)
    pkt[0:4] = b"VBAN"
    pkt[4] = 0x60
    pkt[5] = 0x80
    pkt[6] = 0x00
    pkt[7] = 0x00
    pkt[8:20] = b"VBAN Service"
    pkt[24:28] = (request_id_echo & 0xFFFFFFFF).to_bytes(4, "little")
    payload = pkt[HEADER_SIZE:]
    payload[0:4] = (32).to_bytes(4, "little")
    payload[4:8] = (66305).to_bytes(4, "little")
    payload[28:32] = bytes([1, 0, 0, 5])
    app = b"VBAN-Receptor"
    payload[292 : 292 + len(app)] = app
    pkt[HEADER_SIZE:] = payload
    return bytes(pkt)


class VbanProtocolTests(unittest.TestCase):
    def test_header_roundtrip_48k_stereo(self) -> None:
        hdr = encode_header(b"Stream1", 200, 123456)
        self.assertEqual(len(hdr), HEADER_SIZE)
        decoded = decode_header(hdr)
        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertEqual(decoded.sample_rate_index, SAMPLE_RATE_INDEX)
        self.assertEqual(decoded.samples_per_packet, 200)
        self.assertEqual(decoded.channels, CHANNELS)
        self.assertEqual(decoded.bit_format, 1)
        self.assertEqual(decoded.stream_name, "Stream1")
        self.assertEqual(decoded.nu_frame, 123456)

    def test_prepare_stereo_preserves_stereo_and_resamples(self) -> None:
        mono = np.array([0.25, -0.5, 0.75, float("nan")], dtype=np.float32)
        out_mono = prepare_stereo_48k(mono, RATE)
        self.assertEqual(out_mono.shape, (4, 2))
        self.assertTrue(np.allclose(out_mono[:, 0], out_mono[:, 1]))
        self.assertEqual(out_mono[3, 0], 0.0)

        stereo = np.array([[0.1, -0.1], [0.2, -0.2]], dtype=np.float32)
        out_stereo = prepare_stereo_48k(stereo, RATE)
        self.assertEqual(out_stereo.shape, (2, 2))
        self.assertAlmostEqual(float(out_stereo[0, 0]), 0.1, places=5)
        self.assertAlmostEqual(float(out_stereo[0, 1]), -0.1, places=5)

        sine_24k = np.sin(2.0 * np.pi * 440.0 * np.arange(2400, dtype=np.float32) / 24_000.0)
        resampled = prepare_stereo_48k(sine_24k, 24_000)
        self.assertEqual(resampled.shape, (4800, 2))


class VbanOutputLoopbackTests(unittest.TestCase):
    def _bind_listener(self, timeout: float = 0.5) -> tuple[socket.socket, int]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        sock.settimeout(timeout)
        return sock, sock.getsockname()[1]

    def test_sequential_packets_and_persistent_nu_frame(self) -> None:
        listener, port = self._bind_listener()
        output = VbanOutput()
        try:
            pcm1 = np.full(300, 0.5, dtype=np.float32)
            h1 = output.start(pcm1, RATE, "127.0.0.1", port, "LOOP", kind="speech")
            packets_1 = []
            for _ in range(2):
                data, _ = listener.recvfrom(2048)
                packets_1.append((len(data), decode_header(data)))
            self.assertEqual([p[0] for p in packets_1], [HEADER_SIZE + 256 * 4, HEADER_SIZE + 44 * 4])
            self.assertEqual([p[1].nu_frame for p in packets_1], [1, 2])
            self.assertTrue(h1.wait(timeout=2.0))
            self.assertEqual(h1.status, "finished")

            # Next phrase on the same stream must continue nu_frame instead of resetting to 0/1.
            pcm2 = np.full(300, -0.25, dtype=np.float32)
            h2 = output.start(pcm2, RATE, "127.0.0.1", port, "LOOP", kind="speech")
            packets_2 = []
            for _ in range(2):
                data, _ = listener.recvfrom(2048)
                packets_2.append(decode_header(data))
            self.assertEqual([p.nu_frame for p in packets_2], [3, 4])
            self.assertTrue(h2.wait(timeout=2.0))
            self.assertEqual(h2.status, "finished")
        finally:
            output.shutdown()
            listener.close()

    def test_stop_is_prompt_and_halts_stream(self) -> None:
        listener, port = self._bind_listener(timeout=0.25)
        output = VbanOutput()
        try:
            pcm = np.full(RATE, 0.4, dtype=np.float32)  # 1 second
            handle = output.start(pcm, RATE, "127.0.0.1", port, "STOP")
            data, _ = listener.recvfrom(2048)
            self.assertIsNotNone(decode_header(data))

            t0 = time.perf_counter()
            handle.stop()
            self.assertTrue(handle.wait(timeout=0.5))
            elapsed = time.perf_counter() - t0
            self.assertLess(elapsed, 0.2)
            self.assertEqual(handle.status, "stopped")

            # Drain any in-flight packet, then ensure silence.
            for _ in range(8):
                try:
                    listener.recvfrom(2048)
                except socket.timeout:
                    break
            with self.assertRaises(socket.timeout):
                listener.recvfrom(2048)
        finally:
            output.shutdown()
            listener.close()

    def test_mixing_and_test_conflict(self) -> None:
        listener, port = self._bind_listener(timeout=0.5)
        output = VbanOutput()
        try:
            speech = np.full(RATE // 4, 0.25, dtype=np.float32)
            cue = np.full(RATE // 4, 0.25, dtype=np.float32)
            h_speech = output.start(speech, RATE, "127.0.0.1", port, "MIX", kind="speech")
            h_cue = output.start(cue, RATE, "127.0.0.1", port, "MIX", kind="cue")

            # Wait until speech starts playing, then verify test on occupied stream is rejected.
            deadline = time.monotonic() + 1.0
            while h_speech.status == "pending" and time.monotonic() < deadline:
                time.sleep(0.005)
            with self.assertRaises(RuntimeError):
                output.start(speech, RATE, "127.0.0.1", port, "MIX", kind="test", repeat=True)

            data, _ = listener.recvfrom(2048)
            samples = np.frombuffer(data[HEADER_SIZE:], dtype="<i2")
            # Either first packet already mixes both (~0.5 -> ~16384) or second packet does.
            mixed_seen = np.any(samples > 12000)
            if not mixed_seen:
                data2, _ = listener.recvfrom(2048)
                samples2 = np.frombuffer(data2[HEADER_SIZE:], dtype="<i2")
                mixed_seen = np.any(samples2 > 12000)
            self.assertTrue(mixed_seen)

            self.assertTrue(h_speech.wait(timeout=2.0))
            self.assertTrue(h_cue.wait(timeout=2.0))
        finally:
            output.shutdown()
            listener.close()


class VbanPing0Tests(unittest.TestCase):
    def test_ping0_encode_decode(self) -> None:
        req = encode_ping0_request(7)
        self.assertEqual(len(req), PING0_PACKET_SIZE)
        self.assertEqual(req[:4], b"VBAN")
        self.assertEqual(req[4], 0x60)

        reply_bytes = _build_receptor_ping0_reply(0)
        parsed = decode_ping0_reply(reply_bytes, "127.0.0.1", 6980, ("127.0.0.1", 6980))
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.application_name, "VBAN-Receptor")
        self.assertEqual(parsed.version_str, "1.0.0.5")

    def test_ping_receiver_and_monitor(self) -> None:
        responder = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        responder.bind(("127.0.0.1", 0))
        responder_port = responder.getsockname()[1]

        # Pick an ephemeral reply port for test isolation.
        probe_bind = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe_bind.bind(("127.0.0.1", 0))
        reply_port = probe_bind.getsockname()[1]
        probe_bind.close()

        stop_flag = threading.Event()

        def _serve() -> None:
            responder.settimeout(0.05)
            while not stop_flag.is_set():
                try:
                    data, _ = responder.recvfrom(2048)
                except socket.timeout:
                    continue
                if len(data) == PING0_PACKET_SIZE and data[:4] == b"VBAN" and data[4] == 0x60:
                    responder.sendto(_build_receptor_ping0_reply(0), ("127.0.0.1", reply_port))

        t = threading.Thread(target=_serve, daemon=True)
        t.start()
        monitor = VbanHealthMonitor(reply_port=reply_port, probe_timeout=0.3)
        try:
            status, reply, _ = ping_receiver("127.0.0.1", port=responder_port, timeout=0.5, reply_port=reply_port)
            self.assertEqual(status, "alive")
            assert reply is not None
            self.assertEqual(reply.application_name, "VBAN-Receptor")

            events: list[tuple[str, str, str]] = []
            evt_ready = threading.Event()

            def _on_change(st: str, rsn: str, dtl: str) -> None:
                events.append((st, rsn, dtl))
                if st == "alive":
                    evt_ready.set()

            monitor.set_callback(_on_change)
            monitor.set_target("127.0.0.1", port=responder_port, interval_secs=0.2)
            self.assertTrue(evt_ready.wait(timeout=2.0))
            self.assertEqual(monitor.state[0], "alive")

            # Verify local port conflict returns local_socket cleanly.
            occupier = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            occupier.bind(("0.0.0.0", 0))
            busy_port = occupier.getsockname()[1]
            try:
                st_busy, _, _ = ping_receiver("127.0.0.1", port=responder_port, timeout=0.1, reply_port=busy_port)
                self.assertEqual(st_busy, "local_socket")
            finally:
                occupier.close()
        finally:
            stop_flag.set()
            monitor.shutdown()
            t.join(timeout=1.0)
            responder.close()


class AudioRouterVbanTests(unittest.TestCase):
    def test_router_vban_only_playback_and_stop(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listener.bind(("127.0.0.1", 0))
        listener.settimeout(0.5)
        port = listener.getsockname()[1]

        router = DualAudioRouter()
        overrides = {
            "primary_output_device": "__off__",
            "secondary_enabled": False,
            "vban_enabled": True,
            "vban_address": "127.0.0.1",
            "vban_port": port,
            "vban_stream_name": "RouterTest",
            "vban_volume": 50,
        }

        def _fake_get(key: str, default=None):
            return overrides.get(key, default)

        try:
            with patch.object(config, "get", side_effect=_fake_get):
                long_audio = np.full(RATE, 0.8, dtype=np.float32)  # 1 second
                play_thread = threading.Thread(target=router.play_samples, args=(long_audio, RATE), daemon=True)
                play_thread.start()

                data, _ = listener.recvfrom(2048)
                hdr = decode_header(data)
                self.assertIsNotNone(hdr)
                assert hdr is not None
                self.assertEqual(hdr.stream_name, "RouterTest")

                # Stop must abort VBAN playback immediately instead of blocking for 1s.
                t0 = time.perf_counter()
                router.stop()
                play_thread.join(timeout=0.5)
                self.assertFalse(play_thread.is_alive())
                self.assertLess(time.perf_counter() - t0, 0.35)
        finally:
            router.stop_vban()
            listener.close()

    def test_viewer_chat_routing_skips_primary_and_overrides_vban_stream(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listener.bind(("127.0.0.1", 0))
        listener.settimeout(0.25)
        port = listener.getsockname()[1]

        router = DualAudioRouter()
        overrides = {
            "primary_output_device": "CABLE Input (VB-Audio Virtual Cable)",
            "primary_volume": 100,
            "secondary_enabled": True,
            "secondary_output_device": "",
            "secondary_volume": 80,
            "vban_enabled": True,
            "vban_address": "127.0.0.1",
            "vban_port": port,
            "vban_stream_name": "SilentVoice",
            "vban_volume": 50,
        }

        def _fake_get(key: str, default=None):
            return overrides.get(key, default)

        started_devices: list[tuple[object, bool]] = []
        with patch.object(config, "get", side_effect=_fake_get), patch(
            "core.audio_router.resolve_output_device_index", return_value=11
        ), patch("core.audio_router.resolve_secondary_output_targets", return_value=[22]), patch.object(
            router,
            "_start_on_single_device",
            side_effect=lambda _samples, _sr, dev_idx, allow_fb=True: started_devices.append((dev_idx, allow_fb)) or None,
        ):
            try:
                short = np.full(int(RATE * 0.05), 0.2, dtype=np.float32)

                # 1) Viewer chat with send_primary=False and separate VBAN stream "SilentChat"
                router.play_samples(short, RATE, send_primary=False, vban_stream_override="SilentChat")
                self.assertEqual(started_devices, [(22, True)])  # skipped primary=11 (VB-Cable)
                data, _ = listener.recvfrom(2048)
                hdr = decode_header(data)
                self.assertIsNotNone(hdr)
                assert hdr is not None
                self.assertEqual(hdr.stream_name, "SilentChat")

                # Drain any remaining packets from SilentChat
                while True:
                    try:
                        listener.recvfrom(2048)
                    except socket.timeout:
                        break

                # 2) Viewer chat with VBAN disabled (vban_stream_override="") and send_primary=False
                started_devices.clear()
                router.play_samples(short, RATE, send_primary=False, vban_stream_override="")
                self.assertEqual(started_devices, [(22, True)])
                with self.assertRaises(socket.timeout):
                    listener.recvfrom(2048)
            finally:
                router.stop_vban()
                listener.close()

    def test_twitch_chat_vban_ui_has_two_stream_options_with_checkbox(self) -> None:
        from PySide6.QtWidgets import QApplication
        from ui.main_window import MainWindow

        _app = QApplication.instance() or QApplication([])
        win = MainWindow.__new__(MainWindow)
        win._twitch_chat_paused = False
        win._refresh_twitch_chat_voice_selectors = lambda: None
        win._rebuild_twitch_voice_rules = lambda: None
        win._save_twitch_settings = lambda _src="": win._update_twitch_allowed_row()
        with patch(
            "ui.main_window.config.get",
            side_effect=lambda k, d=None: {"twitch_chat_send_vban": False, "twitch_chat_vban_mode": "same"}.get(k, d),
        ):
            page = win._build_tts_tab_twitch_chat()
            items = [win.twitch_chat_vban_combo.itemData(i) for i in range(win.twitch_chat_vban_combo.count())]
            self.assertEqual(items, ["same", "separate"])
            self.assertFalse(win.twitch_chat_vban_chk.isChecked())
            self.assertTrue(win.twitch_chat_vban_combo.isHidden())
            self.assertTrue(win.twitch_chat_vban_stream_edit.isHidden())

            win.twitch_chat_vban_chk.setChecked(True)
            self.assertFalse(win.twitch_chat_vban_combo.isHidden())
            self.assertTrue(win.twitch_chat_vban_stream_edit.isHidden())

            win.twitch_chat_vban_combo.setCurrentIndex(1)
            self.assertFalse(win.twitch_chat_vban_stream_edit.isHidden())


if __name__ == "__main__":
    unittest.main()
