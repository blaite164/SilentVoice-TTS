"""Unit and loopback TCP IRC tests for Twitch chat integration."""

from __future__ import annotations

import socket
import threading
import time
import unittest

from core.twitch_chat import (
    TwitchChatClient,
    TwitchChatMessage,
    evaluate_chat_message,
    format_and_filter_chat_message,
    normalize_twitch_name,
    normalize_twitch_token,
    parse_moderation_line,
    parse_own_privmsg,
    parse_privmsg,
    sanitize_irc_text,
    split_twitch_message,
)


class _MockIrcServer:
    def __init__(self, fail_auth: bool = False) -> None:
        self.fail_auth = fail_auth
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(5)
        self.port = self.sock.getsockname()[1]
        self.stop_flag = threading.Event()
        self.received_lines: list[str] = []
        self.privmsgs: list[str] = []
        self.pongs: list[str] = []
        self._clients: list[socket.socket] = []
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _accept_loop(self) -> None:
        self.sock.settimeout(0.05)
        while not self.stop_flag.is_set():
            try:
                conn, _ = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self._clients.append(conn)
            threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()

    def _handle_client(self, conn: socket.socket) -> None:
        conn.settimeout(0.05)
        buf = ""
        nick = "user"
        try:
            while not self.stop_flag.is_set():
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk.decode("utf-8", errors="replace")
                while "\r\n" in buf:
                    line, buf = buf.split("\r\n", 1)
                    self.received_lines.append(line)
                    if line.startswith("NICK "):
                        nick = line.split(" ", 1)[1].strip()
                        if self.fail_auth:
                            conn.sendall(b":tmi.twitch.tv NOTICE * :Login authentication failed\r\n")
                        else:
                            conn.sendall(f":tmi.twitch.tv 001 {nick} :Welcome, GLHF!\r\n".encode("utf-8"))
                    elif line.startswith("JOIN "):
                        chan = line.split(" ", 1)[1].strip()
                        conn.sendall(f":{nick}!{nick}@{nick}.tmi.twitch.tv JOIN {chan}\r\n".encode("utf-8"))
                        conn.sendall(f":tmi.twitch.tv 366 {nick} {chan} :End of /NAMES list\r\n".encode("utf-8"))
                        conn.sendall(b"PING :tmi.twitch.tv\r\n")
                    elif line.startswith("PONG "):
                        self.pongs.append(line)
                    elif line.startswith("PRIVMSG "):
                        self.privmsgs.append(line)
                    elif line.startswith("QUIT"):
                        return
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def close(self) -> None:
        self.stop_flag.set()
        try:
            self.sock.close()
        except OSError:
            pass
        for c in self._clients:
            try:
                c.close()
            except OSError:
                pass
        self._thread.join(timeout=1.0)

    def broadcast_raw(self, raw_line: str) -> None:
        data = (raw_line.rstrip("\r\n") + "\r\n").encode("utf-8")
        for c in list(self._clients):
            try:
                c.sendall(data)
            except OSError:
                pass


class TwitchHelpersTests(unittest.TestCase):
    def test_normalization_and_sanitization(self) -> None:
        self.assertEqual(normalize_twitch_name("https://www.twitch.tv/MyStream?foo=1"), "mystream")
        self.assertEqual(normalize_twitch_name("#Cool_User"), "cool_user")
        self.assertEqual(normalize_twitch_name("@Bot123"), "bot123")
        with self.assertRaises(ValueError):
            normalize_twitch_name("bad name!")

        self.assertEqual(normalize_twitch_token("secret123"), "oauth:secret123")
        self.assertEqual(normalize_twitch_token("oauth:secret123"), "oauth:secret123")
        with self.assertRaises(ValueError):
            normalize_twitch_token("bad\ntoken")

        sanitized = sanitize_irc_text("Hello\r\nJOIN #evil\x00world")
        self.assertEqual(sanitized, "Hello JOIN #evil world")

        long_text = " ".join(["word"] * 150)  # > 500 chars
        parts = split_twitch_message(long_text, max_len=500)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(p) <= 500 for p in parts))


class TwitchClientLoopbackTests(unittest.TestCase):
    def test_connect_send_pong_and_test_connection(self) -> None:
        srv = _MockIrcServer(fail_auth=False)
        events: list[tuple[str, str]] = []
        connected_evt = threading.Event()

        def _on_status(st: str, dtl: str) -> None:
            events.append((st, dtl))
            if st == "connected":
                connected_evt.set()

        client = TwitchChatClient(host="127.0.0.1", port=srv.port, use_tls=False, on_status=_on_status)
        try:
            client.configure(True, "MyBot", "tok123", "https://twitch.tv/TestChan")
            self.assertTrue(connected_evt.wait(timeout=2.0))
            self.assertEqual(client.status, ("connected", "testchan"))

            self.assertTrue(client.send_message("Привіт чату!\r\nДругий рядок"))
            deadline = time.monotonic() + 2.0
            while (not srv.privmsgs or not srv.pongs) and time.monotonic() < deadline:
                time.sleep(0.02)

            self.assertTrue(any("PRIVMSG #testchan :Привіт чату! Другий рядок" in m for m in srv.privmsgs))
            self.assertTrue(any("PONG :tmi.twitch.tv" in p for p in srv.pongs))

            ok, chan = client.test_connection("MyBot", "oauth:tok123", "#TestChan", test_message="Тест!")
            self.assertTrue(ok)
            self.assertEqual(chan, "testchan")
            deadline = time.monotonic() + 2.0
            while not any("PRIVMSG #testchan :Тест!" in m for m in srv.privmsgs) and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(any("PRIVMSG #testchan :Тест!" in m for m in srv.privmsgs))
        finally:
            client.shutdown()
            srv.close()

    def test_auth_failure_reports_error(self) -> None:
        srv = _MockIrcServer(fail_auth=True)
        err_evt = threading.Event()

        def _on_status(st: str, _dtl: str) -> None:
            if st == "error":
                err_evt.set()

        client = TwitchChatClient(host="127.0.0.1", port=srv.port, use_tls=False, on_status=_on_status)
        try:
            client.configure(True, "MyBot", "wrong", "TestChan")
            self.assertTrue(err_evt.wait(timeout=2.0))
            st, dtl = client.status
            self.assertEqual(st, "error")
            self.assertIn("authentication failed", dtl.lower())
        finally:
            client.shutdown()
            srv.close()

    def test_token_only_auto_resolves_username_and_channel(self) -> None:
        srv = _MockIrcServer(fail_auth=False)
        resolved: list[tuple[str, str]] = []
        connected_evt = threading.Event()
        client = TwitchChatClient(
            host="127.0.0.1",
            port=srv.port,
            use_tls=False,
            on_status=lambda st, _dtl: connected_evt.set() if st == "connected" else None,
            on_resolved=lambda u, c: resolved.append((u, c)),
            token_resolver=lambda _tok: "streamer_me",
        )
        try:
            client.configure(True, "", "just_my_token", "")
            self.assertTrue(connected_evt.wait(timeout=2.0))
            self.assertEqual(client.status, ("connected", "streamer_me"))
            self.assertEqual(resolved, [("streamer_me", "streamer_me")])
            self.assertTrue(client.send_message("Привіт від себе!"))
            deadline = time.monotonic() + 2.0
            while not srv.privmsgs and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(any("PRIVMSG #streamer_me :Привіт від себе!" in m for m in srv.privmsgs))
        finally:
            client.shutdown()
            srv.close()

    def test_intercept_own_messages_and_suppress_echo_loop(self) -> None:
        srv = _MockIrcServer(fail_auth=False)
        received: list[str] = []
        got_evt = threading.Event()
        connected_evt = threading.Event()

        def _on_msg(txt: str) -> None:
            received.append(txt)
            got_evt.set()

        client = TwitchChatClient(
            host="127.0.0.1",
            port=srv.port,
            use_tls=False,
            on_status=lambda st, _dtl: connected_evt.set() if st == "connected" else None,
            on_message=_on_msg,
            token_resolver=lambda _tok: "streamer_me",
        )
        try:
            # Test both outgoing mirror and incoming intercept active together.
            client.configure(True, "", "tok_abc", "", intercept_own=True)
            self.assertTrue(connected_evt.wait(timeout=2.0))

            # Outgoing send works, and manual same-text message typed in Twitch web chat shortly after IS spoken.
            self.assertTrue(client.send_message("куку"))
            deadline = time.monotonic() + 2.0
            while not srv.privmsgs and time.monotonic() < deadline:
                time.sleep(0.02)
            # Message from another viewer must be ignored.
            srv.broadcast_raw(":viewer1!viewer1@viewer1.tmi.twitch.tv PRIVMSG #streamer_me :Чуже повідомлення")
            # Message typed directly by the same user in Twitch web/phone chat MUST be intercepted.
            srv.broadcast_raw("@badge-info=;badges=broadcaster/1 :streamer_me!streamer_me@streamer_me.tmi.twitch.tv PRIVMSG #streamer_me :куку")

            self.assertTrue(got_evt.wait(timeout=2.0))
            time.sleep(0.1)
            self.assertEqual(received, ["куку"])

            # Test parse_own_privmsg helper directly for ACTION messages.
            self.assertEqual(
                parse_own_privmsg(":streamer_me!u@h PRIVMSG #streamer_me :\x01ACTION махає рукою\x01", "streamer_me", "streamer_me"),
                "махає рукою",
            )

            # Intercept-only mode (outgoing mirror OFF, intercept ON) keeps connection alive.
            client.configure(False, "", "tok_abc", "", intercept_own=True)
            self.assertFalse(client.send_message("Не повинно йти в чат"))
            got_evt.clear()
            srv.broadcast_raw(":streamer_me!u@h PRIVMSG #streamer_me :Тільки озвучка")
            self.assertTrue(got_evt.wait(timeout=2.0))
            self.assertEqual(received[-1], "Тільки озвучка")
        finally:
            client.shutdown()
            srv.close()

    def test_parse_privmsg_tags_and_viewer_filters(self) -> None:
        raw_sub = "@badges=subscriber/12,vip/1;display-name=Olena_UA;mod=0;subscriber=1 :olena_ua!u@h PRIVMSG #mychan :Привіт стрім!"
        msg = parse_privmsg(raw_sub, "streamer_me", "mychan")
        self.assertIsNotNone(msg)
        assert msg is not None
        self.assertEqual(msg.username, "olena_ua")
        self.assertEqual(msg.display_name, "Olena_UA")
        self.assertTrue(msg.is_vip)
        self.assertTrue(msg.is_subscriber)
        self.assertFalse(msg.is_mod)
        self.assertFalse(msg.is_own)

        spoken, reason = format_and_filter_chat_message(
            msg,
            read_own=False,
            read_viewers=True,
            viewer_filter="subscribers_vips_mods",
            read_format="user_says",
        )
        self.assertEqual(reason, "ok")
        self.assertEqual(spoken, "Olena_UA каже: Привіт стрім!")

        spoken_colon, _ = format_and_filter_chat_message(
            msg, read_own=False, read_viewers=True, read_format="user_colon"
        )
        self.assertEqual(spoken_colon, "Olena_UA: Привіт стрім!")

        spoken_only, _ = format_and_filter_chat_message(
            msg, read_own=False, read_viewers=True, read_format="text_only"
        )
        self.assertEqual(spoken_only, "Привіт стрім!")

        # Ignored bot
        bot_msg = TwitchChatMessage("nightbot", "Nightbot", "Команда виконана", "mychan", is_own=False, is_mod=True)
        self.assertEqual(
            format_and_filter_chat_message(
                bot_msg, read_own=False, read_viewers=True, ignored_users="Nightbot, moobot"
            ),
            (None, "ignored_user"),
        )

        # Commands and links filtering
        cmd_msg = TwitchChatMessage("user1", "User1", "!uptime", "mychan", is_own=False)
        self.assertEqual(
            format_and_filter_chat_message(cmd_msg, read_own=False, read_viewers=True, ignore_commands=True),
            (None, "command"),
        )
        link_msg = TwitchChatMessage("user1", "User1", "дивись https://example.com тут", "mychan", is_own=False)
        self.assertEqual(
            format_and_filter_chat_message(link_msg, read_own=False, read_viewers=True, ignore_links=True),
            (None, "link"),
        )

        # Whitelist + role filter
        pleb = TwitchChatMessage("pleb1", "Pleb1", "Привіт", "mychan", is_own=False)
        self.assertEqual(
            format_and_filter_chat_message(pleb, read_own=False, read_viewers=True, viewer_filter="vips_mods"),
            (None, "role_filtered"),
        )
        self.assertEqual(
            format_and_filter_chat_message(
                pleb, read_own=False, read_viewers=True, viewer_filter="whitelist", allowed_users="@pleb1, friend"
            )[1],
            "ok",
        )

    def test_read_viewers_loopback_delivery(self) -> None:
        srv = _MockIrcServer(fail_auth=False)
        chat_events: list[TwitchChatMessage] = []
        evt = threading.Event()
        connected_evt = threading.Event()

        client = TwitchChatClient(
            host="127.0.0.1",
            port=srv.port,
            use_tls=False,
            on_status=lambda st, _dtl: connected_evt.set() if st == "connected" else None,
            token_resolver=lambda _tok: "streamer_me",
        )
        client.set_chat_message_callback(lambda m: (chat_events.append(m), evt.set()))
        try:
            client.configure(False, "", "tok_v", "", intercept_own=False, read_viewers=True)
            self.assertTrue(connected_evt.wait(timeout=2.0))

            srv.broadcast_raw("@badges=vip/1;display-name=Alex;subscriber=1 :alex!u@h PRIVMSG #streamer_me :Як справи?")
            self.assertTrue(evt.wait(timeout=2.0))
            self.assertEqual(len(chat_events), 1)
            self.assertEqual(chat_events[0].display_name, "Alex")
            self.assertTrue(chat_events[0].is_vip)
            self.assertEqual(chat_events[0].text, "Як справи?")
        finally:
            client.shutdown()
            srv.close()

    def test_command_prefixes_and_multi_voice_rules(self) -> None:
        rules = [
            {"command": "!t", "engine_id": "online:edge", "voice_id": "uk-UA-PolinaNeural", "volume": 90},
            {"command": "!1t", "engine_id": "piper", "voice_id": "uk_UA-lada-x_low", "volume": 130},
            {"command": "!!!", "engine_id": "silero", "voice_id": "v4_ua", "volume": 110},
        ]

        # 1) read_all=False: plain messages and unknown !commands are skipped; !t / !1t / !!! work & strip prefix
        plain = TwitchChatMessage("u1", "User1", "звичайний чат", "c", is_own=False)
        self.assertEqual(
            evaluate_chat_message(
                plain, read_own=False, read_viewers=True, read_all=False, command_prefix="!t", voice_rules=rules
            ),
            (None, "no_prefix", None),
        )
        other_cmd = TwitchChatMessage("u1", "User1", "!uptime", "c", is_own=False)
        self.assertEqual(
            evaluate_chat_message(
                other_cmd, read_own=False, read_viewers=True, read_all=False, command_prefix="!t", voice_rules=rules
            ),
            (None, "no_prefix", None),
        )
        # "!test" must NOT match "!t"
        word_cmd = TwitchChatMessage("u1", "User1", "!test привіт", "c", is_own=False)
        self.assertEqual(
            evaluate_chat_message(
                word_cmd, read_own=False, read_viewers=True, read_all=False, command_prefix="!t", voice_rules=rules
            ),
            (None, "no_prefix", None),
        )

        # Match "!1t" (longer prefix beats "!1" or "!t")
        m_1t = TwitchChatMessage("u1", "User1", "!1t голос два", "c", is_own=False)
        spoken, reason, matched = evaluate_chat_message(
            m_1t, read_own=False, read_viewers=True, read_all=False, command_prefix="!t", voice_rules=rules
        )
        self.assertEqual(reason, "ok")
        self.assertEqual(spoken, "User1 каже: голос два")
        self.assertIsNotNone(matched)
        self.assertEqual(matched["voice_id"], "uk_UA-lada-x_low")  # type: ignore[index]

        # Symbol prefix "!!!" matches with or without space
        m_bang = TwitchChatMessage("u2", "Alex", "!!!вау привіт", "c", is_own=False)
        spoken2, reason2, matched2 = evaluate_chat_message(
            m_bang, read_own=False, read_viewers=True, read_all=True, command_prefix="!t", voice_rules=rules
        )
        self.assertEqual(reason2, "ok")
        self.assertEqual(spoken2, "Alex каже: вау привіт")
        self.assertEqual(matched2["engine_id"], "silero")  # type: ignore[index]

        # 2) read_all=True: plain text speaks with matched=None (default chat voice)
        spoken_plain, reason_plain, matched_plain = evaluate_chat_message(
            plain, read_own=False, read_viewers=True, read_all=True, command_prefix="!t", voice_rules=rules
        )
        self.assertEqual(reason_plain, "ok")
        self.assertEqual(spoken_plain, "User1 каже: звичайний чат")
        self.assertIsNone(matched_plain)

    def test_moderation_clearchat_and_clearmsg_cancel_queued_tts(self) -> None:
        from core.tts_manager import tts_manager

        raw_msg = "@id=msg-123;display-name=Spammer :spammer!u@h PRIVMSG #mychan :поганий текст"
        parsed = parse_privmsg(raw_msg, "streamer", "mychan")
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.msg_id, "msg-123")

        self.assertEqual(
            parse_moderation_line("@login=spammer;target-msg-id=msg-123 :tmi.twitch.tv CLEARMSG #mychan :bad", "mychan"),
            ("clearmsg", "spammer", "msg-123"),
        )
        self.assertEqual(
            parse_moderation_line("@ban-duration=600 :tmi.twitch.tv CLEARCHAT #mychan :spammer", "mychan"),
            ("clearchat", "spammer", ""),
        )

        # Drain existing queue safely in test and verify queued moderation pruning
        tts_manager.stop_chat_tts()
        tts_manager._queue.put(("own text", False, True, False, False, "", "", None, "", ""))
        tts_manager._queue.put(("chat spam", False, True, False, True, "", "", None, "spammer", "msg-123"))
        tts_manager._queue.put(("good viewer", False, True, False, True, "", "", None, "gooduser", "msg-456"))
        try:
            removed = tts_manager.cancel_chat_by_moderation("clearmsg", "spammer", "msg-123")
            self.assertEqual(removed, 1)

            # Timeout/ban clears remaining queued messages from that user, preserving own & gooduser
            tts_manager._queue.put(("more spam", False, True, False, True, "", "", None, "spammer", "msg-789"))
            removed_ban = tts_manager.cancel_chat_by_moderation("clearchat", "spammer", "")
            self.assertEqual(removed_ban, 1)

            remaining = []
            while not tts_manager._queue.empty():
                remaining.append(tts_manager._queue.get_nowait()[0])
                tts_manager._queue.task_done()
            self.assertEqual(remaining, ["own text", "good viewer"])
        finally:
            tts_manager.stop_chat_tts()


if __name__ == "__main__":
    unittest.main()
