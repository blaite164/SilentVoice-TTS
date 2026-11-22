"""Simple Twitch chat sender over Twitch IRC (TLS) without requiring dev.twitch.tv."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import queue
import re
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

TWITCH_IRC_HOST = "irc.chat.twitch.tv"
TWITCH_IRC_PORT = 6697
TWITCH_VALIDATE_URL = "https://id.twitch.tv/oauth2/validate"
MAX_MESSAGE_CHARS = 500
_RATE_WINDOW_SECS = 30.0
_RATE_MAX_MESSAGES = 18
_NAME_RE = re.compile(r"^[a-z0-9_]{1,25}$")
_URL_RE = re.compile(r"^(?:https?://)?(?:www\.)?twitch\.tv/([a-zA-Z0-9_]+)(?:[/?#].*)?$", re.IGNORECASE)
_CHAT_LINK_RE = re.compile(r"(?:https?://\S+|www\.\S+)", re.IGNORECASE)

StatusCallback = Callable[[str, str], None]
ResolvedCallback = Callable[[str, str], None]
MessageCallback = Callable[[str], None]


@dataclass(frozen=True)
class TwitchChatMessage:
    username: str
    display_name: str
    text: str
    channel: str
    is_own: bool
    is_mod: bool = False
    is_vip: bool = False
    is_subscriber: bool = False
    is_broadcaster: bool = False
    is_action: bool = False
    msg_id: str = ""


ChatMessageCallback = Callable[[TwitchChatMessage], None]
ModerationCallback = Callable[[str, str, str], None]  # (event_type, target_user, msg_id)


def normalize_twitch_name(raw: str) -> str:
    """Normalize a Twitch username or channel (accepts URLs, @user, #channel)."""
    text = str(raw or "").strip()
    m = _URL_RE.match(text)
    if m:
        text = m.group(1)
    text = text.lstrip("@#").strip().lower()
    if not _NAME_RE.match(text):
        raise ValueError("Twitch name must be 1–25 Latin letters, digits, or '_'")
    return text


def normalize_twitch_token(raw: str) -> str:
    """Normalize an OAuth chat token into `oauth:<secret>` form without whitespace."""
    token = str(raw or "").strip()
    if not token:
        raise ValueError("Twitch OAuth token is empty")
    if any(ch in token for ch in ("\r", "\n", "\x00", " ")):
        raise ValueError("Twitch OAuth token contains invalid characters")
    if token.lower().startswith("oauth:"):
        secret = token[6:]
    else:
        secret = token
    if not secret:
        raise ValueError("Twitch OAuth token is empty")
    return f"oauth:{secret}"


def sanitize_irc_text(text: str) -> str:
    """Strip CR/LF/NUL so user text can never inject raw IRC commands."""
    cleaned = str(text or "").replace("\r", " ").replace("\n", " ").replace("\x00", " ")
    return " ".join(cleaned.split())


def _unescape_irc_tag_value(raw: str) -> str:
    if "\\" not in raw:
        return raw
    out: List[str] = []
    i = 0
    n = len(raw)
    escapes = {":": ";", "s": " ", "\\": "\\", "r": "", "n": ""}
    while i < n:
        ch = raw[i]
        if ch == "\\" and i + 1 < n:
            nxt = raw[i + 1]
            out.append(escapes.get(nxt, nxt))
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def parse_irc_tags(tag_blob: str) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    for part in tag_blob.split(";"):
        if "=" not in part:
            if part:
                tags[part] = ""
            continue
        k, v = part.split("=", 1)
        tags[k] = _unescape_irc_tag_value(v)
    return tags


def parse_privmsg(line: str, expected_user: str, expected_channel: str) -> Optional[TwitchChatMessage]:
    """Parse any PRIVMSG in `#expected_channel`, including IRCv3 badges/display-name tags."""
    raw = line.rstrip("\r\n")
    tags: Dict[str, str] = {}
    if raw.startswith("@"):
        if " " not in raw:
            return None
        tag_part, raw = raw.split(" ", 1)
        tags = parse_irc_tags(tag_part[1:])
    if not raw.startswith(":"):
        return None
    parts = raw.split(" ", 3)
    if len(parts) < 4:
        return None
    prefix, command, target, trailing = parts
    if command.upper() != "PRIVMSG":
        return None
    sender = prefix[1:].split("!", 1)[0].split("@", 1)[0].strip().lower()
    chan = target.lstrip("#").strip().lower()
    if not sender or chan != expected_channel.strip().lower():
        return None
    msg = trailing[1:] if trailing.startswith(":") else trailing
    is_action = False
    if msg.startswith("\x01ACTION ") and msg.endswith("\x01"):
        msg = msg[8:-1]
        is_action = True
    clean = sanitize_irc_text(msg)
    if not clean:
        return None

    badges_raw = tags.get("badges", "")
    badge_names = {b.split("/", 1)[0].strip().lower() for b in badges_raw.split(",") if b}
    display_name = sanitize_irc_text(tags.get("display-name", "")) or sender
    is_broadcaster = "broadcaster" in badge_names or (chan == sender)
    is_vip = "vip" in badge_names
    is_mod = tags.get("mod") == "1" or "moderator" in badge_names or is_broadcaster
    is_subscriber = tags.get("subscriber") == "1" or "subscriber" in badge_names or "founder" in badge_names
    return TwitchChatMessage(
        username=sender,
        display_name=display_name,
        text=clean,
        channel=chan,
        is_own=(sender == expected_user.strip().lower()),
        is_mod=is_mod,
        is_vip=is_vip,
        is_subscriber=is_subscriber,
        is_broadcaster=is_broadcaster,
        is_action=is_action,
        msg_id=str(tags.get("id", "") or ""),
    )


def parse_moderation_line(line: str, expected_channel: str) -> Optional[Tuple[str, str, str]]:
    """Parse Twitch IRC CLEARCHAT / CLEARMSG moderation events for `#expected_channel`.

    Returns `(event_type, target_user_lower, target_msg_id)` or `None`.
    """
    raw = line.rstrip("\r\n")
    tags: Dict[str, str] = {}
    if raw.startswith("@"):
        if " " not in raw:
            return None
        tag_part, raw = raw.split(" ", 1)
        tags = parse_irc_tags(tag_part[1:])
    if not raw.startswith(":"):
        return None
    parts = raw.split(" ", 3)
    if len(parts) < 3:
        return None
    command = parts[1].upper()
    chan = parts[2].lstrip("#").strip().lower()
    if chan != expected_channel.strip().lower():
        return None
    trailing = parts[3][1:].strip() if len(parts) >= 4 and parts[3].startswith(":") else (parts[3].strip() if len(parts) >= 4 else "")
    if command == "CLEARCHAT":
        return "clearchat", trailing.lower(), ""
    if command == "CLEARMSG":
        target_user = sanitize_irc_text(tags.get("login", "")).lower()
        target_msg_id = str(tags.get("target-msg-id", "") or "").strip()
        if not target_msg_id and not target_user:
            return None
        return "clearmsg", target_user, target_msg_id
    return None


def parse_own_privmsg(line: str, expected_user: str, expected_channel: str) -> Optional[str]:
    """Extract clean text if `line` is a PRIVMSG from `expected_user` in `#expected_channel`."""
    parsed = parse_privmsg(line, expected_user, expected_channel)
    if parsed is None or not parsed.is_own:
        return None
    return parsed.text


def _parse_user_list(raw: str) -> Set[str]:
    tokens = re.split(r"[,;\n\r]+", str(raw or ""))
    return {t.strip().lstrip("@").lower() for t in tokens if t.strip().lstrip("@")}


def _match_command_prefix(text: str, prefix: str) -> Optional[str]:
    """Match `prefix` at the start of `text` (case-insensitive) and return stripped remainder."""
    p = str(prefix or "").strip()
    if not p:
        return None
    low_t = text.lower()
    low_p = p.lower()
    if not low_t.startswith(low_p):
        return None
    # Alphanumeric endings (like !t, !1t) need whitespace or end-of-message after prefix
    # so "!test" does not match "!t". Symbol endings (like !!!, >>) match both ways.
    if low_p[-1].isalnum() and len(text) > len(p) and not text[len(p)].isspace():
        return None
    return text[len(p) :].strip()


def evaluate_chat_message(
    msg: TwitchChatMessage,
    *,
    read_own: bool,
    read_viewers: bool,
    viewer_filter: str = "all",
    allowed_users: str = "",
    ignored_users: str = "",
    ignore_commands: bool = True,
    ignore_links: bool = True,
    read_format: str = "user_says",
    max_chars: int = 220,
    say_template: str = "{user} каже: {text}",
    read_all: bool = True,
    command_prefix: str = "",
    voice_rules: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[str], str, Optional[Dict[str, Any]]]:
    """Return `(spoken_text, status_code, matched_voice_rule)` for a chat message."""
    if msg.is_own:
        if not read_own:
            return None, "disabled", None
        text = msg.text.strip()
        return (text, "ok", None) if text else (None, "empty", None)

    if not read_viewers:
        return None, "disabled", None

    ignored = _parse_user_list(ignored_users)
    if msg.username.lower() in ignored or msg.display_name.lower() in ignored:
        return None, "ignored_user", None

    mode = str(viewer_filter or "all").strip().lower()
    if mode == "subscribers_vips_mods":
        if not (msg.is_subscriber or msg.is_vip or msg.is_mod or msg.is_broadcaster):
            return None, "role_filtered", None
    elif mode == "vips_mods":
        if not (msg.is_vip or msg.is_mod or msg.is_broadcaster):
            return None, "role_filtered", None
    elif mode == "whitelist":
        allowed = _parse_user_list(allowed_users)
        if msg.username.lower() not in allowed and msg.display_name.lower() not in allowed:
            return None, "not_whitelisted", None

    text = msg.text.strip()
    if not text:
        return None, "empty", None

    matched_rule: Optional[Dict[str, Any]] = None
    matched_command = False
    clean_rules = [
        r for r in (voice_rules or [])
        if isinstance(r, dict) and str(r.get("command", "")).strip()
    ]
    clean_rules.sort(key=lambda r: len(str(r.get("command", "")).strip()), reverse=True)
    for rule in clean_rules:
        rem = _match_command_prefix(text, str(rule.get("command", "")))
        if rem is not None:
            matched_rule = rule
            matched_command = True
            text = rem
            break

    if not matched_command and not read_all:
        rem = _match_command_prefix(text, command_prefix)
        if rem is not None:
            matched_command = True
            text = rem

    if not matched_command:
        if not read_all:
            return None, "no_prefix", None
        if ignore_commands and text.startswith(("!", "/")):
            return None, "command", None

    if not text:
        return None, "empty", None
    if ignore_links and _CHAT_LINK_RE.search(text):
        return None, "link", None

    if int(max_chars or 0) > 0 and len(text) > int(max_chars):
        text = text[: int(max_chars)].rstrip() + "…"

    speaker = msg.display_name or msg.username
    fmt = str(read_format or "user_says").strip().lower()
    if fmt == "text_only":
        return text, "ok", matched_rule
    if fmt == "user_colon":
        return f"{speaker}: {text}", "ok", matched_rule
    return say_template.format(user=speaker, text=text), "ok", matched_rule


def format_and_filter_chat_message(
    msg: TwitchChatMessage,
    *,
    read_own: bool,
    read_viewers: bool,
    viewer_filter: str = "all",
    allowed_users: str = "",
    ignored_users: str = "",
    ignore_commands: bool = True,
    ignore_links: bool = True,
    read_format: str = "user_says",
    max_chars: int = 220,
    say_template: str = "{user} каже: {text}",
    read_all: bool = True,
    command_prefix: str = "",
    voice_rules: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[str], str]:
    """Return `(spoken_text, status_code)` according to chat-reading filters and format."""
    spoken, reason, _ = evaluate_chat_message(
        msg,
        read_own=read_own,
        read_viewers=read_viewers,
        viewer_filter=viewer_filter,
        allowed_users=allowed_users,
        ignored_users=ignored_users,
        ignore_commands=ignore_commands,
        ignore_links=ignore_links,
        read_format=read_format,
        max_chars=max_chars,
        say_template=say_template,
        read_all=read_all,
        command_prefix=command_prefix,
        voice_rules=voice_rules,
    )
    return spoken, reason


def resolve_token_login(raw_token: str, timeout: float = 6.0) -> str:
    """Resolve the Twitch account username from an OAuth token via id.twitch.tv/oauth2/validate."""
    oauth = normalize_twitch_token(raw_token)
    secret = oauth[6:]
    req = urllib.request.Request(
        TWITCH_VALIDATE_URL,
        headers={"Authorization": f"OAuth {secret}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise PermissionError("Twitch OAuth token is invalid or expired") from exc
        raise ConnectionError(f"Twitch token validation failed (HTTP {exc.code})") from exc
    login = str(payload.get("login", "") or "").strip()
    if not login:
        raise PermissionError("Twitch token response did not include login")
    return normalize_twitch_name(login)


def split_twitch_message(text: str, max_len: int = MAX_MESSAGE_CHARS) -> List[str]:
    """Split a sanitized chat message into <= max_len chunks on word boundaries."""
    clean = sanitize_irc_text(text)
    if not clean:
        return []
    if len(clean) <= max_len:
        return [clean]
    parts: List[str] = []
    remaining = clean
    while len(remaining) > max_len:
        cut = remaining.rfind(" ", 0, max_len + 1)
        if cut <= 0:
            cut = max_len
        part = remaining[:cut].strip()
        if part:
            parts.append(part)
        remaining = remaining[cut:].strip()
    if remaining:
        parts.append(remaining)
    return parts


class TwitchChatClient:
    """Background Twitch IRC client that forwards spoken TTS phrases to a channel."""

    def __init__(
        self,
        host: str = TWITCH_IRC_HOST,
        port: int = TWITCH_IRC_PORT,
        use_tls: bool = True,
        on_status: Optional[StatusCallback] = None,
        on_resolved: Optional[ResolvedCallback] = None,
        on_message: Optional[MessageCallback] = None,
        token_resolver: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._host = host
        self._port = int(port)
        self._use_tls = bool(use_tls)
        self._cond = threading.Condition()
        self._on_status = on_status
        self._on_resolved = on_resolved
        self._on_message = on_message
        self._on_chat_message: Optional[ChatMessageCallback] = None
        self._on_moderation: Optional[ModerationCallback] = None
        self._token_resolver = token_resolver or resolve_token_login
        self._desired: Optional[Tuple[str, str, str]] = None  # (username_or_empty, oauth_token, channel_or_empty)
        self._mirror_outgoing = False
        self._intercept_own = False
        self._read_viewers = False
        self._generation = 0
        self._state = "disconnected"
        self._detail = ""
        self._send_queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=100)
        self._send_timestamps: deque[float] = deque()
        self._sock: Optional[socket.socket] = None
        self._shutdown = False
        self._thread: Optional[threading.Thread] = None
        self._connected_event = threading.Event()

    @property
    def status(self) -> Tuple[str, str]:
        with self._cond:
            return self._state, self._detail

    def set_status_callback(self, callback: Optional[StatusCallback]) -> None:
        with self._cond:
            self._on_status = callback

    def set_resolved_callback(self, callback: Optional[ResolvedCallback]) -> None:
        with self._cond:
            self._on_resolved = callback

    def set_message_callback(self, callback: Optional[MessageCallback]) -> None:
        with self._cond:
            self._on_message = callback

    def set_chat_message_callback(self, callback: Optional[ChatMessageCallback]) -> None:
        with self._cond:
            self._on_chat_message = callback

    def set_moderation_callback(self, callback: Optional[ModerationCallback]) -> None:
        with self._cond:
            self._on_moderation = callback

    def _emit_status(self, state: str, detail: str = "") -> None:
        cb: Optional[StatusCallback] = None
        with self._cond:
            if (self._state, self._detail) == (state, detail):
                return
            self._state = state
            self._detail = detail
            cb = self._on_status
        if state == "connected":
            self._connected_event.set()
        else:
            self._connected_event.clear()
        if cb is not None:
            cb(state, detail)

    def configure(
        self,
        enabled: bool,
        username: str,
        token: str,
        channel: str,
        *,
        intercept_own: bool = False,
        read_viewers: bool = False,
    ) -> None:
        """Start/update or stop the Twitch IRC connection based on user settings.

        Connects whenever `enabled` (outgoing chat mirror), `intercept_own`
        (speaking user's own Twitch chat messages), or `read_viewers`
        (reading viewer chat messages) is True.
        `username` and `channel` may be empty when `token` is provided; the worker
        resolves the token's login via Twitch and defaults `channel` to that login.
        """
        mirror = bool(enabled)
        intercept = bool(intercept_own)
        viewers = bool(read_viewers)
        should_connect = mirror or intercept or viewers
        desired: Optional[Tuple[str, str, str]] = None
        if should_connect:
            tok = normalize_twitch_token(token)
            user = normalize_twitch_name(username) if str(username or "").strip() else ""
            chan = normalize_twitch_name(channel) if str(channel or "").strip() else user
            desired = (user, tok, chan)

        with self._cond:
            if (
                desired == self._desired
                and mirror == self._mirror_outgoing
                and intercept == self._intercept_own
                and viewers == self._read_viewers
                and not self._shutdown
            ):
                return
            creds_changed = desired != self._desired
            self._shutdown = False
            self._mirror_outgoing = mirror
            self._intercept_own = intercept
            self._read_viewers = viewers
            self._desired = desired
            if creds_changed:
                self._generation += 1
                if desired is None:
                    self._drain_send_queue()
                sock = self._sock
            else:
                if not mirror:
                    self._drain_send_queue()
                sock = None

            if desired is not None and (self._thread is None or not self._thread.is_alive()):
                self._thread = threading.Thread(
                    target=self._run,
                    name="twitch-irc",
                    daemon=True,
                )
                self._thread.start()
            self._cond.notify_all()

        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        if desired is None:
            self._emit_status("disconnected", "")

    def send_message(self, text: str) -> bool:
        """Queue a spoken phrase for Twitch chat (non-blocking)."""
        with self._cond:
            if self._desired is None or not self._mirror_outgoing or self._shutdown:
                return False
        parts = split_twitch_message(text)
        if not parts:
            return False
        queued_any = False
        with self._cond:
            for part in parts:
                try:
                    self._send_queue.put_nowait(part)
                    queued_any = True
                except queue.Full:
                    break
            if queued_any:
                self._cond.notify_all()
        return queued_any

    def _resolve_credentials(self, user: str, tok: str, chan: str) -> Tuple[str, str]:
        resolved_user = user
        if not resolved_user:
            resolved_user = self._token_resolver(tok)
        resolved_chan = chan or resolved_user
        cb: Optional[ResolvedCallback] = None
        with self._cond:
            cb = self._on_resolved
        if cb is not None and (resolved_user != user or resolved_chan != chan):
            cb(resolved_user, resolved_chan)
        return resolved_user, resolved_chan

    def test_connection(
        self,
        username: str,
        token: str,
        channel: str,
        test_message: str = "SilentVoice: тест чату Twitch!",
        timeout: float = 8.0,
    ) -> Tuple[bool, str]:
        """Verify credentials on a direct short-lived session and optionally send a test line."""
        tok = normalize_twitch_token(token)
        raw_user = normalize_twitch_name(username) if str(username or "").strip() else ""
        raw_chan = normalize_twitch_name(channel) if str(channel or "").strip() else raw_user
        user, chan = self._resolve_credentials(raw_user, tok, raw_chan)

        sock = self._connect_and_auth(user, tok, chan, timeout=timeout)
        try:
            clean_msg = sanitize_irc_text(test_message)
            if clean_msg:
                self._send_raw(sock, f"PRIVMSG #{chan} :{clean_msg}\r\n")
            return True, chan
        finally:
            try:
                self._send_raw(sock, "QUIT\r\n")
                sock.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            try:
                sock.settimeout(0.05)
                while sock.recv(1024):
                    pass
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    def shutdown(self, timeout: float = 2.0) -> None:
        with self._cond:
            self._shutdown = True
            self._desired = None
            self._generation += 1
            self._drain_send_queue()
            sock = self._sock
            thread = self._thread
            self._cond.notify_all()
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        with self._cond:
            if self._thread is thread:
                self._thread = None
        self._emit_status("disconnected", "")

    def _drain_send_queue(self) -> None:
        while True:
            try:
                self._send_queue.get_nowait()
            except queue.Empty:
                break

    def _open_socket(self, timeout: float = 10.0) -> socket.socket:
        raw = socket.create_connection((self._host, self._port), timeout=timeout)
        if not self._use_tls:
            return raw
        ctx = ssl.create_default_context()
        return ctx.wrap_socket(raw, server_hostname=self._host)

    @staticmethod
    def _send_raw(sock: socket.socket, line: str) -> None:
        sock.sendall(line.encode("utf-8"))

    def _connect_and_auth(
        self,
        username: str,
        oauth_token: str,
        channel: str,
        timeout: float = 10.0,
    ) -> socket.socket:
        sock = self._open_socket(timeout=timeout)
        try:
            sock.settimeout(timeout)
            self._send_raw(sock, "CAP REQ :twitch.tv/commands twitch.tv/tags\r\n")
            self._send_raw(sock, f"PASS {oauth_token}\r\n")
            self._send_raw(sock, f"NICK {username}\r\n")

            buf = ""
            welcomed = False
            joined = False
            deadline = time.monotonic() + timeout
            while not (welcomed and joined):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for Twitch IRC handshake")
                sock.settimeout(remaining)
                chunk = sock.recv(4096)
                if not chunk:
                    raise ConnectionError("Twitch IRC closed connection during login")
                buf += chunk.decode("utf-8", errors="replace")
                while "\r\n" in buf:
                    line, buf = buf.split("\r\n", 1)
                    if not line:
                        continue
                    if line.startswith("PING"):
                        payload = line.split(" ", 1)[1] if " " in line else ":tmi.twitch.tv"
                        self._send_raw(sock, f"PONG {payload}\r\n")
                        continue
                    if " NOTICE " in line:
                        msg_text = line.rsplit(" :", 1)[-1]
                        low = msg_text.lower()
                        if "authentication failed" in low or "improperly formatted auth" in low or "login unsuccessful" in low:
                            raise PermissionError(msg_text)
                    if " 001 " in line:
                        welcomed = True
                        self._send_raw(sock, f"JOIN #{channel}\r\n")
                    if f" JOIN #{channel}" in line.lower() or f" 366 {username} #{channel}" in line.lower() or " 366 " in line:
                        joined = True
            return sock
        except Exception:
            try:
                sock.close()
            except OSError:
                pass
            raise

    def _wait_rate_slot(self) -> None:
        now = time.monotonic()
        while self._send_timestamps and (now - self._send_timestamps[0]) >= _RATE_WINDOW_SECS:
            self._send_timestamps.popleft()
        if len(self._send_timestamps) >= _RATE_MAX_MESSAGES:
            wait = _RATE_WINDOW_SECS - (now - self._send_timestamps[0]) + 0.05
            if wait > 0:
                time.sleep(min(wait, 2.0))
        self._send_timestamps.append(time.monotonic())

    def _run(self) -> None:
        backoff = 1.0
        while True:
            with self._cond:
                if self._shutdown:
                    break
                desired = self._desired
                gen = self._generation
                if desired is None:
                    self._cond.wait()
                    continue

            raw_user, tok, raw_chan = desired
            self._emit_status("connecting", raw_chan or raw_user)
            try:
                user, chan = self._resolve_credentials(raw_user, tok, raw_chan)
                sock = self._connect_and_auth(user, tok, chan, timeout=10.0)
            except PermissionError as exc:
                self._emit_status("error", str(exc))
                # Auth failure will not fix itself on retry; wait for reconfigure/shutdown.
                with self._cond:
                    if not self._shutdown and self._generation == gen:
                        self._cond.wait(timeout=60.0)
                continue
            except Exception as exc:
                self._emit_status("error", str(exc))
                with self._cond:
                    if self._shutdown or self._generation != gen:
                        continue
                    self._cond.wait(timeout=backoff)
                backoff = min(30.0, backoff * 2.0)
                continue

            backoff = 1.0
            with self._cond:
                if self._shutdown or self._generation != gen:
                    try:
                        sock.close()
                    except OSError:
                        pass
                    continue
                self._sock = sock

            self._emit_status("connected", chan)
            try:
                self._session_loop(sock, user, chan, gen)
            except Exception as exc:
                with self._cond:
                    reconfig = self._shutdown or self._generation != gen
                if not reconfig:
                    self._emit_status("error", str(exc))
            finally:
                with self._cond:
                    if self._sock is sock:
                        self._sock = None
                try:
                    sock.close()
                except OSError:
                    pass

    def _session_loop(self, sock: socket.socket, username: str, channel: str, gen: int) -> None:
        buf = ""
        sock.settimeout(0.15)
        while True:
            with self._cond:
                if self._shutdown or self._generation != gen:
                    return
                intercept = self._intercept_own
                read_viewers = self._read_viewers
                msg_cb = self._on_message
                chat_cb = self._on_chat_message
                mod_cb = self._on_moderation

            # 1. Flush queued outgoing chat messages
            while True:
                try:
                    msg = self._send_queue.get_nowait()
                except queue.Empty:
                    break
                if msg is None:
                    continue
                self._wait_rate_slot()
                self._send_raw(sock, f"PRIVMSG #{channel} :{msg}\r\n")

            # 2. Poll incoming IRC control frames (PING / RECONNECT / NOTICE / own PRIVMSG)
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    raise ConnectionError("Disconnected by Twitch IRC")
                buf += chunk.decode("utf-8", errors="replace")
                while "\r\n" in buf:
                    line, buf = buf.split("\r\n", 1)
                    if line.startswith("PING"):
                        payload = line.split(" ", 1)[1] if " " in line else ":tmi.twitch.tv"
                        self._send_raw(sock, f"PONG {payload}\r\n")
                    elif line.startswith("RECONNECT") or " RECONNECT" in line:
                        raise ConnectionError("Twitch requested RECONNECT")
                    elif " NOTICE " in line:
                        notice_txt = line.rsplit(" :", 1)[-1]
                        low = notice_txt.lower()
                        if "authentication failed" in low or "msg_banned" in low or "msg_channel_suspended" in low:
                            raise PermissionError(notice_txt)
                    elif (intercept or read_viewers) and " PRIVMSG " in line:
                        parsed = parse_privmsg(line, username, channel)
                        if parsed is not None:
                            if chat_cb is not None and (
                                (parsed.is_own and intercept) or (not parsed.is_own and read_viewers)
                            ):
                                chat_cb(parsed)
                            if msg_cb is not None and intercept and parsed.is_own:
                                msg_cb(parsed.text)
                    elif read_viewers and mod_cb is not None and (" CLEARCHAT " in line or " CLEARMSG " in line):
                        mod_evt = parse_moderation_line(line, channel)
                        if mod_evt is not None:
                            mod_cb(*mod_evt)
            except socket.timeout:
                continue


twitch_chat = TwitchChatClient()
