# Security Policy & Privacy Overview

This document explains how **SilentVoice TTS** handles privileges, local secrets, and network connections.

---

## 1. Administrator Privileges (Optional, Windows UIPI)

By default, SilentVoice requests elevation via standard Windows UAC (`run_as_admin: true`).

**Why this is needed:**
- **Windows UIPI (User Interface Privilege Isolation):** Without elevation, Windows prevents global low-level keyboard hooks (`WH_KEYBOARD_LL`) from capturing hotkeys and overlay typing while a game, anti-cheat launcher, or Discord window running as administrator has focus.
- **Virtual Audio Cable Driver:** One-click installation of VB-Audio Virtual Cable requires administrator rights.

**How to run without administrator rights:**
- Disable **"Run as administrator"** in **Audio & System** settings (`"run_as_admin": false`), or
- Launch with the command-line flag `--no-admin-relaunch`.

All non-elevated applications (browsers, standard games, Discord without admin mode) continue to work normally without elevation.

---

## 2. Local Secret Storage (Windows DPAPI)

All sensitive credentials stored locally by SilentVoice are encrypted at rest using **Windows DPAPI** (`CryptProtectData` / `CryptUnprotectData`, scoped to the current Windows user account) via [`core/secret_store.py`](core/secret_store.py):

- Cloud TTS & AI API keys (`online_items`, `ai_items`)
- Twitch IRC OAuth token (`twitch_token`)
- Custom HTTP/SOCKS5 proxy URLs (which may contain embedded `user:pass@` credentials)
- Telegram API Hash (`tg_app_keys.json`) and Telethon session (`telegram_silero.session.dpapi`, stored as a DPAPI-encrypted `StringSession` rather than a plaintext `.session` SQLite database)

Existing plaintext secrets from earlier versions are automatically migrated to DPAPI on first load. Portable settings exports ([`core/settings_transfer.py`](core/settings_transfer.py)) strip all credential fields unless the user explicitly opts in.

---

## 3. Network Connections & Local Ports

SilentVoice contains **no telemetry, analytics, or advertising SDKs**. All external network access is feature-driven and optional:

| Feature | Destination / Port | Default State | How to Disable |
| :--- | :--- | :--- | :--- |
| **Local TTS** (Piper, Silero, HolosTTS, TeraTTS, Qwen3-TTS, OpenVoice v2) | 100% offline after one-time model download (Hugging Face / Silero / GitHub) | Off until downloaded | Do not download or delete model in **Voice & TTS** |
| **Cloud TTS** (Edge TTS, OpenAI, ElevenLabs, Google Cloud, Fish Audio, Telegram Silero) | Respective provider APIs (sends text to synthesize speech) | Opt-in (Edge TTS enabled by default if selected) | Select a local offline TTS engine |
| **AI Orthography** | Configured cloud LLM API or local `127.0.0.1` Ollama / LM Studio / llama.cpp | **Off** | Keep **AI Orthography** unchecked |
| **Twitch Chat Integration** | `irc.chat.twitch.tv` (TLS) | **Off** | Keep Twitch integration disabled |
| **OBS Subtitles Server** | Local loopback HTTP + WebSocket on `127.0.0.1:8765` (same-origin checked, never exposed to LAN) | On (loopback only) | Uncheck OBS Subtitles in **OBS Subtitles** tab |
| **VBAN Network Audio** | Outbound UDP audio stream (default port `6980`, user-configured target IP) | **Off** | Keep VBAN disabled in **Audio & System** |
| **Auto-Update Check** | GitHub / GitLab Releases API (HTTPS) + SHA-256 verification of downloaded installer | On | Uncheck **Auto-check updates** in **Audio & System** |
| **Download Proxy Bypass** | Optional mirror/proxy fallback when downloading models in restricted networks | On for downloads | Uncheck proxy bypass in **Voice & TTS** |

---

## 4. Reporting a Vulnerability

If you discover a security issue in SilentVoice TTS, please report it via GitHub Issues (for non-sensitive bugs) or contact the maintainer directly before public disclosure so it can be fixed promptly.
