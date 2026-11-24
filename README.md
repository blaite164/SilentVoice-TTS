<div align="center">

# 🎙️ SilentVoice

**Type to speak — naturally, instantly, and without touching a microphone.**

[![Platform](https://img.shields.io/badge/Platform-Windows%2010%2F11-0078D6?style=for-the-badge&logo=windows&logoColor=white)](#-system-requirements)
[![TTS](https://img.shields.io/badge/TTS-Local%20%26%20Cloud%20AI-8A2BE2?style=for-the-badge)](#%EF%B8%8F-voice--tts-engine)
[![OBS Ready](https://img.shields.io/badge/OBS-Lip--Sync%20Subtitles-302E31?style=for-the-badge&logo=obsstudio&logoColor=white)](#-obs-subtitles-with-lip-sync)
[![Audio](https://img.shields.io/badge/Audio-Virtual%20Cable%20%26%20VBAN-2EA44F?style=for-the-badge)](#-audio-routing--system)

🌐 **Language:** **English** | [Українська](README.uk.md) | [Русский](README.ru.md)

</div>

> **Озвучка тексту для німих людей у реальному часі** | **Бесплатная озвучка текста для немых людей**

---

**Can't or don't want to speak into a microphone? Just type, and SilentVoice will speak for you — in voice calls, multiplayer games, or live streams.**

Built specifically for **non-verbal individuals**, people with speech difficulties, VTubers, and gamers who prefer text communication, **SilentVoice** turns your keyboard into a real-time expressive voice without ever minimizing or breaking your game.

## 🎬 Video Overview

Watch the overview covering installation, local & cloud TTS models, AI typo correction, Twitch chat integration, and animated OBS subtitles:

> ℹ️ *Note: Currently, a quick video overview is only available in [Ukrainian](https://youtu.be/qGdbhC7rr3I) and [Russian](https://youtu.be/mXmoLLr2lJI).*

<div align="center">

[![SilentVoice — Video Overview](https://img.youtube.com/vi/YOUR_VIDEO_ID/maxresdefault.jpg)](https://youtu.be/YOUR_VIDEO_ID)

*Click the preview above to watch the overview on YouTube*

</div>

---

## ✨ Key Features

### 🗣️ Voice & TTS Engine
- **One-Click Local TTS Installer:** Download and set up offline TTS models in a single click with progress tracking, automatic mirrors, and proxy fallback.
  - **Language-Tailored Model Library:** Includes **4 local TTS models** each for Ukrainian and Russian (3 fast CPU models with instant voice previews + 1 GPU model with voice cloning) and **3 local models** for English and other languages (4 CPU models and 1 GPU voice-cloning model in total; **OpenVoice v2** works separately as a tone-transfer add-on).
  - **Voice Cloning & Tone Transfer:** GPU-accelerated voice cloning with curated starter voice packs (EN / UK / RU), custom audio sample support, plus **OpenVoice v2** tone-color transfer to give lightweight TTS voices a unique vocal identity.
- **Cloud TTS via API (6+ Providers):** One-click **Edge TTS** installation plus support for 6+ online TTS services with built-in step-by-step API key setup guides.
- **Primary & Fallback Failover + DSP:** Configure a backup TTS engine that takes over automatically if your primary engine fails, and enhance your voice with real-time **DSP audio filters**.
- **Smart Speech Caching:** Caches up to ~1,000 recent generations for instant zero-latency replay and permanently saves frequently repeated phrases.
- **Instant Filtering:** Filter available voices by language and speaker gender directly from the sidebar.

### 🧠 AI Orthography (Real-Time Typo & Grammar Fixer)
- **Type Fast, Sound Clean:** Automatically corrects spelling mistakes, missing punctuation, and typos via an LLM *before* your text is sent to the TTS engine.
- **Cloud & Local LLM Support:** Connect **7+ cloud AI providers** (with built-in API guides) or run 100% offline using **Ollama**, **LM Studio**, or **llama.cpp**.
- **Customizable Pipeline:** Configure primary and fallback LLMs and edit the correction system prompt to match your speaking style.

### 💬 Twitch Chat Integration
- **1-Click Token Setup:** Connect your Twitch channel in seconds with guided in-app instructions.
- **Command-Based Multi-Voice Reading:** Assign specific TTS engines and voices to custom chat commands (e.g., `!t`), configure what gets read aloud, or enable auto-voicing for your own chat messages.
- **Separate VBAN Routing:** Route your own voice and Twitch chat TTS to separate audio channels/PCs.

### ⚡ Quick Phrases & Soundboard
- **Instant Hotkey Triggers:** Bind frequently used text phrases to any keyboard shortcut, pre-render them to `.wav` for instant playback, or load custom `.mp3`/audio files as a soundboard.
- **Spoken History:** Browse previously spoken phrases, replay them in one click, save them directly to **Quick Phrases**, or export them as permanent audio clips.

### ⌨️ Game-Safe Hotkeys & Input Modes
- **Zero-Interference Overlay Input:** Press <kbd>Ctrl</kbd> + <kbd>Space</kbd> (customizable) to start typing right over your screen without stealing window focus or breaking fullscreen/borderless games.
- **Conversation Mode:** Stay in active typing mode automatically after pressing <kbd>Enter</kbd>—have natural back-and-forth conversations without repeatedly pressing your activation hotkey (disabled by default; toggle in settings or via <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>Space</kbd>).
- **Audio Feedback:** Optional subtle sound cues when opening and closing text input with adjustable volume (`0–200%`) and quick preview.
- **Smart Game Mode (Panic-Cancel & Draft Recovery):** Automatically detects active games (stays standard in browsers, players, and Discord) and can be toggled anytime via <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>G</kbd>. If you shoot (LMB/RMB) or make a sharp mouse flick while typing, capture immediately stops so movement keys (`WASD`) are never hijacked—and your unfinished text is kept for **20 seconds** (configurable) to resume after combat.

### 🖥️ On-Screen Text & Chat Mirroring
- **Flexible Overlay Styles:** Choose between minimalist floating text (with a discreet purple oval indicator), text on a dark bordered frame, or a **resizable multi-line interactive input window** with persistent dimensions and compact Repeat/Stop/Speak controls.
- **Full Visual Customization & Accidental-Close Draft:** Optional **mouse-drag toggle** (works for plain text, framed box, or interactive window), custom colors, outlines, live preview, an **OBS Visibility Toggle**, and **Accidental-Close Draft Recovery** (**5 s** default, configurable) so closing input accidentally never loses unfinished text.
- **Mirror to Twitch Chat:** Automatically post your spoken messages into your Twitch chat so viewers feel like they're chatting directly with you.

### 🎬 OBS Subtitles with Lip-Sync
- **Browser Source Ready:** Connect animated subtitles to OBS via a local port in seconds.
- **Dynamic Lip-Sync Animation:** The default letter-by-letter reveal automatically adapts its speed to match the exact duration of the spoken TTS audio.
- **Rich Animations & Custom CSS:** Combine entrance and idle animations, pick from 4 blinking cursor styles, enable live typing preview, or edit `index.html` and `style.css` directly.
- **Interactive Browser Customizer:** Launch a built-in web configurator in your browser to tweak animations, preview results live, and save settings straight back into SilentVoice.

### 🔊 Audio Routing & System
- **One-Click Dependencies:** Install **Virtual Audio Cable** and **FFmpeg** directly from the app.
- **3-Channel Audio Mixer & Live Headphone Hot-Plug:** Independent volume sliders for Virtual Cable output (Discord, games, OBS), local headphone monitoring (automatically adapts if you plug/unplug headphones mid-playback), and **Microphone Mixing** (blend your real mic and SilentVoice into a single virtual mic).
- **VBAN Network Audio:** Stream TTS audio over your local network to a dedicated streaming PC.
- **Accessible & Multilingual UI:** Auto-detects your OS language. If your language isn't built in yet, an auto-translated welcome window lets you translate the entire UI via AI or add your own language file. Includes contextual `!` help buttons, hover tooltips, system tray controls, Windows autostart, and modular settings import/export.

---

## 🚀 Quick Start

1. **Download & Install:** Run the SilentVoice installer and launch the app (follow the 3-step welcome guide on first launch).
2. **Install a Voice:** Open **Voice & TTS**, click to install a local TTS engine or **Edge TTS** in one click, and pick your preferred voice.
3. **Set Up Audio & Speak:** Install **Virtual Audio Cable** under **Audio & System**, select it as your input device in Discord/games, press <kbd>Ctrl</kbd> + <kbd>Space</kbd>, type, and hit <kbd>Enter</kbd>!

---

## 💻 System Requirements

| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| **OS** | Windows 10 (64-bit) | Windows 10 / 11 (64-bit) |
| **CPU** | Dual-core x64 (for Cloud & CPU TTS) | Quad-core+ x64 |
| **GPU** | Integrated graphics | NVIDIA GPU with CUDA (for Voice Cloning & local LLMs) |
| **RAM** | 4 GB | 8 GB – 16 GB |
| **Audio** | Any sound card | Virtual Audio Cable (1-click install inside app) |

---

## 💡 The Story Behind SilentVoice

SilentVoice started with a simple realization while watching VTuber podcasts and live streams: non-verbal creators and gamers who rely on text-to-speech often face an unfair choice between clunky, robotic free tools and expensive, overly complex setups. When a non-verbal viewer reached out asking for a way to have an AI speak for her on stream, it sparked the mission to build a tool that *anyone*—even without technical expertise—could install in minutes and rely on daily.

What began as an effort to help one non-verbal streamer grew into a multilingual, accessible desktop companion designed so that anyone who cannot or prefers not to speak out loud can still have a clear, expressive voice in games, calls, and communities.

---

## 💖 Support the Project

If SilentVoice helps you communicate, stream, or game more comfortably, consider supporting further development:

- ⭐ **Star this repository** to help others discover SilentVoice
- 🐛 **Report bugs or suggest features** via [Issues](https://github.com/blaite164/SilentVoice-TTS/issues)
- ☕ **Support the developer:**
  - 💳 **Bank Card (Visa / Mastercard):** [Donatello](https://donatello.to/B1M)
  - 🅿️ **PayPal (International):** [Ko-fi](https://ko-fi.com/b1m164/tip)
  - 🪙 **Cryptocurrency (Worldwide):** [Crypto wallets](https://b1m164.notion.site/Donate-via-cryptocurrency-3f6eff284d6780289505cfaa2b43f748)

---

## 📄 License

This project is licensed under the [GNU General Public License v3.0](LICENSE) — see the `LICENSE` file for details.
