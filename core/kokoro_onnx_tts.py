"""
core/kokoro_onnx_tts.py — Локальний CPU ONNX FP32 рушій Kokoro-82M (24 kHz).

Працює на вже вбудованих залежностях (`onnxruntime`, `numpy`, `piper.phonemize_espeak`)
без збільшення розміру інсталятора. Файли моделі (`~337 МБ`) завантажуються on-demand
у `data/engines/kokoro_82m/onnx/`.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.config import ENGINES_DIR
from core.i18n import tr

KOKORO_ONNX_DIR = ENGINES_DIR / "kokoro_82m" / "onnx"
KOKORO_ONNX_MODEL_FILE = "kokoro-v1.0.onnx"
KOKORO_ONNX_VOICES_FILE = "voices-v1.0.bin"
KOKORO_ONNX_CONFIG_FILE = "config.json"

KOKORO_ONNX_RELEASE_BASE = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"
)
KOKORO_ONNX_CONFIG_URL = "https://huggingface.co/hexgrad/Kokoro-82M/raw/main/config.json"

KOKORO_ONNX_DOWNLOAD_FILES: List[Tuple[str, str, int, int]] = [
    (KOKORO_ONNX_MODEL_FILE, f"{KOKORO_ONNX_RELEASE_BASE}/{KOKORO_ONNX_MODEL_FILE}", 2, 86),
    (KOKORO_ONNX_VOICES_FILE, f"{KOKORO_ONNX_RELEASE_BASE}/{KOKORO_ONNX_VOICES_FILE}", 88, 10),
    (KOKORO_ONNX_CONFIG_FILE, KOKORO_ONNX_CONFIG_URL, 98, 1),
]

KOKORO_ONNX_SAMPLE_RATE = 24000
_MAX_PHONEMES_PER_CHUNK = 480

_MISAKI_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
    ("tʃ", "ʧ"),
    ("dʒ", "ʤ"),
    ("eɪ", "A"),
    ("aɪ", "I"),
    ("ɔɪ", "Y"),
    ("aʊ", "W"),
    ("oʊ", "O"),
    ("əʊ", "Q"),
)

_CLAUSE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+")


def is_kokoro_onnx_installed() -> bool:
    """Повертає True, якщо всі 3 файли Kokoro ONNX FP32 встановлені."""
    if not KOKORO_ONNX_DIR.is_dir():
        return False
    for fname in (KOKORO_ONNX_MODEL_FILE, KOKORO_ONNX_VOICES_FILE, KOKORO_ONNX_CONFIG_FILE):
        p = KOKORO_ONNX_DIR / fname
        if not p.is_file() or p.stat().st_size == 0:
            return False
    return True


def normalize_espeak_ipa_for_kokoro(ipa: str) -> str:
    """Конвертує IPA з espeak-ng у односимвольний формат словника Kokoro/Misaki."""
    out = str(ipa or "")
    for src, dst in _MISAKI_REPLACEMENTS:
        out = out.replace(src, dst)
    return out


def split_kokoro_text(text: str, max_chars: int = 260) -> List[str]:
    """Розбиває довгий текст по межах речень, щоб не перевищити ліміт 510 фонем Kokoro."""
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]
    parts = [p.strip() for p in _CLAUSE_SPLIT_RE.split(clean) if p.strip()]
    chunks: List[str] = []
    cur = ""
    for part in parts:
        if not cur:
            cur = part
        elif len(cur) + 1 + len(part) <= max_chars:
            cur = f"{cur} {part}"
        else:
            chunks.append(cur)
            cur = part
    if cur:
        chunks.append(cur)
    return chunks or [clean]


class KokoroOnnxEngine:
    """Потокобезпечний ONNX Runtime CPU рушій для Kokoro-82M FP32 (24 kHz)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._loaded_dir: Optional[Path] = None
        self._session: Any = None
        self._tok_input_name: str = "tokens"
        self._vocab: Dict[str, int] = {}
        self._voices: Dict[str, np.ndarray] = {}
        self._phonemizer: Any = None

    def unload(self) -> None:
        with self._lock:
            self._loaded_dir = None
            self._session = None
            self._vocab = {}
            self._voices = {}
            self._phonemizer = None

    def _ensure_loaded_locked(self) -> None:
        if not is_kokoro_onnx_installed():
            raise RuntimeError(
                tr(
                    "tts.err.model_not_installed",
                    "Модель {name} не встановлена. Натисніть «Встановити» у списку локальних TTS.",
                    name="Kokoro-82M (CPU ONNX)",
                )
            )
        if self._loaded_dir == KOKORO_ONNX_DIR and self._session is not None:
            return

        import onnxruntime as ort
        from piper.phonemize_espeak import ESPEAK_DATA_DIR, EspeakPhonemizer

        model_path = KOKORO_ONNX_DIR / KOKORO_ONNX_MODEL_FILE
        voices_path = KOKORO_ONNX_DIR / KOKORO_ONNX_VOICES_FILE
        cfg_path = KOKORO_ONNX_DIR / KOKORO_ONNX_CONFIG_FILE

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        raw_vocab = cfg.get("vocab") or {}
        vocab: Dict[str, int] = {str(k): int(v) for k, v in raw_vocab.items()}

        voices_npz = np.load(str(voices_path), allow_pickle=False)
        try:
            voices = {str(k): np.asarray(voices_npz[k], dtype=np.float32) for k in voices_npz.files}
        finally:
            voices_npz.close()

        # За результатами CPU-бенчмарків стандартний ORT пул потоків дає найкращий час для Kokoro FP32.
        opts = ort.SessionOptions()
        sess = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        inp_names = [i.name for i in sess.get_inputs()]
        self._tok_input_name = "input_ids" if "input_ids" in inp_names else "tokens"
        self._session = sess
        self._vocab = vocab
        self._voices = voices
        self._phonemizer = EspeakPhonemizer(ESPEAK_DATA_DIR)
        self._loaded_dir = KOKORO_ONNX_DIR

    def _text_to_token_ids_locked(self, text: str, espeak_lang: str) -> List[int]:
        assert self._phonemizer is not None
        sents = self._phonemizer.phonemize(espeak_lang, text)
        ipa = normalize_espeak_ipa_for_kokoro(" ".join("".join(s) for s in sents))
        return [self._vocab[ch] for ch in ipa if ch in self._vocab]

    def _infer_ids_locked(self, ids: List[int], voice_table: np.ndarray, eff_speed: float) -> np.ndarray:
        assert self._session is not None
        if not ids:
            return np.zeros(0, dtype=np.float32)
        trimmed = ids[:510]
        style_idx = min(len(trimmed), len(voice_table) - 1)
        style = voice_table[style_idx].reshape(1, -1).astype(np.float32, copy=False)
        tokens = np.array([[0, *trimmed, 0]], dtype=np.int64)
        speed_arr = np.array([eff_speed], dtype=np.float32)
        out = self._session.run(
            None,
            {
                self._tok_input_name: tokens,
                "style": style,
                "speed": speed_arr,
            },
        )[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)

    def synthesize(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        clean = str(text or "").strip()
        if not clean:
            return np.zeros(0, dtype=np.float32), KOKORO_ONNX_SAMPLE_RATE

        v_name = str(voice or "af_heart").strip() or "af_heart"
        espeak_lang = "en-gb" if v_name.startswith("b") else "en-us"
        eff_speed = max(0.5, min(2.0, float(speed)))
        chunks = split_kokoro_text(clean)

        with self._lock:
            self._ensure_loaded_locked()
            voice_table = self._voices.get(v_name)
            if voice_table is None:
                voice_table = self._voices.get("af_heart") or next(iter(self._voices.values()))

            parts: List[np.ndarray] = []
            sil_samples = int(0.12 * KOKORO_ONNX_SAMPLE_RATE)
            for chunk in chunks:
                ids = self._text_to_token_ids_locked(chunk, espeak_lang)
                if not ids:
                    continue
                for offset in range(0, len(ids), _MAX_PHONEMES_PER_CHUNK):
                    sub_ids = ids[offset : offset + _MAX_PHONEMES_PER_CHUNK]
                    wav = self._infer_ids_locked(sub_ids, voice_table, eff_speed)
                    if len(wav) == 0:
                        continue
                    if parts and sil_samples > 0:
                        parts.append(np.zeros(sil_samples, dtype=np.float32))
                    parts.append(wav)

            if not parts:
                return np.zeros(0, dtype=np.float32), KOKORO_ONNX_SAMPLE_RATE
            merged = np.concatenate(parts).astype(np.float32, copy=False)
            return merged, KOKORO_ONNX_SAMPLE_RATE


kokoro_onnx_engine = KokoroOnnxEngine()
