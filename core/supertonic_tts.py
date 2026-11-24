"""
core/supertonic_tts.py — Локальний CPU ONNX рушій Supertonic 3 (Supertone, 44.1 kHz, 31 мова).

Працює на вже вбудованому в додаток `onnxruntime` + `numpy` без GPU та без додаткових залежностей.
Файли моделі та стилі голосів завантажуються on-demand у `data/engines/supertonic3/`.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unicodedata import normalize

import numpy as np

from core.config import ENGINES_DIR
from core.i18n import tr

SUPERTONIC3_DIR = ENGINES_DIR / "supertonic3"
SUPERTONIC3_HF_BASE = "https://huggingface.co/Supertone/supertonic-3/resolve/main"
SUPERTONIC3_DEFAULT_STEPS = 6

SUPERTONIC3_AVAILABLE_LANGS = frozenset(
    (
        "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi",
        "fr", "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl", "pt", "ro",
        "ru", "sk", "sl", "sv", "tr", "uk", "vi", "na",
    )
)

SUPERTONIC3_VOICE_NAMES: Tuple[str, ...] = (
    "F1", "F2", "F3", "F4", "F5",
    "M1", "M2", "M3", "M4", "M5",
)

SUPERTONIC3_REQUIRED_ONNX: Tuple[str, ...] = (
    "onnx/duration_predictor.onnx",
    "onnx/text_encoder.onnx",
    "onnx/vector_estimator.onnx",
    "onnx/vocoder.onnx",
    "onnx/tts.json",
    "onnx/unicode_indexer.json",
)

SUPERTONIC3_DOWNLOAD_FILES: List[Tuple[str, str]] = [
    (rel, f"{SUPERTONIC3_HF_BASE}/{rel}")
    for rel in (
        *SUPERTONIC3_REQUIRED_ONNX,
        *(f"voice_styles/{v}.json" for v in SUPERTONIC3_VOICE_NAMES),
    )
]

_EMOJI_RE = re.compile(
    "[\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\u2600-\u26ff"
    "\u2700-\u27bf"
    "\U0001f1e6-\U0001f1ff]+",
    flags=re.UNICODE,
)

_CHAR_REPLACEMENTS: Dict[str, str] = {
    "–": "-",
    "‑": "-",
    "—": "-",
    "_": " ",
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
    "´": "'",
    "`": "'",
    "[": " ",
    "]": " ",
    "|": " ",
    "/": " ",
    "#": " ",
    "→": " ",
    "←": " ",
}

_EXPR_REPLACEMENTS: Dict[str, str] = {
    "@": " at ",
    "e.g.,": "for example, ",
    "i.e.,": "that is, ",
}

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)"
    r"(?<!Ph\.D\.)(?<!etc\.)(?<!e\.g\.)(?<!i\.e\.)(?<!vs\.)(?<!Inc\.)"
    r"(?<!Ltd\.)(?<!Co\.)(?<!Corp\.)(?<!St\.)(?<!Ave\.)(?<!Blvd\.)"
    r"(?<!\b[A-Z]\.)(?<=[.!?])\s+"
)


def is_supertonic3_installed() -> bool:
    """Повертає True, якщо всі ONNX-файли, конфіг і хоча б один стиль голосу Supertonic 3 встановлені."""
    if not SUPERTONIC3_DIR.is_dir():
        return False
    for rel in SUPERTONIC3_REQUIRED_ONNX:
        p = SUPERTONIC3_DIR / rel
        if not p.is_file() or p.stat().st_size == 0:
            return False
    styles_dir = SUPERTONIC3_DIR / "voice_styles"
    if not styles_dir.is_dir():
        return False
    return any((styles_dir / f"{v}.json").is_file() for v in SUPERTONIC3_VOICE_NAMES)


def _length_to_mask(lengths: np.ndarray, max_len: Optional[int] = None) -> np.ndarray:
    m = int(max_len if max_len is not None else int(np.max(lengths)))
    ids = np.arange(0, m, dtype=np.int64)
    mask = (ids < np.expand_dims(lengths, axis=1)).astype(np.float32)
    return mask.reshape(-1, 1, m)


def _get_latent_mask(
    wav_lengths: np.ndarray,
    base_chunk_size: int,
    chunk_compress_factor: int,
) -> np.ndarray:
    latent_size = base_chunk_size * chunk_compress_factor
    latent_lengths = (wav_lengths + latent_size - 1) // latent_size
    return _length_to_mask(latent_lengths)


def _chunk_text(text: str, max_len: int = 300) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    chunks: List[str] = []
    for paragraph in paragraphs:
        sentences = _SENTENCE_SPLIT_RE.split(paragraph)
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current) + len(sentence) + 1 <= max_len:
                current = f"{current} {sentence}".strip()
            else:
                if current:
                    chunks.append(current)
                current = sentence
        if current:
            chunks.append(current)
    return chunks or [text.strip()]


def _normalize_supertonic_lang(lang: str) -> str:
    code = str(lang or "en").strip().lower()
    if code == "ua":
        code = "uk"
    return code if code in SUPERTONIC3_AVAILABLE_LANGS else "na"


def preprocess_supertonic_text(text: str, lang: str) -> str:
    clean = normalize("NFKD", str(text or ""))
    clean = _EMOJI_RE.sub("", clean)
    for k, v in _CHAR_REPLACEMENTS.items():
        clean = clean.replace(k, v)
    clean = re.sub(r"[♥☆♡©\\]", "", clean)
    for k, v in _EXPR_REPLACEMENTS.items():
        clean = clean.replace(k, v)
    for punct in (",", ".", "!", "?", ";", ":", "'"):
        clean = clean.replace(f" {punct}", punct)
    while '""' in clean:
        clean = clean.replace('""', '"')
    while "''" in clean:
        clean = clean.replace("''", "'")
    while "``" in clean:
        clean = clean.replace("``", "`")
    clean = re.sub(r"\s+", " ", clean).strip()
    if clean and not re.search(r"[.!?;:,'\"')\]}…。」』】〉》›»]$", clean):
        clean += "."
    eff_lang = _normalize_supertonic_lang(lang)
    return f"<{eff_lang}>{clean}</{eff_lang}>"


class _UnicodeProcessor:
    def __init__(self, unicode_indexer_path: Path) -> None:
        raw = json.loads(unicode_indexer_path.read_text(encoding="utf-8"))
        self.indexer: List[int] = [int(x) for x in raw]
        self._space_id = self.indexer[ord(" ")] if ord(" ") < len(self.indexer) and self.indexer[ord(" ")] >= 0 else 0

    def __call__(self, text_list: List[str], lang_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        processed = [preprocess_supertonic_text(t, lg) for t, lg in zip(text_list, lang_list)]
        encoded: List[List[int]] = []
        idx_len = len(self.indexer)
        for text in processed:
            ids: List[int] = []
            for ch in text:
                code = ord(ch)
                if code < idx_len:
                    mapped = self.indexer[code]
                    if mapped >= 0:
                        ids.append(mapped)
                        continue
                # Пропускаємо невідомі символи замість падіння IndexError
                if self._space_id >= 0:
                    ids.append(self._space_id)
            if not ids:
                ids.append(self._space_id)
            encoded.append(ids)

        lengths = np.array([len(seq) for seq in encoded], dtype=np.int64)
        max_l = max(1, int(lengths.max()))
        text_ids = np.zeros((len(encoded), max_l), dtype=np.int64)
        for i, seq in enumerate(encoded):
            text_ids[i, : len(seq)] = np.array(seq, dtype=np.int64)
        text_mask = _length_to_mask(lengths, max_len=max_l)
        return text_ids, text_mask


class Supertonic3Engine:
    """Потокобезпечний ONNX CPU рушій для Supertonic 3 (44.1 kHz)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._loaded_dir: Optional[Path] = None
        self._cfgs: Optional[Dict[str, Any]] = None
        self._processor: Optional[_UnicodeProcessor] = None
        self._dp_ort: Any = None
        self._text_enc_ort: Any = None
        self._vector_est_ort: Any = None
        self._vocoder_ort: Any = None
        self._styles_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.sample_rate: int = 44100
        self._base_chunk_size: int = 512
        self._chunk_compress_factor: int = 6
        self._ldim: int = 24

    def unload(self) -> None:
        with self._lock:
            self._loaded_dir = None
            self._cfgs = None
            self._processor = None
            self._dp_ort = None
            self._text_enc_ort = None
            self._vector_est_ort = None
            self._vocoder_ort = None
            self._styles_cache.clear()

    def _ensure_loaded_locked(self) -> None:
        if not is_supertonic3_installed():
            raise RuntimeError(
                tr(
                    "tts.err.model_not_installed",
                    "Модель {name} не встановлена. Натисніть «Встановити» у списку локальних TTS.",
                    name="Supertonic 3",
                )
            )
        if self._loaded_dir == SUPERTONIC3_DIR and self._vocoder_ort is not None:
            return

        import onnxruntime as ort

        onnx_dir = SUPERTONIC3_DIR / "onnx"
        cfgs = json.loads((onnx_dir / "tts.json").read_text(encoding="utf-8"))
        opts = ort.SessionOptions()
        try:
            phys_threads = max(2, min(8, (os.cpu_count() or 4) // 2))
            opts.intra_op_num_threads = phys_threads
            opts.inter_op_num_threads = 1
            if hasattr(ort, "GraphOptimizationLevel"):
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if hasattr(opts, "add_session_config_entry"):
                opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
                opts.add_session_config_entry("session.set_denormal_as_zero", "1")
        except Exception:
            pass
        providers = ["CPUExecutionProvider"]

        self._dp_ort = ort.InferenceSession(str(onnx_dir / "duration_predictor.onnx"), sess_options=opts, providers=providers)
        self._text_enc_ort = ort.InferenceSession(str(onnx_dir / "text_encoder.onnx"), sess_options=opts, providers=providers)
        self._vector_est_ort = ort.InferenceSession(str(onnx_dir / "vector_estimator.onnx"), sess_options=opts, providers=providers)
        self._vocoder_ort = ort.InferenceSession(str(onnx_dir / "vocoder.onnx"), sess_options=opts, providers=providers)
        self._processor = _UnicodeProcessor(onnx_dir / "unicode_indexer.json")
        self._cfgs = cfgs
        self.sample_rate = int(cfgs["ae"]["sample_rate"])
        self._base_chunk_size = int(cfgs["ae"]["base_chunk_size"])
        self._chunk_compress_factor = int(cfgs["ttl"]["chunk_compress_factor"])
        self._ldim = int(cfgs["ttl"]["latent_dim"])
        self._styles_cache.clear()
        self._loaded_dir = SUPERTONIC3_DIR

    def _get_style_locked(self, voice_name: str) -> Tuple[np.ndarray, np.ndarray]:
        v_clean = str(voice_name or "F1").strip().upper()
        if v_clean.startswith("SUPERTONIC3_"):
            v_clean = v_clean[len("SUPERTONIC3_") :]
        if v_clean not in SUPERTONIC3_VOICE_NAMES:
            v_clean = "F1"

        cached = self._styles_cache.get(v_clean)
        if cached is not None:
            return cached

        style_path = SUPERTONIC3_DIR / "voice_styles" / f"{v_clean}.json"
        if not style_path.is_file():
            for cand in SUPERTONIC3_VOICE_NAMES:
                alt = SUPERTONIC3_DIR / "voice_styles" / f"{cand}.json"
                if alt.is_file():
                    style_path = alt
                    v_clean = cand
                    break
        if not style_path.is_file():
            raise FileNotFoundError(f"Supertonic 3 voice style not found: {style_path}")

        data = json.loads(style_path.read_text(encoding="utf-8"))
        ttl_dims = data["style_ttl"]["dims"]
        dp_dims = data["style_dp"]["dims"]
        ttl_arr = np.array(data["style_ttl"]["data"], dtype=np.float32).reshape(1, int(ttl_dims[1]), int(ttl_dims[2]))
        dp_arr = np.array(data["style_dp"]["data"], dtype=np.float32).reshape(1, int(dp_dims[1]), int(dp_dims[2]))
        self._styles_cache[v_clean] = (ttl_arr, dp_arr)
        return ttl_arr, dp_arr

    def _infer_chunk_locked(
        self,
        chunk: str,
        lang: str,
        style_ttl: np.ndarray,
        style_dp: np.ndarray,
        *,
        total_step: int,
        speed: float,
    ) -> np.ndarray:
        assert self._processor is not None
        text_ids, text_mask = self._processor([chunk], [lang])
        dur_onnx, *_ = self._dp_ort.run(
            None,
            {"text_ids": text_ids, "style_dp": style_dp, "text_mask": text_mask},
        )
        eff_speed = max(0.5, min(2.0, float(speed) * 1.05))
        dur_onnx = np.maximum(0.15, np.asarray(dur_onnx, dtype=np.float32) / eff_speed)
        text_emb, *_ = self._text_enc_ort.run(
            None,
            {"text_ids": text_ids, "style_ttl": style_ttl, "text_mask": text_mask},
        )

        wav_len_max = float(np.max(dur_onnx)) * self.sample_rate
        wav_lengths = np.maximum(1, (dur_onnx * self.sample_rate).astype(np.int64))
        chunk_size = self._base_chunk_size * self._chunk_compress_factor
        latent_len = max(1, int((wav_len_max + chunk_size - 1) // chunk_size))
        latent_dim = self._ldim * self._chunk_compress_factor

        xt = np.random.randn(1, latent_dim, latent_len).astype(np.float32)
        latent_mask = _get_latent_mask(wav_lengths, self._base_chunk_size, self._chunk_compress_factor)
        if latent_mask.shape[-1] != latent_len:
            latent_len = int(latent_mask.shape[-1])
            xt = np.random.randn(1, latent_dim, latent_len).astype(np.float32)
        xt = xt * latent_mask

        steps = max(2, int(total_step))
        total_step_np = np.array([float(steps)], dtype=np.float32)
        for step in range(steps):
            current_step = np.array([float(step)], dtype=np.float32)
            xt, *_ = self._vector_est_ort.run(
                None,
                {
                    "noisy_latent": xt,
                    "text_emb": text_emb,
                    "style_ttl": style_ttl,
                    "text_mask": text_mask,
                    "latent_mask": latent_mask,
                    "current_step": current_step,
                    "total_step": total_step_np,
                },
            )
        wav, *_ = self._vocoder_ort.run(None, {"latent": xt})
        out = np.asarray(wav, dtype=np.float32).reshape(-1)
        expected_len = int(wav_lengths[0])
        if 0 < expected_len < len(out):
            out = out[:expected_len]
        return out

    def synthesize(
        self,
        text: str,
        voice_name: str = "F1",
        lang: str = "en",
        speed: float = 1.0,
        total_step: int = SUPERTONIC3_DEFAULT_STEPS,
    ) -> Tuple[np.ndarray, int]:
        clean = str(text or "").strip()
        if not clean:
            return np.zeros(0, dtype=np.float32), self.sample_rate
        eff_lang = _normalize_supertonic_lang(lang)
        max_len = 120 if eff_lang in ("ko", "ja") else 300
        chunks = _chunk_text(clean, max_len=max_len)

        with self._lock:
            self._ensure_loaded_locked()
            style_ttl, style_dp = self._get_style_locked(voice_name)
            parts: List[np.ndarray] = []
            sil_samples = int(0.18 * self.sample_rate)
            for idx, chunk in enumerate(chunks):
                wav_chunk = self._infer_chunk_locked(
                    chunk,
                    eff_lang,
                    style_ttl,
                    style_dp,
                    total_step=total_step,
                    speed=speed,
                )
                if len(wav_chunk) == 0:
                    continue
                if parts and sil_samples > 0:
                    parts.append(np.zeros(sil_samples, dtype=np.float32))
                parts.append(wav_chunk)
            if not parts:
                return np.zeros(0, dtype=np.float32), self.sample_rate
            merged = np.concatenate(parts).astype(np.float32, copy=False)
            return merged, self.sample_rate


supertonic3_engine = Supertonic3Engine()
