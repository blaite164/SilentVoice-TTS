"""Two-tier (RAM + disk) smart audio cache with frequency pinning for repeated TTS phrases."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import soundfile as sf

_MAX_SINGLE_DISK_BYTES = 10 * 1024 * 1024


def hash_cache_key(key_parts: Tuple[Any, ...]) -> str:
    """Create a stable SHA-256 hex digest (32 chars) from structured TTS cache key parts."""
    payload = json.dumps(key_parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class TTSAudioCache:
    """
    L1 (RAM LRU) + L2 (Disk WAV + index.json) cache for synthesized TTS chunks.

    - Keeps up to `disk_recent_limit` recent unpinned phrases on disk.
    - Promotes any phrase used at least `pin_threshold` times (e.g. 3+) to `pinned`,
      storing up to `disk_pinned_limit` frequent phrases long-term across restarts.
    - Enforces `disk_max_bytes` by evicting oldest unpinned entries before pinned ones.
    """

    def __init__(
        self,
        cache_dir: Optional[Path],
        *,
        ram_max_items: int = 100,
        ram_max_bytes: int = 128 * 1024 * 1024,
        disk_recent_limit: int = 100,
        disk_pinned_limit: int = 250,
        disk_max_bytes: int = 150 * 1024 * 1024,
        pin_threshold: int = 3,
    ) -> None:
        self._dir = Path(cache_dir) if cache_dir is not None else None
        self._index_path = (self._dir / "index.json") if self._dir is not None else None
        self.ram_max_items = max(1, int(ram_max_items))
        self.ram_max_bytes = max(1024, int(ram_max_bytes))
        self.disk_recent_limit = max(1, int(disk_recent_limit))
        self.disk_pinned_limit = max(0, int(disk_pinned_limit))
        self.disk_max_bytes = max(1024, int(disk_max_bytes))
        self.pin_threshold = max(2, int(pin_threshold))

        self._lock = threading.Lock()
        self._ram: OrderedDict[str, Tuple[np.ndarray, int, str]] = OrderedDict()
        self._ram_bytes = 0
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._last_index_save = 0.0

        if self._dir is not None:
            self._load_index()

    def _wav_path(self, key_hash: str) -> Path:
        assert self._dir is not None
        return self._dir / f"{key_hash}.wav"

    def _load_index(self) -> None:
        assert self._dir is not None and self._index_path is not None
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return
        if not self._index_path.exists():
            return
        try:
            raw = json.loads(self._index_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(raw, dict):
            return

        cleaned: Dict[str, Dict[str, Any]] = {}
        dirty = False
        for key_hash, meta in raw.items():
            if not isinstance(meta, dict) or not isinstance(key_hash, str):
                dirty = True
                continue
            wav = self._wav_path(key_hash)
            if not wav.exists():
                dirty = True
                continue
            hits = max(1, int(meta.get("hits", 1) or 1))
            pinned = bool(meta.get("pinned", False)) or hits >= self.pin_threshold
            cleaned[key_hash] = {
                "hits": hits,
                "pinned": pinned,
                "last_used": float(meta.get("last_used", 0.0) or 0.0),
                "bytes": int(meta.get("bytes", wav.stat().st_size) or wav.stat().st_size),
                "title": str(meta.get("title", "") or ""),
            }
        self._entries = cleaned
        if self._evict_disk_locked() or dirty:
            self._save_index_locked(force=True)

    def _save_index_locked(self, force: bool = False) -> None:
        if self._dir is None or self._index_path is None:
            return
        now = time.monotonic()
        if not force and (now - self._last_index_save) < 1.5:
            return
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp = self._index_path.with_name(f"index_{os.getpid()}_{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(self._entries, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self._index_path)
            self._last_index_save = now
        except Exception:
            pass

    def _ram_put_locked(self, key_hash: str, arr: np.ndarray, sr: int, title: str) -> None:
        arr_bytes = int(arr.nbytes)
        if arr_bytes > self.ram_max_bytes:
            return
        old = self._ram.pop(key_hash, None)
        if old is not None:
            self._ram_bytes = max(0, self._ram_bytes - int(old[0].nbytes))
        self._ram[key_hash] = (arr, int(sr), title)
        self._ram_bytes += arr_bytes
        while len(self._ram) > self.ram_max_items or (
            self._ram_bytes > self.ram_max_bytes and len(self._ram) > 1
        ):
            _, evicted = self._ram.popitem(last=False)
            self._ram_bytes = max(0, self._ram_bytes - int(evicted[0].nbytes))

    def _remove_disk_entry_locked(self, key_hash: str) -> None:
        self._entries.pop(key_hash, None)
        if self._dir is not None:
            try:
                self._wav_path(key_hash).unlink(missing_ok=True)
            except OSError:
                pass

    def _evict_disk_locked(self) -> bool:
        if self._dir is None:
            return False
        evicted_any = False

        unpinned = [(k, v) for k, v in self._entries.items() if not v.get("pinned", False)]
        if len(unpinned) > self.disk_recent_limit:
            unpinned.sort(key=lambda item: float(item[1].get("last_used", 0.0)))
            for k, _ in unpinned[: len(unpinned) - self.disk_recent_limit]:
                self._remove_disk_entry_locked(k)
                evicted_any = True

        pinned = [(k, v) for k, v in self._entries.items() if v.get("pinned", False)]
        if len(pinned) > self.disk_pinned_limit:
            pinned.sort(key=lambda item: (int(item[1].get("hits", 0)), float(item[1].get("last_used", 0.0))))
            for k, _ in pinned[: len(pinned) - self.disk_pinned_limit]:
                self._remove_disk_entry_locked(k)
                evicted_any = True

        total_bytes = sum(int(v.get("bytes", 0)) for v in self._entries.values())
        if total_bytes > self.disk_max_bytes:
            # Always evict unpinned recent entries before pinned frequent entries.
            candidates = sorted(
                self._entries.items(),
                key=lambda item: (
                    1 if item[1].get("pinned", False) else 0,
                    float(item[1].get("last_used", 0.0)),
                ),
            )
            for k, meta in candidates:
                if total_bytes <= self.disk_max_bytes or len(self._entries) <= 1:
                    break
                total_bytes -= int(meta.get("bytes", 0))
                self._remove_disk_entry_locked(k)
                evicted_any = True

        return evicted_any

    def _bump_hit_locked(self, key_hash: str) -> None:
        meta = self._entries.get(key_hash)
        if meta is None:
            return
        was_pinned = bool(meta.get("pinned", False))
        hits = int(meta.get("hits", 1)) + 1
        meta["hits"] = hits
        meta["last_used"] = time.time()
        if hits >= self.pin_threshold:
            meta["pinned"] = True
        self._save_index_locked(force=(not was_pinned and bool(meta.get("pinned", False))))

    def lookup(self, key_parts: Tuple[Any, ...]) -> Optional[Tuple[np.ndarray, int, str]]:
        """Look up cached audio in L1 RAM or L2 Disk, updating LRU and hit counts."""
        key_hash = hash_cache_key(key_parts)
        disk_wav: Optional[Path] = None
        disk_title = ""

        with self._lock:
            ram_hit = self._ram.get(key_hash)
            if ram_hit is not None:
                self._ram.move_to_end(key_hash)
                self._bump_hit_locked(key_hash)
                samples, sr, title = ram_hit
                return samples.copy(), sr, title

            if self._dir is not None and key_hash in self._entries:
                wav = self._wav_path(key_hash)
                if wav.exists():
                    disk_wav = wav
                    disk_title = str(self._entries[key_hash].get("title", "") or "")
                else:
                    self._entries.pop(key_hash, None)
                    self._save_index_locked(force=True)

        if disk_wav is None:
            return None

        try:
            data, sr = sf.read(str(disk_wav), dtype="float32", always_2d=False)
            arr = np.array(data, dtype=np.float32, copy=True, order="C")
            if len(arr) == 0 or int(sr) <= 0:
                raise ValueError("Empty cached WAV")
        except Exception:
            with self._lock:
                self._remove_disk_entry_locked(key_hash)
                self._save_index_locked(force=True)
            return None

        with self._lock:
            self._ram_put_locked(key_hash, arr, int(sr), disk_title)
            if key_hash in self._entries:
                self._bump_hit_locked(key_hash)
                self._save_index_locked(force=True)
            return arr.copy(), int(sr), disk_title

    def store(self, key_parts: Tuple[Any, ...], samples: np.ndarray, sr: int, title: str) -> None:
        """Save synthesized audio to L1 RAM and L2 Disk (as PCM16 WAV)."""
        if samples is None or len(samples) == 0 or int(sr) <= 0:
            return
        key_hash = hash_cache_key(key_parts)
        arr = np.array(samples, dtype=np.float32, copy=True, order="C")

        with self._lock:
            self._ram_put_locked(key_hash, arr, int(sr), title)

        if self._dir is None:
            return
        est_pcm16_bytes = int(arr.size) * 2 + 44
        if est_pcm16_bytes > _MAX_SINGLE_DISK_BYTES or est_pcm16_bytes > self.disk_max_bytes:
            return

        wav_path = self._wav_path(key_hash)
        tmp_wav = wav_path.with_name(f"{key_hash}_{os.getpid()}_{threading.get_ident()}.tmp.wav")
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            clipped = np.clip(arr, -1.0, 1.0)
            sf.write(str(tmp_wav), clipped, int(sr), subtype="PCM_16", format="WAV")
            os.replace(tmp_wav, wav_path)
            disk_bytes = int(wav_path.stat().st_size)
        except Exception:
            try:
                tmp_wav.unlink(missing_ok=True)
            except OSError:
                pass
            return

        with self._lock:
            old_meta = self._entries.get(key_hash)
            hits = max(1, int(old_meta.get("hits", 1))) if old_meta else 1
            pinned = (bool(old_meta.get("pinned", False)) if old_meta else False) or hits >= self.pin_threshold
            self._entries[key_hash] = {
                "hits": hits,
                "pinned": pinned,
                "last_used": time.time(),
                "bytes": disk_bytes,
                "title": title,
            }
            self._evict_disk_locked()
            self._save_index_locked(force=True)

    def flush(self) -> None:
        """Persist pending hit/timestamp updates to index.json."""
        with self._lock:
            self._save_index_locked(force=True)

    def clear(self) -> None:
        """Clear both RAM and disk caches."""
        with self._lock:
            self._ram.clear()
            self._ram_bytes = 0
            self._entries.clear()
            if self._dir is not None and self._dir.exists():
                for p in list(self._dir.glob("*.wav")) + list(self._dir.glob("*.tmp")):
                    try:
                        p.unlink(missing_ok=True)
                    except OSError:
                        pass
                if self._index_path is not None:
                    try:
                        self._index_path.unlink(missing_ok=True)
                    except OSError:
                        pass

    def stats(self) -> Dict[str, int]:
        with self._lock:
            pinned = sum(1 for v in self._entries.values() if v.get("pinned", False))
            disk_bytes = sum(int(v.get("bytes", 0)) for v in self._entries.values())
            return {
                "ram_items": len(self._ram),
                "ram_bytes": self._ram_bytes,
                "disk_items": len(self._entries),
                "pinned_items": pinned,
                "recent_items": len(self._entries) - pinned,
                "disk_bytes": disk_bytes,
            }
