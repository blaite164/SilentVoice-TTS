"""
app_logger.py — Система логування SilentVoice TTS.

Кожен запуск програми створює окремий файл логу у папці ``logs/``
(наприклад, ``app_2026-11-23_20-25-10.log``), додає точний час до кожного рядка
та автоматично зберігає лише останні ``MAX_LOG_FILES`` (5) логів.
"""

from __future__ import annotations

import datetime
import os
import threading
from pathlib import Path
from typing import TextIO

MAX_LOG_FILES = 5
LOG_FILE_PREFIX = "app_"
LOG_FILE_SUFFIX = ".log"


def format_log_timestamp(now: datetime.datetime | None = None) -> str:
    """Повертає мітку часу у форматі ``[YYYY-MM-DD HH:MM:SS]``."""
    dt = now or datetime.datetime.now()
    return dt.strftime("[%Y-%m-%d %H:%M:%S]")


def prune_old_logs(logs_dir: Path, keep_count: int = MAX_LOG_FILES) -> list[Path]:
    """
    Видаляє старі файли ``app_*.log`` у ``logs_dir``, залишаючи не більше ``keep_count``
    найновіших файлів. Повертає список видалених файлів.
    """
    removed: list[Path] = []
    if keep_count < 0:
        keep_count = 0
    if not logs_dir.exists():
        return removed

    candidates = [
        p
        for p in logs_dir.iterdir()
        if p.is_file() and p.name.startswith(LOG_FILE_PREFIX) and p.name.endswith(LOG_FILE_SUFFIX)
    ]

    def _sort_key(path: Path) -> tuple[float, str]:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (mtime, path.name)

    # Найновіші спочатку
    candidates.sort(key=_sort_key, reverse=True)
    for stale in candidates[keep_count:]:
        try:
            stale.unlink()
            removed.append(stale)
        except OSError:
            # Файл може бути зайнятий іншим процесом — пропускаємо
            pass
    return removed


def create_session_log_path(
    logs_dir: Path,
    now: datetime.datetime | None = None,
    max_files: int = MAX_LOG_FILES,
) -> Path:
    """
    Готує папку ``logs_dir``, очищує старі логи (залишаючи ``max_files - 1``)
    та повертає унікальний шлях для нового файлу логу поточного запуску.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    prune_old_logs(logs_dir, keep_count=max(0, max_files - 1))

    dt = now or datetime.datetime.now()
    stamp = dt.strftime("%Y-%m-%d_%H-%M-%S")
    base_path = logs_dir / f"{LOG_FILE_PREFIX}{stamp}{LOG_FILE_SUFFIX}"
    if not base_path.exists():
        return base_path

    pid = os.getpid()
    pid_path = logs_dir / f"{LOG_FILE_PREFIX}{stamp}_{pid}{LOG_FILE_SUFFIX}"
    if not pid_path.exists():
        return pid_path

    counter = 1
    while True:
        candidate = logs_dir / f"{LOG_FILE_PREFIX}{stamp}_{pid}_{counter}{LOG_FILE_SUFFIX}"
        if not candidate.exists():
            return candidate
        counter += 1


class TimestampedLogStream:
    """
    Потокобезпечна обгортка над файлом логу (та опційною консоллю),
    яка додає ``[YYYY-MM-DD HH:MM:SS] `` на початок кожного нового рядка.
    """

    def __init__(
        self,
        log_file: TextIO,
        console_stream: TextIO | None = None,
        lock: threading.RLock | None = None,
    ) -> None:
        self._log_file = log_file
        self._console_stream = console_stream
        self._lock = lock or threading.RLock()
        self._at_line_start = True

    @property
    def encoding(self) -> str:
        return getattr(self._log_file, "encoding", "utf-8") or "utf-8"

    @property
    def errors(self) -> str:
        return getattr(self._log_file, "errors", "replace") or "replace"

    @property
    def closed(self) -> bool:
        return bool(getattr(self._log_file, "closed", False))

    def isatty(self) -> bool:
        if self._console_stream is not None:
            try:
                return bool(self._console_stream.isatty())
            except Exception:
                return False
        return False

    def fileno(self) -> int:
        if self._console_stream is not None:
            try:
                return self._console_stream.fileno()
            except Exception:
                pass
        return self._log_file.fileno()

    def reconfigure(self, *args, **kwargs) -> None:
        if self._console_stream is not None and hasattr(self._console_stream, "reconfigure"):
            try:
                self._console_stream.reconfigure(*args, **kwargs)
            except Exception:
                pass

    def _format_text(self, text: str) -> str:
        if not text:
            return ""
        parts: list[str] = []
        for chunk in text.splitlines(keepends=True):
            if self._at_line_start and chunk not in ("\n", "\r\n", "\r"):
                parts.append(f"{format_log_timestamp()} {chunk}")
            else:
                parts.append(chunk)
            self._at_line_start = chunk.endswith(("\n", "\r"))
        return "".join(parts)

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return 0
        with self._lock:
            formatted = self._format_text(text)
            try:
                if not self._log_file.closed:
                    self._log_file.write(formatted)
                    if "\n" in formatted:
                        self._log_file.flush()
            except Exception:
                pass

            if self._console_stream is not None:
                try:
                    self._console_stream.write(formatted)
                    if "\n" in formatted:
                        self._console_stream.flush()
                except UnicodeEncodeError:
                    try:
                        enc = getattr(self._console_stream, "encoding", None) or "utf-8"
                        safe = formatted.encode(enc, errors="replace").decode(enc, errors="replace")
                        self._console_stream.write(safe)
                        if "\n" in safe:
                            self._console_stream.flush()
                    except Exception:
                        pass
                except Exception:
                    pass
        return len(text)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        with self._lock:
            try:
                if not self._log_file.closed:
                    self._log_file.flush()
            except Exception:
                pass
            if self._console_stream is not None:
                try:
                    self._console_stream.flush()
                except Exception:
                    pass

    def close(self) -> None:
        with self._lock:
            try:
                if not self._log_file.closed:
                    self._log_file.flush()
                    self._log_file.close()
            except Exception:
                pass


def open_session_log_streams(
    logs_dir: Path,
    orig_stdout: TextIO | None,
    orig_stderr: TextIO | None,
    max_files: int = MAX_LOG_FILES,
) -> tuple[Path, TextIO, TimestampedLogStream, TimestampedLogStream]:
    """
    Відкриває новий файл логу в ``logs_dir`` та повертає
    ``(log_path, raw_file, stdout_stream, stderr_stream)``.
    """
    log_path = create_session_log_path(logs_dir, max_files=max_files)
    raw_file = open(log_path, "a", encoding="utf-8", errors="replace", buffering=1)
    shared_lock = threading.RLock()

    for stream in (orig_stdout, orig_stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    stdout_stream = TimestampedLogStream(raw_file, console_stream=orig_stdout, lock=shared_lock)
    stderr_stream = TimestampedLogStream(raw_file, console_stream=orig_stderr, lock=shared_lock)
    return log_path, raw_file, stdout_stream, stderr_stream
