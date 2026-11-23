"""
app_logger.py — Система логування SilentVoice TTS.

Кожен запуск програми створює окремий файл логу у папці ``logs/``
(наприклад, ``app_2026-11-23_20-25-10.log``), додає точний час до кожного рядка
та автоматично зберігає лише останні ``MAX_LOG_FILES`` (5) логів.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import re
import sys
import threading
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Callable, TextIO
from urllib.parse import urlencode

MAX_LOG_FILES = 5
MAX_CRASH_REPORT_ZIPS = 5
LOG_FILE_PREFIX = "app_"
LOG_FILE_SUFFIX = ".log"
MAX_CRASH_LOG_BYTES = 2 * 1024 * 1024
PENDING_CRASH_FILENAME = "pending_crash.json"
LAST_SESSION_FILENAME = "last_session.json"
_WIN_USER_PATH_RE = re.compile(r"([A-Za-z]:[\\/]+Users[\\/]+)([^\\/\r\n\"'<>|]+)", re.IGNORECASE)
_UNIX_USER_PATH_RE = re.compile(r"(/(?:home|Users)/)([^/\r\n\"'<>|]+)")
_SECRET_TOKEN_RE = re.compile(
    r"((?:Bearer\s+|(?:Token|api[_-]?key|secret)\s*[:=]\s*)['\"]?)([A-Za-z0-9_\-\.]{12,})",
    re.IGNORECASE,
)


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


def rotate_crash_log_if_needed(
    crash_log_path: Path,
    max_bytes: int = MAX_CRASH_LOG_BYTES,
) -> bool:
    """
    Якщо ``crash.log`` перевищує ``max_bytes``, переносить його у ``crash.log.old``,
    щоб файл не розростався безмежно, але остання історія крашів зберігалася.
    """
    try:
        if crash_log_path.exists() and crash_log_path.stat().st_size > max_bytes:
            backup = crash_log_path.with_name(crash_log_path.name + ".old")
            crash_log_path.replace(backup)
            return True
    except OSError:
        pass
    return False


def format_all_threads_dump() -> str:
    """
    Повертає повний стек викликів усіх активних Python-потоків
    (починаючи з MainThread) для діагностики зависань та deadlock'ів.
    """
    frames = sys._current_frames()
    threads_by_id = {t.ident: t for t in threading.enumerate() if t.ident is not None}
    main_ident = threading.main_thread().ident

    ordered_ids = sorted(
        frames.keys(),
        key=lambda tid: (0 if tid == main_ident else 1, threads_by_id[tid].name if tid in threads_by_id else str(tid)),
    )

    sections: list[str] = []
    for tid in ordered_ids:
        frame = frames.get(tid)
        if frame is None:
            continue
        t = threads_by_id.get(tid)
        if t is not None:
            role = "daemon" if t.daemon else "non-daemon"
            header = f"--- Thread: {t.name} (id={tid}, {role}) ---"
        else:
            header = f"--- Thread: unknown (id={tid}) ---"
        stack_txt = "".join(traceback.format_stack(frame)).rstrip()
        sections.append(f"{header}\n{stack_txt}")
    return "\n\n".join(sections)


class UIWatchdog:
    """
    Фоновий сторожовий потік, що відстежує пульс головного GUI-потоку Qt.
    Якщо UI не відповідає довше ``timeout_sec``, записує повний стек усіх потоків
    через ``on_freeze`` (у ``crash.log``), щоб можна було точно побачити причину зависання.
    """

    def __init__(
        self,
        on_freeze: Callable[[float, str], None],
        on_recover: Callable[[float], None] | None = None,
        timeout_sec: float = 12.0,
        check_interval_sec: float = 2.0,
        repeat_dump_sec: float = 30.0,
    ) -> None:
        self._on_freeze = on_freeze
        self._on_recover = on_recover
        self._timeout_sec = max(0.05, float(timeout_sec))
        self._check_interval_sec = max(0.02, float(check_interval_sec))
        self._repeat_dump_sec = max(self._timeout_sec, float(repeat_dump_sec))
        self._lock = threading.Lock()
        self._last_heartbeat = time.monotonic()
        self._freeze_started_at: float | None = None
        self._dump_count = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        with self._lock:
            self._last_heartbeat = time.monotonic()
            self._freeze_started_at = None
            self._dump_count = 0
        self._thread = threading.Thread(target=self._run, daemon=True, name="UIWatchdog")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=min(1.0, self._check_interval_sec * 2.0))

    def heartbeat(self) -> None:
        now = time.monotonic()
        recovered_after: float | None = None
        with self._lock:
            if self._freeze_started_at is not None:
                recovered_after = max(0.0, now - self._freeze_started_at)
                self._freeze_started_at = None
                self._dump_count = 0
            self._last_heartbeat = now
        if recovered_after is not None and self._on_recover is not None:
            try:
                self._on_recover(recovered_after)
            except Exception:
                pass

    def _run(self) -> None:
        while not self._stop_event.is_set():
            t_before = time.monotonic()
            if self._stop_event.wait(self._check_interval_sec):
                break
            now = time.monotonic()
            # Якщо система щойно вийшла зі сну/гібернації — скидаємо таймер, щоб уникнути хибної тривоги
            if (now - t_before) > max(self._check_interval_sec * 4.0, self._timeout_sec):
                with self._lock:
                    self._last_heartbeat = now
                continue

            should_dump = False
            stalled_sec = 0.0
            with self._lock:
                stalled_sec = now - self._last_heartbeat
                if stalled_sec >= self._timeout_sec:
                    if self._freeze_started_at is None:
                        self._freeze_started_at = self._last_heartbeat
                    if self._dump_count == 0:
                        self._dump_count = 1
                        should_dump = True
                    elif self._dump_count == 1 and stalled_sec >= self._repeat_dump_sec:
                        self._dump_count = 2
                        should_dump = True

            if should_dump:
                try:
                    dump = format_all_threads_dump()
                    self._on_freeze(stalled_sec, dump)
                except Exception:
                    pass


def sanitize_log_text(text: str) -> str:
    """Маскує ім'я користувача ОС у шляхах та потенційні токени перед пакуванням у звіт."""
    cleaned = _WIN_USER_PATH_RE.sub(r"\1<USER>", text)
    cleaned = _UNIX_USER_PATH_RE.sub(r"\1<USER>", cleaned)
    cleaned = _SECRET_TOKEN_RE.sub(r"\1<REDACTED>", cleaned)
    return cleaned


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return None


def _write_json_file(path: Path, data: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def mark_session_started(
    data_dir: Path,
    session_log_name: str,
    crash_log_path: Path,
) -> None:
    """Фіксує старт сесії, щоб при наступному запуску виявити нативний краш без чистого виходу."""
    try:
        crash_size = crash_log_path.stat().st_size if crash_log_path.exists() else 0
    except OSError:
        crash_size = 0
    _write_json_file(
        data_dir / LAST_SESSION_FILENAME,
        {
            "session_log": session_log_name,
            "crash_log_size": crash_size,
            "clean_exit": False,
            "started_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


def mark_session_clean_exit(data_dir: Path) -> None:
    """Позначає поточну сесію як коректно завершену."""
    state_path = data_dir / LAST_SESSION_FILENAME
    state = _read_json_file(state_path)
    if state is not None:
        state["clean_exit"] = True
        state["ended_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _write_json_file(state_path, state)


def record_pending_crash(
    data_dir: Path,
    kind: str,
    summary: str,
    session_log_name: str = "",
) -> None:
    """
    Зберігає маркер серйозної помилки/зависання, щоб показати вікно звіту
    при наступному запуску програми.
    """
    _write_json_file(
        data_dir / PENDING_CRASH_FILENAME,
        {
            "kind": str(kind or "crash"),
            "summary": str(summary or "").strip()[:400],
            "session_log": str(session_log_name or ""),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


def clear_pending_freeze(data_dir: Path) -> None:
    """Прибирає маркер зависання, якщо головний потік самостійно відновив роботу."""
    pending_path = data_dir / PENDING_CRASH_FILENAME
    data = _read_json_file(pending_path)
    if data is not None and data.get("kind") == "freeze":
        try:
            pending_path.unlink(missing_ok=True)
        except OSError:
            pass


def consume_pending_crash_report(
    data_dir: Path,
    crash_log_path: Path,
    logs_dir: Path,
) -> dict[str, Any] | None:
    """
    Перевіряє, чи завершився попередній запуск серйозним крашем або зависанням.
    Повертає словник з деталями інциденту й одразу очищує маркер, щоб діалог не повторювався.
    """
    pending_path = data_dir / PENDING_CRASH_FILENAME
    pending = _read_json_file(pending_path)
    if pending is not None:
        try:
            pending_path.unlink(missing_ok=True)
        except OSError:
            pass
        return pending

    last_state = _read_json_file(data_dir / LAST_SESSION_FILENAME)
    if last_state is not None and not bool(last_state.get("clean_exit", True)):
        prev_crash_size = int(last_state.get("crash_log_size", 0) or 0)
        try:
            cur_crash_size = crash_log_path.stat().st_size if crash_log_path.exists() else 0
        except OSError:
            cur_crash_size = 0
        if cur_crash_size > prev_crash_size:
            return {
                "kind": "native_crash",
                "summary": "Abnormal termination detected (crash.log updated)",
                "session_log": str(last_state.get("session_log", "") or ""),
                "timestamp": str(last_state.get("started_at", "") or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            }

    return None


def _prune_old_crash_zips(reports_dir: Path, keep_count: int = MAX_CRASH_REPORT_ZIPS) -> None:
    if not reports_dir.exists():
        return
    zips = sorted(
        (p for p in reports_dir.glob("SilentVoice_crash_*.zip") if p.is_file()),
        key=lambda p: (p.stat().st_mtime if p.exists() else 0.0, p.name),
        reverse=True,
    )
    for stale in zips[keep_count:]:
        try:
            stale.unlink()
        except OSError:
            pass


def _read_tail_text(path: Path, max_bytes: int = 512 * 1024) -> str:
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = b"... [truncated] ...\n" + data[-max_bytes:]
    return data.decode("utf-8", errors="replace")


def build_crash_report_zip(
    data_dir: Path,
    crash_log_path: Path,
    logs_dir: Path,
    report_info: dict[str, Any],
    app_version: str = "",
) -> Path:
    """
    Пакує лише діагностичні логи (без config.json і секретів) у готовий ZIP-архів
    у папці ``data_dir / "crash_reports"``.
    """
    reports_dir = data_dir / "crash_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    _prune_old_crash_zips(reports_dir, keep_count=max(0, MAX_CRASH_REPORT_ZIPS - 1))

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    zip_path = reports_dir / f"SilentVoice_crash_{stamp}.zip"

    sys_lines = [
        f"App: SilentVoice TTS {app_version or 'unknown'}",
        f"OS: {platform.system()} {platform.release()} ({platform.version()})",
        f"Python: {sys.version.splitlines()[0]}",
        f"Incident kind: {report_info.get('kind', 'crash')}",
        f"Timestamp: {report_info.get('timestamp', '')}",
        f"Summary: {sanitize_log_text(str(report_info.get('summary', '') or ''))}",
    ]

    session_log_name = str(report_info.get("session_log", "") or "").strip()
    session_log_path: Path | None = None
    if session_log_name:
        candidate = logs_dir / Path(session_log_name).name
        if candidate.is_file():
            session_log_path = candidate

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("system_info.txt", "\n".join(sys_lines) + "\n")
        if crash_log_path.is_file() and crash_log_path.stat().st_size > 0:
            zf.writestr("crash.log", sanitize_log_text(_read_tail_text(crash_log_path)))
        if session_log_path is not None and session_log_path.stat().st_size > 0:
            zf.writestr(session_log_path.name, sanitize_log_text(_read_tail_text(session_log_path)))

    return zip_path


def build_github_issue_url(
    repo_url: str,
    report_info: dict[str, Any],
    app_version: str = "",
    zip_name: str = "",
) -> str:
    """Формує посилання на створення GitHub Issue із заповненим заголовком та шаблоном."""
    base = str(repo_url or "").rstrip("/")
    kind = str(report_info.get("kind", "crash") or "crash")
    tag = "Freeze" if kind == "freeze" else "Crash"
    summary = sanitize_log_text(str(report_info.get("summary", "") or "")).splitlines()
    first_line = (summary[0] if summary else "Unexpected error")[:90]
    ver_label = f"v{app_version.lstrip('v')}" if app_version else ""
    title = f"[{tag}] {ver_label}: {first_line}".strip()

    body_lines = [
        f"**Version:** {ver_label or 'unknown'}",
        f"**OS:** {platform.system()} {platform.release()}",
        f"**Incident:** `{kind}` ({report_info.get('timestamp', '')})",
        "",
        "### Summary",
        "```",
        sanitize_log_text(str(report_info.get("summary", "") or "No details"))[:500],
        "```",
        "",
        "### What were you doing when this happened?",
        "<!-- Briefly describe your last action -->",
        "",
        "### Diagnostic Archive",
        f"<!-- Drag and drop `{zip_name or 'SilentVoice_crash_*.zip'}` here -->",
    ]
    query = urlencode({"title": title, "body": "\n".join(body_lines)})
    return f"{base}/issues/new?{query}"


