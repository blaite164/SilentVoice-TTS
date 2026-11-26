import datetime
import io
import os
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path

from core.app_logger import (
    MAX_LOG_FILES,
    TimestampedLogStream,
    UIWatchdog,
    build_crash_report_zip,
    build_github_issue_url,
    clear_pending_freeze,
    consume_pending_crash_report,
    create_session_log_path,
    format_all_threads_dump,
    mark_session_clean_exit,
    mark_session_started,
    open_session_log_streams,
    prune_old_logs,
    record_pending_crash,
    rotate_crash_log_if_needed,
    sanitize_log_text,
)


class _StrictAsciiStream:
    encoding = "ascii"

    def __init__(self) -> None:
        self.chunks: list[str] = []

    def write(self, text: str) -> int:
        text.encode("ascii")
        self.chunks.append(text)
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


class AppLoggerTests(unittest.TestCase):
    def test_creates_unique_per_launch_log_and_keeps_max_five(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp) / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            # Нецільовий файл не повинен видалятися
            unrelated = logs_dir / "notes.txt"
            unrelated.write_text("keep", encoding="utf-8")

            base_dt = datetime.datetime(2026, 11, 23, 20, 0, 0)
            created: list[Path] = []
            for idx in range(7):
                dt = base_dt + datetime.timedelta(seconds=idx)
                p = create_session_log_path(logs_dir, now=dt, max_files=MAX_LOG_FILES)
                p.write_text(f"run {idx}\n", encoding="utf-8")
                ts = dt.timestamp()
                os.utime(p, (ts, ts))
                created.append(p)

            remaining = sorted(p.name for p in logs_dir.glob("app_*.log"))
            self.assertEqual(len(remaining), MAX_LOG_FILES)
            self.assertEqual(remaining, [p.name for p in created[-MAX_LOG_FILES:]])
            self.assertTrue(unrelated.exists())

    def test_same_second_launch_does_not_overwrite_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp) / "logs"
            dt = datetime.datetime(2026, 11, 23, 20, 15, 30)
            first = create_session_log_path(logs_dir, now=dt, max_files=5)
            first.write_text("first\n", encoding="utf-8")

            second = create_session_log_path(logs_dir, now=dt, max_files=5)
            second.write_text("second\n", encoding="utf-8")

            self.assertNotEqual(first, second)
            self.assertEqual(first.read_text(encoding="utf-8"), "first\n")
            self.assertEqual(second.read_text(encoding="utf-8"), "second\n")

    def test_timestamped_stream_formats_lines_and_handles_partial_writes(self) -> None:
        buf = io.StringIO()
        stream = TimestampedLogStream(buf)
        stream.write("Hello ")
        stream.write("world\nSecond line\n\nThird")
        stream.write("\n")
        stream.flush()

        lines = buf.getvalue().splitlines()
        self.assertEqual(len(lines), 4)
        ts_pattern = r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] "
        self.assertRegex(lines[0], ts_pattern + r"Hello world$")
        self.assertRegex(lines[1], ts_pattern + r"Second line$")
        self.assertEqual(lines[2], "")
        self.assertRegex(lines[3], ts_pattern + r"Third$")

    def test_console_encoding_error_does_not_break_file_logging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp) / "logs"
            ascii_console = _StrictAsciiStream()
            log_path, raw_file, out_stream, err_stream = open_session_log_streams(
                logs_dir,
                orig_stdout=ascii_console,  # type: ignore[arg-type]
                orig_stderr=None,
                max_files=5,
            )
            try:
                out_stream.write("[SilentVoice] Запуск програми...\n")
                err_stream.write("[Qt] Тестове попередження\n")
                out_stream.flush()
                err_stream.flush()
            finally:
                raw_file.close()

            content = log_path.read_text(encoding="utf-8")
            self.assertIn("[SilentVoice] Запуск програми...", content)
            self.assertIn("[Qt] Тестове попередження", content)
            self.assertEqual(len(ascii_console.chunks), 1)
            self.assertRegex(
                ascii_console.chunks[0],
                r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[SilentVoice\] ",
            )

    def test_prune_old_logs_explicit_keep_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp) / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(4):
                p = logs_dir / f"app_2026-11-23_20-00-0{idx}.log"
                p.write_text("x", encoding="utf-8")
                ts = 1_700_000_000 + idx
                os.utime(p, (ts, ts))
            removed = prune_old_logs(logs_dir, keep_count=2)
            self.assertEqual(len(removed), 2)
            self.assertEqual(len(list(logs_dir.glob("app_*.log"))), 2)

    def test_format_all_threads_dump_includes_main_thread(self) -> None:
        dump = format_all_threads_dump()
        self.assertIn("MainThread", dump)
        self.assertIn("test_format_all_threads_dump_includes_main_thread", dump)

    def test_ui_watchdog_detects_freeze_and_recovery(self) -> None:
        freeze_events: list[tuple[float, str]] = []
        recover_events: list[float] = []
        got_freeze = threading.Event()
        got_recover = threading.Event()

        def _on_freeze(stalled: float, dump: str) -> None:
            freeze_events.append((stalled, dump))
            got_freeze.set()

        def _on_recover(stalled: float) -> None:
            recover_events.append(stalled)
            got_recover.set()

        wd = UIWatchdog(
            on_freeze=_on_freeze,
            on_recover=_on_recover,
            timeout_sec=0.08,
            check_interval_sec=0.02,
            repeat_dump_sec=5.0,
        )
        wd.start()
        try:
            self.assertTrue(got_freeze.wait(timeout=1.5), "Watchdog did not detect freeze")
            self.assertEqual(len(freeze_events), 1)
            self.assertGreaterEqual(freeze_events[0][0], 0.08)
            self.assertIn("MainThread", freeze_events[0][1])

            wd.heartbeat()
            self.assertTrue(got_recover.wait(timeout=0.5), "Watchdog did not report recovery")
            self.assertEqual(len(recover_events), 1)
        finally:
            wd.stop()

    def test_rotate_crash_log_if_needed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            crash_path = Path(tmp) / "crash.log"
            crash_path.write_text("x" * 200, encoding="utf-8")
            rotated = rotate_crash_log_if_needed(crash_path, max_bytes=100)
            self.assertTrue(rotated)
            self.assertFalse(crash_path.exists())
            backup = Path(tmp) / "crash.log.old"
            self.assertTrue(backup.exists())
            self.assertEqual(len(backup.read_text(encoding="utf-8")), 200)

    def test_sanitize_log_text_masks_usernames_and_tokens(self) -> None:
        raw = r"Error at C:\Users\AliceSmith\AppData\Local\SilentVoice and Bearer sk-1234567890abcdef"
        cleaned = sanitize_log_text(raw)
        self.assertNotIn("AliceSmith", cleaned)
        self.assertIn(r"C:\Users\<USER>", cleaned)
        self.assertNotIn("sk-1234567890abcdef", cleaned)
        self.assertIn("Bearer <REDACTED>", cleaned)

    def test_pending_crash_lifecycle_and_recovered_freeze_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            logs_dir = data_dir / "logs"
            crash_log = data_dir / "crash.log"

            # 1. Тимчасове зависання, яке відновилося — не повинно показувати діалог
            record_pending_crash(data_dir, "freeze", "UI freeze (12.1s)", "app_1.log")
            clear_pending_freeze(data_dir)
            self.assertIsNone(consume_pending_crash_report(data_dir, crash_log, logs_dir))

            # 2. Справжній краш — повертається один раз і одразу очищується
            record_pending_crash(data_dir, "crash", "RuntimeError: boom", "app_2.log")
            first = consume_pending_crash_report(data_dir, crash_log, logs_dir)
            self.assertIsNotNone(first)
            assert first is not None
            self.assertEqual(first["kind"], "crash")
            self.assertEqual(first["session_log"], "app_2.log")
            self.assertIsNone(consume_pending_crash_report(data_dir, crash_log, logs_dir))

            # 3. Нативний краш (crash.log виріс, але clean_exit=False)
            crash_log.write_text("old\n", encoding="utf-8")
            mark_session_started(data_dir, "app_3.log", crash_log)
            crash_log.write_text("old\nFatal Python error: Segmentation fault\n", encoding="utf-8")
            native = consume_pending_crash_report(data_dir, crash_log, logs_dir)
            self.assertIsNotNone(native)
            assert native is not None
            self.assertEqual(native["kind"], "native_crash")
            self.assertIn("Segmentation fault", native["summary"])

            # 3b. Нефатальний Windows COM first-chance виняток 0x8001010d — НЕ повинен викликати діалог крашу
            mark_session_started(data_dir, "app_3b.log", crash_log)
            with open(crash_log, "a", encoding="utf-8") as fh:
                fh.write("Windows fatal exception: code 0x8001010d\n\nCurrent thread 0x1:\n  File main.py\n")
            self.assertIsNone(consume_pending_crash_report(data_dir, crash_log, logs_dir))

            # 3c. Справжній NTSTATUS краш 0xc0000409 / access violation у потоці з Python-фреймом — повинен показати діалог
            mark_session_started(data_dir, "app_3c.log", crash_log)
            with open(crash_log, "a", encoding="utf-8") as fh:
                fh.write("Windows fatal exception: access violation\n\nCurrent thread 0x1 (most recent call first):\n  File main.py, line 10\n")
            av = consume_pending_crash_report(data_dir, crash_log, logs_dir)
            self.assertIsNotNone(av)
            assert av is not None
            self.assertIn("access violation", av["summary"])

            # 3d. Сторонній нативний потік Windows без Python-фрейму (<no Python frame>, напр. драйвер/оверлей VEH) — НЕ повинен лякати діалогом
            mark_session_started(data_dir, "app_3d.log", crash_log)
            with open(crash_log, "a", encoding="utf-8") as fh:
                fh.write(
                    "Windows fatal exception: access violation\n\n"
                    "Current thread 0x6830 (most recent call first):\n"
                    "  <no Python frame>\n\n"
                    "Thread 0x2030 (most recent call first):\n"
                    '  File "main.py", line 648 in main\n'
                )
            self.assertIsNone(consume_pending_crash_report(data_dir, crash_log, logs_dir))

            # 4. Коректне завершення — не показує діалог
            mark_session_started(data_dir, "app_4.log", crash_log)
            mark_session_clean_exit(data_dir)
            self.assertIsNone(consume_pending_crash_report(data_dir, crash_log, logs_dir))

    def test_build_crash_report_zip_excludes_config_and_sanitizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            logs_dir = data_dir / "logs"
            logs_dir.mkdir()
            crash_log = data_dir / "crash.log"
            (data_dir / "config.json").write_text('{"api_key": "SECRET"}', encoding="utf-8")
            crash_log.write_text(r"Traceback at C:\Users\Tester\main.py", encoding="utf-8")
            (logs_dir / "app_test.log").write_text(r"[2026-11-23 20:00:00] C:\Users\Tester\ok", encoding="utf-8")

            report_info = {
                "kind": "freeze",
                "summary": "UI freeze (15.0s)",
                "session_log": "app_test.log",
                "timestamp": "2026-11-23 20:00:15",
            }
            zp = build_crash_report_zip(data_dir, crash_log, logs_dir, report_info, app_version="1.0.1")
            self.assertTrue(zp.exists())

            with zipfile.ZipFile(zp, "r") as zf:
                names = set(zf.namelist())
                self.assertIn("system_info.txt", names)
                self.assertIn("crash.log", names)
                self.assertIn("app_test.log", names)
                self.assertNotIn("config.json", names)
                self.assertNotIn("Tester", zf.read("crash.log").decode("utf-8"))
                self.assertNotIn("Tester", zf.read("app_test.log").decode("utf-8"))

            url = build_github_issue_url(
                "https://github.com/blaite164/SilentVoice-TTS",
                report_info,
                app_version="1.0.1",
                zip_name=zp.name,
            )
            self.assertTrue(url.startswith("https://github.com/blaite164/SilentVoice-TTS/issues/new?"))
            self.assertIn("Freeze", url)
            self.assertIn(zp.name, url)

    def test_run_in_background_handles_destroyed_parent_safely(self) -> None:
        from PySide6.QtCore import QCoreApplication, QEvent, QObject
        from PySide6.QtWidgets import QApplication
        from shiboken6 import isValid
        from ui.dialogs import _ACTIVE_BG_BRIDGES, run_in_background

        app = QApplication.instance() or QApplication([])
        assert app is not None

        # 1) Normal delivery when parent stays alive
        parent_ok = QObject()
        received: list[tuple[object, str]] = []
        done_ev = threading.Event()

        def on_ok(val: object, err: str) -> None:
            received.append((val, err))
            done_ev.set()

        run_in_background(parent_ok, lambda: "hello", on_done=on_ok)
        deadline = time.monotonic() + 3.0
        while not done_ev.is_set() and time.monotonic() < deadline:
            QCoreApplication.processEvents()
            time.sleep(0.01)
        self.assertEqual(received, [("hello", "")])

        # 2) Parent destroyed before background task finishes — must NOT crash or deliver
        parent_dead = QObject()
        gate = threading.Event()
        worker_done = threading.Event()
        delivered_after_destroy: list[tuple[object, str]] = []

        def slow_task() -> str:
            gate.wait(timeout=2.0)
            worker_done.set()
            return "too_late"

        run_in_background(
            parent_dead,
            slow_task,
            on_done=lambda v, e: delivered_after_destroy.append((v, e)),
        )
        parent_dead.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        QCoreApplication.processEvents()
        self.assertFalse(isValid(parent_dead))

        gate.set()
        self.assertTrue(worker_done.wait(timeout=2.0))
        time.sleep(0.05)
        QCoreApplication.processEvents()
        self.assertEqual(delivered_after_destroy, [])
        self.assertEqual(len(_ACTIVE_BG_BRIDGES), 0)

    def test_console_excepthook_ignores_keyboard_interrupt(self) -> None:
        from unittest.mock import patch
        import main as app_main

        with (
            patch.object(app_main, "record_pending_crash") as mock_record,
            patch.object(app_main, "mark_session_clean_exit") as mock_clean,
            patch.object(app_main.sys, "__excepthook__") as mock_orig,
        ):
            app_main._console_excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
            mock_record.assert_not_called()
            mock_clean.assert_called_once()
            mock_orig.assert_called_once()


if __name__ == "__main__":
    unittest.main()
