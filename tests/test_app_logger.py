import datetime
import io
import os
import re
import tempfile
import unittest
from pathlib import Path

from core.app_logger import (
    MAX_LOG_FILES,
    TimestampedLogStream,
    create_session_log_path,
    open_session_log_streams,
    prune_old_logs,
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


if __name__ == "__main__":
    unittest.main()
