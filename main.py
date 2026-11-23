"""
main.py — Точка входу додатку SilentVoice TTS (PySide6).
Програма для швидкого голосового спілкування без мікрофона на ПК, у програмах, Discord та в іграх.
Усі помилки та стек викликів автоматично виводяться в консоль.
"""

from __future__ import annotations

import ctypes
import faulthandler
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import warnings


def _install_windows_no_window_patch() -> None:
    """
    Гарантує, що жоден виклик subprocess.Popen / subprocess.run (включно зі сторонніми
    бібліотеками torch, onnxruntime, transformers, ffmpeg, ollama тощо) не відкриватиме
    миготливі консольні вікна cmd.exe / conhost.exe у Windows.
    """
    if os.name != "nt" or getattr(subprocess.Popen, "_sv_no_window_patched", False):
        return
    orig_init = subprocess.Popen.__init__
    create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    create_new_console = getattr(subprocess, "CREATE_NEW_CONSOLE", 0x00000010)
    detached_process = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
    new_proc_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)

    def _patched_init(self, *args, **kwargs):
        flags = int(kwargs.get("creationflags", 0) or 0)
        if not (flags & create_new_console):
            flags |= create_no_window
            # У Win32 CreateProcessW прапор DETACHED_PROCESS ігнорує CREATE_NO_WINDOW,
            # через що дочірні процеси можуть створювати власні вікна консолі.
            if flags & detached_process:
                flags &= ~detached_process
                flags |= new_proc_group
            kwargs["creationflags"] = flags
        orig_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = _patched_init  # type: ignore[method-assign]
    subprocess.Popen._sv_no_window_patched = True  # type: ignore[attr-defined]


_install_windows_no_window_patch()

# Додаємо кореневу папку проєкту в sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from PySide6.QtCore import QEvent, QMessageLogContext, QObject, QTimer, QtMsgType, Signal, qInstallMessageHandler
from PySide6.QtWidgets import QAbstractScrollArea, QApplication, QComboBox, QLineEdit

from core.app_logger import (
    MAX_LOG_FILES,
    UIWatchdog,
    clear_pending_freeze,
    consume_pending_crash_report,
    format_log_timestamp,
    mark_session_clean_exit,
    mark_session_started,
    open_session_log_streams,
    record_pending_crash,
    rotate_crash_log_if_needed,
)
from core.config import DATA_DIR, config
from core.i18n import (
    _find_locale_path,
    detect_system_language,
    has_internet_connection,
    set_ui_language,
    tr,
)
from core.updater import APP_VERSION, DEFAULT_GITHUB_REPO
from core.win_api import is_running_as_admin, restart_as_admin
from ui.dialogs import CrashReportDialog, ErrorPopupDialog, FirstRunLanguageDialog
from ui.main_window import MainWindow, create_app_icon

CRASH_LOG_PATH = DATA_DIR / "crash.log"
LOGS_DIR = DATA_DIR / "logs"
APP_LOG_PATH = LOGS_DIR / "app.log"
INSTANCE_PORT_PATH = DATA_DIR / "instance.port"
_CRASH_LOG_FILE = None
_APP_LOG_FILE = None
_ORIG_STDOUT = sys.stdout
_ORIG_STDERR = sys.stderr
_INSTANCE_MUTEX = None
_QUIT_EVENT = None
_QUITTING_NORMALLY = False
_SINGLE_INSTANCE_SOCK = None


def _ensure_stdio_streams() -> None:
    """Гарантує безпечні sys.stdout/sys.stderr на ранньому старті (включно з windowed-збіркою)."""
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    if sys.stdout is None or sys.stderr is None:
        try:
            devnull = open(os.devnull, "w", encoding="utf-8")
            if sys.stdout is None:
                sys.stdout = devnull
            if sys.stderr is None:
                sys.stderr = devnull
        except Exception:
            pass


def _init_session_logging() -> None:
    """
    Відкриває окремий файл логу для поточного запуску в ``DATA_DIR / "logs"``,
    додає мітку часу до кожного рядка та зберігає не більше ``MAX_LOG_FILES`` (5) останніх логів.
    """
    global APP_LOG_PATH, _APP_LOG_FILE
    try:
        log_path, raw_file, stdout_stream, stderr_stream = open_session_log_streams(
            LOGS_DIR,
            orig_stdout=_ORIG_STDOUT,
            orig_stderr=_ORIG_STDERR,
            max_files=MAX_LOG_FILES,
        )
        APP_LOG_PATH = log_path
        _APP_LOG_FILE = raw_file
        sys.stdout = stdout_stream
        sys.stderr = stderr_stream
    except Exception:
        _ensure_stdio_streams()


def _write_crash_log(
    header: str,
    exc_type=None,
    exc_value=None,
    exc_tb=None,
    body: str | None = None,
) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        stamp = format_log_timestamp()
        with open(CRASH_LOG_PATH, "a", encoding="utf-8", errors="replace") as fh:
            fh.write("\n" + "=" * 68 + "\n" + f"{stamp} {header}\n")
            if body:
                fh.write(body.rstrip() + "\n")
            if exc_type is not None:
                traceback.print_exception(exc_type, exc_value, exc_tb, file=fh)
            fh.flush()
    except Exception:
        pass


def _console_excepthook(exc_type, exc_value, exc_tb) -> None:
    """Виводить будь-яку необроблену помилку Python у консоль, crash.log та показує вікно помилки."""
    _write_crash_log("[ERROR] Необроблена помилка в головному потоці:", exc_type, exc_value, exc_tb)
    err_name = getattr(exc_type, "__name__", "Error")
    record_pending_crash(
        DATA_DIR,
        kind="crash",
        summary=f"{err_name}: {exc_value}",
        session_log_name=APP_LOG_PATH.name,
    )
    if sys.stderr is not None:
        print("\n" + "=" * 68, file=sys.stderr)
        print("[ERROR] Необроблена помилка в головному потоці:", file=sys.stderr)
        traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
        print("=" * 68 + "\n", file=sys.stderr)
        sys.stderr.flush()
    try:
        if QApplication.instance() is not None:
            err_msg = str(exc_value).strip() or "".join(traceback.format_exception(exc_type, exc_value, exc_tb)).strip()
            ErrorPopupDialog(err_msg, tr("err.system_title", "Системна помилка")).exec()
    except Exception:
        pass


def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
    """Виводить будь-яку помилку у фонових потоках у консоль та crash.log."""
    tname = args.thread.name if args.thread else "unknown"
    _write_crash_log(f"[THREAD ERROR] Помилка в потоці {tname}:", args.exc_type, args.exc_value, args.exc_traceback)
    if sys.stderr is not None:
        print("\n" + "=" * 68, file=sys.stderr)
        print(f"[THREAD ERROR] Помилка в потоці {tname}:", file=sys.stderr)
        traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=sys.stderr)
        print("=" * 68 + "\n", file=sys.stderr)
        sys.stderr.flush()


def _unraisable_excepthook(unraisable) -> None:
    """Ловить помилки в __del__, weakref та C-колбеках (PortAudio / PySide6 / ctypes), які не доходять до sys.excepthook."""
    err_msg = getattr(unraisable, "err_msg", None) or "Unraisable exception"
    obj_repr = ""
    try:
        if getattr(unraisable, "object", None) is not None:
            obj_repr = f" ({unraisable.object!r})"
    except Exception:
        pass
    header = f"[UNRAISABLE ERROR] {err_msg}{obj_repr}:"
    _write_crash_log(header, unraisable.exc_type, unraisable.exc_value, unraisable.exc_traceback)
    if sys.stderr is not None:
        print(f"{header} {unraisable.exc_value}", file=sys.stderr, flush=True)


def _qt_message_handler(mode: QtMsgType, context: QMessageLogContext, message: str) -> None:
    """Перенаправляє попередження та помилки Qt у лог, а критичні/фатальні — також у crash.log."""
    if "QFont::setPointSize: Point size <= 0" in message:
        return
    if mode in (QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
        level = "FATAL" if mode == QtMsgType.QtFatalMsg else "CRITICAL"
        _write_crash_log(f"[Qt {level}] {message}")
        if mode == QtMsgType.QtFatalMsg:
            record_pending_crash(
                DATA_DIR,
                kind="crash",
                summary=f"QtFatal: {message}",
                session_log_name=APP_LOG_PATH.name,
            )
    if mode in (QtMsgType.QtWarningMsg, QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
        print(f"[Qt] {message}", file=sys.stderr, flush=True)


class _IpcShowBridge(QObject):
    show_requested = Signal()
    quit_requested = Signal()


def _activate_existing_instance() -> bool:
    """Надсилає запущеному екземпляру SilentVoice сигнал показати/активувати вікно."""
    try:
        ctypes.windll.user32.AllowSetForegroundWindow(-1)
    except Exception:
        pass

    for _ in range(8):
        try:
            if INSTANCE_PORT_PATH.exists():
                port = int(INSTANCE_PORT_PATH.read_text(encoding="utf-8").strip())
                if 1 <= port <= 65535:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.3) as sock:
                        sock.sendall(b"SHOW\n")
                        return True
        except Exception:
            pass
        time.sleep(0.06)

    # Резерв через Win32 API, якщо локальний сокет ще не встиг відкритися
    try:
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        found_hwnd = 0

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def _enum_cb(hwnd, _lp):
            nonlocal found_hwnd
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                if "SilentVoice" in buf.value and "Interactive Input" not in buf.value:
                    found_hwnd = hwnd
                    return False
            return True

        user32.EnumWindows(_enum_cb, 0)
        if found_hwnd:
            user32.ShowWindow(found_hwnd, 9)  # SW_RESTORE
            user32.SetForegroundWindow(found_hwnd)
            return True
    except Exception:
        pass
    return False


def main() -> int:
    global _CRASH_LOG_FILE, _INSTANCE_MUTEX, _SINGLE_INSTANCE_SOCK
    warnings.filterwarnings("ignore", category=SyntaxWarning, module=r".*torch_package.*")
    _ensure_stdio_streams()
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        rotate_crash_log_if_needed(CRASH_LOG_PATH)
        _CRASH_LOG_FILE = open(CRASH_LOG_PATH, "a", encoding="utf-8", errors="replace")
        faulthandler.enable(file=_CRASH_LOG_FILE, all_threads=True)
    except Exception:
        pass
    sys.excepthook = _console_excepthook
    threading.excepthook = _thread_excepthook
    sys.unraisablehook = _unraisable_excepthook
    qInstallMessageHandler(_qt_message_handler)

    if "--verify-imports" in sys.argv:
        ok = True
        lines = []
        engines_dir = DATA_DIR / "engines"

        def _verify_holos():
            from core.tts_manager import tts_manager

            HolosTTS = __import__("holos_onnx", fromlist=["HolosTTS"]).HolosTTS
            fast_stress_fn = tts_manager._build_fast_uk_stress()
            ipa_fn = __import__("ipa_uk", fromlist=["ipa"]).ipa
            stressed_fast = fast_stress_fn("привіт")
            phonemes = ipa_fn(stressed_fast)
            full_stress = __import__("ukrainian_word_stress", fromlist=["Stressifier"]).Stressifier()("привіт")
            holos_dir = engines_dir / "holos"
            bundle_dir = holos_dir / "onnx"
            required = ("encoder.onnx", "prosody.onnx", "decoder.onnx", "holos_onnx.json", "voices.npz")
            if all((bundle_dir / name).exists() for name in required):
                t0 = time.perf_counter()
                model = HolosTTS(str(bundle_dir), providers=["CPUExecutionProvider"])
                t_load = time.perf_counter() - t0
                t1 = time.perf_counter()
                audio = model.generate(phonemes, voice="Гаська Шиян", speed=1.0)
                t_gen = time.perf_counter() - t1
                n_samples = len(audio) if hasattr(audio, "__len__") else 0
                return (stressed_fast, full_stress, phonemes, f"load={t_load:.2f}s", f"gen={t_gen:.2f}s", f"samples={n_samples}")
            return (stressed_fast, full_stress, phonemes)

        def _verify_teratts():
            from core.tts_manager import tts_manager

            t_start = time.perf_counter()
            translit = __import__("transliterate", fromlist=["translit"]).translit
            num2words = __import__("num2words", fromlist=["num2words"]).num2words
            tr_res = translit("Hello world", "ru")
            n2w_ru = num2words(123, lang="ru")
            n2w_uk = num2words(123, lang="uk")
            tera_dir = engines_dir / "teratts"
            if (tera_dir / "ready.marker").exists():
                acc, tts = tts_manager._load_teratts_fast(tera_dir)
                t_load = time.perf_counter() - t_start
                t1 = time.perf_counter()
                accented = acc.process_all("Привет мир 123 и абвгдейка")
                audio = tts(accented, lenght_scale=1.1)
                t_gen = time.perf_counter() - t1
                n_samples = len(audio) if hasattr(audio, "__len__") else 0
                return (tr_res, n2w_ru, n2w_uk, accented, f"load={t_load:.2f}s", f"gen={t_gen:.2f}s", f"samples={n_samples}")
            RUAccent = tts_manager._import_ruaccent_fast()
            TTS = __import__("TeraTTS", fromlist=["TTS"]).TTS
            return (tr_res, n2w_ru, n2w_uk, RUAccent, TTS)

        for label, fn in (
            ("teratts_ruaccent", _verify_teratts),
            ("torch_package", lambda: (__import__("unittest"), __import__("torch.testing"), __import__("torch.package", fromlist=["PackageImporter"]))),
            ("transformers_auto_tokenizer", lambda: __import__("transformers", fromlist=["AutoTokenizer"]).AutoTokenizer),
            ("transformers_albert", lambda: __import__("transformers", fromlist=["AlbertConfig", "AlbertModel"]).AlbertModel),
            ("holos", _verify_holos),
            ("piper", lambda: __import__("piper.voice", fromlist=["PiperVoice"]).PiperVoice),
        ):
            try:
                res = fn()
                lines.append(f"OK {label}: {res!r}")
            except Exception as exc:
                ok = False
                lines.append(f"FAIL {label}: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        report = "\n".join(lines) + "\n"
        try:
            (DATA_DIR / "verify_imports.log").write_text(report, encoding="utf-8")
        except Exception:
            pass
        print(report, flush=True)
        return 0 if ok else 1

    # При першому запуску визначаємо локаль Windows; далі мова інтерфейсу зберігається явно і не перемикається автоматично
    _saved_ui_lang = str(config.get("ui_language", "") or "").strip().lower()
    if _saved_ui_lang:
        set_ui_language(_saved_ui_lang)
    else:
        _sys_code, _, _ = detect_system_language()
        _loc_file = _find_locale_path(_sys_code)
        _init_ui = _sys_code if (_loc_file is not None and _loc_file.exists()) else "en"
        set_ui_language(_init_ui)
        if bool(config.get("first_run_lang_checked", False)):
            config.set("ui_language", _init_ui)

    # Опційне підвищення прав (UAC) для коректної роботи глобальних хоткеїв у Windows.
    # Через механізм Windows UIPI (User Interface Privilege Isolation) процес без прав
    # адміністратора не може перехоплювати глобальний ввід (WH_KEYBOARD_LL) поверх ігор
    # або програм (наприклад, Discord / ігри з античітом), запущених з підвищеними правами.
    # Також права адміністратора потрібні для одноразового встановлення драйвера VB-Audio
    # Virtual Cable. Користувач може будь-коли вимкнути цю поведінку в налаштуваннях
    # ("run_as_admin": false) або запустити з аргументом `--no-admin-relaunch`.
    if (
        bool(config.get("run_as_admin", True))
        and not is_running_as_admin()
        and "--no-admin-relaunch" not in sys.argv
    ):
        if restart_as_admin():
            return 0

    # Захист від повторного запуску (Single-Instance Mutex + IPC для активації вже відкритого вікна)
    _mutex_already_exists = False
    try:
        _INSTANCE_MUTEX = ctypes.windll.kernel32.CreateMutexW(None, False, "SilentVoiceTTS_Mutex")
        _mutex_already_exists = ctypes.windll.kernel32.GetLastError() == 183  # ERROR_ALREADY_EXISTS
    except Exception:
        _mutex_already_exists = False

    # Встановлюємо AppUserModelID для коректної іконки на панелі завдань Windows
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("SilentVoice.TTS.DesktopApp.1")
    except Exception:
        pass

    app = QApplication(sys.argv)

    class _NoWheelComboFilter(QObject):
        """Глобально вимикає зміну значень QComboBox коліщатком миші у всіх вікнах додатку."""

        def eventFilter(self, obj: QObject, event: QEvent) -> bool:
            if event.type() == QEvent.Type.Wheel:
                combo = obj if isinstance(obj, QComboBox) else (obj.parent() if isinstance(obj, QLineEdit) and isinstance(obj.parent(), QComboBox) else None)
                if combo is not None:
                    event.ignore()
                    parent = combo.parentWidget()
                    while parent is not None:
                        if isinstance(parent, QAbstractScrollArea):
                            QApplication.sendEvent(parent.viewport(), event)
                            break
                        parent = parent.parentWidget()
                    return True
            return False

    _no_wheel_filter = _NoWheelComboFilter(app)
    app.installEventFilter(_no_wheel_filter)

    if _mutex_already_exists:
        print("[SilentVoice] Програма вже запущена — активуємо існуюче вікно.", flush=True)
        _activate_existing_instance()
        return 0

    pending_crash_report = consume_pending_crash_report(DATA_DIR, CRASH_LOG_PATH, LOGS_DIR)
    _init_session_logging()
    mark_session_started(DATA_DIR, APP_LOG_PATH.name, CRASH_LOG_PATH)
    print(f"[SilentVoice] Запуск програми... (лог: {APP_LOG_PATH.name})", flush=True)

    def _release_instance_lock() -> None:
        global _INSTANCE_MUTEX, _QUIT_EVENT, _QUITTING_NORMALLY, _SINGLE_INSTANCE_SOCK
        _QUITTING_NORMALLY = True
        try:
            if _QUIT_EVENT:
                ctypes.windll.kernel32.SetEvent(_QUIT_EVENT)
                ctypes.windll.kernel32.CloseHandle(_QUIT_EVENT)
                _QUIT_EVENT = None
        except Exception:
            pass
        try:
            if _SINGLE_INSTANCE_SOCK is not None:
                _SINGLE_INSTANCE_SOCK.close()
                _SINGLE_INSTANCE_SOCK = None
        except Exception:
            pass
        try:
            INSTANCE_PORT_PATH.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            if _INSTANCE_MUTEX:
                ctypes.windll.kernel32.CloseHandle(_INSTANCE_MUTEX)
                _INSTANCE_MUTEX = None
        except Exception:
            pass

    app.setProperty("_release_instance_lock", _release_instance_lock)

    app.setApplicationName("SilentVoice TTS")
    app.setOrganizationName("SilentVoice")
    app.setWindowIcon(create_app_icon())
    # Не завершувати роботу при закритті вікна, якщо програма згорнута в трей
    app.setQuitOnLastWindowClosed(False)

    orig_cfg_backup: str | None = None
    sim_arg = next((a for a in sys.argv[1:] if a.startswith("--simulate-lang=")), "")
    if sim_arg:
        from PySide6.QtCore import QLocale
        from core.config import CONFIG_FILE
        sim_code = sim_arg.split("=", 1)[1].strip().lower() or "it"
        ql = QLocale(sim_code)
        nat = (ql.nativeLanguageName() or "").strip()
        nat = (nat[:1].upper() + nat[1:]) if nat and nat[:1].islower() else (nat or sim_code.upper())
        eng = (QLocale.languageToString(ql.language()) or "").strip() or sim_code.upper()
        orig_cfg_backup = CONFIG_FILE.read_text(encoding="utf-8") if CONFIG_FILE.exists() else None
        dlg = FirstRunLanguageDialog(sim_code, nat, eng)
        dlg.raise_()
        dlg.activateWindow()
        dlg.exec()
    elif not bool(config.get("first_run_lang_checked", False)):
        sys_code, native_name, eng_name = detect_system_language()
        loc_file = _find_locale_path(sys_code)
        if loc_file is not None and loc_file.exists():
            from core.config import sync_default_quick_phrases_for_lang
            set_ui_language(sys_code)
            config.update({
                "ui_language": sys_code,
                "app_language": sys_code,
                "language_confirmed": True,
                "first_run_lang_checked": True,
            })
            sync_default_quick_phrases_for_lang(sys_code)
        elif has_internet_connection():
            dlg = FirstRunLanguageDialog(sys_code, native_name, eng_name)
            dlg.exec()
        else:
            from core.config import sync_default_quick_phrases_for_lang
            set_ui_language("en")
            config.update({
                "ui_language": "en",
                "app_language": "en",
                "language_confirmed": True,
                "first_run_lang_checked": False,
            })
            sync_default_quick_phrases_for_lang("en")

    window = MainWindow()
    app.aboutToQuit.connect(window.force_quit_app)
    ipc_bridge = _IpcShowBridge(app)
    ipc_bridge.show_requested.connect(window.restore_from_tray)
    ipc_bridge.quit_requested.connect(window.force_quit_app)

    try:
        if os.name == "nt":
            _QUIT_EVENT = ctypes.windll.kernel32.CreateEventW(None, True, False, "SilentVoiceTTS_QuitEvent")
            if _QUIT_EVENT:
                def _quit_event_waiter() -> None:
                    ev = _QUIT_EVENT
                    if not ev:
                        return
                    try:
                        ctypes.windll.kernel32.WaitForSingleObject(ev, 0xFFFFFFFF)
                        if not _QUITTING_NORMALLY:
                            print("[SilentVoice] Отримано зовнішній запит на завершення (інсталятор).", flush=True)
                            ipc_bridge.quit_requested.emit()
                    except Exception:
                        pass

                threading.Thread(target=_quit_event_waiter, daemon=True, name="InstallerQuitWaiter").start()
    except Exception:
        pass

    try:
        _SINGLE_INSTANCE_SOCK = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _SINGLE_INSTANCE_SOCK.bind(("127.0.0.1", 0))
        _SINGLE_INSTANCE_SOCK.listen(5)
        ipc_port = int(_SINGLE_INSTANCE_SOCK.getsockname()[1])
        INSTANCE_PORT_PATH.write_text(str(ipc_port), encoding="utf-8")

        def _ipc_accept_loop() -> None:
            while True:
                srv = _SINGLE_INSTANCE_SOCK
                if srv is None:
                    break
                try:
                    conn, _ = srv.accept()
                except OSError:
                    break
                try:
                    conn.settimeout(0.3)
                    data = conn.recv(64)
                    conn.close()
                    if b"QUIT" in data:
                        ipc_bridge.quit_requested.emit()
                    elif b"SHOW" in data:
                        ipc_bridge.show_requested.emit()
                except Exception:
                    try:
                        conn.close()
                    except Exception:
                        pass

        threading.Thread(target=_ipc_accept_loop, daemon=True, name="SingleInstanceIPC").start()
    except Exception:
        pass

    start_minimized = ("--minimized" in sys.argv) or bool(config.get("start_minimized", False))
    if not start_minimized:
        window.show()
        window.raise_()
        window.activateWindow()
        app.processEvents()
        window.repaint()

    def _on_ui_freeze(stalled_sec: float, thread_dump: str) -> None:
        header = f"[UI FREEZE] Головний потік не відповідає вже {stalled_sec:.1f} с (лог сесії: {APP_LOG_PATH.name}):"
        _write_crash_log(header, body=thread_dump)
        record_pending_crash(
            DATA_DIR,
            kind="freeze",
            summary=f"UI freeze ({stalled_sec:.1f}s)",
            session_log_name=APP_LOG_PATH.name,
        )
        print(f"[Watchdog] Виявлено зависання інтерфейсу ({stalled_sec:.1f} с) — стек усіх потоків записано в crash.log.", flush=True)

    def _on_ui_recover(stalled_sec: float) -> None:
        clear_pending_freeze(DATA_DIR)
        _write_crash_log(f"[UI RECOVERED] Головний потік відновив роботу після {stalled_sec:.1f} с.")
        print(f"[Watchdog] Інтерфейс відновив роботу після {stalled_sec:.1f} с.", flush=True)

    ui_watchdog = UIWatchdog(
        on_freeze=_on_ui_freeze,
        on_recover=_on_ui_recover,
        timeout_sec=12.0,
        check_interval_sec=2.0,
        repeat_dump_sec=30.0,
    )
    watchdog_timer = QTimer(app)
    watchdog_timer.setInterval(2000)
    watchdog_timer.timeout.connect(ui_watchdog.heartbeat)
    watchdog_timer.start()
    ui_watchdog.start()

    if pending_crash_report is not None:
        def _show_pending_crash_dialog() -> None:
            try:
                dlg = CrashReportDialog(
                    report_info=pending_crash_report,
                    data_dir=DATA_DIR,
                    crash_log_path=CRASH_LOG_PATH,
                    logs_dir=LOGS_DIR,
                    app_version=APP_VERSION,
                    repo_url=DEFAULT_GITHUB_REPO,
                    parent=window if window.isVisible() else None,
                )
                dlg.exec()
            except Exception:
                pass

        QTimer.singleShot(450, _show_pending_crash_dialog)

    try:
        return app.exec()
    finally:
        try:
            watchdog_timer.stop()
        except Exception:
            pass
        try:
            ui_watchdog.stop()
        except Exception:
            pass
        mark_session_clean_exit(DATA_DIR)
        print("[SilentVoice] Завершення роботи...", flush=True)
        try:
            window.force_quit_app()
        except Exception:
            pass
        try:
            from core.tts_manager import tts_manager
            tts_manager.shutdown()
        except Exception:
            pass
        try:
            from core.llama_tts import shutdown_llama_tts
            shutdown_llama_tts()
        except Exception:
            pass
        try:
            from core.local_llm import stop_managed_llama_server
            stop_managed_llama_server()
        except Exception:
            pass
        _release_instance_lock()
        if orig_cfg_backup is not None:
            from core.config import CONFIG_FILE
            CONFIG_FILE.write_text(orig_cfg_backup, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
