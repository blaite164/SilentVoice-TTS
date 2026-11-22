"""
core/win_api.py — Інтеграція з Windows API (ctypes):
- SetWindowDisplayAffinity (WDA_EXCLUDEFROMCAPTURE) для повної невидимості в OBS / Discord / записі екрана
- Прозорий click-through оверлей без втрати фокусу гри (WS_EX_TOPMOST, WS_EX_TRANSPARENT, WS_EX_NOACTIVATE)
- Низькорівневий хук клавіатури WH_KEYBOARD_LL + ToUnicodeEx (UA / RU / EN) з автовизначенням і перемиканням розкладки
- Запасний режим вводу з автоповерненням фокусу у гру
- Автозапуск Windows та перевірка прав адміністратора
"""

from __future__ import annotations

from collections import deque
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import math
import os
import re
import sys
import threading
import time
import winreg
from typing import Deque, Dict, FrozenSet, List, Optional, Set, Tuple

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QGuiApplication

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
shell32 = ctypes.windll.shell32

# WinAPI константи
WDA_NONE = 0x00000000
WDA_MONITOR = 0x00000001
WDA_EXCLUDEFROMCAPTURE = 0x00000011

GWL_EXSTYLE = -20
WS_EX_TOPMOST = 0x00000008
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000

HWND_TOPMOST = -1
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020
SWP_SHOWWINDOW = 0x0040

WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_SYSKEYDOWN = 0x0104
WM_SYSKEYUP = 0x0105
WM_INPUT = 0x00FF
WM_TIMER = 0x0113
WM_QUIT = 0x0012
_WM_GAME_GUARD_ARM = 0x800A
_WM_GAME_GUARD_DISARM = 0x800B
_GAME_VERIFY_TIMER_ID = 1

_RI_MOUSE_LEFT_BUTTON_DOWN = 0x0001
_RI_MOUSE_RIGHT_BUTTON_DOWN = 0x0004
_RIDEV_REMOVE = 0x00000001
_RIDEV_INPUTSINK = 0x00000100
_RID_INPUT = 0x10000003
_RIM_TYPEMOUSE = 0
_MOUSE_MOVE_ABSOLUTE = 0x0001
_HWND_MESSAGE = wintypes.HWND(-3)

_QUNS_RUNNING_D3D_FULL_SCREEN = 3
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_MONITOR_DEFAULTTONEAREST = 2
_CURSOR_SHOWING = 0x00000001

_NON_GAME_EXES: FrozenSet[str] = frozenset({
    "explorer.exe", "shellexperiencehost.exe", "searchhost.exe",
    "startmenuexperiencehost.exe", "applicationframehost.exe", "lockapp.exe",
    "taskmgr.exe", "silentvoice.exe",
    "chrome.exe", "msedge.exe", "firefox.exe", "opera.exe", "brave.exe", "vivaldi.exe", "yandex.exe",
    "vlc.exe", "mpc-hc.exe", "mpc-hc64.exe", "mpc-be.exe", "mpc-be64.exe",
    "potplayermini.exe", "potplayermini64.exe", "mpv.exe", "mpvnet.exe",
    "smplayer.exe", "kmplayer.exe", "kmplayer64.exe", "kodi.exe", "wmplayer.exe", "video.ui.exe",
    "discord.exe", "discordptb.exe", "discordcanary.exe", "telegram.exe",
    "obs64.exe", "obs32.exe", "streamlabs obs.exe", "spotify.exe",
    "code.exe", "devenv.exe", "pycharm64.exe", "windowsterminal.exe", "cmd.exe", "powershell.exe",
    "notepad.exe", "notepad++.exe", "mstsc.exe",
})

_LAUNCHER_EXES: FrozenSet[str] = frozenset({
    "steam.exe", "steamwebhelper.exe", "epicgameslauncher.exe", "epicwebhelper.exe",
    "riotclientservices.exe", "riotclientux.exe", "battle.net.exe", "eadesktop.exe",
    "upc.exe", "ubisoftconnect.exe", "galaxyclient.exe", "xboxpcapp.exe",
})

_GAME_PATH_MARKERS: Tuple[str, ...] = (
    "\\steamapps\\common\\",
    "\\epic games\\",
    "\\riot games\\",
    "\\xboxgames\\",
    "\\gog galaxy\\games\\",
    "\\ubisoft game launcher\\games\\",
    "\\ea games\\",
    "\\origin games\\",
    "\\hoyoverse\\",
    "\\mihoyo\\",
    "\\escapefromtarkov\\",
    "\\games\\",
)

_GAME_MOUSE_THRESHOLDS: Dict[str, Optional[float]] = {
    "high": 90.0,
    "medium": 180.0,
    "low": 340.0,
    "click_only": None,
}

VK_BACK = 0x08
VK_TAB = 0x09
VK_RETURN = 0x0D
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12
VK_PAUSE = 0x13
VK_CAPITAL = 0x14
VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_PRIOR = 0x21
VK_NEXT = 0x22
VK_END = 0x23
VK_HOME = 0x24
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_INSERT = 0x2D
VK_DELETE = 0x2E
VK_LWIN = 0x5B
VK_RWIN = 0x5C
VK_LSHIFT = 0xA0
VK_RSHIFT = 0xA1
VK_LCONTROL = 0xA2
VK_RCONTROL = 0xA3
VK_LMENU = 0xA4
VK_RMENU = 0xA5

# Назви віртуальних клавіш для красивого відображення та розпізнавання хоткеїв
VK_TO_NAME: Dict[int, str] = {
    VK_SHIFT: "Shift",
    VK_CONTROL: "Ctrl",
    VK_MENU: "Alt",
    VK_LWIN: "Win",
    VK_BACK: "Backspace",
    VK_TAB: "Tab",
    VK_RETURN: "Enter",
    VK_PAUSE: "Pause",
    VK_CAPITAL: "CapsLock",
    VK_ESCAPE: "Esc",
    VK_SPACE: "Space",
    VK_PRIOR: "PageUp",
    VK_NEXT: "PageDown",
    VK_END: "End",
    VK_HOME: "Home",
    VK_LEFT: "Left",
    VK_UP: "Up",
    VK_RIGHT: "Right",
    VK_DOWN: "Down",
    VK_INSERT: "Insert",
    VK_DELETE: "Delete",
    0x2C: "PrintScreen",
    0x90: "NumLock",
    0x91: "ScrollLock",
    0xAD: "VolumeMute",
    0xAE: "VolumeDown",
    0xAF: "VolumeUp",
    0xB0: "MediaNext",
    0xB1: "MediaPrev",
    0xB2: "MediaStop",
    0xB3: "MediaPlayPause",
    0xC0: "`",
    0xBD: "-",
    0xBB: "=",
    0xDB: "[",
    0xDD: "]",
    0xDC: "\\",
    0xBA: ";",
    0xDE: "'",
    0xBC: ",",
    0xBE: ".",
    0xBF: "/",
}

# Додаємо A-Z, 0-9, F1-F24, Numpad 0-9
for _c in range(ord("0"), ord("9") + 1):
    VK_TO_NAME[_c] = chr(_c)
for _c in range(ord("A"), ord("Z") + 1):
    VK_TO_NAME[_c] = chr(_c)
for _f in range(1, 25):
    VK_TO_NAME[0x70 + _f - 1] = f"F{_f}"
for _n in range(10):
    VK_TO_NAME[0x60 + _n] = f"Num{_n}"

VK_TO_NAME[0x6A] = "Num*"
VK_TO_NAME[0x6B] = "Num+"
VK_TO_NAME[0x6D] = "Num-"
VK_TO_NAME[0x6E] = "Num."
VK_TO_NAME[0x6F] = "Num/"

NAME_TO_VK: Dict[str, int] = {v.lower(): k for k, v in VK_TO_NAME.items()}
NAME_TO_VK["return"] = VK_RETURN
NAME_TO_VK["escape"] = VK_ESCAPE
NAME_TO_VK["caps"] = VK_CAPITAL
NAME_TO_VK["capslock"] = VK_CAPITAL
NAME_TO_VK["pgup"] = VK_PRIOR
NAME_TO_VK["pgdn"] = VK_NEXT
NAME_TO_VK["del"] = VK_DELETE
NAME_TO_VK["ins"] = VK_INSERT


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


LRESULT = ctypes.c_ssize_t
HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

user32.SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD]
user32.SetWindowsHookExW.restype = wintypes.HHOOK
user32.CallNextHookEx.argtypes = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
user32.CallNextHookEx.restype = LRESULT
user32.UnhookWindowsHookEx.argtypes = [wintypes.HHOOK]
user32.UnhookWindowsHookEx.restype = wintypes.BOOL
user32.GetForegroundWindow.argtypes = []
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.GetKeyboardLayout.argtypes = [wintypes.DWORD]
user32.GetKeyboardLayout.restype = wintypes.HKL
user32.GetKeyboardLayoutList.argtypes = [ctypes.c_int, ctypes.POINTER(wintypes.HKL)]
user32.GetKeyboardLayoutList.restype = ctypes.c_int
user32.ToUnicodeEx.argtypes = [
    wintypes.UINT,
    wintypes.UINT,
    ctypes.POINTER(ctypes.c_ubyte),
    wintypes.LPWSTR,
    ctypes.c_int,
    wintypes.UINT,
    wintypes.HKL,
]
user32.ToUnicodeEx.restype = ctypes.c_int
user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
if hasattr(user32, "GetWindowLongPtrW"):
    user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
if hasattr(user32, "SetWindowLongPtrW"):
    user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
    user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetWindowLongW.restype = wintypes.LONG
user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.LONG]
user32.SetWindowLongW.restype = wintypes.LONG
user32.SetWindowPos.argtypes = [
    wintypes.HWND,
    wintypes.HWND,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.UINT,
]
user32.SetWindowPos.restype = wintypes.BOOL
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.SetForegroundWindow.restype = wintypes.BOOL
user32.SetFocus.argtypes = [wintypes.HWND]
user32.SetFocus.restype = wintypes.HWND
user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
user32.AttachThreadInput.restype = wintypes.BOOL
user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = wintypes.SHORT
user32.GetKeyState.argtypes = [ctypes.c_int]
user32.GetKeyState.restype = wintypes.SHORT
user32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
user32.MapVirtualKeyW.restype = wintypes.UINT
user32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.PostThreadMessageW.restype = wintypes.BOOL
user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
user32.GetMessageW.restype = wintypes.BOOL
user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
user32.TranslateMessage.restype = wintypes.BOOL
user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
user32.DispatchMessageW.restype = LRESULT
shell32.ShellExecuteW.argtypes = [
    wintypes.HWND,
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    ctypes.c_int,
]
shell32.ShellExecuteW.restype = wintypes.HINSTANCE


def set_window_capture_excluded(hwnd: int, excluded: bool) -> bool:
    """
    Вмикає або вимикає невидимість вікна для OBS Studio, Discord Screen Share та запису екрана.
    Використовує WDA_EXCLUDEFROMCAPTURE (Windows 10 2004+ / Windows 11).
    """
    if not hwnd:
        return False
    try:
        affinity = WDA_EXCLUDEFROMCAPTURE if excluded else WDA_NONE
        res = user32.SetWindowDisplayAffinity(wintypes.HWND(int(hwnd)), affinity)
        return bool(res)
    except Exception as exc:
        print(f"[WinAPI] SetWindowDisplayAffinity error: {exc}")
        return False


def apply_overlay_ex_styles(hwnd: int, click_through: bool = True, no_activate: bool = True) -> None:
    """
    Налаштовує розширені стилі вікна оверлею:
    - WS_EX_TOPMOST: поверх усіх вікон та ігор у режимі Borderless
    - WS_EX_TRANSPARENT: пропускає всі кліки миші крізь себе у гру
    - WS_EX_NOACTIVATE: ніколи не забирає фокус у гри
    - WS_EX_TOOLWINDOW: не відображається в Alt+Tab та на панелі завдань
    """
    if not hwnd:
        return
    try:
        get_long = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
        set_long = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
        h = wintypes.HWND(int(hwnd))
        ex_style = get_long(h, GWL_EXSTYLE)
        ex_style |= WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_LAYERED
        if click_through:
            ex_style |= WS_EX_TRANSPARENT
        else:
            ex_style &= ~WS_EX_TRANSPARENT
        if no_activate:
            ex_style |= WS_EX_NOACTIVATE
        else:
            ex_style &= ~WS_EX_NOACTIVATE
        set_long(h, GWL_EXSTYLE, ex_style)
        user32.SetWindowPos(
            h,
            wintypes.HWND(HWND_TOPMOST),
            0, 0, 0, 0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED,
        )
    except Exception as exc:
        print(f"[WinAPI] apply_overlay_ex_styles error: {exc}")


def get_foreground_hwnd() -> int:
    try:
        return int(user32.GetForegroundWindow() or 0)
    except Exception:
        return 0


def restore_foreground_hwnd(hwnd: int) -> None:
    """Повертає фокус попередньому активному вікну (грі/програмі)."""
    if not hwnd:
        return
    try:
        h = wintypes.HWND(int(hwnd))
        fg = user32.GetForegroundWindow()
        if fg == h:
            return
        fg_tid = user32.GetWindowThreadProcessId(fg, None)
        cur_tid = kernel32.GetCurrentThreadId()
        attached = False
        if fg_tid and cur_tid and fg_tid != cur_tid:
            attached = bool(user32.AttachThreadInput(cur_tid, fg_tid, True))
        user32.SetForegroundWindow(h)
        user32.SetFocus(h)
        if attached:
            user32.AttachThreadInput(cur_tid, fg_tid, False)
    except Exception as exc:
        print(f"[WinAPI] restore_foreground_hwnd error: {exc}")


@dataclass(frozen=True)
class HotkeySpec:
    """Опис гарячої клавіші: стандартна (mods+vk), лише модифікатори (vk=None) або подвійне натискання (is_double=True)."""

    mods: FrozenSet[str]
    vk: Optional[int]
    is_double: bool = False


def _canonical_mod_vk(vk: int) -> Optional[int]:
    if vk in (VK_SHIFT, VK_LSHIFT, VK_RSHIFT):
        return VK_SHIFT
    if vk in (VK_CONTROL, VK_LCONTROL, VK_RCONTROL):
        return VK_CONTROL
    if vk in (VK_MENU, VK_LMENU, VK_RMENU):
        return VK_MENU
    if vk in (VK_LWIN, VK_RWIN):
        return VK_LWIN
    return None


def _vk_to_mod_name(vk: int) -> Optional[str]:
    cvk = _canonical_mod_vk(vk)
    if cvk == VK_SHIFT:
        return "shift"
    if cvk == VK_CONTROL:
        return "ctrl"
    if cvk == VK_MENU:
        return "alt"
    if cvk == VK_LWIN:
        return "win"
    return None


_MOD_NAME_TO_VK: Dict[str, int] = {
    "shift": VK_SHIFT,
    "ctrl": VK_CONTROL,
    "alt": VK_MENU,
    "win": VK_LWIN,
}


def parse_hotkey_string(hotkey_str: str) -> Optional[HotkeySpec]:
    """
    Розбирає рядок хоткею:
    - стандартний ('Ctrl+Space', 'CapsLock', 'Alt+1', 'Ctrl+Shift+S')
    - лише модифікатори ('Alt', 'Ctrl+Alt', 'Ctrl+Shift')
    - подвійне натискання ('2x Shift', '2x Ctrl', '2x Space', 'Double+Shift')
    """
    if not hotkey_str or not hotkey_str.strip():
        return None
    raw = hotkey_str.strip()
    is_double = False
    low_raw = raw.lower()
    for prefix in ("2x ", "2х ", "double+", "double "):
        if low_raw.startswith(prefix):
            is_double = True
            raw = raw[len(prefix) :].strip()
            break

    parts = [p.strip() for p in raw.replace(" ", "").split("+") if p.strip()]
    if not parts:
        return None
    mods: Set[str] = set()
    main_vk: Optional[int] = None
    for part in parts:
        low = part.lower()
        if low in ("ctrl", "control", "lctrl", "rctrl"):
            mods.add("ctrl")
        elif low in ("alt", "menu", "lalt", "ralt"):
            mods.add("alt")
        elif low in ("shift", "lshift", "rshift"):
            mods.add("shift")
        elif low in ("win", "meta", "super", "lwin", "rwin"):
            mods.add("win")
        else:
            if low in NAME_TO_VK:
                main_vk = NAME_TO_VK[low]
            elif low.startswith("vk_") and low[3:].isdigit():
                # Keep uncommon keys round-trippable when format_hotkey has no friendly name.
                main_vk = int(low[3:])
            elif len(part) == 1:
                main_vk = ord(part.upper())

    if is_double:
        if main_vk is not None:
            return HotkeySpec(frozenset(), main_vk, is_double=True)
        if len(mods) == 1:
            mod_name = next(iter(mods))
            return HotkeySpec(frozenset(), _MOD_NAME_TO_VK[mod_name], is_double=True)
        return None

    if main_vk is not None:
        return HotkeySpec(frozenset(mods), main_vk, is_double=False)
    if mods:
        return HotkeySpec(frozenset(mods), None, is_double=False)
    return None


def format_hotkey(mods: Set[str], vk: Optional[int], is_double: bool = False) -> str:
    """Форматує комбінацію клавіш у читабельний рядок ('Ctrl+Shift+F1', 'Ctrl+Alt', '2x Shift')."""
    if is_double and vk is not None:
        key_name = VK_TO_NAME.get(vk, f"VK_{vk}")
        return f"2x {key_name}"
    ordered_mods = []
    if "ctrl" in mods:
        ordered_mods.append("Ctrl")
    if "alt" in mods:
        ordered_mods.append("Alt")
    if "shift" in mods:
        ordered_mods.append("Shift")
    if "win" in mods:
        ordered_mods.append("Win")
    if vk is None:
        return "+".join(ordered_mods)
    key_name = VK_TO_NAME.get(vk, f"VK_{vk}")
    return "+".join(ordered_mods + [key_name])


def hkl_to_lang_code(hkl: int) -> str:
    """Перетворює HKL у короткий код мови ('UA', 'RU', 'EN' тощо)."""
    lang_id = int(hkl) & 0xFFFF
    primary = lang_id & 0x3FF
    if primary == 0x22:
        return "UA"
    if primary == 0x19:
        return "RU"
    if primary == 0x09:
        return "EN"
    if primary == 0x15:
        return "PL"
    if primary == 0x07:
        return "DE"
    return f"L{primary:02X}"


def get_system_hkls() -> List[int]:
    """Повертає список усіх встановлених розкладок клавіатури в системі."""
    try:
        count = user32.GetKeyboardLayoutList(0, None)
        if count <= 0:
            return []
        arr = (wintypes.HKL * count)()
        user32.GetKeyboardLayoutList(count, arr)
        return [int(x) for x in arr]
    except Exception:
        return []


class _MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
    ]


class _CURSORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hCursor", wintypes.HANDLE),
        ("ptScreenPos", wintypes.POINT),
    ]


class _RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wintypes.USHORT),
        ("usUsage", wintypes.USHORT),
        ("dwFlags", wintypes.DWORD),
        ("hwndTarget", wintypes.HWND),
    ]


class _RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", wintypes.DWORD),
        ("dwSize", wintypes.DWORD),
        ("hDevice", wintypes.HANDLE),
        ("wParam", wintypes.WPARAM),
    ]


class _RAWMOUSE_BUTTONS_STRUCT(ctypes.Structure):
    _fields_ = [
        ("usButtonFlags", wintypes.USHORT),
        ("usButtonData", wintypes.USHORT),
    ]


class _RAWMOUSE_BUTTONS_UNION(ctypes.Union):
    _anonymous_ = ("_s",)
    _fields_ = [
        ("ulButtons", wintypes.ULONG),
        ("_s", _RAWMOUSE_BUTTONS_STRUCT),
    ]


class _RAWMOUSE(ctypes.Structure):
    _anonymous_ = ("_btn",)
    _fields_ = [
        ("usFlags", wintypes.USHORT),
        ("_btn", _RAWMOUSE_BUTTONS_UNION),
        ("ulRawButtons", wintypes.ULONG),
        ("lLastX", wintypes.LONG),
        ("lLastY", wintypes.LONG),
        ("ulExtraInformation", wintypes.ULONG),
    ]


class _RAWINPUT_DATA(ctypes.Union):
    _fields_ = [("mouse", _RAWMOUSE)]


class _RAWINPUT(ctypes.Structure):
    _fields_ = [
        ("header", _RAWINPUTHEADER),
        ("data", _RAWINPUT_DATA),
    ]


class _WNDCLASSEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.UINT),
        ("style", wintypes.UINT),
        ("lpfnWndProc", ctypes.c_void_p),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HANDLE),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
        ("hIconSm", wintypes.HICON),
    ]


_WIN32_GAME_PROTOS_READY = False


def _ensure_win32_game_prototypes() -> None:
    global _WIN32_GAME_PROTOS_READY
    if _WIN32_GAME_PROTOS_READY or sys.platform != "win32":
        return
    try:
        user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user32.DefWindowProcW.restype = ctypes.c_ssize_t
        user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user32.PostMessageW.restype = wintypes.BOOL
        user32.RegisterRawInputDevices.argtypes = [ctypes.c_void_p, wintypes.UINT, wintypes.UINT]
        user32.RegisterRawInputDevices.restype = wintypes.BOOL
        user32.GetRawInputData.argtypes = [
            wintypes.HANDLE, wintypes.UINT, ctypes.c_void_p, ctypes.POINTER(wintypes.UINT), wintypes.UINT,
        ]
        user32.GetRawInputData.restype = wintypes.UINT
        user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        user32.GetWindowRect.restype = wintypes.BOOL
        user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
        user32.MonitorFromWindow.restype = wintypes.HMONITOR
        user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.c_void_p]
        user32.GetMonitorInfoW.restype = wintypes.BOOL
        user32.IsIconic.argtypes = [wintypes.HWND]
        user32.IsIconic.restype = wintypes.BOOL
        user32.GetClipCursor.argtypes = [ctypes.POINTER(wintypes.RECT)]
        user32.GetClipCursor.restype = wintypes.BOOL
        user32.GetCursorInfo.argtypes = [ctypes.c_void_p]
        user32.GetCursorInfo.restype = wintypes.BOOL
        user32.InSendMessageEx.argtypes = [wintypes.LPVOID]
        user32.InSendMessageEx.restype = wintypes.DWORD
        user32.RegisterClassExW.argtypes = [ctypes.c_void_p]
        user32.RegisterClassExW.restype = wintypes.ATOM
        user32.CreateWindowExW.argtypes = [
            wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, ctypes.c_void_p,
        ]
        user32.CreateWindowExW.restype = wintypes.HWND
        user32.DestroyWindow.argtypes = [wintypes.HWND]
        user32.DestroyWindow.restype = wintypes.BOOL
        user32.UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HINSTANCE]
        user32.UnregisterClassW.restype = wintypes.BOOL
        user32.SetTimer.argtypes = [wintypes.HWND, ctypes.c_size_t, wintypes.UINT, ctypes.c_void_p]
        user32.SetTimer.restype = ctypes.c_size_t
        user32.KillTimer.argtypes = [wintypes.HWND, ctypes.c_size_t]
        user32.KillTimer.restype = wintypes.BOOL
        kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        kernel32.GetModuleHandleW.restype = wintypes.HINSTANCE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        shell32.SHQueryUserNotificationState.argtypes = [ctypes.POINTER(wintypes.DWORD)]
        shell32.SHQueryUserNotificationState.restype = ctypes.c_long
    except Exception:
        pass
    _WIN32_GAME_PROTOS_READY = True


def _read_raw_mouse_input(lparam: int) -> Tuple[int, int, int, bool]:
    size = wintypes.UINT(0)
    hdr_sz = ctypes.sizeof(_RAWINPUTHEADER)
    hraw = wintypes.HANDLE(lparam)
    if user32.GetRawInputData(hraw, _RID_INPUT, None, ctypes.byref(size), hdr_sz) != 0 or size.value == 0:
        return 0, 0, 0, False
    buf = (ctypes.c_ubyte * size.value)()
    written = user32.GetRawInputData(hraw, _RID_INPUT, buf, ctypes.byref(size), hdr_sz)
    if written == 0xFFFFFFFF or written < ctypes.sizeof(_RAWINPUTHEADER) + ctypes.sizeof(_RAWMOUSE):
        return 0, 0, 0, False
    raw = _RAWINPUT.from_buffer(buf)
    if raw.header.dwType != _RIM_TYPEMOUSE:
        return 0, 0, 0, False
    mouse = raw.data.mouse
    return (
        int(mouse.lLastX),
        int(mouse.lLastY),
        int(mouse.usButtonFlags),
        bool(int(mouse.usFlags) & _MOUSE_MOVE_ABSOLUTE),
    )


_GAME_STORE_CACHE_LOCK = threading.Lock()
_GAME_STORE_CACHE_AT: float = 0.0
_GAME_STORE_CACHE_PATHS: FrozenSet[str] = frozenset()


def _get_windows_game_store_paths(ttl_sec: float = 30.0) -> FrozenSet[str]:
    """Читає список підтверджених ігор із реєстру Windows GameConfigStore (із кешем 30 с)."""
    global _GAME_STORE_CACHE_AT, _GAME_STORE_CACHE_PATHS
    if sys.platform != "win32":
        return frozenset()
    now = time.monotonic()
    with _GAME_STORE_CACHE_LOCK:
        if _GAME_STORE_CACHE_AT > 0.0 and (now - _GAME_STORE_CACHE_AT) < ttl_sec:
            return _GAME_STORE_CACHE_PATHS

    paths: Set[str] = set()
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"System\GameConfigStore\Children") as root_key:
            sub_count = int(winreg.QueryInfoKey(root_key)[0] or 0)
            for idx in range(sub_count):
                try:
                    sub_name = winreg.EnumKey(root_key, idx)
                    with winreg.OpenKey(root_key, sub_name) as sub_key:
                        exe_val, _ = winreg.QueryValueEx(sub_key, "MatchedExeFullPath")
                        norm = str(exe_val or "").strip().lower().replace("/", "\\")
                        if norm:
                            paths.add(norm)
                except OSError:
                    continue
    except OSError:
        pass

    frozen = frozenset(paths)
    with _GAME_STORE_CACHE_LOCK:
        _GAME_STORE_CACHE_AT = time.monotonic()
        _GAME_STORE_CACHE_PATHS = frozen
    return frozen


def _is_cursor_still_confined_or_hidden() -> bool:
    """Перевіряє, чи курсор досі прихований або затиснутий грою (на відміну від плеєра при русі миші)."""
    if sys.platform != "win32":
        return False
    try:
        _ensure_win32_game_prototypes()
        ci = _CURSORINFO(cbSize=ctypes.sizeof(_CURSORINFO))
        if user32.GetCursorInfo(ctypes.byref(ci)):
            if (int(ci.flags) & _CURSOR_SHOWING) == 0 or not ci.hCursor:
                return True
        hwnd = user32.GetForegroundWindow()
        if hwnd:
            mon = user32.MonitorFromWindow(hwnd, _MONITOR_DEFAULTTONEAREST)
            if mon:
                mi = _MONITORINFO(cbSize=ctypes.sizeof(_MONITORINFO))
                clip = wintypes.RECT()
                if user32.GetMonitorInfoW(mon, ctypes.byref(mi)) and user32.GetClipCursor(ctypes.byref(clip)):
                    m = mi.rcMonitor
                    mon_w = max(1, m.right - m.left)
                    mon_h = max(1, m.bottom - m.top)
                    clip_w = max(0, clip.right - clip.left)
                    clip_h = max(0, clip.bottom - clip.top)
                    if clip_w > 0 and clip_h > 0 and (clip_w < mon_w - 8 or clip_h < mon_h - 8):
                        return True
    except Exception:
        pass
    return False


def classify_foreground_game_detail(
    *,
    is_self: bool,
    exe_name: str,
    exe_path: str,
    quns_state: Optional[int],
    is_fullscreen: bool,
    cursor_hidden: bool,
    cursor_clipped: bool,
    in_game_store: bool = False,
) -> Tuple[bool, str]:
    """
    Визначає, чи зараз у фокусі гра, та рівень впевненості ('strong' або 'weak_hidden').
    """
    if is_self:
        return False, ""
    low_name = str(exe_name or "").strip().lower()
    low_path = str(exe_path or "").strip().lower().replace("/", "\\")
    if low_name in _NON_GAME_EXES or low_name in _LAUNCHER_EXES:
        return False, ""
    if quns_state == _QUNS_RUNNING_D3D_FULL_SCREEN:
        return True, "strong"
    if in_game_store:
        return True, "strong"
    if low_path and any(marker in low_path for marker in _GAME_PATH_MARKERS):
        return True, "strong"
    if is_fullscreen and cursor_clipped:
        return True, "strong"
    if is_fullscreen and cursor_hidden:
        return True, "weak_hidden"
    return False, ""


def classify_foreground_game(
    *,
    is_self: bool,
    exe_name: str,
    exe_path: str,
    quns_state: Optional[int],
    is_fullscreen: bool,
    cursor_hidden: bool,
    cursor_clipped: bool,
    in_game_store: bool = False,
) -> bool:
    """Безпечно визначає за лише зовнішніми даними Windows, чи зараз у фокусі гра."""
    return classify_foreground_game_detail(
        is_self=is_self,
        exe_name=exe_name,
        exe_path=exe_path,
        quns_state=quns_state,
        is_fullscreen=is_fullscreen,
        cursor_hidden=cursor_hidden,
        cursor_clipped=cursor_clipped,
        in_game_store=in_game_store,
    )[0]


def probe_foreground_game_state() -> Tuple[bool, str]:
    """
    Повертає (is_game, match_kind): 'strong' (DirectX FSE, GameConfigStore,
    ігрова папка чи ClipCursor) або 'weak_hidden' (лише повний екран + прихований курсор).
    """
    if sys.platform != "win32":
        return False, ""
    try:
        _ensure_win32_game_prototypes()
        hwnd = user32.GetForegroundWindow()
        if not hwnd or user32.IsIconic(hwnd):
            return False, ""

        pid = wintypes.DWORD(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        is_self = int(pid.value) == os.getpid()
        if is_self:
            return classify_foreground_game_detail(
                is_self=True,
                exe_name="",
                exe_path="",
                quns_state=None,
                is_fullscreen=False,
                cursor_hidden=False,
                cursor_clipped=False,
                in_game_store=False,
            )

        exe_path = ""
        exe_name = ""
        if pid.value:
            hproc = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
            if hproc:
                try:
                    cch = wintypes.DWORD(512)
                    buf = ctypes.create_unicode_buffer(512)
                    if kernel32.QueryFullProcessImageNameW(hproc, 0, buf, ctypes.byref(cch)):
                        exe_path = buf.value
                        exe_name = os.path.basename(exe_path)
                finally:
                    kernel32.CloseHandle(hproc)

        low_name = exe_name.strip().lower()
        if low_name in _NON_GAME_EXES or low_name in _LAUNCHER_EXES:
            return classify_foreground_game_detail(
                is_self=False,
                exe_name=exe_name,
                exe_path=exe_path,
                quns_state=None,
                is_fullscreen=False,
                cursor_hidden=False,
                cursor_clipped=False,
                in_game_store=False,
            )

        norm_path = exe_path.strip().lower().replace("/", "\\")
        has_game_path = bool(norm_path and any(marker in norm_path for marker in _GAME_PATH_MARKERS))
        in_game_store = False if has_game_path else bool(norm_path and norm_path in _get_windows_game_store_paths())

        is_fullscreen = False
        cursor_clipped = False
        mon = user32.MonitorFromWindow(hwnd, _MONITOR_DEFAULTTONEAREST)
        if mon:
            mi = _MONITORINFO(cbSize=ctypes.sizeof(_MONITORINFO))
            wr = wintypes.RECT()
            if user32.GetMonitorInfoW(mon, ctypes.byref(mi)) and user32.GetWindowRect(hwnd, ctypes.byref(wr)):
                m = mi.rcMonitor
                is_fullscreen = (
                    wr.left <= m.left + 2
                    and wr.top <= m.top + 2
                    and wr.right >= m.right - 2
                    and wr.bottom >= m.bottom - 2
                )
                clip = wintypes.RECT()
                if user32.GetClipCursor(ctypes.byref(clip)):
                    mon_w = max(1, m.right - m.left)
                    mon_h = max(1, m.bottom - m.top)
                    clip_w = max(0, clip.right - clip.left)
                    clip_h = max(0, clip.bottom - clip.top)
                    cursor_clipped = clip_w > 0 and clip_h > 0 and (clip_w < mon_w - 8 or clip_h < mon_h - 8)

        cursor_hidden = False
        ci = _CURSORINFO(cbSize=ctypes.sizeof(_CURSORINFO))
        if user32.GetCursorInfo(ctypes.byref(ci)):
            cursor_hidden = (int(ci.flags) & _CURSOR_SHOWING) == 0 or not ci.hCursor

        quns_val: Optional[int] = None
        if is_fullscreen and not in_game_store and not has_game_path and not cursor_clipped:
            if user32.InSendMessageEx(None) == 0:
                st = wintypes.DWORD(0)
                if shell32.SHQueryUserNotificationState(ctypes.byref(st)) == 0:
                    quns_val = int(st.value)

        return classify_foreground_game_detail(
            is_self=is_self,
            exe_name=exe_name,
            exe_path=exe_path,
            quns_state=quns_val,
            is_fullscreen=is_fullscreen,
            cursor_hidden=cursor_hidden,
            cursor_clipped=cursor_clipped,
            in_game_store=in_game_store,
        )
    except Exception:
        return False, ""


def is_foreground_game_active() -> bool:
    """
    Визначає, чи зараз користувач у грі (DirectX Fullscreen, Windows GameConfigStore,
    Borderless із захопленням курсора або процес із відомих ігрових бібліотек).
    """
    return probe_foreground_game_state()[0]


class _GameMouseGuard:
    """Відстежує постріл/приціл (ЛКМ/ПКМ) або різкий ривок миші за коротке вікно (120 мс)."""

    _WINDOW_SEC = 0.12
    _ARM_GRACE_SEC = 0.04

    def __init__(self, sensitivity: str = "low") -> None:
        self._sensitivity = sensitivity if sensitivity in _GAME_MOUSE_THRESHOLDS else "low"
        self._events: Deque[Tuple[float, float]] = deque()
        self._armed_at: float = 0.0

    def set_sensitivity(self, sensitivity: str) -> None:
        self._sensitivity = sensitivity if sensitivity in _GAME_MOUSE_THRESHOLDS else "low"

    def reset(self, now: Optional[float] = None) -> None:
        self._events.clear()
        self._armed_at = time.monotonic() if now is None else now

    def feed(
        self,
        dx: int,
        dy: int,
        button_flags: int,
        *,
        is_absolute: bool = False,
        now: Optional[float] = None,
    ) -> bool:
        ts = time.monotonic() if now is None else now
        if button_flags & (_RI_MOUSE_LEFT_BUTTON_DOWN | _RI_MOUSE_RIGHT_BUTTON_DOWN):
            return True
        if (ts - self._armed_at) < self._ARM_GRACE_SEC:
            return False
        thresh = _GAME_MOUSE_THRESHOLDS.get(self._sensitivity, _GAME_MOUSE_THRESHOLDS["low"])
        if thresh is None or is_absolute:
            return False
        dist = math.hypot(dx, dy)
        if dist <= 0.0:
            return False
        self._events.append((ts, dist))
        cutoff = ts - self._WINDOW_SEC
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()
        total = sum(d for _, d in self._events)
        return total >= thresh


class KeyboardHookManager(QObject):
    """
    Менеджер глобального перехоплення клавіатури (WH_KEYBOARD_LL) та набору тексту без втрати фокусу.
    Випромінює сигнали Qt у головний потік UI.
    """

    typing_mode_changed = Signal(bool, str)      # (is_typing, lang_code)
    typing_text_changed = Signal(str, str)       # (current_text, lang_code)
    send_triggered = Signal(str)                 # (final_text)
    cancel_triggered = Signal()
    stop_tts_triggered = Signal()
    repeat_last_triggered = Signal()
    toggle_game_mode_triggered = Signal()
    stop_chat_tts_triggered = Signal()
    toggle_chat_tts_triggered = Signal()
    quick_phrase_triggered = Signal(str)         # (phrase_text)
    hotkey_recorded = Signal(str)                # Для діалогу запису нової комбінації клавіш
    _layout_refresh_requested = Signal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._hook_handle: Optional[wintypes.HHOOK] = None
        self._thread: Optional[threading.Thread] = None
        self._thread_id: Optional[int] = None
        self._running = False

        # Стан набору
        self._is_typing = False
        self._buffer: str = ""
        self._input_mode: str = "hook"  # "hook" або "fallback"
        self._overlay_style: str = "text_only"
        self._config_continuous: bool = False
        self._session_continuous: bool = False
        # Ігровий захист (автодетект гри + аварійне скасування в бою + чернетка)
        self._game_mode_enabled: bool = True
        self._game_auto_detect_only: bool = True
        self._game_emergency_cancel: bool = True
        self._game_mouse_sensitivity: str = "low"
        self._game_save_draft: bool = True
        self._game_draft_ttl_sec: int = 20
        self._save_draft_on_close: bool = True
        self._draft_keep_ttl_sec: int = 5
        self._saved_draft_text: str = ""
        self._saved_draft_expires_at: float = 0.0
        self._game_guard = _GameMouseGuard()
        self._game_guard_armed: bool = False
        self._game_guard_weak_mode: bool = False
        self._game_guard_pending_verify: bool = False
        self._raw_hwnd: int = 0
        self._raw_wndclass_atom: int = 0
        self._raw_wndproc_ref = None
        # Запис хоткею
        self._recording_hotkey = False
        self._recording_double = False
        self._rec_mod_peak: Set[str] = set()
        self._rec_saw_non_mod = False

        # Стан натиснутих клавіш і подвійного натискання
        self._pressed_vks: Set[int] = set()
        self._last_tap_vk: Optional[int] = None
        self._last_tap_ts: float = 0.0
        self._mod_combo_fired = False

        # Відкладене оновлення індикатора мови після системного перемикання розкладки
        self._layout_timer = QTimer(self)
        self._layout_timer.setSingleShot(True)
        self._layout_timer.setInterval(55)
        self._layout_timer.timeout.connect(self._emit_live_layout_refresh)
        self._layout_refresh_requested.connect(self._layout_timer.start)

        # Хоткеї
        self._hk_toggle: Optional[HotkeySpec] = parse_hotkey_string("Ctrl+Space")
        self._hk_conversation: Optional[HotkeySpec] = parse_hotkey_string("Ctrl+Shift+Space")
        self._hk_send: Optional[HotkeySpec] = parse_hotkey_string("Enter")
        self._hk_cancel: Optional[HotkeySpec] = parse_hotkey_string("Esc")
        self._hk_stop: Optional[HotkeySpec] = parse_hotkey_string("Ctrl+Shift+S")
        self._hk_repeat: Optional[HotkeySpec] = parse_hotkey_string("Ctrl+Shift+R")
        self._hk_toggle_game_mode: Optional[HotkeySpec] = parse_hotkey_string("Ctrl+Shift+G")
        self._hk_stop_chat: Optional[HotkeySpec] = None
        self._hk_toggle_chat: Optional[HotkeySpec] = None
        self._chat_tts_enabled: bool = False
        self._quick_hotkeys: List[Tuple[HotkeySpec, str]] = []
        # Відкладаємо одиночну фразу, якщо для тієї ж клавіші є хоткей 2x.
        # Інакше одиночна фраза встигає спрацювати до другого натискання.
        self._pending_quick_lock = threading.Lock()
        self._pending_quick_singles: Dict[int, Tuple[threading.Timer, str]] = {}

        # Тримаємо посилання на callback, щоб GC не видалив його
        self._proc_ref = HOOKPROC(self._low_level_keyboard_proc)

    def update_from_config(self, cfg_data: dict) -> None:
        """Оновлює гарячі клавіші та режим вводу з конфігурації."""
        self._cancel_pending_quick_singles()
        self._input_mode = cfg_data.get("input_mode", "hook")
        self._overlay_style = cfg_data.get("overlay_style", "text_only")
        self._config_continuous = bool(cfg_data.get("continuous_chat_mode", False))
        self._hk_toggle = parse_hotkey_string(cfg_data.get("hotkey_toggle_input", "Ctrl+Space"))
        self._hk_conversation = parse_hotkey_string(cfg_data.get("hotkey_conversation_mode", "Ctrl+Shift+Space"))
        self._hk_send = parse_hotkey_string(cfg_data.get("hotkey_send", "Enter"))
        self._hk_cancel = parse_hotkey_string(cfg_data.get("hotkey_cancel", "Esc"))
        self._hk_stop = parse_hotkey_string(cfg_data.get("hotkey_stop_tts", "Ctrl+Shift+S"))
        self._hk_repeat = parse_hotkey_string(cfg_data.get("hotkey_repeat_last", "Ctrl+Shift+R"))
        self._hk_toggle_game_mode = parse_hotkey_string(cfg_data.get("hotkey_toggle_game_mode", "Ctrl+Shift+G"))
        self._hk_stop_chat = parse_hotkey_string(cfg_data.get("hotkey_stop_chat_tts", ""))
        self._hk_toggle_chat = parse_hotkey_string(cfg_data.get("hotkey_toggle_chat_tts", ""))
        self._chat_tts_enabled = bool(cfg_data.get("twitch_read_viewers", False))

        self._game_mode_enabled = bool(cfg_data.get("game_mode_enabled", True))
        self._game_auto_detect_only = bool(cfg_data.get("game_auto_detect_only", True))
        self._game_emergency_cancel = bool(cfg_data.get("game_emergency_cancel", True))
        sens = str(cfg_data.get("game_mouse_sensitivity", "low") or "low").strip().lower()
        self._game_mouse_sensitivity = sens if sens in _GAME_MOUSE_THRESHOLDS else "low"
        self._game_guard.set_sensitivity(self._game_mouse_sensitivity)
        self._game_save_draft = bool(cfg_data.get("game_save_draft", True))
        try:
            self._game_draft_ttl_sec = max(5, min(300, int(cfg_data.get("game_draft_ttl_sec", 20))))
        except Exception:
            self._game_draft_ttl_sec = 20
        self._save_draft_on_close = bool(cfg_data.get("save_draft_on_close", True))
        try:
            self._draft_keep_ttl_sec = max(1, min(300, int(cfg_data.get("draft_keep_ttl_sec", 5))))
        except Exception:
            self._draft_keep_ttl_sec = 5
        if not self._save_draft_on_close and not (self._game_mode_enabled and self._game_save_draft):
            self._clear_saved_draft()
        self._sync_game_guard()

        q_list: List[Tuple[HotkeySpec, str]] = []
        for item in cfg_data.get("quick_phrases", []):
            hk_str = item.get("hotkey", "")
            text = item.get("text", "")
            parsed = parse_hotkey_string(hk_str)
            if parsed and text:
                q_list.append((parsed, text))
        self._quick_hotkeys = q_list

    def _cancel_pending_quick_singles(self, vk: Optional[int] = None) -> None:
        """Скасовує відкладені одиночні фрази (усі або для конкретної клавіші)."""
        with self._pending_quick_lock:
            keys = list(self._pending_quick_singles) if vk is None else [vk]
            for key in keys:
                pending = self._pending_quick_singles.pop(key, None)
                if pending:
                    pending[0].cancel()

    def _fire_pending_quick_single(self, vk: int, timer: threading.Timer) -> None:
        with self._pending_quick_lock:
            pending = self._pending_quick_singles.get(vk)
            if pending is None or pending[0] is not timer:
                return
            self._pending_quick_singles.pop(vk, None)
        self.quick_phrase_triggered.emit(pending[1])

    def _defer_quick_single(self, vk: int, text: str) -> None:
        """Запускає одиночну фразу після інтервалу розпізнавання подвійного натискання."""
        # Передаємо той самий об'єкт таймера в callback для захисту від застарілих таймерів.
        timer = threading.Timer(0.42, lambda: self._fire_pending_quick_single(vk, timer))
        timer.daemon = True
        with self._pending_quick_lock:
            previous = self._pending_quick_singles.pop(vk, None)
            if previous:
                previous[0].cancel()
            self._pending_quick_singles[vk] = (timer, text)
        timer.start()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._hook_loop, daemon=True, name="KeyboardHookThread")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread_id:
            try:
                user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            except Exception:
                pass

    def start_recording_hotkey(self, double_mode: bool = False) -> None:
        """Вмикає режим одноразового перехоплення наступної комбінації клавіш для налаштувань."""
        self._recording_hotkey = True
        self._recording_double = bool(double_mode)
        self._rec_mod_peak = set()
        self._rec_saw_non_mod = False

    def set_recording_double(self, double_mode: bool) -> None:
        self._recording_double = bool(double_mode)
        self._rec_mod_peak = set()
        self._rec_saw_non_mod = False

    def stop_recording_hotkey(self) -> None:
        self._recording_hotkey = False
        self._recording_double = False
        self._rec_mod_peak = set()
        self._rec_saw_non_mod = False

    @property
    def is_typing(self) -> bool:
        return self._is_typing

    @property
    def uses_popup_window(self) -> bool:
        return self._input_mode == "fallback" or self._overlay_style == "interactive_window"

    @property
    def is_continuous_active(self) -> bool:
        return self._session_continuous or self._config_continuous

    @property
    def current_text(self) -> str:
        return self._buffer

    def set_typing_mode(self, active: bool, continuous: Optional[bool] = None) -> None:
        """Примусово вмикає або вимикає режим набору (наприклад, через кнопку UI або HTTP API)."""
        if active == self._is_typing and (continuous is None or self._session_continuous == continuous):
            return
        self._is_typing = active
        if active:
            self._buffer = self._consume_valid_draft()
            self._session_continuous = bool(continuous) if continuous is not None else self._config_continuous
        else:
            self._stash_draft_on_close(emergency=False)
            self._buffer = ""
            self._session_continuous = False
        self._sync_game_guard()
        lang = self.get_current_lang_code()
        self.typing_mode_changed.emit(self._is_typing, lang)
        if active and not self.uses_popup_window:
            self.typing_text_changed.emit(self._buffer, lang)
        elif not active:
            self.cancel_triggered.emit()

    def _clear_saved_draft(self) -> None:
        self._saved_draft_text = ""
        self._saved_draft_expires_at = 0.0

    def _stash_draft_on_close(self, text: Optional[str] = None, *, emergency: bool = False) -> None:
        """Зберігає недописаний текст у чернетку на відповідний TTL залежно від режиму закриття."""
        buf = self._buffer if text is None else text
        if not str(buf or "").strip():
            self._clear_saved_draft()
            return
        if emergency:
            if self._game_mode_enabled and self._game_save_draft:
                ttl = float(max(5, self._game_draft_ttl_sec))
            elif self._save_draft_on_close:
                ttl = float(max(1, self._draft_keep_ttl_sec))
            else:
                self._clear_saved_draft()
                return
        elif self._save_draft_on_close:
            ttl = float(max(1, self._draft_keep_ttl_sec))
        else:
            self._clear_saved_draft()
            return
        self._saved_draft_text = str(buf)
        self._saved_draft_expires_at = time.monotonic() + ttl

    def _consume_valid_draft(self, now: Optional[float] = None) -> str:
        cur = time.monotonic() if now is None else now
        draft = self._saved_draft_text
        exp = self._saved_draft_expires_at
        self._clear_saved_draft()
        draft_enabled = self._save_draft_on_close or (self._game_mode_enabled and self._game_save_draft)
        if draft_enabled and draft and cur <= exp:
            return draft
        return ""

    def _should_arm_game_guard_state(self) -> Tuple[bool, bool]:
        if not (self._is_typing and not self.uses_popup_window):
            return False, False
        if not (self._game_mode_enabled and self._game_emergency_cancel):
            return False, False
        if self._game_auto_detect_only:
            is_game, kind = probe_foreground_game_state()
            if not is_game:
                return False, False
            return True, (kind == "weak_hidden")
        return True, False

    def _should_arm_game_guard(self) -> bool:
        return self._should_arm_game_guard_state()[0]

    def _disarm_game_guard_session(self) -> None:
        self._game_guard_armed = False
        self._game_guard_weak_mode = False
        self._game_guard_pending_verify = False
        self._game_guard.reset()
        hwnd = self._raw_hwnd
        if hwnd:
            try:
                user32.PostMessageW(wintypes.HWND(hwnd), _WM_GAME_GUARD_DISARM, 0, 0)
            except Exception:
                pass

    def _sync_game_guard(self) -> None:
        want, weak_mode = self._should_arm_game_guard_state()
        self._game_guard_weak_mode = weak_mode
        if want == self._game_guard_armed:
            return
        self._game_guard_armed = want
        self._game_guard_pending_verify = False
        self._game_guard.reset()
        hwnd = self._raw_hwnd
        if hwnd:
            try:
                user32.PostMessageW(
                    wintypes.HWND(hwnd),
                    _WM_GAME_GUARD_ARM if want else _WM_GAME_GUARD_DISARM,
                    0,
                    0,
                )
            except Exception:
                pass

    def _verify_weak_game_trigger(self) -> None:
        """
        Перевіряє через ~30 мс після руху/кліку в невідомій повноекранній програмі,
        чи курсор досі прихований/затиснутий (гра) або з'явився назад (відеоплеєр).
        """
        self._game_guard_pending_verify = False
        if not (self._game_guard_armed and self._is_typing):
            return
        if _is_cursor_still_confined_or_hidden():
            self._emergency_cancel_typing()
        else:
            self._disarm_game_guard_session()

    def _emergency_cancel_typing(self) -> None:
        """Миттєво скасовує сліпий набір у бою та за потреби зберігає чернетку з таймером."""
        if not self._is_typing or self.uses_popup_window:
            return
        self._stash_draft_on_close(emergency=True)
        self._buffer = ""
        self._is_typing = False
        self._session_continuous = False
        self._sync_game_guard()
        lang = self.get_current_lang_code()
        self.typing_mode_changed.emit(False, lang)
        self.cancel_triggered.emit()

    def _emit_live_layout_refresh(self) -> None:
        if self._is_typing and not self.uses_popup_window:
            self.typing_text_changed.emit(self._buffer, self.get_current_lang_code())

    def get_active_hkl(self) -> int:
        try:
            fg = user32.GetForegroundWindow()
            if fg:
                fg_tid = user32.GetWindowThreadProcessId(fg, None)
                if fg_tid:
                    # Для UWP/багатопотокових вікон спочатку перевіряємо фокусний потік через GetGUIThreadInfo
                    class _GUITHREADINFO(ctypes.Structure):
                        _fields_ = [
                            ("cbSize", wintypes.DWORD),
                            ("flags", wintypes.DWORD),
                            ("hwndActive", wintypes.HWND),
                            ("hwndFocus", wintypes.HWND),
                            ("hwndCapture", wintypes.HWND),
                            ("hwndMenuOwner", wintypes.HWND),
                            ("hwndMoveSize", wintypes.HWND),
                            ("hwndCaret", wintypes.HWND),
                            ("rcCaret", wintypes.RECT),
                        ]

                    gti = _GUITHREADINFO()
                    gti.cbSize = ctypes.sizeof(_GUITHREADINFO)
                    if user32.GetGUIThreadInfo(fg_tid, ctypes.byref(gti)) and gti.hwndFocus:
                        focus_tid = user32.GetWindowThreadProcessId(gti.hwndFocus, None)
                        if focus_tid:
                            hkl = user32.GetKeyboardLayout(focus_tid)
                            if hkl:
                                return int(hkl)
                    hkl = user32.GetKeyboardLayout(fg_tid)
                    if hkl:
                        return int(hkl)
        except Exception:
            pass
        return int(user32.GetKeyboardLayout(0))

    def get_current_lang_code(self) -> str:
        return hkl_to_lang_code(self.get_active_hkl())

    def _get_pressed_modifiers(self) -> Set[str]:
        mods: Set[str] = set()
        if (user32.GetAsyncKeyState(VK_CONTROL) & 0x8000) or (user32.GetAsyncKeyState(VK_LCONTROL) & 0x8000) or (user32.GetAsyncKeyState(VK_RCONTROL) & 0x8000):
            mods.add("ctrl")
        if (user32.GetAsyncKeyState(VK_MENU) & 0x8000) or (user32.GetAsyncKeyState(VK_LMENU) & 0x8000) or (user32.GetAsyncKeyState(VK_RMENU) & 0x8000):
            mods.add("alt")
        if (user32.GetAsyncKeyState(VK_SHIFT) & 0x8000) or (user32.GetAsyncKeyState(VK_LSHIFT) & 0x8000) or (user32.GetAsyncKeyState(VK_RSHIFT) & 0x8000):
            mods.add("shift")
        if (user32.GetAsyncKeyState(VK_LWIN) & 0x8000) or (user32.GetAsyncKeyState(VK_RWIN) & 0x8000):
            mods.add("win")
        return mods

    def _vk_to_unicode(self, vk: int, scan: int) -> str:
        """Перетворює натиснуту клавішу в символ Unicode згідно з активною розкладкою (UA / RU / EN)."""
        state = (ctypes.c_ubyte * 256)()
        shift_down = bool(user32.GetAsyncKeyState(VK_SHIFT) & 0x8000)
        caps_on = bool(user32.GetKeyState(VK_CAPITAL) & 0x0001)
        ralt_down = bool(user32.GetAsyncKeyState(VK_RMENU) & 0x8000)
        ctrl_down = bool(user32.GetAsyncKeyState(VK_CONTROL) & 0x8000)

        if shift_down:
            state[VK_SHIFT] = 0x80
            state[VK_LSHIFT] = 0x80
        if caps_on:
            state[VK_CAPITAL] = 0x01
        # Підтримка AltGr (наприклад, правий Alt + Г = Ґ в українській розкладці)
        if ralt_down and ctrl_down:
            state[VK_CONTROL] = 0x80
            state[VK_MENU] = 0x80
            state[VK_RMENU] = 0x80

        buf = ctypes.create_unicode_buffer(8)
        hkl = wintypes.HKL(self.get_active_hkl())
        # Прапорець 0x04 = не змінювати стан dead-key у системі (Win 10 1607+)
        res = user32.ToUnicodeEx(vk, scan, state, buf, len(buf), 0x04, hkl)
        if res > 0:
            text = buf.value[:res]
            # Відсіюємо керуючі символи
            return "".join(ch for ch in text if ch >= " " and ch != "\x7f")
        return ""

    @staticmethod
    def _matches_hotkey(
        spec: Optional[HotkeySpec],
        mods: Set[str],
        vk: int,
        is_modifier_vk: bool,
        is_repeat: bool,
        double_hit_vk: Optional[int],
    ) -> bool:
        if spec is None:
            return False
        if spec.is_double:
            return double_hit_vk is not None and spec.vk == double_hit_vk
        if spec.vk is None:
            return is_modifier_vk and not is_repeat and mods == spec.mods
        return (not is_modifier_vk) and mods == spec.mods and vk == spec.vk

    def _low_level_keyboard_proc(self, nCode: int, wParam: int, lParam: int) -> int:
        if nCode >= 0 and wParam in (WM_KEYDOWN, WM_SYSKEYDOWN):
            try:
                kb = ctypes.cast(lParam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                vk = int(kb.vkCode)
                scan = int(kb.scanCode)

                is_repeat = vk in self._pressed_vks
                self._pressed_vks.add(vk)

                mod_name = _vk_to_mod_name(vk)
                is_modifier_vk = mod_name is not None

                mods = self._get_pressed_modifiers()
                if mod_name:
                    mods.add(mod_name)

                # 1. Якщо активний діалог запису нового хоткею в налаштуваннях
                if self._recording_hotkey:
                    if self._recording_double:
                        if not is_repeat:
                            cap_vk = _canonical_mod_vk(vk) if is_modifier_vk else vk
                            hk_str = format_hotkey(set(), cap_vk, is_double=True)
                            self.stop_recording_hotkey()
                            self.hotkey_recorded.emit(hk_str)
                        return 1 if not is_modifier_vk else user32.CallNextHookEx(self._hook_handle, nCode, wParam, lParam)

                    if is_modifier_vk:
                        self._rec_mod_peak |= mods
                        return user32.CallNextHookEx(self._hook_handle, nCode, wParam, lParam)

                    self._rec_saw_non_mod = True
                    hk_str = format_hotkey(mods, vk)
                    self.stop_recording_hotkey()
                    self.hotkey_recorded.emit(hk_str)
                    return 1

                # Визначаємо подвійне натискання (2x, напр. 2x Shift або 2x Space) без проміжних інших клавіш
                double_hit_vk: Optional[int] = None
                if not is_repeat:
                    tap_vk = _canonical_mod_vk(vk) if is_modifier_vk else vk
                    now = time.monotonic()
                    clean_tap = (len(mods) <= 1) if is_modifier_vk else (not mods)
                    if clean_tap and self._last_tap_vk == tap_vk and (now - self._last_tap_ts) <= 0.38:
                        double_hit_vk = tap_vk
                        self._last_tap_vk = None
                    elif clean_tap:
                        self._last_tap_vk = tap_vk
                        self._last_tap_ts = now
                    else:
                        self._last_tap_vk = None

                # 2. Глобальні хоткеї (зупинка озвучки, повтор останньої фрази)
                if self._matches_hotkey(self._hk_stop, mods, vk, is_modifier_vk, is_repeat, double_hit_vk):
                    self.stop_tts_triggered.emit()
                    return user32.CallNextHookEx(self._hook_handle, nCode, wParam, lParam) if is_modifier_vk else 1

                if self._matches_hotkey(self._hk_repeat, mods, vk, is_modifier_vk, is_repeat, double_hit_vk):
                    self.repeat_last_triggered.emit()
                    return user32.CallNextHookEx(self._hook_handle, nCode, wParam, lParam) if is_modifier_vk else 1

                if self._matches_hotkey(self._hk_toggle_game_mode, mods, vk, is_modifier_vk, is_repeat, double_hit_vk):
                    self.toggle_game_mode_triggered.emit()
                    return user32.CallNextHookEx(self._hook_handle, nCode, wParam, lParam) if is_modifier_vk else 1

                if self._chat_tts_enabled:
                    if self._matches_hotkey(self._hk_stop_chat, mods, vk, is_modifier_vk, is_repeat, double_hit_vk):
                        self.stop_chat_tts_triggered.emit()
                        return user32.CallNextHookEx(self._hook_handle, nCode, wParam, lParam) if is_modifier_vk else 1

                    if self._matches_hotkey(self._hk_toggle_chat, mods, vk, is_modifier_vk, is_repeat, double_hit_vk):
                        self.toggle_chat_tts_triggered.emit()
                        return user32.CallNextHookEx(self._hook_handle, nCode, wParam, lParam) if is_modifier_vk else 1

                # 3a. Режим постійного спілкування (діалогу)
                if self._matches_hotkey(self._hk_conversation, mods, vk, is_modifier_vk, is_repeat, double_hit_vk):
                    self.set_typing_mode(not self._is_typing, continuous=True)
                    return user32.CallNextHookEx(self._hook_handle, nCode, wParam, lParam) if is_modifier_vk else 1

                # 3b. Старт / перемикання режиму набору
                if self._matches_hotkey(self._hk_toggle, mods, vk, is_modifier_vk, is_repeat, double_hit_vk):
                    self.set_typing_mode(not self._is_typing, continuous=self._config_continuous)
                    return user32.CallNextHookEx(self._hook_handle, nCode, wParam, lParam) if is_modifier_vk else 1

                # 4. Швидкі фрази за хоткеями
                if not self._is_typing or mods or double_hit_vk is not None:
                    if double_hit_vk is not None:
                        # Друге натискання перемагає одиночне: скасовуємо фразу,
                        # яку відклали після першого натискання.
                        self._cancel_pending_quick_singles(double_hit_vk)
                    for q_spec, q_text in self._quick_hotkeys:
                        if double_hit_vk is not None and not q_spec.is_double:
                            continue
                        if self._matches_hotkey(q_spec, mods, vk, is_modifier_vk, is_repeat, double_hit_vk):
                            paired_double = False
                            if not q_spec.is_double:
                                tap_vk = _canonical_mod_vk(vk) if is_modifier_vk else vk
                                for double_spec, _ in self._quick_hotkeys:
                                    if not double_spec.is_double or double_spec.vk != tap_vk:
                                        continue
                                    if is_modifier_vk:
                                        paired_double = q_spec.vk is None and q_spec.mods == frozenset({mod_name})
                                    else:
                                        paired_double = q_spec.vk == vk and not q_spec.mods and not mods
                                    if paired_double:
                                        break
                            if paired_double:
                                self._defer_quick_single(tap_vk, q_text)
                            else:
                                self.quick_phrase_triggered.emit(q_text)
                            return user32.CallNextHookEx(self._hook_handle, nCode, wParam, lParam) if is_modifier_vk else 1

                # 5. Обробка клавіш у режимі сліпого набору без втрати фокусу (hook mode)
                if self._is_typing and not self.uses_popup_window:
                    # Відправка на озвучку (за замовчуванням Enter)
                    if self._matches_hotkey(self._hk_send, mods, vk, is_modifier_vk, is_repeat, double_hit_vk):
                        final_text = self._buffer.strip()
                        self._buffer = ""
                        self._clear_saved_draft()
                        lang = self.get_current_lang_code()
                        if self.is_continuous_active:
                            self.typing_text_changed.emit("", lang)
                            if final_text:
                                self.send_triggered.emit(final_text)
                        else:
                            self._is_typing = False
                            self._session_continuous = False
                            self._sync_game_guard()
                            self.typing_mode_changed.emit(False, lang)
                            if final_text:
                                self.send_triggered.emit(final_text)
                        return user32.CallNextHookEx(self._hook_handle, nCode, wParam, lParam) if is_modifier_vk else 1

                    # Скасування набору (за замовчуванням Esc)
                    if self._matches_hotkey(self._hk_cancel, mods, vk, is_modifier_vk, is_repeat, double_hit_vk):
                        self.set_typing_mode(False)
                        return user32.CallNextHookEx(self._hook_handle, nCode, wParam, lParam) if is_modifier_vk else 1

                if not is_modifier_vk:
                    # 6. Подальша обробка символів і редагування у режимі сліпого набору
                    if self._is_typing and not self.uses_popup_window:

                        # Backspace (та Ctrl+Backspace для видалення слова)
                        if vk == VK_BACK:
                            if "ctrl" in mods:
                                stripped = self._buffer.rstrip()
                                last_space = stripped.rfind(" ")
                                self._buffer = stripped[: last_space + 1] if last_space != -1 else ""
                            else:
                                self._buffer = self._buffer[:-1]
                            self.typing_text_changed.emit(self._buffer, self.get_current_lang_code())
                            return 1

                        # Delete або Ctrl+U — очистити весь рядок
                        if vk == VK_DELETE or (vk == ord("U") and "ctrl" in mods):
                            self._buffer = ""
                            self.typing_text_changed.emit(self._buffer, self.get_current_lang_code())
                            return 1

                        # Ctrl+V — вставка з буфера обміну
                        if vk == ord("V") and "ctrl" in mods and not ("alt" in mods):
                            try:
                                cb = QGuiApplication.clipboard()
                                if cb and cb.text():
                                    clean_paste = cb.text().replace("\r", " ").replace("\n", " ")
                                    self._buffer += clean_paste
                                    self.typing_text_changed.emit(self._buffer, self.get_current_lang_code())
                            except Exception:
                                pass
                            return 1

                        # Системні комбінації Windows (перемикання робочих столів Ctrl+Win+Left/Right, Win+Tab, Alt+Tab, Alt+Esc, Ctrl+Shift+Esc, Alt+F4 тощо)
                        # завжди пропускаємо в систему, не блокуючи їх під час набору
                        if "win" in mods:
                            return user32.CallNextHookEx(self._hook_handle, nCode, wParam, lParam)
                        if "alt" in mods and vk in (VK_TAB, VK_ESCAPE, 0x73):
                            return user32.CallNextHookEx(self._hook_handle, nCode, wParam, lParam)
                        if "ctrl" in mods and "shift" in mods and vk == VK_ESCAPE:
                            return user32.CallNextHookEx(self._hook_handle, nCode, wParam, lParam)

                        # Ігноруємо інші комбінації з Ctrl/Alt (крім AltGr), але блокуємо їх для гри під час набору
                        ralt_down = bool(user32.GetAsyncKeyState(VK_RMENU) & 0x8000)
                        is_altgr = ralt_down and ("ctrl" in mods) and ("alt" in mods)
                        if ("ctrl" in mods or "alt" in mods) and not is_altgr:
                            return 1

                        # Перетворення клавіші у символ (UA / RU / EN)
                        ch = self._vk_to_unicode(vk, scan)
                        if ch:
                            self._buffer += ch
                            self.typing_text_changed.emit(self._buffer, self.get_current_lang_code())
                            return 1

                        # Блокуємо навігаційні клавіші під час набору, щоб не рухати персонажа в грі
                        if vk in (VK_SPACE, VK_TAB, VK_LEFT, VK_RIGHT, VK_UP, VK_DOWN, VK_HOME, VK_END, VK_PRIOR, VK_NEXT):
                            return 1

            except Exception as exc:
                print(f"[KeyboardHook] Error in hook proc: {exc}")
        elif nCode >= 0 and wParam in (WM_KEYUP, WM_SYSKEYUP):
            try:
                kb = ctypes.cast(lParam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                vk = int(kb.vkCode)
                self._pressed_vks.discard(vk)

                # Завершення запису однієї або кількох клавіш-модифікаторів при відпусканні.
                if self._recording_hotkey and not self._recording_double and not self._rec_saw_non_mod:
                    mod_name = _vk_to_mod_name(vk)
                    if mod_name is not None:
                        remaining = self._get_pressed_modifiers()
                        remaining.discard(mod_name)
                        if not remaining:
                            if self._rec_mod_peak:
                                hk_str = format_hotkey(self._rec_mod_peak, None)
                                self.stop_recording_hotkey()
                                self.hotkey_recorded.emit(hk_str)
                                return user32.CallNextHookEx(self._hook_handle, nCode, wParam, lParam)
                            self._rec_mod_peak.clear()

                if self._is_typing and not self.uses_popup_window:
                    if vk in (
                        VK_SHIFT, VK_LSHIFT, VK_RSHIFT,
                        VK_CONTROL, VK_LCONTROL, VK_RCONTROL,
                        VK_MENU, VK_LMENU, VK_RMENU,
                        VK_LWIN, VK_RWIN,
                        VK_SPACE,
                    ):
                        self._layout_refresh_requested.emit()
            except Exception:
                pass

        return user32.CallNextHookEx(self._hook_handle, nCode, wParam, lParam)

    def _raw_wnd_proc(self, hwnd: int, msg: int, wparam: int, lparam: int) -> int:
        if msg == _WM_GAME_GUARD_ARM:
            self._set_raw_mouse_registered(hwnd, True)
            return 0
        if msg == _WM_GAME_GUARD_DISARM:
            self._set_raw_mouse_registered(hwnd, False)
            return 0
        if msg == WM_TIMER and int(wparam) == _GAME_VERIFY_TIMER_ID:
            try:
                user32.KillTimer(wintypes.HWND(hwnd), _GAME_VERIFY_TIMER_ID)
            except Exception:
                pass
            self._verify_weak_game_trigger()
            return 0
        if msg == WM_INPUT and self._game_guard_armed:
            dx, dy, btn_flags, is_abs = _read_raw_mouse_input(lparam)
            if (dx or dy or btn_flags) and self._game_guard.feed(dx, dy, btn_flags, is_absolute=is_abs):
                if self._game_guard_weak_mode:
                    if not self._game_guard_pending_verify:
                        self._game_guard_pending_verify = True
                        try:
                            user32.SetTimer(wintypes.HWND(hwnd), _GAME_VERIFY_TIMER_ID, 30, None)
                        except Exception:
                            self._verify_weak_game_trigger()
                else:
                    self._emergency_cancel_typing()
            return user32.DefWindowProcW(wintypes.HWND(hwnd), msg, wparam, lparam)
        return user32.DefWindowProcW(wintypes.HWND(hwnd), msg, wparam, lparam)

    @staticmethod
    def _set_raw_mouse_registered(hwnd: int, enabled: bool) -> bool:
        if not hwnd:
            return False
        if not enabled:
            try:
                user32.KillTimer(wintypes.HWND(hwnd), _GAME_VERIFY_TIMER_ID)
            except Exception:
                pass
        try:
            rid = _RAWINPUTDEVICE(
                usUsagePage=0x01,
                usUsage=0x02,
                dwFlags=_RIDEV_INPUTSINK if enabled else _RIDEV_REMOVE,
                hwndTarget=wintypes.HWND(hwnd) if enabled else None,
            )
            ok = bool(user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(_RAWINPUTDEVICE)))
            if not ok:
                err = kernel32.GetLastError()
                print(f"[KeyboardHook] RegisterRawInputDevices({enabled}) failed: err={err}")
            return ok
        except Exception as exc:
            print(f"[KeyboardHook] _set_raw_mouse_registered error: {exc}")
            return False

    def _create_raw_message_window(self) -> int:
        try:
            _ensure_win32_game_prototypes()
            hinst = kernel32.GetModuleHandleW(None)
            cls_name = f"SilentVoiceRawMouseGuard_{id(self)}"
            wndproc_type = ctypes.WINFUNCTYPE(
                ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
            )
            self._raw_wndproc_ref = wndproc_type(self._raw_wnd_proc)
            wc = _WNDCLASSEXW()
            wc.cbSize = ctypes.sizeof(_WNDCLASSEXW)
            wc.lpfnWndProc = ctypes.cast(self._raw_wndproc_ref, ctypes.c_void_p)
            wc.hInstance = hinst
            wc.lpszClassName = cls_name
            atom = user32.RegisterClassExW(ctypes.byref(wc))
            if atom:
                self._raw_wndclass_atom = int(atom)
                hwnd = user32.CreateWindowExW(0, cls_name, cls_name, 0, 0, 0, 0, 0, _HWND_MESSAGE, None, hinst, None)
                if not hwnd:
                    err = kernel32.GetLastError()
                    print(f"[KeyboardHook] CreateWindowExW failed: err={err}")
                return int(hwnd or 0)
            err = kernel32.GetLastError()
            print(f"[KeyboardHook] RegisterClassExW failed: err={err}")
        except Exception as exc:
            print(f"[KeyboardHook] _create_raw_message_window error: {exc}")
        return 0

    def _destroy_raw_message_window(self) -> None:
        hwnd = self._raw_hwnd
        self._raw_hwnd = 0
        if hwnd:
            self._set_raw_mouse_registered(hwnd, False)
            try:
                user32.DestroyWindow(wintypes.HWND(hwnd))
            except Exception:
                pass
        if self._raw_wndclass_atom:
            try:
                hinst = kernel32.GetModuleHandleW(None)
                user32.UnregisterClassW(f"SilentVoiceRawMouseGuard_{id(self)}", hinst)
            except Exception:
                pass
            self._raw_wndclass_atom = 0

    def _hook_loop(self) -> None:
        self._thread_id = kernel32.GetCurrentThreadId()
        self._hook_handle = user32.SetWindowsHookExW(
            WH_KEYBOARD_LL,
            self._proc_ref,
            None,
            0,
        )
        if not self._hook_handle:
            err = kernel32.GetLastError()
            print(f"[KeyboardHook] Не вдалося встановити WH_KEYBOARD_LL (код {err})")
            return

        self._raw_hwnd = self._create_raw_message_window()
        if self._game_guard_armed and self._raw_hwnd:
            self._set_raw_mouse_registered(self._raw_hwnd, True)

        msg = wintypes.MSG()
        while self._running:
            ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if ret <= 0:
                break
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

        self._destroy_raw_message_window()
        if self._hook_handle:
            user32.UnhookWindowsHookEx(self._hook_handle)
            self._hook_handle = None


# ──────────────────────────────────────────────────────────────────────
# Системні утиліти Windows (Автозапуск та права Адміністратора)
# ──────────────────────────────────────────────────────────────────────

RUN_REG_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"
APP_REG_NAME = "SilentVoiceTTS"


def is_autostart_enabled() -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_REG_PATH, 0, winreg.KEY_READ) as key:
            val, _ = winreg.QueryValueEx(key, APP_REG_NAME)
            return bool(val)
    except Exception:
        return False


def set_windows_autostart(enabled: bool) -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_REG_PATH, 0, winreg.KEY_SET_VALUE) as key:
            if enabled:
                if getattr(sys, "frozen", False):
                    cmd = f'"{sys.executable}" --minimized'
                else:
                    import os
                    main_py = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "main.py"))
                    pythonw = sys.executable.replace("python.exe", "pythonw.exe")
                    if not os.path.exists(pythonw):
                        pythonw = sys.executable
                    cmd = f'"{pythonw}" "{main_py}" --minimized'
                winreg.SetValueEx(key, APP_REG_NAME, 0, winreg.REG_SZ, cmd)
            else:
                try:
                    winreg.DeleteValue(key, APP_REG_NAME)
                except FileNotFoundError:
                    pass
        return True
    except Exception as exc:
        print(f"[WinAPI] set_windows_autostart error: {exc}")
        return False


def is_running_as_admin() -> bool:
    try:
        return bool(shell32.IsUserAnAdmin())
    except Exception:
        return False


def restart_as_admin() -> bool:
    """Перезапускає поточну програму з правами адміністратора (UAC)."""
    try:
        import os
        if getattr(sys, "frozen", False):
            exe = sys.executable
            params = " ".join(f'"{a}"' for a in sys.argv[1:])
        else:
            exe = sys.executable
            main_py = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "main.py"))
            params = f'"{main_py}" ' + " ".join(f'"{a}"' for a in sys.argv[1:])
        res = shell32.ShellExecuteW(None, "runas", exe, params, None, 1)
        return int(res) > 32
    except Exception as exc:
        print(f"[WinAPI] restart_as_admin error: {exc}")
        return False


class _DISPLAY_DEVICEW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("DeviceName", wintypes.WCHAR * 32),
        ("DeviceString", wintypes.WCHAR * 128),
        ("StateFlags", wintypes.DWORD),
        ("DeviceID", wintypes.WCHAR * 128),
        ("DeviceKey", wintypes.WCHAR * 128),
    ]


_SOFTWARE_GPU_MARKERS = (
    "microsoft basic",
    "basic display",
    "basic render",
    "remote display",
    "virtual",
    "mirror",
    "rdp",
    "vnc",
    "citrix",
    "parsec",
    "spacedesk",
    "luminon",
)
_DISCRETE_AMD_RE = re.compile(
    r"\b(?:rx\s*\d{3,4}|r9\b|radeon\s+pro\b|radeon\s+vii\b|vega\s*(?:56|64|frontier)|firepro\b|wx\s*\d{4}|w[567]\d{3}|hd\s*[78]\d{3})\b",
    re.IGNORECASE,
)
_DISCRETE_INTEL_ARC_RE = re.compile(r"\barc\b.*\b(?:[ab]\d{3}|pro\s*[ab]\d{2,3})\b", re.IGNORECASE)
_GPU_DETECT_CACHE: Optional[bool] = None
_NVIDIA_GPU_DETECT_CACHE: Optional[bool] = None
_GPU_VRAM_BYTES_CACHE: Optional[int] = None
_GPU_VRAM_DETECTED: bool = False
_KILL_ON_CLOSE_JOB_HANDLE: Optional[int] = None
_KILL_ON_CLOSE_JOB_LOCK = threading.Lock()


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def assign_process_to_kill_on_close_job(proc: object) -> bool:
    """
    Прив'язує дочірній процес (subprocess.Popen або PID) до Win32 Job Object із прапором
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE (0x2000), щоб ядро Windows гарантовано завершувало
    дочірні сервери (llama-server.exe, llama-tts.exe, python.exe) при виході з SilentVoice.
    """
    global _KILL_ON_CLOSE_JOB_HANDLE
    if sys.platform != "win32" or proc is None:
        return False
    try:
        with _KILL_ON_CLOSE_JOB_LOCK:
            if not _KILL_ON_CLOSE_JOB_HANDLE:
                create_job = kernel32.CreateJobObjectW
                create_job.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
                create_job.restype = wintypes.HANDLE
                set_info = kernel32.SetInformationJobObject
                set_info.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
                set_info.restype = wintypes.BOOL
                job = create_job(None, None)
                if not job:
                    return False
                info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
                info.BasicLimitInformation.LimitFlags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                if not set_info(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
                    kernel32.CloseHandle(job)
                    return False
                _KILL_ON_CLOSE_JOB_HANDLE = int(job)

            job_handle = wintypes.HANDLE(_KILL_ON_CLOSE_JOB_HANDLE)

        assign_fn = kernel32.AssignProcessToJobObject
        assign_fn.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        assign_fn.restype = wintypes.BOOL

        raw_handle = getattr(proc, "_handle", None)
        if raw_handle:
            return bool(assign_fn(job_handle, wintypes.HANDLE(int(raw_handle))))

        pid = int(getattr(proc, "pid", 0) or (proc if isinstance(proc, int) else 0))
        if pid <= 0:
            return False
        open_proc = kernel32.OpenProcess
        open_proc.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        open_proc.restype = wintypes.HANDLE
        # PROCESS_SET_QUOTA (0x0100) | PROCESS_TERMINATE (0x0001)
        h_proc = open_proc(0x0100 | 0x0001, False, pid)
        if not h_proc:
            return False
        try:
            return bool(assign_fn(job_handle, h_proc))
        finally:
            kernel32.CloseHandle(h_proc)
    except Exception:
        return False


class _DXGI_LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class _DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", wintypes.WCHAR * 128),
        ("VendorId", wintypes.UINT),
        ("DeviceId", wintypes.UINT),
        ("SubSysId", wintypes.UINT),
        ("Revision", wintypes.UINT),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _DXGI_LUID),
        ("Flags", wintypes.UINT),
    ]


class _DXGI_GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]


def is_real_gpu_adapter(device_string: str, device_id: str = "", flags: int = 0) -> bool:
    """Повертає True лише для дискретної відеокарти (відсіює вбудовану графіку CPU та віртуальні драйвери)."""
    if flags & 0x2:  # DXGI_ADAPTER_FLAG_SOFTWARE
        return False
    name = str(device_string or "").strip().lower()
    dev_id = str(device_id or "").strip().lower()
    if not name:
        return False
    if any(tok in name for tok in _SOFTWARE_GPU_MARKERS):
        return False
    if "ven_1414" in dev_id:  # Microsoft Basic Render Driver
        return False

    # NVIDIA discrete GPUs (GeForce GTX/RTX, Quadro, TITAN, etc.)
    if "ven_10de" in dev_id or any(tok in name for tok in ("nvidia", "geforce", "gtx", "rtx", "quadro", "titan")):
        return True

    # Discrete AMD GPUs only (exclude integrated APU "AMD Radeon(TM) Graphics", Vega 8, 680M/780M, etc.)
    if "ven_1002" in dev_id or "radeon" in name or "amd" in name:
        return bool(_DISCRETE_AMD_RE.search(name))

    # Discrete Intel Arc GPUs only (exclude UHD/Iris/generic Arc Graphics iGPU)
    if "ven_8086" in dev_id or "intel" in name or "arc" in name:
        return bool(_DISCRETE_INTEL_ARC_RE.search(name))

    return False


def is_nvidia_gpu_adapter(device_string: str, device_id: str = "", flags: int = 0) -> bool:
    """Повертає True лише для дискретної відеокарти NVIDIA (GeForce GTX/RTX, Quadro, TITAN тощо)."""
    if flags & 0x2:  # DXGI_ADAPTER_FLAG_SOFTWARE
        return False
    name = str(device_string or "").strip().lower()
    dev_id = str(device_id or "").strip().lower()
    if not name:
        return False
    if any(tok in name for tok in _SOFTWARE_GPU_MARKERS):
        return False
    if "ven_1414" in dev_id:
        return False
    return "ven_10de" in dev_id or any(
        tok in name for tok in ("nvidia", "geforce", "gtx", "rtx", "quadro", "titan")
    )


def _detect_dxgi_discrete_gpu() -> Optional[bool]:
    """Перевіряє всі GPU-адаптери через DXGI (включно з гібридними ноутбуками Optimus)."""
    try:
        dxgi = ctypes.windll.dxgi
    except Exception:
        return None
    factory = ctypes.c_void_p()
    iid = _DXGI_GUID(
        0x770AAE78,
        0xF26F,
        0x4DBA,
        (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87),
    )
    if dxgi.CreateDXGIFactory1(ctypes.byref(iid), ctypes.byref(factory)) != 0 or not factory.value:
        return None
    try:
        vt = ctypes.cast(factory, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        enum_adapters1 = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))(vt[12])
        release_fn = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(vt[2])
        idx = 0
        saw_any = False
        found_discrete = False
        while True:
            adapter = ctypes.c_void_p()
            if enum_adapters1(factory, idx, ctypes.byref(adapter)) != 0 or not adapter.value:
                break
            saw_any = True
            avt = ctypes.cast(adapter, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            get_desc1 = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(_DXGI_ADAPTER_DESC1))(avt[10])
            arelease = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(avt[2])
            try:
                desc = _DXGI_ADAPTER_DESC1()
                if get_desc1(adapter, ctypes.byref(desc)) == 0:
                    vid_tag = f"pci\\ven_{desc.VendorId:04x}&dev_{desc.DeviceId:04x}"
                    if is_real_gpu_adapter(desc.Description, vid_tag, flags=int(desc.Flags)):
                        found_discrete = True
                        break
            finally:
                arelease(adapter)
            idx += 1
        return found_discrete if saw_any else None
    except Exception:
        return None
    finally:
        try:
            vt = ctypes.cast(factory, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(vt[2])(factory)
        except Exception:
            pass


def _read_test_vram_override_gb() -> Optional[float]:
    raw = str(os.environ.get("SILENTVOICE_TEST_VRAM_GB", "") or "").strip()
    if not raw:
        return None
    try:
        val = float(raw.replace(",", "."))
        return val if val > 0 else None
    except ValueError:
        return None


def has_hardware_gpu(force_refresh: bool = False) -> bool:
    """Визначає наявність дискретної відеокарти через DXGI (з fallback на EnumDisplayDevicesW)."""
    global _GPU_DETECT_CACHE
    if _read_test_vram_override_gb() is not None:
        return True
    if _GPU_DETECT_CACHE is not None and not force_refresh:
        return _GPU_DETECT_CACHE
    if sys.platform != "win32":
        _GPU_DETECT_CACHE = True
        return True
    try:
        dxgi_res = _detect_dxgi_discrete_gpu()
        if dxgi_res is not None:
            _GPU_DETECT_CACHE = dxgi_res
            return _GPU_DETECT_CACHE

        idx = 0
        saw_any = False
        found_hw = False
        while True:
            dd = _DISPLAY_DEVICEW()
            dd.cb = ctypes.sizeof(dd)
            if not user32.EnumDisplayDevicesW(None, idx, ctypes.byref(dd), 0):
                break
            saw_any = True
            if is_real_gpu_adapter(dd.DeviceString, dd.DeviceID):
                found_hw = True
                break
            idx += 1
        _GPU_DETECT_CACHE = found_hw if saw_any else False
    except Exception:
        _GPU_DETECT_CACHE = True
    return bool(_GPU_DETECT_CACHE)


def _detect_dxgi_nvidia_gpu() -> Optional[bool]:
    """Перевіряє наявність дискретної відеокарти NVIDIA серед усіх адаптерів DXGI."""
    try:
        dxgi = ctypes.windll.dxgi
    except Exception:
        return None
    factory = ctypes.c_void_p()
    iid = _DXGI_GUID(
        0x770AAE78,
        0xF26F,
        0x4DBA,
        (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87),
    )
    if dxgi.CreateDXGIFactory1(ctypes.byref(iid), ctypes.byref(factory)) != 0 or not factory.value:
        return None
    try:
        vt = ctypes.cast(factory, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        enum_adapters1 = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))(vt[12])
        idx = 0
        saw_any = False
        found_nvidia = False
        while True:
            adapter = ctypes.c_void_p()
            if enum_adapters1(factory, idx, ctypes.byref(adapter)) != 0 or not adapter.value:
                break
            saw_any = True
            avt = ctypes.cast(adapter, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            get_desc1 = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(_DXGI_ADAPTER_DESC1))(avt[10])
            arelease = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(avt[2])
            try:
                desc = _DXGI_ADAPTER_DESC1()
                if get_desc1(adapter, ctypes.byref(desc)) == 0:
                    vid_tag = f"pci\\ven_{desc.VendorId:04x}&dev_{desc.DeviceId:04x}"
                    if is_nvidia_gpu_adapter(desc.Description, vid_tag, flags=int(desc.Flags)):
                        found_nvidia = True
                        break
            finally:
                arelease(adapter)
            idx += 1
        return found_nvidia if saw_any else None
    except Exception:
        return None
    finally:
        try:
            vt = ctypes.cast(factory, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(vt[2])(factory)
        except Exception:
            pass


def has_nvidia_gpu(force_refresh: bool = False) -> bool:
    """Визначає наявність дискретної відеокарти NVIDIA через DXGI (з fallback на EnumDisplayDevicesW)."""
    global _NVIDIA_GPU_DETECT_CACHE
    if _NVIDIA_GPU_DETECT_CACHE is not None and not force_refresh:
        return _NVIDIA_GPU_DETECT_CACHE
    if sys.platform != "win32":
        _NVIDIA_GPU_DETECT_CACHE = False
        return False
    try:
        dxgi_res = _detect_dxgi_nvidia_gpu()
        if dxgi_res is not None:
            _NVIDIA_GPU_DETECT_CACHE = dxgi_res
            return _NVIDIA_GPU_DETECT_CACHE

        idx = 0
        saw_any = False
        found_nv = False
        while True:
            dd = _DISPLAY_DEVICEW()
            dd.cb = ctypes.sizeof(dd)
            if not user32.EnumDisplayDevicesW(None, idx, ctypes.byref(dd), 0):
                break
            saw_any = True
            if is_nvidia_gpu_adapter(dd.DeviceString, dd.DeviceID):
                found_nv = True
                break
            idx += 1
        _NVIDIA_GPU_DETECT_CACHE = found_nv if saw_any else False
    except Exception:
        _NVIDIA_GPU_DETECT_CACHE = False
    return bool(_NVIDIA_GPU_DETECT_CACHE)


def _detect_dxgi_discrete_gpu_vram_bytes() -> Optional[int]:
    """Повертає максимальний обсяг виділеної відеопам'яті (DedicatedVideoMemory, байти) серед дискретних GPU через DXGI."""
    try:
        dxgi = ctypes.windll.dxgi
    except Exception:
        return None
    factory = ctypes.c_void_p()
    iid = _DXGI_GUID(
        0x770AAE78,
        0xF26F,
        0x4DBA,
        (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87),
    )
    if dxgi.CreateDXGIFactory1(ctypes.byref(iid), ctypes.byref(factory)) != 0 or not factory.value:
        return None
    try:
        vt = ctypes.cast(factory, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        enum_adapters1 = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))(vt[12])
        idx = 0
        max_vram = 0
        while True:
            adapter = ctypes.c_void_p()
            if enum_adapters1(factory, idx, ctypes.byref(adapter)) != 0 or not adapter.value:
                break
            avt = ctypes.cast(adapter, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            get_desc1 = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(_DXGI_ADAPTER_DESC1))(avt[10])
            arelease = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(avt[2])
            try:
                desc = _DXGI_ADAPTER_DESC1()
                if get_desc1(adapter, ctypes.byref(desc)) == 0:
                    vid_tag = f"pci\\ven_{desc.VendorId:04x}&dev_{desc.DeviceId:04x}"
                    if is_real_gpu_adapter(desc.Description, vid_tag, flags=int(desc.Flags)):
                        vram = int(desc.DedicatedVideoMemory or 0)
                        if vram > max_vram:
                            max_vram = vram
            finally:
                arelease(adapter)
            idx += 1
        return max_vram if max_vram > 0 else None
    except Exception:
        return None
    finally:
        try:
            vt = ctypes.cast(factory, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(vt[2])(factory)
        except Exception:
            pass


def get_discrete_gpu_vram_gb(force_refresh: bool = False) -> Optional[float]:
    """Повертає обсяг відеопам'яті дискретної відеокарти в ГіБ (або None, якщо визначити не вдалося)."""
    global _GPU_VRAM_BYTES_CACHE, _GPU_VRAM_DETECTED
    override_gb = _read_test_vram_override_gb()
    if override_gb is not None:
        return override_gb
    if _GPU_VRAM_DETECTED and not force_refresh:
        return (_GPU_VRAM_BYTES_CACHE / float(1024 ** 3)) if _GPU_VRAM_BYTES_CACHE else None
    if sys.platform != "win32":
        _GPU_VRAM_BYTES_CACHE = None
        _GPU_VRAM_DETECTED = True
        return None
    vram_bytes = _detect_dxgi_discrete_gpu_vram_bytes()
    _GPU_VRAM_BYTES_CACHE = vram_bytes
    _GPU_VRAM_DETECTED = True
    return (vram_bytes / float(1024 ** 3)) if vram_bytes else None
