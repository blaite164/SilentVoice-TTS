"""
ui/overlay_window.py — Екранний оверлей поверх ігор та програм:
- За замовчуванням: «Без фону (тільки текст)» знизу по центру (або там, куди його перетягнув користувач)
- При виклику хоткею (поки текст ще порожній): показує лише ледь помітний міні-індикатор «• UA ▌», щоб людина бачила, що набір увімкнувся
- Під час вводу тексту: красивий текст із векторною обводкою QPainterPath та тінню без жодного фону (або з плашкою за бажанням)
- Вільне перетягування мишкою під час налаштування позиції + 100% click-through без втрати фокусу в іграх (WS_EX_TOPMOST, WS_EX_TRANSPARENT, WS_EX_NOACTIVATE)
- Повна невидимість для OBS / Discord Screen Share через SetWindowDisplayAffinity (WDA_EXCLUDEFROMCAPTURE)
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import math
import time
from typing import Optional

from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QGuiApplication,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QRegion,
    QTextCursor,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.config import config
from core.i18n import tr
from core.omnivoice_tags import (
    OMNIVOICE_REACTION_TAGS,
    OmniVoiceTagSpec,
    build_active_reaction_specs,
    build_all_configured_reaction_specs,
    extract_active_slash_query,
    format_tags_for_display,
    get_primary_cmd_name,
    get_primary_slash_cmd,
    get_spec_display_label,
    get_tag_description,
    match_reaction_tags,
)
from core.win_api import (
    apply_overlay_ex_styles,
    get_foreground_hwnd,
    restore_foreground_hwnd,
    set_window_capture_excluded,
)


def resolve_screen_by_name(screen_name: str = ""):
    """Повертає збережений монітор за назвою або основний екран за замовчуванням."""
    screens = QGuiApplication.screens() or []
    target_name = str(screen_name or "").strip()
    if target_name:
        for s in screens:
            if (s.name() or "").strip() == target_name:
                return s
    return QGuiApplication.primaryScreen() or (screens[0] if screens else None)


def find_screen_for_rect(rect: QRect):
    """Знаходить монітор, у межах якого лежить центр прямокутника (або найближчий за відстанню)."""
    center = rect.center()
    screen = QGuiApplication.screenAt(center)
    if screen is not None:
        return screen
    screens = QGuiApplication.screens() or []
    if not screens:
        return QGuiApplication.primaryScreen()
    best_screen = screens[0]
    best_dist = None
    for s in screens:
        g = s.geometry()
        cx = max(g.left(), min(center.x(), g.right()))
        cy = max(g.top(), min(center.y(), g.bottom()))
        dist = (center.x() - cx) ** 2 + (center.y() - cy) ** 2
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_screen = s
    return best_screen


def calc_screen_relative_offsets(rect: QRect) -> tuple[str, int, int]:
    """Повертає `(screen_name, x_offset, y_offset)` відносно монітора, на якому розташоване вікно."""
    screen = find_screen_for_rect(rect)
    if not screen:
        return "", 0, 85
    geom = screen.geometry()
    default_center_x = geom.x() + (geom.width() - rect.width()) // 2
    x_off = rect.x() - default_center_x
    y_off = max(5, (geom.y() + geom.height()) - (rect.y() + rect.height()))
    return (screen.name() or ""), int(x_off), int(y_off)


TAG_HINT_MODES: tuple[str, ...] = (
    "emoji_dock",
    "menu_mini",
    "menu_micro",
    "menu_v_compact",
    "emoji_grid",
)
INTERACTIVE_TAG_HINT_MODES: tuple[str, ...] = TAG_HINT_MODES
TAG_HINT_MODE_LABELS: dict[str, str] = {
    "emoji_dock": "Смайлики без фону",
    "menu_mini": "Міні-список (5 рядків)",
    "menu_micro": "Мікро-список (4 рядки)",
    "menu_v_compact": "Вузький список без описів",
    "emoji_grid": "Сітка смайликів (2×4)",
}


class OmniVoiceTagPickerPopup(QWidget):
    """
    Нефокусне інтерактивне спливаюче віконце вибору тегів OmniVoice (без втрати фокусу в іграх/програмах):
    - `menu_v`: вертикальний список зі скролом та описами (як командна палітра).
    - `menu_h`: горизонтальна стрічка плиток зі скролом коліщатком.
    - `emoji_wheel`: компактне віконце на 3 великих смайлики з круговою прокруткою коліщатком і кліком.
    """

    tag_chosen = Signal(str)
    selection_changed = Signal(int)
    visibility_hwnd_changed = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("SilentVoice Tag Picker")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._style: str = "menu_mini"
        self._matches: list[OmniVoiceTagSpec] = []
        self._ui_lang: str = "uk"
        self._selected_idx: int = 0
        self._scroll_offset: int = 0
        self._hover_idx: Optional[int] = None
        self._hover_arrow: str = ""
        self._mouse_inside: bool = False
        self._scroll_revealed: bool = False
        self._scroll_reveal_timer = QTimer(self)
        self._scroll_reveal_timer.setSingleShot(True)
        self._scroll_reveal_timer.timeout.connect(self._on_scroll_reveal_timeout)
        self._toast_text: str = ""
        self._item_rects: list[tuple[int, QRectF]] = []
        self._left_arrow_rect: QRectF = QRectF()
        self._right_arrow_rect: QRectF = QRectF()

    def _on_scroll_reveal_timeout(self) -> None:
        self._scroll_revealed = False
        if self.isVisible():
            self.update()

    def _is_scrollbar_visible(self) -> bool:
        return self._mouse_inside or self._scroll_revealed

    def nativeEvent(self, eventType, message):
        # Гарантуємо MA_NOACTIVATE (3) на WM_MOUSEACTIVATE (0x0021), щоб клік по смайлику не забирав фокус
        try:
            if eventType in (b"windows_generic_MSG", b"windows_dispatcher_MSG"):
                msg = ctypes.cast(int(message), ctypes.POINTER(wintypes.MSG)).contents
                if int(msg.message) == 0x0021:
                    return True, 3
        except Exception:
            pass
        return super().nativeEvent(eventType, message)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._apply_win32_flags()

    def _apply_win32_flags(self) -> None:
        hwnd = int(self.winId())
        apply_overlay_ex_styles(hwnd, click_through=False, no_activate=True)
        set_window_capture_excluded(hwnd, bool(config.get("hide_from_capture", False)))

    def update_capture_affinity(self) -> None:
        if self.isVisible():
            set_window_capture_excluded(int(self.winId()), bool(config.get("hide_from_capture", False)))

    def _header_height(self) -> float:
        return 22.0 if self._toast_text else 0.0

    def _compute_dimensions(self) -> tuple[int, int]:
        n = max(1, len(self._matches))
        hdr = int(self._header_height())
        if self._style == "menu_mini":
            vis = min(5, n)
            return 252, hdr + 3 + vis * 25
        if self._style == "menu_micro":
            vis = min(4, n)
            return 228, hdr + 3 + vis * 23
        if self._style == "menu_v_compact":
            vis = min(6, n)
            return 176, hdr + 8 + vis * 26
        if self._style == "emoji_grid":
            cols = min(4, n)
            rows = min(2, (n + 3) // 4)
            tile_w, tile_h, gap = 50, 44, 6
            w = max(210, 20 + cols * tile_w + max(0, cols - 1) * gap)
            h = hdr + 10 + rows * tile_h + max(0, rows - 1) * gap + 28
            return w, h
        # emoji_dock — чисті смайлики (або `[команда]`) без фонів
        vis_indices = self._visible_indices() or list(range(min(8, n)))
        font_family = str(config.get("overlay_font_family", "Segoe UI") or "Segoe UI")
        fm_txt = QFontMetrics(QFont(font_family, 11, QFont.Weight.Bold))
        total_slots = 0
        for idx in vis_indices:
            spec = self._matches[idx]
            if spec.emoji:
                total_slots += 38
            else:
                lbl = get_spec_display_label(spec, self._ui_lang)
                total_slots += max(44, int(fm_txt.horizontalAdvance(lbl)) + 16)
        w = max(150, 32 + total_slots + 12)
        return w, hdr + 66

    def _ensure_selected_visible(self) -> None:
        n = len(self._matches)
        if n <= 0:
            self._selected_idx = 0
            self._scroll_offset = 0
            return
        self._selected_idx %= n
        if self._style == "emoji_dock":
            max_off = max(0, n - 8)
            self._scroll_offset = max(0, min(max_off, self._scroll_offset))
            if self._selected_idx < self._scroll_offset:
                self._scroll_offset = self._selected_idx
            elif self._selected_idx >= self._scroll_offset + 8:
                self._scroll_offset = self._selected_idx - 7
            return
        if self._style == "emoji_grid":
            if n <= 8:
                self._scroll_offset = 0
            else:
                sel_row = self._selected_idx // 4
                max_row = max(0, ((n + 3) // 4) - 2)
                self._scroll_offset = max(0, min(max_row, self._scroll_offset))
                if sel_row < self._scroll_offset:
                    self._scroll_offset = sel_row
                elif sel_row >= self._scroll_offset + 2:
                    self._scroll_offset = sel_row - 1
            return
        if self._style == "emoji_wheel":
            if n <= 3:
                self._scroll_offset = 0
            else:
                vis_indices = [(self._scroll_offset + k) % n for k in range(3)]
                if self._selected_idx not in vis_indices:
                    self._scroll_offset = (self._selected_idx - 1) % n
            return
        vis = 5 if self._style == "menu_mini" else (6 if self._style == "menu_v_compact" else 4)
        max_off = max(0, n - vis)
        self._scroll_offset = max(0, min(max_off, self._scroll_offset))
        if self._selected_idx < self._scroll_offset:
            self._scroll_offset = self._selected_idx
        elif self._selected_idx >= self._scroll_offset + vis:
            self._scroll_offset = self._selected_idx - vis + 1

    def _visible_indices(self) -> list[int]:
        n = len(self._matches)
        if n <= 0:
            return []
        if self._style == "emoji_dock":
            start = max(0, min(max(0, n - 8), self._scroll_offset))
            return list(range(start, min(n, start + 8)))
        if self._style == "emoji_grid":
            start = self._scroll_offset * 4
            return list(range(start, min(n, start + 8)))
        if self._style == "emoji_wheel":
            if n <= 3:
                return list(range(n))
            return [(self._scroll_offset + k) % n for k in range(3)]
        vis = 5 if self._style == "menu_mini" else (6 if self._style == "menu_v_compact" else 4)
        start = max(0, min(max(0, n - vis), self._scroll_offset))
        return list(range(start, min(n, start + vis)))

    def _rebuild_hit_rects(self) -> None:
        self._item_rects = []
        self._left_arrow_rect = QRectF()
        self._right_arrow_rect = QRectF()
        vis_indices = self._visible_indices()
        if not vis_indices:
            return
        hdr = self._header_height()
        w = float(self.width())
        n = len(self._matches)

        if self._style in ("menu_mini", "menu_micro"):
            pad_top = hdr + 1.5
            row_h = 25.0 if self._style == "menu_mini" else 23.0
            for row_i, idx in enumerate(vis_indices):
                self._item_rects.append((idx, QRectF(1.0, pad_top + row_i * row_h, w - 2.0, row_h)))
            return

        if self._style == "menu_v_compact":
            pad_top = hdr + 4.0
            row_h = 26.0
            for row_i, idx in enumerate(vis_indices):
                self._item_rects.append((idx, QRectF(4.0, pad_top + row_i * row_h, w - 8.0, row_h)))
            return

        if self._style == "emoji_grid":
            tile_w, tile_h, gap = 50.0, 44.0, 6.0
            cols = min(4, len(vis_indices))
            grid_w = cols * tile_w + max(0, cols - 1) * gap
            start_x = (w - grid_w) / 2.0
            start_y = hdr + 8.0
            for pos_i, idx in enumerate(vis_indices):
                r, c = divmod(pos_i, 4)
                tx = start_x + c * (tile_w + gap)
                ty = start_y + r * (tile_h + gap)
                self._item_rects.append((idx, QRectF(tx, ty, tile_w, tile_h)))
            return

        # emoji_dock
        dock_y = hdr + 22.0
        dock_h = 42.0
        active_zoom_idx = self._hover_idx if self._hover_idx is not None else self._selected_idx
        font_family = str(config.get("overlay_font_family", "Segoe UI") or "Segoe UI")
        fm_txt = QFontMetrics(QFont(font_family, 11, QFont.Weight.Bold))
        widths: list[float] = []
        for idx in vis_indices:
            spec = self._matches[idx]
            is_act = idx == active_zoom_idx
            if spec.emoji:
                widths.append(46.0 if is_act else 36.0)
            else:
                lbl = get_spec_display_label(spec, self._ui_lang)
                base_w = max(44.0, float(fm_txt.horizontalAdvance(lbl)) + 14.0)
                widths.append(base_w + (8.0 if is_act else 0.0))
        total_slots_w = sum(widths)
        cur_x = (w - total_slots_w) / 2.0
        for idx, sw in zip(vis_indices, widths):
            self._item_rects.append((idx, QRectF(cur_x, dock_y, sw, dock_h)))
            cur_x += sw

    def sync_state(
        self,
        *,
        style: str,
        matches: list[OmniVoiceTagSpec],
        ui_lang: str,
        selected_idx: int,
        anchor_top_center: QPoint,
        anchor_bottom_y: int,
        toast_text: str = "",
    ) -> None:
        if not matches:
            self.hide_picker()
            return

        old_tags = [m.tag for m in self._matches]
        new_tags = [m.tag for m in matches]
        style_changed = style != self._style
        if style_changed or old_tags != new_tags:
            self._scroll_offset = 0
            self._hover_idx = None
            self._hover_arrow = ""

        self._style = style
        self._matches = list(matches)
        self._ui_lang = ui_lang
        self._selected_idx = max(0, int(selected_idx)) % len(self._matches)
        self._toast_text = str(toast_text or "")
        self._ensure_selected_visible()

        w, h = self._compute_dimensions()
        if self.width() != w or self.height() != h:
            self.resize(w, h)
        self._rebuild_hit_rects()

        screen = find_screen_for_rect(QRect(anchor_top_center.x() - 10, anchor_top_center.y() - 10, 20, 20))
        geom = screen.geometry() if screen else QRect(0, 0, 1920, 1080)
        px = anchor_top_center.x() - w // 2
        py = anchor_top_center.y() - h - 8
        if py < geom.y() + 6:
            py = anchor_bottom_y + 8
        px = max(geom.x() + 6, min(geom.x() + geom.width() - w - 6, px))
        py = max(geom.y() + 6, min(geom.y() + geom.height() - h - 6, py))
        if self.pos() != QPoint(px, py):
            self.move(px, py)

        if not self.isVisible():
            self.setWindowOpacity(0.0)
            self.show()
            self._apply_win32_flags()
            self.raise_()
            self.repaint()
            self.setWindowOpacity(1.0)
            self.visibility_hwnd_changed.emit(int(self.winId()))
        else:
            self._apply_win32_flags()
            self.raise_()
            self.update()
            self.visibility_hwnd_changed.emit(int(self.winId()))

    def hide_picker(self) -> None:
        self._hover_idx = None
        self._hover_arrow = ""
        self._mouse_inside = False
        self._scroll_revealed = False
        self._scroll_reveal_timer.stop()
        if self.isVisible():
            self.hide()
        self.visibility_hwnd_changed.emit(0)

    def _step_scroll(self, step: int) -> None:
        n = len(self._matches)
        if n <= 1:
            return
        if self._style == "emoji_dock":
            self._selected_idx = (self._selected_idx + step) % n
            self._ensure_selected_visible()
        elif self._style == "emoji_grid":
            self._selected_idx = (self._selected_idx + step) % n
            self._ensure_selected_visible()
        elif self._style == "emoji_wheel":
            if n > 3:
                self._scroll_offset = (self._scroll_offset + step) % n
            self._selected_idx = (self._selected_idx + step) % n
        else:
            vis = 5 if self._style == "menu_mini" else (6 if self._style == "menu_v_compact" else 4)
            max_off = max(0, n - vis)
            self._selected_idx = max(0, min(n - 1, self._selected_idx + step))
            if max_off > 0:
                self._scroll_offset = max(0, min(max_off, self._scroll_offset + step))
                if self._selected_idx < self._scroll_offset:
                    self._scroll_offset = self._selected_idx
                elif self._selected_idx >= self._scroll_offset + vis:
                    self._scroll_offset = self._selected_idx - vis + 1
        self._scroll_revealed = True
        self._scroll_reveal_timer.start(850)
        self._rebuild_hit_rects()
        self.selection_changed.emit(self._selected_idx)
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        dy = event.angleDelta().y()
        if dy == 0:
            dy = event.angleDelta().x()
        if dy != 0:
            self._step_scroll(-1 if dy > 0 else 1)
            event.accept()
            return
        super().wheelEvent(event)

    def enterEvent(self, event) -> None:
        if not self._mouse_inside:
            self._mouse_inside = True
            self.update()
        super().enterEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self._mouse_inside:
            self._mouse_inside = True
            self.update()
        pos = event.position()
        new_hover_arrow = ""
        if not self._left_arrow_rect.isEmpty() and self._left_arrow_rect.contains(pos):
            new_hover_arrow = "left"
        elif not self._right_arrow_rect.isEmpty() and self._right_arrow_rect.contains(pos):
            new_hover_arrow = "right"

        hit_idx: Optional[int] = None
        if not new_hover_arrow:
            for idx, rect in self._item_rects:
                if rect.contains(pos):
                    hit_idx = idx
                    break

        changed = False
        if new_hover_arrow != self._hover_arrow:
            self._hover_arrow = new_hover_arrow
            changed = True
        if hit_idx is not None and (hit_idx != self._hover_idx or hit_idx != self._selected_idx):
            self._hover_idx = hit_idx
            self._selected_idx = hit_idx
            if self._style == "emoji_dock":
                self._rebuild_hit_rects()
            self.selection_changed.emit(hit_idx)
            changed = True
        elif hit_idx is None and self._hover_idx is not None:
            self._hover_idx = None
            changed = True
        if changed:
            self.update()
        event.accept()

    def leaveEvent(self, event) -> None:
        had_state = self._mouse_inside or self._scroll_revealed or self._hover_idx is not None or bool(self._hover_arrow)
        self._mouse_inside = False
        self._scroll_revealed = False
        self._scroll_reveal_timer.stop()
        self._hover_idx = None
        self._hover_arrow = ""
        if had_state:
            self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            if not self._left_arrow_rect.isEmpty() and self._left_arrow_rect.contains(pos):
                self._step_scroll(-1)
                event.accept()
                return
            if not self._right_arrow_rect.isEmpty() and self._right_arrow_rect.contains(pos):
                self._step_scroll(1)
                event.accept()
                return
            for idx, rect in self._item_rects:
                if rect.contains(pos) and 0 <= idx < len(self._matches):
                    self.tag_chosen.emit(self._matches[idx].tag)
                    event.accept()
                    return
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:
        if not self._matches:
            return
        self._rebuild_hit_rects()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        font_family = str(config.get("overlay_font_family", "Segoe UI") or "Segoe UI")
        border_hex = str(config.get("overlay_border_color", "#7C5CFF") or "#7C5CFF")
        accent = QColor(border_hex)
        w = float(self.width())
        h = float(self.height())
        hdr = self._header_height()
        n = len(self._matches)

        if self._style == "emoji_dock":
            self._paint_emoji_dock(painter, font_family, accent, w, h, hdr, n)
            painter.end()
            return

        outer = QRectF(1.0, 1.0, w - 2.0, h - 2.0)
        is_minimal_list = self._style in ("menu_mini", "menu_micro", "menu_v_compact")
        radius = 10.0 if self._style in ("menu_mini", "menu_micro") else (
            15.0 if self._style in ("emoji_wheel", "emoji_grid") else (13.0 if self._style == "menu_h" else 11.0)
        )
        bg_col = QColor(30, 31, 34, 244) if is_minimal_list else QColor(22, 24, 35, 238)
        brd_col = (
            QColor(255, 255, 255, 24)
            if is_minimal_list
            else QColor(accent.red(), accent.green(), accent.blue(), 110)
        )
        painter.setBrush(bg_col)
        painter.setPen(QPen(brd_col, 1.0))
        painter.drawRoundedRect(outer, radius, radius)

        if self._toast_text:
            hdr_font = QFont(font_family, 8 if is_minimal_list else 9, QFont.Weight.DemiBold)
            painter.setFont(hdr_font)
            painter.setPen(QColor(196, 181, 253, 235))
            painter.drawText(QRectF(8.0, 2.0, w - 16.0, hdr - 2.0), Qt.AlignmentFlag.AlignCenter, self._toast_text)
            painter.setPen(QPen(QColor(255, 255, 255, 16), 1.0))
            painter.drawLine(QPointF(8.0, hdr), QPointF(w - 8.0, hdr))

        if self._style in ("menu_mini", "menu_micro"):
            self._paint_menu_v(painter, font_family, accent, w, h, hdr, n)
        elif self._style == "menu_v_compact":
            self._paint_menu_v_compact(painter, font_family, accent, w, h, hdr, n)
        elif self._style == "emoji_grid":
            self._paint_emoji_grid(painter, font_family, accent, w, h, hdr, n)
        else:
            self._paint_emoji_wheel(painter, font_family, accent, w, h, hdr, n)

        painter.end()

    def _paint_menu_v_compact(
        self,
        painter: QPainter,
        font_family: str,
        accent: QColor,
        w: float,
        h: float,
        hdr: float,
        n: int,
    ) -> None:
        emoji_font = QFont(font_family, 12)
        cmd_font = QFont(font_family, 10, QFont.Weight.DemiBold)
        tab_font = QFont(font_family, 8, QFont.Weight.DemiBold)

        for idx, rect in self._item_rects:
            spec = self._matches[idx]
            is_sel = idx == self._selected_idx
            if is_sel:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(accent.red(), accent.green(), accent.blue(), 66))
                painter.drawRoundedRect(rect, 7.0, 7.0)

            painter.setFont(emoji_font)
            painter.setPen(QColor(255, 255, 255) if spec.emoji else QColor(148, 163, 184))
            painter.drawText(QRectF(rect.left() + 9.0, rect.top(), 24.0, rect.height()), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, spec.emoji or "♪")

            cmd_s = get_primary_slash_cmd(spec, self._ui_lang)
            painter.setFont(cmd_font)
            painter.setPen(QColor(255, 255, 255) if is_sel else QColor(215, 223, 236))
            painter.drawText(QRectF(rect.left() + 36.0, rect.top(), rect.width() - 68.0, rect.height()), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, cmd_s)

            if is_sel:
                painter.setFont(tab_font)
                painter.setPen(QColor(196, 181, 253, 210))
                painter.drawText(QRectF(rect.right() - 32.0, rect.top(), 26.0, rect.height()), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, "Tab")

        if n > 6 and self._is_scrollbar_visible():
            track_x = w - 6.8
            track_y = hdr + 6.0
            track_h = max(20.0, h - track_y - 6.0)
            thumb_h = max(18.0, track_h * (6.0 / float(n)))
            max_off = max(1, n - 6)
            frac = float(self._scroll_offset) / float(max_off)
            thumb_y = track_y + frac * (track_h - thumb_h)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255, 76))
            painter.drawRoundedRect(QRectF(track_x, thumb_y, 3.4, thumb_h), 1.7, 1.7)

    def _paint_emoji_grid(
        self,
        painter: QPainter,
        font_family: str,
        accent: QColor,
        w: float,
        h: float,
        hdr: float,
        n: int,
    ) -> None:
        emoji_norm = QFont(font_family, 17)
        emoji_sel = QFont(font_family, 20)
        for idx, rect in self._item_rects:
            spec = self._matches[idx]
            is_sel = idx == self._selected_idx
            if is_sel:
                painter.setBrush(QColor(accent.red(), accent.green(), accent.blue(), 78))
                painter.setPen(QPen(QColor(accent.red(), accent.green(), accent.blue(), 190), 1.15))
            else:
                painter.setBrush(QColor(255, 255, 255, 9))
                painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(rect, 10.0, 10.0)
            if spec.emoji:
                painter.setFont(emoji_sel if is_sel else emoji_norm)
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, spec.emoji)
            else:
                lbl_font = QFont(font_family, 9 if is_sel else 8, QFont.Weight.Bold)
                fm_l = QFontMetrics(lbl_font)
                lbl = fm_l.elidedText(get_spec_display_label(spec, self._ui_lang), Qt.TextElideMode.ElideRight, int(rect.width() - 6.0))
                painter.setFont(lbl_font)
                painter.setPen(QColor(248, 250, 252) if is_sel else QColor(203, 213, 225))
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, lbl)

        foot_y = h - 25.0
        painter.setPen(QPen(QColor(255, 255, 255, 16), 1.0))
        painter.drawLine(QPointF(12.0, foot_y), QPointF(w - 12.0, foot_y))

        active_spec = self._matches[self._selected_idx % n]
        cmd_s = get_primary_slash_cmd(active_spec, self._ui_lang)
        desc_s = get_tag_description(active_spec, self._ui_lang)
        cap_font = QFont(font_family, 9, QFont.Weight.DemiBold)
        fm_cap = QFontMetrics(cap_font)
        caption = f"{cmd_s} · {desc_s}" if desc_s else cmd_s
        elided = fm_cap.elidedText(caption, Qt.TextElideMode.ElideRight, int(w - 20.0))
        painter.setFont(cap_font)
        painter.setPen(QColor(215, 223, 236, 235))
        painter.drawText(QRectF(10.0, foot_y + 2.0, w - 20.0, 20.0), Qt.AlignmentFlag.AlignCenter, elided)

        total_rows = (n + 3) // 4
        if total_rows > 2 and self._is_scrollbar_visible():
            track_x = w - 6.5
            track_y = hdr + 8.0
            track_h = max(20.0, foot_y - track_y - 4.0)
            thumb_h = max(18.0, track_h * (2.0 / float(total_rows)))
            max_row = max(1, total_rows - 2)
            frac = float(self._scroll_offset) / float(max_row)
            thumb_y = track_y + frac * (track_h - thumb_h)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255, 76))
            painter.drawRoundedRect(QRectF(track_x, thumb_y, 3.4, thumb_h), 1.7, 1.7)

    @staticmethod
    def _draw_clean_outlined(
        painter: QPainter,
        x: float,
        y: float,
        text: str,
        font: QFont,
        fg: QColor,
        outline_r: float = 1.35,
    ) -> None:
        painter.setFont(font)
        painter.setPen(QColor(0, 0, 0, 145))
        painter.drawText(QPointF(x, y + 1.4), text)
        painter.setPen(QColor(0, 0, 0, 230))
        d = outline_r * 0.72
        for dx, dy in (
            (-outline_r, 0.0), (outline_r, 0.0), (0.0, -outline_r), (0.0, outline_r),
            (-d, -d), (d, -d), (-d, d), (d, d),
        ):
            painter.drawText(QPointF(x + dx, y + dy), text)
        painter.setPen(fg)
        painter.drawText(QPointF(x, y), text)

    def _paint_emoji_dock(
        self,
        painter: QPainter,
        font_family: str,
        accent: QColor,
        w: float,
        h: float,
        hdr: float,
        n: int,
    ) -> None:
        # Тимчасовий підпис режиму при перемиканні Numpad Ins (чистий текст з обводкою)
        if self._toast_text:
            t_font = QFont(font_family, 8, QFont.Weight.DemiBold)
            fm_t = QFontMetrics(t_font)
            tx = (w - float(fm_t.horizontalAdvance(self._toast_text))) / 2.0
            ty = 2.0 + float(fm_t.ascent())
            self._draw_clean_outlined(painter, tx, ty, self._toast_text, t_font, QColor(196, 181, 253, 240), outline_r=1.15)

        dock_y = hdr + 22.0
        dock_h = 42.0
        # Невидима 1/255 альфа-смуга під рядом смайликів, щоб hover/колесо/клік не переривалися у проміжках між іконками
        if self._item_rects:
            hit_left = self._item_rects[0][1].left() - 6.0
            hit_right = self._item_rects[-1][1].right() + 6.0
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 1))
            painter.drawRoundedRect(QRectF(hit_left, dock_y, hit_right - hit_left, dock_h), 16.0, 16.0)

        emoji_norm = QFont(font_family, 15)
        emoji_sel = QFont(font_family, 20)
        active_idx = self._hover_idx if self._hover_idx is not None else self._selected_idx
        sel_center_x = w / 2.0

        txt_norm = QFont(font_family, 10, QFont.Weight.Bold)
        txt_sel = QFont(font_family, 12, QFont.Weight.Bold)
        for idx, rect in self._item_rects:
            spec = self._matches[idx]
            is_sel = idx == active_idx
            if is_sel:
                sel_center_x = rect.center().x()
            if spec.emoji:
                painter.setFont(emoji_sel if is_sel else emoji_norm)
                painter.setPen(QColor(255, 255, 255))
                draw_r = QRectF(rect.left(), dock_y - (4.0 if is_sel else 0.0), rect.width(), dock_h + (2.0 if is_sel else 0.0))
                painter.drawText(draw_r, Qt.AlignmentFlag.AlignCenter, spec.emoji)
            else:
                lbl = get_spec_display_label(spec, self._ui_lang)
                fnt = txt_sel if is_sel else txt_norm
                fm_b = QFontMetrics(fnt)
                bw = float(fm_b.horizontalAdvance(lbl))
                bx = rect.center().x() - bw / 2.0
                by = dock_y + (dock_h - float(fm_b.height())) / 2.0 + float(fm_b.ascent()) - (3.0 if is_sel else 0.0)
                fg_c = QColor(255, 255, 255, 252) if is_sel else QColor(220, 226, 238, 220)
                self._draw_clean_outlined(painter, bx, by, lbl, fnt, fg_c, outline_r=1.35)

        # Чистий напис зверху без фону (з темною обводкою, щоб читався на будь-якому фоні)
        active_spec = self._matches[active_idx % n]
        tip_text = (
            get_primary_cmd_name(active_spec, self._ui_lang)
            if active_spec.emoji
            else (get_tag_description(active_spec, self._ui_lang) or get_primary_cmd_name(active_spec, self._ui_lang))
        )
        tip_font = QFont(font_family, 9, QFont.Weight.Bold)
        fm_tip = QFontMetrics(tip_font)
        tip_w = float(fm_tip.horizontalAdvance(tip_text))
        tip_x = max(6.0, min(w - tip_w - 6.0, sel_center_x - tip_w / 2.0))
        tip_y = hdr + 3.0 + float(fm_tip.ascent())
        self._draw_clean_outlined(painter, tip_x, tip_y, tip_text, tip_font, QColor(255, 255, 255, 248), outline_r=1.35)

    def _paint_menu_v(
        self,
        painter: QPainter,
        font_family: str,
        accent: QColor,
        w: float,
        h: float,
        hdr: float,
        n: int,
    ) -> None:
        is_micro = self._style == "menu_micro"
        vis_limit = 4 if is_micro else 5
        emoji_font = QFont(font_family, 9 if is_micro else 10)
        cmd_font = QFont(font_family, 9 if is_micro else 9, QFont.Weight.DemiBold)
        desc_font = QFont(font_family, 8 if is_micro else 9)
        fm_cmd = QFontMetrics(cmd_font)
        fm_desc = QFontMetrics(desc_font)

        # Обрізаємо підсвітку активного рядка по закруглених кутах картки (точно як на скріні)
        clip_path = QPainterPath()
        clip_path.addRoundedRect(QRectF(1.5, 1.5, w - 3.0, h - 3.0), 9.5, 9.5)
        painter.save()
        painter.setClipPath(clip_path)

        show_scroll = (n > vis_limit) and self._is_scrollbar_visible()
        right_pad = 13.0 if show_scroll else 10.0

        for idx, rect in self._item_rects:
            spec = self._matches[idx]
            is_sel = idx == self._selected_idx
            if is_sel:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(255, 255, 255, 20))
                painter.drawRect(rect)

            left_x = rect.left() + (8.0 if is_micro else 10.0)
            painter.setFont(emoji_font)
            painter.setPen(QColor(255, 255, 255) if spec.emoji else QColor(148, 163, 184))
            painter.drawText(
                QRectF(left_x, rect.top(), 18.0, rect.height()),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                spec.emoji or "♪",
            )

            cmd_name = get_primary_cmd_name(spec, self._ui_lang)
            cmd_x = left_x + 22.0
            cmd_w = float(fm_cmd.horizontalAdvance(cmd_name)) + 2.0
            painter.setFont(cmd_font)
            painter.setPen(QColor(238, 240, 244) if is_sel else QColor(220, 224, 232))
            painter.drawText(
                QRectF(cmd_x, rect.top(), cmd_w, rect.height()),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                cmd_name,
            )

            desc = get_tag_description(spec, self._ui_lang)
            desc_x = cmd_x + cmd_w + 7.0
            desc_w = max(24.0, rect.right() - desc_x - right_pad)
            elided = fm_desc.elidedText(desc, Qt.TextElideMode.ElideRight, int(desc_w))
            painter.setFont(desc_font)
            painter.setPen(QColor(154, 160, 172) if is_sel else QColor(134, 140, 152))
            painter.drawText(
                QRectF(desc_x, rect.top(), desc_w, rect.height()),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                elided,
            )

        painter.restore()

        # Тонкий повзунок прокрутки без фону — показується ТІЛЬКИ при наведенні мишки або під час прокрутки
        if show_scroll:
            track_x = w - 7.0
            track_y = hdr + 5.0
            track_h = max(18.0, h - track_y - 5.0)
            thumb_h = max(18.0, track_h * (float(vis_limit) / float(n)))
            max_off = max(1, n - vis_limit)
            frac = float(self._scroll_offset) / float(max_off)
            thumb_y = track_y + frac * (track_h - thumb_h)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255, 76))
            painter.drawRoundedRect(QRectF(track_x, thumb_y, 3.6, thumb_h), 1.8, 1.8)


class ScreenOverlayWindow(QWidget):
    """
    Click-through оверлей, який ніколи не забирає фокус у ігор.
    Коли користувач вмикає набір — з'являється лише ледь помітний індикатор,
    а при наборі — чистий красивий текст без фону (або з рамкою, якщо обрано в налаштуваннях).
    Підтримує перетягування мишкою в режимі налаштування позиції.
    """

    position_changed = Signal(int, int)       # (x_offset, y_offset)
    move_mode_changed = Signal(bool)          # Чи активний режим перетягування мишкою
    tag_picker_chosen = Signal(str)           # Обраний кліком тег OmniVoice (`[laughter]`)
    tag_picker_selection_changed = Signal(int)
    tag_picker_hwnd_changed = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("SilentVoice Overlay")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self._mode: str = "idle"          # "typing" | "speaking" | "preview" | "move" | "idle"
        self._text: str = ""
        self._lang_code: str = "UA"
        self._cursor_visible: bool = True
        self._interactive_move: bool = False
        self._drag_start_pos: Optional[QPoint] = None
        self._win_start_pos: Optional[QPoint] = None
        self._last_box_rect: QRectF = QRectF()
        self._last_paint_rect: QRectF = QRectF()
        self._last_mask_rect: QRect = QRect()
        self._text_cache_key: Optional[tuple] = None
        self._text_cache_img: Optional[QImage] = None
        saved_picker_style = str(config.get("reaction_picker_style", "emoji_dock") or "emoji_dock").strip()
        self._tag_hint_style: str = saved_picker_style if saved_picker_style in TAG_HINT_MODES else "emoji_dock"
        self._tag_hint_selected_idx: int = 0
        self._tag_mode_toast_text: str = ""
        self._tag_mode_toast_timer = QTimer(self)
        self._tag_mode_toast_timer.setSingleShot(True)
        self._tag_mode_toast_timer.timeout.connect(self._clear_tag_mode_toast)

        self._tag_picker = OmniVoiceTagPickerPopup()
        self._tag_picker.tag_chosen.connect(self._on_tag_picker_chosen)
        self._tag_picker.selection_changed.connect(self._on_tag_picker_selection_changed)
        self._tag_picker.visibility_hwnd_changed.connect(self.tag_picker_hwnd_changed.emit)
        self._picker_sync_pending: bool = False
        self._picker_preview_active: bool = False
        self._picker_preview_timer = QTimer(self)
        self._picker_preview_timer.setSingleShot(True)
        self._picker_preview_timer.timeout.connect(self._end_picker_preview)

        self._cursor_timer = QTimer(self)
        self._cursor_timer.setInterval(500)
        self._cursor_timer.timeout.connect(self._toggle_cursor)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide_overlay)

        self.resize(960, 185)
        self.reposition()

    @staticmethod
    def get_tag_hint_mode_label(mode: str) -> str:
        return TAG_HINT_MODE_LABELS.get(mode, mode)

    def _on_tag_picker_chosen(self, tag: str) -> None:
        if self._picker_preview_active:
            self._end_picker_preview()
            return
        self.tag_picker_chosen.emit(tag)

    def _on_tag_picker_selection_changed(self, idx: int) -> None:
        self._tag_hint_selected_idx = max(0, int(idx))
        if not self._picker_preview_active:
            self.tag_picker_selection_changed.emit(self._tag_hint_selected_idx)

    def _clear_tag_mode_toast(self) -> None:
        self._tag_mode_toast_text = ""
        if self.isVisible():
            self.update()
        if self._tag_picker.isVisible() and not self._picker_preview_active:
            self._schedule_tag_picker_sync()

    def _end_picker_preview(self) -> None:
        if self._picker_preview_active and self._tag_picker.isVisible() and self._tag_picker.underMouse():
            self._picker_preview_timer.start(1400)
            return
        self._picker_preview_active = False
        self._picker_preview_timer.stop()
        if not (self.isVisible() and self._mode == "typing"):
            self._tag_picker.hide_picker()

    def set_tag_hint_style(self, style: str) -> str:
        clean = str(style or "emoji_dock").strip()
        self._tag_hint_style = clean if clean in TAG_HINT_MODES else "emoji_dock"
        if self.isVisible() and self._mode == "typing":
            self._schedule_tag_picker_sync()
        return self._tag_hint_style

    def preview_tag_picker_style(
        self,
        style: Optional[str] = None,
        *,
        duration_ms: int = 4000,
        anchor_global: Optional[QPoint] = None,
    ) -> None:
        """Показує живий попередній перегляд обраного стилю підказки команд."""
        if style is not None:
            self.set_tag_hint_style(style)
        ui_lang = str(config.get("app_language", "uk") or "uk")
        specs = build_active_reaction_specs(config.data, ui_lang=ui_lang)
        if not specs:
            specs = build_all_configured_reaction_specs(config.data, ui_lang=ui_lang)
        if not specs:
            specs = list(OMNIVOICE_REACTION_TAGS)

        if anchor_global is not None:
            top_pt = anchor_global
            bot_y = anchor_global.y() + 40
        else:
            self.reposition()
            g = self.geometry()
            top_pt = QPoint(g.x() + g.width() // 2, g.y() + g.height() - 62)
            bot_y = g.y() + g.height() - 10

        self._picker_preview_active = True
        self._tag_picker.sync_state(
            style=self._tag_hint_style,
            matches=specs,
            ui_lang=ui_lang,
            selected_idx=0,
            anchor_top_center=top_pt,
            anchor_bottom_y=bot_y,
            toast_text="",
        )
        self._picker_preview_timer.start(max(1200, int(duration_ms)))

    def set_tag_hint_selected_index(self, idx: int) -> None:
        self._tag_hint_selected_idx = max(0, int(idx))
        if self.isVisible() and self._mode == "typing":
            self.update()
            self._schedule_tag_picker_sync()

    def toggle_tag_hint_style(self) -> str:
        try:
            cur_i = TAG_HINT_MODES.index(self._tag_hint_style)
        except ValueError:
            cur_i = 0
        self._tag_hint_style = TAG_HINT_MODES[(cur_i + 1) % len(TAG_HINT_MODES)]
        config.set("reaction_picker_style", self._tag_hint_style)
        if self.isVisible() and self._mode == "typing":
            self._schedule_tag_picker_sync()
        return self._tag_hint_style

    def _schedule_tag_picker_sync(self) -> None:
        if self._picker_sync_pending:
            return
        self._picker_sync_pending = True
        QTimer.singleShot(0, self._sync_tag_picker)

    def _sync_tag_picker(self) -> None:
        self._picker_sync_pending = False
        if not self.isVisible() or self._mode != "typing":
            if not self._picker_preview_active:
                self._tag_picker.hide_picker()
            return
        self._picker_preview_active = False
        self._picker_preview_timer.stop()

        slash_state = extract_active_slash_query(self._text)
        if slash_state is None:
            self._tag_picker.hide_picker()
            return
        ui_lang = str(config.get("app_language", "uk") or "uk")
        active_specs = build_active_reaction_specs(config.data, ui_lang=ui_lang)
        if not active_specs:
            self._tag_picker.hide_picker()
            return
        matches = match_reaction_tags(slash_state[1], ui_lang, specs=active_specs)
        if not matches:
            self._tag_picker.hide_picker()
            return

        style_cfg = str(config.get("reaction_picker_style", self._tag_hint_style) or self._tag_hint_style).strip()
        if style_cfg in TAG_HINT_MODES:
            self._tag_hint_style = style_cfg

        if not self._last_box_rect.isEmpty():
            top_local = QPoint(int(round(self._last_box_rect.center().x())), int(round(self._last_box_rect.top())))
            bot_local = QPoint(int(round(self._last_box_rect.center().x())), int(round(self._last_box_rect.bottom())))
        else:
            top_local = QPoint(self.width() // 2, self.height() - 56)
            bot_local = QPoint(self.width() // 2, self.height() - 8)
        anchor_top = self.mapToGlobal(top_local)
        anchor_bot = self.mapToGlobal(bot_local)

        self._tag_picker.sync_state(
            style=self._tag_hint_style,
            matches=matches,
            ui_lang=ui_lang,
            selected_idx=self._tag_hint_selected_idx,
            anchor_top_center=anchor_top,
            anchor_bottom_y=anchor_bot.y(),
            toast_text="",
        )

    @property
    def is_move_mode(self) -> bool:
        return self._interactive_move

    @property
    def is_draggable_now(self) -> bool:
        return self._interactive_move or bool(config.get("overlay_draggable", False)) or str(config.get("overlay_style", "text_only")) == "interactive_box"

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._apply_win32_flags()

    def _apply_win32_flags(self) -> None:
        hwnd = int(self.winId())
        # У режимі перетягування мишкою або у режимі «Інтерактивне вікно 1 (Рухома плашка)»
        # дозволяємо захоплення мишкою, щоб плашку можна було пересувати в будь-який момент
        draggable = self.is_draggable_now
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not draggable)
        if draggable:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.unsetCursor()
        click_through = not draggable
        no_activate = not self._interactive_move
        apply_overlay_ex_styles(hwnd, click_through=click_through, no_activate=no_activate)
        excluded = bool(config.get("hide_from_capture", False))
        set_window_capture_excluded(hwnd, excluded)

    def set_interactive_move(self, enabled: bool) -> None:
        """
        Вмикає або вимикає режим ручного перетягування оверлею мишкою по екрану.
        Після вимкнення миттєво повертає WS_EX_TRANSPARENT (повний пропуск кліків для ігор).
        """
        self._interactive_move = enabled
        self._hide_timer.stop()

        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not enabled)

        if enabled:
            self._tag_picker.hide_picker()
            self._mode = "move"
            self._text = tr(
                "overlay.drag_hint",
                "Перетягніть цей текст мишкою у зручне місце (Подвійний клік — зафіксувати)",
            )
            self._cursor_timer.stop()
            self.setCursor(Qt.CursorShape.SizeAllCursor)
            self.reposition()
            self._show_cleanly()
            self.raise_()
        else:
            self.unsetCursor()
            self._apply_win32_flags()
            self.hide_overlay()

        self.move_mode_changed.emit(enabled)

    def update_capture_affinity(self) -> None:
        hwnd = int(self.winId())
        excluded = bool(config.get("hide_from_capture", False))
        set_window_capture_excluded(hwnd, excluded)
        self._tag_picker.update_capture_affinity()

    def closeEvent(self, event) -> None:
        self._tag_picker.hide_picker()
        self._tag_picker.close()
        super().closeEvent(event)

    def retranslate_ui(self) -> None:
        """Оновлює системні підказки оверлею при зміні мови інтерфейсу на льоту."""
        if self._mode == "move":
            self._text = tr(
                "overlay.drag_hint",
                "Перетягніть цей текст мишкою у зручне місце (Подвійний клік — зафіксувати)",
            )
            self.update()
        elif self._mode == "preview":
            self._text = tr(
                "overlay.preview_sample",
                "Привіт! Ось так виглядає ваш текст на екрані ✨ Так відображатимуться ваші повідомлення поверх ігор та програм: шрифт, розмір, кольори, тінь, обводка та плашка. Налаштуйте все під себе!",
            )
            self.update()
        elif self._mode == "typing":
            self.update()

    def reposition(self) -> None:
        screen = resolve_screen_by_name(str(config.get("overlay_screen_name", "") or ""))
        if not screen:
            return
        geom = screen.geometry()
        max_w_cfg = max(320, min(2400, int(config.get("overlay_max_width", 1120))))
        w = min(max(320, int(geom.width() - 40)), max_w_cfg)
        font_size = int(config.get("overlay_font_size", 18))
        h = max(185, int(font_size * 5.6) + 74)
        x_offset = int(config.get("overlay_x_offset", 0))
        y_offset = int(config.get("overlay_y_offset", 85))

        center_x = geom.x() + (geom.width() - w) // 2 + x_offset
        pos_y = geom.y() + geom.height() - h - y_offset

        # Обмежуємо в межах екрана
        clamped_x = max(geom.x() - w // 3, min(geom.x() + geom.width() - w * 2 // 3, center_x))
        clamped_y = max(geom.y() + 5, min(geom.y() + geom.height() - 60, pos_y))
        target_rect = QRect(clamped_x, clamped_y, w, h)
        if self.geometry() != target_rect:
            self.setGeometry(target_rect)

    def _show_cleanly(self) -> None:
        """Показує оверлей без миготіння попереднього кадру з кешу шарованого вікна Windows."""
        if not self.isVisible():
            self.setWindowOpacity(0.0)
            self.show()
            self._apply_win32_flags()
            self.repaint()
            self.setWindowOpacity(1.0)
        else:
            self._apply_win32_flags()
            self.repaint()

    def show_typing(self, text: str, lang_code: str = "") -> None:
        if not config.get("overlay_enabled", True):
            return
        if self._interactive_move:
            self.set_interactive_move(False)
        self._hide_timer.stop()
        self._cursor_timer.stop()
        was_visible_typing = self.isVisible() and self._mode == "typing"
        if text != self._text:
            self._tag_hint_selected_idx = 0
        self._mode = "typing"
        self._text = text
        self._lang_code = lang_code
        if was_visible_typing:
            # Асинхронне коалесцоване оновлення без Win32-рефлагів та синхронного repaint на кожну клавішу
            self.update()
        else:
            self.reposition()
            self._show_cleanly()
        if self._tag_hint_style in INTERACTIVE_TAG_HINT_MODES or self._tag_picker.isVisible():
            self._schedule_tag_picker_sync()

    def show_speaking(self, text: str, duration_sec: float = 2.5) -> None:
        if not config.get("overlay_enabled", True):
            return
        if self._interactive_move:
            self.set_interactive_move(False)
        self._tag_picker.hide_picker()
        self._cursor_timer.stop()
        self._cursor_visible = False
        self._mode = "speaking"
        self._text = text
        self.reposition()
        self._show_cleanly()

        auto_hide_sec = float(config.get("overlay_auto_hide_sec", 3.0))
        total_ms = int((max(0.5, duration_sec) + auto_hide_sec) * 1000)
        self._hide_timer.start(total_ms)

    def show_preview(self, sample_text: Optional[str] = None) -> None:
        """Показує тестовий вигляд оверлею під час налаштування повзунків."""
        if self._interactive_move:
            self.update()
            return
        self._tag_picker.hide_picker()
        self._cursor_timer.stop()
        self._cursor_visible = False
        self._mode = "preview"
        self._text = sample_text or tr(
            "overlay.preview_sample",
            "Привіт! Ось так виглядає ваш текст на екрані ✨ Так відображатимуться ваші повідомлення поверх ігор та програм: шрифт, розмір, кольори, тінь, обводка та плашка. Налаштуйте все під себе!",
        )
        self.reposition()
        self._show_cleanly()
        self._hide_timer.start(4000)

    def hide_overlay(self) -> None:
        if self._interactive_move:
            return
        self._tag_picker.hide_picker()
        self._cursor_timer.stop()
        self._hide_timer.stop()
        self._mode = "idle"
        self._text = ""
        self._tag_hint_selected_idx = 0
        self._last_box_rect = QRectF()
        self._last_paint_rect = QRectF()
        self._sync_drag_mask()
        if self.isVisible():
            self.repaint()
        self.hide()

    def _sync_drag_mask(self) -> None:
        """Обмежує область захоплення мишки виключно самою плашкою (та підказкою тегів), а не порожнім фоном навколо."""
        mask_src = self._last_paint_rect if not self._last_paint_rect.isEmpty() else self._last_box_rect
        if self.is_draggable_now and self.isVisible() and not mask_src.isEmpty():
            target = mask_src.toAlignedRect().adjusted(-3, -3, 3, 3)
            if target != self._last_mask_rect:
                self._last_mask_rect = target
                self.setMask(QRegion(target))
                # Перемальовуємо, щоб розширені лівий і правий краї плашки не лишалися обрізаними старою маскою
                self.update()
        else:
            if not self._last_mask_rect.isEmpty():
                self._last_mask_rect = QRect()
                self.clearMask()

    def _toggle_cursor(self) -> None:
        self._cursor_visible = not self._cursor_visible
        self.update()

    # ── Перетягування мишкою (у режимі налаштування або в режимі «Інтерактивне вікно 1 — Рухома плашка») ──

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if (
            self.is_draggable_now
            and event.button() == Qt.MouseButton.LeftButton
            and not self._last_box_rect.isEmpty()
            and self._last_box_rect.contains(event.position())
        ):
            self._drag_start_pos = event.globalPosition().toPoint()
            self._win_start_pos = self.pos()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.is_draggable_now and self._drag_start_pos is not None and self._win_start_pos is not None:
            delta = event.globalPosition().toPoint() - self._drag_start_pos
            new_pos = self._win_start_pos + delta
            self.move(new_pos)

            screen_name, x_off, y_off = calc_screen_relative_offsets(QRect(new_pos, self.size()))
            config.update(
                {
                    "overlay_screen_name": screen_name,
                    "overlay_x_offset": int(x_off),
                    "overlay_y_offset": int(y_off),
                },
                auto_save=False,
            )
            self.position_changed.emit(int(x_off), int(y_off))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self.is_draggable_now and event.button() == Qt.MouseButton.LeftButton:
            if self._drag_start_pos is not None:
                config.schedule_save(0.15)
            self._drag_start_pos = None
            self._win_start_pos = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self._interactive_move:
            self.set_interactive_move(False)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    # ── Малювання оверлею ──

    def paintEvent(self, event) -> None:
        if self._mode == "idle":
            self._last_box_rect = QRectF()
            self._last_paint_rect = QRectF()
            painter = QPainter(self)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            painter.fillRect(self.rect(), Qt.GlobalColor.transparent)
            painter.end()
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        style_mode = str(config.get("overlay_style", "text_only"))
        bg_opacity = int(config.get("overlay_bg_opacity", 78))
        text_opacity = int(config.get("overlay_text_opacity", 100))
        font_family = str(config.get("overlay_font_family", "Segoe UI"))
        font_size = int(config.get("overlay_font_size", 18))
        outline_width = int(config.get("overlay_outline_width", 3))
        shadow_blur = int(config.get("overlay_shadow_blur", 10))
        text_color_hex = str(config.get("overlay_text_color", "#FFFFFF"))
        bg_color_hex = str(config.get("overlay_bg_color", "#13151F"))
        border_color_hex = str(config.get("overlay_border_color", "#7C5CFF"))

        # Коли користувач щойно натиснув хоткей набору, але ще не почав писати:
        # у звичайному режимі — невеликий овал (22x8px);
        # у режимі «Рухома плашка» — компактна спокійна міні-капсула, щоб не відволікати увагу,
        # але її все одно можна було зручно перетягувати мишкою за саму плашку.
        if self._mode == "typing" and not self._text:
            if style_mode == "interactive_box":
                pill_w, pill_h = 68.0, 24.0
                pill_x = (self.width() - pill_w) / 2.0
                pill_y = self.height() - pill_h - 10.0
                pill_rect = QRectF(pill_x, pill_y, pill_w, pill_h)
                self._last_box_rect = pill_rect

                pill_bg = QColor(bg_color_hex)
                pill_bg.setAlpha(max(55, int(115 * (bg_opacity / 100.0))))
                pill_border = QColor(border_color_hex)
                pill_border.setAlpha(max(45, int(85 * (bg_opacity / 100.0))))
                painter.setBrush(pill_bg)
                painter.setPen(QPen(pill_border, 1.0))
                painter.drawRoundedRect(pill_rect, 12, 12)

                # Ненав'язливий овал готовності + ледь помітний маркер перетягування (⋮⋮)
                dot_col = QColor(border_color_hex)
                dot_col.setAlpha(int(140 * (text_opacity / 100.0)))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(dot_col)
                painter.drawEllipse(QRectF(pill_x + 12.0, pill_y + 9.0, 14.0, 6.0))

                grip_col = QColor(203, 213, 225, int(95 * (text_opacity / 100.0)))
                painter.setBrush(grip_col)
                for col_i in range(3):
                    gx = pill_x + 33.0 + col_i * 6.0
                    painter.drawEllipse(QRectF(gx, pill_y + 8.5, 2.2, 2.2))
                    painter.drawEllipse(QRectF(gx, pill_y + 13.5, 2.2, 2.2))

                paint_rect = QRectF(pill_rect)
                if self._tag_mode_toast_text:
                    toast_r = self._paint_tag_mode_toast(
                        painter, pill_y, font_family, text_opacity, bg_color_hex, border_color_hex
                    )
                    if not toast_r.isEmpty():
                        paint_rect = paint_rect.united(toast_r)
                self._last_paint_rect = paint_rect

                painter.end()
                target_mask = paint_rect.toAlignedRect().adjusted(-3, -3, 3, 3)
                if target_mask != self._last_mask_rect:
                    QTimer.singleShot(0, self._sync_drag_mask)
                return
            if self.is_draggable_now:
                oval_w, oval_h = 22.0, 8.0
                px = (self.width() - oval_w) / 2.0
                py = self.height() - oval_h - 17.0
                hit_rect = QRectF(px - 8.0, py - 6.0, oval_w + 16.0, oval_h + 12.0)
                self._last_box_rect = hit_rect
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(0, 0, 0, 1))
                painter.drawRoundedRect(hit_rect, 8, 8)
                self._paint_subtle_ready_indicator(painter, text_opacity, border_color_hex)
                paint_rect = QRectF(hit_rect)
                if self._tag_mode_toast_text:
                    toast_r = self._paint_tag_mode_toast(
                        painter, py - 6.0, font_family, text_opacity, bg_color_hex, border_color_hex
                    )
                    if not toast_r.isEmpty():
                        paint_rect = paint_rect.united(toast_r)
                self._last_paint_rect = paint_rect
                painter.end()
                target_mask = paint_rect.toAlignedRect().adjusted(-3, -3, 3, 3)
                if target_mask != self._last_mask_rect:
                    QTimer.singleShot(0, self._sync_drag_mask)
                return
            self._last_box_rect = QRectF()
            self._last_paint_rect = QRectF()
            self._paint_subtle_ready_indicator(painter, text_opacity, border_color_hex)
            if self._tag_mode_toast_text:
                self._paint_tag_mode_toast(
                    painter, self.height() - 28.0, font_family, text_opacity, bg_color_hex, border_color_hex
                )
            painter.end()
            if not self._last_mask_rect.isEmpty():
                QTimer.singleShot(0, self._sync_drag_mask)
            return

        is_omnivoice = str(config.get("active_engine", "") or "").strip() == "omnivoice"
        display_text = self._text
        # Для довгого тексту беремо лише видимий хвіст, щоб розбиття та рендер завжди виконувались за O(1)
        if len(display_text) > 320:
            tail = display_text[-320:]
            sp = tail.find(" ")
            if 0 <= sp < 60:
                tail = tail[sp + 1 :]
            display_text = "… " + tail
        if self._mode == "typing" and display_text:
            ui_lang_cur = str(config.get("app_language", "uk") or "uk")
            all_specs = build_all_configured_reaction_specs(config.data, ui_lang=ui_lang_cur)
            display_text = format_tags_for_display(display_text, specs=all_specs, ui_lang=ui_lang_cur)
            if is_omnivoice and bool(config.get("omnivoice_whisper", False)):
                display_text = f"🤫 {display_text}"

        font = QFont(font_family, font_size, QFont.Weight.Bold)
        painter.setFont(font)
        fm = QFontMetrics(font)

        ow = float(max(0, min(8, outline_width)))
        max_box_w = self.width() - 36
        pad_x = int(math.ceil(ow + 16))
        pad_y = int(math.ceil(ow + 8))
        max_text_w = max(120, max_box_w - pad_x * 2)
        lines = self._wrap_text_lines(display_text, fm, max_text_w)
        line_h = fm.height() + max(0, int((ow - 2.0) * 1.4))
        max_line_w = max((fm.horizontalAdvance(ln) for ln in lines), default=0)

        box_w = max(220, min(max_box_w, max_line_w + pad_x * 2))
        box_h = min(self.height() - 10, len(lines) * line_h + pad_y * 2)

        box_x = (self.width() - box_w) / 2.0
        box_y = self.height() - box_h - 8
        box_rect = QRectF(box_x, box_y, box_w, box_h)
        self._last_box_rect = box_rect

        # Рамка перетягування у режимі налаштування позиції мишкою
        if self._mode == "move":
            move_bg = QColor(19, 21, 31, 210)
            painter.setBrush(move_bg)
            painter.setPen(QPen(QColor("#7C5CFF"), 2.0, Qt.PenStyle.DashLine))
            painter.drawRoundedRect(box_rect, 14, 14)
        elif style_mode in ("boxed", "interactive_box"):
            # Режим «З рамкою і плашкою» або спокійний режим «Рухома плашка»
            is_interactive_box = style_mode == "interactive_box"
            bg_col = QColor(bg_color_hex)
            bg_scale = 0.78 if is_interactive_box else 1.0
            bg_col.setAlpha(int(255 * (bg_opacity / 100.0) * bg_scale))
            border_col = QColor(border_color_hex)
            border_col.setAlpha(
                max(50, int(115 * (bg_opacity / 100.0)))
                if is_interactive_box
                else max(90, int(220 * (bg_opacity / 100.0)))
            )

            painter.setBrush(bg_col)
            painter.setPen(QPen(border_col, 1.2 if is_interactive_box else 1.6))
            painter.drawRoundedRect(box_rect, 14, 14)
        elif self.is_draggable_now:
            # Невидима 1/255 альфа-зона в межах тексту, щоб Windows передавав кліки миші при перетягуванні без плашки
            painter.setBrush(QColor(0, 0, 0, 1))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(box_rect, 12, 12)

        # Малювання тексту
        painter.setFont(font)
        text_area = QRectF(box_x + pad_x * 0.5, box_y + 4, box_w - pad_x, box_h - 8)

        text_col = QColor(text_color_hex)
        eff_opacity = text_opacity
        text_col.setAlpha(int(255 * (eff_opacity / 100.0)))
        outline_col = QColor(0, 0, 0, int(235 * (eff_opacity / 100.0)))
        sh_col = QColor(0, 0, 0, int(145 * (eff_opacity / 100.0)))

        total_text_h = len(lines) * line_h
        start_y = text_area.y() + (text_area.height() - total_text_h) / 2.0 + fm.ascent()

        if ow > 2.0:
            # Для товстої обводки (> 2 px) використовуємо векторний QPainterPath з RoundJoin,
            # інакше 8 зсувів drawText дають «рвані» краї й роздвоєння букв. Результат кешується в QImage
            # без запікання eff_opacity — зміна прозорості тексту застосовується миттєво через painter.setOpacity.
            cache_key = (
                display_text,
                font_family,
                font_size,
                ow,
                shadow_blur,
                text_color_hex,
                int(box_w),
                int(box_h),
            )
            if self._text_cache_key != cache_key or self._text_cache_img is None:
                dpr = max(1.0, float(self.devicePixelRatioF()))
                cw = max(1, int(round(box_w)))
                ch = max(1, int(round(box_h)))
                img = QImage(int(round(cw * dpr)), int(round(ch * dpr)), QImage.Format.Format_ARGB32_Premultiplied)
                img.setDevicePixelRatio(dpr)
                img.fill(Qt.GlobalColor.transparent)

                cp = QPainter(img)
                cp.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                cp.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

                local_start_y = (ch - total_text_h) / 2.0 + fm.ascent()
                path = QPainterPath()
                for i, line in enumerate(lines):
                    lw = fm.horizontalAdvance(line)
                    lx = (cw - lw) / 2.0
                    ly = local_start_y + i * line_h
                    path.addText(QPointF(lx, ly), font, line)

                full_text_col = QColor(text_color_hex)
                full_text_col.setAlpha(255)
                full_outline_col = QColor(0, 0, 0, 235)
                full_sh_col = QColor(0, 0, 0, 145)

                if shadow_blur > 0:
                    sh_off = max(1.5, min(4.0, shadow_blur / 5.0))
                    cp.save()
                    cp.translate(0.0, sh_off + ow * 0.4)
                    cp.strokePath(
                        path,
                        QPen(full_sh_col, ow * 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin),
                    )
                    cp.fillPath(path, full_sh_col)
                    cp.restore()

                cp.strokePath(
                    path,
                    QPen(full_outline_col, ow * 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin),
                )
                cp.fillPath(path, full_text_col)
                cp.end()

                self._text_cache_key = cache_key
                self._text_cache_img = img

            if eff_opacity < 100:
                painter.save()
                painter.setOpacity(eff_opacity / 100.0)
                painter.drawImage(QPointF(box_x, box_y), self._text_cache_img)
                painter.restore()
            else:
                painter.drawImage(QPointF(box_x, box_y), self._text_cache_img)
        else:
            outline_offsets = (
                (-ow, 0.0), (ow, 0.0), (0.0, -ow), (0.0, ow),
                (-ow * 0.72, -ow * 0.72), (ow * 0.72, -ow * 0.72),
                (-ow * 0.72, ow * 0.72), (ow * 0.72, ow * 0.72),
            ) if ow > 0 else ()

            for i, line in enumerate(lines):
                lw = fm.horizontalAdvance(line)
                lx = text_area.x() + (text_area.width() - lw) / 2.0
                ly = start_y + i * line_h

                if shadow_blur > 0:
                    sh_off = max(1.5, min(4.0, shadow_blur / 5.0))
                    painter.setPen(sh_col)
                    painter.drawText(QPointF(lx, ly + sh_off + ow * 0.5), line)

                if outline_offsets:
                    painter.setPen(outline_col)
                    for dx, dy in outline_offsets:
                        painter.drawText(QPointF(lx + dx, ly + dy), line)

                painter.setPen(text_col)
                painter.drawText(QPointF(lx, ly), line)

        # Підказка звукових команд і реакцій (показується поки активний токен `/...`, `\...` або `|...`)
        paint_rect = QRectF(box_rect)
        hint_rect = QRectF()
        popup_handles_toast = False
        if self._mode == "typing":
            slash_state = extract_active_slash_query(self._text)
            if slash_state is not None:
                popup_handles_toast = True
                self._schedule_tag_picker_sync()
            elif self._tag_picker.isVisible() and not self._picker_preview_active:
                self._schedule_tag_picker_sync()

        if self._tag_mode_toast_text and not popup_handles_toast:
            toast_anchor_y = hint_rect.top() if not hint_rect.isEmpty() else box_y
            toast_r = self._paint_tag_mode_toast(
                painter, toast_anchor_y, font_family, text_opacity, bg_color_hex, border_color_hex
            )
            if not toast_r.isEmpty():
                paint_rect = paint_rect.united(toast_r)

        self._last_paint_rect = paint_rect
        if self.is_draggable_now:
            target_mask = paint_rect.toAlignedRect().adjusted(-3, -3, 3, 3)
            if target_mask != self._last_mask_rect:
                QTimer.singleShot(0, self._sync_drag_mask)
        elif not self._last_mask_rect.isEmpty():
            QTimer.singleShot(0, self._sync_drag_mask)

        painter.end()

    @staticmethod
    def _draw_outlined_text(
        painter: QPainter,
        x: float,
        y: float,
        text: str,
        font: QFont,
        fg_color: QColor,
        alpha_scale: float,
        outline_r: float = 1.35,
    ) -> None:
        painter.setFont(font)
        sh_col = QColor(0, 0, 0, int(135 * alpha_scale))
        painter.setPen(sh_col)
        painter.drawText(QPointF(x, y + 1.5), text)

        out_col = QColor(0, 0, 0, int(225 * alpha_scale))
        painter.setPen(out_col)
        d = outline_r * 0.72
        for dx, dy in (
            (-outline_r, 0.0), (outline_r, 0.0), (0.0, -outline_r), (0.0, outline_r),
            (-d, -d), (d, -d), (-d, d), (d, d),
        ):
            painter.drawText(QPointF(x + dx, y + dy), text)

        painter.setPen(fg_color)
        painter.drawText(QPointF(x, y), text)

    def _paint_omnivoice_tag_hint(
        self,
        painter: QPainter,
        anchor_y: float,
        font_family: str,
        base_font_size: int,
        text_opacity: int,
        bg_opacity: int,
        bg_color_hex: str,
        border_color_hex: str,
        query: str,
    ) -> QRectF:
        """
        Малює 1-рядкову підказку тегів OmniVoice прямо над текстом набору:
        - Варіант A (`"text"`): чистий текст із темною обводкою, без жодної рамки/плашки.
        - Варіант B (`"pill"`): тонка напівпрозора 1-рядкова капсула.
        """
        ui_lang = str(config.get("app_language", "uk") or "uk")
        matches = match_reaction_tags(query, ui_lang)
        if not matches:
            return QRectF()

        is_pill = self._tag_hint_style == "pill"
        is_emoji_row = self._tag_hint_style == "emoji_row"
        sel_idx = self._tag_hint_selected_idx % len(matches)
        alpha_scale = max(0.25, min(1.0, text_opacity / 100.0))

        hint_pt = max(10, min(13, int(round(base_font_size * 0.68))))
        font_norm = QFont(font_family, hint_pt + (2 if is_emoji_row else 0), QFont.Weight.DemiBold)
        font_sel = QFont(font_family, hint_pt + (1 if is_emoji_row else 0), QFont.Weight.Bold)
        font_tab = QFont(font_family, max(9, hint_pt - 2), QFont.Weight.DemiBold)

        fm_norm = QFontMetrics(font_norm)
        fm_sel = QFontMetrics(font_sel)
        fm_tab = QFontMetrics(font_tab)

        item_pad_x = 8.0 if is_pill else 3.0
        item_gap = 6.0 if is_pill else (11.0 if is_emoji_row else 13.0)
        outer_pad_x = 11.0 if is_pill else 6.0
        tab_label = "Tab" if is_pill else "· Tab"
        tab_w = float(fm_tab.horizontalAdvance(tab_label)) + (12.0 if is_pill else 4.0)
        tab_gap = 8.0 if is_pill else 12.0

        entries: list[tuple[int, str, str, float, float, float]] = []
        for idx, spec in enumerate(matches):
            cmd = "" if (is_emoji_row and idx != sel_idx) else get_primary_slash_cmd(spec, ui_lang)
            fm = fm_sel if idx == sel_idx else fm_norm
            ew = float(fm.horizontalAdvance(spec.emoji))
            cw = float(fm.horizontalAdvance(f" {cmd}")) if cmd else 0.0
            slot_w = ew + cw + item_pad_x * 2.0
            entries.append((idx, spec.emoji, cmd, ew, cw, slot_w))

        max_bar_w = max(200.0, float(self.width() - 36))
        avail_items_w = max_bar_w - outer_pad_x * 2.0 - tab_gap - tab_w

        # Якщо підказка ширша за оверлей — беремо вікно елементів навколо обраного тегу
        lo = sel_idx
        hi = sel_idx
        used_w = entries[sel_idx][5]
        while True:
            grew = False
            if hi + 1 < len(entries):
                cand = used_w + item_gap + entries[hi + 1][5]
                if cand <= avail_items_w:
                    hi += 1
                    used_w = cand
                    grew = True
            if lo - 1 >= 0:
                cand = used_w + item_gap + entries[lo - 1][5]
                if cand <= avail_items_w:
                    lo -= 1
                    used_w = cand
                    grew = True
            if not grew:
                break

        visible_entries = entries[lo : hi + 1]
        total_w = outer_pad_x * 2.0 + used_w + tab_gap + tab_w
        bar_h = max(24.0, float(fm_sel.height()) + (10.0 if is_pill else 6.0))
        bar_y = max(3.0, anchor_y - bar_h - (7.0 if is_pill else 5.0))
        bar_x = (self.width() - total_w) / 2.0
        bar_rect = QRectF(bar_x, bar_y, total_w, bar_h)

        if is_pill:
            pill_bg = QColor(bg_color_hex)
            eff_bg_alpha = max(145, min(225, int(195 * (0.45 + 0.55 * (bg_opacity / 100.0)))))
            pill_bg.setAlpha(int(eff_bg_alpha * alpha_scale))
            pill_brd = QColor(border_color_hex)
            pill_brd.setAlpha(int(135 * alpha_scale))
            painter.setBrush(pill_bg)
            painter.setPen(QPen(pill_brd, 1.1))
            painter.drawRoundedRect(bar_rect, bar_h / 2.0, bar_h / 2.0)

        baseline_y = bar_y + (bar_h - float(fm_sel.height())) / 2.0 + float(fm_sel.ascent())
        cur_x = bar_x + outer_pad_x

        for idx, emoji_s, cmd_s, ew, _cw, slot_w in visible_entries:
            is_sel = idx == sel_idx
            item_font = font_sel if is_sel else font_norm
            fm = fm_sel if is_sel else fm_norm
            cmd_with_sp = f" {cmd_s}"

            if is_pill:
                if is_sel:
                    chip_rect = QRectF(cur_x, bar_y + 3.0, slot_w, bar_h - 6.0)
                    chip_bg = QColor(border_color_hex)
                    chip_bg.setAlpha(int(90 * alpha_scale))
                    chip_brd = QColor(border_color_hex)
                    chip_brd.setAlpha(int(165 * alpha_scale))
                    painter.setBrush(chip_bg)
                    painter.setPen(QPen(chip_brd, 0.9))
                    painter.drawRoundedRect(chip_rect, (bar_h - 6.0) / 2.0, (bar_h - 6.0) / 2.0)

                fg = QColor(255, 255, 255, int(250 * alpha_scale)) if is_sel else QColor(215, 223, 236, int(208 * alpha_scale))
                painter.setFont(item_font)
                painter.setPen(fg)
                painter.drawText(QPointF(cur_x + item_pad_x, baseline_y), emoji_s)
                painter.drawText(QPointF(cur_x + item_pad_x + ew, baseline_y), cmd_with_sp)
            else:
                # Варіант A: емодзі малюється один раз (чітко в кольорі), а команда `/...` — з темною обводкою
                painter.setFont(item_font)
                painter.setPen(QColor(255, 255, 255, int(255 * alpha_scale)))
                painter.drawText(QPointF(cur_x + item_pad_x, baseline_y), emoji_s)

                if cmd_s:
                    fg = QColor(255, 255, 255, int(252 * alpha_scale)) if is_sel else QColor(226, 232, 240, int(196 * alpha_scale))
                    self._draw_outlined_text(
                        painter, cur_x + item_pad_x + ew, baseline_y, cmd_with_sp, item_font, fg, alpha_scale, outline_r=1.35
                    )

                if is_sel:
                    sp_w = float(fm.horizontalAdvance(" "))
                    ul_x = cur_x + item_pad_x + ew + sp_w
                    ul_w = max(14.0, float(fm.horizontalAdvance(cmd_s)))
                    ul_y = baseline_y + 3.0
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor(0, 0, 0, int(175 * alpha_scale)))
                    painter.drawRoundedRect(QRectF(ul_x - 1.0, ul_y - 0.5, ul_w + 2.0, 3.0), 1.5, 1.5)
                    acc = QColor(border_color_hex)
                    acc.setAlpha(int(235 * alpha_scale))
                    painter.setBrush(acc)
                    painter.drawRoundedRect(QRectF(ul_x, ul_y, ul_w, 2.0), 1.0, 1.0)

            cur_x += slot_w + item_gap

        # Мітка `Tab` праворуч
        tab_x = bar_x + total_w - outer_pad_x - tab_w
        if is_pill:
            key_h = 16.0
            key_rect = QRectF(tab_x, bar_y + (bar_h - key_h) / 2.0, tab_w, key_h)
            painter.setBrush(QColor(255, 255, 255, int(22 * alpha_scale)))
            painter.setPen(QPen(QColor(255, 255, 255, int(55 * alpha_scale)), 0.8))
            painter.drawRoundedRect(key_rect, 4.5, 4.5)
            painter.setFont(font_tab)
            painter.setPen(QColor(196, 181, 253, int(230 * alpha_scale)))
            painter.drawText(key_rect, Qt.AlignmentFlag.AlignCenter, tab_label)
        else:
            tab_baseline = bar_y + (bar_h - float(fm_tab.height())) / 2.0 + float(fm_tab.ascent())
            self._draw_outlined_text(
                painter,
                tab_x + 2.0,
                tab_baseline,
                tab_label,
                font_tab,
                QColor(196, 181, 253, int(215 * alpha_scale)),
                alpha_scale,
                outline_r=1.2,
            )

        return bar_rect

    def _paint_tag_mode_toast(
        self,
        painter: QPainter,
        anchor_y: float,
        font_family: str,
        text_opacity: int,
        bg_color_hex: str,
        border_color_hex: str,
    ) -> QRectF:
        """Коротке спливаюче повідомлення на 1.8 с при перемиканні Numpad Ins (Варіант A ↔ Варіант B)."""
        if not self._tag_mode_toast_text:
            return QRectF()
        alpha_scale = max(0.35, min(1.0, text_opacity / 100.0))
        t_font = QFont(font_family, 10, QFont.Weight.DemiBold)
        fm = QFontMetrics(t_font)
        tw = float(fm.horizontalAdvance(self._tag_mode_toast_text))
        pill_w = tw + 18.0
        pill_h = max(20.0, float(fm.height()) + 6.0)
        pill_x = (self.width() - pill_w) / 2.0
        pill_y = max(2.0, anchor_y - pill_h - 5.0)
        rect = QRectF(pill_x, pill_y, pill_w, pill_h)

        bg = QColor(bg_color_hex)
        bg.setAlpha(int(205 * alpha_scale))
        brd = QColor(border_color_hex)
        brd.setAlpha(int(175 * alpha_scale))
        painter.setBrush(bg)
        painter.setPen(QPen(brd, 1.0))
        painter.drawRoundedRect(rect, pill_h / 2.0, pill_h / 2.0)

        painter.setFont(t_font)
        painter.setPen(QColor(226, 232, 240, int(245 * alpha_scale)))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._tag_mode_toast_text)
        return rect

    def _paint_subtle_ready_indicator(
        self,
        painter: QPainter,
        text_opacity: int,
        accent_hex: str,
    ) -> None:
        """
        Малює ледь помітний міні-овал (22x8 px) без фону й без миготіння,
        щоб було зрозуміло, що режим набору увімкнувся.
        """
        oval_w = 22.0
        oval_h = 8.0
        px = (self.width() - oval_w) / 2.0
        py = self.height() - oval_h - 17.0

        alpha_scale = text_opacity / 100.0
        shadow_col = QColor(0, 0, 0, int(90 * alpha_scale))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(shadow_col)
        painter.drawEllipse(QRectF(px - 1.0, py - 1.0, oval_w + 2.0, oval_h + 2.0))

        oval_col = QColor(accent_hex)
        oval_col.setAlpha(int(135 * alpha_scale))
        painter.setBrush(oval_col)
        painter.drawEllipse(QRectF(px, py, oval_w, oval_h))

    @staticmethod
    def _wrap_text_lines(text: str, fm: QFontMetrics, max_width: int) -> list[str]:
        words = text.split(" ")
        lines: list[str] = []
        current = ""
        for w in words:
            if fm.horizontalAdvance(w) > max_width and len(w) > 1:
                if current:
                    lines.append(current)
                    current = ""
                part = ""
                for ch in w:
                    if fm.horizontalAdvance(part + ch) <= max_width or not part:
                        part += ch
                    else:
                        lines.append(part)
                        part = ch
                current = part
                continue
            cand = f"{current} {w}".strip() if current else w
            if fm.horizontalAdvance(cand) <= max_width or not current:
                current = cand
            else:
                lines.append(current)
                current = w
        if current:
            lines.append(current)
        return lines[-3:] if len(lines) > 3 else (lines or [""])


class _ExpandingInputEdit(QPlainTextEdit):
    """Багаторядкове поле вводу, яке масштабується під розмір вікна; Enter озвучує, Shift+Enter переносить рядок."""

    returnPressed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setTabChangesFocus(True)

    def text(self) -> str:
        return self.toPlainText()

    def setText(self, value: str) -> None:
        self.setPlainText(value)

    def end(self, _mark: bool = False) -> None:
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.setTextCursor(cursor)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            win = self.window()
            if win is not None:
                win.keyPressEvent(event)
                return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            event.accept()
            self.returnPressed.emit()
            return
        super().keyPressEvent(event)


class FallbackInputWindow(QWidget):
    """
    Інтерактивне вікно 2 — активне вікно з полем вводу та кнопками керування.
    Відкривається одразу з активним фокусом у полі вводу (можна одразу писати),
    підтримує перетягування, зміну розміру за краї/кути зі збереженням розмірів та режим спілкування.
    """

    _RESIZE_MARGIN = 7

    text_submitted = Signal(str)
    cancelled = Signal()
    open_state_changed = Signal(bool)
    position_changed = Signal(int, int)
    stop_requested = Signal()
    repeat_requested = Signal()
    live_text_changed = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("SilentVoice Interactive Input")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.setMinimumSize(260, 80)
        self._prev_hwnd: int = 0
        self._continuous_mode: bool = False
        self._drag_start_pos: Optional[QPoint] = None
        self._win_start_pos: Optional[QPoint] = None
        self._resize_edges: str = ""
        self._resize_start_pos: Optional[QPoint] = None
        self._resize_start_geom: Optional[QRect] = None
        self._saved_draft_text: str = ""
        self._saved_draft_expires_at: float = 0.0
        self._last_theme_key: Optional[tuple] = None
        self._last_edit_theme_key: Optional[tuple] = None
        self._last_el_op: Optional[int] = None
        self._bg_paint_color = QColor(19, 21, 31, 199)
        self._border_paint_color = QColor(124, 92, 255, 135)

        self._size_save_timer = QTimer(self)
        self._size_save_timer.setSingleShot(True)
        self._size_save_timer.setInterval(150)
        self._size_save_timer.timeout.connect(self._persist_window_size)

        self.container = QWidget(self)
        self.container.setObjectName("fallbackContainer")
        self.container.setMouseTracking(True)

        lay = QVBoxLayout(self.container)
        lay.setContentsMargins(10, 6, 10, 8)
        lay.setSpacing(5)

        # Верхній рядок: порожня зона перетягування + компактні квадратні іконки керування
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(5)

        self.top_lbl = QLabel("")
        self.top_lbl.setMouseTracking(True)
        self.top_lbl.setCursor(Qt.CursorShape.SizeAllCursor)
        top_row.addWidget(self.top_lbl, 1)

        def _make_top_btn(icon: str, obj_name: str, tip: str, slot) -> QPushButton:
            btn = QPushButton(icon)
            btn.setObjectName(obj_name)
            btn.setFixedSize(24, 24)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            top_row.addWidget(btn, 0, Qt.AlignmentFlag.AlignVCenter)
            return btn

        self.send_btn = _make_top_btn("🔊", "sendBtn", tr("overlay.speak_tooltip", "Озвучити написаний текст (Enter)"), self._on_submit)
        self.repeat_btn = _make_top_btn("🔁", "toolBtn", tr("overlay.repeat_tooltip", "Повторити останню озвучену фразу"), self._on_repeat_clicked)
        self.stop_btn = _make_top_btn("🛑", "toolBtn", tr("overlay.stop_tooltip", "Миттєво зупинити поточну озвучку"), self._on_stop_clicked)
        self.close_btn = _make_top_btn("✕", "closeBtn", tr("overlay.close_tooltip", "Закрити вікно (Esc)"), self._on_close_clicked)

        lay.addLayout(top_row)

        # Основна область: багаторядкове масштабоване поле вводу на всю ширину й висоту
        self.input_edit = _ExpandingInputEdit()
        self.input_edit.returnPressed.connect(self._on_submit)
        self.input_edit.textChanged.connect(lambda: self.live_text_changed.emit(self.input_edit.text()))
        lay.addWidget(self.input_edit, 1)

        root_lay = QVBoxLayout(self)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.addWidget(self.container)

        w = max(self.minimumWidth(), int(config.get("overlay_input_width", 700) or 700))
        h = max(self.minimumHeight(), int(config.get("overlay_input_height", 110) or 110))
        self.resize(w, h)
        # Статичні стилі кнопок і контейнера встановлюються один раз, щоб повзунки вигляду не перепарсювали QSS усього дерева
        self.container.setStyleSheet(
            """
            QWidget#fallbackContainer {
                background: transparent;
                border: none;
            }
            QLabel {
                color: rgba(203, 213, 225, 210);
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton#sendBtn {
                background-color: rgba(99, 91, 255, 170);
                color: #FFFFFF;
                border: 1px solid rgba(129, 123, 255, 150);
                border-radius: 6px;
                padding: 0px;
                font-size: 12px;
                font-weight: 700;
                text-align: center;
            }
            QPushButton#sendBtn:hover {
                background-color: rgba(117, 110, 255, 220);
            }
            QPushButton#toolBtn {
                background-color: rgba(46, 53, 75, 150);
                color: #E2E8F0;
                border: 1px solid rgba(69, 80, 115, 130);
                border-radius: 6px;
                padding: 0px;
                font-size: 12px;
                font-weight: 600;
                text-align: center;
            }
            QPushButton#toolBtn:hover {
                background-color: rgba(58, 67, 94, 210);
                border-color: rgba(124, 92, 255, 200);
                color: #FFFFFF;
            }
            QPushButton#closeBtn {
                background-color: rgba(58, 36, 48, 150);
                color: #FCA5A5;
                border: 1px solid rgba(110, 50, 68, 130);
                border-radius: 6px;
                padding: 0px 0px 1px 0px;
                font-size: 12px;
                font-weight: 700;
                text-align: center;
            }
            QPushButton#closeBtn:hover {
                background-color: rgba(153, 27, 27, 220);
                color: #FFFFFF;
            }
            """
        )
        self.apply_theme_from_config()
        self.sync_draggable_cursor()
        self.container.installEventFilter(self)
        self.top_lbl.installEventFilter(self)

    @staticmethod
    def _set_widget_opacity(widget: QWidget, opacity: float) -> None:
        eff = widget.graphicsEffect()
        if not isinstance(eff, QGraphicsOpacityEffect):
            eff = QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(eff)
        eff.setOpacity(opacity)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
        painter.setBrush(self._bg_paint_color)
        painter.setPen(QPen(self._border_paint_color, 2.0))
        painter.drawRoundedRect(rect, 12.0, 12.0)
        painter.end()

    def apply_theme_from_config(self) -> None:
        bg_hex = str(config.get("overlay_bg_color", "#13151F"))
        border_hex = str(config.get("overlay_border_color", "#7C5CFF"))
        text_hex = str(config.get("overlay_text_color", "#FFFFFF"))
        font_sz = max(14, min(28, int(config.get("overlay_font_size", 18))))
        bg_op = max(0, min(100, int(config.get("overlay_bg_opacity", 78))))
        text_op = max(20, min(100, int(config.get("overlay_text_opacity", 100))))
        el_op = max(15, min(100, int(config.get("overlay_elements_opacity", 30))))
        theme_key = (bg_hex, border_hex, text_hex, font_sz, bg_op, text_op, el_op)
        if theme_key == self._last_theme_key:
            return
        self._last_theme_key = theme_key
        el_alpha = el_op / 100.0

        bg_col = QColor(bg_hex)
        bg_col.setAlpha(int(255 * bg_op / 100.0))
        brd_col = QColor(border_hex)
        brd_col.setAlpha(int(180 * el_alpha))
        if bg_col != self._bg_paint_color or brd_col != self._border_paint_color:
            self._bg_paint_color = bg_col
            self._border_paint_color = brd_col
            self.update()

        edit_key = (text_hex, font_sz, text_op, el_op)
        if edit_key != self._last_edit_theme_key:
            self._last_edit_theme_key = edit_key
            txt_col = QColor(text_hex)
            txt_alpha = int(255 * (text_op / 100.0) * el_alpha)
            rgba_txt = f"rgba({txt_col.red()}, {txt_col.green()}, {txt_col.blue()}, {txt_alpha})"
            edit_bg_alpha = int(145 * el_alpha)
            edit_brd_alpha = int(115 * el_alpha)
            edit_focus_alpha = int(185 * el_alpha)
            self.input_edit.setStyleSheet(
                f"""
                QPlainTextEdit {{
                    background-color: rgba(20, 24, 38, {edit_bg_alpha});
                    color: {rgba_txt};
                    border: 1px solid rgba(124, 92, 255, {edit_brd_alpha});
                    border-radius: 8px;
                    padding: 5px 10px;
                    font-size: {font_sz}px;
                    font-weight: 600;
                }}
                QPlainTextEdit:focus {{
                    border: 1.5px solid rgba(138, 132, 255, {edit_focus_alpha});
                }}
                """
            )

        if self.input_edit.graphicsEffect() is not None:
            self.input_edit.setGraphicsEffect(None)
        if el_op != self._last_el_op:
            self._last_el_op = el_op
            for widget in (self.send_btn, self.repeat_btn, self.stop_btn, self.close_btn):
                self._set_widget_opacity(widget, el_alpha)
        self._update_continuous_ui()

    def retranslate_ui(self) -> None:
        """Оновлює всі тексти і підказки інтерактивного вікна вводу при зміні мови інтерфейсу на льоту."""
        self.send_btn.setText("🔊")
        self.send_btn.setToolTip(tr("overlay.speak_tooltip", "Озвучити написаний текст (Enter)"))
        self.repeat_btn.setText("🔁")
        self.repeat_btn.setToolTip(tr("overlay.repeat_tooltip", "Повторити останню озвучену фразу"))
        self.stop_btn.setText("🛑")
        self.stop_btn.setToolTip(tr("overlay.stop_tooltip", "Миттєво зупинити поточну озвучку"))
        self.close_btn.setText("✕")
        self.close_btn.setToolTip(tr("overlay.close_tooltip", "Закрити вікно (Esc)"))
        self.input_edit.setPlaceholderText("")
        self._update_continuous_ui()

    def set_continuous_mode(self, enabled: bool) -> None:
        self._continuous_mode = enabled
        self._update_continuous_ui()

    def _update_continuous_ui(self) -> None:
        self.top_lbl.setText("")

    def _on_repeat_clicked(self) -> None:
        self.repeat_requested.emit()
        self.input_edit.setFocus()

    def _on_stop_clicked(self) -> None:
        self.stop_requested.emit()
        self.input_edit.setFocus()

    def _on_close_clicked(self) -> None:
        self.close_and_restore_focus()
        self.cancelled.emit()

    def reposition_only(self) -> None:
        """Оновлює позицію вікна за збереженим екраном і overlay_x_offset/overlay_y_offset без крадіжки фокусу чи стирання тексту."""
        screen = resolve_screen_by_name(str(config.get("overlay_screen_name", "") or ""))
        if not screen:
            return
        geom = screen.geometry()
        x_offset = int(config.get("overlay_x_offset", 0))
        y_offset = int(config.get("overlay_y_offset", 85))
        x = geom.x() + (geom.width() - self.width()) // 2 + x_offset
        y = geom.y() + geom.height() - self.height() - y_offset
        clamped_x = max(geom.x() + 5, min(geom.x() + geom.width() - self.width() - 5, x))
        clamped_y = max(geom.y() + 5, min(geom.y() + geom.height() - self.height() - 5, y))
        target_pos = QPoint(clamped_x, clamped_y)
        if self.pos() != target_pos:
            self.move(target_pos)

    def open_and_focus(self, continuous: Optional[bool] = None) -> None:
        self._prev_hwnd = get_foreground_hwnd()
        if continuous is not None:
            self._continuous_mode = continuous
        else:
            self._continuous_mode = bool(config.get("continuous_chat_mode", False))
        w = max(self.minimumWidth(), int(config.get("overlay_input_width", 700) or 700))
        h = max(self.minimumHeight(), int(config.get("overlay_input_height", 110) or 110))
        if self.width() != w or self.height() != h:
            self.resize(w, h)
        self.apply_theme_from_config()
        self.sync_draggable_cursor()
        self.reposition_only()

        self.input_edit.clear()
        draft = ""
        if bool(config.get("save_draft_on_close", True)) and self._saved_draft_text and time.monotonic() <= self._saved_draft_expires_at:
            draft = self._saved_draft_text
        self._saved_draft_text = ""
        self._saved_draft_expires_at = 0.0
        if draft:
            self.input_edit.setText(draft)
            self.input_edit.end(False)
        if not self.isVisible():
            self.setWindowOpacity(0.0)
            self.show()
            self.repaint()
            self.setWindowOpacity(1.0)
        else:
            self.repaint()
        hwnd = int(self.winId())
        set_window_capture_excluded(hwnd, bool(config.get("hide_from_capture", False)))
        restore_foreground_hwnd(hwnd)
        self.raise_()
        self.activateWindow()
        self.input_edit.setFocus()
        QTimer.singleShot(25, lambda: (self.activateWindow(), self.input_edit.setFocus()))
        self.open_state_changed.emit(True)

    def update_capture_affinity(self) -> None:
        if self.isVisible():
            set_window_capture_excluded(int(self.winId()), bool(config.get("hide_from_capture", False)))

    def close_and_restore_focus(self, save_draft: bool = True) -> None:
        self._continuous_mode = False
        if self._size_save_timer.isActive():
            self._size_save_timer.stop()
        self._persist_window_size()
        cur_txt = self.input_edit.text()
        if save_draft and bool(config.get("save_draft_on_close", True)) and cur_txt.strip():
            ttl = max(1, min(300, int(config.get("draft_keep_ttl_sec", 10) or 10)))
            self._saved_draft_text = cur_txt
            self._saved_draft_expires_at = time.monotonic() + float(ttl)
        else:
            self._saved_draft_text = ""
            self._saved_draft_expires_at = 0.0
        self.input_edit.clear()
        if self.isVisible():
            self.repaint()
        self.hide()
        self.open_state_changed.emit(False)
        if self._prev_hwnd:
            QTimer.singleShot(30, lambda: restore_foreground_hwnd(self._prev_hwnd))

    def _on_submit(self) -> None:
        text = self.input_edit.text().strip()
        is_cont = self._continuous_mode or bool(config.get("continuous_chat_mode", False))
        if is_cont:
            self._saved_draft_text = ""
            self._saved_draft_expires_at = 0.0
            self.input_edit.clear()
            self.input_edit.setFocus()
            if text:
                self.text_submitted.emit(text)
            return

        self.close_and_restore_focus(save_draft=False)
        if text:
            self.text_submitted.emit(text)
        else:
            self.cancelled.emit()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close_and_restore_focus()
            self.cancelled.emit()
            return
        super().keyPressEvent(event)

    # ── Перетягування та зміна розміру вікна зі збереженням позиції та габаритів ──

    def _persist_window_size(self) -> None:
        config.update({
            "overlay_input_width": int(self.width()),
            "overlay_input_height": int(self.height()),
        })

    def _emit_position_from_current_pos(self) -> None:
        screen_name, x_off, y_off = calc_screen_relative_offsets(self.geometry())
        config.update(
            {
                "overlay_screen_name": screen_name,
                "overlay_x_offset": int(x_off),
                "overlay_y_offset": int(y_off),
            },
            auto_save=False,
        )
        self.position_changed.emit(int(x_off), int(y_off))

    def _hit_test_edges(self, local_pos: QPoint) -> str:
        m = self._RESIZE_MARGIN
        edges = ""
        if local_pos.y() <= m:
            edges += "t"
        elif local_pos.y() >= self.height() - m:
            edges += "b"
        if local_pos.x() <= m:
            edges += "l"
        elif local_pos.x() >= self.width() - m:
            edges += "r"
        return edges

    @staticmethod
    def _cursor_for_edges(edges: str) -> Optional[Qt.CursorShape]:
        if edges in ("tl", "br"):
            return Qt.CursorShape.SizeFDiagCursor
        if edges in ("tr", "bl"):
            return Qt.CursorShape.SizeBDiagCursor
        if edges in ("l", "r"):
            return Qt.CursorShape.SizeHorCursor
        if edges in ("t", "b"):
            return Qt.CursorShape.SizeVerCursor
        return None

    def sync_draggable_cursor(self) -> None:
        draggable = bool(config.get("overlay_draggable", False))
        self.top_lbl.setCursor(Qt.CursorShape.SizeAllCursor if draggable else Qt.CursorShape.ArrowCursor)

    def _update_hover_cursor(self, global_pos: QPoint) -> None:
        edges = self._hit_test_edges(self.mapFromGlobal(global_pos))
        cur = self._cursor_for_edges(edges)
        if cur is not None:
            self.setCursor(cur)
            self.container.setCursor(cur)
            self.top_lbl.setCursor(cur)
        else:
            self.unsetCursor()
            self.container.unsetCursor()
            self.sync_draggable_cursor()

    def eventFilter(self, watched: object, event: QEvent) -> bool:
        if watched in (getattr(self, "container", None), getattr(self, "top_lbl", None)) and isinstance(event, QMouseEvent):
            etype = event.type()
            if etype == QEvent.Type.MouseMove and not self._resize_edges and self._drag_start_pos is None:
                self._update_hover_cursor(event.globalPosition().toPoint())
            elif etype == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                gpos = event.globalPosition().toPoint()
                edges = self._hit_test_edges(self.mapFromGlobal(gpos))
                if edges:
                    self._resize_edges = edges
                    self._resize_start_pos = gpos
                    self._resize_start_geom = self.geometry()
                    return True
        return super().eventFilter(watched, event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._size_save_timer.start()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            gpos = event.globalPosition().toPoint()
            edges = self._hit_test_edges(self.mapFromGlobal(gpos))
            if edges:
                self._resize_edges = edges
                self._resize_start_pos = gpos
                self._resize_start_geom = self.geometry()
            elif bool(config.get("overlay_draggable", False)):
                self._drag_start_pos = gpos
                self._win_start_pos = self.pos()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        gpos = event.globalPosition().toPoint()
        if self._resize_edges and self._resize_start_pos is not None and self._resize_start_geom is not None:
            delta = gpos - self._resize_start_pos
            g = QRect(self._resize_start_geom)
            min_w, min_h = self.minimumWidth(), self.minimumHeight()
            if "l" in self._resize_edges:
                new_left = min(g.right() - min_w + 1, g.left() + delta.x())
                g.setLeft(new_left)
            if "r" in self._resize_edges:
                new_right = max(g.left() + min_w - 1, g.right() + delta.x())
                g.setRight(new_right)
            if "t" in self._resize_edges:
                new_top = min(g.bottom() - min_h + 1, g.top() + delta.y())
                g.setTop(new_top)
            if "b" in self._resize_edges:
                new_bottom = max(g.top() + min_h - 1, g.bottom() + delta.y())
                g.setBottom(new_bottom)
            self.setGeometry(g)
            self._emit_position_from_current_pos()
            event.accept()
            return
        if self._drag_start_pos is not None and self._win_start_pos is not None:
            delta = gpos - self._drag_start_pos
            self.move(self._win_start_pos + delta)
            self._emit_position_from_current_pos()
            event.accept()
            return
        self._update_hover_cursor(gpos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self._resize_edges:
                if self._size_save_timer.isActive():
                    self._size_save_timer.stop()
                self._persist_window_size()
            elif self._drag_start_pos is not None:
                config.schedule_save(0.15)
            self._resize_edges = ""
            self._resize_start_pos = None
            self._resize_start_geom = None
            self._drag_start_pos = None
            self._win_start_pos = None
            event.accept()
            return
        super().mouseReleaseEvent(event)
