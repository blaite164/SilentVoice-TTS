"""
ui/overlay_window.py — Екранний оверлей поверх ігор та програм:
- За замовчуванням: «Без фону (тільки текст)» знизу по центру (або там, куди його перетягнув користувач)
- При виклику хоткею (поки текст ще порожній): показує лише ледь помітний міні-індикатор «• UA ▌», щоб людина бачила, що набір увімкнувся
- Під час вводу тексту: красивий текст із векторною обводкою QPainterPath та тінню без жодного фону (або з плашкою за бажанням)
- Вільне перетягування мишкою під час налаштування позиції + 100% click-through без втрати фокусу в іграх (WS_EX_TOPMOST, WS_EX_TRANSPARENT, WS_EX_NOACTIVATE)
- Повна невидимість для OBS / Discord Screen Share через SetWindowDisplayAffinity (WDA_EXCLUDEFROMCAPTURE)
"""

from __future__ import annotations

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


class ScreenOverlayWindow(QWidget):
    """
    Click-through оверлей, який ніколи не забирає фокус у ігор.
    Коли користувач вмикає набір — з'являється лише ледь помітний індикатор,
    а при наборі — чистий красивий текст без фону (або з рамкою, якщо обрано в налаштуваннях).
    Підтримує перетягування мишкою в режимі налаштування позиції.
    """

    position_changed = Signal(int, int)       # (x_offset, y_offset)
    move_mode_changed = Signal(bool)          # Чи активний режим перетягування мишкою

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
        self._last_mask_rect: QRect = QRect()
        self._text_cache_key: Optional[tuple] = None
        self._text_cache_img: Optional[QImage] = None

        self._cursor_timer = QTimer(self)
        self._cursor_timer.setInterval(500)
        self._cursor_timer.timeout.connect(self._toggle_cursor)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide_overlay)

        self.resize(960, 155)
        self.reposition()

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
        h = max(155, int(font_size * 5.2) + 48)
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
        self._mode = "typing"
        self._text = text
        self._lang_code = lang_code
        if was_visible_typing:
            # Асинхронне коалесцоване оновлення без Win32-рефлагів та синхронного repaint на кожну клавішу
            self.update()
        else:
            self.reposition()
            self._show_cleanly()

    def show_speaking(self, text: str, duration_sec: float = 2.5) -> None:
        if not config.get("overlay_enabled", True):
            return
        if self._interactive_move:
            self.set_interactive_move(False)
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
        self._cursor_timer.stop()
        self._hide_timer.stop()
        self._mode = "idle"
        self._text = ""
        self._last_box_rect = QRectF()
        self._sync_drag_mask()
        if self.isVisible():
            self.repaint()
        self.hide()

    def _sync_drag_mask(self) -> None:
        """Обмежує область захоплення мишки виключно самою плашкою, а не порожнім фоном навколо."""
        if self.is_draggable_now and self.isVisible() and not self._last_box_rect.isEmpty():
            target = self._last_box_rect.toAlignedRect().adjusted(-3, -3, 3, 3)
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

                painter.end()
                target_mask = pill_rect.toAlignedRect().adjusted(-3, -3, 3, 3)
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
                painter.end()
                target_mask = hit_rect.toAlignedRect().adjusted(-3, -3, 3, 3)
                if target_mask != self._last_mask_rect:
                    QTimer.singleShot(0, self._sync_drag_mask)
                return
            self._last_box_rect = QRectF()
            self._paint_subtle_ready_indicator(painter, text_opacity, border_color_hex)
            painter.end()
            if not self._last_mask_rect.isEmpty():
                QTimer.singleShot(0, self._sync_drag_mask)
            return

        display_text = self._text
        # Для довгого тексту беремо лише видимий хвіст, щоб розбиття та рендер завжди виконувались за O(1)
        if len(display_text) > 320:
            tail = display_text[-320:]
            sp = tail.find(" ")
            if 0 <= sp < 60:
                tail = tail[sp + 1 :]
            display_text = "… " + tail

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

        if self.is_draggable_now:
            target_mask = box_rect.toAlignedRect().adjusted(-3, -3, 3, 3)
            if target_mask != self._last_mask_rect:
                QTimer.singleShot(0, self._sync_drag_mask)
        elif not self._last_mask_rect.isEmpty():
            QTimer.singleShot(0, self._sync_drag_mask)

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

        painter.end()

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
