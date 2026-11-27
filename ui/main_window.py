"""
ui/main_window.py — Головне вікно додатку SilentVoice TTS (PySide6):
- Комфортна сіро-синя палітра (не занадто темна, приємна для очей)
- Компактна панель навігації СПРАВА зі справжніми об'ємними кнопками та підказками
- Перша вкладка при вході — «Голос та Озвучка» (налаштування для нового користувача)
- Круглі кнопки підказок (!) біля кожного розділу та налаштування для простих пояснень без термінів
"""

from __future__ import annotations

import os
import json
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Tuple

import requests

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QCursor,
    QDesktopServices,
    QFont,
    QFontMetrics,
    QGuiApplication,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)

from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSpinBox,
    QApplication,
    QBoxLayout,
    QCheckBox,
    QColorDialog,
    QComboBox as QtComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox as QtDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QLayout,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox as QtSpinBox,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QSystemTrayIcon,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.ai_corrector import (
    AI_PROVIDERS,
    fetch_ai_provider_models,
    get_ai_item_title,
    normalize_ollama_model_name,
    ollama_model_family_key,
)
from core.audio_router import (
    audio_router,
    check_vbcable_installed,
    download_and_install_vbcable,
    get_core_audio_active_all_device_names,
    get_input_devices,
    get_output_devices,
)
from core.config import BASE_DIR, BUNDLE_DIR, config
from core.settings_transfer import (
    CATEGORY_KEYS as TRANSFER_CATEGORIES,
    apply_import as apply_settings_import,
    build_export as build_settings_export,
    read_import_file as read_settings_import,
    write_export_file as write_settings_export,
)
from core.local_llm import (
    start_llama_cpp_server,
    start_lmstudio_model,
    stop_managed_llama_server,
    unload_saved_local_models,
)
from core.i18n import get_locales_dir, get_ui_language, list_available_ui_languages, set_ui_language, tr
from core.obs_server import obs_server
from core.proxy_helper import ensure_socks5_proxy_ready, invalidate_socks5_cache
from core.tts_manager import (
    ENGINE_SPECS,
    FFMPEG_ENGINE_ID,
    ONLINE_PROVIDERS,
    _resolve_dsp_key_and_defaults,
    get_engine_dsp_settings,
    is_ffmpeg_installed,
    register_custom_clone_sample,
    save_engine_dsp_settings,
    tts_manager,
)
from core.win_api import (
    KeyboardHookManager,
    get_discrete_gpu_vram_gb,
    has_hardware_gpu,
    is_autostart_enabled,
    is_running_as_admin,
    restart_as_admin,
    set_window_capture_excluded,
    set_windows_autostart,
)
from ui.dialogs import (
    AddSileroVoiceDialog,
    AICorrectorTestDialog,
    AISettingsDialog,
    ApiGuideDialog,
    AppUpdateDialog,
    DonateDialog,
    ElidedLabel,
    ErrorPopupDialog,
    FFmpegInstallDialog,
    HelpButton,
    HotkeyCaptureDialog,
    LocalModelDownloadNoticeDialog,
    LocalModelsDialog,
    LanguageConfirmDialog,
    ObsAnimationSettingsDialog,
    OmniVoiceSettingsDialog,
    ProxySettingsDialog,
    QuickPhraseDialog,
    Qwen3QuantChoiceDialog,
    SettingsTransferDialog,
    TelegramAuthDialog,
    VoiceSamplesDialog,
    get_check_svg_qss_url,
    run_in_background,
    toggle_collapsible_widget,
)

from ui.overlay_window import FallbackInputWindow, ScreenOverlayWindow, find_screen_for_rect

OLLAMA_RECOMMENDED_MODELS = [
    "gemma3:4b",
    "qwen2.5:3b-instruct",
    "qwen3.5:4b",
    "qwen3.5:9b",
]

DEFAULT_WINDOW_WIDTH_RATIO = 1247.0 / 1920.0


def calculate_default_window_width(screen_width: int) -> int:
    """Обчислює базову ширину вікна пропорційно до ширини екрана (1247 при 1920)."""
    return max(540, round(max(1, int(screen_width)) * DEFAULT_WINDOW_WIDTH_RATIO))


MAIN_STYLESHEET = """
QMainWindow, QWidget#centralRoot {
    background-color: #1E2230;
    color: #F8FAFC;
    font-family: 'Segoe UI', system-ui, sans-serif;
}
QScrollArea {
    background: transparent;
    border: none;
}
QWidget#scrollContent {
    background: transparent;
}
/* Права панель навігації (компактна, зі справжніми кнопками без нижньої тіні) */
QFrame#rightSidebar {
    background-color: #252A3B;
    border-left: 1px solid #363E57;
}
QPushButton.navBtn {
    background-color: #2E354B;
    color: #E2E8F0;
    border: 1px solid #414B69;
    border-radius: 10px;
    padding: 9px 11px;
    text-align: left;
    font-size: 13px;
    font-weight: 600;
}
QPushButton.navBtn:hover {
    background-color: #38415C;
    border-color: #6366F1;
    color: #FFFFFF;
}
QPushButton.navBtn[active="true"] {
    background-color: #635BFF;
    color: #FFFFFF;
    border: 1px solid #8A84FF;
    font-weight: 700;
}
/* Картки */
QFrame.card {
    background-color: #262B3D;
    border: 1px solid #373F59;
    border-radius: 12px;
}
QFrame.cardHighlight {
    background-color: #2A3047;
    border: 1px solid #55589C;
    border-radius: 12px;
}
QLabel.sectionTitle {
    color: #FFFFFF;
    font-size: 19px;
    font-weight: 700;
}
QLabel.sectionSub {
    color: #CBD5E1;
    font-size: 13px;
}
QLabel.cardHeader {
    color: #FFFFFF;
    font-size: 14px;
    font-weight: 700;
}
QLabel.muted {
    color: #CBD5E1;
    font-size: 12px;
}
/* Поля вводу та селектори */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #1B1F2D;
    color: #FFFFFF;
    border: 1px solid #3E4766;
    border-radius: 8px;
    padding: 6px 8px;
    font-size: 13px;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #7C5CFF;
}
QComboBox QAbstractItemView {
    background-color: #262B3D;
    color: #FFFFFF;
    selection-background-color: #635BFF;
    border: 1px solid #3E4766;
    outline: none;
}
QComboBox QLineEdit {
    background-color: transparent;
    color: #F8FAFC;
    border: none;
    border-radius: 0px;
    padding: 0px 4px;
}
/* Стандартні кнопки */
QPushButton {
    background-color: #30374F;
    color: #F8FAFC;
    border: 1px solid #454F70;
    border-radius: 8px;
    padding: 7px 14px;
    font-size: 13px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #3B4461;
    border-color: #6A76A3;
}
QPushButton:disabled {
    background-color: #252A38;
    color: #7C8499;
    border: 1px solid #343B50;
}
QPushButton.primaryBtn {
    background-color: #635BFF;
    color: #FFFFFF;
    border: 1px solid #827BFF;
    font-weight: 700;
}
QPushButton.primaryBtn:hover {
    background-color: #5249F5;
}
QPushButton.successBtn {
    background-color: #10B981;
    color: #FFFFFF;
    border: 1px solid #34D399;
    font-weight: 700;
}
QPushButton.successBtn:hover {
    background-color: #059669;
}
QPushButton.dangerBtn {
    background-color: #422631;
    color: #FCA5A5;
    border: 1px solid #7F394B;
}
QPushButton.dangerBtn:hover {
    background-color: #592E3D;
}
QLabel:disabled {
    color: #64748B;
}
QLineEdit:disabled {
    background-color: #161925;
    color: #64748B;
    border-color: #2B324A;
}
QPushButton:disabled {
    background-color: #232838;
    color: #64748B;
    border-color: #323952;
}
/* Повзунки */
QSlider::groove:horizontal {
    height: 6px;
    background: #1B1F2D;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #635BFF;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #FFFFFF;
    border: 2px solid #635BFF;
    width: 15px;
    height: 15px;
    margin: -5px 0;
    border-radius: 8px;
}
/* Чекбокси */
QCheckBox {
    color: #F1F5F9;
    font-size: 13px;
    spacing: 9px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 5px;
    border: 1px solid #525E85;
    background-color: #1B1F2D;
}
QCheckBox::indicator:hover {
    border-color: #7C5CFF;
}
QCheckBox::indicator:checked {
    background-color: #635BFF;
    border-color: #827BFF;
    image: url("__CHECK_SVG__");
}
QCheckBox::indicator:disabled {
    background-color: #161A26;
    border-color: #303750;
}
QCheckBox::indicator:checked:disabled {
    background-color: #3A3866;
    border-color: #4D4B80;
    image: url("__CHECK_SVG__");
}
/* Вкладки всередині розділу Голосів */
QTabWidget::pane {
    border: 1px solid #373F59;
    border-top-left-radius: 0px;
    border-top-right-radius: 10px;
    border-bottom-left-radius: 10px;
    border-bottom-right-radius: 10px;
    background-color: #222738;
    top: -1px;
}
QTabBar::tab {
    background-color: #2A3045;
    color: #CBD5E1;
    border: 1px solid #3B4463;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 8px 14px;
    margin-right: 4px;
    font-size: 13px;
    font-weight: 600;
}
QTabBar::tab:selected {
    background-color: #635BFF;
    color: #FFFFFF;
    border-color: #827BFF;
}
QProgressBar {
    background-color: #1B1F2D;
    border: 1px solid #38415E;
    border-radius: 6px;
    text-align: center;
    color: #FFFFFF;
    font-size: 11px;
    font-weight: 700;
    height: 15px;
}
QProgressBar::chunk {
    background-color: #635BFF;
    border-radius: 5px;
}
QToolTip {
    background-color: #191D2B;
    color: #F8FAFC;
    border: 1px solid #635BFF;
    border-radius: 6px;
    padding: 6px 8px;
    font-size: 12px;
}
"""
MAIN_STYLESHEET = MAIN_STYLESHEET.replace("__CHECK_SVG__", get_check_svg_qss_url())


def _find_app_icon_path() -> Optional[str]:
    for cand in (
        BUNDLE_DIR / "assets" / "icon.ico",
        BASE_DIR / "assets" / "icon.ico",
    ):
        if cand.exists():
            return str(cand)
    return None


def create_app_icon() -> QIcon:
    """Повертає багаторозмірну іконку додатку для панелі задач Windows, вікна та трею."""
    ico_path = _find_app_icon_path()
    if ico_path:
        ico = QIcon(ico_path)
        if not ico.isNull():
            return ico

    icon = QIcon()
    for sz in (16, 20, 24, 32, 48, 64, 128, 256):
        pix = QPixmap(sz, sz)
        pix.fill(Qt.GlobalColor.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        scale = sz / 64.0
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#635BFF"))
        margin = max(1.0, 4.0 * scale)
        p.drawRoundedRect(margin, margin, sz - 2 * margin, sz - 2 * margin, 16.0 * scale, 16.0 * scale)
        pen = QPen(QColor("#FFFFFF"), max(1.5, 4.5 * scale), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        bars = [(16, 26, 38), (24, 18, 46), (32, 12, 52), (40, 20, 44), (48, 26, 38)]
        for x, y1, y2 in bars:
            p.drawLine(int(round(x * scale)), int(round(y1 * scale)), int(round(x * scale)), int(round(y2 * scale)))
        p.end()
        icon.addPixmap(pix)
    return icon


def wrap_long_tokens(text: str, chunk: int = 38) -> str:
    """Додає невидимі точки переносу у довгі суцільні слова/URL, щоб QLabel завжди переносив їх по рядках."""
    out_words: List[str] = []
    for word in text.split(" "):
        if len(word) > chunk:
            pieces = [word[i : i + chunk] for i in range(0, len(word), chunk)]
            out_words.append("\u200b".join(pieces))
        else:
            out_words.append(word)
    return " ".join(out_words)


def make_header_with_help(title_text: str, help_title: str, help_body: str) -> QWidget:
    """Створює заголовок розділу або блоку разом із круглою кнопкою підказки (!)."""
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(8)
    lbl = QLabel(title_text)
    lbl.setProperty("class", "sectionTitle")
    lay.addWidget(lbl)
    lay.addWidget(HelpButton(help_title, help_body))
    lay.addStretch(1)
    return w


GENDER_ROLE = Qt.ItemDataRole.UserRole + 10


class VoiceGenderDelegate(QStyledItemDelegate):
    """Малює елементи випадаючого списку з кольоровою позначкою статі: Ж (рожева) або М (синя)."""

    def paint(self, painter: QPainter, option, index) -> None:
        opt = option
        self.initStyleOption(opt, index)
        gender = str(index.data(GENDER_ROLE) or "")
        text = str(opt.text or "")

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        if opt.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(opt.rect, QColor("#635BFF"))
        elif opt.state & QStyle.StateFlag.State_MouseOver:
            painter.fillRect(opt.rect, QColor("#363F59"))
        else:
            painter.fillRect(opt.rect, QColor("#252B3D"))

        r = opt.rect.adjusted(8, 0, -8, 0)
        badge_w = 18 if gender in ("female", "male") else 0

        max_text_w = max(16, r.width() - (badge_w + 5 if badge_w else 0))
        painter.setPen(QColor("#F8FAFC"))
        painter.setFont(opt.font)
        fm = painter.fontMetrics()
        elided = fm.elidedText(text, Qt.TextElideMode.ElideRight, max_text_w)
        text_rect = r.adjusted(0, 0, -(badge_w + 5) if badge_w else 0, 0)
        painter.drawText(text_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), elided)

        if gender in ("female", "male"):
            letter = tr("voice.gender_f", "Ж") if gender == "female" else tr("voice.gender_m", "М")
            color = QColor("#F472B6") if gender == "female" else QColor("#38BDF8")
            b_font = QFont(opt.font)
            b_font.setBold(True)
            painter.setFont(b_font)
            painter.setPen(color)
            drawn_w = min(max_text_w, fm.horizontalAdvance(elided))
            badge_x = r.left() + drawn_w + 6
            badge_rect = r.adjusted(badge_x - r.left(), 0, 0, 0)
            painter.drawText(badge_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), letter)

        painter.restore()


class VoiceComboBox(QtComboBox):
    """Сучасний випадаючий список без стандартної сірої кнопки Windows, з підтримкою позначок 'Ж' (рожева) та 'М' (синя)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setItemDelegate(VoiceGenderDelegate(self))
        self.setMinimumHeight(33)

    def add_gender_item(self, text: str, data, gender: str = "") -> None:
        self.addItem(text, data)
        idx = self.count() - 1
        self.setItemData(idx, gender, GENDER_ROLE)

    def wheelEvent(self, event) -> None:
        event.ignore()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        rect = self.rect().adjusted(1, 1, -1, -1)
        le = self.lineEdit()
        is_focused = self.hasFocus() or (le is not None and le.hasFocus())
        border_col = QColor("#7C5CFF") if (is_focused or self.underMouse()) else QColor("#3E4766")
        painter.setBrush(QColor("#1B1F2D"))
        painter.setPen(QPen(border_col, 1.0))
        painter.drawRoundedRect(rect, 8, 8)

        is_compact = self.width() <= 64
        arrow_w = 14 if is_compact else 22

        if not self.isEditable():
            idx = self.currentIndex()
            full_text = self.currentText()
            gender = str(self.itemData(idx, GENDER_ROLE) or "") if idx >= 0 else ""

            content_rect = rect.adjusted(6 if is_compact else 10, 0, -(arrow_w + 3), 0)
            badge_w = 16 if gender in ("female", "male") else 0
            max_text_w = max(12, content_rect.width() - (badge_w + 5 if badge_w else 0))

            painter.setPen(QColor("#F8FAFC"))
            painter.setFont(self.font())
            fm = painter.fontMetrics()
            elided = fm.elidedText(full_text, Qt.TextElideMode.ElideRight, max_text_w)
            painter.drawText(content_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), elided)

            if gender in ("female", "male"):
                letter = tr("voice.gender_f", "Ж") if gender == "female" else tr("voice.gender_m", "М")
                color = QColor("#F472B6") if gender == "female" else QColor("#38BDF8")
                b_font = QFont(self.font())
                b_font.setBold(True)
                painter.setFont(b_font)
                painter.setPen(color)
                drawn_w = min(max_text_w, fm.horizontalAdvance(elided))
                badge_rect = content_rect.adjusted(drawn_w + 5, 0, 0, 0)
                painter.drawText(badge_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), letter)

        # Охайна роздільна лінія та стрілочка справа
        x_div = rect.right() - arrow_w
        if not is_compact:
            painter.setPen(QPen(QColor("#323A54"), 1.0))
            painter.drawLine(x_div, rect.top() + 5, x_div, rect.bottom() - 5)

        arrow_col = QColor("#C7D2FE") if self.underMouse() else QColor("#94A3B8")
        cx = x_div + arrow_w / 2.0
        cy = rect.top() + rect.height() / 2.0
        chev = QPainterPath()
        chev.moveTo(cx - 3.6, cy - 1.8)
        chev.lineTo(cx, cy + 2.2)
        chev.lineTo(cx + 3.6, cy - 1.8)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(arrow_col, 1.7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawPath(chev)
        painter.end()


class ModelComboBox(QtComboBox):
    """Editable model selector with a wide, clearly visible drop-down button."""

    _DROP_WIDTH = 44

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            "QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; "
            "width: 44px; border-left: 1px solid #323A54; "
            "border-top-right-radius: 8px; border-bottom-right-radius: 8px; background-color: #23283B; }"
            "QComboBox::drop-down:hover { background-color: #384263; }"
            "QComboBox::down-arrow { image: none; width: 0px; height: 0px; }"
        )

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(
            QPen(
                QColor("#F8FAFC" if self.underMouse() else "#C7D2FE"),
                2.0,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        )
        cx = self.width() - self._DROP_WIDTH / 2
        cy = self.height() / 2
        chevron = QPainterPath()
        chevron.moveTo(cx - 4.5, cy - 2.0)
        chevron.lineTo(cx, cy + 2.5)
        chevron.lineTo(cx + 4.5, cy - 2.0)
        painter.drawPath(chevron)
        painter.end()


def _paint_spinbox_arrows(widget: QWidget, hover_part: int) -> None:
    """Малює акуратні сучасні кнопки збільшення/зменшення для числових полів без застарілих 3D-рамок Windows."""
    painter = QPainter(widget)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    w = widget.width()
    h = widget.height()
    btn_w = 24
    x_div = w - btn_w - 1
    mid_y = h // 2

    # Фон області кнопок справа, щоб стандартні кнопки ніколи не проглядали
    bg_rect = QRect(x_div, 2, btn_w - 1, h - 4)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#23283B"))
    painter.drawRoundedRect(bg_rect, 6, 6)

    up_rect = QRect(x_div + 1, 2, btn_w - 3, mid_y - 2)
    down_rect = QRect(x_div + 1, mid_y, btn_w - 3, h - mid_y - 2)

    if hover_part == 1:
        painter.setBrush(QColor("#384263"))
        painter.drawRoundedRect(up_rect, 4, 4)
    elif hover_part == 2:
        painter.setBrush(QColor("#384263"))
        painter.drawRoundedRect(down_rect, 4, 4)

    painter.setPen(QPen(QColor("#38415E"), 1.0))
    painter.drawLine(x_div, 3, x_div, h - 3)
    painter.drawLine(x_div + 2, mid_y, w - 3, mid_y)

    cx = up_rect.center().x() + 0.5
    up_cy = up_rect.center().y() + 0.5
    dn_cy = down_rect.center().y() + 0.5

    up_path = QPainterPath()
    up_path.moveTo(cx - 3.5, up_cy + 1.8)
    up_path.lineTo(cx, up_cy - 1.8)
    up_path.lineTo(cx + 3.5, up_cy + 1.8)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(QColor("#FFFFFF") if hover_part == 1 else QColor("#CBD5E1"), 1.7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.drawPath(up_path)

    dn_path = QPainterPath()
    dn_path.moveTo(cx - 3.5, dn_cy - 1.8)
    dn_path.lineTo(cx, dn_cy + 1.8)
    dn_path.lineTo(cx + 3.5, dn_cy - 1.8)
    painter.setPen(QPen(QColor("#FFFFFF") if hover_part == 2 else QColor("#CBD5E1"), 1.7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.drawPath(dn_path)
    painter.end()


class ModernSpinBox(QtSpinBox):
    """Числове поле без старих 3D-кнопок Windows, з сучасними векторними стрілками ▴ / ▾."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setMouseTracking(True)
        self.setMinimumHeight(33)
        self.setMinimumWidth(92)
        self.setStyleSheet("padding-right: 26px;")
        self._hover_part = 0  # 0 = none, 1 = up, 2 = down

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        _paint_spinbox_arrows(self, self._hover_part)

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        if pos.x() >= self.width() - 25:
            part = 1 if pos.y() < self.height() / 2 else 2
        else:
            part = 0
        if part != self._hover_part:
            self._hover_part = part
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        if self._hover_part != 0:
            self._hover_part = 0
            self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and event.position().x() >= self.width() - 25:
            if event.position().y() < self.height() / 2:
                self.stepUp()
            else:
                self.stepDown()
            event.accept()
            self.update()
            return
        super().mousePressEvent(event)


class ModernDoubleSpinBox(QtDoubleSpinBox):
    """Дробове числове поле (напр. 1.00x) без старих 3D-кнопок Windows, з векторними стрілками ▴ / ▾."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setMouseTracking(True)
        self.setMinimumHeight(33)
        self.setMinimumWidth(96)
        self.setStyleSheet("padding-right: 26px;")
        self._hover_part = 0

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        _paint_spinbox_arrows(self, self._hover_part)

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        if pos.x() >= self.width() - 25:
            part = 1 if pos.y() < self.height() / 2 else 2
        else:
            part = 0
        if part != self._hover_part:
            self._hover_part = part
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        if self._hover_part != 0:
            self._hover_part = 0
            self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and event.position().x() >= self.width() - 25:
            if event.position().y() < self.height() / 2:
                self.stepUp()
            else:
                self.stepDown()
            event.accept()
            self.update()
            return
        super().mousePressEvent(event)


# Підміняємо стандартні віджети у цьому модулі на сучасні стилізовані версії
QComboBox = VoiceComboBox
QSpinBox = ModernSpinBox
QDoubleSpinBox = ModernDoubleSpinBox


class _NativeSpinBoxModernizer(QObject):
    """Перетворює вбудовані Qt-спінбокси (наприклад, у QColorDialog) на сучасні з векторними стрілками."""

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if not isinstance(obj, (QtSpinBox, QtDoubleSpinBox)):
            return False
        et = event.type()
        if et == QEvent.Type.Paint:
            obj.removeEventFilter(self)
            QApplication.sendEvent(obj, event)
            obj.installEventFilter(self)
            _paint_spinbox_arrows(obj, int(obj.property("_sv_hover") or 0))
            return True
        if et == QEvent.Type.MouseMove:
            pos = event.position()
            part = (1 if pos.y() < obj.height() / 2 else 2) if pos.x() >= obj.width() - 25 else 0
            if part != int(obj.property("_sv_hover") or 0):
                obj.setProperty("_sv_hover", part)
                obj.update()
        elif et == QEvent.Type.Leave:
            if int(obj.property("_sv_hover") or 0) != 0:
                obj.setProperty("_sv_hover", 0)
                obj.update()
        elif et == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            if event.position().x() >= obj.width() - 25:
                if event.position().y() < obj.height() / 2:
                    obj.stepUp()
                else:
                    obj.stepDown()
                obj.update()
                return True
        return False


def _modernize_child_spinboxes(root: QWidget) -> None:
    flt = _NativeSpinBoxModernizer(root)
    for sp in root.findChildren(QAbstractSpinBox):
        if isinstance(sp, (ModernSpinBox, ModernDoubleSpinBox)):
            continue
        sp.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        sp.setMouseTracking(True)
        sp.setMinimumHeight(30)
        sp.setStyleSheet("padding-right: 26px;")
        sp.installEventFilter(flt)


class _QuickPhraseButton(QPushButton):
    """Дворядокова кнопка швидкої фрази, яка скорочує текст під ширину колонки без горизонтального скролу."""

    def __init__(self, line1: str, line2: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._line1 = line1
        self._line2 = line2
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._update_elided_text()

    def sizeHint(self) -> QSize:
        fm = QFontMetrics(self.font())
        h = fm.lineSpacing() * 2 + 10
        return QSize(180, max(38, h))

    def minimumSizeHint(self) -> QSize:
        return QSize(0, self.sizeHint().height())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_elided_text()

    def _update_elided_text(self) -> None:
        avail_w = max(40, self.width() - 8) if self.width() > 20 else 220
        fm = QFontMetrics(self.font())
        el1 = fm.elidedText(self._line1, Qt.TextElideMode.ElideRight, avail_w)
        el2 = fm.elidedText(self._line2, Qt.TextElideMode.ElideRight, avail_w)
        super().setText(f"{el1}\n{el2}" if el2 else el1)


class _QuickPhraseScrollArea(QScrollArea):
    """Внутрішня область прокрутки сітки 3×9: при ≤9 рядках прокручує сторінку, при >9 — сам блок."""

    def wheelEvent(self, event) -> None:
        sb = self.verticalScrollBar()
        if self.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff or sb.maximum() <= 0:
            event.ignore()
            return
        dy = event.angleDelta().y()
        if (dy > 0 and sb.value() <= sb.minimum()) or (dy < 0 and sb.value() >= sb.maximum()):
            event.ignore()
            return
        super().wheelEvent(event)


class _NoWheelWidgetFilter(QObject):
    """Блокує зміну значення числового поля коліщатком і передає прокрутку сторінці."""

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Wheel and isinstance(obj, QWidget):
            event.ignore()
            parent = obj.parentWidget()
            while parent is not None:
                if isinstance(parent, QAbstractScrollArea):
                    QApplication.sendEvent(parent.viewport(), event)
                    break
                parent = parent.parentWidget()
            return True
        return False


def _disable_wheel_scroll(widget: QWidget) -> None:
    widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    flt = _NoWheelWidgetFilter(widget)
    widget.installEventFilter(flt)
    le = widget.findChild(QLineEdit)
    if le is not None:
        le.installEventFilter(flt)


class _VoiceDropCard(QFrame):
    """Card supporting drag-and-drop selection of a reference voice audio file."""

    _AUDIO_EXTS = (".wav", ".mp3", ".ogg", ".flac", ".m4a")

    def __init__(self, on_file_dropped, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._on_file_dropped = on_file_dropped
        self.setAcceptDrops(True)

    @classmethod
    def _first_audio_url(cls, mime_data) -> str:
        if mime_data is None or not mime_data.hasUrls():
            return ""
        for url in mime_data.urls():
            local = url.toLocalFile()
            if local and local.lower().endswith(cls._AUDIO_EXTS):
                return local
        return ""

    def dragEnterEvent(self, event) -> None:
        if self._first_audio_url(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dropEvent(self, event) -> None:
        path = self._first_audio_url(event.mimeData())
        if path:
            event.acceptProposedAction()
            self._on_file_dropped(path)
            return
        super().dropEvent(event)


class CreditsFlowLayout(QLayout):
    """Lay out credits in one left-aligned row, or two rows when space is short."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._items = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(10)

    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientation:
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        margins = self.contentsMargins()
        available = max(0, width - margins.left() - margins.right())
        sizes = [item.sizeHint() for item in self._items]
        if not sizes:
            return margins.top() + margins.bottom()
        one_row_width = sum(size.width() for size in sizes) + self.spacing() * (len(sizes) - 1)
        if available >= one_row_width:
            content_height = max(size.height() for size in sizes)
        else:
            content_height = max(sizes[0].height(), sizes[1].height()) + self.spacing() + sizes[2].height()
        return margins.top() + content_height + margins.bottom()

    def sizeHint(self) -> QSize:
        margins = self.contentsMargins()
        sizes = [item.sizeHint() for item in self._items]
        return QSize(
            margins.left() + sum(size.width() for size in sizes) + self.spacing() * max(0, len(sizes) - 1) + margins.right(),
            margins.top() + max((size.height() for size in sizes), default=0) + margins.bottom(),
        )

    def minimumSize(self) -> QSize:
        margins = self.contentsMargins()
        sizes = [item.minimumSize() for item in self._items]
        if len(sizes) >= 3:
            width = max(sizes[0].width() + sizes[1].width() + self.spacing(), sizes[2].width())
        else:
            width = sum(size.width() for size in sizes) + self.spacing() * max(0, len(sizes) - 1)
        height = max((size.height() for size in sizes), default=0)
        return QSize(margins.left() + width + margins.right(), margins.top() + height + margins.bottom())

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        if not self._items:
            return
        margins = self.contentsMargins()
        x = rect.x() + margins.left()
        y = rect.y() + margins.top()
        available = rect.width() - margins.left() - margins.right()
        sizes = [item.sizeHint() for item in self._items]
        one_row_width = sum(size.width() for size in sizes) + self.spacing() * (len(sizes) - 1)
        if available >= one_row_width:
            for item, size in zip(self._items, sizes):
                item.setGeometry(QRect(x, y, size.width(), size.height()))
                x += size.width() + self.spacing()
        else:
            first_height = max(sizes[0].height(), sizes[1].height())
            for item, size in zip(self._items[:2], sizes[:2]):
                item.setGeometry(QRect(x, y, size.width(), size.height()))
                x += size.width() + self.spacing()
            self._items[2].setGeometry(
                QRect(rect.x() + margins.left(), y + first_height + self.spacing(), sizes[2].width(), sizes[2].height())
            )


class MainWindow(QMainWindow):
    """Головне вікно програми SilentVoice з компактною правою панеллю навігації."""

    fish_voices_fetched = Signal(str, object, str, object)
    google_voices_fetched = Signal(str, object, str, object)
    provider_catalog_fetched = Signal(str, object, str, object, bool)
    ai_models_fetched = Signal(str, object, str, object, bool)
    # Ollama reports byte counts larger than Qt's signed 32-bit int range.
    ollama_pull_progress = Signal(str, str, str, object, object)
    ollama_pull_finished = Signal(str, str, str)
    ollama_server_ready = Signal(bool)
    local_llm_restore_finished = Signal()
    socks5_status_changed = Signal(bool, str)
    tg_bot_voice_detected = Signal(str)
    ollama_installer_progress = Signal(int)
    ollama_installer_finished = Signal(str, str)
    vban_health_changed = Signal(str, str, str)
    vban_test_finished = Signal(str)
    vban_ping_finished = Signal(str, str)
    twitch_status_changed = Signal(str, str)
    twitch_resolved = Signal(str, str)
    twitch_chat_event_received = Signal(object)
    twitch_moderation_received = Signal(str, str, str)
    twitch_test_finished = Signal(bool, str)
    startup_update_found = Signal(object)
    vbcable_progress = Signal(int)
    vbcable_finished = Signal(bool, str)

    def __init__(self) -> None:
        super().__init__()
        if not config.get("_ollama_qwen35_2b_default_migrated", False):
            ai_items = list(config.get("ai_items", []))
            for item in ai_items:
                if (
                    isinstance(item, dict)
                    and str(item.get("provider", "")) == "ollama"
                    and str(item.get("model", "")).strip() == "qwen3.5:4b"
                ):
                    item["model"] = "qwen3.5:2b"
            config.update({
                "ai_items": ai_items,
                "_ollama_qwen35_2b_default_migrated": True,
            })
        if not config.get("_ollama_gemma3_4b_default_migrated", False):
            ai_items = list(config.get("ai_items", []))
            for item in ai_items:
                if (
                    isinstance(item, dict)
                    and str(item.get("provider", "")) == "ollama"
                    and str(item.get("model", "")).strip() == "qwen3.5:2b"
                ):
                    item["model"] = "gemma3:4b"
            config.update({
                "ai_items": ai_items,
                "_ollama_gemma3_4b_default_migrated": True,
            })
        self._ollama_download_cancel = threading.Event()
        self._ollama_download_active = False
        self.ollama_installer_progress.connect(self._on_ollama_installer_progress)
        self.ollama_installer_finished.connect(self._on_ollama_installer_finished)
        self.vbcable_progress.connect(lambda pct: self.live_status_label.setText(f"VB-Cable: {pct}%"))
        self.vbcable_finished.connect(self._on_vbcable_done)
        self.setWindowTitle(tr("ui_084", "SilentVoice — Озвучення тексту для спілкування на ПК"))
        self.setWindowIcon(create_app_icon())
        saved_x = config.get("main_window_x")
        saved_y = config.get("main_window_y")
        try:
            saved_w = int(config.get("main_window_width", 0) or 0)
            saved_h = int(config.get("main_window_height", 0) or 0)
        except (TypeError, ValueError):
            saved_w, saved_h = 0, 0

        screen_obj = QGuiApplication.primaryScreen()
        if saved_x is not None and saved_y is not None:
            try:
                saved_rect = QRect(int(saved_x), int(saved_y), max(540, saved_w or 800), max(380, saved_h or 700))
                screen_obj = find_screen_for_rect(saved_rect) or screen_obj
            except (TypeError, ValueError):
                pass
        if screen_obj is not None:
            avail_geo = screen_obj.availableGeometry()
            screen_w = int(screen_obj.geometry().width() or avail_geo.width())
            target_w = calculate_default_window_width(screen_w)
            init_w = min(target_w, max(540, avail_geo.width() - 24))
            init_h = min(740, max(380, avail_geo.height() - 48))
        else:
            init_w, init_h = calculate_default_window_width(1920), 740
        if saved_w > 0:
            init_w = min(saved_w, max(540, avail_geo.width() - 24)) if screen_obj is not None else max(540, saved_w)
        if saved_h > 0:
            init_h = min(saved_h, max(380, avail_geo.height() - 48)) if screen_obj is not None else max(380, saved_h)
        self.resize(init_w, init_h)
        self.setMinimumSize(540, 380)
        if screen_obj is not None:
            cx = avail_geo.x() + max(0, (avail_geo.width() - init_w) // 2)
            cy = avail_geo.y()
            self.move(cx, cy)
        self.setStyleSheet(MAIN_STYLESHEET)

        self._force_quit = False
        self._engine_cards_ui: Dict[str, Dict[str, QWidget]] = {}
        self._ffmpeg_ui_bars: List[Dict[str, Any]] = []
        self._ffmpeg_prompt_open = False
        self._ffmpeg_startup_checked = False
        self._ffmpeg_last_declined_voice: Optional[Tuple[str, str]] = None
        self._download_ui_state: Dict[str, Tuple[int, str]] = {}
        self._api_debounce_timers: Dict[str, QTimer] = {}
        self._online_status_labels: Dict[str, QLabel] = {}
        self._ai_debounce_timers: Dict[str, QTimer] = {}
        self._ai_status_labels: Dict[str, QLabel] = {}
        self._ai_model_combos: Dict[str, QComboBox] = {}
        self._ollama_connections: Dict[str, bool] = {}
        self._ollama_model_pull_active: Dict[str, str] = {}
        self._ollama_start_in_progress = False
        self._local_llm_restore_in_progress = False
        self._ollama_model_buttons: Dict[str, QPushButton] = {}
        self._ollama_install_buttons: Dict[str, tuple[QPushButton, QPushButton]] = {}
        self._ollama_install_check_timer = QTimer(self)
        self._ollama_install_check_timer.setInterval(2500)
        self._ollama_install_check_timer.timeout.connect(self._refresh_all_ollama_install_buttons)
        self._ollama_install_check_timer.start()
        self._ai_test_buttons: Dict[str, QPushButton] = {}
        self._ollama_download_widgets: Dict[str, Dict[str, QWidget]] = {}
        self._socks5_checkboxes: List[QCheckBox] = []
        self._socks5_status_labels: List[QLabel] = []
        self._socks5_url_edits: List[QLineEdit] = []
        self._socks5_refresh_btns: List[QPushButton] = []
        self._silero_limits: Optional[Dict[str, str]] = None
        self._silero_current_voice: Optional[Dict[str, str]] = None
        self._silero_error: str = ""
        self._silero_connecting: bool = False

        # 1. Екранний текст, запасне вікно вводу та глобальний перехоплювач клавіш
        self.overlay = ScreenOverlayWindow()
        self.fallback_window = FallbackInputWindow()
        self.keyboard_hook = KeyboardHookManager(self)
        self.keyboard_hook.update_from_config(config.data)

        # 2. Будуємо інтерфейс (робоча зона зліва + компактна панель кнопок справа)
        self._build_ui()
        self._setup_tray()
        self._connect_signals()
        self.fish_voices_fetched.connect(self._on_fish_voices_fetched)
        self.google_voices_fetched.connect(self._on_google_voices_fetched)
        self.provider_catalog_fetched.connect(self._on_provider_catalog_fetched)
        self.ai_models_fetched.connect(self._on_ai_models_fetched)
        self.ollama_pull_progress.connect(self._on_ollama_pull_progress)
        self.ollama_pull_finished.connect(self._on_ollama_pull_finished)
        self.ollama_server_ready.connect(self._on_ollama_server_ready)
        self.local_llm_restore_finished.connect(self._on_local_llm_restore_finished)
        self.socks5_status_changed.connect(self._on_socks5_probe_done)
        if bool(config.get("use_socks5_proxy", False)):
            self._start_socks5_probe()
        QTimer.singleShot(900, self._auto_connect_silero_bot)
        QTimer.singleShot(1400, self._auto_sync_all_online_providers)
        QTimer.singleShot(1600, self._auto_sync_all_ai_providers)
        if bool(config.get("ai_spellcheck_enabled", False)):
            QTimer.singleShot(1800, self._ensure_ollama_server)
            QTimer.singleShot(2200, self._start_saved_local_llm_servers)

        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        # 3. Запускаємо хук клавіатури та сервер субтитрів OBS
        self.keyboard_hook.start()
        if config.get("obs_enabled", True):
            obs_server.start(int(config.get("obs_port", 8765)))

        self.tg_bot_voice_detected.connect(self._on_tg_bot_voice_detected)

        self._first_run_lang_prompt_active = False
        self._first_run_help_scheduled = False
        self._lang_net_retry_timer = QTimer(self)
        self._lang_net_retry_timer.setInterval(15000)
        self._lang_net_retry_timer.timeout.connect(self._check_pending_first_run_lang)
        if not bool(config.get("first_run_lang_checked", False)):
            self._lang_net_retry_timer.start()
        from core.telegram_silero import telegram_silero

        telegram_silero.set_speaker_changed_callback(self.tg_bot_voice_detected.emit)

        QTimer.singleShot(150, self._apply_window_capture_protection)
        QTimer.singleShot(160, self._apply_native_win_icon)
        QTimer.singleShot(1200, lambda: tts_manager.schedule_smart_warmup(0.3, "startup"))
        self.startup_update_found.connect(self._on_startup_update_found)
        QTimer.singleShot(700, self._check_post_update_notice)
        QTimer.singleShot(4500, self._auto_check_for_updates_on_startup)

    def eventFilter(self, watched, event) -> bool:
        """Знімає фокус із текстових та числових полів при натисканні Esc або при кліку поза полем вводу."""
        try:
            ev_type = event.type()
            if ev_type == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
                fw = QApplication.focusWidget()
                if fw is not None and fw.window() is self:
                    fw.clearFocus()
                    return True
            elif ev_type == QEvent.Type.MouseButtonPress:
                fw = QApplication.focusWidget()
                if fw is not None and fw.window() is self and isinstance(
                    fw, (QLineEdit, QTextEdit, QAbstractSpinBox, QtComboBox)
                ):
                    if isinstance(watched, QWidget) and watched.window() is self:
                        cur: QWidget | None = watched
                        clicked_inside_editable = False
                        while cur is not None and cur is not self:
                            if isinstance(cur, (QLineEdit, QTextEdit, QAbstractSpinBox, QtComboBox)):
                                clicked_inside_editable = True
                                break
                            cur = cur.parentWidget()
                        if not clicked_inside_editable:
                            fw.clearFocus()
        except RuntimeError:
            return False
        return False

    # ──────────────────────────────────────────────────────────────────
    # Побудова головного макета (Робоча зона + Компактне праве меню)
    # ──────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("centralRoot")
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ЛІВА / ЦЕНТРАЛЬНА ЧАСТИНА: відразу сторінки налаштувань (без зайвих верхніх панелей)
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(14, 12, 8, 12)
        left_layout.setSpacing(10)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_page_tts())         # 0: 🎙️ Голос та Озвучка (відкривається першою!)
        self.pages.addWidget(self._build_page_phrases())     # 1: 💬 Швидкі фрази
        self.pages.addWidget(self._build_page_hotkeys())     # 2: ⌨️ Кнопки та Ввід
        self.pages.addWidget(self._build_page_overlay())     # 3: ✨ Текст на екрані
        self.pages.addWidget(self._build_page_obs())         # 4: 🎬 Субтитри для OBS
        self.pages.addWidget(self._build_page_audio())       # 5: 🔊 Звук та Система

        left_layout.addWidget(self.pages, 1)
        root_layout.addWidget(left_container, 1)

        # ПРАВА ПАНЕЛЬ НАВІГАЦІЇ (компактна 175px, або вузька панель кнопок-іконок при фізичній ширині < 1138px)
        sidebar = QFrame()
        sidebar.setObjectName("rightSidebar")
        sidebar.setFixedWidth(175)
        self.right_sidebar = sidebar
        self._sidebar_compact = False
        self._sidebar_short_h = False
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(8, 12, 8, 12)
        sb_layout.setSpacing(6)
        self._sb_layout = sb_layout

        # Назва програми + кнопка (!) про програму
        brand_row = QHBoxLayout()
        brand_row.setContentsMargins(4, 0, 2, 4)
        brand_title = QLabel("🎙️ SilentVoice")
        brand_title.setStyleSheet("color: #FFFFFF; font-size: 16px; font-weight: 800;")
        brand_title.setToolTip("SilentVoice")
        self._sb_brand_title = brand_title
        brand_row.addWidget(brand_title)
        brand_row.addStretch(1)
        brand_help_btn = HelpButton(
            tr("ui_085", "Як працює ця програма?"),
            tr(
                "ui_086",
                "<b>Привіт! Це SilentVoice — твій текстовий голос</b><br>"
                "Не можеш або не хочеш говорити в мікрофон? Просто друкуй, а програма озвучить усе за тебе — у дзвінках, іграх чи на стрімі. Чудово виручає і тих, кому складно або неможливо говорити голосом, і тих, хто просто хоче спілкуватися текстом.<br><br>"
                "<b>Налаштування за хвилинку:</b><br>"
                "1. <b>Обери, чим озвучувати.</b> Зазирни в <b>«🎙️ Голос та Озвучка»</b> та встанови будь-який TTS-рушій: офлайн у <b>«💻 Завантажити на ПК»</b> або онлайн у <b>«🌐 Онлайн-голоси»</b> (найпростіше почати з безкоштовного <b>Edge-TTS</b>).<br>"
                "2. <b>Підбери голос.</b> У самому верху в <b>«Ваш активний голос»</b> вибери диктора до смаку й перевір звучання.<br>"
                "3. <b>Користуйся де завгодно.</b> Натисни <b>{hk_toggle}</b> поверх будь-якої програми, набери повідомлення й відправ через <b>{hk_send}</b> — і тебе почують!<br><br>"
                "<i>💡 Порада: щоб звук йшов у Discord чи гру, увімкни <b>VB-Cable</b> у вкладці <b>«🔊 Звук та Система»</b>.</i>",
            ),
        )
        self._sb_brand_help_btn = brand_help_btn
        brand_row.addWidget(brand_help_btn)
        sb_layout.addLayout(brand_row)

        nav_items = [
            (tr("ui_087", "🎙️ Голос та Озвучка"), 0, tr("ui_088", "Вибір голосу, прослуховування прикладів та завантаження голосових моделей")),
            (tr("ui_089", "💬 Швидкі фрази"), 1, tr("ui_090", "Готові фрази («Привіт», «Дякую», «Секунду») для миттєвої озвучки в 1 клік")),
            (tr("ui_091", "⌨️ Кнопки та Ввід"), 2, tr("ui_092", "Налаштування клавіш, якими ви викликаєте набір тексту та озвучку на ПК")),
            (tr("ui_093", "✨ Текст на екран/чат"), 3, tr("ui_094", "Як виглядає текст знизу екрана, коли ви його друкуєте (розмір, колір, місце)")),
            (tr("ui_095", "🎬 Субтитри OBS"), 4, tr("ui_096", "Окремі красиві субтитри для тих, хто веде трансляції (стріми) в OBS Studio")),
            (tr("ui_097", "🔊 Звук та Система"), 5, tr("ui_098", "Налаштування мікрофона (щоб вас чули інші), навушників та автозапуску")),
        ]
        self.nav_buttons: List[QPushButton] = []
        for label, idx, tip in nav_items:
            btn = QPushButton(label)
            btn._full_text = label  # type: ignore[attr-defined]
            btn._icon_text = label.split()[0] if label.split() else label  # type: ignore[attr-defined]
            btn.setProperty("class", "navBtn")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(tip)
            btn.clicked.connect(lambda checked=False, i=idx: self.switch_page(i))
            sb_layout.addWidget(btn)
            self.nav_buttons.append(btn)

        # Кнопка «Підтримати» відразу після «Звук та Система», а під нею 2 списки поряд: Мова (2 великі букви) та Стать
        donate_label = tr("ui_099", "❤️ Підтримати")
        self.donate_btn = QPushButton(donate_label)
        self.donate_btn._full_text = donate_label  # type: ignore[attr-defined]
        self.donate_btn._icon_text = donate_label.split()[0] if donate_label.split() else "❤️"  # type: ignore[attr-defined]
        self.donate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.donate_btn.setToolTip(tr("ui_100", "Підтримати розробку проєкту"))
        self.donate_btn.setStyleSheet(
            "QPushButton { background-color: #362B4C; color: #F5D0FE; border: 1px solid #7E57C2; border-radius: 10px; padding: 8px 10px; font-size: 13px; font-weight: 700; }"
            "QPushButton:hover { background-color: #463763; border-color: #A855F7; color: #FFFFFF; }"
        )
        self.donate_btn.clicked.connect(self._open_donate_dialog)
        sb_layout.addWidget(self.donate_btn)

        self._sb_filters_container = QWidget()
        self._sb_filters_container.setStyleSheet("background: transparent;")
        filters_row = QBoxLayout(QBoxLayout.Direction.LeftToRight, self._sb_filters_container)
        filters_row.setContentsMargins(0, 0, 0, 0)
        filters_row.setSpacing(6)
        self._sb_filters_row = filters_row

        self.sidebar_lang_combo = VoiceComboBox()
        self.sidebar_lang_combo.setFixedWidth(54)
        self.sidebar_lang_combo.setFixedHeight(32)
        self.sidebar_lang_combo.setToolTip(tr("ui_101", "Мова голосів, озвучки та AI-коректора"))
        seen_sb_langs = set()
        for loc_item in list_available_ui_languages():
            c_code = str(loc_item.get("code", "")).strip().lower()
            c_short = str(loc_item.get("short", "") or c_code.upper()).strip().upper()
            if c_code and c_code not in seen_sb_langs:
                seen_sb_langs.add(c_code)
                self.sidebar_lang_combo.add_gender_item(c_short, c_code, "")
        cur_lang = str(config.get("app_language", "en")).lower()
        if cur_lang not in seen_sb_langs:
            cur_lang = "en" if "en" in seen_sb_langs else (next(iter(seen_sb_langs), "en"))
            config.set("app_language", cur_lang)
        lang_idx = self.sidebar_lang_combo.findData(cur_lang)
        self.sidebar_lang_combo.setCurrentIndex(lang_idx if lang_idx >= 0 else 0)
        self.sidebar_lang_combo.currentIndexChanged.connect(self._on_sidebar_lang_changed)
        filters_row.addWidget(self.sidebar_lang_combo)

        self.sidebar_gender_combo = VoiceComboBox()
        self.sidebar_gender_combo.setFixedHeight(32)
        self.sidebar_gender_combo.setToolTip(
            tr("ui_102", "Які голоси показувати у списку:\n• Всі голоси (і жіночі, і чоловічі)\n• Жіночі (Ж)\n• Чоловічі (М)")
        )
        self.sidebar_gender_combo.add_gender_item(tr("ui_103", "Всі голоси"), "both", "")
        self.sidebar_gender_combo.add_gender_item(tr("ui_104", "Жіночі"), "female", "female")
        self.sidebar_gender_combo.add_gender_item(tr("ui_105", "Чоловічі"), "male", "male")
        self.sidebar_gender_combo.view().setMinimumWidth(135)
        cur_gender = str(config.get("voice_gender_filter", "both")).lower()
        gen_idx = self.sidebar_gender_combo.findData(cur_gender)
        self.sidebar_gender_combo.setCurrentIndex(gen_idx if gen_idx >= 0 else 0)
        self.sidebar_gender_combo.currentIndexChanged.connect(self._on_sidebar_gender_changed)
        filters_row.addWidget(self.sidebar_gender_combo, 1)

        sb_layout.addWidget(self._sb_filters_container)

        sb_layout.addStretch(1)

        # Зрозумілий блок швидких дій унизу бокової панелі
        action_card = QFrame()
        action_card.setProperty("class", "card")
        self._sb_action_card = action_card
        ac_lay = QVBoxLayout(action_card)
        ac_lay.setContentsMargins(7, 9, 7, 9)
        ac_lay.setSpacing(6)
        self._sb_ac_lay = ac_lay

        ac_hdr_row = QHBoxLayout()
        ac_title = QLabel(tr("ui_106", "Швидкі дії:"))
        ac_title.setStyleSheet("color: #CBD5E1; font-size: 12px; font-weight: 700;")
        self._sb_ac_title = ac_title
        ac_hdr_row.addWidget(ac_title)
        ac_hdr_row.addStretch(1)
        ac_help_btn = HelpButton(
            tr("ui_107", "Як писати і озвучувати текст на ПК?"),
            tr(
                "ui_108",
                "<b>Вікно програми можна згорнути!</b><br>"
                "Друкувати можна прямо поверх ігор, браузера чи месенджерів.<br><br>"
                "• <b>Почати набір:</b> натисніть <b>{hk_toggle}</b> або кнопку <b>«Почати писати»</b> нижче.<br>"
                "• <b>Індикатор:</b> унизу екрана з'явиться крапка — можна друкувати.<br>"
                "• <b>Озвучити:</b> натисніть <b>{hk_send}</b>.<br>"
                "• <b>Скасувати:</b> натисніть <b>{hk_cancel}</b>, якщо передумали.<br><br>"
                "<i>Якщо у захищеній грі текст не набирається, запустіть програму від імені Адміністратора або перемкніть спосіб набору у «Гарячих клавішах».</i>",
            ),
        )
        self._sb_ac_help_btn = ac_help_btn
        ac_hdr_row.addWidget(ac_help_btn)
        ac_lay.addLayout(ac_hdr_row)

        self.sidebar_hk_label = QLabel()
        self.sidebar_hk_label.setWordWrap(True)
        self.sidebar_hk_label.setStyleSheet("color: #E2E8F0; font-size: 11px;")
        ac_lay.addWidget(self.sidebar_hk_label)

        self.toggle_typing_sidebar_btn = QPushButton(tr("ui_109", "⌨️ Почати писати"))
        self.toggle_typing_sidebar_btn.setProperty("class", "primaryBtn")
        self.toggle_typing_sidebar_btn.setStyleSheet("padding: 7px 8px;")
        self.toggle_typing_sidebar_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_typing_sidebar_btn.setToolTip(
            tr("ui_110", "Вмикає режим набору тексту прямо поверх екрана.\n"
            "Надрукуйте фразу на клавіатурі та натисніть Enter, щоб озвучити її.")
        )
        self.toggle_typing_sidebar_btn.clicked.connect(self._on_manual_toggle_typing)
        ac_lay.addWidget(self.toggle_typing_sidebar_btn)

        self.stop_sound_btn = QPushButton(tr("ui_111", "⏹ Зупинити голос"))
        self.stop_sound_btn._full_text = tr("ui_111", "⏹ Зупинити голос")  # type: ignore[attr-defined]
        self.stop_sound_btn._icon_text = "⏹"  # type: ignore[attr-defined]
        self.stop_sound_btn.setProperty("class", "dangerBtn")
        self.stop_sound_btn.setStyleSheet("padding: 7px 8px;")
        self.stop_sound_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_sound_btn.setToolTip(
            tr("ui_112", "Миттєво перериває озвучення поточної фрази, якщо програма зараз щось говорить.")
        )
        self.stop_sound_btn.clicked.connect(tts_manager.stop_current)
        ac_lay.addWidget(self.stop_sound_btn)

        self.live_status_label = QLabel(tr("ui_113", "Система готова"))
        self.live_status_label.setWordWrap(True)
        self.live_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.live_status_label.setStyleSheet("color: #A5B4FC; font-size: 11px; margin-top: 2px;")
        ac_lay.addWidget(self.live_status_label)

        sb_layout.addWidget(action_card)

        root_layout.addWidget(sidebar)
        self.switch_page(0)
        self._refresh_status_banners()
        self._update_sidebar_compact_mode(force=True)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_sidebar_compact_mode()

    def _update_sidebar_compact_mode(self, force: bool = False) -> None:
        """При фізичній ширині < 1138px звужує бокове меню до іконок; при висоті <= 773px лишає в «Швидких діях» лише 2 кнопки і статус під ними."""
        if not hasattr(self, "right_sidebar"):
            return
        dpr = max(1.0, float(self.devicePixelRatioF()))
        phys_w = round(self.width() * dpr)
        phys_h = round(self.height() * dpr)
        compact = phys_w < 1138
        short_h = phys_h <= 773
        if not force and compact == self._sidebar_compact and short_h == self._sidebar_short_h:
            return
        self._sidebar_compact = compact
        self._sidebar_short_h = short_h

        if compact:
            self.right_sidebar.setFixedWidth(44)
            self._sb_layout.setContentsMargins(3, 8, 3, 8)
            self._sb_layout.setSpacing(4)
            self._sb_brand_title.setText("🎙️")
            self._sb_brand_title.setStyleSheet("color: #FFFFFF; font-size: 14px; font-weight: 800;")
            self._sb_brand_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._sb_brand_help_btn.setVisible(False)
            for btn in self.nav_buttons:
                btn.setText(getattr(btn, "_icon_text", btn.text()))
                btn.setStyleSheet("padding: 5px 0px; text-align: center; font-size: 14px; border-radius: 8px;")
            self.donate_btn.setText(getattr(self.donate_btn, "_icon_text", "❤️"))
            self.donate_btn.setStyleSheet(
                "QPushButton { background-color: #362B4C; color: #F5D0FE; border: 1px solid #7E57C2; border-radius: 8px; padding: 5px 0px; font-size: 14px; font-weight: 700; }"
                "QPushButton:hover { background-color: #463763; border-color: #A855F7; color: #FFFFFF; }"
            )
            self._sb_filters_row.setDirection(QBoxLayout.Direction.TopToBottom)
            self.sidebar_lang_combo.setFixedWidth(38)
            self.sidebar_lang_combo.setFixedHeight(26)
            self.sidebar_gender_combo.setFixedWidth(38)
            self.sidebar_gender_combo.setFixedHeight(26)
            self._sb_ac_lay.setContentsMargins(2, 4, 2, 4)
            self._sb_ac_lay.setSpacing(4)
            self._sb_ac_title.setVisible(False)
            self._sb_ac_help_btn.setVisible(False)
            self.sidebar_hk_label.setVisible(False)
            self.live_status_label.setVisible(False)
            self.stop_sound_btn.setText(getattr(self.stop_sound_btn, "_icon_text", "⏹"))
            self.stop_sound_btn.setStyleSheet("padding: 5px 0px; font-size: 14px; border-radius: 8px;")
            self.toggle_typing_sidebar_btn.setStyleSheet("padding: 5px 0px; font-size: 14px; border-radius: 8px;")
        else:
            self.right_sidebar.setFixedWidth(175)
            self._sb_layout.setContentsMargins(6, 12, 6, 12)
            self._sb_layout.setSpacing(6)
            self._sb_brand_title.setText("🎙️ SilentVoice")
            self._sb_brand_title.setStyleSheet("color: #FFFFFF; font-size: 16px; font-weight: 800;")
            self._sb_brand_title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self._sb_brand_help_btn.setVisible(True)
            for btn in self.nav_buttons:
                btn.setText(getattr(btn, "_full_text", btn.text()))
                btn.setStyleSheet("")
            self.donate_btn.setText(getattr(self.donate_btn, "_full_text", self.donate_btn.text()))
            self.donate_btn.setStyleSheet(
                "QPushButton { background-color: #362B4C; color: #F5D0FE; border: 1px solid #7E57C2; border-radius: 10px; padding: 8px 10px; font-size: 13px; font-weight: 700; }"
                "QPushButton:hover { background-color: #463763; border-color: #A855F7; color: #FFFFFF; }"
            )
            self._sb_filters_row.setDirection(QBoxLayout.Direction.LeftToRight)
            self.sidebar_lang_combo.setFixedWidth(54)
            self.sidebar_lang_combo.setFixedHeight(32)
            self.sidebar_gender_combo.setMinimumWidth(0)
            self.sidebar_gender_combo.setMaximumWidth(16777215)
            self.sidebar_gender_combo.setFixedHeight(32)
            self._sb_ac_lay.setContentsMargins(7, 6 if short_h else 9, 7, 6 if short_h else 9)
            self._sb_ac_lay.setSpacing(4 if short_h else 6)
            self._sb_ac_title.setVisible(not short_h)
            self._sb_ac_help_btn.setVisible(not short_h)
            self.sidebar_hk_label.setVisible(not short_h)
            self.live_status_label.setVisible(True)
            self.stop_sound_btn.setText(getattr(self.stop_sound_btn, "_full_text", self.stop_sound_btn.text()))
            self.stop_sound_btn.setStyleSheet("padding: 7px 8px;")
            self.toggle_typing_sidebar_btn.setStyleSheet("padding: 7px 8px;")

        self._update_typing_sidebar_btn(bool(getattr(self, "_typing_btn_active", False)))
        if hasattr(self, "pages"):
            self._update_sidebar_filters_visibility(self.pages.currentIndex())

    def _get_default_test_phrase(self, lang: Optional[str] = None) -> str:
        """Повертає стандартну тестову фразу мовою вибраних голосів (app_language)."""
        import json
        target = str(lang or config.get("app_language", "uk") or "uk").strip().lower() or "uk"
        loc_file = get_locales_dir() / f"{target}.json"
        if loc_file.exists():
            try:
                data = json.loads(loc_file.read_text(encoding="utf-8"))
                val = str(data.get("ui_120", "")).strip()
                if val:
                    return val
            except Exception:
                pass
        return tr("ui_120", "Привіт! Тепер цей голос говоритиме замість мене.")

    def _all_default_test_phrases(self) -> set[str]:
        import json
        phrases = {tr("ui_120", "Привіт! Тепер цей голос говоритиме замість мене.")}
        loc_dir = get_locales_dir()
        if loc_dir.exists():
            for p in loc_dir.glob("*.json"):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    v = str(data.get("ui_120", "")).strip()
                    if v:
                        phrases.add(v)
                except Exception:
                    pass
        return phrases

    def _on_sidebar_lang_changed(self) -> None:
        from core.config import sync_default_quick_phrases_for_lang
        lang = str(self.sidebar_lang_combo.currentData() or "uk").strip().lower() or "uk"
        config.update({"app_language": lang, "language_confirmed": True})
        if sync_default_quick_phrases_for_lang(lang):
            self.keyboard_hook.update_from_config(config.data)
            self._rebuild_quick_phrases_grid()
        if hasattr(self, "test_voice_input"):
            cur_txt = self.test_voice_input.text().strip()
            if not cur_txt or cur_txt in self._all_default_test_phrases():
                self.test_voice_input.setText(self._get_default_test_phrase(lang))
        # Мова інтерфейсу не змінюється автоматично при перемиканні мови голосів — лише вручну в налаштуваннях
        self._rebuild_local_engine_cards()
        self._rebuild_online_cards()
        self._sync_tts_page_selectors()
        self._populate_api_add_combos()
        tts_manager.schedule_smart_warmup(1.5, "voice_lang_changed")

    def _check_pending_first_run_lang(self) -> None:
        """Мова інтерфейсу змінюється виключно вручну, без автоперемикання за локаллю ОС."""
        if hasattr(self, "_lang_net_retry_timer"):
            self._lang_net_retry_timer.stop()
        return

        from core.i18n import detect_system_language, get_locales_dir, has_internet_connection
        sys_code, native_name, eng_name = detect_system_language()
        if (get_locales_dir() / f"{sys_code}.json").exists():
            config.set("first_run_lang_checked", True)
            if hasattr(self, "_lang_net_retry_timer"):
                self._lang_net_retry_timer.stop()
            return

        def _probe() -> bool:
            return has_internet_connection(timeout=2.0)

        def _on_probe_done(online: Any, err: str) -> None:
            if not online or err or bool(config.get("first_run_lang_checked", False)):
                return
            if not self.isVisible() or getattr(self, "_first_run_lang_prompt_active", False):
                return
            self._first_run_lang_prompt_active = True
            if hasattr(self, "_lang_net_retry_timer"):
                self._lang_net_retry_timer.stop()
            try:
                from ui.dialogs import FirstRunLanguageDialog
                prev_ui = get_ui_language()
                dlg = FirstRunLanguageDialog(sys_code, native_name, eng_name, self)
                dlg.exec()
                new_ui = get_ui_language()
                if new_ui != prev_ui:
                    set_ui_language(prev_ui)
                    self._schedule_live_ui_language(new_ui)
                else:
                    self._rebuild_local_engine_cards()
                    self._rebuild_online_cards()
                    self._sync_tts_page_selectors()
            finally:
                self._first_run_lang_prompt_active = False

        run_in_background(self, _probe, _on_probe_done)

    def _ensure_language_confirmed(self) -> bool:
        """
        Якщо новий користувач вперше встановлює голосову модель, питає підтвердження мови.
        Після вибору мови більше ніколи не перепитує, застосовує мову усюди та встановлює лише голоси для неї.
        """
        if bool(config.get("language_confirmed", False)):
            return True
        dlg = LanguageConfirmDialog(self)
        if dlg.exec() != LanguageConfirmDialog.DialogCode.Accepted:
            return False
        chosen_lang = dlg.selected_language()
        if hasattr(self, "sidebar_lang_combo"):
            self.sidebar_lang_combo.blockSignals(True)
            idx = self.sidebar_lang_combo.findData(chosen_lang)
            if idx >= 0:
                self.sidebar_lang_combo.setCurrentIndex(idx)
            self.sidebar_lang_combo.blockSignals(False)
        tts_manager.clear_silero_cache()
        tts_manager._piper_cache.clear()
        self._rebuild_local_engine_cards()
        self._sync_tts_page_selectors()
        self._populate_api_add_combos()
        return True

    def _on_sidebar_gender_changed(self) -> None:
        gender = str(self.sidebar_gender_combo.currentData() or "both")
        config.set("voice_gender_filter", gender)
        self._sync_tts_page_selectors()

    def _populate_api_add_combos(self) -> None:
        """Оновлює порядок сервісів у випадаючих списках онлайн-TTS та AI відповідно до обраної мови."""
        self._populate_online_add_combo()
        self._populate_ai_add_combo()

    def _populate_online_add_combo(self) -> None:
        if not hasattr(self, "online_add_combo"):
            return
        cur = str(self.online_add_combo.currentData() or "")
        lang = str(config.get("app_language", "uk")).lower()
        google_label = tr("online.add.google", "Google Cloud TTS API (безкоштовно)")
        fish_label = tr("online.add.fish_audio", "Fish Audio API (безкоштовно) (клонування)")
        fish_ru_label = tr("online.add.fish_audio_ru", "Fish Audio API (безкоштовно) (клонування) (доступно в РФ)")
        elevenlabs_label = tr("online.add.elevenlabs", "ElevenLabs API (клонування)")
        openai_label = tr("online.add.openai", "OpenAI TTS API (тільки платно)")
        items = (
            [
                ("Microsoft Edge-TTS", "edge"),
                ("Silero Bot (Telegram — @silero_voice_bot)", "silero_bot"),
                (fish_ru_label, "fish_audio"),
                (elevenlabs_label, "elevenlabs"),
                (openai_label, "openai"),
                ("Google AI Studio (Gemini TTS)", "gemini_tts"),
                (google_label, "google"),
            ]
            if lang == "ru"
            else [
                ("Microsoft Edge-TTS", "edge"),
                (google_label, "google"),
                (elevenlabs_label, "elevenlabs"),
                (openai_label, "openai"),
                ("Google AI Studio (Gemini TTS)", "gemini_tts"),
                (fish_label, "fish_audio"),
            ]
        )
        existing_providers = {
            str(item.get("provider", "")).strip()
            for item in config.get("online_items", [])
            if isinstance(item, dict)
        }
        self.online_add_combo.blockSignals(True)
        self.online_add_combo.clear()
        for title, key in items:
            if key not in existing_providers:
                self.online_add_combo.addItem(title, key)
        if cur:
            idx = self.online_add_combo.findData(cur)
            if idx >= 0:
                self.online_add_combo.setCurrentIndex(idx)
        self.online_add_combo.blockSignals(False)

    def _populate_ai_add_combo(self) -> None:
        if not hasattr(self, "ai_add_combo"):
            return
        cur = str(self.ai_add_combo.currentData() or "")
        lang = str(config.get("app_language", "uk")).lower()
        ru_order = ("deepseek", "zai", "openrouter", "ollama", "groq", "gemini", "openai")
        keys = [k for k in ru_order if k in AI_PROVIDERS] if lang == "ru" else list(AI_PROVIDERS.keys())
        # Local OpenAI-compatible servers are added through the simplified local-model wizard.
        keys = [k for k in keys if k != "local_api"]
        for k in AI_PROVIDERS.keys():
            if k != "local_api" and k not in keys:
                keys.append(k)
        existing_providers = {
            str(item.get("provider", "")).strip()
            for item in config.get("ai_items", [])
            if isinstance(item, dict)
        }
        self.ai_add_combo.blockSignals(True)
        self.ai_add_combo.clear()
        for prov_key in keys:
            if prov_key in existing_providers:
                continue
            prov_title = (
                tr(f"ai.prov.{prov_key}.title", str(AI_PROVIDERS[prov_key]["title"]))
                if prov_key in ("ollama", "local_api", "openai") else str(AI_PROVIDERS[prov_key]["title"])
            )
            self.ai_add_combo.addItem(prov_title, prov_key)
        if cur:
            idx = self.ai_add_combo.findData(cur)
            if idx >= 0:
                self.ai_add_combo.setCurrentIndex(idx)
        self.ai_add_combo.blockSignals(False)

    def _update_sidebar_filters_visibility(self, page_index: int) -> None:
        """
        Показує фільтри мови та статі лише на тих розділах, де вони реально впливають на вміст:
        - 0 («Голос та Озвучка»): мова + стать
        - 1 («Швидкі фрази»): лише мова (оновлює стандартні фрази й мову скидання)
        - 2..5 («Кнопки та Ввід», «Текст на екран», «Субтитри OBS», «Звук та Система»): приховані
        """
        if not hasattr(self, "_sb_filters_container"):
            return
        show_lang = page_index in (0, 1)
        show_gender = page_index == 0
        self._sb_filters_container.setVisible(show_lang or show_gender)
        self.sidebar_lang_combo.setVisible(show_lang)
        self.sidebar_gender_combo.setVisible(show_gender)
        if show_lang:
            if getattr(self, "_sidebar_compact", False):
                self.sidebar_lang_combo.setFixedWidth(38)
            elif show_gender:
                self.sidebar_lang_combo.setFixedWidth(54)
            else:
                self.sidebar_lang_combo.setMinimumWidth(0)
                self.sidebar_lang_combo.setMaximumWidth(16777215)

    def switch_page(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            btn.setProperty("active", "true" if i == index else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        self._update_sidebar_filters_visibility(index)
        # При переході на вкладку «Звук та Система» (index == 5) автоматично оновлюємо список
        # аудіопристроїв, щоб підключені вже після запуску навушники одразу відображалися у списку.
        if index == 5 and hasattr(self, "primary_dev_combo"):
            self._populate_audio_devices(refresh_backend=True)

    # ──────────────────────────────────────────────────────────────────
    # РОЗДІЛ 1 (Перший при запуску): Голос та Озвучка
    # ──────────────────────────────────────────────────────────────────

    def _build_page_tts(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(2, 2, 4, 2)
        lay.setSpacing(10)

        lay.addWidget(
            make_header_with_help(
                tr("ui_087", "🎙️ Голос та Озвучка"),
                tr("ui_114", "Налаштування голосу (TTS)"),
                tr(
                    "ui_115",
                    "<b>Як влаштований вибір голосів:</b><br>"
                    "Угорі у блоці <b>«Ваш активний голос»</b> показуються лише готові до роботи голоси.<br><br>"
                    "• <b>Завантажити на ПК</b> — офлайн-голоси (працюють без інтернету). Завантажуються у 1 клік, можна попередньо прослухати.<br>"
                    "• <b>Онлайн-голоси</b> — безкоштовний Edge-TTS або сервіси за API-ключем (OpenAI, ElevenLabs, Google Cloud, Fish Audio) чи Telegram-бот.<br><br>"
                    "<i>Найпростіший старт без налаштувань — безкоштовний Edge-TTS або локальний голос Piper / Silero.</i>",
                ),
            )
        )

        # Верхній блок активного голосу + швидка перевірка
        active_card = QFrame()
        active_card.setProperty("class", "cardHighlight")
        ac_lay = QVBoxLayout(active_card)
        ac_lay.setContentsMargins(14, 10, 14, 10)
        ac_lay.setSpacing(8)

        row1 = QHBoxLayout()
        row1.setSpacing(10)
        row1.addWidget(QLabel(tr("ui_116", "<b>Ваш активний голос:</b>")))

        self.tts_engine_combo = QComboBox()
        self.tts_engine_combo.setToolTip(tr("ui_117", "Список встановлених TTS"))
        self.tts_engine_combo.setMinimumWidth(190)
        self.tts_engine_combo.currentIndexChanged.connect(self._on_tts_page_engine_changed)
        row1.addWidget(self.tts_engine_combo, 1)

        self.tts_voice_combo = VoiceComboBox()
        self.tts_voice_combo.setToolTip(tr("ui_118", "Виберіть голос обраного TTS (Ж — жіночий, М — чоловічий)"))
        self.tts_voice_combo.setMinimumWidth(190)
        self.tts_voice_combo.currentIndexChanged.connect(self._on_tts_page_voice_changed)
        row1.addWidget(self.tts_voice_combo, 1)

        self.save_tts_audio_btn = QPushButton("💾")
        self.save_tts_audio_btn.setFixedSize(36, 32)
        self.save_tts_audio_btn.setStyleSheet("padding: 0px; font-size: 15px;")
        self.save_tts_audio_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_tts_audio_btn.setToolTip(
            tr("tts.save_audio_tooltip", "Згенерувати та записати вписаний текст у .wav файл (вибрати куди зберегти)")
        )
        self.save_tts_audio_btn.clicked.connect(self._on_save_tts_audio_clicked)
        row1.addWidget(self.save_tts_audio_btn)
        ac_lay.addLayout(row1)

        # Рядок для швидкої перевірки голосу прямо тут
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        self.test_voice_input = QLineEdit()
        self.test_voice_input.setPlaceholderText(tr("ui_119", "Напишіть сюди будь-яку фразу для перевірки вибраного голосу..."))
        self.test_voice_input.setText(self._get_default_test_phrase())
        self.test_voice_input.returnPressed.connect(self._on_test_voice_clicked)
        row2.addWidget(self.test_voice_input, 1)

        test_voice_btn = QPushButton(tr("ui_121", "🔊 Прослухати тест"))
        test_voice_btn.setProperty("class", "primaryBtn")
        test_voice_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        test_voice_btn.clicked.connect(self._on_test_voice_clicked)
        row2.addWidget(test_voice_btn)
        ac_lay.addLayout(row2)

        lay.addWidget(active_card)

        # Нижня частина: 5 вкладок (Завантажити на ПК / Онлайн-голоси / Усі налаштування / AI Орфографія / Чат Twitch)
        self.tts_tabs = QTabWidget()
        self.tts_tabs.addTab(self._build_tts_tab_models(), tr("ui_123", "💻 Завантажити на ПК"))
        self.tts_tabs.addTab(self._build_tts_tab_online(), tr("ui_124", "🌐 Онлайн-голоси"))
        self.tts_tabs.addTab(self._build_tts_tab_all_settings(), tr("ui_125", "⚙️ Усі налаштування"))
        self.tts_tabs.addTab(self._build_tts_tab_ai_spellcheck(), tr("ui_126", "🪄 AI Орфографія"))
        self.tts_tabs.addTab(self._build_tts_tab_twitch_chat(), tr("ui_tts_tab_twitch", "💬 Чат Twitch"))
        lay.addWidget(self.tts_tabs, 1)

        self._sync_tts_page_selectors()
        return page

    def _on_test_voice_clicked(self) -> None:
        txt = self.test_voice_input.text().strip() or self._get_default_test_phrase()
        tts_manager.speak(txt)

    def _on_save_tts_audio_clicked(self) -> None:
        txt = self.test_voice_input.text().strip() if hasattr(self, "test_voice_input") else ""
        if not txt:
            QMessageBox.information(
                self,
                tr("tts.save_audio_title", "Зберегти озвучку"),
                tr("tts.err.enter_text_record", "Введіть текст для запису озвучки."),
            )
            if hasattr(self, "test_voice_input"):
                self.test_voice_input.setFocus()
            return

        from pathlib import Path
        desktop_dir = Path.home() / "Desktop"
        init_dir = desktop_dir if desktop_dir.exists() else Path.home()
        dest_str, _ = QFileDialog.getSaveFileName(
            self,
            tr("tts.save_audio_dialog", "Зберегти озвучку у WAV-файл"),
            str(init_dir / "voice_output.wav"),
            "WAV Audio (*.wav)",
        )
        if not dest_str:
            return
        dest_path = Path(dest_str)
        if dest_path.suffix.lower() != ".wav":
            dest_path = dest_path.with_suffix(".wav")

        self.save_tts_audio_btn.setEnabled(False)
        self.save_tts_audio_btn.setText("⏳")
        self.live_status_label.setText(tr("tts.save_audio_generating", "💾 Генерація аудіофайлу..."))

        def _worker():
            return tts_manager.synthesize_to_file(txt, dest_path)

        def _done(res: Any, err: str) -> None:
            self.save_tts_audio_btn.setEnabled(True)
            self.save_tts_audio_btn.setText("💾")
            if err:
                self.live_status_label.setText(tr("tts.status.synth_error", "Помилка синтезу"))
                QMessageBox.warning(self, tr("common.error", "Помилка"), str(err))
                return
            ok, info = res if isinstance(res, tuple) and len(res) == 2 else (False, "")
            if ok:
                self.live_status_label.setText(tr("tts.save_audio_done", "✅ Файл збережено: {name}", name=dest_path.name))
                QMessageBox.information(
                    self,
                    tr("tts.save_audio_title", "Зберегти озвучку"),
                    tr("tts.save_audio_success", "Озвучений текст успішно збережено:\n{path}", path=str(info)),
                )
            else:
                self.live_status_label.setText(tr("tts.status.synth_error", "Помилка синтезу"))
                QMessageBox.warning(self, tr("common.error", "Помилка"), str(info))

        run_in_background(self, _worker, _done)

    def _build_tts_tab_models(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content.setObjectName("scrollContent")
        lay = QVBoxLayout(content)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        self.local_models_layout = QVBoxLayout()
        self.local_models_layout.setSpacing(8)
        lay.addLayout(self.local_models_layout)
        lay.addStretch(1)

        self._rebuild_local_engine_cards()
        scroll.setWidget(content)
        return scroll

    def _rebuild_local_engine_cards(self) -> None:
        if not hasattr(self, "local_models_layout"):
            return
        while self.local_models_layout.count():
            item = self.local_models_layout.takeAt(0)
            w = item.widget() if item else None
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()

        for k in list(self._engine_cards_ui.keys()):
            if k in ENGINE_SPECS:
                self._engine_cards_ui.pop(k, None)

        visible_engines = tts_manager.get_visible_local_engines()
        if not visible_engines:
            empty_lbl = QLabel(
                tr("ui_128", "Для обраної мови локальні моделі не передбачені — скористайтеся вкладкою «🌐 Онлайн-голоси» (Microsoft Edge-TTS, Google, ElevenLabs або OpenAI).")
            )
            empty_lbl.setWordWrap(True)
            empty_lbl.setStyleSheet("color: #94A3B8; font-size: 13px; padding: 18px;")
            self.local_models_layout.addWidget(empty_lbl)

        clone_gguf_min_vram = {
            "omnivoice": 2.0,
            "qwen3_tts": 4.0,
            "zonos3_mini": 4.0,
        }
        has_gpu = has_hardware_gpu()
        vram_gb = get_discrete_gpu_vram_gb() if has_gpu else None
        for spec in visible_engines:
            eng_id = spec["id"]
            installed = tts_manager.is_engine_installed(eng_id)
            is_clone_gguf = eng_id in clone_gguf_min_vram
            min_vram_gb = clone_gguf_min_vram.get(eng_id)
            if min_vram_gb is not None and not installed and has_gpu and vram_gb is not None and vram_gb < min_vram_gb:
                continue

            card = (
                _VoiceDropCard(lambda p, eid=eng_id: self._set_clone_ref_file(eid, p))
                if is_clone_gguf
                else QFrame()
            )
            card.setProperty("class", "card")
            c_lay = QVBoxLayout(card)
            c_lay.setContentsMargins(14, 10, 14, 10)
            c_lay.setSpacing(6)

            top_r = QHBoxLayout()
            top_r.setSpacing(10)

            card_title = (
                tr(f"tts.title.{eng_id}", str(spec["title"]))
                if is_clone_gguf
                else str(spec["title"])
            )
            t_lbl = QLabel(f"<b>{card_title}</b>")
            t_lbl.setStyleSheet("color: #FFFFFF; font-size: 14px;")
            top_r.addWidget(t_lbl)

            meta_lbl = QLabel(tr(f"tts.size.{eng_id}", str(spec["size_label"])))
            meta_lbl.setStyleSheet("color: #94A3B8; font-size: 12px;")
            top_r.addWidget(meta_lbl)
            top_r.addStretch(1)

            listen_btn = QPushButton(tr("ui_129", "🎧 Прослухати"))
            listen_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            listen_btn.setToolTip(tr("ui_130", "Відкрити список доступних голосів для прослуховування"))
            if is_clone_gguf:
                listen_btn.clicked.connect(
                    lambda checked=False, eid=eng_id, title=str(spec["title"]): VoiceSamplesDialog(
                        eid,
                        title,
                        self,
                        on_select_sample=lambda p, e=eid: self._set_clone_ref_file(e, p),
                        allow_install=tts_manager.is_engine_installed(eid),
                        selected_sample_path=str(config.get(f"{eid}_ref_path", "") or ""),
                        on_delete_sample=lambda _p, e=eid: self._on_clone_sample_deleted(e),
                    ).exec()
                )
            else:
                listen_btn.clicked.connect(
                    lambda checked=False, eid=eng_id, title=spec["title"]: VoiceSamplesDialog(eid, title, self).exec()
                )
            top_r.addWidget(listen_btn)

            action_btn = QPushButton()
            action_btn.setMinimumWidth(115)
            action_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            action_btn.clicked.connect(lambda checked=False, eid=eng_id: self._on_local_engine_action_clicked(eid))
            top_r.addWidget(action_btn)

            cancel_btn = QPushButton(tr("tts.dl.cancel_btn", "✕ Скасувати"))
            cancel_btn.setProperty("class", "dangerBtn")
            cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            cancel_btn.setVisible(False)
            cancel_btn.clicked.connect(lambda checked=False, eid=eng_id: self._on_cancel_model_download_clicked(eid))
            top_r.addWidget(cancel_btn)

            c_lay.addLayout(top_r)

            clone_ui: Dict[str, Any] = {}
            if is_clone_gguf:
                clone_ui = self._build_llama_tts_clone_opts(eng_id, str(spec["title"]), c_lay)

            prog_bar = QProgressBar()
            prog_bar.setRange(0, 100)
            prog_bar.setValue(0)
            prog_bar.setVisible(False)
            c_lay.addWidget(prog_bar)

            prog_lbl = QLabel("")
            prog_lbl.setProperty("class", "muted")
            prog_lbl.setWordWrap(True)
            prog_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            prog_lbl.setVisible(False)
            c_lay.addWidget(prog_lbl)

            card_entry: Dict[str, Any] = {
                "progress": prog_bar,
                "prog_label": prog_lbl,
                "action_btn": action_btn,
                "cancel_btn": cancel_btn,
                "listen_btn": listen_btn,
            }
            card_entry.update(clone_ui)
            self._engine_cards_ui[eng_id] = card_entry
            self._update_engine_card_state(eng_id, installed)
            if tts_manager.is_downloading(eng_id):
                pct, label = self._download_ui_state.get(
                    eng_id,
                    tts_manager.get_download_progress(eng_id) or (1, tr("ui_411", "Підготовка до встановлення...")),
                )
                self._on_model_download_progress(eng_id, pct, label)
            self.local_models_layout.addWidget(card)

    def _build_llama_tts_clone_opts(self, eng_id: str, eng_title: str, parent_layout: QVBoxLayout) -> Dict[str, Any]:
        """Додає панель статусу файлу клонування, кнопку додавання голосу та транскрипт (для Qwen3-TTS)."""
        opts_widget = QWidget()
        opts_v = QVBoxLayout(opts_widget)
        opts_v.setContentsMargins(0, 2, 0, 0)
        opts_v.setSpacing(6)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        ref_lbl = ElidedLabel("")
        row.addWidget(ref_lbl, 1)

        clear_ref_btn = QPushButton("✕")
        clear_ref_btn.setProperty("class", "dangerBtn")
        clear_ref_btn.setFixedSize(28, 28)
        clear_ref_btn.setStyleSheet("padding: 0px; font-size: 13px; font-weight: 700;")
        clear_ref_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_ref_btn.setToolTip(tr("llama_tts.clear_ref_tip", "Скинути вибраний файл і використовувати стандартний голос"))
        clear_ref_btn.clicked.connect(lambda checked=False, eid=eng_id: self._clear_clone_ref_file(eid))
        row.addWidget(clear_ref_btn)

        choose_ref_btn = QPushButton(tr("ov.pick_file", "📂 Обрати голос"))
        choose_ref_btn.setMinimumWidth(115)
        choose_ref_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        choose_ref_btn.setToolTip(
            tr("ov.pick_file_tip", "Оберіть або перетягніть .wav/.mp3/.ogg файл зразка голосу для клонування")
        )
        choose_ref_btn.clicked.connect(lambda checked=False, eid=eng_id: self._pick_clone_ref_file(eid))
        row.addWidget(choose_ref_btn)

        settings_btn: Optional[QPushButton] = None
        if eng_id == "omnivoice":
            settings_btn = QPushButton(tr("omnivoice.settings_btn", "⚙ Налаштування"))
            settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            settings_btn.setToolTip(
                tr("omnivoice.settings_btn_tip", "Режим клонування / конструктора голосу, текст референсу, кроки (16/32) та шепіт")
            )
            settings_btn.clicked.connect(self._open_omnivoice_settings_dialog)
            row.addWidget(settings_btn)

        opts_v.addLayout(row)

        parent_layout.addWidget(opts_widget)
        res: Dict[str, Any] = {
            "opts_widget": opts_widget,
            "ref_lbl": ref_lbl,
            "clear_ref_btn": clear_ref_btn,
            "choose_ref_btn": choose_ref_btn,
        }
        if settings_btn is not None:
            res["settings_btn"] = settings_btn
        return res

    def _open_omnivoice_settings_dialog(self) -> None:
        dlg = OmniVoiceSettingsDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._refresh_clone_card_ui("omnivoice")

    def _pick_clone_ref_file(self, engine_id: str) -> None:
        ref_key = f"{engine_id}_ref_path"
        cur = str(config.get(ref_key, "") or "").strip()
        start_dir = os.path.dirname(cur) if cur and os.path.exists(cur) else ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr("ov.pick_dlg_title", "Оберіть файл голосу для клонування"),
            start_dir,
            tr("ov.pick_dlg_filter", "Аудіофайли (*.wav *.mp3 *.ogg *.flac *.m4a);;Усі файли (*.*)"),
        )
        if path:
            self._set_clone_ref_file(engine_id, path)

    def _set_clone_ref_file(self, engine_id: str, path: str) -> None:
        clean = str(path or "").strip()
        if not clean or not os.path.isfile(clean):
            return
        prev_ref = str(config.get(f"{engine_id}_ref_path", "") or "").strip()
        resolved = register_custom_clone_sample(engine_id, clean) or clean
        config.set(f"{engine_id}_ref_path", resolved)
        if engine_id == "omnivoice":
            config.set("omnivoice_mode", "clone")
            if prev_ref != resolved:
                from core.omnivoice_tts import get_default_omnivoice_ref_text

                auto_txt = get_default_omnivoice_ref_text(resolved)
                config.set("omnivoice_ref_text", auto_txt)
                config.set("omnivoice_ref_text_path", resolved if auto_txt else "")
            if str(config.get("active_engine", "") or "").strip() == "omnivoice":
                tts_manager.schedule_smart_warmup(0.2, "omnivoice_ref_changed")
        self._refresh_clone_card_ui(engine_id)

    def _on_clone_sample_deleted(self, engine_id: str) -> None:
        self._refresh_clone_card_ui(engine_id)

    def _clear_clone_ref_file(self, engine_id: str) -> None:
        config.set(f"{engine_id}_ref_path", "")
        if engine_id == "omnivoice":
            config.set("omnivoice_ref_text", "")
            config.set("omnivoice_ref_text_path", "")
        self._refresh_clone_card_ui(engine_id)

    def _refresh_clone_card_ui(self, engine_id: str) -> None:
        ui = self._engine_cards_ui.get(engine_id)
        if not ui:
            return
        installed = tts_manager.is_engine_installed(engine_id)
        has_gpu = has_hardware_gpu()
        listen_btn = ui.get("listen_btn")
        if isinstance(listen_btn, QPushButton):
            listen_btn.setVisible(has_gpu)
        choose_btn = ui.get("choose_ref_btn")
        if isinstance(choose_btn, QPushButton):
            choose_btn.setVisible(installed)
        opts_w = ui.get("opts_widget")
        if isinstance(opts_w, QWidget):
            opts_w.setVisible(installed)
        ref_path = str(config.get(f"{engine_id}_ref_path", "") or "").strip()
        has_ref = bool(ref_path and os.path.isfile(ref_path))
        clear_btn = ui.get("clear_ref_btn")
        if isinstance(clear_btn, QPushButton):
            clear_btn.setVisible(installed and has_ref)
        ref_lbl = ui.get("ref_lbl")
        if isinstance(ref_lbl, QLabel):
            if engine_id == "omnivoice":
                from core.omnivoice_tts import (
                    format_omnivoice_design_summary_localized,
                    normalize_omnivoice_mode,
                    normalize_omnivoice_steps,
                )

                ov_mode = normalize_omnivoice_mode()
                ov_steps = normalize_omnivoice_steps()
                ov_whisper = " · 🤫" if bool(config.get("omnivoice_whisper", False)) else ""
                if ov_mode == "clone" and has_ref:
                    name = os.path.basename(ref_path)
                    ref_lbl.setStyleSheet("color: #34D399; font-size: 12px;")
                    ref_lbl.setText(
                        tr(
                            "omnivoice.card.clone_status",
                            "🎤 {name} · {steps} кр.{whisper}",
                            name=name,
                            steps=ov_steps,
                            whisper=ov_whisper,
                        )
                    )
                    ref_lbl.setToolTip(ref_path)
                else:
                    instruct_loc = format_omnivoice_design_summary_localized()
                    ref_lbl.setStyleSheet("color: #60A5FA; font-size: 12px;")
                    design_txt = tr(
                        "omnivoice.card.design_status",
                        "✨ Конструктор голосу ({instruct}) · {steps} кр.",
                        instruct=instruct_loc,
                        steps=ov_steps,
                    )
                    ref_lbl.setText(design_txt)
                    ref_lbl.setToolTip(design_txt)
            elif has_ref:
                name = os.path.basename(ref_path)
                ref_lbl.setStyleSheet("color: #34D399; font-size: 12px;")
                ref_lbl.setText(tr("ov.ref_selected", "🎤 {name}", name=name))
                ref_lbl.setToolTip(ref_path)
            else:
                ref_lbl.setStyleSheet("color: #94A3B8; font-size: 12px;")
                default_text = tr("llama_tts.ref_default", "🎙️ Стандартний голос (без клонування)")
                ref_lbl.setText(default_text)
                ref_lbl.setToolTip(default_text)

    def _build_tts_tab_online(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content.setObjectName("scrollContent")
        lay = QVBoxLayout(content)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        # Верхній рядок вибору онлайн-сервісу + кнопка "Додати"
        add_bar = QFrame()
        add_bar.setProperty("class", "card")
        ab_lay = QHBoxLayout(add_bar)
        ab_lay.setContentsMargins(14, 9, 14, 9)
        ab_lay.setSpacing(10)

        self.online_add_combo = QComboBox()
        self._populate_online_add_combo()
        ab_lay.addWidget(self.online_add_combo, 1)

        add_btn = QPushButton(tr("ui_131", "+ Додати"))
        add_btn.setProperty("class", "primaryBtn")
        add_btn.setMinimumWidth(115)
        add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_btn.clicked.connect(self._on_add_online_provider)
        ab_lay.addWidget(add_btn)

        lay.addWidget(add_bar)
        lay.addWidget(self._create_socks5_bar())

        # Контейнер доданих карток онлайн-голосів
        self.online_cards_layout = QVBoxLayout()
        self.online_cards_layout.setSpacing(8)
        lay.addLayout(self.online_cards_layout)
        lay.addStretch(1)

        self._rebuild_online_cards()
        scroll.setWidget(content)
        return scroll

    def _rebuild_online_cards(self) -> None:
        if not hasattr(self, "online_cards_layout"):
            return
        while self.online_cards_layout.count():
            item = self.online_cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._online_status_labels.clear()
        for k in list(self._engine_cards_ui.keys()):
            if k not in ENGINE_SPECS:
                self._engine_cards_ui.pop(k, None)
        online_items: List[Dict[str, Any]] = list(config.get("online_items", []))
        for item in online_items:
            item_id = str(item.get("id", ""))
            prov_key = str(item.get("provider", "google"))
            prov_spec = ONLINE_PROVIDERS.get(prov_key, ONLINE_PROVIDERS["google"])
            title_str = str(item.get("title", prov_spec["title"]))
            saved_key = str(item.get("api_key", ""))

            card = QFrame()
            card.setProperty("class", "card")
            c_lay = QVBoxLayout(card)
            c_lay.setContentsMargins(14, 10, 14, 10)
            c_lay.setSpacing(8)

            if prov_key == "edge":
                # Для Microsoft Edge-TTS немає поля ключа і немає гайду — тільки кнопки «Встановити» та «Видалити»
                is_installed = bool(item.get("installed", False))

                top_r = QHBoxLayout()
                top_r.setSpacing(8)

                t_lbl = QLabel(f"<b>{title_str}</b>")
                t_lbl.setStyleSheet("color: #FFFFFF; font-size: 14px;")
                top_r.addWidget(t_lbl)
                top_r.addStretch(1)

                install_btn = QPushButton(tr("ui_133", "✅ Встановлено") if is_installed else tr("ui_134", "Встановити")) 
                install_btn.setMinimumWidth(115)
                install_btn.setProperty("class", "primaryBtn")
                install_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                install_btn.setEnabled(not is_installed)
                install_btn.clicked.connect(lambda checked=False, iid=item_id: self._on_install_online_edge(iid))
                top_r.addWidget(install_btn)

                del_btn = QPushButton(tr("ui_137", "Видалити"))
                del_btn.setProperty("class", "dangerBtn")
                del_btn.setMinimumWidth(100)
                del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                del_btn.clicked.connect(lambda checked=False, iid=item_id: self._on_delete_online_item(iid))
                top_r.addWidget(del_btn)

                c_lay.addLayout(top_r)

                if is_installed:
                    edge_r = QHBoxLayout()
                    edge_r.setSpacing(8)
                    edge_fetch_btn = QPushButton(tr("ui_139", "🔄 Оновити голоси"))
                    edge_fetch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    edge_fetch_btn.clicked.connect(
                        lambda checked=False, iid=item_id, btn=edge_fetch_btn: self._start_fetch_provider_catalog(
                            iid, btn, silent=False
                        )
                    )
                    edge_r.addWidget(edge_fetch_btn)
                    e_cnt = len(list(item.get("custom_voices") or []))
                    edge_st = QLabel(
                        tr("ui_f_001", "🟢 Авто-пошук: знайдено {count} голосів", count=e_cnt) if e_cnt
                        else tr("ui_140", "⚪ Голоси оновляться автоматично")
                    )
                    edge_st.setStyleSheet("color: #94A3B8; font-size: 12px;")
                    self._online_status_labels[item_id] = edge_st
                    edge_r.addWidget(edge_st, 1)
                    c_lay.addLayout(edge_r)

                prog_bar = QProgressBar()
                prog_bar.setRange(0, 100)
                prog_bar.setValue(0)
                prog_bar.setVisible(False)
                c_lay.addWidget(prog_bar)

                prog_lbl = QLabel("")
                prog_lbl.setProperty("class", "muted")
                prog_lbl.setWordWrap(True)
                prog_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
                prog_lbl.setVisible(False)
                c_lay.addWidget(prog_lbl)

                self._engine_cards_ui[item_id] = {
                    "progress": prog_bar,
                    "prog_label": prog_lbl,
                    "action_btn": install_btn,
                }
                self.online_cards_layout.addWidget(card)
                continue

            if prov_key == "silero_bot":
                if str(config.get("app_language", "en")).strip().lower() != "ru":
                    continue
                self._build_silero_bot_card(c_lay, item, item_id, title_str)
                self.online_cards_layout.addWidget(card)
                continue

            # Верхній рядок блоку (Google / ElevenLabs / OpenAI): Назва зліва, справа кнопка "Гайд" та "Видалити"
            top_r = QHBoxLayout()
            top_r.setSpacing(8)

            t_lbl = QLabel(f"<b>{title_str}</b>")
            t_lbl.setStyleSheet("color: #FFFFFF; font-size: 14px;")
            top_r.addWidget(t_lbl)
            top_r.addStretch(1)

            guide_btn = QPushButton(tr("ui_141", "📘 Гайд"))
            guide_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            guide_btn.setToolTip(tr("ui_f_002", "Як отримати API ключ для {title}", title=title_str))
            guide_btn.clicked.connect(lambda checked=False, p=prov_key: ApiGuideDialog(p, self).exec())
            top_r.addWidget(guide_btn)

            del_btn = QPushButton(tr("ui_137", "Видалити"))
            del_btn.setProperty("class", "dangerBtn")
            del_btn.setMinimumWidth(100)
            del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            del_btn.clicked.connect(lambda checked=False, iid=item_id: self._on_delete_online_item(iid))
            top_r.addWidget(del_btn)

            c_lay.addLayout(top_r)

            # Нижній рядок блоку: Поле вводу API ключа (приховане) + кнопка "Показати"
            key_r = QHBoxLayout()
            key_r.setSpacing(8)

            key_lbl = QLabel(tr("ui_143", "API ключ:"))
            key_lbl.setStyleSheet("color: #CBD5E1; font-size: 13px;")
            key_r.addWidget(key_lbl)

            key_edit = QLineEdit(saved_key)
            key_edit.setEchoMode(QLineEdit.EchoMode.Password)
            key_edit.setPlaceholderText(tr("ui_144", "Вставте ваш API ключ сюди..."))
            key_edit.textChanged.connect(lambda val, iid=item_id: self._on_online_item_key_changed(iid, val))
            key_r.addWidget(key_edit, 1)

            show_btn = QPushButton(tr("ui_145", "Показати"))
            show_btn.setMinimumWidth(96)
            show_btn.setCursor(Qt.CursorShape.PointingHandCursor)

            def _toggle_echo(checked: bool = False, edit: QLineEdit = key_edit, btn: QPushButton = show_btn) -> None:
                if edit.echoMode() == QLineEdit.EchoMode.Password:
                    edit.setEchoMode(QLineEdit.EchoMode.Normal)
                    btn.setText(tr("ui_146", "Сховати"))
                else:
                    edit.setEchoMode(QLineEdit.EchoMode.Password)
                    btn.setText(tr("ui_145", "Показати"))

            show_btn.clicked.connect(_toggle_echo)
            key_r.addWidget(show_btn)

            c_lay.addLayout(key_r)

            if prov_key == "fish_audio":
                fish_paid = bool(item.get("paid_enabled", False))
                fish_r = QHBoxLayout()
                fish_r.setSpacing(8)

                ref_lbl = QLabel(tr("ui_147", "ID голосу (опц.):"))
                ref_lbl.setStyleSheet("color: #CBD5E1; font-size: 13px;")
                ref_tip = tr(
                    "ui_148",
                    "Необов’язкове поле. Вставте reference_id з fish.audio, якщо хочете використати конкретний голос. Якщо залишити порожнім, використається голос із верхнього списку «Голос».",
                )
                ref_lbl.setToolTip(ref_tip)
                fish_r.addWidget(ref_lbl)

                ref_edit = QLineEdit(str(item.get("reference_id", "")))
                ref_edit.setPlaceholderText(tr("ui_149", "Порожньо = брати зі списку «Голос» (або вставте ID з fish.audio)"))
                ref_edit.setToolTip(ref_tip)
                ref_edit.textChanged.connect(
                    lambda val, iid=item_id: self._on_online_item_field_changed(iid, "reference_id", val.strip())
                )
                fish_r.addWidget(ref_edit, 1)

                model_lbl = QLabel(tr("ui_150", "Модель:"))
                model_lbl.setStyleSheet("color: #CBD5E1; font-size: 13px;")
                fish_r.addWidget(model_lbl)

                model_combo = QComboBox()
                model_combo.setMinimumWidth(165)
                fish_models = [("s2.1-pro-free", tr("ui_151", "s2.1-pro-free (Безкоштовна)"))]
                if fish_paid:
                    fish_models.extend([
                        ("s2.1-pro", tr("ui_152", "s2.1-pro (Платна) 💎")),
                        ("s2-pro", tr("ui_153", "s2-pro (Платна) 💎")),
                        ("s1", tr("ui_154", "s1 (Застаріла)")),
                        ("speech-1.6", "speech-1.6"),
                        ("speech-1.5", "speech-1.5"),
                    ])
                for m_val, m_disp in fish_models:
                    model_combo.addItem(m_disp, m_val)
                cur_m = str(item.get("model", "s2.1-pro-free")).strip() or "s2.1-pro-free"
                if not fish_paid and cur_m != "s2.1-pro-free":
                    cur_m = "s2.1-pro-free"
                m_idx = model_combo.findData(cur_m)
                if m_idx >= 0:
                    model_combo.setCurrentIndex(m_idx)
                model_combo.currentIndexChanged.connect(
                    lambda _idx, iid=item_id, cb=model_combo: self._on_online_item_field_changed(
                        iid, "model", str(cb.currentData() or "s2.1-pro-free")
                    )
                )
                fish_r.addWidget(model_combo)

                fetch_btn = QPushButton(tr("ui_139", "🔄 Оновити голоси"))
                fetch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                fetch_btn.setToolTip(tr("ui_155", "Завантажити ваші та популярні голоси з Fish Audio у список голосів"))
                fetch_btn.clicked.connect(
                    lambda checked=False, iid=item_id, btn=fetch_btn: self._start_fetch_provider_catalog(
                        iid, btn, silent=False
                    )
                )
                fish_r.addWidget(fetch_btn)

                fish_paid_btn = QPushButton(
                    tr("ui_156", "✅ Платний режим (s2.1-pro)") if fish_paid else tr("ui_157", "💳 Перейти на платно")
                )
                
                fish_paid_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                if fish_paid:
                    fish_paid_btn.setStyleSheet(
                        "QPushButton { background-color: #7C2D12; color: #FED7AA; border: 1px solid #EA580C; "
                        "border-radius: 6px; padding: 5px 10px; font-weight: 600; }"
                        "QPushButton:hover { background-color: #9A3412; }"
                    )
                else:
                    fish_paid_btn.setStyleSheet(
                        "QPushButton { background-color: #1E293B; color: #FBBF24; border: 1px solid #B45309; "
                        "border-radius: 6px; padding: 5px 10px; font-weight: 600; }"
                        "QPushButton:hover { background-color: #334155; }"
                    )
                fish_paid_btn.clicked.connect(
                    lambda checked=False, iid=item_id: self._on_toggle_fish_paid_mode(iid)
                )
                fish_r.addWidget(fish_paid_btn)

                c_lay.addLayout(fish_r)

                fish_hint = QLabel(
                    tr("ui_158", "💡 Безкоштовна модель: s2.1-pro-free. Поле «ID голосу» можна не заповнювати — "
                    "програма сама підтягне голоси у верхній список «Голос» (або вставте туди свій reference_id з сайту fish.audio).")
                )
                fish_hint.setStyleSheet("color: #94A3B8; font-size: 11.5px;")
                fish_hint.setWordWrap(True)
                c_lay.addWidget(fish_hint)

            if prov_key == "google":
                paid_enabled = bool(item.get("paid_enabled", False))
                g_row = QHBoxLayout()
                g_row.setSpacing(8)

                g_mod_lbl = QLabel(tr("ui_150", "Модель:"))
                g_mod_lbl.setStyleSheet("color: #CBD5E1; font-size: 13px;")
                g_row.addWidget(g_mod_lbl)

                g_mod_combo = QComboBox()
                g_mod_combo.setMinimumWidth(145)
                g_mod_combo.addItem(tr("ui_159", "Усі доступні"), "all")

                custom_v = list(item.get("custom_voices") or [])
                base_models = ["Chirp 3: HD", "WaveNet", "Standard", "Neural2", "Polyglot", "Studio"]
                if paid_enabled:
                    base_models.extend(["Chirp HD", "Journey", "News", "Casual"])
                seen_models = set(base_models)
                for cv in custom_v:
                    m_name = str(cv.get("model", "")).strip()
                    if not m_name:
                        continue
                    if not paid_enabled and not bool(cv.get("is_free", True)):
                        continue
                    if m_name not in seen_models:
                        seen_models.add(m_name)
                        base_models.append(m_name)

                for m_name in base_models:
                    g_mod_combo.addItem(m_name, m_name)

                cur_gm = str(item.get("model_filter", "all")).strip() or "all"
                gm_idx = g_mod_combo.findData(cur_gm)
                if gm_idx >= 0:
                    g_mod_combo.setCurrentIndex(gm_idx)

                def _on_g_model_changed(_idx: int, iid: str = item_id, cb: QComboBox = g_mod_combo) -> None:
                    self._on_online_item_field_changed(iid, "model_filter", str(cb.currentData() or "all"))
                    self._sync_tts_page_selectors()

                g_mod_combo.currentIndexChanged.connect(_on_g_model_changed)
                g_row.addWidget(g_mod_combo)

                g_fetch_btn = QPushButton(tr("ui_160", "🔄 Знайти голоси (API)"))
                g_fetch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                g_fetch_btn.setToolTip(
                    tr("ui_161", "Завантажити моделі й голоси Google Cloud TTS з урахуванням мови, статі та тарифу")
                )
                g_fetch_btn.clicked.connect(
                    lambda checked=False, iid=item_id, btn=g_fetch_btn: self._on_fetch_google_voices(iid, btn)
                )
                g_row.addWidget(g_fetch_btn)

                paid_btn = QPushButton(
                    tr("ui_162", "✅ Платний режим (Neural2 / Studio)") if paid_enabled else tr("ui_157", "💳 Перейти на платно")
                )
                
                paid_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                if paid_enabled:
                    paid_btn.setStyleSheet(
                        "QPushButton { background-color: #7C2D12; color: #FED7AA; border: 1px solid #EA580C; "
                        "border-radius: 6px; padding: 5px 10px; font-weight: 600; }"
                        "QPushButton:hover { background-color: #9A3412; }"
                    )
                    paid_btn.setToolTip(tr("ui_163", "Платні моделі (Neural2, Studio, Chirp HD тощо) увімкнено. Натисніть, щоб повернутися до безкоштовних."))
                else:
                    paid_btn.setStyleSheet(
                        "QPushButton { background-color: #1E293B; color: #FBBF24; border: 1px solid #B45309; "
                        "border-radius: 6px; padding: 5px 10px; font-weight: 600; }"
                        "QPushButton:hover { background-color: #334155; }"
                    )
                    paid_btn.setToolTip(tr("ui_164", "Відкрити платні моделі Google Cloud (Neural2, Studio, Polyglot тощо)"))

                paid_btn.clicked.connect(
                    lambda checked=False, iid=item_id, btn=g_fetch_btn: self._on_toggle_google_paid_mode(iid, btn)
                )
                g_row.addWidget(paid_btn)
                g_row.addStretch(1)

                c_lay.addLayout(g_row)

                from core.tts_manager import GOOGLE_FREE_SAFE_LIMITS, load_google_month_usage_map
                usage_map = load_google_month_usage_map()
                active_model_f = str(item.get("model_filter", "all")).strip()
                show_fam = active_model_f if active_model_f in GOOGLE_FREE_SAFE_LIMITS else "Chirp 3: HD"
                used_c = int(usage_map.get(show_fam, 0))
                lim_c = int(GOOGLE_FREE_SAFE_LIMITS.get(show_fam, 950_000))
                quota_lbl = QLabel(
                    tr("ui_f_003", "📊 Ліміт ({fam}, скидання 1-го числа): {used} / {lim} симв. (запас -50 000)", fam=show_fam, used=f"{used_c:,}", lim=f"{lim_c:,}")
                    + (tr("ui_165", " • Захист вимкнено (Платний режим)") if paid_enabled else tr("ui_166", " • Захист від перевищення увімкнено"))
                )
                quota_lbl.setStyleSheet("color: #A5B4FC; font-size: 11.5px;")
                quota_lbl.setWordWrap(True)
                c_lay.addWidget(quota_lbl)

            if prov_key in ("elevenlabs", "openai", "gemini_tts"):
                if prov_key == "elevenlabs":
                    el_ref_row = QHBoxLayout()
                    el_ref_row.setSpacing(8)
                    el_ref_lbl = QLabel(tr("ui_147", "ID голосу (опц.):"))
                    el_ref_lbl.setStyleSheet("color: #CBD5E1; font-size: 13px;")
                    el_ref_tip = tr(
                        "ui_el_ref_tip",
                        "Необов’язкове поле. Вставте Voice ID з elevenlabs.io (три крапки → Copy Voice ID), якщо хочете використати конкретний голос. Якщо залишити порожнім, використається голос із верхнього списку «Голос».",
                    )
                    el_ref_lbl.setToolTip(el_ref_tip)
                    el_ref_row.addWidget(el_ref_lbl)

                    el_ref_edit = QLineEdit(str(item.get("reference_id", "")))
                    el_ref_edit.setPlaceholderText(
                        tr("ui_el_ref_ph", "Порожньо = брати зі списку «Голос» (або вставте Voice ID з elevenlabs.io)")
                    )
                    el_ref_edit.setToolTip(el_ref_tip)
                    el_ref_edit.textChanged.connect(
                        lambda val, iid=item_id: self._on_online_item_field_changed(iid, "reference_id", val.strip())
                    )
                    el_ref_row.addWidget(el_ref_edit, 1)
                    c_lay.addLayout(el_ref_row)

                api_row = QHBoxLayout()
                api_row.setSpacing(8)

                m_lbl = QLabel(tr("ui_150", "Модель:"))
                m_lbl.setStyleSheet("color: #CBD5E1; font-size: 13px;")
                api_row.addWidget(m_lbl)

                m_combo = QComboBox()
                m_combo.setMinimumWidth(165)
                if prov_key == "elevenlabs":
                    default_models = ["eleven_flash_v2_5", "eleven_turbo_v2_5", "eleven_multilingual_v2"]
                elif prov_key == "gemini_tts":
                    default_models = [
                        "gemini-3.8-flash-lite-tts",
                        "gemini-3.8-flash-tts",
                        "gemini-3.1-flash-tts-preview",
                        "gemini-2.5-pro-preview-tts",
                        "gemini-2.5-flash-preview-tts",
                    ]
                else:
                    default_models = ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]
                disc_models = list(item.get("available_models") or [])
                all_models = disc_models if disc_models else default_models
                for m_val in all_models:
                    m_combo.addItem(m_val, m_val)
                if prov_key == "elevenlabs" and "eleven_flash_v2_5" in all_models:
                    default_pick = "eleven_flash_v2_5"
                else:
                    default_pick = all_models[0]
                cur_m = str(item.get("model", "")).strip() or default_pick
                m_idx = m_combo.findData(cur_m)
                if m_idx >= 0:
                    m_combo.setCurrentIndex(m_idx)
                m_combo.currentIndexChanged.connect(
                    lambda _idx, iid=item_id, cb=m_combo, dflt=all_models[0]: self._on_online_item_field_changed(
                        iid, "model", str(cb.currentData() or dflt)
                    )
                )
                api_row.addWidget(m_combo)

                u_fetch_btn = QPushButton(tr("ui_139", "🔄 Оновити голоси"))
                u_fetch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                u_fetch_btn.clicked.connect(
                    lambda checked=False, iid=item_id, btn=u_fetch_btn: self._start_fetch_provider_catalog(
                        iid, btn, silent=False
                    )
                )
                api_row.addWidget(u_fetch_btn)
                api_row.addStretch(1)
                c_lay.addLayout(api_row)

            c_cnt = len(list(item.get("custom_voices") or []))
            if c_cnt > 0:
                st_text = tr("ui_f_004", "🟢 Авто-пошук: знайдено {count} голосів з API", count=c_cnt)
            elif str(item.get("api_key", "")).strip():
                st_text = tr("ui_167", "🟡 Очікується авто-пошук голосів з API...")
            else:
                st_text = tr("ui_168", "⚪ Введіть API ключ — програма сама знайде доступні голоси та моделі")
            st_lbl = QLabel(st_text)
            st_lbl.setStyleSheet("color: #94A3B8; font-size: 12px;")
            self._online_status_labels[item_id] = st_lbl
            c_lay.addWidget(st_lbl)

            self.online_cards_layout.addWidget(card)

    def _on_install_online_edge(self, item_id: str) -> None:
        if not self._ensure_language_confirmed():
            return
        ui = self._engine_cards_ui.get(item_id)
        if ui:
            ui["progress"].setVisible(True)
            ui["progress"].setValue(5)
            ui["prog_label"].setVisible(True)
            ui["prog_label"].setText(tr("ui_169", "Підготовка до встановлення edge-tts..."))
            btn: QPushButton = ui["action_btn"]  # type: ignore
            btn.setEnabled(False)
            btn.setText(tr("ui_170", "⏳ Встановлення..."))
        tts_manager.install_edge_tts_library(item_id)

    def _build_silero_bot_card(self, c_lay: QVBoxLayout, item: Dict[str, Any], item_id: str, title_str: str) -> None:
        """Будує картку налаштувань та керування @silero_voice_bot (аналогічно TtsSileroCard з TTSBard)."""
        from core.telegram_silero import SESSION_FILE, is_telethon_installed, telegram_silero

        tg_installed = is_telethon_installed()
        tg_status = telegram_silero.get_cached_status()
        connected = bool(tg_status.get("connected", False))
        user_info = tg_status.get("user") or {}

        # Верхній рядок: Назва + Гайд + Видалити
        top_r = QHBoxLayout()
        top_r.setSpacing(8)
        t_lbl = QLabel(f"<b>{title_str}</b>")
        t_lbl.setStyleSheet("color: #FFFFFF; font-size: 14px;")
        top_r.addWidget(t_lbl)
        top_r.addStretch(1)

        guide_btn = QPushButton(tr("ui_141", "📘 Гайд"))
        guide_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        guide_btn.setToolTip(tr("tg.card.guide_tooltip", "Як отримати API ID/Hash і підключити @silero_voice_bot"))
        guide_btn.clicked.connect(lambda checked=False: ApiGuideDialog("silero_bot", self).exec())
        top_r.addWidget(guide_btn)

        del_btn = QPushButton(tr("ui_137", "Видалити"))
        del_btn.setProperty("class", "dangerBtn")
        del_btn.setMinimumWidth(100)
        del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        del_btn.setToolTip(tr("tg.card.delete_tooltip", "Видалити Silero Bot (Telegram) зі списку"))
        del_btn.clicked.connect(lambda checked=False, iid=item_id: self._on_delete_online_item(iid))
        top_r.addWidget(del_btn)
        c_lay.addLayout(top_r)

        if not tg_installed:
            inst_r = QHBoxLayout()
            inst_r.setSpacing(8)
            inst_lbl = QLabel(tr("tg.card.telethon_needed", "⚪ Для роботи з Telegram потрібна бібліотека Telethon"))
            inst_lbl.setStyleSheet("color: #CBD5E1; font-size: 12.5px;")
            inst_r.addWidget(inst_lbl, 1)

            inst_btn = QPushButton(tr("tg.card.install_telethon", "📦 Встановити Telethon"))
            inst_btn.setProperty("class", "primaryBtn")
            inst_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            inst_btn.clicked.connect(lambda checked=False, iid=item_id: self._on_install_telethon(iid))
            inst_r.addWidget(inst_btn)
            c_lay.addLayout(inst_r)

            prog_bar = QProgressBar()
            prog_bar.setRange(0, 100)
            prog_bar.setValue(0)
            prog_bar.setVisible(False)
            c_lay.addWidget(prog_bar)

            prog_lbl = QLabel("")
            prog_lbl.setProperty("class", "muted")
            prog_lbl.setWordWrap(True)
            prog_lbl.setVisible(False)
            c_lay.addWidget(prog_lbl)

            self._engine_cards_ui[item_id] = {
                "progress": prog_bar,
                "prog_label": prog_lbl,
                "action_btn": inst_btn,
            }
            return

        # Статус підключення Telegram + кнопки Підключити / Перепідключити / Вийти
        conn_r = QHBoxLayout()
        conn_r.setSpacing(8)
        if connected:
            name_parts = [str(user_info.get("first_name") or "").strip(), str(user_info.get("last_name") or "").strip()]
            full_name = " ".join(p for p in name_parts if p).strip() or "Telegram"
            uname = str(user_info.get("username") or "").strip()
            disp = f"{full_name} (@{uname})" if uname else full_name
            conn_lbl = QLabel(tr("tg.card.connected_as", "🟢 Підключено: {user}", user=disp))
            conn_lbl.setStyleSheet("color: #34D399; font-size: 13px; font-weight: 600;")
            conn_r.addWidget(conn_lbl, 1)

            reconn_btn = QPushButton(tr("tg.card.reconnect", "🔄 Перепідключити"))
            reconn_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            reconn_btn.clicked.connect(lambda checked=False, iid=item_id: self._on_silero_reconnect(iid))
            conn_r.addWidget(reconn_btn)

            logout_btn = QPushButton(tr("tg.card.sign_out", "Вийти"))
            logout_btn.setProperty("class", "dangerBtn")
            logout_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            logout_btn.clicked.connect(lambda checked=False, iid=item_id: self._on_silero_sign_out(iid))
            conn_r.addWidget(logout_btn)
        else:
            if self._silero_connecting:
                conn_lbl = QLabel(tr("tg.card.connecting", "⏳ Підключення до Telegram..."))
                conn_lbl.setStyleSheet("color: #A5B4FC; font-size: 13px; font-weight: 600;")
            else:
                conn_lbl = QLabel(
                    tr("tg.card.session_saved_disconnected", "🟡 Збережена сесія (не підключено)")
                    if SESSION_FILE.exists()
                    else tr("tg.card.not_connected", "⚪ Telegram не підключено")
                )
                conn_lbl.setStyleSheet("color: #CBD5E1; font-size: 13px;")
            conn_r.addWidget(conn_lbl, 1)

            if SESSION_FILE.exists() and telegram_silero.has_effective_credentials():
                reconn_btn = QPushButton(tr("tg.card.reconnect", "🔄 Перепідключити"))
                reconn_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                reconn_btn.setEnabled(not self._silero_connecting)
                reconn_btn.clicked.connect(lambda checked=False, iid=item_id: self._on_silero_reconnect(iid))
                conn_r.addWidget(reconn_btn)

            conn_btn = QPushButton(tr("tg.card.connect", "🔗 Підключити Telegram"))
            conn_btn.setProperty("class", "primaryBtn")
            conn_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            conn_btn.clicked.connect(lambda checked=False, iid=item_id: self._on_silero_connect_dialog(iid))
            conn_r.addWidget(conn_btn)
        c_lay.addLayout(conn_r)

        if self._silero_error:
            err_lbl = QLabel(f"⚠️ {self._silero_error}")
            err_lbl.setWordWrap(True)
            err_lbl.setStyleSheet("color: #FCA5A5; font-size: 12px;")
            c_lay.addWidget(err_lbl)

        # Налаштування проксі та таймауту (як у TTSBard TelegramConnectionStatus)
        cfg_r = QHBoxLayout()
        cfg_r.setSpacing(8)

        pm_lbl = QLabel(tr("tg.card.proxy", "Проксі:"))
        pm_lbl.setStyleSheet("color: #CBD5E1; font-size: 12.5px;")
        cfg_r.addWidget(pm_lbl)

        pm_combo = QComboBox()
        pm_combo.setMinimumWidth(145)
        pm_combo.addItem(tr("tg.card.proxy_auto", "Авто (з налаштувань)"), "auto")
        pm_combo.addItem(tr("tg.card.proxy_none", "Без проксі"), "none")
        pm_combo.addItem("SOCKS5", "socks5")
        cur_pm = str(item.get("proxy_mode", "auto") or "auto").strip().lower()
        pm_idx = pm_combo.findData(cur_pm)
        if pm_idx >= 0:
            pm_combo.setCurrentIndex(pm_idx)

        purl_edit = QLineEdit(str(item.get("proxy_url", "") or ""))
        purl_edit.setPlaceholderText("socks5://127.0.0.1:1080")
        purl_edit.setVisible(cur_pm == "socks5")
        purl_edit.textChanged.connect(
            lambda val, iid=item_id: self._on_online_item_field_changed(iid, "proxy_url", val.strip())
        )

        def _on_pm_changed(_idx: int, iid: str = item_id, cb: QComboBox = pm_combo, ed: QLineEdit = purl_edit) -> None:
            mode = str(cb.currentData() or "auto")
            self._on_online_item_field_changed(iid, "proxy_mode", mode)
            ed.setVisible(mode == "socks5")

        pm_combo.currentIndexChanged.connect(_on_pm_changed)
        cfg_r.addWidget(pm_combo)
        cfg_r.addWidget(purl_edit, 1)

        to_lbl = QLabel(tr("tg.card.timeout", "Таймаут (с):"))
        to_lbl.setStyleSheet("color: #CBD5E1; font-size: 12.5px;")
        cfg_r.addWidget(to_lbl)

        to_spin = QSpinBox()
        to_spin.setRange(10, 300)
        to_spin.setValue(int(item.get("timeout_sec", 60) or 60))
        to_spin.valueChanged.connect(
            lambda val, iid=item_id: self._on_online_item_field_changed(iid, "timeout_sec", int(val))
        )
        cfg_r.addWidget(to_spin)
        if cur_pm != "socks5":
            cfg_r.addStretch(1)
        c_lay.addLayout(cfg_r)

        if not connected:
            return

        # Рядок лімітів (/limits)
        lim_r = QHBoxLayout()
        lim_r.setSpacing(8)
        if self._silero_limits:
            v_lim = self._silero_limits.get("voices", "—")
            r_ts = self._silero_limits.get("reset_timestamp", "")
            lim_txt = tr("tg.card.limits_val", "📊 Ліміт символів: {voices}", voices=v_lim)
            if r_ts:
                lim_txt += tr("tg.card.limits_reset", " · оновлення {when}", when=r_ts)
        else:
            lim_txt = tr("tg.card.limits_empty", "📊 Ліміт символів: —")
        lim_lbl = QLabel(lim_txt)
        lim_lbl.setStyleSheet("color: #A5B4FC; font-size: 12.5px;")
        lim_r.addWidget(lim_lbl)
        lim_r.addWidget(
            HelpButton(
                tr("tg.card.limits_help_title", "Денний ліміт @silero_voice_bot"),
                tr(
                    "tg.card.limits_help_body",
                    "<b>Що це показує?</b><br>"
                    "Залишок безкоштовних озвучок і символів на сьогодні у боті @silero_voice_bot та час оновлення ліміту.<br><br>"
                    "• Ліміт прив'язаний до вашого Telegram-акаунту.<br>"
                    "• Натисніть <b>«🔄 Оновити ліміти»</b>, щоб побачити актуальний залишок.<br><br>"
                    "<i>Коли ліміт вичерпано, можна перемкнутися на Запасний TTS.</i>",
                ),
            )
        )
        lim_r.addStretch(1)

        lim_btn = QPushButton(tr("tg.card.refresh_limits", "🔄 Оновити ліміти"))
        lim_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        lim_btn.setToolTip(tr("tg.card.refresh_limits_tip", "Запитати у бота актуальний залишок добового ліміту (/limits)"))
        lim_btn.clicked.connect(lambda checked=False, btn=lim_btn: self._on_silero_refresh_limits(btn))
        lim_r.addWidget(lim_btn)
        c_lay.addLayout(lim_r)

        # Рядок поточного голосу бота (/speaker) + додавання збережених голосів
        vh_r = QHBoxLayout()
        vh_r.setSpacing(8)
        cur_v = self._silero_current_voice
        if cur_v and cur_v.get("id"):
            v_str = tr("tg.card.cur_voice", "🎙️ Поточний голос бота: {id} ({name})", id=cur_v["id"], name=cur_v.get("name", cur_v["id"]))
        else:
            v_str = tr("tg.card.saved_voices_hdr", "🎙️ Збережені голоси @silero_voice_bot:")
        vh_lbl = QLabel(v_str)
        vh_lbl.setStyleSheet("color: #E2E8F0; font-size: 12.5px; font-weight: 600;")
        vh_lbl.setWordWrap(True)
        vh_r.addWidget(vh_lbl, 1)
        vh_r.addWidget(
            HelpButton(
                tr("tg.card.voices_help_title", "Голоси @silero_voice_bot"),
                tr(
                    "tg.card.voices_help_body",
                    "<b>Як керувати голосами бота:</b><br><br>"
                    "• <b>🔄 З бота (/speaker)</b> — підтягує голос, який зараз вибрано у вашому чаті з @silero_voice_bot.<br>"
                    "• <b>+ Додати голос</b> — додає інший голос за кодом (список кодів дивіться у бота командою <code>/speakers</code>, напр. <code>hamster_clerk</code>, <code>rene</code>, <code>baya</code>).<br>"
                    "• У списку нижче можна у 1 клік обрати активний голос або видалити зайві.",
                ),
            )
        )

        sync_v_btn = QPushButton(tr("tg.card.sync_bot_voice", "🔄 З бота (/speaker)"))
        sync_v_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        sync_v_btn.setToolTip(tr("tg.card.sync_bot_voice_tip", "Підтягнути поточний активний голос із бота @silero_voice_bot"))
        sync_v_btn.clicked.connect(lambda checked=False, iid=item_id, btn=sync_v_btn: self._on_silero_refresh_voice(iid, btn))
        vh_r.addWidget(sync_v_btn)

        add_v_btn = QPushButton(tr("tg.card.add_voice", "+ Додати голос"))
        add_v_btn.setProperty("class", "primaryBtn")
        add_v_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_v_btn.setToolTip(tr("tg.card.add_voice_tip", "Додати голос @silero_voice_bot за кодом (напр. hamster_clerk, rene, baya)"))
        add_v_btn.clicked.connect(lambda checked=False, iid=item_id: self._on_silero_add_voice(iid))
        vh_r.addWidget(add_v_btn)
        c_lay.addLayout(vh_r)

        saved_voices = list(item.get("saved_voices") or [])
        if saved_voices:
            sv_box = QFrame()
            sv_box.setObjectName("sileroVoicesBox")
            sv_box.setStyleSheet(
                "#sileroVoicesBox { background-color: #191E2D; border: 1px solid #2D354D; border-radius: 8px; }"
            )
            sv_lay = QVBoxLayout(sv_box)
            sv_lay.setContentsMargins(10, 8, 10, 8)
            sv_lay.setSpacing(6)

            sv_title = QLabel(tr("tg.card.saved_voices_list", "Збережені голоси для швидкого перемикання:"))
            sv_title.setStyleSheet("color: #94A3B8; font-size: 11.5px; font-weight: 600;")
            sv_lay.addWidget(sv_title)

            cur_bot_id = str((cur_v or {}).get("id", "")).strip().lower()
            for sv in saved_voices:
                vid = str(sv.get("id", "")).strip()
                if not vid:
                    continue
                vdesc = str(sv.get("description", "")).strip()
                vrow = QHBoxLayout()
                vrow.setSpacing(8)
                is_cur = vid.lower() == cur_bot_id
                prefix = "✅ " if is_cur else "• "
                vlbl = QLabel(f"{prefix}{vid}" + (f" ({vdesc})" if vdesc else ""))
                vlbl.setStyleSheet("color: #34D399; font-weight: 600;" if is_cur else "color: #CBD5E1;")
                vrow.addWidget(vlbl, 1)

                if is_cur:
                    use_btn = QPushButton(tr("tg.card.active_voice", "✓ Активний"))
                    use_btn.setEnabled(False)
                    use_btn.setToolTip(tr("tg.card.active_voice_tip", "Цей голос зараз активний у @silero_voice_bot"))
                    use_btn.setStyleSheet(
                        "QPushButton { background-color: #13392D; color: #34D399; border: 1px solid #1F6F54; "
                        "border-radius: 6px; padding: 4px 10px; font-weight: 600; }"
                    )
                else:
                    use_btn = QPushButton(tr("tg.card.use_voice", "Обрати"))
                    use_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    use_btn.setToolTip(tr("tg.card.use_voice_tip", "Увімкнути цей голос у боті @silero_voice_bot"))
                    use_btn.clicked.connect(lambda checked=False, iid=item_id, vc=vid, btn=use_btn: self._on_silero_select_voice(iid, vc, btn))
                vrow.addWidget(use_btn)

                rm_btn = QPushButton(tr("tg.card.remove_voice", "Прибрати"))
                rm_btn.setProperty("class", "dangerBtn")
                rm_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                rm_btn.setToolTip(tr("tg.card.remove_voice_tip", "Прибрати цей голос зі списку збережених"))
                rm_btn.clicked.connect(lambda checked=False, iid=item_id, vc=vid: self._on_silero_remove_voice(iid, vc))
                vrow.addWidget(rm_btn)
                sv_lay.addLayout(vrow)

            c_lay.addWidget(sv_box)
        else:
            empty_lbl = QLabel(
                tr("tg.card.no_saved_voices", "Список голосів порожній — натисніть «🔄 З бота (/speaker)» або «+ Додати голос» (код виду hamster_clerk, rene, baya).")
            )
            empty_lbl.setWordWrap(True)
            empty_lbl.setStyleSheet("color: #94A3B8; font-size: 11.5px;")
            c_lay.addWidget(empty_lbl)

    def _on_install_telethon(self, item_id: str) -> None:
        ui = self._engine_cards_ui.get(item_id)
        if ui:
            ui["progress"].setVisible(True)
            ui["progress"].setValue(10)
            ui["prog_label"].setVisible(True)
            ui["prog_label"].setText(tr("tg.dl.telethon_installing", "Встановлення бібліотеки Telethon..."))
            btn: QPushButton = ui["action_btn"]  # type: ignore
            btn.setEnabled(False)
        tts_manager.install_telethon_library(item_id)

    def _auto_connect_silero_bot(self) -> None:
        if str(config.get("app_language", "en")).strip().lower() != "ru":
            return
        from core.telegram_silero import SESSION_FILE, is_telethon_installed, telegram_silero

        items = [x for x in config.get("online_items", []) if str(x.get("provider", "")) == "silero_bot"]
        if not items or not is_telethon_installed() or not SESSION_FILE.exists():
            return
        item = items[0]
        if not telegram_silero.has_effective_credentials():
            return

        self._silero_connecting = True
        self._rebuild_online_cards()

        def _done(res: Any, err: str) -> None:
            self._silero_connecting = False
            if err:
                self._silero_error = err
            elif isinstance(res, dict) and res.get("connected"):
                self._silero_error = ""
                self._silero_fetch_initial_meta(str(item.get("id", "")))
            self._rebuild_online_cards()
            self._sync_tts_page_selectors()

        run_in_background(self, telegram_silero.connect_existing, _done)

    def _silero_fetch_initial_meta(self, item_id: str) -> None:
        from core.telegram_silero import telegram_silero

        def _done_limits(res: Any, _err: str) -> None:
            if isinstance(res, dict):
                self._silero_limits = res
                self._rebuild_online_cards()

        def _done_voice(res: Any, _err: str) -> None:
            if isinstance(res, dict) and res.get("id"):
                self._silero_current_voice = res
                self._ensure_silero_voice_saved(item_id, res["id"], res.get("name", ""))
                self._rebuild_online_cards()
                self._sync_tts_page_selectors()

        run_in_background(self, telegram_silero.get_limits, _done_limits)
        run_in_background(self, telegram_silero.get_current_voice, _done_voice)

    def _ensure_silero_voice_saved(self, item_id: str, voice_id: str, desc: str = "") -> None:
        vid = str(voice_id or "").strip().lower()
        if not vid:
            return
        items = list(config.get("online_items", []))
        changed = False
        for x in items:
            if str(x.get("id")) == item_id:
                sv = list(x.get("saved_voices") or [])
                if not any(str(v.get("id", "")).strip().lower() == vid for v in sv):
                    sv.append({"id": vid, "description": desc})
                    x["saved_voices"] = sv
                    changed = True
                break
        if changed:
            config.set("online_items", items)

    def _on_silero_connect_dialog(self, item_id: str) -> None:
        dlg = TelegramAuthDialog(item_id, self)
        if dlg.exec() == TelegramAuthDialog.DialogCode.Accepted:
            self._silero_error = ""
            config.set("active_engine", f"online:{item_id}")
            self._silero_fetch_initial_meta(item_id)
            self._rebuild_online_cards()
            self._sync_tts_page_selectors()

    def _on_silero_reconnect(self, item_id: str) -> None:
        if self._silero_connecting:
            return
        from core.telegram_silero import telegram_silero

        self._silero_error = ""
        self._silero_connecting = True
        self._rebuild_online_cards()

        def _done(res: Any, err: str) -> None:
            self._silero_connecting = False
            if err:
                self._silero_error = err
            elif isinstance(res, dict) and res.get("connected"):
                self._silero_fetch_initial_meta(item_id)
            elif isinstance(res, dict) and res.get("state") == "auth_required":
                self._rebuild_online_cards()
                self._on_silero_connect_dialog(item_id)
                return
            self._rebuild_online_cards()
            self._sync_tts_page_selectors()

        run_in_background(self, telegram_silero.connect_existing, _done)

    def _on_silero_sign_out(self, item_id: str) -> None:
        ans = QMessageBox.question(
            self,
            tr("tg.card.sign_out_title", "Вийти з Telegram"),
            tr("tg.card.sign_out_confirm", "Завершити поточну Telegram-сесію та видалити локальний файл сесії?"),
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        from core.telegram_silero import telegram_silero

        def _done(_res: Any, err: str) -> None:
            self._silero_limits = None
            self._silero_current_voice = None
            self._silero_error = err or ""
            self._rebuild_online_cards()
            self._sync_tts_page_selectors()

        run_in_background(self, telegram_silero.sign_out, _done)

    def _on_silero_refresh_limits(self, btn: QPushButton) -> None:
        from core.telegram_silero import telegram_silero

        btn.setEnabled(False)

        def _done(res: Any, err: str) -> None:
            btn.setEnabled(True)
            if err:
                self._silero_error = err
            elif isinstance(res, dict):
                self._silero_error = ""
                self._silero_limits = res
            self._rebuild_online_cards()

        run_in_background(self, telegram_silero.get_limits, _done)

    def _on_silero_refresh_voice(self, item_id: str, btn: QPushButton) -> None:
        from core.telegram_silero import telegram_silero

        btn.setEnabled(False)

        def _done(res: Any, err: str) -> None:
            btn.setEnabled(True)
            if err:
                self._silero_error = err
            elif isinstance(res, dict) and res.get("id"):
                self._silero_error = ""
                self._silero_current_voice = res
                self._ensure_silero_voice_saved(item_id, res["id"], res.get("name", ""))
                self._sync_tts_page_selectors()
            self._rebuild_online_cards()

        run_in_background(self, telegram_silero.get_current_voice, _done)

    def _refresh_silero_ui_deferred(self) -> None:
        self._rebuild_online_cards()
        self._sync_tts_page_selectors()

    def _on_tg_bot_voice_detected(self, voice_id: str) -> None:
        vid = str(voice_id or "").strip().lower()
        if not vid:
            return
        items = [x for x in config.get("online_items", []) if str(x.get("provider", "")) == "silero_bot"]
        if not items:
            return
        item = items[0]
        item_id = str(item.get("id", ""))
        self._ensure_silero_voice_saved(item_id, vid, "")
        sv = next((v for v in (item.get("saved_voices") or []) if str(v.get("id", "")).lower() == vid), {})
        desc = str(sv.get("description") or "").strip()
        self._silero_current_voice = {"id": vid, "name": desc or vid}
        config.update({"active_engine": f"online:{item_id}", "active_voice": vid})
        active_dlg = getattr(self, "_active_silero_add_dlg", None)
        if active_dlg is not None:
            self._active_silero_add_dlg = None
            QTimer.singleShot(0, active_dlg.accept)
        QTimer.singleShot(0, self._refresh_silero_ui_deferred)

    def _on_silero_add_voice(self, item_id: str) -> None:
        item = next((x for x in config.get("online_items", []) if str(x.get("id")) == item_id), {})
        existing = [str(v.get("id", "")) for v in (item.get("saved_voices") or [])]
        dlg = AddSileroVoiceDialog(existing, self)
        self._active_silero_add_dlg = dlg
        try:
            res_code = dlg.exec()
        finally:
            if getattr(self, "_active_silero_add_dlg", None) is dlg:
                self._active_silero_add_dlg = None
        if res_code == AddSileroVoiceDialog.DialogCode.Accepted and dlg.added_voice:
            vid = dlg.added_voice["id"]
            desc = dlg.added_voice.get("description", "")
            self._ensure_silero_voice_saved(item_id, vid, desc)
            self._silero_current_voice = {"id": vid, "name": desc or vid}
            config.update({"active_engine": f"online:{item_id}", "active_voice": vid})
            QTimer.singleShot(0, self._refresh_silero_ui_deferred)

    def _on_silero_select_voice(self, item_id: str, voice_code: str, btn: QPushButton) -> None:
        from core.telegram_silero import telegram_silero

        btn.setEnabled(False)

        def _done(res: Any, err: str) -> None:
            try:
                btn.setEnabled(True)
            except RuntimeError:
                pass
            if err:
                QMessageBox.warning(self, tr("common.error", "Помилка"), err)
                return
            ok, msg = res
            if not ok:
                QMessageBox.warning(self, tr("common.error", "Помилка"), msg)
                return
            item = next((x for x in config.get("online_items", []) if str(x.get("id")) == item_id), {})
            sv = next((v for v in (item.get("saved_voices") or []) if str(v.get("id", "")).lower() == msg.lower()), {})
            self._silero_current_voice = {"id": msg, "name": str(sv.get("description") or msg)}
            config.update({"active_engine": f"online:{item_id}", "active_voice": msg})
            QTimer.singleShot(0, self._refresh_silero_ui_deferred)

        run_in_background(self, lambda: telegram_silero.set_speaker(voice_code), _done)

    def _on_silero_remove_voice(self, item_id: str, voice_code: str) -> None:
        if QMessageBox.question(self, tr("common.confirm", "Підтвердження"), tr("confirm.delete_voice", "Точно прибрати цей голос зі списку?")) != QMessageBox.StandardButton.Yes:
            return
        items = list(config.get("online_items", []))
        for x in items:
            if str(x.get("id")) == item_id:
                x["saved_voices"] = [
                    v for v in (x.get("saved_voices") or [])
                    if str(v.get("id", "")).strip().lower() != voice_code.strip().lower()
                ]
                break
        config.set("online_items", items)
        QTimer.singleShot(0, self._refresh_silero_ui_deferred)

    def _on_add_online_provider(self) -> None:
        prov_key = str(self.online_add_combo.currentData())
        prov_spec = ONLINE_PROVIDERS.get(prov_key)
        if not prov_spec:
            return
        items: List[Dict[str, Any]] = list(config.get("online_items", []))
        # Each online provider can be configured only once.
        if any(str(item.get("provider", "")).strip() == prov_key for item in items if isinstance(item, dict)):
            self._populate_online_add_combo()
            return

        base_title = prov_spec["title"]
        same_count = sum(1 for x in items if x.get("provider") == prov_key)
        title = base_title if same_count == 0 else f"{base_title} #{same_count + 1}"
        new_id = f"{prov_key}_{int(QTimer().remainingTime())}_{len(items) + 1}_{id(title) % 10000}"

        new_entry: Dict[str, Any] = {
            "id": new_id,
            "provider": prov_key,
            "title": title,
            "api_key": "",
        }
        if prov_key == "edge":
            new_entry["installed"] = False
        elif prov_key == "silero_bot":
            new_entry["proxy_mode"] = "auto"
            new_entry["proxy_url"] = ""
            new_entry["timeout_sec"] = 60
            new_entry["saved_voices"] = []

        items.append(new_entry)
        config.set("online_items", items)
        if prov_key not in ("edge", "silero_bot"):
            config.set("active_engine", f"online:{new_id}")
        self._populate_online_add_combo()
        self._rebuild_online_cards()
        self._sync_tts_page_selectors()

    def _on_delete_online_item(self, item_id: str) -> None:
        all_items = list(config.get("online_items", []))
        removed = next((x for x in all_items if str(x.get("id")) == item_id), None)
        if removed and QMessageBox.question(self, tr("common.confirm", "Підтвердження"), tr("confirm.delete_service", "Точно видалити сервіс «{name}»?", name=str(removed.get("title", removed.get("provider", ""))))) != QMessageBox.StandardButton.Yes:
            return
        old_t = self._api_debounce_timers.pop(item_id, None)
        if old_t is not None:
            old_t.stop()
            old_t.deleteLater()
        items: List[Dict[str, Any]] = [x for x in all_items if str(x.get("id")) != item_id]
        config.set("online_items", items)
        self._populate_online_add_combo()
        if removed and removed.get("provider") == "silero_bot":
            from core.telegram_silero import telegram_silero
            run_in_background(self, telegram_silero.disconnect, lambda _r, _e: None)
        self._rebuild_online_cards()
        self._sync_tts_page_selectors()

    def _on_online_item_key_changed(self, item_id: str, new_key: str) -> None:
        clean_key = new_key.strip()
        items: List[Dict[str, Any]] = list(config.get("online_items", []))
        for x in items:
            if str(x.get("id")) == item_id:
                x["api_key"] = clean_key
                break
        config.set("online_items", items)

        if len(clean_key) >= 12:
            st_lbl = self._online_status_labels.get(item_id)
            if st_lbl is not None:
                try:
                    st_lbl.setText(tr("ui_171", "⏳ Автоматичний пошук голосів і моделей через API..."))
                except RuntimeError:
                    pass
            old_t = self._api_debounce_timers.get(item_id)
            if old_t is not None:
                old_t.stop()
            t = QTimer(self)
            t.setSingleShot(True)
            t.timeout.connect(lambda iid=item_id: self._start_fetch_provider_catalog(iid, None, silent=True))
            self._api_debounce_timers[item_id] = t
            t.start(900)

    def _on_online_item_field_changed(self, item_id: str, field: str, value: Any) -> None:
        items: List[Dict[str, Any]] = list(config.get("online_items", []))
        for x in items:
            if str(x.get("id")) == item_id:
                x[field] = value
                break
        config.set("online_items", items)

    def _on_fetch_fish_audio_voices(self, item_id: str, btn: QPushButton) -> None:
        items: List[Dict[str, Any]] = list(config.get("online_items", []))
        item = next((x for x in items if str(x.get("id")) == item_id), None)
        api_key = str((item or {}).get("api_key", "")).strip()
        if not api_key:
            QMessageBox.warning(self, "Fish Audio API", tr("ui_172", "Спочатку введіть API ключ Fish Audio."))
            return

        btn.setEnabled(False)
        btn.setText(tr("ui_173", "⏳ Завантаження..."))

        def _worker() -> None:
            try:
                voices = tts_manager.fetch_fish_audio_voices(api_key)
                self.fish_voices_fetched.emit(item_id, voices, "", btn)
            except Exception as exc:
                self.fish_voices_fetched.emit(item_id, [], str(exc), btn)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_fish_voices_fetched(self, item_id: str, voices: object, err: str, btn: object) -> None:
        if isinstance(btn, QPushButton):
            try:
                btn.setEnabled(True)
                btn.setText(tr("ui_139", "🔄 Оновити голоси"))
            except RuntimeError:
                pass

        if err:
            QMessageBox.warning(self, "Fish Audio API", tr("ui_f_005", "Не вдалося завантажити голоси:\n{err}", err=err))
            return
        voice_list = list(voices) if isinstance(voices, list) else []
        if not voice_list:
            QMessageBox.information(self, "Fish Audio API", tr("ui_174", "Список голосів порожній."))
            return

        items: List[Dict[str, Any]] = list(config.get("online_items", []))
        for x in items:
            if str(x.get("id")) == item_id:
                x["custom_voices"] = voice_list
                break
        config.set("online_items", items)
        self._sync_tts_page_selectors()
        QMessageBox.information(
            self,
            "Fish Audio API",
            tr("ui_f_006", "Завантажено {count} голосів. Вони вже доступні у випадаючому списку «Голос».", count=len(voice_list)),
        )

    def _on_fetch_google_voices(self, item_id: str, btn: QPushButton) -> None:
        items: List[Dict[str, Any]] = list(config.get("online_items", []))
        item = next((x for x in items if str(x.get("id")) == item_id), None)
        api_key = str((item or {}).get("api_key", "")).strip()
        if not api_key:
            QMessageBox.warning(self, "Google Cloud TTS API", tr("ui_175", "Спочатку введіть API ключ Google Cloud TTS."))
            return

        btn.setEnabled(False)
        btn.setText(tr("ui_176", "⏳ Пошук..."))

        def _worker() -> None:
            try:
                voices = tts_manager.fetch_google_cloud_voices(api_key)
                self.google_voices_fetched.emit(item_id, voices, "", btn)
            except Exception as exc:
                self.google_voices_fetched.emit(item_id, [], str(exc), btn)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_google_voices_fetched(self, item_id: str, voices: object, err: str, btn: object) -> None:
        if isinstance(btn, QPushButton):
            try:
                btn.setEnabled(True)
                btn.setText(tr("ui_160", "🔄 Знайти голоси (API)"))
            except RuntimeError:
                pass

        if err:
            QMessageBox.warning(self, "Google Cloud TTS API", tr("ui_f_005", "Не вдалося завантажити голоси:\n{err}", err=err))
            return
        voice_list = list(voices) if isinstance(voices, list) else []
        if not voice_list:
            QMessageBox.information(self, "Google Cloud TTS API", tr("ui_174", "Список голосів порожній."))
            return

        items: List[Dict[str, Any]] = list(config.get("online_items", []))
        paid_enabled = False
        for x in items:
            if str(x.get("id")) == item_id:
                x["custom_voices"] = voice_list
                paid_enabled = bool(x.get("paid_enabled", False))
                break
        config.set("online_items", items)
        self._rebuild_online_cards()
        self._sync_tts_page_selectors()

        matched_count = len(tts_manager.get_voices_for_engine(f"online:{item_id}"))
        mode_note = (
            tr("ui_177", "Увімкнено платний режим (без блокування за лімітом, доступні також Chirp HD, Journey тощо).")
            if paid_enabled
            else tr("ui_178", "Увімкнено захист безкоштовного ліміту (запас -50 000 симв., скидання 1-го числа): "
                 "Chirp 3: HD / Neural2 / Polyglot / Studio — 950k, WaveNet / Standard — 3.95M.")
        )
        QMessageBox.information(
            self,
            "Google Cloud TTS API",
            tr("ui_f_007", "Отримано з API: {total} голосів.\nДоступно за поточними фільтрами (мова/стать/режим): {matched}.\n\n{mode_note}", total=len(voice_list), matched=matched_count, mode_note=mode_note),
        )

    def _on_toggle_google_paid_mode(self, item_id: str, fetch_btn: QPushButton) -> None:
        items: List[Dict[str, Any]] = list(config.get("online_items", []))
        item = next((x for x in items if str(x.get("id")) == item_id), None)
        if not item:
            return

        currently_paid = bool(item.get("paid_enabled", False))
        if not currently_paid:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle(tr("ui_179", "Увімкнути платні моделі Google Cloud TTS?"))
            box.setText(
                tr("ui_180", "Ви збираєтесь вимкнути захист безкоштовного ліміту Google Cloud TTS "
                "та дозволити платне використання.")
            )
            box.setInformativeText(
                tr("ui_181", "⚠️ Застереження: у платному режимі програма НЕ зупинятиме озвучку при перевищенні "
                "безкоштовного місячного ліміту (1M / 4M символів), а також відкриє моделі без безкоштовного ліміту.\n\n"
                "Якщо до вашого Google Cloud акаунту прив'язана банківська картка (активний Billing), "
                "після вичерпання безкоштовного ліміту кошти списуватимуться з неї автоматично згідно з тарифами Google.\n\n"
                "Ви точно хочете перейти в платний режим?")
            )
            yes_btn = box.addButton(tr("ui_182", "Так, увімкнути платний режим"), QMessageBox.ButtonRole.AcceptRole)
            box.addButton(tr("dlg_026", "Скасувати"), QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() != yes_btn:
                return
            item["paid_enabled"] = True
        else:
            ans = QMessageBox.question(
                self,
                tr("ui_183", "Повернутися до безкоштовного режиму із захистом?"),
                tr("ui_184", "Увімкнути захист безкоштовного ліміту (блокування при досягненні 950k / 3.95M символів за місяць) "
                "та залишити лише моделі з безкоштовним лімітом (Chirp 3: HD, WaveNet, Standard, Neural2, Polyglot, Studio)?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return
            item["paid_enabled"] = False
            if str(item.get("model_filter", "all")) not in ("all", "Chirp 3: HD", "WaveNet", "Standard", "Neural2", "Polyglot", "Studio"):
                item["model_filter"] = "all"

        config.set("online_items", items)
        has_custom = bool(item.get("custom_voices"))
        has_key = bool(str(item.get("api_key", "")).strip())
        self._rebuild_online_cards()
        self._sync_tts_page_selectors()

        if item.get("paid_enabled") and has_key:
            self._start_fetch_provider_catalog(item_id, fetch_btn, silent=True)

    def _on_toggle_fish_paid_mode(self, item_id: str) -> None:
        items: List[Dict[str, Any]] = list(config.get("online_items", []))
        item = next((x for x in items if str(x.get("id")) == item_id), None)
        if not item:
            return

        currently_paid = bool(item.get("paid_enabled", False))
        if not currently_paid:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle(tr("ui_185", "Увімкнути платні моделі Fish Audio?"))
            box.setText(tr("ui_186", "Ви збираєтесь увімкнути платні моделі Fish Audio (s2.1-pro, s2-pro тощо)."))
            box.setInformativeText(
                tr("ui_187", "⚠️ Застереження: безкоштовною для API є лише модель s2.1-pro-free.\n\n"
                "За використання s2.1-pro / s2-pro кошти списуються з балансу вашого акаунту Fish Audio "
                "або прив'язаної картки згідно з тарифами сервісу.\n\n"
                "Ви точно хочете відкрити платні моделі?")
            )
            yes_btn = box.addButton(tr("ui_188", "Так, увімкнути платні моделі"), QMessageBox.ButtonRole.AcceptRole)
            box.addButton(tr("dlg_026", "Скасувати"), QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() != yes_btn:
                return
            item["paid_enabled"] = True
        else:
            ans = QMessageBox.question(
                self,
                tr("ui_189", "Повернутися до безкоштовної моделі?"),
                tr("ui_190", "Вимкнути платні моделі та залишити лише безкоштовну s2.1-pro-free?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            
            if ans != QMessageBox.StandardButton.Yes:
                return
            item["paid_enabled"] = False
            item["model"] = "s2.1-pro-free"

        config.set("online_items", items)
        self._rebuild_online_cards()
        self._sync_tts_page_selectors()

    def _auto_sync_all_online_providers(self) -> None:
        """Автоматично підтягує доступні голоси й моделі для всіх підключених онлайн TTS API."""
        cur_lang = str(config.get("app_language", "uk")).strip().lower() or "uk"
        items: List[Dict[str, Any]] = list(config.get("online_items", []))
        for item in items:
            item_id = str(item.get("id", "")).strip()
            prov = str(item.get("provider", "")).strip()
            if not item_id or not prov:
                continue
            needs_fish_lang_refresh = (prov == "fish_audio" and item.get("voices_lang") != cur_lang)
            if item.get("custom_voices") and not needs_fish_lang_refresh:
                continue
            if prov == "edge":
                if bool(item.get("installed", False)):
                    self._start_fetch_provider_catalog(item_id, None, silent=True)
            elif str(item.get("api_key", "")).strip():
                self._start_fetch_provider_catalog(item_id, None, silent=True)

    def _start_fetch_provider_catalog(self, item_id: str, btn: Optional[QPushButton], silent: bool = False) -> None:
        items: List[Dict[str, Any]] = list(config.get("online_items", []))
        item = next((x for x in items if str(x.get("id")) == item_id), None)
        if not item:
            return
        prov = str(item.get("provider", "")).strip()
        api_key = str(item.get("api_key", "")).strip()
        if prov != "edge" and not api_key:
            if not silent:
                QMessageBox.warning(self, tr("ui_191", "Пошук голосів"), tr("ui_192", "Спочатку введіть API ключ."))
            return

        if isinstance(btn, QPushButton):
            try:
                btn.setEnabled(False)
                btn.setText(tr("ui_176", "⏳ Пошук..."))
            except RuntimeError:
                pass
        st_lbl = self._online_status_labels.get(item_id)
        if st_lbl is not None:
            try:
                st_lbl.setText(tr("ui_193", "⏳ Оновлення голосів і моделей через API..."))
            except RuntimeError:
                pass

        def _worker() -> None:
            try:
                catalog = tts_manager.fetch_provider_catalog(prov, api_key)
                self.provider_catalog_fetched.emit(item_id, catalog, "", btn, silent)
            except Exception as exc:
                self.provider_catalog_fetched.emit(item_id, {}, str(exc), btn, silent)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_provider_catalog_fetched(
        self, item_id: str, catalog: object, err: str, btn: object, silent: bool
    ) -> None:
        if isinstance(btn, QPushButton):
            try:
                btn.setEnabled(True)
                btn.setText(tr("ui_139", "🔄 Оновити голоси"))
            except RuntimeError:
                pass

        if err:
            st_lbl = self._online_status_labels.get(item_id)
            if st_lbl is not None:
                try:
                    st_lbl.setText(tr("ui_f_008", "🔴 Помилка API: {err}", err=err[:95]))
                except RuntimeError:
                    pass
            if not silent:
                QMessageBox.warning(self, tr("ui_194", "Пошук голосів через API"), tr("ui_f_005", "Не вдалося завантажити голоси:\n{err}", err=err))
            return
        cat = catalog if isinstance(catalog, dict) else {}
        voice_list = list(cat.get("voices") or [])
        models_list = list(cat.get("models") or [])
        if not voice_list:
            if not silent:
                QMessageBox.information(self, tr("ui_191", "Пошук голосів"), tr("ui_174", "Список голосів порожній."))
            return

        items: List[Dict[str, Any]] = list(config.get("online_items", []))
        cur_lang = str(config.get("app_language", "uk")).strip().lower() or "uk"
        prov = ""
        for x in items:
            if str(x.get("id")) == item_id:
                x["custom_voices"] = voice_list
                if models_list:
                    x["available_models"] = models_list
                    if str(x.get("provider", "")) == "gemini_tts" and not str(x.get("model", "")).strip():
                        x["model"] = models_list[0]
                prov = str(x.get("provider", ""))
                if prov == "fish_audio":
                    x["voices_lang"] = cur_lang
                break
        config.set("online_items", items)
        self._rebuild_online_cards()
        self._sync_tts_page_selectors()

        if not silent:
            matched_count = len(tts_manager.get_voices_for_engine(f"online:{item_id}"))
            QMessageBox.information(
                self,
                ONLINE_PROVIDERS.get(prov, {}).get("title", "TTS API"),
                tr("ui_f_009", "Знайдено {total} голосів через API.\nДоступно за поточними фільтрами (мова/стать/режим): {matched}.", total=len(voice_list), matched=matched_count),
            )

    # ──────────────────────────────────────────────────────────────────
    # ВКЛАДКА 3 У ГОЛОСАХ: Усі налаштування (Головний TTS, Запасний TTS та DSP)
    # ──────────────────────────────────────────────────────────────────

    def _build_tts_tab_all_settings(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content.setObjectName("scrollContent")
        lay = QVBoxLayout(content)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        # Картка 1: Вибір Головного та Запасного TTS
        routing_card = QFrame()
        routing_card.setProperty("class", "card")
        rc_lay = QVBoxLayout(routing_card)
        rc_lay.setContentsMargins(14, 12, 14, 12)
        rc_lay.setSpacing(10)

        rc_hdr = QHBoxLayout()
        rc_title = QLabel(tr("ui_195", "🎙️ Вибір Головного та Запасного голосу (TTS)"))
        rc_title.setProperty("class", "cardHeader")
        rc_hdr.addWidget(rc_title)
        rc_hdr.addWidget(
            HelpButton(
                tr("ui_196", "Для чого потрібен Головний та Запасний TTS?"),
                tr(
                    "ui_197",
                    "<b>Подвійний захист озвучки:</b><br><br>"
                    "• <b>Головний TTS</b> — голос, який звучить постійно.<br>"
                    "• <b>Запасний TTS</b> — автоматично підхоплює, якщо головний не спрацював (зник інтернет, помилка API або скінчився ліміт).<br><br>"
                    "<b>💡 Порада:</b> Якщо головний голос онлайн (Edge-TTS, ElevenLabs, OpenAI тощо), встановіть один локальний голос (напр. <b>Silero</b> або <b>Piper</b>) і поставте його як <b>Запасний TTS</b> — тоді ви ніколи не залишитесь без голосу.<br><br>"
                    "<i>Нижче для кожного вибраного TTS з'являються його власні налаштування звучання.</i>",
                ),
            )
        )
        rc_hdr.addStretch(1)
        rc_lay.addLayout(rc_hdr)

        r_grid = QGridLayout()
        r_grid.setHorizontalSpacing(10)
        r_grid.setVerticalSpacing(8)

        prim_lbl = QLabel(tr("ui_198", "<b>Головний TTS (основний):</b>"))
        r_grid.addWidget(prim_lbl, 0, 0)

        self.all_prim_eng_combo = VoiceComboBox()
        self.all_prim_eng_combo.setToolTip(tr("ui_199", "Основна система озвучки"))
        self.all_prim_eng_combo.currentIndexChanged.connect(self._on_all_prim_eng_changed)
        r_grid.addWidget(self.all_prim_eng_combo, 0, 1)

        self.all_prim_voice_combo = VoiceComboBox()
        self.all_prim_voice_combo.setToolTip(tr("ui_200", "Голос для основного TTS (Ж — жіночий, М — чоловічий)"))
        self.all_prim_voice_combo.currentIndexChanged.connect(self._on_all_prim_voice_changed)
        r_grid.addWidget(self.all_prim_voice_combo, 0, 2)

        fb_lbl = QLabel(tr("ui_201", "<b>Запасний TTS (при збої першого):</b>"))
        r_grid.addWidget(fb_lbl, 1, 0)

        self.all_fb_eng_combo = VoiceComboBox()
        self.all_fb_eng_combo.setToolTip(
            tr("ui_202", "Запасна система озвучки, яка автоматично увімкнеться, якщо основна дасть збій")
        )
        self.all_fb_eng_combo.currentIndexChanged.connect(self._on_all_fb_eng_changed)
        r_grid.addWidget(self.all_fb_eng_combo, 1, 1)

        self.all_fb_voice_combo = VoiceComboBox()
        self.all_fb_voice_combo.setToolTip(tr("ui_203", "Голос для запасного TTS (Ж — жіночий, М — чоловічий)"))
        self.all_fb_voice_combo.currentIndexChanged.connect(self._on_all_fb_voice_changed)
        r_grid.addWidget(self.all_fb_voice_combo, 1, 2)

        r_grid.setColumnStretch(1, 1)
        r_grid.setColumnStretch(2, 1)
        rc_lay.addLayout(r_grid)
        lay.addWidget(routing_card)

        # Компактний блок обробки тексту перед налаштуваннями звучання (DSP)
        split_card = QFrame()
        split_card.setProperty("class", "card")
        sp_card_lay = QVBoxLayout(split_card)
        sp_card_lay.setContentsMargins(14, 10, 14, 10)
        sp_card_lay.setSpacing(6)

        sp_lay = QHBoxLayout()
        sp_lay.setSpacing(8)
        self.chk_split_sentences = QCheckBox(tr("ui_204", "✂️ Розбивати текст на речення для озвучки (по крапках, ! та ?)"))
        self.chk_split_sentences.setChecked(bool(config.get("split_sentences_enabled", True)))
        self.chk_split_sentences.toggled.connect(lambda v: config.set("split_sentences_enabled", v))
        sp_lay.addWidget(self.chk_split_sentences)
        sp_lay.addWidget(
            HelpButton(
                tr("ui_206", "Розбиття тексту на речення"),
                tr(
                    "ui_207",
                    "<b>Як це працює?</b><br>"
                    "Довгий текст автоматично ділиться по крапках (<code>.</code>), знаках оклику (<code>!</code>) та питання (<code>?</code>), не розриваючи фрази по комах.<br><br>"
                    "• <b>Швидший старт:</b> перше речення звучить майже одразу, поки готуються наступні.<br>"
                    "• <b>Без обрізань:</b> локальні голоси стабільніше читають довгі тексти.<br>"
                    "• <b>Кращі субтитри:</b> у OBS текст з'являється акуратно по одному реченню.<br><br>"
                    "<i>Рекомендуємо тримати увімкненим.</i>",
                ),
            )
        )
        sp_lay.addStretch(1)
        sp_card_lay.addLayout(sp_lay)

        norm_lay = QHBoxLayout()
        norm_lay.setSpacing(8)
        self.chk_normalize_numbers = QCheckBox(tr("ui_208", "🔢 Перетворювати числа та знаки (%, $, °, №) у слова для озвучки"))
        self.chk_normalize_numbers.setChecked(bool(config.get("normalize_numbers_enabled", True)))
        self.chk_normalize_numbers.toggled.connect(lambda v: config.set("normalize_numbers_enabled", v))
        norm_lay.addWidget(self.chk_normalize_numbers)
        norm_lay.addWidget(
            HelpButton(
                tr("ui_210", "Перетворення чисел та знаків"),
                tr(
                    "ui_211",
                    "<b>Навіщо це?</b><br>"
                    "Деякі голоси (особливо локальні) плутаються в цифрах або пропускають символи на кшталт <code>%</code>, <code>$</code>, <code>€</code>, <code>₴</code>, <code>°C</code> чи <code>№</code>.<br><br>"
                    "• <b>В озвучку</b> число чи знак іде словами, щоб звучало правильно.<br>"
                    "• <b>На екрані, в OBS та історії</b> залишається оригінальний запис (напр. <code>Знижка 50%</code>).<br><br>"
                    "<i>Якщо голос уже сам читає числа добре — опцію можна вимкнути.</i>",
                ),
            )
        )
        norm_lay.addStretch(1)
        sp_card_lay.addLayout(norm_lay)

        lay.addWidget(split_card)

        # Контейнер індивідуальних карток налаштувань звучання (DSP) для Головного і (якщо обрано) Запасного TTS
        self.dsp_cards_layout = QVBoxLayout()
        self.dsp_cards_layout.setSpacing(10)
        lay.addLayout(self.dsp_cards_layout)
        lay.addStretch(1)

        scroll.setWidget(content)
        return scroll

    # ──────────────────────────────────────────────────────────────────
    # ВКЛАДКА 4 У ГОЛОСАХ: AI Орфографія (як Онлайн-голоси + кнопка ⚙️)
    # ──────────────────────────────────────────────────────────────────

    def _build_tts_tab_ai_spellcheck(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content.setObjectName("scrollContent")
        lay = QVBoxLayout(content)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        # Швидкий перемикач увімкнення AI-орфографії прямо на вкладці
        self.chk_ai_spellcheck = QCheckBox(tr("dlg_067", "Увімкнути AI-коректор тексту перед відправкою в TTS та OBS"))
        self.chk_ai_spellcheck.setChecked(bool(config.get("ai_spellcheck_enabled", False)))
        self.chk_ai_spellcheck.setStyleSheet("font-weight: 700; color: #F8FAFC; font-size: 13px; padding: 2px 4px;")
        self.chk_ai_spellcheck.toggled.connect(self._on_ai_spellcheck_toggled)
        lay.addWidget(self.chk_ai_spellcheck)

        # Верхній рядок вибору AI-сервісу + "+ Додати" + кнопка налаштувань "⚙️"
        add_bar = QFrame()
        add_bar.setProperty("class", "card")
        ab_lay = QHBoxLayout(add_bar)
        ab_lay.setContentsMargins(14, 9, 14, 9)
        ab_lay.setSpacing(8)

        self.ai_add_combo = QComboBox()
        self._populate_ai_add_combo()
        ab_lay.addWidget(self.ai_add_combo, 1)

        add_btn = QPushButton(tr("ui_131", "+ Додати"))
        add_btn.setProperty("class", "primaryBtn")
        add_btn.setMinimumWidth(115)
        add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_btn.clicked.connect(self._on_add_ai_provider)
        ab_lay.addWidget(add_btn)

        settings_btn = QPushButton("⚙️")
        settings_btn.setFixedWidth(42)
        settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        settings_btn.clicked.connect(self._open_ai_settings_dialog)
        ab_lay.addWidget(settings_btn)

        ab_lay.addWidget(
            HelpButton(
                tr("ui_215", "Як працює AI Орфографія?"),
                tr(
                    "ui_216",
                    "<b>Що робить?</b><br>"
                    "Автоматично виправляє помилки, розділові знаки та переплутану розкладку (напр. <code>ghbdsn</code> → <code>привіт</code>) <b>перед озвученням і показом в OBS</b>.<br><br>"
                    "<b>Як налаштувати:</b><br>"
                    "1. Оберіть провайдера (найшвидший і безкоштовний — <b>Groq</b>) та натисніть <b>+ Додати</b>.<br>"
                    "2. Вставте API-ключ (інструкція є в кнопці <b>📘 Гайд</b>).<br>"
                    "3. Увімкніть галочку <b>«AI Орфографія»</b> (або натисніть <b>⚙️</b>, щоб обрати головний/запасний сервіс і промпт).<br><br>"
                    "<i>Якщо текст не потребує виправлень — просто вимкніть галочку.</i>",
                ),
            )
        )

        lay.addWidget(add_bar)
        lay.addWidget(self._create_socks5_bar())

        self.ai_status_banner = QLabel("")
        self.ai_status_banner.setWordWrap(True)
        self.ai_status_banner.setStyleSheet("padding: 4px 8px; font-size: 12.5px;")
        lay.addWidget(self.ai_status_banner)

        self.ai_cards_layout = QVBoxLayout()
        self.ai_cards_layout.setSpacing(8)
        lay.addLayout(self.ai_cards_layout)
        lay.addStretch(1)

        self._rebuild_ai_cards()
        self._refresh_ai_status_banner()
        scroll.setWidget(content)
        return scroll

    def _build_tts_tab_twitch_chat(self) -> QWidget:
        """Вкладка налаштувань читання чату Twitch (хто, як, фільтри та жива стрічка)."""
        self._twitch_user_last_spoken: Dict[str, float] = {}
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        content.setObjectName("scrollContent")
        lay = QVBoxLayout(content)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        # 1. Підключення та основні перемикачі читання
        enable_card = QFrame()
        enable_card.setProperty("class", "card")
        en_lay = QVBoxLayout(enable_card)
        en_lay.setContentsMargins(14, 12, 14, 12)
        en_lay.setSpacing(8)

        en_hdr = QHBoxLayout()
        en_title = QLabel(tr("twitch.read.title", "💬 Читання чату Twitch у голос"))
        en_title.setProperty("class", "cardHeader")
        en_hdr.addWidget(en_title)
        en_hdr.addStretch(1)

        self.twitch_tab_status_lbl = QLabel()
        self.twitch_tab_status_lbl.setStyleSheet("font-size: 12px; color: #AAB4C8;")
        en_hdr.addWidget(self.twitch_tab_status_lbl)

        open_conn_btn = QPushButton(tr("twitch.read.open_conn", "⚙️ Токен і канал"))
        open_conn_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_conn_btn.clicked.connect(self._open_twitch_connection_settings)
        en_hdr.addWidget(open_conn_btn)
        en_lay.addLayout(en_hdr)

        self.twitch_read_viewers_chk = QCheckBox(
            tr("twitch.read.viewers_enable", "Читати та озвучувати повідомлення глядачів у чаті")
        )
        self.twitch_read_viewers_chk.setChecked(bool(config.get("twitch_read_viewers", False)))
        en_lay.addWidget(self.twitch_read_viewers_chk)

        self.twitch_chat_intercept_chk = QCheckBox(
            tr("twitch.intercept_own", "Перехоплювати мої повідомлення з чату Twitch та озвучувати")
        )
        self.twitch_chat_intercept_chk.setChecked(bool(config.get("twitch_intercept_own", False)))
        en_lay.addWidget(self.twitch_chat_intercept_chk)

        en_note = QLabel(
            tr(
                "twitch.read.note",
                "Для роботи потрібен Access Token у розділі «✨ Текст на екран/чат» → «Надсилання фраз у чат Twitch».",
            )
        )
        en_note.setWordWrap(True)
        en_note.setProperty("class", "muted")
        en_lay.addWidget(en_note)
        lay.addWidget(enable_card)

        # 1.5. Команда запуску, голос за замовчуванням і голоси за командами
        self._twitch_chat_paused = False
        self._twitch_voice_rule_widgets: List[Dict[str, Any]] = []

        voice_card = QFrame()
        voice_card.setProperty("class", "card")
        vc_lay = QVBoxLayout(voice_card)
        vc_lay.setContentsMargins(14, 12, 14, 12)
        vc_lay.setSpacing(8)

        vc_title = QLabel(tr("twitch.read.voice_title", "🎯 Команда запуску, голос і гучність чату"))
        vc_title.setProperty("class", "cardHeader")
        vc_lay.addWidget(vc_title)

        self.twitch_chat_read_all_chk = QCheckBox(
            tr("twitch.read.read_all", "Озвучувати всі повідомлення (без команди)")
        )
        self.twitch_chat_read_all_chk.setChecked(bool(config.get("twitch_chat_read_all", False)))
        vc_lay.addWidget(self.twitch_chat_read_all_chk)

        self.twitch_chat_prefix_row = QWidget()
        pfx_lay = QHBoxLayout(self.twitch_chat_prefix_row)
        pfx_lay.setContentsMargins(24, 0, 0, 0)
        pfx_lay.setSpacing(8)
        pfx_lay.addWidget(QLabel(tr("twitch.read.prefix_lbl", "Команда для озвучки:")))
        self.twitch_chat_prefix_edit = QLineEdit(str(config.get("twitch_chat_command_prefix", "!!!") or "!!!"))
        self.twitch_chat_prefix_edit.setFixedWidth(130)
        self.twitch_chat_prefix_edit.setPlaceholderText("!!! / !t")
        pfx_lay.addWidget(self.twitch_chat_prefix_edit)
        pfx_hint = QLabel(tr("twitch.read.prefix_hint", "Напр.: «!!! привіт» або «!t привіт»"))
        pfx_hint.setProperty("class", "muted")
        pfx_lay.addWidget(pfx_hint, 1)
        vc_lay.addWidget(self.twitch_chat_prefix_row)

        vc_grid = QGridLayout()
        vc_grid.setColumnStretch(1, 1)
        vc_grid.setSpacing(8)

        vc_grid.addWidget(QLabel(tr("twitch.read.tts_engine_lbl", "TTS для чату:")), 0, 0)
        self.twitch_chat_engine_combo = QComboBox()
        vc_grid.addWidget(self.twitch_chat_engine_combo, 0, 1)

        self.twitch_chat_voice_lbl = QLabel(tr("twitch.read.tts_voice_lbl", "Голос для чату:"))
        self.twitch_chat_voice_combo = QComboBox()
        vc_grid.addWidget(self.twitch_chat_voice_lbl, 1, 0)
        vc_grid.addWidget(self.twitch_chat_voice_combo, 1, 1)

        vc_grid.addWidget(QLabel(tr("twitch.read.volume_lbl", "Гучність озвучки чату:")), 2, 0)
        self.twitch_chat_vol_slider = self._make_slider_row(
            vc_grid,
            2,
            1,
            0,
            200,
            int(config.get("twitch_chat_volume", 100)),
            " %",
            lambda: self._save_twitch_settings("chat_tab"),
        )
        vc_lay.addLayout(vc_grid)

        rules_hdr = QHBoxLayout()
        rules_hdr.setContentsMargins(0, 6, 0, 0)
        rules_hdr.setSpacing(8)
        rules_title = QLabel(tr("twitch.read.rules_title", "🎭 Окремі голоси за командами (напр. !t, !1t):"))
        rules_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #E2E8F0;")
        rules_hdr.addWidget(rules_title)
        rules_hdr.addStretch(1)
        add_rule_btn = QPushButton(tr("twitch.read.add_rule", "+ Додати команду-голос"))
        add_rule_btn.setProperty("class", "primaryBtn")
        add_rule_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_rule_btn.clicked.connect(self._on_add_twitch_voice_rule)
        rules_hdr.addWidget(add_rule_btn)
        vc_lay.addLayout(rules_hdr)

        rules_hint = QLabel(
            tr(
                "twitch.read.rules_hint",
                "Кожна команда може запускати свій TTS, голос і гучність (працює навіть коли «Озвучувати всі повідомлення» увімкнено).",
            )
        )
        rules_hint.setWordWrap(True)
        rules_hint.setProperty("class", "muted")
        vc_lay.addWidget(rules_hint)

        self.twitch_voice_rules_lay = QVBoxLayout()
        self.twitch_voice_rules_lay.setContentsMargins(0, 2, 0, 0)
        self.twitch_voice_rules_lay.setSpacing(6)
        vc_lay.addLayout(self.twitch_voice_rules_lay)
        lay.addWidget(voice_card)

        # 2. Хто може бути озвучений
        who_card = QFrame()
        who_card.setProperty("class", "card")
        who_lay = QVBoxLayout(who_card)
        who_lay.setContentsMargins(14, 12, 14, 12)
        who_lay.setSpacing(8)

        who_title = QLabel(tr("twitch.read.who_title", "👤 Хто (фільтр користувачів)"))
        who_title.setProperty("class", "cardHeader")
        who_lay.addWidget(who_title)

        who_grid = QGridLayout()
        who_grid.setColumnStretch(1, 1)
        who_grid.setSpacing(8)

        who_grid.addWidget(QLabel(tr("twitch.read.who_lbl", "Читати повідомлення від:")), 0, 0)
        self.twitch_viewer_filter_combo = QComboBox()
        for label, key in (
            (tr("twitch.read.filter.all", "Усі глядачі чату"), "all"),
            (tr("twitch.read.filter.subs_vips_mods", "Підписники, VIP і модератори"), "subscribers_vips_mods"),
            (tr("twitch.read.filter.vips_mods", "Тільки VIP і модератори"), "vips_mods"),
            (tr("twitch.read.filter.whitelist", "Лише зі списку дозволених"), "whitelist"),
        ):
            self.twitch_viewer_filter_combo.addItem(label, key)
        cur_filter = str(config.get("twitch_viewer_filter", "all") or "all")
        f_idx = self.twitch_viewer_filter_combo.findData(cur_filter)
        if f_idx >= 0:
            self.twitch_viewer_filter_combo.setCurrentIndex(f_idx)
        who_grid.addWidget(self.twitch_viewer_filter_combo, 0, 1)

        self.twitch_allowed_lbl = QLabel(tr("twitch.read.allowed_lbl", "Дозволені ніки:"))
        self.twitch_allowed_edit = QLineEdit(str(config.get("twitch_allowed_users", "")))
        self.twitch_allowed_edit.setPlaceholderText(tr("twitch.read.allowed_ph", "user1, user2 (через кому)"))
        who_grid.addWidget(self.twitch_allowed_lbl, 1, 0)
        who_grid.addWidget(self.twitch_allowed_edit, 1, 1)

        who_grid.addWidget(QLabel(tr("twitch.read.ignored_lbl", "Ігнорувати ботів / ніки:")), 2, 0)
        self.twitch_ignored_edit = QLineEdit(
            str(config.get("twitch_ignored_users", "nightbot, streamelements, streamlabs, moobot, fossabot, wizebot"))
        )
        self.twitch_ignored_edit.setPlaceholderText("nightbot, streamelements, moobot...")
        who_grid.addWidget(self.twitch_ignored_edit, 2, 1)
        who_lay.addLayout(who_grid)
        lay.addWidget(who_card)

        # 3. Як читати та обмеження
        how_card = QFrame()
        how_card.setProperty("class", "card")
        how_lay = QVBoxLayout(how_card)
        how_lay.setContentsMargins(14, 12, 14, 12)
        how_lay.setSpacing(8)

        how_title = QLabel(tr("twitch.read.how_title", "🗣️ Як читати та фільтри тексту"))
        how_title.setProperty("class", "cardHeader")
        how_lay.addWidget(how_title)

        how_grid = QGridLayout()
        how_grid.setColumnStretch(1, 1)
        how_grid.setSpacing(8)

        how_grid.addWidget(QLabel(tr("twitch.read.format_lbl", "Формат озвучення:")), 0, 0)
        self.twitch_read_format_combo = QComboBox()
        for label, key in (
            (tr("twitch.read.fmt.user_says", "«Нік каже: текст»"), "user_says"),
            (tr("twitch.read.fmt.user_colon", "«Нік: текст»"), "user_colon"),
            (tr("twitch.read.fmt.text_only", "Тільки текст повідомлення"), "text_only"),
        ):
            self.twitch_read_format_combo.addItem(label, key)
        cur_fmt = str(config.get("twitch_read_format", "user_says") or "user_says")
        fmt_idx = self.twitch_read_format_combo.findData(cur_fmt)
        if fmt_idx >= 0:
            self.twitch_read_format_combo.setCurrentIndex(fmt_idx)
        how_grid.addWidget(self.twitch_read_format_combo, 0, 1)

        how_grid.addWidget(QLabel(tr("twitch.read.max_chars_lbl", "Макс. символів (0 = без ліміту):")), 1, 0)
        self.twitch_max_chars_spin = QSpinBox()
        self.twitch_max_chars_spin.setRange(0, 500)
        self.twitch_max_chars_spin.setSingleStep(20)
        self.twitch_max_chars_spin.setValue(int(config.get("twitch_max_chars", 220) or 0))
        _disable_wheel_scroll(self.twitch_max_chars_spin)
        how_grid.addWidget(self.twitch_max_chars_spin, 1, 1)

        cooldown_lbl = QLabel(tr("twitch.read.cooldown_lbl", "Пауза на одного глядача (сек):"))
        cooldown_tip = tr(
            "twitch.read.cooldown_tip",
            "Мінімальна пауза (у секундах) між озвученими повідомленнями від одного й того ж глядача.\n"
            "Захищає від спаму в черзі одним користувачем; 0 — без паузи.",
        )
        cooldown_lbl.setToolTip(cooldown_tip)
        how_grid.addWidget(cooldown_lbl, 2, 0)
        self.twitch_cooldown_spin = QSpinBox()
        self.twitch_cooldown_spin.setRange(0, 120)
        self.twitch_cooldown_spin.setToolTip(cooldown_tip)
        self.twitch_cooldown_spin.setValue(int(config.get("twitch_user_cooldown_sec", 0) or 0))
        _disable_wheel_scroll(self.twitch_cooldown_spin)
        how_grid.addWidget(self.twitch_cooldown_spin, 2, 1)

        how_grid.addWidget(QLabel(tr("twitch.read.max_queue_lbl", "Макс. черга повідомлень чату:")), 3, 0)
        self.twitch_max_queue_spin = QSpinBox()
        self.twitch_max_queue_spin.setRange(1, 20)
        self.twitch_max_queue_spin.setValue(max(1, int(config.get("twitch_max_queue", 5) or 5)))
        _disable_wheel_scroll(self.twitch_max_queue_spin)
        how_grid.addWidget(self.twitch_max_queue_spin, 3, 1)
        how_lay.addLayout(how_grid)

        self.twitch_ignore_cmds_chk = QCheckBox(
            tr("twitch.read.ignore_cmds", "Ігнорувати команди чату (починаються з ! або /)")
        )
        self.twitch_ignore_cmds_chk.setChecked(bool(config.get("twitch_ignore_commands", True)))
        how_lay.addWidget(self.twitch_ignore_cmds_chk)

        self.twitch_ignore_links_chk = QCheckBox(
            tr("twitch.read.ignore_links", "Не читати повідомлення з посиланнями (http / www)")
        )
        self.twitch_ignore_links_chk.setChecked(bool(config.get("twitch_ignore_links", True)))
        how_lay.addWidget(self.twitch_ignore_links_chk)

        self.twitch_save_viewers_hist_chk = QCheckBox(
            tr("twitch.read.save_history", "Зберігати повідомлення глядачів в історію фраз")
        )
        self.twitch_save_viewers_hist_chk.setChecked(bool(config.get("twitch_save_viewers_history", False)))
        how_lay.addWidget(self.twitch_save_viewers_hist_chk)
        lay.addWidget(how_card)

        # 3.5. Аудіо-маршрут для чату глядачів (VB-Cable та VBAN)
        route_card = QFrame()
        route_card.setProperty("class", "card")
        rt_lay = QVBoxLayout(route_card)
        rt_lay.setContentsMargins(14, 12, 14, 12)
        rt_lay.setSpacing(8)

        rt_title = QLabel(tr("twitch.read.audio_title", "🔊 Звук чату: OBS, Discord та VBAN"))
        rt_title.setProperty("class", "cardHeader")
        rt_lay.addWidget(rt_title)

        self.twitch_chat_vbcable_chk = QCheckBox(
            tr("twitch.read.send_vbcable", "Передавати озвучку чату у VB-Cable (щоб чули в Discord / іграх)")
        )
        self.twitch_chat_vbcable_chk.setChecked(bool(config.get("twitch_chat_send_vbcable", False)))
        rt_lay.addWidget(self.twitch_chat_vbcable_chk)

        raw_vm = str(config.get("twitch_chat_vban_mode", "same") or "same").strip().lower()
        send_vban_cfg = config.get("twitch_chat_send_vban", None)
        vban_chat_enabled = (raw_vm in ("same", "separate")) if send_vban_cfg is None else bool(send_vban_cfg)
        self.twitch_chat_vban_chk = QCheckBox(
            tr("twitch.read.send_vban", "Передавати озвучку чату у VBAN (через локальну мережу)")
        )
        self.twitch_chat_vban_chk.setChecked(vban_chat_enabled)
        rt_lay.addWidget(self.twitch_chat_vban_chk)

        rt_grid = QGridLayout()
        rt_grid.setColumnStretch(1, 1)
        rt_grid.setSpacing(8)

        self.twitch_chat_vban_mode_lbl = QLabel(tr("twitch.read.vban_mode_lbl", "VBAN для чату глядачів:"))
        rt_grid.addWidget(self.twitch_chat_vban_mode_lbl, 0, 0)
        self.twitch_chat_vban_combo = QComboBox()
        for label, key in (
            (tr("twitch.read.vban.same", "Основний (у той самий потік)"), "same"),
            (tr("twitch.read.vban.separate", "В окремий потік VBAN"), "separate"),
        ):
            self.twitch_chat_vban_combo.addItem(label, key)
        cur_vm = raw_vm if raw_vm in ("same", "separate") else "same"
        vm_idx = self.twitch_chat_vban_combo.findData(cur_vm)
        if vm_idx >= 0:
            self.twitch_chat_vban_combo.setCurrentIndex(vm_idx)
        rt_grid.addWidget(self.twitch_chat_vban_combo, 0, 1)

        self.twitch_chat_vban_stream_lbl = QLabel(tr("twitch.read.vban_stream_lbl", "Назва окремого потоку:"))
        self.twitch_chat_vban_stream_edit = QLineEdit(
            str(config.get("twitch_chat_vban_stream", "SilentChat") or "SilentChat")
        )
        self.twitch_chat_vban_stream_edit.setMaxLength(16)
        self.twitch_chat_vban_stream_edit.setPlaceholderText("SilentChat")
        rt_grid.addWidget(self.twitch_chat_vban_stream_lbl, 1, 0)
        rt_grid.addWidget(self.twitch_chat_vban_stream_edit, 1, 1)
        rt_lay.addLayout(rt_grid)

        rt_note = QLabel(
            tr(
                "twitch.read.audio_note",
                "Коли VB-Cable вимкнено, чат чуєте лише ви в навушниках та OBS (через звук робочого столу), а співрозмовники в Discord — ні.",
            )
        )
        rt_note.setWordWrap(True)
        rt_note.setProperty("class", "muted")
        rt_lay.addWidget(rt_note)
        lay.addWidget(route_card)

        # 4. Жива стрічка чату
        feed_card = QFrame()
        feed_card.setProperty("class", "card")
        feed_lay = QVBoxLayout(feed_card)
        feed_lay.setContentsMargins(14, 12, 14, 12)
        feed_lay.setSpacing(8)

        feed_hdr = QHBoxLayout()
        feed_title = QLabel(tr("twitch.read.feed_title", "📜 Жива стрічка чату"))
        feed_title.setProperty("class", "cardHeader")
        feed_hdr.addWidget(feed_title)
        feed_hdr.addStretch(1)

        sample_btn = QPushButton(tr("twitch.read.test_sample", "🔊 Тест фрази"))
        sample_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        sample_btn.clicked.connect(self._test_twitch_reader_sample)
        feed_hdr.addWidget(sample_btn)

        skip_btn = QPushButton(tr("twitch.read.skip", "⏭️ Пропустити"))
        skip_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        skip_btn.clicked.connect(tts_manager.stop_current)
        feed_hdr.addWidget(skip_btn)

        clear_feed_btn = QPushButton(tr("twitch.read.clear_feed", "Очистити"))
        clear_feed_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_feed_btn.clicked.connect(lambda: self.twitch_live_feed.clear())
        feed_hdr.addWidget(clear_feed_btn)
        feed_lay.addLayout(feed_hdr)

        self.twitch_live_feed = QListWidget()
        self.twitch_live_feed.setMinimumHeight(140)
        self.twitch_live_feed.setMaximumHeight(200)
        self.twitch_live_feed.setStyleSheet(
            "QListWidget { background-color: #1B1F2D; border: 1px solid #38415E; border-radius: 8px; padding: 4px; }"
            "QListWidget::item { padding: 3px 6px; border-radius: 5px; color: #E2E8F0; }"
        )
        feed_lay.addWidget(self.twitch_live_feed)
        lay.addWidget(feed_card)
        lay.addStretch(1)

        self._update_twitch_allowed_row()
        self._update_twitch_command_row()
        self._refresh_twitch_chat_voice_selectors()
        self._refresh_omnivoice_whisper_hotkey_availability()
        if hasattr(self, "keyboard_hook"):
            self.keyboard_hook.update_from_config(config.data)
        self._rebuild_twitch_voice_rules()
        self.twitch_read_viewers_chk.toggled.connect(lambda _c: self._save_twitch_settings("chat_tab"))
        self.twitch_chat_intercept_chk.toggled.connect(lambda _c: self._save_twitch_settings("chat_tab"))
        self.twitch_chat_read_all_chk.toggled.connect(lambda _c: self._save_twitch_settings("chat_tab"))
        self.twitch_chat_prefix_edit.editingFinished.connect(lambda: self._save_twitch_settings("chat_tab"))
        self.twitch_chat_engine_combo.currentIndexChanged.connect(self._on_twitch_chat_engine_changed)
        self.twitch_chat_voice_combo.currentIndexChanged.connect(lambda _i: self._save_twitch_settings("chat_tab"))
        self.twitch_viewer_filter_combo.currentIndexChanged.connect(lambda _i: self._save_twitch_settings("chat_tab"))
        self.twitch_allowed_edit.editingFinished.connect(lambda: self._save_twitch_settings("chat_tab"))
        self.twitch_ignored_edit.editingFinished.connect(lambda: self._save_twitch_settings("chat_tab"))
        self.twitch_read_format_combo.currentIndexChanged.connect(lambda _i: self._save_twitch_settings("chat_tab"))
        self.twitch_max_chars_spin.valueChanged.connect(lambda _v: self._save_twitch_settings("chat_tab"))
        self.twitch_cooldown_spin.valueChanged.connect(lambda _v: self._save_twitch_settings("chat_tab"))
        self.twitch_max_queue_spin.valueChanged.connect(lambda _v: self._save_twitch_settings("chat_tab"))
        self.twitch_ignore_cmds_chk.toggled.connect(lambda _c: self._save_twitch_settings("chat_tab"))
        self.twitch_ignore_links_chk.toggled.connect(lambda _c: self._save_twitch_settings("chat_tab"))
        self.twitch_save_viewers_hist_chk.toggled.connect(lambda _c: self._save_twitch_settings("chat_tab"))
        self.twitch_chat_vbcable_chk.toggled.connect(lambda _c: self._save_twitch_settings("chat_tab"))
        self.twitch_chat_vban_chk.toggled.connect(lambda _c: self._save_twitch_settings("chat_tab"))
        self.twitch_chat_vban_combo.currentIndexChanged.connect(lambda _i: self._save_twitch_settings("chat_tab"))
        self.twitch_chat_vban_stream_edit.editingFinished.connect(lambda: self._save_twitch_settings("chat_tab"))

        scroll.setWidget(content)
        return scroll

    def _update_twitch_allowed_row(self) -> None:
        if hasattr(self, "twitch_viewer_filter_combo"):
            is_wl = str(self.twitch_viewer_filter_combo.currentData() or "") == "whitelist"
            self.twitch_allowed_lbl.setVisible(is_wl)
            self.twitch_allowed_edit.setVisible(is_wl)
        if hasattr(self, "twitch_chat_vban_combo"):
            is_vban_on = self.twitch_chat_vban_chk.isChecked() if hasattr(self, "twitch_chat_vban_chk") else True
            is_sep = str(self.twitch_chat_vban_combo.currentData() or "") == "separate"
            if hasattr(self, "twitch_chat_vban_mode_lbl"):
                self.twitch_chat_vban_mode_lbl.setVisible(is_vban_on)
            self.twitch_chat_vban_combo.setVisible(is_vban_on)
            self.twitch_chat_vban_stream_lbl.setVisible(is_vban_on and is_sep)
            self.twitch_chat_vban_stream_edit.setVisible(is_vban_on and is_sep)

    def _update_twitch_command_row(self) -> None:
        if not hasattr(self, "twitch_chat_read_all_chk"):
            return
        self.twitch_chat_prefix_row.setVisible(not self.twitch_chat_read_all_chk.isChecked())

    def _populate_voice_combo_for_engine(self, combo: QComboBox, engine_id: str, selected_voice: str = "") -> str:
        combo.blockSignals(True)
        combo.clear()
        voices = tts_manager.get_voices_for_engine(engine_id) if engine_id else []
        chosen_idx = 0
        for i, v in enumerate(voices):
            combo.add_gender_item(v["name"], v["id"], str(v.get("gender", "")))
            if v["id"] == selected_voice:
                chosen_idx = i
        out_voice = ""
        if voices:
            combo.setCurrentIndex(chosen_idx)
            out_voice = str(combo.currentData() or "")
        combo.blockSignals(False)
        return out_voice

    def _refresh_twitch_chat_voice_selectors(self) -> None:
        if not hasattr(self, "twitch_chat_engine_combo"):
            return
        available = tts_manager.get_available_active_engines()
        cur_eng = str(config.get("twitch_chat_tts_engine", "") or "").strip()
        cur_voice = str(config.get("twitch_chat_tts_voice", "") or "").strip()

        self.twitch_chat_engine_combo.blockSignals(True)
        self.twitch_chat_engine_combo.clear()
        if not available:
            if cur_eng or cur_voice:
                config.update({"twitch_chat_tts_engine": "", "twitch_chat_tts_voice": ""})
            self.twitch_chat_engine_combo.addItem(tr("twitch.read.no_tts", "Немає доступних TTS"), "")
            self.twitch_chat_engine_combo.blockSignals(False)
            self._populate_voice_combo_for_engine(self.twitch_chat_voice_combo, "", "")
            self.twitch_chat_voice_lbl.setVisible(False)
            self.twitch_chat_voice_combo.setVisible(False)
            if hasattr(self, "twitch_voice_rules_lay"):
                self._rebuild_twitch_voice_rules()
            return

        eng_idx = 0
        for i, entry in enumerate(available):
            self.twitch_chat_engine_combo.addItem(entry["title"], entry["id"])
            if entry["id"] == cur_eng:
                eng_idx = i
        self.twitch_chat_engine_combo.setCurrentIndex(eng_idx)
        sel_eng = str(self.twitch_chat_engine_combo.currentData() or "")
        self.twitch_chat_engine_combo.blockSignals(False)

        sel_voice = self._populate_voice_combo_for_engine(self.twitch_chat_voice_combo, sel_eng, cur_voice)
        self.twitch_chat_voice_lbl.setVisible(True)
        self.twitch_chat_voice_combo.setVisible(True)
        if sel_eng != cur_eng or sel_voice != cur_voice:
            config.update({"twitch_chat_tts_engine": sel_eng, "twitch_chat_tts_voice": sel_voice})

        if hasattr(self, "twitch_voice_rules_lay"):
            self._rebuild_twitch_voice_rules()

    def _on_twitch_chat_engine_changed(self) -> None:
        sel_eng = str(self.twitch_chat_engine_combo.currentData() or "")
        self._populate_voice_combo_for_engine(
            self.twitch_chat_voice_combo, sel_eng, str(config.get("twitch_chat_tts_voice", "") or "")
        )
        has_eng = bool(sel_eng)
        self.twitch_chat_voice_lbl.setVisible(has_eng)
        self.twitch_chat_voice_combo.setVisible(has_eng)
        self._save_twitch_settings("chat_tab")

    def _rebuild_twitch_voice_rules(self) -> None:
        if not hasattr(self, "twitch_voice_rules_lay"):
            return
        while self.twitch_voice_rules_lay.count():
            item = self.twitch_voice_rules_lay.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self._twitch_voice_rule_widgets = []

        rules: List[Dict[str, Any]] = [
            dict(r) for r in config.get("twitch_chat_voice_rules", []) if isinstance(r, dict)
        ]
        if not rules:
            empty_lbl = QLabel(
                tr(
                    "twitch.read.rules_empty",
                    "Поки не додано окремих команд — використовується голос за замовчуванням вище.",
                )
            )
            empty_lbl.setWordWrap(True)
            empty_lbl.setProperty("class", "muted")
            self.twitch_voice_rules_lay.addWidget(empty_lbl)
            return

        available = tts_manager.get_available_active_engines()
        for idx, rule in enumerate(rules):
            row_card = QFrame()
            row_card.setStyleSheet(
                "QFrame { background-color: #1B1F2D; border: 1px solid #38415E; border-radius: 10px; }"
                "QFrame:hover { border-color: #635BFF; }"
            )
            r_lay = QHBoxLayout(row_card)
            r_lay.setContentsMargins(10, 8, 8, 8)
            r_lay.setSpacing(6)

            cmd_edit = QLineEdit(str(rule.get("command", "!1t") or ""))
            cmd_edit.setFixedWidth(74)
            cmd_edit.setPlaceholderText("!1t")
            cmd_edit.setToolTip(tr("twitch.read.rule_cmd_tip", "Префікс команди в чаті (напр. !t, !1t, !!!)"))
            r_lay.addWidget(cmd_edit)

            eng_combo = QComboBox()
            eng_combo.setMinimumWidth(145)
            if not available:
                eng_combo.addItem(tr("twitch.read.use_active_tts", "Активний голос"), "")
            else:
                for entry in available:
                    eng_combo.addItem(entry["title"], entry["id"])
                cur_e = str(rule.get("engine_id", "") or "")
                e_idx = eng_combo.findData(cur_e)
                if e_idx >= 0:
                    eng_combo.setCurrentIndex(e_idx)
            r_lay.addWidget(eng_combo)

            voice_combo = QComboBox()
            sel_e = str(eng_combo.currentData() or "")
            self._populate_voice_combo_for_engine(voice_combo, sel_e, str(rule.get("voice_id", "") or ""))
            r_lay.addWidget(voice_combo, 1)

            vol_spin = QSpinBox()
            vol_spin.setRange(0, 200)
            vol_spin.setSingleStep(10)
            vol_spin.setSuffix("%")
            vol_spin.setFixedWidth(78)
            vol_spin.setValue(max(0, min(200, int(rule.get("volume", 100)))))
            vol_spin.setToolTip(tr("twitch.read.rule_vol_tip", "Гучність для цієї команди (%)"))
            _disable_wheel_scroll(vol_spin)
            r_lay.addWidget(vol_spin)

            del_btn = QPushButton()
            del_btn.setFixedSize(26, 26)
            del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            del_btn.setIcon(QIcon(str(BUNDLE_DIR / "assets" / "delete.svg")))
            del_btn.setIconSize(QSize(15, 15))
            del_btn.setStyleSheet("QPushButton { padding: 0px; text-align: center; }")
            del_btn.clicked.connect(lambda checked=False, i=idx: self._on_delete_twitch_voice_rule(i))
            r_lay.addWidget(del_btn)

            def _on_rule_eng_change(_i: int, ec=eng_combo, vc=voice_combo) -> None:
                self._populate_voice_combo_for_engine(vc, str(ec.currentData() or ""), "")
                self._commit_twitch_voice_rules()

            cmd_edit.editingFinished.connect(self._commit_twitch_voice_rules)
            eng_combo.currentIndexChanged.connect(_on_rule_eng_change)
            voice_combo.currentIndexChanged.connect(lambda _i: self._commit_twitch_voice_rules())
            vol_spin.valueChanged.connect(lambda _v: self._commit_twitch_voice_rules())

            self._twitch_voice_rule_widgets.append(
                {"cmd": cmd_edit, "eng": eng_combo, "voice": voice_combo, "vol": vol_spin}
            )
            self.twitch_voice_rules_lay.addWidget(row_card)

    def _commit_twitch_voice_rules(self) -> None:
        collected: List[Dict[str, Any]] = []
        for row in getattr(self, "_twitch_voice_rule_widgets", []):
            cmd = row["cmd"].text().strip()
            eng = str(row["eng"].currentData() or "")
            voice = str(row["voice"].currentData() or "")
            vol = int(row["vol"].value())
            collected.append({"command": cmd, "engine_id": eng, "voice_id": voice, "volume": vol})
        config.set("twitch_chat_voice_rules", collected)

    def _on_add_twitch_voice_rule(self) -> None:
        rules = [dict(r) for r in config.get("twitch_chat_voice_rules", []) if isinstance(r, dict)]
        existing = {str(r.get("command", "")).strip().lower() for r in rules}
        idx = 1
        while f"!{idx}t" in existing:
            idx += 1
        available = tts_manager.get_available_active_engines()
        def_eng = str(config.get("twitch_chat_tts_engine", "") or config.get("active_engine", "")).strip()
        if not def_eng and available:
            def_eng = str(available[0]["id"])
        voices = tts_manager.get_voices_for_engine(def_eng) if def_eng else []
        def_voice = str(voices[0]["id"]) if voices else ""
        rules.append(
            {
                "command": f"!{idx}t",
                "engine_id": def_eng,
                "voice_id": def_voice,
                "volume": int(config.get("twitch_chat_volume", 100)),
            }
        )
        config.set("twitch_chat_voice_rules", rules)
        self._rebuild_twitch_voice_rules()

    def _on_delete_twitch_voice_rule(self, index: int) -> None:
        rules = [dict(r) for r in config.get("twitch_chat_voice_rules", []) if isinstance(r, dict)]
        if 0 <= index < len(rules):
            rules.pop(index)
            config.set("twitch_chat_voice_rules", rules)
            self._rebuild_twitch_voice_rules()

    def _open_twitch_connection_settings(self) -> None:
        self.switch_page(3)
        if hasattr(self, "twitch_expand_btn") and not self.twitch_expand_btn.isChecked():
            self.twitch_expand_btn.setChecked(True)

    def _apply_twitch_status_ui(self, state: str, detail: str) -> None:
        enabled = bool(
            (hasattr(self, "twitch_enable_chk") and self.twitch_enable_chk.isChecked())
            or (hasattr(self, "twitch_intercept_chk") and self.twitch_intercept_chk.isChecked())
            or (hasattr(self, "twitch_chat_intercept_chk") and self.twitch_chat_intercept_chk.isChecked())
            or (hasattr(self, "twitch_read_viewers_chk") and self.twitch_read_viewers_chk.isChecked())
        )
        if not enabled:
            if hasattr(self, "twitch_badge_lbl"):
                self.twitch_badge_lbl.setVisible(False)
                self.twitch_badge_lbl.setText("")
            if hasattr(self, "twitch_tab_status_lbl"):
                self.twitch_tab_status_lbl.setText(tr("twitch.status.off", "⚪ Вимкнено"))
                self.twitch_tab_status_lbl.setStyleSheet("font-size: 12px; color: #AAB4C8;")
            return

        if state == "connected":
            txt = tr("twitch.status.connected", "🟢 Підключено (#{channel})", channel=detail)
            col = "#4ADE80"
        elif state == "connecting":
            txt = tr("twitch.status.connecting", "🟡 Підключення...")
            col = "#FACC15"
        elif state == "error":
            txt = tr("twitch.status.error", "🔴 Помилка підключення")
            col = "#F87171"
            if detail and hasattr(self, "twitch_status_lbl"):
                self.twitch_status_lbl.setStyleSheet("color: #F87171;")
                self.twitch_status_lbl.setText(tr("twitch.test_fail", "Помилка: {error}", error=detail))
        else:
            tok_missing = not str(config.get("twitch_token", "")).strip()
            txt = (
                tr("twitch.status.no_token", "🟡 Потрібен Access Token")
                if tok_missing
                else tr("twitch.status.off", "⚪ Вимкнено")
            )
            col = "#FACC15" if tok_missing else "#AAB4C8"

        if hasattr(self, "twitch_badge_lbl"):
            self.twitch_badge_lbl.setVisible(True)
            self.twitch_badge_lbl.setText(txt)
            self.twitch_badge_lbl.setStyleSheet(f"font-size: 12px; color: {col};")
        if hasattr(self, "twitch_tab_status_lbl"):
            self.twitch_tab_status_lbl.setText(txt)
            self.twitch_tab_status_lbl.setStyleSheet(f"font-size: 12px; color: {col};")

    def _sync_twitch_connection(self) -> None:
        from core.twitch_chat import twitch_chat

        enabled = bool(self.twitch_enable_chk.isChecked()) if hasattr(self, "twitch_enable_chk") else bool(config.get("twitch_enabled", False))
        intercept = bool(self.twitch_intercept_chk.isChecked()) if hasattr(self, "twitch_intercept_chk") else bool(config.get("twitch_intercept_own", False))
        read_viewers = bool(self.twitch_read_viewers_chk.isChecked()) if hasattr(self, "twitch_read_viewers_chk") else bool(config.get("twitch_read_viewers", False))
        tok = self.twitch_token_edit.text().strip() if hasattr(self, "twitch_token_edit") else str(config.get("twitch_token", "")).strip()
        chan = self.twitch_channel_edit.text().strip() if hasattr(self, "twitch_channel_edit") else str(config.get("twitch_channel", "")).strip()

        if (enabled or intercept or read_viewers) and tok:
            try:
                twitch_chat.configure(
                    enabled,
                    "",
                    tok,
                    chan,
                    intercept_own=intercept,
                    read_viewers=read_viewers,
                )
                st, dtl = twitch_chat.status
                self._apply_twitch_status_ui(st, dtl)
            except ValueError as exc:
                twitch_chat.configure(False, "", "", "", intercept_own=False, read_viewers=False)
                self._apply_twitch_status_ui("error", str(exc))
        else:
            twitch_chat.configure(False, "", "", "", intercept_own=False, read_viewers=False)
            self._apply_twitch_status_ui("disconnected", "")
            if (enabled or intercept or read_viewers) and hasattr(self, "twitch_status_lbl"):
                self.twitch_status_lbl.setStyleSheet("color: #FACC15;")
                self.twitch_status_lbl.setText(tr("twitch.err.fill_fields", "Вставте Access Token Twitch."))

    def _save_twitch_settings(self, source: str = "overlay", *_args: Any) -> None:
        if source == "chat_tab" and hasattr(self, "twitch_chat_intercept_chk") and hasattr(self, "twitch_intercept_chk"):
            val = self.twitch_chat_intercept_chk.isChecked()
            if self.twitch_intercept_chk.isChecked() != val:
                self.twitch_intercept_chk.blockSignals(True)
                self.twitch_intercept_chk.setChecked(val)
                self.twitch_intercept_chk.blockSignals(False)
        elif source == "overlay" and hasattr(self, "twitch_intercept_chk") and hasattr(self, "twitch_chat_intercept_chk"):
            val = self.twitch_intercept_chk.isChecked()
            if self.twitch_chat_intercept_chk.isChecked() != val:
                self.twitch_chat_intercept_chk.blockSignals(True)
                self.twitch_chat_intercept_chk.setChecked(val)
                self.twitch_chat_intercept_chk.blockSignals(False)

        self._update_twitch_allowed_row()
        self._update_twitch_command_row()
        updates: Dict[str, Any] = {}
        if hasattr(self, "twitch_enable_chk"):
            updates["twitch_enabled"] = self.twitch_enable_chk.isChecked()
        if hasattr(self, "twitch_intercept_chk"):
            updates["twitch_intercept_own"] = self.twitch_intercept_chk.isChecked()
        elif hasattr(self, "twitch_chat_intercept_chk"):
            updates["twitch_intercept_own"] = self.twitch_chat_intercept_chk.isChecked()
        if hasattr(self, "twitch_token_edit"):
            updates["twitch_token"] = self.twitch_token_edit.text().strip()
        if hasattr(self, "twitch_channel_edit"):
            updates["twitch_channel"] = self.twitch_channel_edit.text().strip()
        if hasattr(self, "twitch_read_viewers_chk"):
            read_viewers_on = self.twitch_read_viewers_chk.isChecked()
            if not read_viewers_on:
                self._twitch_chat_paused = False
            updates["twitch_read_viewers"] = read_viewers_on
            updates["twitch_chat_read_all"] = self.twitch_chat_read_all_chk.isChecked()
            updates["twitch_chat_command_prefix"] = self.twitch_chat_prefix_edit.text().strip()
            updates["twitch_chat_tts_engine"] = str(self.twitch_chat_engine_combo.currentData() or "")
            updates["twitch_chat_tts_voice"] = str(self.twitch_chat_voice_combo.currentData() or "")
            updates["twitch_chat_volume"] = int(self.twitch_chat_vol_slider.value())
            updates["twitch_viewer_filter"] = str(self.twitch_viewer_filter_combo.currentData() or "all")
            updates["twitch_allowed_users"] = self.twitch_allowed_edit.text().strip()
            updates["twitch_ignored_users"] = self.twitch_ignored_edit.text().strip()
            updates["twitch_read_format"] = str(self.twitch_read_format_combo.currentData() or "user_says")
            updates["twitch_max_chars"] = int(self.twitch_max_chars_spin.value())
            updates["twitch_user_cooldown_sec"] = int(self.twitch_cooldown_spin.value())
            updates["twitch_max_queue"] = int(self.twitch_max_queue_spin.value())
            updates["twitch_ignore_commands"] = self.twitch_ignore_cmds_chk.isChecked()
            updates["twitch_ignore_links"] = self.twitch_ignore_links_chk.isChecked()
            updates["twitch_save_viewers_history"] = self.twitch_save_viewers_hist_chk.isChecked()
            updates["twitch_chat_send_vbcable"] = self.twitch_chat_vbcable_chk.isChecked()
            updates["twitch_chat_send_vban"] = self.twitch_chat_vban_chk.isChecked()
            updates["twitch_chat_vban_mode"] = str(self.twitch_chat_vban_combo.currentData() or "same")
            vban_chat_name = self.twitch_chat_vban_stream_edit.text().strip()[:16] or "SilentChat"
            updates["twitch_chat_vban_stream"] = vban_chat_name
        if updates:
            config.update(updates)
        if "twitch_read_viewers" in updates:
            self.keyboard_hook.update_from_config(config.data)
            self._refresh_chat_hotkey_availability()
        if bool(config.get("twitch_read_viewers", False)) and (
            "twitch_read_viewers" in updates or "twitch_chat_tts_engine" in updates or "twitch_chat_tts_voice" in updates
        ):
            tts_manager.schedule_smart_warmup(0.6, "twitch_changed")
        if hasattr(self, "twitch_status_lbl"):
            self.twitch_status_lbl.setText("")
        self._sync_twitch_connection()

    def _on_toggle_chat_tts_hotkey(self) -> None:
        if not bool(config.get("twitch_read_viewers", False)):
            return
        self._twitch_chat_paused = not getattr(self, "_twitch_chat_paused", False)
        if self._twitch_chat_paused:
            tts_manager.stop_chat_tts()
            msg = tr("twitch.read.paused", "🔇 Озвучку чату призупинено")
            self._append_twitch_feed("⏸️", "SilentVoice", msg)
        else:
            msg = tr("twitch.read.resumed", "🔊 Озвучку чату відновлено")
            self._append_twitch_feed("▶️", "SilentVoice", msg)
        if hasattr(self, "live_status_label"):
            self.live_status_label.setText(msg)

    def _on_toggle_game_mode_hotkey(self) -> None:
        new_state = not bool(config.get("game_mode_enabled", True))
        if hasattr(self, "chk_game_mode"):
            self.chk_game_mode.setChecked(new_state)
        else:
            config.set("game_mode_enabled", new_state)
            self.keyboard_hook.update_from_config(config.data)
        audio_router.play_feedback_beep("start" if new_state else "cancel")
        if hasattr(self, "live_status_label"):
            msg = (
                tr("game_mode.enabled_status", "🛡️ Ігровий режим увімкнено")
                if new_state
                else tr("game_mode.disabled_status", "🛡️ Ігровий режим вимкнено")
            )
            self.live_status_label.setText(msg)

    def _on_toggle_omnivoice_whisper_hotkey(self) -> None:
        if str(config.get("active_engine", "") or "").strip() != "omnivoice":
            return
        new_state = not bool(config.get("omnivoice_whisper", False))
        config.set("omnivoice_whisper", new_state)
        self._refresh_clone_card_ui("omnivoice")
        if hasattr(self, "overlay") and self.overlay.isVisible():
            self.overlay.update()
        audio_router.play_feedback_beep("start" if new_state else "cancel")
        if hasattr(self, "live_status_label"):
            msg = (
                tr("omnivoice.whisper.enabled_status", "🤫 Шепіт (OmniVoice) увімкнено")
                if new_state
                else tr("omnivoice.whisper.disabled_status", "🗣️ Шепіт (OmniVoice) вимкнено")
            )
            self.live_status_label.setText(msg)

    def _on_reaction_commands_changed(self) -> None:
        """Оновлює активні звукові команди у хуку клавіатури та оверлеї після редагування списку."""
        if hasattr(self, "keyboard_hook"):
            self.keyboard_hook.update_from_config(config.data)
        if hasattr(self, "overlay") and self.overlay.isVisible():
            self.overlay.update()

    def _on_reaction_style_preview(self, style: str) -> None:
        """Показує живий попередній перегляд обраного стилю підказки команд поверх екрана."""
        if hasattr(self, "overlay"):
            self.overlay.preview_tag_picker_style(style)

    def _on_reaction_hotkey_triggered(self, tag: str) -> None:
        """Вставляє звукову команду за біндом у відкрите вікно вводу."""
        if hasattr(self, "fallback_window") and self.fallback_window.isVisible():
            self.fallback_window.insert_reaction_tag(tag)

    def _append_twitch_feed(self, icon: str, author: str, text: str, note: str = "") -> None:
        if not hasattr(self, "twitch_live_feed"):
            return
        ts = time.strftime("%H:%M")
        suffix = f" ({note})" if note else ""
        self.twitch_live_feed.insertItem(0, f"[{ts}] {icon} {author}: {text}{suffix}")
        while self.twitch_live_feed.count() > 120:
            self.twitch_live_feed.takeItem(self.twitch_live_feed.count() - 1)

    def _test_twitch_reader_sample(self) -> None:
        from core.twitch_chat import TwitchChatMessage, evaluate_chat_message

        sample = TwitchChatMessage(
            username="viewer",
            display_name=tr("twitch.read.sample_viewer", "Глядач"),
            text=tr("twitch.read.sample_text", "Привіт! Класний стрім!"),
            channel=str(config.get("twitch_channel", "stream") or "stream"),
            is_own=False,
            is_mod=True,
            is_vip=True,
            is_subscriber=True,
        )
        spoken, _, _ = evaluate_chat_message(
            sample,
            read_own=True,
            read_viewers=True,
            viewer_filter="all",
            ignore_commands=False,
            ignore_links=False,
            read_format=str(config.get("twitch_read_format", "user_says") or "user_says"),
            max_chars=int(config.get("twitch_max_chars", 220) or 0),
            say_template=tr("twitch.read.template_says", "{user} каже: {text}"),
            read_all=True,
        )
        if spoken:
            self._append_twitch_feed("🔊", sample.display_name, sample.text, tr("twitch.read.test_badge", "тест"))
            tts_manager.speak(
                spoken,
                save_history=False,
                skip_spellcheck=True,
                mirror_twitch=False,
                is_viewer_chat=True,
                engine_override=str(config.get("twitch_chat_tts_engine", "") or ""),
                voice_override=str(config.get("twitch_chat_tts_voice", "") or ""),
                volume_override=int(config.get("twitch_chat_volume", 100)),
            )

    def _on_twitch_chat_message(self, msg: Any) -> None:
        from core.twitch_chat import TwitchChatMessage, evaluate_chat_message

        if not isinstance(msg, TwitchChatMessage):
            return
        read_own = bool(config.get("twitch_intercept_own", False))
        read_viewers = bool(config.get("twitch_read_viewers", False))
        if not msg.is_own and getattr(self, "_twitch_chat_paused", False):
            return

        spoken, reason, matched_rule = evaluate_chat_message(
            msg,
            read_own=read_own,
            read_viewers=read_viewers,
            viewer_filter=str(config.get("twitch_viewer_filter", "all") or "all"),
            allowed_users=str(config.get("twitch_allowed_users", "")),
            ignored_users=str(config.get("twitch_ignored_users", "")),
            ignore_commands=bool(config.get("twitch_ignore_commands", True)),
            ignore_links=bool(config.get("twitch_ignore_links", True)),
            read_format=str(config.get("twitch_read_format", "user_says") or "user_says"),
            max_chars=int(config.get("twitch_max_chars", 220) or 0),
            say_template=tr("twitch.read.template_says", "{user} каже: {text}"),
            read_all=bool(config.get("twitch_chat_read_all", False)),
            command_prefix=str(config.get("twitch_chat_command_prefix", "!!!") or ""),
            voice_rules=list(config.get("twitch_chat_voice_rules", [])),
        )
        if spoken is None:
            if not msg.is_own and read_viewers and reason != "disabled":
                self._append_twitch_feed("🔇", msg.display_name, msg.text, reason)
            return

        if not msg.is_own:
            cooldown = max(0, int(config.get("twitch_user_cooldown_sec", 0) or 0))
            now = time.monotonic()
            if cooldown > 0:
                last_ts = self._twitch_user_last_spoken.get(msg.username, 0.0)
                if (now - last_ts) < cooldown:
                    self._append_twitch_feed("🔇", msg.display_name, msg.text, tr("twitch.read.skip_cd", "пауза"))
                    return
            max_q = max(1, int(config.get("twitch_max_queue", 5) or 5))
            if tts_manager.pending_queue_size() >= max_q:
                self._append_twitch_feed("🔇", msg.display_name, msg.text, tr("twitch.read.skip_queue", "черга"))
                return
            self._twitch_user_last_spoken[msg.username] = now
            rule_note = str(matched_rule.get("command", "")) if matched_rule else ""
            self._append_twitch_feed("🔊", msg.display_name, msg.text, rule_note)

            eng_ovr = (
                str(matched_rule.get("engine_id", "") or "")
                if matched_rule
                else str(config.get("twitch_chat_tts_engine", "") or "")
            )
            voice_ovr = (
                str(matched_rule.get("voice_id", "") or "")
                if matched_rule
                else str(config.get("twitch_chat_tts_voice", "") or "")
            )
            vol_ovr = (
                int(matched_rule.get("volume", 100))
                if matched_rule
                else int(config.get("twitch_chat_volume", 100))
            )
            tts_manager.speak(
                spoken,
                save_history=bool(config.get("twitch_save_viewers_history", False)),
                skip_spellcheck=True,
                mirror_twitch=False,
                is_viewer_chat=True,
                engine_override=eng_ovr,
                voice_override=voice_ovr,
                volume_override=vol_ovr,
                chat_user=msg.username,
                chat_msg_id=msg.msg_id,
            )
            return

        self._append_twitch_feed("🔊", f"{msg.display_name} ({tr('twitch.read.you', 'ви')})", msg.text)
        tts_manager.speak(spoken, save_history=True, mirror_twitch=False)

    def _on_twitch_moderation(self, event_type: str, target_user: str, target_msg_id: str) -> None:
        dropped = tts_manager.cancel_chat_by_moderation(event_type, target_user, target_msg_id)
        if dropped > 0:
            who = target_user or tr("twitch.read.chat", "чат")
            self._append_twitch_feed(
                "🚫",
                who,
                tr("twitch.read.moderated", "повідомлення видалено/таймаут модератором"),
            )

    def _refresh_ai_status_banner(self) -> None:
        if not hasattr(self, "ai_status_banner"):
            return
        enabled = bool(config.get("ai_spellcheck_enabled", False))
        items: List[Dict[str, Any]] = list(config.get("ai_items", []))
        if not enabled:
            self.ai_status_banner.setText(tr("ui_217", "ℹ️ AI-коректор вимкнено. Додайте сервіс і увімкніть його через кнопку ⚙️."))
            self.ai_status_banner.setStyleSheet("color: #94A3B8; padding: 2px 6px; font-size: 12.5px;")
            return
        if not items:
            self.ai_status_banner.setText(tr("ui_218", "⚠️ AI-коректор увімкнено, але не додано жодного сервісу."))
            self.ai_status_banner.setStyleSheet("color: #FBBF24; padding: 2px 6px; font-size: 12.5px;")
            return
        prim_id = str(config.get("ai_primary_id", "")).strip()
        fb_id = str(config.get("ai_fallback_id", "")).strip()
        prim_item = next((x for x in items if str(x.get("id")) == prim_id), items[0])
        fb_item = next((x for x in items if str(x.get("id")) == fb_id), None) if fb_id else None
        prim_title = get_ai_item_title(prim_item)
        fb_part = tr("ui_f_010", " • Запасний: <b>{title}</b>", title=get_ai_item_title(fb_item)) if fb_item else ""
        self.ai_status_banner.setText(tr("ui_f_011", "✅ Активний AI-коректор: <b>{prim_title}</b>{fb_part}", prim_title=prim_title, fb_part=fb_part))
        self.ai_status_banner.setStyleSheet("color: #34D399; padding: 2px 6px; font-size: 12.5px;")

    def _on_ai_spellcheck_toggled(self, checked: bool) -> None:
        config.set("ai_spellcheck_enabled", bool(checked))
        if checked and not str(config.get("ai_primary_id", "")).strip():
            items = list(config.get("ai_items", []))
            if items:
                config.set("ai_primary_id", str(items[0].get("id", "")))
        if checked:
            self._ensure_ollama_server()
            self._start_saved_local_llm_servers()
        self._refresh_ai_status_banner()

    def _start_saved_local_llm_servers(self) -> None:
        if not bool(config.get("ai_spellcheck_enabled", False)) or self._local_llm_restore_in_progress:
            return
        items = [
            dict(item) for item in config.get("ai_items", [])
            if isinstance(item, dict)
            and str(item.get("provider", "")) == "local_api"
            and (
                (str(item.get("managed_backend", "")) == "llama_cpp" and str(item.get("model_path", "")).strip())
                or (str(item.get("managed_backend", "")) == "lmstudio" and str(item.get("model", "")).strip())
            )
        ]
        if not items:
            return
        self._local_llm_restore_in_progress = True

        def _worker() -> None:
            try:
                for item in items:
                    try:
                        if str(item.get("managed_backend", "")) == "lmstudio":
                            start_lmstudio_model(str(item.get("model", "")))
                            continue
                        raw_url = str(item.get("base_url", "")).strip()
                        port = urlparse(raw_url).port if raw_url else None
                        start_llama_cpp_server(str(item["model_path"]), port=port)
                    except Exception as exc:
                        print(f"[LocalLLM] Could not restart saved local model: {exc}", file=sys.stderr, flush=True)
            finally:
                self.local_llm_restore_finished.emit()

        threading.Thread(target=_worker, daemon=True, name="RestoreLocalLLMServers").start()

    def _on_local_llm_restore_finished(self) -> None:
        self._local_llm_restore_in_progress = False

    @staticmethod
    def _is_ollama_server_running() -> bool:
        try:
            with socket.create_connection(("127.0.0.1", 11434), timeout=0.6):
                return True
        except OSError:
            return False

    def _ensure_ollama_server(self) -> None:
        if self._ollama_start_in_progress or self._ollama_executable() is None:
            return
        self._ollama_start_in_progress = True

        def _start_or_detect() -> None:
            ready = self._is_ollama_server_running()
            if not ready:
                executable = self._ollama_executable()
                if executable:
                    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    try:
                        subprocess.Popen(
                            [executable, "serve"],
                            creationflags=flags,
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    except Exception:
                        pass
                    for _ in range(30):
                        if self._is_ollama_server_running():
                            ready = True
                            break
                        threading.Event().wait(0.5)
            self.ollama_server_ready.emit(ready)

        threading.Thread(target=_start_or_detect, daemon=True).start()

    def _on_ollama_server_ready(self, ready: bool) -> None:
        self._ollama_start_in_progress = False
        ollama_items = [
            item for item in config.get("ai_items", [])
            if str(item.get("provider", "")) == "ollama"
        ]
        if ready:
            for item in ollama_items:
                self._start_fetch_ai_models(str(item.get("id", "")), None, silent=True)
        else:
            for item in ollama_items:
                status = self._ai_status_labels.get(str(item.get("id", "")))
                if status is not None:
                    status.setText(tr("ui_ollama_start_failed", "⚠️ Не вдалося запустити Ollama. Перевірте встановлення."))

    def _open_ai_settings_dialog(self) -> None:
        dlg = AISettingsDialog(self)
        if dlg.exec():
            enabled = bool(config.get("ai_spellcheck_enabled", False))
            if hasattr(self, "chk_ai_spellcheck"):
                self.chk_ai_spellcheck.blockSignals(True)
                self.chk_ai_spellcheck.setChecked(enabled)
                self.chk_ai_spellcheck.blockSignals(False)
            if enabled:
                self._ensure_ollama_server()
                self._start_saved_local_llm_servers()
            self._refresh_ai_status_banner()

    def _rebuild_ai_cards(self) -> None:
        if not hasattr(self, "ai_cards_layout"):
            return
        while self.ai_cards_layout.count():
            item = self.ai_cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._ai_status_labels.clear()
        self._ai_model_combos.clear()
        self._ollama_model_buttons.clear()
        self._ollama_install_buttons.clear()
        self._ai_test_buttons.clear()
        self._ollama_download_widgets.clear()
        ai_items: List[Dict[str, Any]] = list(config.get("ai_items", []))
        for item in ai_items:
            item_id = str(item.get("id", ""))
            prov_key = str(item.get("provider", "groq"))
            spec = AI_PROVIDERS.get(prov_key, AI_PROVIDERS["groq"])
            title_str = get_ai_item_title(item)
            saved_key = str(item.get("api_key", ""))
            saved_model = str(item.get("model", spec["default_model"]))

            card = QFrame()
            card.setProperty("class", "card")
            c_lay = QVBoxLayout(card)
            c_lay.setContentsMargins(14, 10, 14, 10)
            c_lay.setSpacing(8)

            top_r = QHBoxLayout()
            top_r.setSpacing(8)
            t_lbl = QLabel(f"<b>{title_str}</b>")
            t_lbl.setStyleSheet("color: #FFFFFF; font-size: 14px;")
            top_r.addWidget(t_lbl)
            top_r.addStretch(1)

            if prov_key in ("ollama", "local_api"):
                local_models_btn = QPushButton("…")
                local_models_btn.setMinimumWidth(44)
                local_models_btn.setMinimumHeight(33)
                local_models_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                local_models_btn.setToolTip(tr("ui_local_models_tooltip", "Підключити інші локальні моделі з ПК"))
                local_models_btn.clicked.connect(lambda checked=False: self._open_local_models_dialog())
                top_r.addWidget(local_models_btn)

            guide_btn = QPushButton(tr("ui_141", "📘 Гайд"))
            guide_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            guide_btn.setToolTip(tr("ui_f_012", "Як отримати API ключ / підключити {title}", title=title_str))
            guide_btn.clicked.connect(lambda checked=False, p=prov_key: ApiGuideDialog(p, self).exec())
            top_r.addWidget(guide_btn)

            del_btn = QPushButton(tr("ui_137", "Видалити"))
            del_btn.setProperty("class", "dangerBtn")
            del_btn.setMinimumWidth(100)
            del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            del_btn.setToolTip(tr("ui_219", "Видалити цей AI-сервіс зі списку"))
            del_btn.clicked.connect(lambda checked=False, iid=item_id: self._on_delete_ai_item(iid))
            top_r.addWidget(del_btn)
            c_lay.addLayout(top_r)

            if spec.get("needs_key", True):
                key_r = QHBoxLayout()
                key_r.setSpacing(8)
                key_lbl = QLabel(tr("ui_143", "API ключ:"))
                key_lbl.setStyleSheet("color: #CBD5E1; font-size: 13px;")
                key_r.addWidget(key_lbl)

                key_edit = QLineEdit(saved_key)
                key_edit.setEchoMode(QLineEdit.EchoMode.Password)
                key_edit.setPlaceholderText(tr("ui_144", "Вставте ваш API ключ сюди..."))

                def _on_ai_key_changed(val: str, iid: str = item_id) -> None:
                    cleaned = val.strip()
                    self._on_ai_item_field_changed(iid, "api_key", cleaned)
                    if len(cleaned) >= 12:
                        t = self._ai_debounce_timers.get(iid)
                        if t is None:
                            t = QTimer(self)
                            t.setSingleShot(True)
                            t.timeout.connect(lambda i=iid: self._start_fetch_ai_models(i, None, silent=True))
                            self._ai_debounce_timers[iid] = t
                        t.start(900)

                key_edit.textChanged.connect(_on_ai_key_changed)
                key_r.addWidget(key_edit, 1)

                show_btn = QPushButton(tr("ui_145", "Показати"))
                show_btn.setMinimumWidth(96)
                show_btn.setCursor(Qt.CursorShape.PointingHandCursor)

                def _toggle_ai_echo(checked: bool = False, edit: QLineEdit = key_edit, btn: QPushButton = show_btn) -> None:
                    if edit.echoMode() == QLineEdit.EchoMode.Password:
                        edit.setEchoMode(QLineEdit.EchoMode.Normal)
                        btn.setText(tr("ui_146", "Сховати"))
                    else:
                        edit.setEchoMode(QLineEdit.EchoMode.Password)
                        btn.setText(tr("ui_145", "Показати"))

                show_btn.clicked.connect(_toggle_ai_echo)
                key_r.addWidget(show_btn)
                c_lay.addLayout(key_r)
            elif prov_key == "ollama":
                ollama_r = QHBoxLayout()
                ollama_r.setSpacing(8)
                ollama_status = QLabel(tr("ui_ollama_local", "Локальний сервер: http://localhost:11434"))
                ollama_status.setStyleSheet("color: #94A3B8; font-size: 12px;")
                ollama_r.addWidget(ollama_status, 1)
                install_ollama_btn = QPushButton(tr("ui_ollama_install", "⬇ Встановити Ollama"))
                install_ollama_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                install_ollama_btn.clicked.connect(
                    lambda checked=False, iid=item_id: self._manage_ollama_install(iid, remove=False)
                )
                ollama_r.addWidget(install_ollama_btn)
                remove_ollama_btn = QPushButton(tr("ui_ollama_remove", "🗑 Видалити Ollama"))
                remove_ollama_btn.setProperty("class", "dangerBtn")
                remove_ollama_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                remove_ollama_btn.clicked.connect(
                    lambda checked=False, iid=item_id: self._manage_ollama_install(iid, remove=True)
                )
                ollama_r.addWidget(remove_ollama_btn)
                c_lay.addLayout(ollama_r)
                install_progress = QProgressBar()
                install_progress.setRange(0, 100)
                install_progress.setValue(0)
                install_progress.setVisible(False)
                c_lay.addWidget(install_progress)
                install_status = QLabel("")
                install_status.setStyleSheet("color: #CBD5E1; font-size: 12px;")
                install_status.setWordWrap(True)
                install_status.setVisible(False)
                c_lay.addWidget(install_status)
                install_cancel_btn = QPushButton(tr("tts.dl.cancel_btn", "✕ Скасувати"))
                install_cancel_btn.setProperty("class", "dangerBtn")
                install_cancel_btn.setVisible(False)
                install_cancel_btn.clicked.connect(self._ollama_download_cancel.set)
                c_lay.addWidget(install_cancel_btn)
                self._ollama_download_widgets[item_id] = {
                    "progress": install_progress,
                    "status": install_status,
                    "cancel": install_cancel_btn,
                }
                if self._ollama_download_active and getattr(self, "_ollama_install_item_id", "") == item_id:
                    install_progress.setVisible(True)
                    install_status.setVisible(True)
                    install_cancel_btn.setVisible(True)
                self._ollama_install_buttons[item_id] = (install_ollama_btn, remove_ollama_btn)
                self._refresh_ollama_install_buttons(item_id)
            elif prov_key == "local_api":
                url_r = QHBoxLayout()
                url_r.setSpacing(8)
                url_lbl = QLabel(tr("ui_local_api_url", "Адреса сервера:"))
                url_lbl.setStyleSheet("color: #CBD5E1; font-size: 13px;")
                url_r.addWidget(url_lbl)
                url_edit = QLineEdit(str(item.get("base_url", spec.get("default_base_url", ""))))
                url_edit.setPlaceholderText(str(spec.get("default_base_url", "http://localhost:1234/v1")))
                url_edit.setToolTip(tr("ui_local_api_url_tip", "Адреса локального API-сервера з його налаштувань."))
                url_edit.textChanged.connect(
                    lambda val, iid=item_id: self._on_ai_item_field_changed(iid, "base_url", val.strip())
                )
                url_r.addWidget(url_edit, 1)
                c_lay.addLayout(url_r)

                key_checkbox = QCheckBox(tr("ui_local_api_key", "Сервер просить ключ доступу"))
                key_checkbox.setChecked(bool(saved_key))
                c_lay.addWidget(key_checkbox)
                local_key_edit = QLineEdit(saved_key)
                local_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
                local_key_edit.setPlaceholderText(tr("ui_local_api_key_ph", "Вставте ключ доступу, якщо сервер його вимагає"))
                local_key_edit.setVisible(bool(saved_key))
                local_key_edit.textChanged.connect(
                    lambda val, iid=item_id: self._on_ai_item_field_changed(iid, "api_key", val.strip())
                )

                def _toggle_local_api_key(enabled: bool, edit: QLineEdit = local_key_edit) -> None:
                    edit.setVisible(enabled)
                    if not enabled:
                        edit.clear()

                key_checkbox.toggled.connect(_toggle_local_api_key)
                c_lay.addWidget(local_key_edit)

            mod_r = QHBoxLayout()
            mod_r.setSpacing(8)
            mod_lbl = QLabel(tr("ui_150", "Модель:"))
            mod_lbl.setStyleSheet("color: #CBD5E1; font-size: 13px;")
            mod_r.addWidget(mod_lbl)

            # Ollama's model selector is editable. Use the native Qt combo here:
            # VoiceComboBox custom-paints its arrow, which makes the small drop-down
            # hit area awkward next to the line editor.
            mod_combo = ModelComboBox() if prov_key == "ollama" else QComboBox()
            mod_combo.setEditable(prov_key == "ollama")
            if prov_key == "ollama":
                mod_combo.setMaxVisibleItems(12)
                mod_combo.setInsertPolicy(QtComboBox.InsertPolicy.NoInsert)
                mod_combo.setCompleter(None)
            mod_combo.setToolTip(
                tr(
                    "ui_ollama_model_tip",
                    "Оберіть модель зі списку або вставте назву з тегом зі сторінки бібліотеки Ollama.",
                )
                if prov_key == "ollama"
                else tr("ui_223", "Оберіть AI-модель зі списку для виправлення тексту")
            )
            avail_models = [str(m).strip() for m in (item.get("available_models") or []) if str(m).strip()]
            model_choices: List[str] = []
            if prov_key == "ollama":
                if self._ollama_connections.get(item_id, False):
                    for cand in [*OLLAMA_RECOMMENDED_MODELS, *avail_models]:
                        if cand and cand not in model_choices:
                            model_choices.append(cand)
            else:
                for cand in [saved_model, str(spec["default_model"]), *avail_models]:
                    if cand and cand not in model_choices:
                        model_choices.append(cand)
            if prov_key == "ollama":
                installed_models = set(avail_models)
                installed_model_keys = {normalize_ollama_model_name(model_name) for model_name in installed_models}
                choice_keys: set[str] = set()
                for model_name in model_choices:
                    model_key = normalize_ollama_model_name(model_name)
                    if not model_key or model_key in choice_keys:
                        continue
                    choice_keys.add(model_key)
                    installed_label = tr("ui_ollama_model_installed", "встановлено")
                    label = f"{model_name} ({installed_label})" if model_key in installed_model_keys else model_name
                    mod_combo.addItem(label, model_name)
            else:
                mod_combo.addItems(model_choices)
            if model_choices:
                wanted_model = next(
                    (
                        choice for choice in model_choices
                        if (
                            normalize_ollama_model_name(choice) == normalize_ollama_model_name(saved_model)
                            if prov_key == "ollama"
                            else choice == saved_model
                        )
                    ),
                    model_choices[0],
                )
                if prov_key == "ollama":
                    mod_combo.setCurrentIndex(mod_combo.findData(wanted_model))
                else:
                    mod_combo.setCurrentText(wanted_model)
            mod_combo.currentTextChanged.connect(
                lambda _text, iid=item_id, cb=mod_combo: self._on_ai_item_field_changed(
                    iid, "model", self._selected_ai_model(cb)
                )
            )
            self._ai_model_combos[item_id] = mod_combo
            mod_r.addWidget(mod_combo, 1)
            test_btn = QPushButton(tr("ai.test.open", "Перевірити"))
            test_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            test_btn.setToolTip(tr("ai.test.tooltip", "Перевірити вибрану модель на прикладі або власному тексті"))
            test_btn.clicked.connect(lambda checked=False, iid=item_id: self._open_ai_test_dialog(iid))
            self._ai_test_buttons[item_id] = test_btn
            mod_r.addWidget(test_btn)
            mod_combo.currentTextChanged.connect(lambda _text, iid=item_id: self._refresh_ai_test_button(iid))
            self._refresh_ai_test_button(item_id)

            refresh_mod_btn = QPushButton(tr("ui_224", "🔄 Оновити моделі"))
            refresh_mod_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            refresh_mod_btn.clicked.connect(
                lambda checked=False, iid=item_id, b=refresh_mod_btn: self._start_fetch_ai_models(iid, b, silent=False)
            )
            mod_r.addWidget(refresh_mod_btn)
            if prov_key == "ollama":
                pull_model_btn = QPushButton(tr("ui_ollama_pull", "⬇ Завантажити модель"))
                pull_model_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                pull_model_btn.clicked.connect(
                    lambda checked=False, iid=item_id, cb=mod_combo: self._pull_ollama_model(
                        iid, self._selected_ai_model(cb)
                    )
                )
                mod_r.addWidget(pull_model_btn)
                self._ollama_model_buttons[item_id] = pull_model_btn
                mod_combo.currentTextChanged.connect(
                    lambda _text, iid=item_id: self._refresh_ollama_model_button(iid)
                )
                self._refresh_ollama_model_button(item_id)
                self._refresh_ai_test_button(item_id)
            c_lay.addLayout(mod_r)

            if prov_key == "ollama" and not self._ollama_connections.get(item_id, False):
                st_text = tr("ui_ollama_disconnected", "⚪ Ollama не підключена — список моделей з’явиться після підключення")
            elif prov_key == "local_api" and not avail_models:
                st_text = tr("ui_local_api_disconnected", "⚪ Запустіть локальний API-сервер і натисніть «Оновити моделі»")
            elif avail_models:
                st_text = tr("ui_f_014", "🟢 Завантажено моделей: {count}", count=len(avail_models))
            elif saved_key.strip():
                st_text = tr("ui_227", "⚪ Натисніть 'Оновити моделі' для завантаження списку з API")
            else:
                st_text = tr("ui_228", "⚪ Введіть API ключ — список моделей підтягнеться автоматично")
            st_lbl = QLabel(st_text)
            st_lbl.setStyleSheet("color: #94A3B8; font-size: 11px;")
            self._ai_status_labels[item_id] = st_lbl
            c_lay.addWidget(st_lbl)

            self.ai_cards_layout.addWidget(card)

    def _refresh_ai_test_button(self, item_id: str) -> None:
        btn = self._ai_test_buttons.get(item_id)
        combo = self._ai_model_combos.get(item_id)
        if btn is None or combo is None:
            return
        item = next((x for x in config.get("ai_items", []) if str(x.get("id", "")) == item_id), {})
        provider = str(item.get("provider", ""))
        spec = AI_PROVIDERS.get(provider, {})
        selected_model = self._selected_ai_model(combo)
        has_model = bool(selected_model)
        if provider == "ollama":
            installed_models = {
                normalize_ollama_model_name(str(model).strip())
                for model in item.get("available_models", [])
                if str(model).strip()
            }
            ready = (
                bool(self._ollama_connections.get(item_id, False))
                and has_model
                and normalize_ollama_model_name(selected_model) in installed_models
            )
        elif provider == "local_api":
            ready = bool(str(item.get("base_url", "")).strip()) and has_model
        else:
            ready = has_model and (not spec.get("needs_key", True) or bool(str(item.get("api_key", "")).strip()))
        btn.setVisible(ready)

    def _open_ai_test_dialog(self, item_id: str) -> None:
        item = next((dict(x) for x in config.get("ai_items", []) if str(x.get("id", "")) == item_id), None)
        if item:
            AICorrectorTestDialog(item, self).exec()

    @staticmethod
    def _ollama_executable() -> Optional[str]:
        executable = shutil.which("ollama")
        if executable:
            return executable
        for candidate in (
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe"),
            os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "Ollama", "ollama.exe"),
        ):
            if candidate and os.path.isfile(candidate):
                return candidate
        return None

    def _refresh_ollama_install_buttons(self, item_id: str) -> None:
        buttons = self._ollama_install_buttons.get(item_id)
        if not buttons:
            return
        install_btn, remove_btn = buttons
        installed = self._ollama_executable() is not None
        install_btn.setVisible(not installed)
        remove_btn.setVisible(installed)

    def _refresh_all_ollama_install_buttons(self) -> None:
        for item_id in tuple(self._ollama_install_buttons):
            self._refresh_ollama_install_buttons(item_id)

    def _manage_ollama_install(self, item_id: str, remove: bool) -> None:
        if remove:
            fallback_url = QUrl("ms-settings:appsfeatures")
            fallback_message = tr(
                "ui_ollama_remove_manual",
                "Відкрийте «Встановлені програми», знайдіть Ollama та виберіть «Видалити».",
            )
        else:
            self._download_ollama_installer(item_id)
            return
        QMessageBox.information(self, tr("ui_ollama_terminal", "Ollama"), fallback_message)
        if not QDesktopServices.openUrl(fallback_url):
            QMessageBox.information(
                self,
                tr("ui_ollama_terminal", "Ollama"),
                f"{fallback_message}\n\n{fallback_url.toString()}",
            )

    def _download_ollama_installer(self, item_id: str) -> None:
        if self._ollama_download_active:
            return
        self._ollama_download_active = True
        self._ollama_download_cancel.clear()
        self._ollama_install_item_id = item_id
        ui = self._ollama_download_widgets.get(item_id, {})
        if isinstance(ui.get("progress"), QProgressBar):
            ui["progress"].setValue(0)
            ui["progress"].setVisible(True)
        if isinstance(ui.get("status"), QLabel):
            ui["status"].setText(tr("ui_ollama_download_start", "Завантаження офіційного інсталятора Ollama…"))
            ui["status"].setVisible(True)
        if isinstance(ui.get("cancel"), QPushButton):
            ui["cancel"].setVisible(True)
        threading.Thread(target=self._ollama_installer_worker, daemon=True, name="OllamaInstallerDownload").start()

    def _ollama_installer_worker(self) -> None:
        installer_path = os.path.join(tempfile.gettempdir(), "OllamaSetup.exe")
        part_path = installer_path + ".part"
        known_total: int = 0
        proxies = None
        custom_proxy = str(config.get("custom_proxy_url", "") or "").strip()
        if custom_proxy:
            proxy_url = custom_proxy if "://" in custom_proxy else f"http://{custom_proxy}"
            proxies = {"http": proxy_url, "https": proxy_url}
        else:
            from core.proxy_helper import get_active_api_proxies
            proxies = get_active_api_proxies()

        try:
            while not self._ollama_download_cancel.is_set():
                offset = os.path.getsize(part_path) if os.path.exists(part_path) else 0
                headers = {"User-Agent": "SilentVoice", "Accept-Encoding": "identity"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                try:
                    with requests.get("https://ollama.com/download/OllamaSetup.exe", headers=headers, proxies=proxies, stream=True, timeout=(15, 30), allow_redirects=True) as response:
                        if offset and response.status_code == 200:
                            offset = 0
                        elif response.status_code not in (200, 206):
                            response.raise_for_status()
                        content_len = int(response.headers.get("Content-Length", "0") or 0)
                        content_range = response.headers.get("Content-Range", "")
                        total = int(content_range.rsplit("/", 1)[-1]) if "/" in content_range else content_len + offset
                        if total > 0:
                            known_total = total
                        mode = "ab" if offset and response.status_code == 206 else "wb"
                        done = offset if mode == "ab" else 0
                        with open(part_path, mode) as out:
                            for chunk in response.iter_content(chunk_size=1024 * 1024):
                                if self._ollama_download_cancel.is_set():
                                    raise InterruptedError(tr("ui_ollama_download_cancelled", "Завантаження скасовано"))
                                if chunk:
                                    out.write(chunk)
                                    done += len(chunk)
                                    if total:
                                        self.ollama_installer_progress.emit(min(99, int(done * 100 / total)))
                    if total and os.path.getsize(part_path) < total:
                        raise IOError(tr("ui_ollama_download_incomplete", "Завантаження перервалося до отримання всього файлу"))
                    os.replace(part_path, installer_path)
                    self.ollama_installer_finished.emit(installer_path, "")
                    return
                except InterruptedError:
                    raise
                except Exception:
                    # Keep the partial file and retry with HTTP Range after the connection returns.
                    partial_size = os.path.getsize(part_path) if os.path.exists(part_path) else 0
                    if known_total > 0:
                        self.ollama_installer_progress.emit(min(99, int(partial_size * 100 / known_total)))
                    for _ in range(10):
                        if self._ollama_download_cancel.wait(1.0):
                            raise InterruptedError(tr("ui_ollama_download_cancelled", "Завантаження скасовано"))
            raise InterruptedError(tr("ui_ollama_download_cancelled", "Завантаження скасовано"))
        except Exception as exc:
            self.ollama_installer_finished.emit("", str(exc))

    def _on_ollama_installer_progress(self, percent: int) -> None:
        ui = self._ollama_download_widgets.get(getattr(self, "_ollama_install_item_id", ""), {})
        try:
            bar = ui.get("progress")
            if isinstance(bar, QProgressBar):
                bar.setValue(percent)
            status = ui.get("status")
            if isinstance(status, QLabel):
                status.setText(tr("ui_ollama_download_progress", "Завантаження інсталятора Ollama… {percent}%", percent=percent))
        except RuntimeError:
            pass

    def _on_ollama_installer_finished(self, path: str, error: str) -> None:
        self._ollama_download_active = False
        ui = self._ollama_download_widgets.get(getattr(self, "_ollama_install_item_id", ""), {})
        try:
            if isinstance(ui.get("progress"), QProgressBar):
                ui["progress"].setVisible(False)
            if isinstance(ui.get("cancel"), QPushButton):
                ui["cancel"].setVisible(False)
            if isinstance(ui.get("status"), QLabel):
                ui["status"].setText(error or tr("ui_ollama_download_done", "Інсталятор завантажено. Запускаємо…"))
                ui["status"].setVisible(True)
        except RuntimeError:
            pass
        if error:
            QMessageBox.warning(self, tr("ui_ollama_install", "Встановлення Ollama"), error)
            return
        try:
            QMessageBox.information(
                self,
                tr("ui_ollama_install", "Встановлення Ollama"),
                tr("ui_ollama_run_installer", "Інсталятор завантажено. Після натискання OK відкриється його вікно для встановлення Ollama."),
            )
            os.startfile(path)
            self._poll_ollama_install_status(getattr(self, "_ollama_install_item_id", ""))
        except Exception as exc:
            QMessageBox.warning(self, tr("ui_ollama_install", "Встановлення Ollama"), str(exc))

    def _poll_ollama_install_status(self, item_id: str, attempts_left: int = 60) -> None:
        self._refresh_ollama_install_buttons(item_id)
        if self._ollama_executable() is not None or attempts_left <= 0:
            return
        QTimer.singleShot(2000, lambda iid=item_id, n=attempts_left - 1: self._poll_ollama_install_status(iid, n))

    def _refresh_ollama_model_button(self, item_id: str) -> None:
        button = self._ollama_model_buttons.get(item_id)
        combo = self._ai_model_combos.get(item_id)
        if button is None or combo is None:
            return
        items = list(config.get("ai_items", []))
        item = next((x for x in items if str(x.get("id")) == item_id), {})
        installed = {
            normalize_ollama_model_name(str(model).strip())
            for model in item.get("available_models", [])
            if str(model).strip()
        }
        selected = self._selected_ai_model(combo)
        connected = self._ollama_connections.get(item_id, False)
        button.setVisible(bool(connected and selected and normalize_ollama_model_name(selected) not in installed))

    @staticmethod
    def _selected_ai_model(combo: QComboBox) -> str:
        index = combo.currentIndex()
        if index >= 0 and combo.currentText() == combo.itemText(index):
            return str(combo.currentData() or combo.currentText()).strip()
        return combo.currentText().strip()

    def _pull_ollama_model(self, item_id: str, model: str) -> None:
        model = str(model).strip()
        if not model:
            return
        if item_id in self._ollama_model_pull_active:
            return
        item = next(
            (x for x in config.get("ai_items", []) if str(x.get("id", "")) == item_id),
            None,
        )
        if not item or str(item.get("provider", "")) != "ollama":
            return
        if not self._ollama_connections.get(item_id, False):
            QMessageBox.information(
                self,
                tr("ui_ollama_pull", "Завантажити модель"),
                tr("ui_ollama_connect_hint", "Спочатку підключіться до Ollama та оновіть список моделей."),
            )
            return
        base_url = str(item.get("base_url", "") or "http://localhost:11434").strip().rstrip("/")
        self._ollama_model_pull_active[item_id] = model
        button = self._ollama_model_buttons.get(item_id)
        if button is not None:
            button.setEnabled(False)
            button.setText(tr("ui_ollama_pull_running", "⏳ Завантаження…"))
        status = self._ai_status_labels.get(item_id)
        if status is not None:
            status.setText(tr("ui_ollama_pull_starting", "Починаємо завантажувати {model}…", model=model))
        threading.Thread(
            target=self._ollama_pull_worker,
            args=(item_id, model, base_url),
            daemon=True,
            name=f"OllamaPull-{item_id}",
        ).start()

    def _ollama_pull_worker(self, item_id: str, model: str, base_url: str) -> None:
        error = ""
        saw_success = False
        try:
            parsed_url = urlparse(base_url)
            executable = self._ollama_executable()
            safe_cli_model = re.fullmatch(r"[A-Za-z0-9._:/+-]+", model) is not None
            if executable and safe_cli_model and parsed_url.hostname in ("localhost", "127.0.0.1", "::1"):
                self.ollama_pull_progress.emit(
                    item_id,
                    model,
                    tr(
                        "ui_ollama_pull_terminal",
                        "Командне вікно відкрито.",
                    ),
                    0,
                    0,
                )
                env = os.environ.copy()
                env["OLLAMA_HOST"] = base_url
                comspec = os.environ.get("COMSPEC", "cmd.exe")
                command = f'"{comspec}" /d /c ""{executable}" pull {model}"'
                process = subprocess.Popen(
                    command,
                    creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
                    close_fds=True,
                    env=env,
                )
                exit_code = process.wait()
                cli_error = ""
                if exit_code != 0:
                    cli_error = tr(
                        "ui_ollama_cli_exit",
                        "Командне вікно Ollama завершилося з кодом {code}.",
                        code=exit_code,
                    )
                else:
                    response = requests.get(f"{base_url}/api/tags", timeout=15)
                    response.raise_for_status()
                    installed_names = {
                        normalize_ollama_model_name(str(entry.get("name", "")))
                        for entry in response.json().get("models", [])
                        if isinstance(entry, dict)
                    }
                    if normalize_ollama_model_name(model) in installed_names:
                        self.ollama_pull_finished.emit(item_id, model, "")
                        return
                    cli_error = tr(
                        "ui_ollama_pull_incomplete",
                        "Сервер завершив передачу без підтвердження успішного встановлення моделі.",
                    )
                # Резервний перехід на вбудований HTTP-потік /api/pull: показує прогрес і точну помилку
                self.ollama_pull_progress.emit(
                    item_id,
                    model,
                    tr("ui_ollama_pull_working", "обробка"),
                    0,
                    0,
                )
            else:
                cli_error = ""

            with requests.post(
                f"{base_url}/api/pull",
                json={"model": model, "stream": True},
                stream=True,
                timeout=(15, None),
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        continue
                    if event.get("error"):
                        raise RuntimeError(str(event["error"]))
                    status = str(event.get("status", ""))
                    completed = int(event.get("completed", 0) or 0)
                    total = int(event.get("total", 0) or 0)
                    self.ollama_pull_progress.emit(item_id, model, status, completed, total)
                    if status.casefold() == "success":
                        saw_success = True
            if not saw_success:
                error = tr(
                    "ui_ollama_pull_incomplete",
                    "Сервер завершив передачу без підтвердження успішного встановлення моделі.",
                )
        except Exception as exc:
            error = str(exc) or locals().get("cli_error", "")
        self.ollama_pull_finished.emit(item_id, model, error)

    def _on_ollama_pull_progress(
        self, item_id: str, model: str, status: str, completed: int, total: int
    ) -> None:
        label = self._ai_status_labels.get(item_id)
        if label is None:
            return
        if total > 0:
            done_mb = completed / (1024 * 1024)
            total_mb = total / (1024 * 1024)
            percent = min(100, round(completed * 100 / total))
            label.setText(
                tr(
                    "ui_ollama_pull_progress",
                    "⬇ {model}: {status} — {percent}% ({done} / {total} МБ поточного файлу)",
                    model=model,
                    status=status or tr("ui_ollama_pull_working", "обробка"),
                    percent=percent,
                    done=f"{done_mb:.0f}",
                    total=f"{total_mb:.0f}",
                )
            )
        else:
            label.setText(
                tr(
                    "ui_ollama_pull_status",
                    "⬇ {model}: {status}",
                    model=model,
                    status=status or tr("ui_ollama_pull_working", "обробка"),
                )
            )

    def _on_ollama_pull_finished(self, item_id: str, model: str, error: str) -> None:
        active_model = self._ollama_model_pull_active.pop(item_id, "")
        button = self._ollama_model_buttons.get(item_id)
        if button is not None:
            button.setEnabled(True)
            button.setText(tr("ui_ollama_pull", "⬇ Завантажити модель"))
        label = self._ai_status_labels.get(item_id)
        if error:
            if label is not None:
                label.setText(tr("ui_ollama_pull_failed", "🔴 Не вдалося встановити {model}: {error}", model=model, error=error))
            self._refresh_ollama_model_button(item_id)
            return
        if label is not None:
            label.setText(tr("ui_ollama_pull_verifying", "Перевіряємо, чи {model} готова до використання…", model=model))
        # Ollama's pull stream sends `success` only after the model is fully available.
        if active_model:
            self._start_fetch_ai_models(item_id, None, silent=True)

    def _open_local_models_dialog(self) -> None:
        dialog = LocalModelsDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        items: List[Dict[str, Any]] = list(config.get("ai_items", []))
        local_item = next(
            (
                item for item in items
                if str(item.get("provider", "")) == "local_api"
                and str(item.get("base_url", "")).rstrip("/") == dialog.server_url.rstrip("/")
            ),
            None,
        )
        if local_item is None:
            spec = AI_PROVIDERS["local_api"]
            item_id = f"ai_local_api_{len(items) + 1}_{id(dialog) % 10000}"
            item_title = f"{tr('ai.prov.local_api.title', spec['title'])} — {dialog.backend_title}"
            local_item = {
                "id": item_id,
                "provider": "local_api",
                "title": item_title,
                "api_key": dialog.api_key,
                "base_url": dialog.server_url,
                "model": dialog.selected_model,
                "available_models": list(dialog.available_models),
                "managed_backend": dialog.managed_backend,
                "model_path": dialog.model_path,
            }
            items.append(local_item)
        else:
            item_id = str(local_item.get("id", ""))
            local_item.update(
                {
                    "api_key": dialog.api_key,
                    "base_url": dialog.server_url,
                    "model": dialog.selected_model,
                    "available_models": list(dialog.available_models),
                    "managed_backend": dialog.managed_backend,
                    "model_path": dialog.model_path,
                }
            )

        updates: Dict[str, Any] = {"ai_items": items}
        if dialog.use_as_primary or not str(config.get("ai_primary_id", "")).strip():
            updates["ai_primary_id"] = item_id
        config.update(updates)
        self._rebuild_ai_cards()
        self._refresh_ai_status_banner()

    def _on_add_ai_provider(self) -> None:
        prov_key = str(self.ai_add_combo.currentData())
        spec = AI_PROVIDERS.get(prov_key)
        if not spec:
            return
        items: List[Dict[str, Any]] = list(config.get("ai_items", []))
        if any(str(item.get("provider", "")).strip() == prov_key for item in items if isinstance(item, dict)):
            self._populate_ai_add_combo()
            return
        same_count = sum(1 for x in items if x.get("provider") == prov_key)
        base_title = (
            tr(f"ai.prov.{prov_key}.title", spec["title"])
            if prov_key in ("ollama", "local_api") else spec["title"]
        )
        title = base_title if same_count == 0 else f"{base_title} #{same_count + 1}"
        new_id = f"ai_{prov_key}_{len(items) + 1}_{id(title) % 10000}"

        new_entry: Dict[str, Any] = {
            "id": new_id,
            "provider": prov_key,
            "title": title,
            "api_key": "",
            "model": spec["default_model"],
        }
        if prov_key == "local_api":
            new_entry["base_url"] = str(spec.get("default_base_url", "http://localhost:1234/v1"))
        items.append(new_entry)
        config.set("ai_items", items)
        self._populate_ai_add_combo()
        if not str(config.get("ai_primary_id", "")).strip():
            config.set("ai_primary_id", new_id)
        self._rebuild_ai_cards()
        self._refresh_ai_status_banner()
        if prov_key in ("ollama", "local_api"):
            self._start_fetch_ai_models(new_id, None, silent=True)

    def _on_delete_ai_item(self, item_id: str) -> None:
        all_items = list(config.get("ai_items", []))
        removed = next((x for x in all_items if str(x.get("id")) == item_id), None)
        if removed is None:
            return
        if QMessageBox.question(
            self,
            tr("common.confirm", "Підтвердження"),
            tr(
                "confirm.delete_service",
                "Точно видалити сервіс «{name}»?",
                name=str(removed.get("title", removed.get("provider", ""))),
            ),
        ) != QMessageBox.StandardButton.Yes:
            return

        old_t = self._ai_debounce_timers.pop(item_id, None)
        if old_t is not None:
            old_t.stop()
            old_t.deleteLater()
        items = [x for x in all_items if str(x.get("id")) != item_id]
        updates: Dict[str, Any] = {"ai_items": items}
        if str(config.get("ai_primary_id", "")) == item_id:
            updates["ai_primary_id"] = str(items[0].get("id", "")) if items else ""
        if str(config.get("ai_fallback_id", "")) == item_id:
            updates["ai_fallback_id"] = ""
        config.update(updates)
        self._populate_ai_add_combo()
        self._rebuild_ai_cards()
        self._refresh_ai_status_banner()

    def _on_ai_item_field_changed(self, item_id: str, field: str, value: Any) -> None:
        items: List[Dict[str, Any]] = list(config.get("ai_items", []))
        for x in items:
            if str(x.get("id")) == item_id:
                x[field] = value
                break
        config.set("ai_items", items)
        self._refresh_ai_test_button(item_id)

    def _auto_sync_all_ai_providers(self) -> None:
        """Автоматично підтягує список моделей для підключених AI-провайдерів."""
        items: List[Dict[str, Any]] = list(config.get("ai_items", []))
        for item in items:
            item_id = str(item.get("id", "")).strip()
            prov = str(item.get("provider", "")).strip()
            if not item_id or not prov:
                continue
            spec = AI_PROVIDERS.get(prov, {})
            if not spec.get("needs_key", True) or str(item.get("api_key", "")).strip():
                self._start_fetch_ai_models(item_id, None, silent=True)

    def _start_fetch_ai_models(self, item_id: str, btn: Optional[QPushButton], silent: bool = False) -> None:
        items: List[Dict[str, Any]] = list(config.get("ai_items", []))
        item = next((x for x in items if str(x.get("id")) == item_id), None)
        if not item:
            return
        prov = str(item.get("provider", "groq")).strip()
        spec = AI_PROVIDERS.get(prov, AI_PROVIDERS["groq"])
        api_key = str(item.get("api_key", "")).strip()
        base_url = str(item.get("base_url", "")).strip()
        if spec.get("needs_key", True) and not api_key:
            if not silent:
                QMessageBox.warning(self, tr("ui_229", "Пошук AI моделей"), tr("ui_192", "Спочатку введіть API ключ."))
            return

        if isinstance(btn, QPushButton):
            try:
                btn.setEnabled(False)
                btn.setText(tr("ui_176", "⏳ Пошук..."))
            except RuntimeError:
                pass
        st_lbl = self._ai_status_labels.get(item_id)
        if st_lbl is not None:
            try:
                st_lbl.setText(tr("ui_230", "⏳ Завантаження списку моделей через API..."))
            except RuntimeError:
                pass

        def _worker() -> None:
            try:
                models = fetch_ai_provider_models(prov, api_key, base_url)
                self.ai_models_fetched.emit(item_id, models, "", btn, silent)
            except Exception as exc:
                self.ai_models_fetched.emit(item_id, [], str(exc), btn, silent)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_ai_models_fetched(
        self, item_id: str, models: object, err: str, btn: object, silent: bool
    ) -> None:
        if isinstance(btn, QPushButton):
            try:
                btn.setEnabled(True)
                btn.setText(tr("ui_224", "🔄 Оновити моделі"))
            except RuntimeError:
                pass

        st_lbl = self._ai_status_labels.get(item_id)
        if err:
            if str(next((x.get("provider") for x in config.get("ai_items", []) if str(x.get("id")) == item_id), "")) == "ollama":
                self._ollama_connections[item_id] = False
                items = list(config.get("ai_items", []))
                for item in items:
                    if str(item.get("id")) == item_id:
                        item["available_models"] = []
                        break
                config.set("ai_items", items)
                combo = self._ai_model_combos.get(item_id)
                if combo is not None:
                    combo.blockSignals(True)
                    combo.clear()
                    combo.blockSignals(False)
                self._refresh_ollama_model_button(item_id)
                self._refresh_ai_test_button(item_id)
                if st_lbl is not None:
                    st_lbl.setText(tr("ui_ollama_disconnected", "⚪ Ollama не підключена — список моделей з’явиться після підключення"))
                if not silent:
                    QMessageBox.information(self, tr("ui_229", "Пошук AI моделей"), tr("ui_ollama_connect_hint", "Не вдалося підключитися до Ollama. Переконайтеся, що вона встановлена й запущена."))
                return
            if st_lbl is not None:
                try:
                    st_lbl.setText(tr("ui_f_008", "🔴 Помилка API: {err}", err=err[:95]))
                except RuntimeError:
                    pass
            if not silent:
                QMessageBox.warning(self, tr("ui_229", "Пошук AI моделей"), tr("ui_f_015", "Не вдалося завантажити моделі:\n{err}", err=err))
            return
        model_list = [str(m).strip() for m in (models if isinstance(models, list) else []) if str(m).strip()]
        active_pull = self._ollama_model_pull_active.get(item_id)
        if active_pull:
            active_key = normalize_ollama_model_name(active_pull)
            model_list = [m for m in model_list if normalize_ollama_model_name(m) != active_key]
        items: List[Dict[str, Any]] = list(config.get("ai_items", []))
        saved_model = ""
        default_model = ""
        for x in items:
            if str(x.get("id")) == item_id:
                x["available_models"] = model_list
                prov = str(x.get("provider", "groq"))
                if prov == "ollama":
                    self._ollama_connections[item_id] = True
                default_model = str(AI_PROVIDERS.get(prov, AI_PROVIDERS["groq"])["default_model"])
                saved_model = str(x.get("model", "")).strip() or default_model
                ollama_choice_keys = {
                    normalize_ollama_model_name(name)
                    for name in (*OLLAMA_RECOMMENDED_MODELS, *model_list)
                }
                if prov == "ollama" and normalize_ollama_model_name(saved_model) not in ollama_choice_keys:
                    saved_model = OLLAMA_RECOMMENDED_MODELS[0]
                    x["model"] = saved_model
                if not str(x.get("model", "")).strip() and model_list:
                    x["model"] = model_list[0]
                    saved_model = model_list[0]
                break
        config.set("ai_items", items)

        combo = self._ai_model_combos.get(item_id)
        if combo is not None:
            try:
                cur_txt = self._selected_ai_model(combo) or saved_model
                choices: List[str] = []
                provider = str(next((x.get("provider", "") for x in items if str(x.get("id")) == item_id), ""))
                candidates = ([*([cur_txt] if any(normalize_ollama_model_name(cur_txt) == normalize_ollama_model_name(name) for name in (*OLLAMA_RECOMMENDED_MODELS, *model_list)) else []), *OLLAMA_RECOMMENDED_MODELS, *model_list] if provider == "ollama"
                              else [cur_txt, default_model, *model_list])
                for cand in candidates:
                    if cand and (
                        normalize_ollama_model_name(cand) not in {normalize_ollama_model_name(choice) for choice in choices}
                        if provider == "ollama"
                        else cand not in choices
                    ):
                        choices.append(cand)
                combo.blockSignals(True)
                combo.clear()
                if provider == "ollama":
                    installed_model_keys = {normalize_ollama_model_name(name) for name in model_list}
                    installed_label = tr("ui_ollama_model_installed", "встановлено")
                    for model_name in choices:
                        model_key = normalize_ollama_model_name(model_name)
                        label = f"{model_name} ({installed_label})" if model_key in installed_model_keys else model_name
                        combo.addItem(label, model_name)
                    selected_model = next(
                        (choice for choice in choices if normalize_ollama_model_name(choice) == normalize_ollama_model_name(cur_txt)),
                        choices[0],
                    )
                    combo.setCurrentIndex(combo.findData(selected_model))
                else:
                    combo.addItems(choices)
                    if cur_txt:
                        combo.setCurrentText(cur_txt if cur_txt in choices else choices[0])
                combo.blockSignals(False)
                if provider == "ollama":
                    self._refresh_ollama_model_button(item_id)
                self._refresh_ai_test_button(item_id)
            except RuntimeError:
                pass

        if st_lbl is not None:
            try:
                if self._ollama_connections.get(item_id, False):
                    family_count = len({ollama_model_family_key(model) for model in model_list if ollama_model_family_key(model)})
                    st_lbl.setText(tr("ui_ollama_connected", "🟢 Ollama підключена • встановлено сімейств моделей: {count}", count=family_count))
                else:
                    st_lbl.setText(tr("ui_f_016", "🟢 Завантажено моделей: {count}", count=len(model_list)))
            except RuntimeError:
                pass

        if not silent and not model_list:
            QMessageBox.information(self, tr("ui_229", "Пошук AI моделей"), tr("ui_231", "Список моделей порожній."))

    def _on_all_prim_eng_changed(self) -> None:
        eng_id = str(self.all_prim_eng_combo.currentData() or "")
        voices = tts_manager.get_voices_for_engine(eng_id)
        v_id = voices[0]["id"] if voices else ""
        self._select_engine_and_voice(eng_id, v_id)

    def _on_all_prim_voice_changed(self) -> None:
        v_id = str(self.all_prim_voice_combo.currentData() or "")
        if v_id:
            config.set("active_voice", v_id)
            if hasattr(self, "tts_voice_combo"):
                self.tts_voice_combo.blockSignals(True)
                idx = self.tts_voice_combo.findData(v_id)
                if idx >= 0:
                    self.tts_voice_combo.setCurrentIndex(idx)
                self.tts_voice_combo.blockSignals(False)
            self._refresh_status_banners()
            self._rebuild_dsp_settings_cards()
            tts_manager.schedule_smart_warmup(0.3, "voice_changed")
            cur_eng = str(config.get("active_engine", "")).strip()
            QTimer.singleShot(0, lambda e=cur_eng, v=v_id: self._maybe_prompt_ffmpeg_install(e, v, from_voice_switch=True))

    def _on_all_fb_eng_changed(self) -> None:
        fb_eng = str(self.all_fb_eng_combo.currentData() or "")
        voices = tts_manager.get_voices_for_engine(fb_eng) if fb_eng else []
        fb_voice = voices[0]["id"] if voices else ""
        config.update({"fallback_engine": fb_eng, "fallback_voice": fb_voice})
        self._sync_tts_page_selectors()
        if fb_eng:
            tts_manager.schedule_smart_warmup(0.5, "fallback_changed")
            QTimer.singleShot(0, lambda e=fb_eng, v=fb_voice: self._maybe_prompt_ffmpeg_install(e, v, from_voice_switch=True))

    def _on_all_fb_voice_changed(self) -> None:
        fb_v = str(self.all_fb_voice_combo.currentData() or "")
        config.set("fallback_voice", fb_v)
        self._rebuild_dsp_settings_cards()
        if fb_v:
            tts_manager.schedule_smart_warmup(0.8, "fallback_voice_changed")
            fb_eng = str(config.get("fallback_engine", "")).strip()
            QTimer.singleShot(0, lambda e=fb_eng, v=fb_v: self._maybe_prompt_ffmpeg_install(e, v, from_voice_switch=True))

    def _rebuild_dsp_settings_cards(self) -> None:
        if not hasattr(self, "dsp_cards_layout"):
            return
        self._primary_dsp_speed_slider = None
        self._primary_dsp_speed_label = None
        self._ffmpeg_ui_bars = [b for b in self._ffmpeg_ui_bars if b.get("scope") != "dsp"]
        while self.dsp_cards_layout.count():
            item = self.dsp_cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        available = tts_manager.get_available_active_engines()
        title_map = {a["id"]: a["title"] for a in available}

        prim_eng = str(config.get("active_engine", "")).strip()
        prim_voice = str(config.get("active_voice", "")).strip()
        fb_eng = str(config.get("fallback_engine", "")).strip()
        fb_voice = str(config.get("fallback_voice", "")).strip()

        if prim_eng and prim_eng in title_map:
            prim_title = (
                tr("ui.dsp.title_primary_custom", "🎛️ Налаштування звучання (DSP custom) — Головний TTS: {title}", title=title_map[prim_eng])
                if prim_voice.endswith("::custom")
                else tr("ui_f_017", "🎛️ Налаштування звучання (DSP) — Головний TTS: {title}", title=title_map[prim_eng])
            )
            self.dsp_cards_layout.addWidget(
                self._create_engine_dsp_card(
                    prim_eng,
                    prim_title,
                    voice_id=prim_voice,
                )
            )
        if fb_eng and fb_eng in title_map:
            # Якщо запасний TTS збігається з головним, використовуємо окремий ключ налаштувань або той самий
            fb_settings_key = f"fallback::{fb_eng}" if fb_eng == prim_eng else fb_eng
            fb_title = (
                tr("ui.dsp.title_fallback_custom", "🛡️ Налаштування звучання (DSP custom) — Запасний TTS: {title}", title=title_map[fb_eng])
                if fb_voice.endswith("::custom")
                else tr("ui_f_018", "🛡️ Налаштування звучання (DSP) — Запасний TTS: {title}", title=title_map[fb_eng])
            )
            self.dsp_cards_layout.addWidget(
                self._create_engine_dsp_card(
                    fb_settings_key,
                    fb_title,
                    voice_id=fb_voice,
                )
            )

    def _create_engine_dsp_card(self, engine_key: str, card_title: str, voice_id: str = "") -> QFrame:
        raw_eng_id = engine_key.split("::", 1)[-1]
        dsp = get_engine_dsp_settings(raw_eng_id, voice_id)

        card = QFrame()
        card.setProperty("class", "card")
        c_lay = QVBoxLayout(card)
        c_lay.setContentsMargins(16, 12, 16, 12)
        c_lay.setSpacing(10)

        hdr_row = QHBoxLayout()
        hdr_lbl = QLabel(card_title)
        hdr_lbl.setProperty("class", "cardHeader")
        hdr_row.addWidget(hdr_lbl)
        hdr_row.addWidget(
            HelpButton(
                tr("ui_232", "Пояснення налаштувань звучання простими словами"),
                tr(
                    "ui_233",
                    "<b>Коли працює?</b><br>"
                    "Обробка вмикається сама, щойно змінено хоча б один параметр нижче.<br><br>"
                    "• <b>Швидкість</b> — робить вимову швидшою (понад 1.00x) або повільнішою (нижче 1.00x) без спотворення тону.<br>"
                    "• <b>Тон (Вище / Нижче)</b> — у плюс (+) голос ніжніший і вищий, у мінус (-) — нижчий і глибший.<br>"
                    "• <b>Шумозаглушення</b> — прибирає металевий призвук і легке шипіння синтезатора.<br>"
                    "• <b>Пом'якшення «С / Ш / Щ»</b> — згладжує занадто різкі свистячі звуки.<br>"
                    "• <b>Студійний нойз-гейт</b> — прибирає фоновий гул у паузах між словами.<br><br>"
                    "<i>Якщо все звучить добре й так — нічого змінювати не потрібно.</i>",
                ),
            )
        )
        hdr_row.addStretch(1)

        reset_btn = QPushButton(tr("ui_234", "↺ Скинути"))
        reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        hdr_row.addWidget(reset_btn)
        c_lay.addLayout(hdr_row)

        vol_row = QHBoxLayout()
        vol_row.setSpacing(14)
        vol_lbl_title = QLabel(tr("tts.dsp.volume_label", "Гучність TTS"))
        vol_lbl_title.setStyleSheet("color: #E2E8F0; font-size: 13px; font-weight: 600;")
        sl_volume = QSlider(Qt.Orientation.Horizontal)
        sl_volume.setRange(0, 200)
        sl_volume.setValue(int(round(float(dsp.get("volume", 100)))))
        lbl_volume = QLabel(f"{sl_volume.value()}%")
        lbl_volume.setFixedWidth(64)
        lbl_volume.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        vol_row.addWidget(vol_lbl_title)
        vol_row.addWidget(sl_volume, 1)
        vol_row.addWidget(lbl_volume)
        c_lay.addLayout(vol_row)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)

        # 1. Швидкість мовлення (0.50x .. 2.00x)
        chk_speed = QCheckBox(tr("ui_241", "Швидкість мовлення"))
        chk_speed.setChecked(bool(dsp.get("speed_enabled", False)))
        sl_speed = QSlider(Qt.Orientation.Horizontal)
        sl_speed.setRange(50, 200)
        sl_speed.setValue(int(round(float(dsp.get("speed_rate", 1.00)) * 100)))
        lbl_speed = QLabel(f"{sl_speed.value() / 100.0:.2f}x")
        lbl_speed.setFixedWidth(64)
        lbl_speed.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(chk_speed, 0, 0)
        grid.addWidget(sl_speed, 0, 1)
        grid.addWidget(lbl_speed, 0, 2)

        # 2. Тон голосу (Pitch: -6.0 st .. +6.0 st)
        chk_pitch = QCheckBox(tr("ui_243", "Тон голосу — вище / нижче (Pitch)"))
        chk_pitch.setChecked(bool(dsp.get("pitch_enabled", False)))
        sl_pitch = QSlider(Qt.Orientation.Horizontal)
        sl_pitch.setRange(-60, 60)
        sl_pitch.setValue(int(round(float(dsp.get("pitch_steps", 0.0)) * 10)))
        lbl_pitch = QLabel(f"{sl_pitch.value() / 10.0:+.1f} st")
        lbl_pitch.setFixedWidth(64)
        lbl_pitch.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(chk_pitch, 1, 0)
        grid.addWidget(sl_pitch, 1, 1)
        grid.addWidget(lbl_pitch, 1, 2)

        # 3. Шумозаглушення (0% .. 100%)
        chk_denoise = QCheckBox(tr("ui_245", "Очищення шуму (Шумозаглушення)"))
        chk_denoise.setChecked(bool(dsp.get("denoise_enabled", False)))
        sl_denoise = QSlider(Qt.Orientation.Horizontal)
        sl_denoise.setRange(0, 100)
        sl_denoise.setValue(int(round(float(dsp.get("denoise_prop", 0.0)) * 100)))
        lbl_denoise = QLabel(f"{sl_denoise.value()}%")
        lbl_denoise.setFixedWidth(64)
        lbl_denoise.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(chk_denoise, 2, 0)
        grid.addWidget(sl_denoise, 2, 1)
        grid.addWidget(lbl_denoise, 2, 2)

        # 4. Пом'якшення шиплячих (-20.0 dB .. 0.0 dB)
        chk_deess = QCheckBox(tr("ui_247", "Пом'якшення шиплячих («С», «Ш»)"))
        chk_deess.setChecked(bool(dsp.get("deess_enabled", False)))
        sl_deess = QSlider(Qt.Orientation.Horizontal)
        sl_deess.setRange(-200, 0)
        sl_deess.setValue(int(round(float(dsp.get("deess_db", -6.0)) * 10)))
        lbl_deess = QLabel(f"{sl_deess.value() / 10.0:.1f} dB")
        lbl_deess.setFixedWidth(64)
        lbl_deess.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(chk_deess, 3, 0)
        grid.addWidget(sl_deess, 3, 1)
        grid.addWidget(lbl_deess, 3, 2)

        # 5. Студійний нойз-гейт (-80 dB .. -20 dB)
        chk_gate = QCheckBox(tr("ui_249", "Прибирати шум у паузах (Нойз-гейт)"))
        chk_gate.setChecked(bool(dsp.get("gate_enabled", False)))
        sl_gate = QSlider(Qt.Orientation.Horizontal)
        sl_gate.setRange(-80, -20)
        sl_gate.setValue(int(round(float(dsp.get("gate_db", -45.0)))))
        lbl_gate = QLabel(f"{sl_gate.value()} dB")
        lbl_gate.setFixedWidth(64)
        lbl_gate.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(chk_gate, 4, 0)
        grid.addWidget(sl_gate, 4, 1)
        grid.addWidget(lbl_gate, 4, 2)

        grid.setColumnStretch(1, 1)
        c_lay.addLayout(grid)

        ffmpeg_notice = self._create_ffmpeg_notice_widget(scope="dsp", show_when_installed=False)
        if ffmpeg_notice is not None:
            c_lay.addWidget(ffmpeg_notice)

        def _save_current_dsp() -> None:
            spd_val = round(sl_speed.value() / 100.0, 2)
            lbl_volume.setText(f"{sl_volume.value()}%")
            lbl_speed.setText(f"{spd_val:.2f}x")
            lbl_pitch.setText(f"{sl_pitch.value() / 10.0:+.1f} st")
            lbl_denoise.setText(f"{sl_denoise.value()}%")
            lbl_deess.setText(f"{sl_deess.value() / 10.0:.1f} dB")
            lbl_gate.setText(f"{sl_gate.value()} dB")
            any_fx = (
                chk_speed.isChecked()
                or chk_pitch.isChecked()
                or chk_denoise.isChecked()
                or chk_deess.isChecked()
                or chk_gate.isChecked()
            )
            save_engine_dsp_settings(
                raw_eng_id,
                {
                    "volume": sl_volume.value(),
                    "enabled": any_fx,
                    "speed_enabled": chk_speed.isChecked(),
                    "speed_rate": spd_val,
                    "pitch_enabled": chk_pitch.isChecked(),
                    "pitch_steps": round(sl_pitch.value() / 10.0, 1),
                    "denoise_enabled": chk_denoise.isChecked(),
                    "denoise_prop": round(sl_denoise.value() / 100.0, 2),
                    "deess_enabled": chk_deess.isChecked(),
                    "deess_db": round(sl_deess.value() / 10.0, 1),
                    "gate_enabled": chk_gate.isChecked(),
                    "gate_db": float(sl_gate.value()),
                },
                voice_id,
            )

        def _reset_current_dsp() -> None:
            _, default_dsp = _resolve_dsp_key_and_defaults(raw_eng_id, voice_id)
            save_engine_dsp_settings(raw_eng_id, dict(default_dsp), voice_id)
            self._rebuild_dsp_settings_cards()

        def _on_speed_or_pitch_toggled(checked: bool) -> None:
            _save_current_dsp()
            if checked:
                QTimer.singleShot(
                    0,
                    lambda: self._maybe_prompt_ffmpeg_install(raw_eng_id, voice_id, from_voice_switch=False),
                )

        sl_volume.valueChanged.connect(lambda _: _save_current_dsp())
        chk_speed.toggled.connect(_on_speed_or_pitch_toggled)
        chk_pitch.toggled.connect(_on_speed_or_pitch_toggled)
        chk_denoise.toggled.connect(lambda _: _save_current_dsp())
        chk_deess.toggled.connect(lambda _: _save_current_dsp())
        chk_gate.toggled.connect(lambda _: _save_current_dsp())

        sl_speed.valueChanged.connect(lambda _: _save_current_dsp())
        sl_pitch.valueChanged.connect(lambda _: _save_current_dsp())
        sl_denoise.valueChanged.connect(lambda _: _save_current_dsp())
        sl_deess.valueChanged.connect(lambda _: _save_current_dsp())
        sl_gate.valueChanged.connect(lambda _: _save_current_dsp())
        reset_btn.clicked.connect(_reset_current_dsp)

        return card

    # ──────────────────────────────────────────────────────────────────
    # РОЗДІЛ 2: Швидкі фрази (натиснув — і програма сказала)
    # ──────────────────────────────────────────────────────────────────

    def _build_page_phrases(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        content.setObjectName("scrollContent")
        lay = QVBoxLayout(content)
        lay.setContentsMargins(2, 2, 6, 6)
        lay.setSpacing(12)

        lay.addWidget(
            make_header_with_help(
                tr("ui_251", "💬 Швидкі фрази в 1 клік"),
                tr("ui_252", "Для чого потрібні швидкі фрази?"),
                tr(
                    "ui_253",
                    "<b>Миттєві відповіді в 1 клік:</b><br>"
                    "Зручно для фраз, які ви часто використовуєте («Привіт», «Секунду», «Дякую»).<br><br>"
                    "• <b>Мишкою:</b> натисніть картку фрази — програма одразу скаже її вголос.<br>"
                    "• <b>Клавішами:</b> натисніть призначену комбінацію (напр. <b>Alt+1</b>), навіть коли вікно згорнуте.<br>"
                    "• <b>Керування:</b> <b>«+ Додати фразу»</b> створює нову кнопку, а <b>«✎»</b> редагує наявну.",
                ),
            )
        )

        # Картка сітки швидких фраз
        qp_card = QFrame()
        qp_card.setProperty("class", "card")
        qp_lay = QVBoxLayout(qp_card)
        qp_lay.setContentsMargins(14, 12, 14, 12)
        qp_lay.setSpacing(10)

        qp_hdr_row = QHBoxLayout()
        qp_title = QLabel(tr("ui_254", "Натисніть на фразу, щоб озвучити її:"))
        qp_title.setProperty("class", "cardHeader")
        qp_hdr_row.addWidget(qp_title)
        qp_hdr_row.addStretch(1)

        reset_qp_btn = QPushButton(tr("ui_255", "↺ Відновити стандартні"))
        reset_qp_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reset_qp_btn.setToolTip(tr("ui_256", "Повернути стандартний набір швидких фраз"))
        reset_qp_btn.clicked.connect(self._on_reset_quick_phrases)
        qp_hdr_row.addWidget(reset_qp_btn)

        # Кнопка відкриває діалог із двома кнопками («Додати файлом» та «Записати»)
        add_qp_btn = QPushButton(tr("ui_257", "🎙️ Додати / Записати фразу"))
        add_qp_btn.setProperty("class", "primaryBtn")
        add_qp_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_qp_btn.setToolTip(
            tr("ui_258", "Створити свою фразу: записати озвучку тексту у файл (щоб не переозвучувати щоразу) або додати звуковим файлом")
        )
        add_qp_btn.clicked.connect(lambda: self._on_add_quick_phrase(auto_pick_file=False))
        qp_hdr_row.addWidget(add_qp_btn)
        qp_lay.addLayout(qp_hdr_row)

        self.quick_phrases_scroll = _QuickPhraseScrollArea()
        self.quick_phrases_scroll.setWidgetResizable(True)
        self.quick_phrases_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.quick_phrases_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.quick_phrases_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        qp_container = QWidget()
        qp_container.setObjectName("scrollContent")
        self._qp_container_lay = QVBoxLayout(qp_container)
        self._qp_container_lay.setContentsMargins(0, 0, 0, 0)
        self._qp_container_lay.setSpacing(0)

        self.quick_phrases_grid = QGridLayout()
        self.quick_phrases_grid.setContentsMargins(0, 0, 0, 0)
        self.quick_phrases_grid.setSpacing(8)
        for col in range(3):
            self.quick_phrases_grid.setColumnStretch(col, 1)
        self._qp_container_lay.addLayout(self.quick_phrases_grid)
        self._qp_container_lay.addStretch(1)
        self.quick_phrases_scroll.setWidget(qp_container)
        qp_lay.addWidget(self.quick_phrases_scroll)
        lay.addWidget(qp_card)

        # Згортуваний блок звукових команд та реакцій (/ \ |) у стилі блоку VBAN
        from ui.reaction_commands import ReactionCommandsCard

        self.reaction_commands_card = ReactionCommandsCard(keyboard_hook=self.keyboard_hook, combo_cls=VoiceComboBox)
        self.reaction_commands_card.commands_changed.connect(self._on_reaction_commands_changed)
        self.reaction_commands_card.style_changed_preview.connect(self._on_reaction_style_preview)
        lay.addWidget(self.reaction_commands_card)

        # Додатковий блок: швидке озвучення довільного тексту та історія
        bottom_card = QFrame()
        bottom_card.setProperty("class", "card")
        b_lay = QVBoxLayout(bottom_card)
        b_lay.setContentsMargins(14, 12, 14, 12)
        b_lay.setSpacing(8)

        b_hdr = QHBoxLayout()
        b_lbl = QLabel(tr("ui_259", "✍️ Написати та озвучити текст прямо з вікна:"))
        b_lbl.setProperty("class", "cardHeader")
        b_hdr.addWidget(b_lbl)
        b_hdr.addStretch(1)
        clear_hist_btn = QPushButton(tr("ui_260", "Очистити історію"))
        clear_hist_btn.clicked.connect(self._on_clear_history)
        b_hdr.addWidget(clear_hist_btn)
        b_lay.addLayout(b_hdr)

        input_row = QHBoxLayout()
        self.quick_speak_input = QLineEdit()
        self.quick_speak_input.setPlaceholderText(tr("ui_262", "Введіть будь-яке речення та натисніть Enter..."))
        self.quick_speak_input.returnPressed.connect(self._on_home_speak_clicked)
        input_row.addWidget(self.quick_speak_input, 1)

        speak_btn = QPushButton(tr("ui_263", "🔊 Сказати (Enter)"))
        speak_btn.setProperty("class", "primaryBtn")
        speak_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        speak_btn.clicked.connect(self._on_home_speak_clicked)
        input_row.addWidget(speak_btn)

        repeat_btn = QPushButton(tr("ui_265", "🔁 Повторити останнє"))
        repeat_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        repeat_btn.clicked.connect(tts_manager.repeat_last)
        input_row.addWidget(repeat_btn)
        b_lay.addLayout(input_row)

        self.history_list = QListWidget()
        self.history_list.setMinimumHeight(200)
        self.history_list.setMaximumHeight(216)
        self.history_list.setStyleSheet(
            "QListWidget { background-color: #1B1F2D; border: 1px solid #38415E; border-radius: 8px; padding: 4px; }"
            "QListWidget::item { padding: 2px 4px; border-radius: 5px; color: #E2E8F0; }"
            "QListWidget::item:hover { background-color: #2E364F; }"
        )
        self.history_list.itemDoubleClicked.connect(
            lambda item: tts_manager.speak(str(item.data(Qt.ItemDataRole.UserRole) or item.text() or ""))
        )
        b_lay.addWidget(self.history_list)
        lay.addWidget(bottom_card)

        lay.addStretch(1)
        scroll.setWidget(content)

        self._rebuild_quick_phrases_grid()
        self._refresh_history_list()
        return scroll

    def _on_home_speak_clicked(self) -> None:
        text = self.quick_speak_input.text().strip()
        if not text:
            return
        self.quick_speak_input.clear()
        tts_manager.speak(text)
        self._refresh_history_list()

    def _rebuild_quick_phrases_grid(self) -> None:
        while self.quick_phrases_grid.count():
            item = self.quick_phrases_grid.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

        phrases: List[Dict[str, str]] = config.get("quick_phrases", [])
        cols = 3
        max_visible_rows = 9
        max_cell_h = 0
        for idx, qp in enumerate(phrases):
            r, c = divmod(idx, cols)
            cell = QFrame()
            cell.setMinimumWidth(0)
            cell.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            cell.setStyleSheet(
                "QFrame { background-color: #2E354B; border: 1px solid #414B6B; border-radius: 10px; }"
                "QFrame:hover { border-color: #7C5CFF; background-color: #363F59; }"
            )
            c_lay = QHBoxLayout(cell)
            c_lay.setContentsMargins(8, 7, 6, 7)
            c_lay.setSpacing(4)

            emoji = qp.get("emoji", "💬")
            title = qp.get("title", tr("ui_268", "Фраза"))
            text = qp.get("text", "")
            hk = qp.get("hotkey", "")

            hk_badge = f" [{hk}]" if hk else ""
            has_audio = bool(qp.get("audio_file"))
            rec_badge = " 🎵" if has_audio else ""
            line1 = f"{emoji} {title}{hk_badge}{rec_badge}"
            play_btn = _QuickPhraseButton(line1, text)
            play_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            play_btn.setToolTip(f"{line1}\n{text}" if text else line1)
            play_btn.setStyleSheet(
                "QPushButton { background: transparent; border: none; text-align: left; color: #FFFFFF; font-size: 12px; font-weight: 600; padding: 2px; }"
            )
            play_btn.clicked.connect(lambda checked=False, item_qp=qp: self._trigger_quick_phrase(item_qp))
            c_lay.addWidget(play_btn, 1)

            edit_btn = QPushButton()
            edit_btn.setFixedSize(24, 24)
            edit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            # In a frozen onedir build, assets are stored under PyInstaller's
            # bundle directory (_internal), not beside the executable.
            edit_btn.setIcon(QIcon(str(BUNDLE_DIR / "assets" / "edit.svg")))
            edit_btn.setIconSize(QSize(14, 14))
            edit_btn.setStyleSheet("padding: 0px; qproperty-iconSize: 14px 14px;")
            edit_btn.clicked.connect(lambda checked=False, i=idx: self._on_edit_quick_phrase(i))
            c_lay.addWidget(edit_btn)

            del_btn = QPushButton()
            del_btn.setFixedSize(24, 24)
            del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            del_btn.setIcon(QIcon(str(BUNDLE_DIR / "assets" / "delete.svg")))
            del_btn.setIconSize(QSize(15, 15))
            del_btn.setContentsMargins(0, 0, 0, 0)
            del_btn.setStyleSheet("QPushButton { padding: 0px; text-align: center; }")
            del_btn.clicked.connect(lambda checked=False, i=idx: self._on_delete_quick_phrase(i))
            c_lay.addWidget(del_btn, 0, Qt.AlignmentFlag.AlignCenter)

            self.quick_phrases_grid.addWidget(cell, r, c)
            play_btn.ensurePolished()
            cell.ensurePolished()
            max_cell_h = max(max_cell_h, cell.sizeHint().height())

        rows = (len(phrases) + cols - 1) // cols
        if rows <= 0:
            self.quick_phrases_scroll.setVisible(False)
            self.quick_phrases_scroll.setFixedHeight(0)
        else:
            needs_scroll = rows > max_visible_rows
            visible_rows = min(rows, max_visible_rows)
            spacing = max(0, self.quick_phrases_grid.verticalSpacing())
            target_h = visible_rows * max_cell_h + max(0, visible_rows - 1) * spacing + 2
            self._qp_container_lay.setContentsMargins(0, 0, 6 if needs_scroll else 0, 0)
            self.quick_phrases_scroll.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded if needs_scroll else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self.quick_phrases_scroll.setFixedHeight(target_h)
            self.quick_phrases_scroll.setVisible(True)

    def _trigger_quick_phrase(self, qp: Dict[str, Any]) -> None:
        tts_manager.speak_quick_phrase(qp)
        self._refresh_history_list()

    def _on_add_quick_phrase(self, auto_pick_file: bool = False) -> None:
        dlg = QuickPhraseDialog(self.keyboard_hook, None, self, auto_pick_file=auto_pick_file)
        if dlg.exec() == QuickPhraseDialog.DialogCode.Accepted:
            phrases = list(config.get("quick_phrases", []))
            phrases.append(dlg.get_result())
            config.set("quick_phrases", phrases)
            self.keyboard_hook.update_from_config(config.data)
            self._rebuild_quick_phrases_grid()
            self._refresh_history_list()

    def _on_save_history_to_quick_phrases(self, phrase_text: str) -> None:
        clean = str(phrase_text or "").strip()
        if not clean:
            return
        title_guess = clean if len(clean) <= 22 else (clean[:21].rstrip() + "…")
        init_data = {
            "title": title_guess,
            "text": clean,
            "hotkey": "",
            "emoji": "⭐",
            "audio_file": "",
        }
        dlg = QuickPhraseDialog(self.keyboard_hook, init_data, self)
        if dlg.exec() == QuickPhraseDialog.DialogCode.Accepted:
            phrases = list(config.get("quick_phrases", []))
            phrases.append(dlg.get_result())
            config.set("quick_phrases", phrases)
            self.keyboard_hook.update_from_config(config.data)
            self._rebuild_quick_phrases_grid()
            self._refresh_history_list()

    def _on_edit_quick_phrase(self, index: int) -> None:
        phrases = list(config.get("quick_phrases", []))
        if not (0 <= index < len(phrases)):
            return
        dlg = QuickPhraseDialog(self.keyboard_hook, phrases[index], self)
        if dlg.exec() == QuickPhraseDialog.DialogCode.Accepted:
            phrases[index] = dlg.get_result()
            config.set("quick_phrases", phrases)
            self.keyboard_hook.update_from_config(config.data)
            self._rebuild_quick_phrases_grid()
            self._refresh_history_list()

    def _on_delete_quick_phrase(self, index: int) -> None:
        phrases = list(config.get("quick_phrases", []))
        if 0 <= index < len(phrases):
            if QMessageBox.question(self, tr("common.confirm", "Підтвердження"), tr("confirm.delete_phrase", "Точно видалити цю швидку фразу?")) != QMessageBox.StandardButton.Yes:
                return
            phrases.pop(index)
            config.set("quick_phrases", phrases)
            self.keyboard_hook.update_from_config(config.data)
            self._rebuild_quick_phrases_grid()
            self._refresh_history_list()

    def _on_reset_quick_phrases(self) -> None:
        if QMessageBox.question(self, tr("common.confirm", "Підтвердження"), tr("confirm.reset_phrases", "Відновити стандартний набір швидких фраз?")) != QMessageBox.StandardButton.Yes:
            return
        from core.config import get_default_quick_phrases_for_lang
        cur_lang = str(config.get("app_language", "en")).strip().lower() or "en"
        config.set("quick_phrases", get_default_quick_phrases_for_lang(cur_lang))
        self.keyboard_hook.update_from_config(config.data)
        self._rebuild_quick_phrases_grid()
        self._refresh_history_list()

    def _refresh_history_list(self) -> None:
        if not hasattr(self, "history_list"):
            return
        self.history_list.clear()
        saved_texts = {
            str(qp.get("text", "") or "").strip().lower()
            for qp in (config.get("quick_phrases", []) or [])
            if isinstance(qp, dict)
        }
        for phrase in config.get("recent_phrases", []):
            text_val = str(phrase or "").strip()
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, text_val)

            row_w = QWidget()
            row_w.setStyleSheet("background: transparent;")
            row_lay = QHBoxLayout(row_w)
            row_lay.setContentsMargins(6, 2, 4, 2)
            row_lay.setSpacing(8)

            lbl = QLabel(text_val)
            lbl.setMinimumWidth(0)
            lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            lbl.setStyleSheet("background: transparent; border: none; color: #E2E8F0; font-size: 12px;")
            lbl.setToolTip(text_val)
            row_lay.addWidget(lbl, 1)

            already_saved = bool(text_val) and text_val.lower() in saved_texts
            btn_label = (
                tr("ui.history.saved_btn", "★ У швидких")
                if already_saved
                else tr("ui.history.save_btn", "⭐ В обране")
            )
            fav_btn = QPushButton(btn_label)
            fav_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            fav_btn.setFixedHeight(22)
            fav_btn.setToolTip(
                tr("ui.history.save_tip", "Записати / додати цю фразу в «Швидкі фрази»")
            )
            if already_saved:
                fav_btn.setStyleSheet(
                    "QPushButton { background-color: #243831; color: #6EE7B7; border: 1px solid #34D399; "
                    "border-radius: 6px; padding: 1px 8px; font-size: 11px; font-weight: 600; }"
                    "QPushButton:hover { background-color: #2D4A3F; }"
                )
            else:
                fav_btn.setStyleSheet(
                    "QPushButton { background-color: #2B334D; color: #E2E8F0; border: 1px solid #445075; "
                    "border-radius: 6px; padding: 1px 8px; font-size: 11px; font-weight: 600; }"
                    "QPushButton:hover { background-color: #7C5CFF; border-color: #8B72FF; color: #FFFFFF; }"
                )
            fav_btn.clicked.connect(lambda checked=False, p=text_val: self._on_save_history_to_quick_phrases(p))
            row_lay.addWidget(fav_btn, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            item.setSizeHint(QSize(0, max(30, row_w.sizeHint().height() + 2)))
            self.history_list.addItem(item)
            self.history_list.setItemWidget(item, row_w)

    def _on_clear_history(self) -> None:
        if not config.get("recent_phrases", []):
            return
        if QMessageBox.question(self, tr("common.confirm", "Підтвердження"), tr("confirm.clear_history", "Точно очистити всю історію?")) != QMessageBox.StandardButton.Yes:
            return
        config.set("recent_phrases", [])
        self._refresh_history_list()

    # ──────────────────────────────────────────────────────────────────
    # РОЗДІЛ 3: Кнопки та Ввід (Як викликати програму на ПК)
    # ──────────────────────────────────────────────────────────────────

    def _build_page_hotkeys(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content.setObjectName("scrollContent")
        lay = QVBoxLayout(content)
        lay.setContentsMargins(2, 2, 6, 6)
        lay.setSpacing(12)

        lay.addWidget(
            make_header_with_help(
                tr("ui_272", "⌨️ Клавіші для набору тексту та озвучки"),
                tr("ui_273", "Як працюють гарячі клавіші?"),
                tr(
                    "ui_274",
                    "<b>Працюють будь-де:</b> у дзвінках, браузері, документах та іграх.<br><br>"
                    "1. Натисніть клавішу набору (зараз: <code>{hk_toggle}</code>).<br>"
                    "2. Надрукуйте текст — він з'явиться внизу екрана без згортання поточної програми.<br>"
                    "3. Натисніть <b>{hk_send}</b> — фраза одразу озвучиться.<br><br>"
                    "<i>Щоб перепризначити будь-яку дію (напр. на <code>CapsLock</code> або <code>F1</code>), натисніть <b>«Змінити клавішу»</b>.</i>",
                ),
            )
        )

        # Таблиця основних гарячих клавіш
        hk_card = QFrame()
        hk_card.setProperty("class", "card")
        hk_lay = QVBoxLayout(hk_card)
        hk_lay.setContentsMargins(16, 14, 16, 14)
        hk_lay.setSpacing(10)

        hk_grid = QGridLayout()
        hk_grid.setHorizontalSpacing(14)
        hk_grid.setVerticalSpacing(9)

        self._hk_edits: Dict[str, QLineEdit] = {}
        hk_specs = [
            (
                "hotkey_toggle_input",
                tr("ui_275", "Почати або завершити набір тексту на екрані:"),
                "Ctrl+Space",
                tr("ui_276", "<b>Головна клавіша набору.</b><br>Натисніть будь-де на ПК, щоб почати друкувати фразу поверх поточної програми."),
            ),
            (
                "hotkey_conversation_mode",
                tr("ui_277", "Режим постійного спілкування (Діалог):"),
                "Ctrl+Shift+Space",
                tr("ui_278", "<b>Режим діалогу без пауз.</b><br>Після озвучення фрази рядок вводу не закривається — можна одразу писати наступну, доки самі не вимкнете режим."),
            ),
            (
                "hotkey_send",
                tr("ui_279", "Озвучити набраний текст:"),
                "Enter",
                tr("ui_280", "<b>Відправка на озвучку.</b><br>Натисніть після набору, щоб програма прочитала фразу вголос (і показала в OBS, якщо увімкнено)."),
            ),
            (
                "hotkey_cancel",
                tr("ui_281", "Скасувати набір (стерти і закрити):"),
                "Esc",
                tr("ui_282", "<b>Швидке скасування.</b><br>Очищає набраний текст і закриває рядок вводу без озвучення."),
            ),
            (
                "hotkey_stop_tts",
                tr("ui_283", "Миттєво зупинити голос:"),
                "Ctrl+Shift+S",
                tr("ui_284", "<b>Екстрена зупинка голосу.</b><br>Миттєво перериває поточну фразу та очищає чергу озвучення."),
            ),
            (
                "hotkey_repeat_last",
                tr("ui_285", "Повторити останню сказану фразу:"),
                "Ctrl+Shift+R",
                tr("ui_286", "<b>Швидкий повтор.</b><br>Ще раз озвучує вашу останню сказану фразу без повторного набору."),
            ),
            (
                "hotkey_toggle_game_mode",
                tr("hk.toggle_game_mode", "Увімкнути / вимкнути ігровий режим:"),
                "Ctrl+Shift+G",
                tr("hk.toggle_game_mode_help", "<b>Швидке перемикання ігрового режиму.</b><br>Миттєво вмикає або вимикає ігровий автозахист прямо під час гри."),
            ),
            (
                "hotkey_stop_chat_tts",
                tr("hk.stop_chat_tts", "Зупинити озвучку чату:"),
                "",
                tr("hk.stop_chat_tts_help", "<b>Зупинка озвучки чату.</b><br>Перериває поточну репліку з чату та очищає чергу чату (не перериває ваш власний голос). Доступно, коли увімкнено «Читати чат»."),
            ),
            (
                "hotkey_toggle_chat_tts",
                tr("hk.toggle_chat_tts", "Призупинити / увімкнути озвучку чату:"),
                "",
                tr("hk.toggle_chat_tts_help", "<b>Швидке вимкнення озвучки чату.</b><br>Тимчасово ставить озвучку глядачів на паузу або знову вмикає її. Доступно, коли увімкнено «Читати чат»."),
            ),
            (
                "hotkey_toggle_omnivoice_whisper",
                tr("hk.toggle_omnivoice_whisper", "Увімкнути / вимкнути шепіт (OmniVoice):"),
                "",
                tr("hk.toggle_omnivoice_whisper_help", "<b>Швидке перемикання шепоту OmniVoice.</b><br>Миттєво вмикає або вимикає стиль шепоту (whisper) для наступних фраз. Доступно, коли активним голосом обрано OmniVoice."),
            ),
        ]
        self._chat_hotkey_widgets: List[QWidget] = []
        self._omnivoice_hotkey_widgets: List[QWidget] = []
        for row_i, (cfg_key, label_text, default_hk, help_txt) in enumerate(hk_specs):
            is_chat_hk = cfg_key in ("hotkey_stop_chat_tts", "hotkey_toggle_chat_tts")
            is_ov_hk = cfg_key == "hotkey_toggle_omnivoice_whisper"
            lbl_wrap = QHBoxLayout()
            lbl_wrap.setSpacing(6)
            lbl_w = QLabel(label_text)
            help_w = HelpButton(label_text.rstrip(":"), help_txt)
            lbl_wrap.addWidget(lbl_w)
            lbl_wrap.addWidget(help_w)
            lbl_wrap.addStretch(1)
            hk_grid.addLayout(lbl_wrap, row_i, 0)

            rst_btn = QPushButton("↺")
            rst_btn.setFixedSize(32, 32)
            rst_btn.setStyleSheet("padding: 0px; font-size: 15px; font-weight: 700;")
            rst_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            rst_btn.clicked.connect(lambda checked=False, k=cfg_key, d=default_hk: self._reset_single_hotkey(k, d))
            hk_grid.addWidget(rst_btn, row_i, 1)

            edit = QLineEdit(str(config.get(cfg_key, default_hk)))
            edit.setReadOnly(True)
            edit.setFixedWidth(150)
            edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if is_chat_hk or is_ov_hk or cfg_key == "hotkey_toggle_game_mode":
                edit.setPlaceholderText(tr("hk.not_set", "Не призначено"))
            edit.setStyleSheet("font-weight: 700; color: #C7D2FE;")
            self._hk_edits[cfg_key] = edit
            hk_grid.addWidget(edit, row_i, 2)

            rec_btn = QPushButton(tr("ui_287", "Змінити клавішу"))
            rec_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            rec_btn.clicked.connect(lambda checked=False, k=cfg_key, lbl=label_text: self._record_config_hotkey(k, lbl))
            hk_grid.addWidget(rec_btn, row_i, 3)

            if is_chat_hk:
                self._chat_hotkey_widgets.extend([lbl_w, rst_btn, edit, rec_btn])
            elif is_ov_hk:
                self._omnivoice_hotkey_widgets.extend([lbl_w, rst_btn, edit, rec_btn])

        hk_lay.addLayout(hk_grid)
        self.chat_hk_disabled_hint = QLabel(
            tr(
                "hk.chat_disabled_hint",
                "ℹ️ Кнопки керування озвучкою чату стають активними після увімкнення «Читати та озвучувати повідомлення глядачів» у вкладці «Голос та Озвучка → Чат Twitch».",
            )
        )
        self.chat_hk_disabled_hint.setWordWrap(True)
        self.chat_hk_disabled_hint.setProperty("class", "muted")
        hk_lay.addWidget(self.chat_hk_disabled_hint)
        self._refresh_chat_hotkey_availability()
        self._refresh_omnivoice_whisper_hotkey_availability()
        lay.addWidget(hk_card)

        # Режим роботи клавіатури + звуковий сигнал
        mode_card = QFrame()
        mode_card.setProperty("class", "card")
        m_lay = QVBoxLayout(mode_card)
        m_lay.setContentsMargins(16, 12, 16, 12)
        m_lay.setSpacing(8)

        m_hdr = QHBoxLayout()
        m_hdr.addWidget(QLabel(tr("ui_288", "<b>Спосіб набору тексту:</b>")))
        m_hdr.addWidget(
            HelpButton(
                tr("ui_289", "Спосіб набору тексту"),
                tr(
                    "ui_290",
                    "<b>Який режим обрати?</b><br><br>"
                    "• <b>Звичайний режим (Рекомендовано)</b> — ви друкуєте прямо поверх гри чи програми: вона не згортається і не реагує на натиснуті літери.<br>"
                    "• <b>Запасний режим (активне віконце)</b> — відкриває маленьке поле вводу й саме повертає фокус назад після <b>{hk_send}</b>.<br><br>"
                    "<i>Перемикайте на «Запасний режим» лише якщо якась захищена гра чи програма блокує перехоплення клавіатури.</i>",
                ),
            )
        )
        m_hdr.addStretch(1)
        m_lay.addLayout(m_hdr)

        self.input_mode_combo = QComboBox()
        self.input_mode_combo.addItem(tr("ui_291", "Звичайний режим — писати прямо поверх активної програми (Рекомендовано)"), "hook")
        self.input_mode_combo.addItem(tr("ui_292", "Активне вікно з полем для вводу та кнопками (відкривається одразу активним)"), "fallback")
        cur_m = config.get("input_mode", "hook")
        self.input_mode_combo.setCurrentIndex(0 if cur_m == "hook" else 1)
        self.input_mode_combo.currentIndexChanged.connect(self._on_input_mode_changed)
        m_lay.addWidget(self.input_mode_combo)

        self.chk_continuous_chat = QCheckBox(
            tr("ui_293", "💬 Режим спілкування: не закривати вікно набору після Enter (написав → відправив → одразу пишеш далі)")
        )
        self.chk_continuous_chat.setToolTip(
            tr("ui_294", "Коли увімкнено, після відправки фрази (Enter) вікно залишається активним для наступного речення, доки ви знову не натиснете комбінацію клавіш або Esc")
        )
        self.chk_continuous_chat.setChecked(bool(config.get("continuous_chat_mode", False)))
        self.chk_continuous_chat.toggled.connect(self._on_continuous_chat_toggled)
        m_lay.addWidget(self.chk_continuous_chat)

        sound_row = QHBoxLayout()
        sound_row.setContentsMargins(0, 0, 0, 0)
        sound_row.setSpacing(18)

        self.chk_sound_feedback = QCheckBox(
            tr("ui_295", "Звуковий сигнал при початку набору тексту")
        )
        self.chk_sound_feedback.setToolTip(
            tr("ui_296", "Вмикає або вимикає сигнал у навушниках на початку набору тексту")
        )
        self.chk_sound_feedback.setChecked(bool(config.get("sound_feedback", True)))
        sound_row.addWidget(self.chk_sound_feedback)

        self.chk_cancel_sound = QCheckBox(
            tr("input.cancel_sound_chk", "Звуковий сигнал при скасуванні вводу")
        )
        self.chk_cancel_sound.setToolTip(
            tr("input.cancel_sound_tooltip", "Вмикає або вимикає сигнал у навушниках при скасуванні або виході з режиму набору (Esc або повторне натискання хоткею)")
        )
        self.chk_cancel_sound.setChecked(bool(config.get("cancel_sound_enabled", True)))
        sound_row.addWidget(self.chk_cancel_sound)
        sound_row.addStretch(1)
        m_lay.addLayout(sound_row)

        fb_vol_row = QWidget()
        fb_grid = QGridLayout(fb_vol_row)
        fb_grid.setContentsMargins(0, 2, 0, 0)
        fb_grid.setColumnStretch(1, 1)
        fb_grid.addWidget(
            QLabel(tr("input.sound_feedback_volume_lbl", "Гучність звукових сигналів (початок / скасування):")),
            0,
            0,
        )
        self.sld_feedback_vol = self._make_slider_row(
            fb_grid,
            0,
            1,
            0,
            200,
            int(config.get("sound_feedback_volume", 100)),
            "%",
            lambda: config.set("sound_feedback_volume", int(self.sld_feedback_vol.value())),
        )
        btn_test_beep = QPushButton("🔊")
        btn_test_beep.setFixedSize(30, 26)
        btn_test_beep.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_test_beep.setToolTip(tr("input.sound_feedback_test_tip", "Перевірити гучність звукового сигналу"))
        btn_test_beep.clicked.connect(
            lambda: audio_router.play_feedback_beep("start" if config.get("sound_feedback", True) else "cancel")
        )
        fb_grid.addWidget(btn_test_beep, 0, 2)
        m_lay.addWidget(fb_vol_row)

        def _sync_feedback_vol_row() -> None:
            fb_vol_row.setEnabled(self.chk_sound_feedback.isChecked() or self.chk_cancel_sound.isChecked())

        self.chk_sound_feedback.toggled.connect(lambda v: (config.set("sound_feedback", v), _sync_feedback_vol_row()))
        self.chk_cancel_sound.toggled.connect(lambda v: (config.set("cancel_sound_enabled", v), _sync_feedback_vol_row()))
        _sync_feedback_vol_row()
        lay.addWidget(mode_card)

        # Ігровий режим (автозахист від зависання в бою + чернетка)
        game_card = QFrame()
        game_card.setProperty("class", "card")
        g_lay = QVBoxLayout(game_card)
        g_lay.setContentsMargins(16, 12, 16, 12)
        g_lay.setSpacing(8)

        g_hdr = QHBoxLayout()
        self.chk_game_mode = QCheckBox(
            tr("input.game_mode_chk", "🛡️ Дозволити ігровий режим (автозахист у грі)")
        )
        self.chk_game_mode.setChecked(bool(config.get("game_mode_enabled", True)))
        self.chk_game_mode.setStyleSheet("font-weight: 700;")
        g_hdr.addWidget(self.chk_game_mode)
        g_hdr.addWidget(
            HelpButton(
                tr("input.game_mode_help_title", "Ігровий режим (захист у грі)"),
                tr(
                    "input.game_mode_help_body",
                    "<b>Навіщо це потрібно в іграх?</b><br>"
                    "Під час сліпого набору тексту програма перехоплює клавіші <code>WASD</code>, <code>Пробіл</code> та інші літери, щоб персонаж не бігав.<br><br>"
                    "• <b>У бою:</b> якщо на вас раптом вийшов ворог — клік миші (постріл/приціл) або різкий ривок прицілу миттєво вимикає набір і повертає керування в гру.<br>"
                    "• <b>Поза грою:</b> коли активні Discord, браузер чи робочий стіл, захист не спрацьовує й ввід працює звичайно.<br>"
                    "• <b>Чернетка:</b> недописану фразу можна автоматично зберегти на вказаний час і дописати після бою.<br><br>"
                    "<i>Поза набором тексту мишка не відстежується взагалі — жодного впливу на FPS, сенсу чи античіти.</i>",
                ),
            )
        )
        g_hdr.addStretch(1)
        g_lay.addLayout(g_hdr)

        self.game_mode_subpanel = QWidget()
        gsub_lay = QVBoxLayout(self.game_mode_subpanel)
        gsub_lay.setContentsMargins(18, 2, 0, 0)
        gsub_lay.setSpacing(8)

        self.lbl_game_emergency_cancel = QLabel(
            tr("input.game_emergency_cancel_chk", "⚡ Аварійне скасування набору при стрільбі або різкому русі миші")
        )
        self.lbl_game_emergency_cancel.setToolTip(
            tr("input.game_emergency_cancel_tip", "Миттєво вимикає перехоплення клавіатури при натисканні ЛКМ/ПКМ або швидкому ривку миші (Raw Input).")
        )
        self.lbl_game_emergency_cancel.setWordWrap(True)
        gsub_lay.addWidget(self.lbl_game_emergency_cancel)

        g_sens_row = QHBoxLayout()
        g_sens_row.setContentsMargins(20, 0, 0, 0)
        g_sens_row.setSpacing(10)
        self.game_mouse_sens_lbl = QLabel(tr("input.game_mouse_sens_lbl", "Чутливість ривка миші:"))
        self.game_mouse_sens_combo = QComboBox()
        for label, key in (
            (tr("input.game_sens_high", "Висока (спрацьовує на невеликий рух)"), "high"),
            (tr("input.game_sens_medium", "Середня"), "medium"),
            (tr("input.game_sens_low", "Низька (лише сильний ривок)"), "low"),
            (tr("input.game_sens_click", "Лише клік миші (ЛКМ / ПКМ)"), "click_only"),
        ):
            self.game_mouse_sens_combo.addItem(label, key)
        cur_gs = str(config.get("game_mouse_sensitivity", "low") or "low").strip().lower()
        gs_idx = self.game_mouse_sens_combo.findData(cur_gs)
        if gs_idx >= 0:
            self.game_mouse_sens_combo.setCurrentIndex(gs_idx)
        g_sens_row.addWidget(self.game_mouse_sens_lbl)
        g_sens_row.addWidget(self.game_mouse_sens_combo, 1)
        gsub_lay.addLayout(g_sens_row)

        g_draft_row = QHBoxLayout()
        g_draft_row.setContentsMargins(0, 0, 0, 0)
        g_draft_row.setSpacing(8)
        self.chk_game_save_draft = QCheckBox(
            tr("input.game_save_draft_chk", "💾 Зберігати недописаний текст як чернетку")
        )
        self.chk_game_save_draft.setToolTip(
            tr("input.game_save_draft_tip", "Якщо набір перервало в бою, недописаний текст повернеться при наступному відкритті набору протягом часу зберігання.")
        )
        self.chk_game_save_draft.setChecked(bool(config.get("game_save_draft", True)))
        g_draft_row.addWidget(self.chk_game_save_draft)

        self.game_draft_ttl_spin = QtSpinBox()
        self.game_draft_ttl_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.game_draft_ttl_spin.setRange(5, 300)
        self.game_draft_ttl_spin.setSuffix(tr("input.game_draft_ttl_suffix", " с"))
        self.game_draft_ttl_spin.setFixedWidth(76)
        self.game_draft_ttl_spin.setToolTip(tr("input.game_draft_ttl_lbl", "Таймер зберігання чернетки:"))
        self.game_draft_ttl_spin.setValue(max(5, min(300, int(config.get("game_draft_ttl_sec", 20) or 20))))
        g_draft_row.addWidget(self.game_draft_ttl_spin)
        g_draft_row.addStretch(1)
        gsub_lay.addLayout(g_draft_row)

        g_lay.addWidget(self.game_mode_subpanel)

        def _sync_and_save_game_mode() -> None:
            enabled = self.chk_game_mode.isChecked()
            draft_on = self.chk_game_save_draft.isChecked()
            self.game_mode_subpanel.setVisible(enabled)
            self.lbl_game_emergency_cancel.setEnabled(enabled)
            self.game_mouse_sens_lbl.setEnabled(enabled)
            self.game_mouse_sens_combo.setEnabled(enabled)
            self.game_draft_ttl_spin.setEnabled(enabled and draft_on)
            config.update({
                "game_mode_enabled": enabled,
                "game_emergency_cancel": True,
                "game_mouse_sensitivity": str(self.game_mouse_sens_combo.currentData() or "low"),
                "game_save_draft": draft_on,
                "game_draft_ttl_sec": int(self.game_draft_ttl_spin.value()),
            })
            self.keyboard_hook.update_from_config(config.data)

        self.chk_game_mode.toggled.connect(lambda _v: _sync_and_save_game_mode())
        self.game_mouse_sens_combo.currentIndexChanged.connect(lambda _i: _sync_and_save_game_mode())
        self.chk_game_save_draft.toggled.connect(lambda _v: _sync_and_save_game_mode())
        self.game_draft_ttl_spin.valueChanged.connect(lambda _v: _sync_and_save_game_mode())

        self.game_mode_subpanel.setVisible(self.chk_game_mode.isChecked())
        self.lbl_game_emergency_cancel.setEnabled(self.chk_game_mode.isChecked())
        self.game_mouse_sens_lbl.setEnabled(self.chk_game_mode.isChecked())
        self.game_mouse_sens_combo.setEnabled(self.chk_game_mode.isChecked())
        self.game_draft_ttl_spin.setEnabled(self.chk_game_mode.isChecked() and self.chk_game_save_draft.isChecked())
        lay.addWidget(game_card)

        lay.addStretch(1)
        scroll.setWidget(content)
        return scroll

    def _sync_interactive_window_options(self, enabled: bool) -> None:
        if enabled:
            self._pre_iw_draggable = bool(config.get("overlay_draggable", False))
            target_drag = True
        else:
            target_drag = bool(getattr(self, "_pre_iw_draggable", False))
        if hasattr(self, "chk_overlay_draggable"):
            self.chk_overlay_draggable.blockSignals(True)
            self.chk_overlay_draggable.setChecked(target_drag)
            self.chk_overlay_draggable.blockSignals(False)
        self._on_overlay_draggable_toggled(target_drag)

    def _on_continuous_chat_toggled(self, checked: bool) -> None:
        config.set("continuous_chat_mode", checked)
        self.keyboard_hook.update_from_config(config.data)
        self.fallback_window.set_continuous_mode(checked)

    def _on_input_mode_changed(self) -> None:
        prev_mode = str(config.get("input_mode", "hook") or "hook")
        new_mode = str(self.input_mode_combo.currentData() or "hook")
        config.set("input_mode", new_mode)
        if prev_mode != "fallback" and new_mode == "fallback":
            self._sync_interactive_window_options(True)
        elif prev_mode == "fallback" and new_mode != "fallback":
            self._sync_interactive_window_options(False)
        if hasattr(self, "ov_style_combo"):
            cur_ov = str(self.ov_style_combo.currentData() or "text_only")
            if new_mode == "fallback" and cur_ov != "interactive_window":
                idx = self.ov_style_combo.findData("interactive_window")
                if idx >= 0:
                    self.ov_style_combo.blockSignals(True)
                    self.ov_style_combo.setCurrentIndex(idx)
                    self.ov_style_combo.blockSignals(False)
                    config.set("overlay_style", "interactive_window")
            elif new_mode == "hook" and cur_ov == "interactive_window":
                self.ov_style_combo.blockSignals(True)
                self.ov_style_combo.setCurrentIndex(0)
                self.ov_style_combo.blockSignals(False)
                config.set("overlay_style", "text_only")
        self._refresh_interactive_window_controls()
        self.keyboard_hook.update_from_config(config.data)

    def _refresh_chat_hotkey_availability(self) -> None:
        enabled = bool(config.get("twitch_read_viewers", False))
        tip = (
            ""
            if enabled
            else tr(
                "hk.chat_disabled_tip",
                "Недоступно, доки вимкнено читання чату у вкладці «Голос та Озвучка → Чат Twitch»",
            )
        )
        for w in getattr(self, "_chat_hotkey_widgets", []):
            w.setEnabled(enabled)
            w.setToolTip(tip)
        if hasattr(self, "chat_hk_disabled_hint"):
            self.chat_hk_disabled_hint.setVisible(not enabled)

    def _refresh_omnivoice_whisper_hotkey_availability(self) -> None:
        enabled = str(config.get("active_engine", "") or "").strip() == "omnivoice"
        tip = (
            ""
            if enabled
            else tr(
                "hk.omnivoice_disabled_tip",
                "Недоступно, доки активним голосом не обрано OmniVoice",
            )
        )
        for w in getattr(self, "_omnivoice_hotkey_widgets", []):
            w.setEnabled(enabled)
            w.setToolTip(tip)

    def _record_config_hotkey(self, cfg_key: str, label_text: str) -> None:
        cur = str(config.get(cfg_key, ""))
        dlg = HotkeyCaptureDialog(self.keyboard_hook, cur, label_text, self)
        if dlg.exec() == HotkeyCaptureDialog.DialogCode.Accepted and dlg.recorded_hotkey is not None:
            config.set(cfg_key, dlg.recorded_hotkey)
            if cfg_key in self._hk_edits:
                self._hk_edits[cfg_key].setText(dlg.recorded_hotkey)
            self.keyboard_hook.update_from_config(config.data)
            self._refresh_status_banners()

    def _reset_single_hotkey(self, cfg_key: str, default_hk: str) -> None:
        if QMessageBox.question(self, tr("common.confirm", "Підтвердження"), tr("confirm.reset_hotkey", "Скинути цю клавішу до стандартного значення?")) != QMessageBox.StandardButton.Yes:
            return
        config.set(cfg_key, default_hk)
        if cfg_key in self._hk_edits:
            self._hk_edits[cfg_key].setText(default_hk)
        self.keyboard_hook.update_from_config(config.data)
        self._refresh_status_banners()

    # ──────────────────────────────────────────────────────────────────
    # РОЗДІЛ 4: Текст на екрані (просто, компактно, зрозуміло)
    # ──────────────────────────────────────────────────────────────────

    def _build_page_overlay(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content.setObjectName("scrollContent")
        lay = QVBoxLayout(content)
        lay.setContentsMargins(2, 2, 6, 6)
        lay.setSpacing(10)

        lay.addWidget(
            make_header_with_help(
                tr("ui_297", "✨ Як виглядає ваш текст на екрані"),
                tr("ui_298", "Текст на екрані під час набору"),
                tr(
                    "ui_299",
                    "<b>Що ви бачите під час набору?</b><br>"
                    "Коли натискаєте <b>{hk_toggle}</b> і друкуєте фразу, текст показується поверх екрана (знизу за замовчуванням).<br><br>"
                    "• Показується лише чистий текст (фон можна налаштувати нижче).<br>"
                    "• Рядок вводу прозорий для кліків мишкою і не заважає іграм чи програмам.<br>"
                    "• Натисніть <b>«✥ Перемістити мишкою»</b>, щоб перетягнути текст у будь-яке зручне місце.",
                ),
            )
        )

        # Основна картка налаштувань вигляду тексту (ко)мпактна, все вміщується на одному екрані!)
        style_card = QFrame()
        style_card.setProperty("class", "card")
        s_lay = QVBoxLayout(style_card)
        s_lay.setContentsMargins(16, 12, 16, 12)
        s_lay.setSpacing(10)

        top_s_row = QHBoxLayout()
        self.chk_overlay_enabled = QCheckBox(tr("ui_300", "Показувати текст на екрані під час набору"))
        self.chk_overlay_enabled.setChecked(bool(config.get("overlay_enabled", True)))
        self.chk_overlay_enabled.toggled.connect(lambda v: config.set("overlay_enabled", v))
        top_s_row.addWidget(self.chk_overlay_enabled)
        top_s_row.addStretch(1)

        self.move_ov_btn = QPushButton(tr("ui_301", "✥ Перемістити мишкою"))
        self.move_ov_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.move_ov_btn.setToolTip(tr("ui_302", "Дозволяє взяти текст мишкою і перетягнути його в будь-яке місце екрана"))
        self.move_ov_btn.clicked.connect(self._on_toggle_overlay_move_mode)
        top_s_row.addWidget(self.move_ov_btn)

        preview_ov_btn = QPushButton(tr("ui_303", "👁 Показати приклад"))
        preview_ov_btn.setProperty("class", "primaryBtn")
        preview_ov_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        preview_ov_btn.clicked.connect(self._on_preview_overlay_clicked)
        top_s_row.addWidget(preview_ov_btn)
        s_lay.addLayout(top_s_row)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)

        grid.addWidget(QLabel(tr("ui_305", "Оформлення:")), 0, 0, Qt.AlignmentFlag.AlignTop)
        ov_style_wrap = QVBoxLayout()
        ov_style_wrap.setContentsMargins(0, 0, 0, 0)
        ov_style_wrap.setSpacing(6)
        self.ov_style_combo = QComboBox()
        self.ov_style_combo.addItem(tr("ui_306", "Тільки текст без фону (Рекомендовано)"), "text_only")
        self.ov_style_combo.addItem(tr("ui_307", "Текст на темній плашці з рамкою"), "boxed")
        self.ov_style_combo.addItem(tr("ui_309", "Активне вікно з полем для вводу та кнопками"), "interactive_window")
        cur_st = str(config.get("overlay_style", "text_only"))
        if cur_st == "interactive_box":
            cur_st = "boxed"
        st_idx = self.ov_style_combo.findData(cur_st)
        self.ov_style_combo.setCurrentIndex(st_idx if st_idx >= 0 else 0)
        self.ov_style_combo.currentIndexChanged.connect(self._on_overlay_setting_changed)
        ov_style_wrap.addWidget(self.ov_style_combo)

        self.chk_overlay_draggable = QCheckBox(tr("overlay.draggable_chk", "✥ Перетягувати вікно мишкою"))
        self.chk_overlay_draggable.setChecked(bool(config.get("overlay_draggable", False)))
        self.chk_overlay_draggable.setToolTip(
            tr(
                "overlay.draggable_tip",
                "Дозволяє в будь-який момент перетягувати мишкою текст, плашку чи активне вікно прямо по екрану.",
            )
        )
        self.chk_overlay_draggable.toggled.connect(self._on_overlay_draggable_toggled)
        ov_style_wrap.addWidget(self.chk_overlay_draggable)
        grid.addLayout(ov_style_wrap, 0, 1)

        grid.addWidget(QLabel(tr("ui_310", "Розмір букв:")), 1, 0)
        self.ov_font_slider = self._make_slider_row(
            grid, 1, 1, 8, 60, int(config.get("overlay_font_size", 18)), " px", self._on_overlay_setting_changed
        )

        grid.addWidget(QLabel(tr("ui_311", "Яскравість (непрозорість) тексту:")), 2, 0)
        self.ov_txt_op_slider = self._make_slider_row(
            grid, 2, 1, 20, 100, int(config.get("overlay_text_opacity", 100)), " %", self._on_overlay_setting_changed
        )

        grid.addWidget(QLabel(tr("ui_312", "Товщина темного контуру букв:")), 3, 0)
        self.ov_outline_slider = self._make_slider_row(
            grid, 3, 1, 0, 8, int(config.get("overlay_outline_width", 3)), " px", self._on_overlay_setting_changed
        )

        grid.addWidget(QLabel(tr("ui_313", "Тінь під буквами:")), 4, 0)
        self.ov_shadow_slider = self._make_slider_row(
            grid, 4, 1, 0, 30, int(config.get("overlay_shadow_blur", 10)), " px", self._on_overlay_setting_changed
        )

        grid.addWidget(QLabel(tr("ui_314", "Непрозорість фону (якщо з плашкою):")), 5, 0)
        self.ov_bg_op_slider = self._make_slider_row(
            grid, 5, 1, 0, 100, int(config.get("overlay_bg_opacity", 78)), " %", self._on_overlay_setting_changed
        )

        grid.addWidget(QLabel(tr("ui_315", "Позиція вліво / вправо (X):")), 6, 0)
        self.ov_x_slider = self._make_slider_row(
            grid, 6, 1, -1000, 1000, int(config.get("overlay_x_offset", 0)), " px", self._on_overlay_setting_changed
        )

        grid.addWidget(QLabel(tr("ui_316", "Висота від низу екрана (Y):")), 7, 0)
        self.ov_y_slider = self._make_slider_row(
            grid, 7, 1, 5, 1000, int(config.get("overlay_y_offset", 85)), " px", self._on_overlay_setting_changed
        )

        grid.addWidget(QLabel(tr("overlay.max_width", "Максимальна ширина тексту:")), 8, 0)
        self.ov_max_width_slider = self._make_slider_row(
            grid, 8, 1, 360, 1800, int(config.get("overlay_max_width", 1120)), " px", self._on_overlay_setting_changed
        )

        self.ov_elements_op_lbl = QLabel(tr("overlay.elements_opacity", "Прозорість вікна (крім фону):"))
        grid.addWidget(self.ov_elements_op_lbl, 9, 0)
        self.ov_elements_op_slider = self._make_slider_row(
            grid, 9, 1, 15, 100, int(config.get("overlay_elements_opacity", 30)), " %", self._on_overlay_setting_changed
        )

        self._refresh_interactive_window_controls()
        s_lay.addLayout(grid)

        # Кольори зліва, а кнопка «Скинути до стандартних» вирівняна до правого боку вікна
        bottom_row = QHBoxLayout()
        self.btn_ov_text_col = QPushButton(tr("ui_317", "🎨 Колір тексту"))
        self.btn_ov_text_col.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_ov_text_col.clicked.connect(lambda: self._pick_color("overlay_text_color"))
        self.btn_ov_bg_col = QPushButton(tr("ui_318", "🎨 Колір плашки"))
        self.btn_ov_bg_col.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_ov_bg_col.clicked.connect(lambda: self._pick_color("overlay_bg_color"))
        self.btn_ov_border_col = QPushButton(tr("ui_319", "🎨 Колір рамки"))
        self.btn_ov_border_col.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_ov_border_col.clicked.connect(lambda: self._pick_color("overlay_border_color"))
        bottom_row.addWidget(self.btn_ov_text_col)
        bottom_row.addWidget(self.btn_ov_bg_col)
        bottom_row.addWidget(self.btn_ov_border_col)
        bottom_row.addStretch(1)

        reset_defaults_btn = QPushButton(tr("ui_320", "↺ Скинути до стандартних"))
        reset_defaults_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reset_defaults_btn.clicked.connect(self._on_reset_overlay_defaults)
        bottom_row.addWidget(reset_defaults_btn)

        s_lay.addLayout(bottom_row)
        lay.addWidget(style_card)

        # Додаткові опції: Приховати від OBS або запису екрана + Збереження чернетки при закритті
        extra_card = QFrame()
        extra_card.setProperty("class", "card")
        ex_col = QVBoxLayout(extra_card)
        ex_col.setContentsMargins(16, 10, 16, 10)
        ex_col.setSpacing(8)

        ex_lay = QHBoxLayout()
        ex_lay.setSpacing(8)

        self.chk_hide_from_capture = QCheckBox(tr("ui_322", "Приховати від OBS або запису екрана"))
        self.chk_hide_from_capture.setChecked(bool(config.get("hide_from_capture", False)))
        self.chk_hide_from_capture.setToolTip(
            tr("ui_323", "Ви бачитимете набраний текст на своєму моніторі, але на стрімі в OBS чи записі екрана рядок вводу буде прихований")
        )
        self.chk_hide_from_capture.toggled.connect(self._on_hide_from_capture_toggled)
        ex_lay.addWidget(self.chk_hide_from_capture)
        ex_lay.addWidget(
            HelpButton(
                tr("ui_322", "Приховати від OBS або запису екрана"),
                tr(
                    "ui_324",
                    "<b>Коли це потрібно?</b><br>"
                    "Лише якщо ви стрімите або записуєте екран і не хочете, щоб глядачі бачили, як ви набираєте текст.<br><br>"
                    "• <b>Коли увімкнено:</b> на вашому моніторі набраний текст і рядок вводу видно, а на записі відео чи стрімі в OBS — ні.<br>"
                    "• <b>Готові субтитри</b> через Browser Source у розділі OBS показуватимуться як зазвичай.<br><br>"
                    "<i>Якщо ви не ведете запис або стріми — вмикати не потрібно.</i>",
                ),
            )
        )
        ex_lay.addStretch(1)
        ex_col.addLayout(ex_lay)

        draft_row = QHBoxLayout()
        draft_row.setSpacing(8)
        self.chk_save_draft_on_close = QCheckBox(
            tr("overlay.save_draft_on_close_chk", "💾 Зберігати недописаний текст при закритті набору")
        )
        self.chk_save_draft_on_close.setChecked(bool(config.get("save_draft_on_close", True)))
        self.chk_save_draft_on_close.setToolTip(
            tr(
                "overlay.save_draft_on_close_tip",
                "Якщо випадково закрили набір без озвучення (Esc або хоткеєм), недописаний текст повернеться при повторному відкритті протягом вказаного часу.",
            )
        )
        draft_row.addWidget(self.chk_save_draft_on_close)
        draft_row.addWidget(
            HelpButton(
                tr("overlay.save_draft_on_close_chk", "💾 Зберігати недописаний текст при закритті набору"),
                tr(
                    "overlay.save_draft_on_close_help",
                    "<b>Захист від випадкового закриття:</b><br>"
                    "Якщо ви не встигли дописати фразу та випадково закрили рядок вводу (<code>Esc</code>, хоткеєм або кнопкою), текст збережеться як тимчасова чернетка.<br><br>"
                    "При повторному відкритті в межах вказаних секунд праворуч недописаний текст автоматично повернеться, щоб не довелося вводити його наново.",
                ),
            )
        )
        self.spin_draft_keep_ttl = QtSpinBox()
        self.spin_draft_keep_ttl.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spin_draft_keep_ttl.setRange(1, 300)
        self.spin_draft_keep_ttl.setSuffix(tr("input.game_draft_ttl_suffix", " с"))
        self.spin_draft_keep_ttl.setFixedWidth(76)
        self.spin_draft_keep_ttl.setToolTip(
            tr("overlay.draft_keep_ttl_tip", "Скільки секунд зберігати недописаний текст після закриття")
        )
        self.spin_draft_keep_ttl.setValue(max(1, min(300, int(config.get("draft_keep_ttl_sec", 10) or 10))))
        draft_row.addWidget(self.spin_draft_keep_ttl)
        draft_row.addStretch(1)
        ex_col.addLayout(draft_row)

        def _sync_general_draft_settings() -> None:
            on = self.chk_save_draft_on_close.isChecked()
            self.spin_draft_keep_ttl.setEnabled(on)
            config.update({
                "save_draft_on_close": on,
                "draft_keep_ttl_sec": int(self.spin_draft_keep_ttl.value()),
            })
            self.keyboard_hook.update_from_config(config.data)

        self.chk_save_draft_on_close.toggled.connect(lambda _v: _sync_general_draft_settings())
        self.spin_draft_keep_ttl.valueChanged.connect(lambda _v: _sync_general_draft_settings())
        self.spin_draft_keep_ttl.setEnabled(self.chk_save_draft_on_close.isChecked())

        lay.addWidget(extra_card)

        # Інтеграція з чатом Twitch (без реєстрації на dev.twitch.tv)
        from core.twitch_chat import twitch_chat

        twitch_card = QFrame()
        twitch_card.setProperty("class", "card")
        tw_lay = QVBoxLayout(twitch_card)
        tw_lay.setContentsMargins(16, 10, 16, 10)
        tw_lay.setSpacing(8)

        tw_hdr_row = QHBoxLayout()
        tw_hdr_row.setContentsMargins(0, 0, 0, 0)
        self.twitch_expand_btn = QToolButton()
        self.twitch_expand_btn.setCheckable(True)
        self.twitch_expand_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.twitch_expand_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.twitch_expand_btn.setArrowType(Qt.ArrowType.RightArrow)
        self.twitch_expand_btn.setText(tr("twitch.card.title", "💬 Надсилання фраз у чат Twitch"))
        self.twitch_expand_btn.setStyleSheet(
            "QToolButton { border: 0; background: transparent; color: #F1F5F9; "
            "font-size: 14px; font-weight: 700; padding: 4px; text-align: left; }"
            "QToolButton:hover { color: #A78BFA; }"
        )
        tw_hdr_row.addWidget(self.twitch_expand_btn)
        tw_hdr_row.addStretch(1)
        self.twitch_badge_lbl = QLabel()
        self.twitch_badge_lbl.setStyleSheet("font-size: 12px; color: #AAB4C8;")
        self.twitch_badge_lbl.setVisible(False)
        tw_hdr_row.addWidget(self.twitch_badge_lbl)
        tw_lay.addLayout(tw_hdr_row)

        self.twitch_body = QWidget()
        tw_body_lay = QVBoxLayout(self.twitch_body)
        tw_body_lay.setContentsMargins(0, 2, 0, 0)
        tw_body_lay.setSpacing(8)

        self.twitch_enable_chk = QCheckBox(tr("twitch.enable", "Дублювати озвучені фрази в чат Twitch"))
        self.twitch_enable_chk.setChecked(bool(config.get("twitch_enabled", False)))
        tw_body_lay.addWidget(self.twitch_enable_chk)

        self.twitch_intercept_chk = QCheckBox(
            tr("twitch.intercept_own", "Перехоплювати мої повідомлення з чату Twitch та озвучувати")
        )
        self.twitch_intercept_chk.setChecked(bool(config.get("twitch_intercept_own", False)))
        tw_body_lay.addWidget(self.twitch_intercept_chk)

        tw_help_row = QHBoxLayout()
        tw_help = QLabel(
            tr(
                "twitch.help",
                "Натисніть «Отримати токен», виберіть Bot Chat Token, увійдіть у Twitch і скопіюйте Access Token.",
            )
        )
        tw_help.setWordWrap(True)
        tw_help.setStyleSheet("color: #AAB4C8;")
        tw_help_row.addWidget(tw_help, 1)
        self.twitch_get_token_btn = QPushButton(tr("twitch.get_token", "🔑 Отримати токен (twitchtokengenerator.com)"))
        self.twitch_get_token_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://twitchtokengenerator.com/"))
        )
        tw_help_row.addWidget(self.twitch_get_token_btn)
        tw_body_lay.addLayout(tw_help_row)

        tw_form = QGridLayout()
        tw_form.setColumnStretch(1, 1)
        tw_form.addWidget(QLabel(tr("twitch.token", "Access Token:")), 0, 0)
        tw_tok_wrap = QHBoxLayout()
        tw_tok_wrap.setContentsMargins(0, 0, 0, 0)
        tw_tok_wrap.setSpacing(6)
        self.twitch_token_edit = QLineEdit(str(config.get("twitch_token", "")))
        self.twitch_token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.twitch_token_edit.setPlaceholderText(tr("twitch.token_ph", "Вставте Access Token із twitchtokengenerator.com"))
        self.twitch_token_eye_btn = QToolButton()
        self.twitch_token_eye_btn.setCheckable(True)
        self.twitch_token_eye_btn.setText("👁")
        self.twitch_token_eye_btn.toggled.connect(
            lambda shown: self.twitch_token_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if shown else QLineEdit.EchoMode.Password
            )
        )
        tw_tok_wrap.addWidget(self.twitch_token_edit, 1)
        tw_tok_wrap.addWidget(self.twitch_token_eye_btn)
        tw_form.addLayout(tw_tok_wrap, 0, 1)

        tw_form.addWidget(QLabel(tr("twitch.channel", "Канал (необов'язково):")), 1, 0)
        self.twitch_channel_edit = QLineEdit(str(config.get("twitch_channel", "")))
        self.twitch_channel_edit.setPlaceholderText(tr("twitch.channel_ph", "Ваш канал (або вкажіть інший)"))
        tw_form.addWidget(self.twitch_channel_edit, 1, 1)
        tw_body_lay.addLayout(tw_form)

        tw_actions = QHBoxLayout()
        self.twitch_test_btn = QPushButton(tr("twitch.test", "Надіслати тест у чат"))
        self.twitch_status_lbl = QLabel()
        self.twitch_status_lbl.setWordWrap(True)
        tw_actions.addWidget(self.twitch_test_btn)
        tw_actions.addWidget(self.twitch_status_lbl, 1)
        tw_body_lay.addLayout(tw_actions)

        self.twitch_body.setVisible(False)
        tw_lay.addWidget(self.twitch_body)

        def _toggle_twitch_body(checked: bool) -> None:
            toggle_collapsible_widget(self.twitch_expand_btn, self.twitch_body, checked)

        self.twitch_expand_btn.toggled.connect(_toggle_twitch_body)

        self._reconnect_signal(self.twitch_status_changed, "_slot_twitch_status", self._apply_twitch_status_ui)
        twitch_chat.set_status_callback(lambda st, dtl: self.twitch_status_changed.emit(st, dtl))

        def _on_twitch_resolved(_user: str, chan: str) -> None:
            if chan and not self.twitch_channel_edit.text().strip():
                self.twitch_channel_edit.blockSignals(True)
                self.twitch_channel_edit.setText(chan)
                self.twitch_channel_edit.blockSignals(False)
                config.set("twitch_channel", chan)

        self._reconnect_signal(self.twitch_resolved, "_slot_twitch_resolved", _on_twitch_resolved)
        twitch_chat.set_resolved_callback(lambda u, c: self.twitch_resolved.emit(u, c))

        self.twitch_enable_chk.toggled.connect(lambda _c: self._save_twitch_settings("overlay"))
        self.twitch_intercept_chk.toggled.connect(lambda _c: self._save_twitch_settings("overlay"))
        self.twitch_token_edit.editingFinished.connect(lambda: self._save_twitch_settings("overlay"))
        self.twitch_channel_edit.editingFinished.connect(lambda: self._save_twitch_settings("overlay"))
        self._sync_twitch_connection()

        def _on_twitch_test_finished(ok: bool, message: str) -> None:
            self.twitch_test_btn.setEnabled(True)
            self.twitch_status_lbl.setStyleSheet("color: #4ADE80;" if ok else "color: #F87171;")
            self.twitch_status_lbl.setText(message)

        self._reconnect_signal(self.twitch_test_finished, "_slot_twitch_test", _on_twitch_test_finished)

        def _send_twitch_test() -> None:
            self._save_twitch_settings("overlay")
            tok = self.twitch_token_edit.text().strip()
            chan = self.twitch_channel_edit.text().strip()
            if not tok:
                self.twitch_status_lbl.setStyleSheet("color: #FACC15;")
                self.twitch_status_lbl.setText(
                    tr("twitch.err.fill_fields", "Вставте Access Token Twitch.")
                )
                return
            self.twitch_test_btn.setEnabled(False)
            self.twitch_status_lbl.setStyleSheet("")
            self.twitch_status_lbl.setText(tr("twitch.testing", "Підключення та надсилання тесту..."))

            def _worker() -> None:
                try:
                    ok, clean_chan = twitch_chat.test_connection(
                        "",
                        tok,
                        chan,
                        test_message=tr("twitch.test_chat_msg", "SilentVoice: тест чату Twitch!"),
                    )
                    msg = tr("twitch.test_ok", "Тестове повідомлення надіслано в #{channel}.", channel=clean_chan)
                    self.twitch_test_finished.emit(ok, msg)
                except Exception as exc:
                    msg = tr("twitch.test_fail", "Помилка: {error}", error=str(exc))
                    self.twitch_test_finished.emit(False, msg)

            threading.Thread(target=_worker, daemon=True).start()

        self.twitch_test_btn.clicked.connect(_send_twitch_test)
        lay.addWidget(twitch_card)

        lay.addStretch(1)
        scroll.setWidget(content)
        return scroll

    def _make_slider_row(
        self,
        grid: QGridLayout,
        row: int,
        col: int,
        min_v: int,
        max_v: int,
        val: int,
        suffix: str,
        callback,
    ) -> QSlider:
        wrap = QWidget()
        h = QHBoxLayout(wrap)
        h.setContentsMargins(0, 0, 0, 0)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setSingleStep(1)
        slider.setPageStep(1)
        slider.setValue(val)
        val_lbl = QLabel(f"{val}{suffix}")
        val_lbl.setFixedWidth(54)
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        slider._val_lbl = val_lbl  # type: ignore[attr-defined]
        slider._val_suffix = suffix  # type: ignore[attr-defined]
        slider.valueChanged.connect(lambda v: (val_lbl.setText(f"{v}{suffix}"), callback()))
        h.addWidget(slider, 1)
        h.addWidget(val_lbl)
        grid.addWidget(wrap, row, col)
        return slider

    @staticmethod
    def _set_slider_silently(slider: QSlider, val: int) -> None:
        slider.blockSignals(True)
        slider.setValue(val)
        slider.blockSignals(False)
        lbl = getattr(slider, "_val_lbl", None)
        suffix = getattr(slider, "_val_suffix", "")
        if lbl is not None:
            lbl.setText(f"{slider.value()}{suffix}")

    def _on_preview_overlay_clicked(self) -> None:
        if str(self.ov_style_combo.currentData()) == "interactive_window":
            self.fallback_window.open_and_focus()
        else:
            self.overlay.show_preview()

    def _refresh_interactive_window_controls(self) -> None:
        if not hasattr(self, "ov_elements_op_lbl") or not hasattr(self, "ov_elements_op_slider"):
            return
        cur_style = str(
            self.ov_style_combo.currentData()
            if hasattr(self, "ov_style_combo")
            else config.get("overlay_style", "text_only")
        )
        is_iw = cur_style == "interactive_window"
        self.ov_elements_op_lbl.setVisible(is_iw)
        wrap = self.ov_elements_op_slider.parentWidget()
        if wrap is not None:
            wrap.setVisible(is_iw)

    def _on_overlay_setting_changed(self) -> None:
        prev_style = str(config.get("overlay_style", "text_only") or "text_only")
        new_style = str(self.ov_style_combo.currentData() or "text_only")
        new_input_mode = "fallback" if new_style == "interactive_window" else "hook"
        style_changed = prev_style != new_style
        config.update({
            "overlay_style": new_style,
            "input_mode": new_input_mode,
            "overlay_font_size": self.ov_font_slider.value(),
            "overlay_bg_opacity": self.ov_bg_op_slider.value(),
            "overlay_elements_opacity": self.ov_elements_op_slider.value() if hasattr(self, "ov_elements_op_slider") else int(config.get("overlay_elements_opacity", 30)),
            "overlay_text_opacity": self.ov_txt_op_slider.value(),
            "overlay_outline_width": self.ov_outline_slider.value(),
            "overlay_shadow_blur": self.ov_shadow_slider.value(),
            "overlay_x_offset": self.ov_x_slider.value(),
            "overlay_y_offset": self.ov_y_slider.value(),
            "overlay_max_width": self.ov_max_width_slider.value(),
        }, auto_save=style_changed)
        if not style_changed:
            config.schedule_save()
        if prev_style != "interactive_window" and new_style == "interactive_window":
            self._sync_interactive_window_options(True)
        elif prev_style == "interactive_window" and new_style != "interactive_window":
            self._sync_interactive_window_options(False)
        if style_changed:
            self._refresh_interactive_window_controls()
            if hasattr(self, "input_mode_combo"):
                self.input_mode_combo.blockSignals(True)
                self.input_mode_combo.setCurrentIndex(1 if new_input_mode == "fallback" else 0)
                self.input_mode_combo.blockSignals(False)
            self.keyboard_hook.update_from_config(config.data)

        is_dragging = any(
            s is not None and s.isSliderDown()
            for s in (
                self.ov_font_slider,
                self.ov_bg_op_slider,
                getattr(self, "ov_elements_op_slider", None),
                self.ov_txt_op_slider,
                self.ov_outline_slider,
                self.ov_shadow_slider,
                self.ov_x_slider,
                self.ov_y_slider,
                self.ov_max_width_slider,
            )
        )
        if is_dragging and not style_changed:
            timer = getattr(self, "_ov_preview_timer", None)
            if timer is None:
                timer = QTimer(self)
                timer.setSingleShot(True)
                timer.setInterval(16)
                timer.timeout.connect(self._flush_overlay_preview_update)
                self._ov_preview_timer = timer
            self._ov_preview_style = new_style
            if not timer.isActive():
                timer.start()
            return

        timer = getattr(self, "_ov_preview_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()
        self._apply_overlay_preview(new_style)

    def _flush_overlay_preview_update(self) -> None:
        style = getattr(self, "_ov_preview_style", str(config.get("overlay_style", "text_only") or "text_only"))
        self._apply_overlay_preview(style)

    def _apply_overlay_preview(self, style: str) -> None:
        if style == "interactive_window":
            self.fallback_window.apply_theme_from_config()
            if self.fallback_window.isVisible():
                self.fallback_window.reposition_only()
        else:
            self.overlay.reposition()
            if self.overlay.isVisible() and self.overlay._mode == "preview":
                self.overlay.update()
                self.overlay._hide_timer.start(4000)
            else:
                self.overlay.show_preview()

    def _on_overlay_draggable_toggled(self, checked: bool) -> None:
        config.set("overlay_draggable", bool(checked))
        if hasattr(self, "overlay"):
            self.overlay._apply_win32_flags()
            if self.overlay.isVisible():
                self.overlay._sync_drag_mask()
                self.overlay.update()
        if hasattr(self, "fallback_window"):
            self.fallback_window.sync_draggable_cursor()

    def _on_toggle_overlay_move_mode(self) -> None:
        new_state = not self.overlay.is_move_mode
        self.overlay.set_interactive_move(new_state)

    def _on_overlay_move_mode_changed(self, active: bool) -> None:
        if active:
            self.move_ov_btn.setText(tr("ui_325", "✅ Зафіксувати місце"))
            self.move_ov_btn.setProperty("class", "successBtn")
        else:
            self.move_ov_btn.setText(tr("ui_301", "✥ Перемістити мишкою"))
            self.move_ov_btn.setProperty("class", "")
        self.move_ov_btn.style().unpolish(self.move_ov_btn)
        self.move_ov_btn.style().polish(self.move_ov_btn)

    def _on_overlay_dragged(self, x_off: int, y_off: int) -> None:
        self._set_slider_silently(self.ov_x_slider, x_off)
        self._set_slider_silently(self.ov_y_slider, y_off)

    DEFAULT_COLOR_VALUES: Dict[str, str] = {
        "overlay_text_color": "#FFFFFF",
        "overlay_bg_color": "#13151F",
        "overlay_border_color": "#7C5CFF",
        "obs_text_color": "#FFFFFF",
        "obs_bg_color": "#13151F",
        "obs_border_color": "#7C5CFF",
        "obs_cursor_color": "#9B82FF",
    }

    def _ask_reset_with_colors(self, default_colors: Dict[str, str]) -> Optional[bool]:
        """
        Якщо хоча б один колір відрізняється від стандартного, запитує:
        True = скинути усе, False = скинути усе крім кольорів, None = скасувати.
        Якщо всі кольори вже стандартні — одразу повертає True.
        """
        has_custom = any(
            str(config.get(k, d)).strip().upper() != d.upper()
            for k, d in default_colors.items()
        )
        if not has_custom:
            answer = QMessageBox.question(
                self,
                tr("ui.reset_colors_title", "Скидання налаштувань"),
                tr("ui.reset_defaults_confirm", "Скинути налаштування цього розділу до стандартних?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            return True if answer == QMessageBox.StandardButton.Yes else None
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(tr("ui.reset_colors_title", "Скидання налаштувань"))
        box.setText(
            tr(
                "ui.reset_colors_msg",
                "Ви налаштували власні кольори. Скинути усе разом із кольорами чи залишити обрані кольори?",
            )
        )
        btn_all = box.addButton(tr("ui.reset_all_btn", "Скинути усе"), QMessageBox.ButtonRole.AcceptRole)
        btn_keep = box.addButton(tr("ui.reset_keep_colors_btn", "Скинути усе, крім кольорів"), QMessageBox.ButtonRole.ActionRole)
        btn_cancel = box.addButton(tr("ui.cancel_btn", "Скасувати"), QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(btn_keep)
        box.exec()
        clicked = box.clickedButton()
        if clicked == btn_all:
            return True
        if clicked == btn_keep:
            return False
        return None

    def _on_reset_overlay_defaults(self) -> None:
        ov_colors = {
            "overlay_text_color": "#FFFFFF",
            "overlay_bg_color": "#13151F",
            "overlay_border_color": "#7C5CFF",
        }
        reset_colors = self._ask_reset_with_colors(ov_colors)
        if reset_colors is None:
            return
        if self.overlay.is_move_mode:
            self.overlay.set_interactive_move(False)
        upd: Dict[str, Any] = {
            "overlay_enabled": True,
            "overlay_style": "text_only",
            "overlay_draggable": False,
            "input_mode": "hook",
            "overlay_font_size": 18,
            "overlay_text_opacity": 100,
            "overlay_outline_width": 3,
            "overlay_shadow_blur": 10,
            "overlay_bg_opacity": 78,
            "overlay_elements_opacity": 30,
            "overlay_x_offset": 0,
            "overlay_y_offset": 85,
            "overlay_screen_name": "",
            "overlay_max_width": 1120,
        }
        if reset_colors:
            upd.update(ov_colors)
        config.update(upd)
        if hasattr(self, "input_mode_combo"):
            self.input_mode_combo.blockSignals(True)
            self.input_mode_combo.setCurrentIndex(0)
            self.input_mode_combo.blockSignals(False)
        self.keyboard_hook.update_from_config(config.data)
        self.fallback_window.apply_theme_from_config()
        self.chk_overlay_enabled.blockSignals(True)
        self.chk_overlay_enabled.setChecked(True)
        self.chk_overlay_enabled.blockSignals(False)
        if hasattr(self, "chk_overlay_draggable"):
            self.chk_overlay_draggable.blockSignals(True)
            self.chk_overlay_draggable.setChecked(False)
            self.chk_overlay_draggable.blockSignals(False)

        self.ov_style_combo.blockSignals(True)
        self.ov_style_combo.setCurrentIndex(0)
        self.ov_style_combo.blockSignals(False)

        self._set_slider_silently(self.ov_font_slider, 18)
        self._set_slider_silently(self.ov_txt_op_slider, 100)
        self._set_slider_silently(self.ov_outline_slider, 3)
        self._set_slider_silently(self.ov_shadow_slider, 10)
        self._set_slider_silently(self.ov_bg_op_slider, 78)
        if hasattr(self, "ov_elements_op_slider"):
            self._set_slider_silently(self.ov_elements_op_slider, 30)
        self._refresh_interactive_window_controls()
        self._set_slider_silently(self.ov_x_slider, 0)
        self._set_slider_silently(self.ov_y_slider, 85)
        self._set_slider_silently(self.ov_max_width_slider, 1120)

        self.overlay.show_preview()

    def _pick_color(self, config_key: str, is_obs: bool = False) -> None:
        default_hex = self.DEFAULT_COLOR_VALUES.get(config_key, "#FFFFFF")
        cur = QColor(str(config.get(config_key, default_hex)))
        dlg = QColorDialog(cur, self)
        dlg.setWindowTitle(tr("ui_326", "Оберіть колір"))
        dlg.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
        _modernize_child_spinboxes(dlg)
        bb = dlg.findChild(QDialogButtonBox)
        if bb is not None:
            reset_btn = bb.addButton(
                tr("ui.color_reset_default", "↺ Стандартний колір"),
                QDialogButtonBox.ButtonRole.ResetRole,
            )
            reset_btn.clicked.connect(lambda: dlg.setCurrentColor(QColor(default_hex)))
        if dlg.exec() == QDialog.DialogCode.Accepted:
            col = dlg.currentColor()
            if col.isValid():
                config.set(config_key, col.name())
                if is_obs:
                    obs_server.broadcast_style()
                else:
                    self.overlay.show_preview()

    def _on_hide_from_capture_toggled(self, checked: bool) -> None:
        config.set("hide_from_capture", checked)
        self._apply_window_capture_protection()

    def _reconnect_signal(self, signal: Any, attr_name: str, slot: Any) -> None:
        old_slot = getattr(self, attr_name, None)
        if old_slot is not None:
            try:
                signal.disconnect(old_slot)
            except (RuntimeError, TypeError):
                pass
        setattr(self, attr_name, slot)
        signal.connect(slot)

    def _apply_native_win_icon(self) -> None:
        """Явно встановлює нативні Win32 ICON_SMALL/ICON_BIG із assets/icon.ico для панелі задач Windows."""
        try:
            import ctypes
            from ctypes import wintypes

            hwnd = int(self.winId())
            if not hwnd:
                return
            user32 = ctypes.windll.user32
            user32.LoadImageW.argtypes = [
                wintypes.HINSTANCE,
                wintypes.LPCWSTR,
                wintypes.UINT,
                ctypes.c_int,
                ctypes.c_int,
                wintypes.UINT,
            ]
            user32.LoadImageW.restype = wintypes.HANDLE
            user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            user32.SendMessageW.restype = ctypes.c_ssize_t

            cached = getattr(self, "_native_win_icons", None)
            if cached is None:
                ico_path = _find_app_icon_path()
                if not ico_path:
                    return
                # IMAGE_ICON = 1, LR_LOADFROMFILE = 0x0010
                h_sm = user32.LoadImageW(None, ico_path, 1, 16, 16, 0x0010)
                h_lg = user32.LoadImageW(None, ico_path, 1, 48, 48, 0x0010) or user32.LoadImageW(None, ico_path, 1, 32, 32, 0x0010)
                cached = (h_sm, h_lg)
                self._native_win_icons = cached
            h_sm, h_lg = cached
            WM_SETICON = 0x0080
            if h_sm:
                user32.SendMessageW(wintypes.HWND(hwnd), WM_SETICON, 0, h_sm)
            if h_lg:
                user32.SendMessageW(wintypes.HWND(hwnd), WM_SETICON, 1, h_lg)
        except Exception:
            pass

    def _apply_window_capture_protection(self) -> None:
        # Захист від запису екрана / OBS діє виключно для вікон вводу тексту (оверлею та інтерактивного вікна),
        # головне вікно програми завжди залишається видимим.
        set_window_capture_excluded(int(self.winId()), False)
        self.overlay.update_capture_affinity()
        self.fallback_window.update_capture_affinity()

    # ──────────────────────────────────────────────────────────────────
    # РОЗДІЛ 5: Субтитри для OBS (для стрімерів)
    # ──────────────────────────────────────────────────────────────────

    def _build_page_obs(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content.setObjectName("scrollContent")
        lay = QVBoxLayout(content)
        lay.setContentsMargins(2, 2, 6, 6)
        lay.setSpacing(10)

        lay.addWidget(
            make_header_with_help(
                tr("ui_327", "🎬 Субтитри для OBS Studio"),
                tr("ui_328", "Для чого потрібен цей розділ?"),
                tr(
                    "ui_329",
                    "<b>Субтитри для глядачів на стрімі:</b><br>"
                    "Виводить озвучені фрази як красиві анімовані субтитри прямо поверх трансляції в <b>OBS Studio</b>.<br><br>"
                    "<b>Як додати за 3 кроки:</b><br>"
                    "1. Натисніть <b>«📋 Копіювати посилання для OBS»</b> нижче.<br>"
                    "2. В OBS додайте джерело <b>«Браузер» (Browser Source)</b> і вставте посилання.<br>"
                    "3. Вкажіть розмір <b>1200 × 200 px</b> (стандартні 800×600 занадто вузькі) і поставте рамку внизу сцени.<br><br>"
                    "<i>Якщо ви не користуєтесь OBS — цей розділ можна не чіпати.</i>",
                ),
            )
        )

        srv_card = QFrame()
        srv_card.setProperty("class", "cardHighlight")
        srv_lay = QVBoxLayout(srv_card)
        srv_lay.setContentsMargins(16, 12, 16, 12)
        srv_lay.setSpacing(10)

        r1 = QHBoxLayout()
        self.chk_obs_enabled = QCheckBox(tr("ui_330", "Увімкнути субтитри для OBS"))
        self.chk_obs_enabled.setChecked(bool(config.get("obs_enabled", True)))
        self.chk_obs_enabled.toggled.connect(self._on_obs_enabled_toggled)
        r1.addWidget(self.chk_obs_enabled)
        r1.addStretch(1)

        r1.addWidget(QLabel(tr("ui_331", "Порт:")))
        self.obs_port_spin = QtSpinBox()
        self.obs_port_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.obs_port_spin.setRange(1024, 65535)
        self.obs_port_spin.setValue(int(config.get("obs_port", 8765)))
        r1.addWidget(self.obs_port_spin)

        restart_srv_btn = QPushButton(tr("ui_332", "🔄 Оновити порт"))
        restart_srv_btn.clicked.connect(self._on_obs_restart_port)
        r1.addWidget(restart_srv_btn)
        srv_lay.addLayout(r1)

        r2 = QHBoxLayout()
        self.obs_url_edit = QLineEdit(obs_server.browser_source_url)
        self.obs_url_edit.setReadOnly(True)
        r2.addWidget(self.obs_url_edit, 1)

        copy_url_btn = QPushButton(tr("ui_333", "📋 Копіювати посилання для OBS"))
        copy_url_btn.setProperty("class", "primaryBtn")
        copy_url_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        copy_url_btn.clicked.connect(self._copy_obs_url)
        r2.addWidget(copy_url_btn)

        open_browser_btn = QPushButton(tr("ui_334", "🌐 Перевірити в браузері"))
        open_browser_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_browser_btn.clicked.connect(self._open_obs_browser_preview)
        r2.addWidget(open_browser_btn)
        srv_lay.addLayout(r2)
        obs_size_hint = QLabel(
            tr(
                "obs.size_hint",
                "💡 Рекомендований розмір у налаштуваннях OBS (Браузер): 1200 × 200 px (ширина 1200, висота 200; стандартні 800×600 занадто вузькі та зависокі).",
            )
        )
        obs_size_hint.setWordWrap(True)
        obs_size_hint.setStyleSheet("color: #A5B4FC; font-size: 12px;")
        srv_lay.addWidget(obs_size_hint)
        lay.addWidget(srv_card)

        # Блок вибору шаблону OBS (вбудовані стилі або власний HTML/CSS)
        from core.obs_server import ensure_custom_obs_templates

        tpl_card = QFrame()
        tpl_card.setProperty("class", "card")
        tpl_lay = QVBoxLayout(tpl_card)
        tpl_lay.setContentsMargins(16, 12, 16, 12)
        tpl_lay.setSpacing(8)

        tpl_hdr = QLabel(tr("obs.tpl.title", "🧩 Шаблон субтитрів OBS (готовий або власний):"))
        tpl_hdr.setProperty("class", "cardHeader")
        tpl_lay.addWidget(tpl_hdr)

        tpl_row = QHBoxLayout()
        tpl_row.setSpacing(8)
        tpl_row.addWidget(QLabel(tr("obs.tpl.mode_lbl", "Режим:")))
        self.obs_template_mode_combo = QComboBox()
        self.obs_template_mode_combo.currentIndexChanged.connect(self._on_obs_template_mode_changed)
        tpl_row.addWidget(self.obs_template_mode_combo, 1)

        self.obs_tpl_list_refresh_btn = QPushButton("↻")
        self.obs_tpl_list_refresh_btn.setFixedSize(32, 32)
        self.obs_tpl_list_refresh_btn.setStyleSheet("padding: 0px; font-size: 15px;")
        self.obs_tpl_list_refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.obs_tpl_list_refresh_btn.setToolTip(tr("obs.tpl.refresh_list_tip", "Оновити список шаблонів із папки"))
        self.obs_tpl_list_refresh_btn.clicked.connect(self._refresh_obs_template_modes)
        tpl_row.addWidget(self.obs_tpl_list_refresh_btn)
        tpl_lay.addLayout(tpl_row)

        tpl_btns = QHBoxLayout()
        tpl_btns.setSpacing(8)
        self.obs_tpl_open_btn = QPushButton(tr("obs.tpl.open_dir", "📂 Відкрити папку шаблонів"))
        self.obs_tpl_open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.obs_tpl_open_btn.clicked.connect(self._open_obs_templates_folder)
        tpl_btns.addWidget(self.obs_tpl_open_btn)

        self.obs_tpl_reload_btn = QPushButton(tr("obs.tpl.reload", "🔄 Оновити OBS"))
        self.obs_tpl_reload_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.obs_tpl_reload_btn.clicked.connect(self._reload_obs_custom_template)
        tpl_btns.addWidget(self.obs_tpl_reload_btn)

        self.obs_tpl_reset_btn = QPushButton(tr("obs.tpl.reset", "↺ Скинути власний шаблон"))
        self.obs_tpl_reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.obs_tpl_reset_btn.clicked.connect(self._reset_obs_custom_template)
        tpl_btns.addWidget(self.obs_tpl_reset_btn)
        tpl_btns.addStretch(1)
        tpl_lay.addLayout(tpl_btns)

        self.obs_custom_hint_lbl = QLabel("")
        self.obs_custom_hint_lbl.setWordWrap(True)
        self.obs_custom_hint_lbl.setTextFormat(Qt.TextFormat.RichText)
        self.obs_custom_hint_lbl.setProperty("class", "muted")
        tpl_lay.addWidget(self.obs_custom_hint_lbl)
        self._refresh_obs_template_modes()
        lay.addWidget(tpl_card)

        st_card = QFrame()
        st_card.setProperty("class", "card")
        st_lay = QVBoxLayout(st_card)
        st_lay.setContentsMargins(16, 12, 16, 12)
        st_lay.setSpacing(10)

        top_st = QHBoxLayout()
        st_hdr = QLabel(tr("ui_335", "Оформлення субтитрів у OBS:"))
        st_hdr.setProperty("class", "cardHeader")
        top_st.addWidget(st_hdr)
        top_st.addStretch(1)

        test_obs_btn = QPushButton(tr("ui_336", "✨ Показати тестовий субтитр в OBS"))
        test_obs_btn.setProperty("class", "primaryBtn")
        test_obs_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        test_obs_btn.clicked.connect(
            lambda: obs_server.broadcast_speech(
                tr("obs.sample_text", "Привіт глядачам! Це перевірка субтитрів в OBS ✨ Так виглядатиме ваш текст на трансляції: шрифт, розмір, кольори, обводка та анімація появи. Налаштуйте все під свій стиль!"),
                8.0,
                force=True,
            )
        )
        top_st.addWidget(test_obs_btn)
        st_lay.addLayout(top_st)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)

        grid.addWidget(QLabel(tr("ui_305", "Оформлення:")), 0, 0)
        self.obs_style_combo = QComboBox()
        self.obs_style_combo.addItem(tr("ui_306", "Тільки текст без фону (Рекомендовано)"), "text_only")
        self.obs_style_combo.addItem(tr("ui_307", "Текст на темній плашці з рамкою"), "boxed")
        cur_obs_st = config.get("obs_style", "text_only")
        self.obs_style_combo.setCurrentIndex(0 if cur_obs_st == "text_only" else 1)
        self.obs_style_combo.currentIndexChanged.connect(self._on_obs_style_changed)
        grid.addWidget(self.obs_style_combo, 0, 1)

        grid.addWidget(QLabel(tr("ui_338", "Анімація появи:")), 1, 0)
        anim_row = QHBoxLayout()
        anim_row.setContentsMargins(0, 0, 0, 0)
        anim_row.setSpacing(6)
        self.obs_anim_combo = QComboBox()
        from core.obs_animations import ENTRY_ANIMATIONS, MOTION_ANIMATIONS

        for anim in ENTRY_ANIMATIONS:
            self.obs_anim_combo.addItem(tr(anim["i18n_key"], anim["name"]), anim["id"])
        cur_anim = str(config.get("obs_anim_style", "typewriter"))
        idx_a = self.obs_anim_combo.findData(cur_anim)
        self.obs_anim_combo.setCurrentIndex(idx_a if idx_a >= 0 else 0)
        self.obs_anim_combo.currentIndexChanged.connect(self._on_obs_style_changed)
        anim_row.addWidget(self.obs_anim_combo, 1)
        self.obs_anim_settings_btn = QPushButton("⚙")
        self.obs_anim_settings_btn.setFixedSize(32, 32)
        self.obs_anim_settings_btn.setStyleSheet("padding: 0px; font-size: 15px;")
        self.obs_anim_settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.obs_anim_settings_btn.clicked.connect(lambda: self._open_obs_anim_settings("entry"))
        anim_row.addWidget(self.obs_anim_settings_btn)
        grid.addLayout(anim_row, 1, 1)

        grid.addWidget(QLabel(tr("ui_345", "Жива анімація (рух тексту):")), 2, 0)
        motion_row = QHBoxLayout()
        motion_row.setContentsMargins(0, 0, 0, 0)
        motion_row.setSpacing(6)
        self.obs_motion_combo = QComboBox()
        for mot in MOTION_ANIMATIONS:
            self.obs_motion_combo.addItem(tr(mot["i18n_key"], mot["name"]), mot["id"])
        cur_motion = str(config.get("obs_motion_effect", "none"))
        idx_m = self.obs_motion_combo.findData(cur_motion)
        self.obs_motion_combo.setCurrentIndex(idx_m if idx_m >= 0 else 0)
        self.obs_motion_combo.currentIndexChanged.connect(self._on_obs_style_changed)
        motion_row.addWidget(self.obs_motion_combo, 1)
        self.obs_motion_settings_btn = QPushButton("⚙")
        self.obs_motion_settings_btn.setFixedSize(32, 32)
        self.obs_motion_settings_btn.setStyleSheet("padding: 0px; font-size: 15px;")
        self.obs_motion_settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.obs_motion_settings_btn.clicked.connect(lambda: self._open_obs_anim_settings("motion"))
        motion_row.addWidget(self.obs_motion_settings_btn)
        grid.addLayout(motion_row, 2, 1)

        grid.addWidget(QLabel(tr("ui_353", "Шрифт:")), 3, 0)
        self.obs_font_combo = QComboBox()
        for f_name in ["Montserrat", "Inter", "Rubik", "Segoe UI", "Arial"]:
            self.obs_font_combo.addItem(f_name, f_name)
        self.obs_font_combo.setCurrentText(str(config.get("obs_font_family", "Montserrat")))
        self.obs_font_combo.currentIndexChanged.connect(self._on_obs_style_changed)
        grid.addWidget(self.obs_font_combo, 3, 1)

        grid.addWidget(QLabel(tr("ui_310", "Розмір букв:")), 4, 0)
        self.obs_size_slider = self._make_slider_row(
            grid, 4, 1, 8, 60, int(config.get("obs_font_size", 24)), " px", self._on_obs_style_changed
        )

        grid.addWidget(QLabel(tr("ui_311", "Яскравість (непрозорість) тексту:")), 5, 0)
        self.obs_txt_op_slider = self._make_slider_row(
            grid, 5, 1, 20, 100, int(config.get("obs_text_opacity", 100)), " %", self._on_obs_style_changed
        )

        grid.addWidget(QLabel(tr("ui_312", "Товщина темного контуру букв:")), 6, 0)
        self.obs_outline_slider = self._make_slider_row(
            grid, 6, 1, 0, 8, int(config.get("obs_outline_width", 3)), " px", self._on_obs_style_changed
        )

        grid.addWidget(QLabel(tr("ui_313", "Тінь під буквами:")), 7, 0)
        self.obs_shadow_slider = self._make_slider_row(
            grid, 7, 1, 0, 30, int(config.get("obs_shadow_blur", 10)), " px", self._on_obs_style_changed
        )

        grid.addWidget(QLabel(tr("ui_314", "Непрозорість фону (якщо з плашкою):")), 8, 0)
        self.obs_bg_op_slider = self._make_slider_row(
            grid, 8, 1, 0, 100, int(config.get("obs_bg_opacity", 78)), " %", self._on_obs_style_changed
        )

        grid.addWidget(QLabel(tr("obs.border_radius", "Заокруглення кутів блоку (якщо з плашкою):")), 9, 0)
        self.obs_border_radius_slider = self._make_slider_row(
            grid, 9, 1, 0, 40, int(config.get("obs_border_radius", 14)), " px", self._on_obs_style_changed
        )

        grid.addWidget(QLabel(tr("ui_354", "Символ курсора (мігання):")), 10, 0)
        self.obs_cursor_combo = QComboBox()
        self.obs_cursor_combo.addItem(tr("ui_355", "▌ (блок)"), "▌")
        self.obs_cursor_combo.addItem(tr("ui_356", "| (риска)"), "|")
        self.obs_cursor_combo.addItem(tr("ui_357", "▮ (вузький блок)"), "▮")
        self.obs_cursor_combo.addItem(tr("ui_358", "_ (підкреслення)"), "_")
        cur_cursor = str(config.get("obs_cursor_char", "▌"))
        idx_cur = self.obs_cursor_combo.findData(cur_cursor)
        if idx_cur >= 0:
            self.obs_cursor_combo.setCurrentIndex(idx_cur)
        self.obs_cursor_combo.currentIndexChanged.connect(self._on_obs_style_changed)
        grid.addWidget(self.obs_cursor_combo, 10, 1)

        st_lay.addLayout(grid)

        self.chk_obs_animate_block = QCheckBox(
            tr("obs.animate_block", "Анімувати блок разом із текстом")
        )
        self.chk_obs_animate_block.setChecked(bool(config.get("obs_animate_block", True)))
        self.chk_obs_animate_block.toggled.connect(self._on_obs_style_changed)
        st_lay.addWidget(self.chk_obs_animate_block)

        self.chk_obs_animate_motion_block = QCheckBox(
            tr("obs.animate_motion_block", "Застосовувати рух також до самого блоку")
        )
        self.chk_obs_animate_motion_block.setChecked(bool(config.get("obs_animate_motion_block", False)))
        self.chk_obs_animate_motion_block.toggled.connect(self._on_obs_style_changed)
        st_lay.addWidget(self.chk_obs_animate_motion_block)

        self.chk_obs_cursor = QCheckBox(tr("ui_359", "Показувати мигаючий курсор під час друку"))
        self.chk_obs_cursor.setChecked(bool(config.get("obs_show_cursor", False)))
        self.chk_obs_cursor.toggled.connect(self._on_obs_style_changed)
        st_lay.addWidget(self.chk_obs_cursor)

        self.chk_obs_top_fade = QCheckBox(tr("ui_360", "Плавне затухання тексту зверху (красиве розмиття при довгому тексті)"))
        self.chk_obs_top_fade.setChecked(bool(config.get("obs_top_fade", True)))
        self.chk_obs_top_fade.toggled.connect(self._on_obs_style_changed)
        st_lay.addWidget(self.chk_obs_top_fade)

        live_row = QHBoxLayout()
        live_row.setContentsMargins(0, 0, 0, 0)
        live_row.setSpacing(8)
        self.chk_obs_live = QCheckBox(tr("ui_361", "Показувати букви в OBS одразу під час набору (ще до натискання Enter)"))
        self.chk_obs_live.setToolTip(tr("obs.live_checkbox_tooltip", "Показує набраний текст в OBS ще до натискання Enter."))
        self.chk_obs_live.setChecked(bool(config.get("obs_show_live_typing", False)))
        self.chk_obs_live.toggled.connect(self._on_obs_style_changed)
        live_row.addWidget(self.chk_obs_live)
        live_row.addWidget(
            HelpButton(
                tr("obs.live_help_title", "Показ набору в OBS наживо"),
                tr(
                    "obs.live_help_body",
                    "<b>Що робить ця опція?</b><br>"
                    "Глядачі на стрімі в OBS бачать текст одразу, поки ви його друкуєте.<br><br>"
                    "• <b>Якщо галочку вимкнено:</b> субтитри в OBS з'являються лише після натискання <b>Enter</b> (під час озвучки).<br>"
                    "• <b>Показувати ввод в OBS та Enter:</b> букви видно в OBS під час друку, а після <b>Enter</b> з'являється фінальний субтитр під час озвучки.<br>"
                    "• <b>Показувати тільки ввод:</b> в OBS видно лише сам набір тексту; після натискання <b>Enter</b> або <b>Esc</b> рядок плавно зникає і не показується повторно.<br><br>"
                    "<i>Оберіть потрібний режим у списку під галочкою.</i>",
                ),
            )
        )
        live_row.addStretch(1)
        st_lay.addLayout(live_row)

        live_mode_row = QHBoxLayout()
        live_mode_row.setContentsMargins(26, 0, 0, 0)
        live_mode_row.setSpacing(10)
        self.obs_live_mode_lbl = QLabel(tr("obs.live_mode_label", "Режим:"))
        self.obs_live_mode_lbl.setVisible(self.chk_obs_live.isChecked())
        live_mode_row.addWidget(self.obs_live_mode_lbl)

        self.obs_live_mode_combo = QComboBox()
        self.obs_live_mode_combo.addItem(tr("obs.live_mode_both", "Показувати ввод в OBS та Enter"), "typing_and_enter")
        self.obs_live_mode_combo.addItem(tr("obs.live_mode_only", "Показувати тільки ввод"), "typing_only")
        cur_live_mode = str(config.get("obs_live_typing_mode", "typing_only"))
        idx_lm = self.obs_live_mode_combo.findData(cur_live_mode)
        self.obs_live_mode_combo.setCurrentIndex(idx_lm if idx_lm >= 0 else 0)
        self.obs_live_mode_combo.setVisible(self.chk_obs_live.isChecked())
        self.obs_live_mode_combo.currentIndexChanged.connect(self._on_obs_style_changed)
        live_mode_row.addWidget(self.obs_live_mode_combo, 1)
        st_lay.addLayout(live_mode_row)

        c_row = QHBoxLayout()
        b_col1 = QPushButton(tr("ui_317", "🎨 Колір тексту"))
        b_col1.setCursor(Qt.CursorShape.PointingHandCursor)
        b_col1.clicked.connect(lambda: self._pick_color("obs_text_color", is_obs=True))
        b_col2 = QPushButton(tr("ui_318", "🎨 Колір плашки"))
        b_col2.setCursor(Qt.CursorShape.PointingHandCursor)
        b_col2.clicked.connect(lambda: self._pick_color("obs_bg_color", is_obs=True))
        b_col3 = QPushButton(tr("ui_319", "🎨 Колір рамки"))
        b_col3.setCursor(Qt.CursorShape.PointingHandCursor)
        b_col3.clicked.connect(lambda: self._pick_color("obs_border_color", is_obs=True))
        b_col4 = QPushButton(tr("ui_362", "🎨 Колір курсора"))
        b_col4.setCursor(Qt.CursorShape.PointingHandCursor)
        b_col4.clicked.connect(lambda: self._pick_color("obs_cursor_color", is_obs=True))
        c_row.addWidget(b_col1)
        c_row.addWidget(b_col2)
        c_row.addWidget(b_col3)
        c_row.addWidget(b_col4)
        c_row.addStretch(1)

        reset_obs_btn = QPushButton(tr("ui_320", "↺ Скинути до стандартних"))
        reset_obs_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reset_obs_btn.clicked.connect(self._on_reset_obs_defaults)
        c_row.addWidget(reset_obs_btn)
        st_lay.addLayout(c_row)

        lay.addWidget(st_card)
        lay.addStretch(1)
        scroll.setWidget(content)
        return scroll

    def _on_obs_enabled_toggled(self, checked: bool) -> None:
        config.set("obs_enabled", checked)
        if checked:
            obs_server.start(int(config.get("obs_port", 8765)))
        else:
            obs_server.stop()

    def _on_obs_restart_port(self) -> None:
        new_port = int(self.obs_port_spin.value())
        config.set("obs_port", new_port)
        obs_server.restart(new_port)

    def _refresh_obs_template_modes(self) -> None:
        from core.obs_server import list_custom_obs_templates

        if not hasattr(self, "obs_template_mode_combo"):
            return
        cur_mode = str(config.get("obs_template_mode", "builtin") or "builtin").strip()
        self.obs_template_mode_combo.blockSignals(True)
        self.obs_template_mode_combo.clear()
        self.obs_template_mode_combo.addItem(tr("obs.tpl.builtin", "🎨 Вбудовані стилі SilentVoice"), "builtin")
        self.obs_template_mode_combo.addItem(tr("obs.tpl.custom", "🛠️ Власний шаблон (HTML + CSS)"), "custom")
        for entry in list_custom_obs_templates():
            self.obs_template_mode_combo.addItem(entry["label"], entry["mode"])
        idx = self.obs_template_mode_combo.findData(cur_mode)
        if idx < 0:
            cur_mode = "builtin"
            idx = 0
        self.obs_template_mode_combo.setCurrentIndex(idx)
        self.obs_template_mode_combo.blockSignals(False)
        self._update_obs_template_controls_state(cur_mode)

    def _update_obs_template_controls_state(self, mode: str) -> None:
        from core.config import OBS_TEMPLATES_DIR
        from core.obs_server import BLAITE_CUSTOM_TEST_NAME

        is_custom = mode == "custom" or mode.startswith("custom:")
        self.obs_custom_hint_lbl.setVisible(is_custom)
        self.obs_tpl_reload_btn.setVisible(is_custom)
        can_reset = mode in ("custom", f"custom:{BLAITE_CUSTOM_TEST_NAME}")
        self.obs_tpl_reset_btn.setVisible(can_reset)
        if mode.startswith("custom:"):
            stem = mode.split(":", 1)[1].strip()
            html_name, css_name = f"{stem}.html", f"{stem}.css"
        else:
            html_name, css_name = "index.html", "style.css"
        self.obs_custom_hint_lbl.setText(
            tr(
                "obs.tpl.hint_named",
                "Редагуйте <b>{html}</b> (<code>/ws</code> і <code>{css_tag}</code>) та <b>{css}</b> у папці:<br><code>{path}</code><br>Можна також додати будь-які власні <b>Назва.html</b> + <b>Назва.css</b> (знак <code>_</code> у списку замінюється на пробіл) і натиснути <b>«🔄 Оновити OBS»</b>.",
                html=html_name,
                css=css_name,
                css_tag="{{CSS}}",
                path=str(OBS_TEMPLATES_DIR),
            )
        )

    def _on_obs_template_mode_changed(self) -> None:
        from core.obs_server import ensure_custom_obs_templates

        mode = str(self.obs_template_mode_combo.currentData() or "builtin")
        if mode == "custom" or mode.startswith("custom:"):
            ensure_custom_obs_templates()
        config.set("obs_template_mode", mode)
        self._update_obs_template_controls_state(mode)
        obs_server.broadcast_reload()

    def _open_obs_templates_folder(self) -> None:
        from core.obs_server import ensure_custom_obs_templates

        tpl_dir = ensure_custom_obs_templates()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(tpl_dir)))

    def _reload_obs_custom_template(self) -> None:
        self._refresh_obs_template_modes()
        obs_server.broadcast_reload()
        self.live_status_label.setText(tr("obs.tpl.reloaded", "Шаблон OBS оновлено!"))

    def _reset_obs_custom_template(self) -> None:
        from core.obs_server import BLAITE_CUSTOM_TEST_NAME, ensure_custom_obs_templates

        mode = str(self.obs_template_mode_combo.currentData() or "custom")
        is_blaite = mode == f"custom:{BLAITE_CUSTOM_TEST_NAME}"
        html_name = f"{BLAITE_CUSTOM_TEST_NAME}.html" if is_blaite else "index.html"
        css_name = f"{BLAITE_CUSTOM_TEST_NAME}.css" if is_blaite else "style.css"
        ans = QMessageBox.question(
            self,
            tr("common.confirm", "Підтвердження"),
            tr(
                "obs.tpl.reset_confirm_named",
                "Перезаписати {html} та {css} початковим шаблоном?",
                html=html_name,
                css=css_name,
            ),
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        ensure_custom_obs_templates(
            overwrite=True,
            reset_name=BLAITE_CUSTOM_TEST_NAME if is_blaite else "index",
        )
        obs_server.broadcast_reload()
        self.live_status_label.setText(tr("obs.tpl.reset_done", "Власний шаблон скинуто до стартового."))

    def _copy_obs_url(self) -> None:
        cb = QGuiApplication.clipboard()
        if cb:
            cb.setText(obs_server.browser_source_url)
            self.live_status_label.setText(tr("ui_364", "Посилання OBS скопійовано!"))

    def _open_obs_browser_preview(self) -> None:
        if not self.chk_obs_enabled.isChecked():
            self.chk_obs_enabled.setChecked(True)
        url = obs_server.browser_source_url
        if not url:
            port = int(config.get("obs_port", 8765))
            obs_server.start(port)
            url = f"http://127.0.0.1:{port}/"
        sep = "&" if "?" in url else "?"
        QDesktopServices.openUrl(QUrl(f"{url}{sep}preview=1"))

    def _open_obs_anim_settings(self, kind: str) -> None:
        anim_id = str(
            (self.obs_anim_combo.currentData() if kind == "entry" else self.obs_motion_combo.currentData())
            or ("typewriter" if kind == "entry" else "none")
        )
        prev = getattr(self, "_obs_anim_dlg", None)
        if prev is not None:
            try:
                prev.close()
            except Exception:
                pass
        dlg = ObsAnimationSettingsDialog(
            kind=kind,
            anim_id=anim_id,
            on_saved=obs_server.broadcast_style,
            parent=self,
        )
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.destroyed.connect(lambda: setattr(self, "_obs_anim_dlg", None))
        self._obs_anim_dlg = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _on_obs_style_changed(self) -> None:
        obs_style = str(self.obs_style_combo.currentData() or "text_only")
        config.update({
            "obs_style": obs_style,
            "obs_bg_enabled": obs_style == "boxed",
            "obs_anim_style": self.obs_anim_combo.currentData(),
            "obs_animate_block": self.chk_obs_animate_block.isChecked(),
            "obs_animate_motion_block": self.chk_obs_animate_motion_block.isChecked(),
            "obs_motion_effect": str(self.obs_motion_combo.currentData() or "none"),
            "obs_font_family": self.obs_font_combo.currentData(),
            "obs_font_size": self.obs_size_slider.value(),
            "obs_text_opacity": self.obs_txt_op_slider.value(),
            "obs_outline_width": self.obs_outline_slider.value(),
            "obs_shadow_blur": self.obs_shadow_slider.value(),
            "obs_shadow_enabled": self.obs_shadow_slider.value() > 0,
            "obs_bg_opacity": self.obs_bg_op_slider.value(),
            "obs_border_radius": self.obs_border_radius_slider.value(),
            "obs_show_live_typing": self.chk_obs_live.isChecked(),
            "obs_live_typing_mode": str(self.obs_live_mode_combo.currentData() or "typing_only"),
            "obs_show_cursor": self.chk_obs_cursor.isChecked(),
            "obs_cursor_char": str(self.obs_cursor_combo.currentData() or "▌"),
            "obs_top_fade": self.chk_obs_top_fade.isChecked(),
        })
        if hasattr(self, "obs_live_mode_combo"):
            live_on = self.chk_obs_live.isChecked()
            self.obs_live_mode_combo.setVisible(live_on)
            self.obs_live_mode_lbl.setVisible(live_on)
            if not live_on:
                obs_server.broadcast_clear_live_typing()
        obs_server.broadcast_style()

    def _on_reset_obs_defaults(self) -> None:
        obs_colors = {
            "obs_text_color": "#FFFFFF",
            "obs_bg_color": "#13151F",
            "obs_border_color": "#7C5CFF",
            "obs_cursor_color": "#9B82FF",
        }
        reset_colors = self._ask_reset_with_colors(obs_colors)
        if reset_colors is None:
            return
        upd: Dict[str, Any] = {
            "obs_template_mode": "builtin",
            "obs_style": "text_only",
            "obs_bg_enabled": False,
            "obs_anim_style": "typewriter",
            "obs_animate_block": True,
            "obs_animate_motion_block": False,
            "obs_motion_effect": "none",
            "obs_font_family": "Montserrat",
            "obs_font_size": 24,
            "obs_text_opacity": 100,
            "obs_outline_width": 3,
            "obs_shadow_blur": 10,
            "obs_shadow_enabled": True,
            "obs_bg_opacity": 78,
            "obs_border_radius": 14,
            "obs_show_live_typing": False,
            "obs_live_typing_mode": "typing_only",
            "obs_show_cursor": False,
            "obs_cursor_char": "▌",
            "obs_top_fade": True,
        }
        if reset_colors:
            upd.update(obs_colors)
        config.update(upd)
        if hasattr(self, "obs_template_mode_combo"):
            prev_mode = str(self.obs_template_mode_combo.currentData() or "builtin")
            self.obs_template_mode_combo.blockSignals(True)
            self.obs_template_mode_combo.setCurrentIndex(0)
            self.obs_template_mode_combo.blockSignals(False)
            self._update_obs_template_controls_state("builtin")
            if prev_mode == "custom" or prev_mode.startswith("custom:"):
                obs_server.broadcast_reload()
        self.obs_style_combo.blockSignals(True)
        self.obs_style_combo.setCurrentIndex(0)
        self.obs_style_combo.blockSignals(False)

        self.obs_anim_combo.blockSignals(True)
        self.obs_anim_combo.setCurrentIndex(0)
        self.obs_anim_combo.blockSignals(False)

        self.obs_motion_combo.blockSignals(True)
        self.obs_motion_combo.setCurrentIndex(0)
        self.obs_motion_combo.blockSignals(False)

        self.chk_obs_animate_block.blockSignals(True)
        self.chk_obs_animate_block.setChecked(True)
        self.chk_obs_animate_block.blockSignals(False)

        self.chk_obs_animate_motion_block.blockSignals(True)
        self.chk_obs_animate_motion_block.setChecked(False)
        self.chk_obs_animate_motion_block.blockSignals(False)

        self.obs_font_combo.blockSignals(True)
        self.obs_font_combo.setCurrentText("Montserrat")
        self.obs_font_combo.blockSignals(False)

        self._set_slider_silently(self.obs_size_slider, 24)
        self._set_slider_silently(self.obs_txt_op_slider, 100)
        self._set_slider_silently(self.obs_outline_slider, 3)
        self._set_slider_silently(self.obs_shadow_slider, 10)
        self._set_slider_silently(self.obs_bg_op_slider, 78)
        self._set_slider_silently(self.obs_border_radius_slider, 14)

        self.chk_obs_live.blockSignals(True)
        self.chk_obs_live.setChecked(False)
        self.chk_obs_live.blockSignals(False)

        self.obs_live_mode_combo.blockSignals(True)
        self.obs_live_mode_combo.setCurrentIndex(self.obs_live_mode_combo.findData("typing_only"))
        self.obs_live_mode_combo.setVisible(False)
        self.obs_live_mode_lbl.setVisible(False)
        self.obs_live_mode_combo.blockSignals(False)
        obs_server.broadcast_clear_live_typing()

        self.obs_cursor_combo.blockSignals(True)
        self.obs_cursor_combo.setCurrentIndex(0)
        self.obs_cursor_combo.blockSignals(False)

        self.chk_obs_cursor.blockSignals(True)
        self.chk_obs_cursor.setChecked(False)
        self.chk_obs_cursor.blockSignals(False)

        self.chk_obs_top_fade.blockSignals(True)
        self.chk_obs_top_fade.setChecked(True)
        self.chk_obs_top_fade.blockSignals(False)

        obs_server.broadcast_style()

    # ──────────────────────────────────────────────────────────────────
    # РОЗДІЛ 6: Звук та Система (Куди йде голос + Віртуальний мікрофон)
    # ──────────────────────────────────────────────────────────────────

    def _build_page_audio(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content.setObjectName("scrollContent")
        lay = QVBoxLayout(content)
        lay.setContentsMargins(2, 2, 6, 6)
        lay.setSpacing(10)

        lay.addWidget(
            make_header_with_help(
                tr("ui_365", "🔊 Налаштування звуку та системи"),
                tr("ui_366", "Як зробити так, щоб інші люди чули вашу озвучку?"),
                tr(
                    "ui_367",
                    "<b>Як працює звук у програмі:</b><br>"
                    "Щоб голос чули у Discord, іграх чи дзвінках, звук подається через віртуальний перехідник <b>VB-Cable</b>.<br><br>"
                    "• <b>Канал 1 (Для інших):</b> сюди йде голос для співрозмовників (<code>CABLE Input</code>). У Discord чи грі оберіть мікрофоном <code>CABLE Output</code>.<br>"
                    "• <b>Канал 2 (Для себе):</b> сюди звук іде у ваші навушники чи колонки, щоб ви самі чули озвучку.",
                ),
            )
        )

        # Картка VB-Cable
        vb_card = QFrame()
        vb_card.setProperty("class", "cardHighlight")
        vb_lay = QVBoxLayout(vb_card)
        vb_lay.setContentsMargins(16, 12, 16, 12)
        vb_lay.setSpacing(8)

        vb_row = QHBoxLayout()
        self.vbcable_status_lbl = QLabel()
        self.vbcable_status_lbl.setProperty("class", "cardHeader")
        vb_row.addWidget(self.vbcable_status_lbl, 1)

        vb_row.addWidget(
            HelpButton(
                tr("ui_368", "Що таке VB-Cable (Віртуальний мікрофон)?"),
                tr(
                    "ui_369",
                    "<b>Простими словами:</b><br>"
                    "Windows не дозволяє одній програмі напряму «говорити» у ваш мікрофон.<br><br>"
                    "• <b>VB-Cable</b> створює віртуальний мікрофон.<br>"
                    "• SilentVoice передає озвучку в нього, а Discord, Skype чи ігри бачать це як ваш звичайний мікрофон.",
                ),
            )
        )

        self.install_vb_btn = QPushButton(tr("ui_370", "⚡ Встановити віртуальний мікрофон (1 клік)"))
        self.install_vb_btn.setProperty("class", "primaryBtn")
        self.install_vb_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.install_vb_btn.setToolTip(tr("ui_371", "Автоматично скачати та встановити безкоштовний драйвер VB-Cable"))
        self.install_vb_btn.clicked.connect(self._on_install_vbcable)
        vb_row.addWidget(self.install_vb_btn)
        vb_lay.addLayout(vb_row)
        lay.addWidget(vb_card)

        ffmpeg_sys_card = self._create_ffmpeg_notice_widget(scope="system", show_when_installed=True)
        if ffmpeg_sys_card is not None:
            lay.addWidget(ffmpeg_sys_card)

        # Картка каналів звуку
        out_card = QFrame()
        out_card.setProperty("class", "card")
        out_lay = QVBoxLayout(out_card)
        out_lay.setContentsMargins(16, 12, 16, 12)
        out_lay.setSpacing(10)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)

        section_hdr_style = "font-weight: 600; color: #F1F5F9; font-size: 13.5px;"

        def _make_divider() -> QFrame:
            line = QFrame()
            line.setFixedHeight(1)
            line.setStyleSheet("background-color: #323B52; margin-top: 2px; margin-bottom: 2px;")
            return line

        # --- Блок 1: Канал 1 (Для інших) ---
        ch1_hdr_row = QHBoxLayout()
        ch1_hdr = QLabel(tr("ui_374", "🎙️ Канал 1 — Для інших (Віртуальний мікрофон)"))
        ch1_hdr.setStyleSheet(section_hdr_style)
        ch1_hdr_row.addWidget(ch1_hdr)
        ch1_hdr_row.addWidget(
            HelpButton(
                tr("ui_374", "Канал 1 — Для інших (Віртуальний мікрофон)"),
                tr(
                    "ui_375",
                    "Основний вивід голосу програми для співрозмовників у Discord, дзвінках або іграх.\n"
                    "Оберіть CABLE Input (VB-Audio Virtual Cable), щоб інші чули озвучку, або «Вимкнено (Off)».",
                ),
            )
        )
        ch1_hdr_row.addStretch(1)
        grid.addLayout(ch1_hdr_row, 0, 0, 1, 2)

        ch1_dev_lbl = QLabel(tr("ui_376", "Пристрій для інших:"))
        ch1_dev_lbl.setToolTip(tr("ui_377", "Пристрій, у який програма передає озвучку для інших людей (зазвичай CABLE Input)"))
        grid.addWidget(ch1_dev_lbl, 1, 0)
        self.primary_dev_combo = QComboBox()
        self.primary_dev_combo.setToolTip(tr("ui_378", "Оберіть CABLE Input (VB-Audio Virtual Cable), щоб звук ішов у віртуальний мікрофон, або Off"))
        self.primary_dev_combo.currentIndexChanged.connect(self._on_audio_devices_changed)
        grid.addWidget(self.primary_dev_combo, 1, 1)

        ch1_vol_lbl = QLabel(tr("ui_379", "Гучність для інших (%):"))
        grid.addWidget(ch1_vol_lbl, 2, 0)
        self.prim_vol_slider = self._make_slider_row(
            grid, 2, 1, 0, 200, int(config.get("primary_volume", 100)), " %", self._on_audio_devices_changed
        )

        grid.addWidget(_make_divider(), 3, 0, 1, 2)

        # --- Блок 2: Канал 2 (Для себе) ---
        self.chk_secondary = QCheckBox(tr("ui_382", "🎧 Канал 2 — Також слухати озвучку у своїх навушниках / колонках"))
        self.chk_secondary.setStyleSheet(section_hdr_style)
        self.chk_secondary.setChecked(bool(config.get("secondary_enabled", True)))
        self.chk_secondary.toggled.connect(self._on_audio_devices_changed)
        grid.addWidget(self.chk_secondary, 4, 0, 1, 2)

        ch2_dev_lbl = QLabel(tr("ui_384", "Ваші навушники або колонки:"))
        grid.addWidget(ch2_dev_lbl, 5, 0)
        self.secondary_dev_combo = QComboBox()
        self.secondary_dev_combo.currentIndexChanged.connect(self._on_audio_devices_changed)
        grid.addWidget(self.secondary_dev_combo, 5, 1)

        ch2_vol_lbl = QLabel(tr("ui_387", "Гучність для себе (%):"))
        grid.addWidget(ch2_vol_lbl, 6, 0)
        self.sec_vol_slider = self._make_slider_row(
            grid, 6, 1, 0, 200, int(config.get("secondary_volume", 85)), " %", self._on_audio_devices_changed
        )

        grid.addWidget(_make_divider(), 7, 0, 1, 2)

        # --- )Блок 3: Мікшування фізичного мікрофона ---
        mic_hdr = QHBoxLayout()
        mic_hdr.setSpacing(6)
        self.chk_mic_passthrough = QCheckBox(tr("ui_390", "🎤 Мікшувати мій мікрофон у Канал 1 (для Discord / ігор)"))
        self.chk_mic_passthrough.setStyleSheet(section_hdr_style)
        self.chk_mic_passthrough.setChecked(bool(config.get("mic_passthrough_enabled", False)))
        self.chk_mic_passthrough.toggled.connect(self._on_audio_devices_changed)
        mic_hdr.addWidget(self.chk_mic_passthrough)
        mic_hdr.addWidget(
            HelpButton(
                tr("ui_392", "Навіщо мікшувати свій мікрофон у Канал 1?"),
                tr(
                    "ui_393",
                    "<b>Коли це потрібно?</b><br>"
                    "Коли у Discord чи грі вибрано мікрофон <code>CABLE Output</code>, співрозмовники за замовчуванням чують лише голос програми.<br><br>"
                    "• <b>Коли увімкнено:</b> програма в реальному часі додає до каналу ще й ваш <b>справжній голос із фізичного мікрофона</b>.<br>"
                    "• У результаті інші чують <b>і вас, і TTS-озвучку одночасно</b> через один мікрофон.<br><br>"
                    "<i>Якщо ви взагалі не говорите у фізичний мікрофон — залишайте вимкненим.</i>",
                ),
            )
        )
        mic_hdr.addStretch(1)
        grid.addLayout(mic_hdr, 8, 0, 1, 2)

        mic_dev_lbl = QLabel(tr("ui_394", "Ваш мікрофон:"))
        grid.addWidget(mic_dev_lbl, 9, 0)
        self.mic_dev_combo = QComboBox()
        self.mic_dev_combo.currentIndexChanged.connect(self._on_audio_devices_changed)
        grid.addWidget(self.mic_dev_combo, 9, 1)

        mic_vol_lbl = QLabel(tr("ui_397", "Гучність мікрофона (%):"))
        grid.addWidget(mic_vol_lbl, 10, 0)
        self.mic_vol_slider = self._make_slider_row(
            grid, 10, 1, 0, 200, int(config.get("mic_passthrough_volume", 100)), " %", self._on_audio_devices_changed
        )

        out_lay.addLayout(grid)
        lay.addWidget(out_card)

        # Optional network output (collapsed by default).
        from core.vban import VbanHealthMonitor, send_test as send_vban_test

        if getattr(self, "_vban_health_monitor", None) is None:
            self._vban_health_monitor = VbanHealthMonitor(
                on_change=lambda st, rsn, dtl: self.vban_health_changed.emit(st, rsn, dtl),
            )
        vban_card = QFrame()
        vban_card.setProperty("class", "card")
        vban_lay = QVBoxLayout(vban_card)
        vban_lay.setContentsMargins(16, 10, 16, 10)
        vban_lay.setSpacing(8)
        vban_hdr_row = QHBoxLayout()
        vban_hdr_row.setContentsMargins(0, 0, 0, 0)
        self.vban_expand_btn = QToolButton()
        self.vban_expand_btn.setCheckable(True)
        self.vban_expand_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.vban_expand_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.vban_expand_btn.setArrowType(Qt.ArrowType.RightArrow)
        self.vban_expand_btn.setText(tr("audio.vban.title", "Передача звуку на інший пристрій (VBAN)"))
        self.vban_expand_btn.setStyleSheet(
            "QToolButton { border: 0; background: transparent; color: #F1F5F9; "
            "font-size: 14px; font-weight: 700; padding: 4px; text-align: left; }"
            "QToolButton:hover { color: #A78BFA; }"
        )
        vban_hdr_row.addWidget(self.vban_expand_btn)
        vban_hdr_row.addStretch(1)
        self.vban_health_badge = QLabel()
        self.vban_health_badge.setStyleSheet("font-size: 12px; color: #AAB4C8;")
        self.vban_health_badge.setVisible(False)
        vban_hdr_row.addWidget(self.vban_health_badge)
        vban_lay.addLayout(vban_hdr_row)
        self.vban_body = QWidget()
        vban_body_lay = QVBoxLayout(self.vban_body)
        vban_body_lay.setContentsMargins(0, 2, 0, 0)
        vban_body_lay.setSpacing(8)
        self.vban_enable_chk = QCheckBox(tr("audio.vban.enable", "Надсилати голос через локальну мережу"))
        self.vban_enable_chk.setChecked(bool(config.get("vban_enabled", False)))
        vban_body_lay.addWidget(self.vban_enable_chk)
        vban_help = QLabel(tr("audio.vban.help", "На іншому пристрої потрібен VBAN-приймач. Обидва пристрої мають бути в одній Wi-Fi або локальній мережі."))
        vban_help.setWordWrap(True)
        vban_help.setStyleSheet("color: #AAB4C8;")
        vban_body_lay.addWidget(vban_help)
        vban_form = QGridLayout()
        vban_form.setColumnStretch(1, 1)
        vban_form.addWidget(QLabel(tr("audio.vban.address", "IP пристрою-приймача:")), 0, 0)
        self.vban_address_edit = QLineEdit(str(config.get("vban_address", "")))
        self.vban_address_edit.setPlaceholderText("192.168.1.20")
        vban_form.addWidget(self.vban_address_edit, 0, 1)
        vban_form.addWidget(QLabel(tr("audio.vban.port", "Порт:")), 1, 0)
        self.vban_port_spin = QtSpinBox()
        self.vban_port_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.vban_port_spin.setRange(1, 65535)
        self.vban_port_spin.setValue(int(config.get("vban_port", 6980) or 6980))
        vban_form.addWidget(self.vban_port_spin, 1, 1)
        vban_form.addWidget(QLabel(tr("audio.vban.stream", "Назва потоку:")), 2, 0)
        saved_vban_name = str(config.get("vban_stream_name", "SilentVoice") or "SilentVoice")
        self.vban_stream_edit = QLineEdit(saved_vban_name)
        self.vban_stream_edit.setMaxLength(16)
        vban_form.addWidget(self.vban_stream_edit, 2, 1)
        vban_form.addWidget(QLabel(tr("audio.vban.volume", "Гучність VBAN (%):")), 3, 0)
        self.vban_vol_slider = self._make_slider_row(
            vban_form, 3, 1, 0, 200, int(config.get("vban_volume", 100)), " %", lambda: None
        )
        vban_body_lay.addLayout(vban_form)
        vban_actions = QHBoxLayout()
        self.vban_test_btn = QPushButton(tr("audio.vban.test", "Надіслати тестовий звук"))
        self.vban_stop_btn = QPushButton(tr("audio.vban.stop", "Зупинити тест"))
        self.vban_stop_btn.setVisible(False)
        self.vban_ping_btn = QPushButton(tr("audio.vban.ping", "Перевірити зв'язок"))
        self.vban_status_lbl = QLabel()
        self.vban_status_lbl.setWordWrap(True)
        vban_actions.addWidget(self.vban_test_btn)
        vban_actions.addWidget(self.vban_stop_btn)
        vban_actions.addWidget(self.vban_ping_btn)
        vban_actions.addWidget(self.vban_status_lbl, 1)
        vban_body_lay.addLayout(vban_actions)
        self.vban_body.setVisible(False)
        vban_lay.addWidget(self.vban_body)

        def _apply_vban_health_ui(state: str, reason: str, detail: str) -> None:
            enabled = self.vban_enable_chk.isChecked() and bool(self.vban_address_edit.text().strip())
            if not enabled:
                self.vban_health_badge.setVisible(False)
                self.vban_health_badge.setText("")
                return
            self.vban_health_badge.setVisible(True)
            if state == "alive":
                text = (
                    tr("audio.vban.health_ok", "🟢 Приймач доступний ({detail})", detail=detail)
                    if detail
                    else tr("audio.vban.health_ok_simple", "🟢 Приймач доступний")
                )
                color = "#4ADE80"
            elif state == "failed" and reason == "local_socket":
                text = tr("audio.vban.health_local_port", "🟡 Порт 6980 зайнятий локально")
                color = "#FACC15"
            elif state == "failed":
                text = tr("audio.vban.health_fail", "🔴 Приймач не відповідає")
                color = "#F87171"
            else:
                text = tr("audio.vban.health_unknown", "⚪ Перевірка приймача...")
                color = "#AAB4C8"
            self.vban_health_badge.setText(text)
            self.vban_health_badge.setStyleSheet(f"font-size: 12px; color: {color};")

        self._reconnect_signal(self.vban_health_changed, "_slot_vban_health", _apply_vban_health_ui)

        def _sync_vban_health_monitor() -> None:
            enabled = self.vban_enable_chk.isChecked()
            address = self.vban_address_edit.text().strip()
            port = self.vban_port_spin.value()
            if enabled and address:
                try:
                    self._vban_health_monitor.set_target(address, port, interval_secs=15.0)
                    st, rsn, dtl = self._vban_health_monitor.state
                    _apply_vban_health_ui(st, rsn, dtl)
                except ValueError:
                    self._vban_health_monitor.set_target(None)
                    self.vban_health_badge.setVisible(False)
            else:
                self._vban_health_monitor.set_target(None)
                self.vban_health_badge.setVisible(False)

        def _toggle_vban_body(checked: bool) -> None:
            toggle_collapsible_widget(self.vban_expand_btn, self.vban_body, checked)
            if not checked and self._vban_test_stop_event is not None:
                self._vban_test_stop_event.set()
        self.vban_expand_btn.toggled.connect(_toggle_vban_body)

        def _save_vban_settings(*_args: Any) -> None:
            config.update({
                "vban_enabled": self.vban_enable_chk.isChecked(),
                "vban_address": self.vban_address_edit.text().strip(),
                "vban_port": self.vban_port_spin.value(),
                "vban_stream_name": self.vban_stream_edit.text().strip() or "SilentVoice",
                "vban_volume": self.vban_vol_slider.value(),
            })
            _sync_vban_health_monitor()
        self.vban_enable_chk.toggled.connect(_save_vban_settings)
        self.vban_address_edit.editingFinished.connect(_save_vban_settings)
        self.vban_port_spin.valueChanged.connect(_save_vban_settings)
        self.vban_stream_edit.editingFinished.connect(_save_vban_settings)
        self.vban_vol_slider.valueChanged.connect(_save_vban_settings)
        _sync_vban_health_monitor()

        prev_stop_ev = getattr(self, "_vban_test_stop_event", None)
        if prev_stop_ev is not None:
            prev_stop_ev.set()
        self._vban_test_stop_event: Optional[threading.Event] = None
        def _stop_vban_test() -> None:
            if self._vban_test_stop_event is not None:
                self._vban_test_stop_event.set()
                self.vban_stop_btn.setEnabled(False)
        self.vban_stop_btn.clicked.connect(_stop_vban_test)

        def _on_vban_test_finished(message: str) -> None:
            self.vban_status_lbl.setText(message)
            self.vban_test_btn.setEnabled(True)
            self.vban_stop_btn.setEnabled(True)
            self.vban_stop_btn.setVisible(False)
            self._vban_test_stop_event = None

        self._reconnect_signal(self.vban_test_finished, "_slot_vban_test", _on_vban_test_finished)

        def _send_vban_test() -> None:
            _save_vban_settings()
            address = self.vban_address_edit.text().strip()
            port = self.vban_port_spin.value()
            stream_name = self.vban_stream_edit.text().strip() or "SilentVoice"
            stop_event = threading.Event()
            self._vban_test_stop_event = stop_event
            self.vban_test_btn.setEnabled(False)
            self.vban_stop_btn.setEnabled(True)
            self.vban_stop_btn.setVisible(True)
            self.vban_status_lbl.setStyleSheet("")
            self.vban_status_lbl.setText(tr("audio.vban.sending", "Тестовий сигнал передається. Натисніть Connect у приймачі."))
            def _worker() -> None:
                try:
                    send_vban_test(address, port, stream_name, stop_event)
                    message = tr("audio.vban.stopped", "Тест зупинено.")
                except Exception as exc:
                    message = tr("audio.vban.failed", "Не вдалося надіслати: {error}", error=str(exc))
                self.vban_test_finished.emit(message)
            threading.Thread(target=_worker, daemon=True).start()
        self.vban_test_btn.clicked.connect(_send_vban_test)

        def _on_vban_ping_finished(message: str, color: str) -> None:
            self.vban_status_lbl.setText(message)
            self.vban_status_lbl.setStyleSheet(f"color: {color};" if color else "")
            self.vban_ping_btn.setEnabled(True)

        self._reconnect_signal(self.vban_ping_finished, "_slot_vban_ping", _on_vban_ping_finished)

        def _ping_vban() -> None:
            _save_vban_settings()
            address = self.vban_address_edit.text().strip()
            port = self.vban_port_spin.value()
            self.vban_ping_btn.setEnabled(False)
            self.vban_status_lbl.setStyleSheet("")
            self.vban_status_lbl.setText(tr("audio.vban.pinging", "Перевірка приймача..."))
            def _worker() -> None:
                try:
                    status, reply, err_detail = self._vban_health_monitor.probe_once_sync(address, port, timeout=1.0)
                    if status == "alive":
                        detail = (
                            f"{reply.application_name} {reply.version_str}".strip()
                            if reply is not None and reply.application_name
                            else f"{address}:{port}"
                        )
                        msg = tr("audio.vban.ping_ok", "Приймач відповів: {detail}", detail=detail)
                        col = "#4ADE80"
                    elif status == "local_socket":
                        msg = tr("audio.vban.ping_local_port", "Локальний UDP-порт 6980 зайнятий (можливо, приймач запущено на цьому ж ПК).")
                        col = "#FACC15"
                    else:
                        msg = tr("audio.vban.ping_fail", "Приймач не відповів на PING0 (перевірте IP, порт {port} та брандмауер).", port=port)
                        col = "#F87171"
                except Exception as exc:
                    msg = tr("audio.vban.failed", "Не вдалося надіслати: {error}", error=str(exc))
                    col = "#F87171"
                self.vban_ping_finished.emit(msg, col)
            threading.Thread(target=_worker, daemon=True).start()
        self.vban_ping_btn.clicked.connect(_ping_vban)
        lay.addWidget(vban_card)

        # Системні налаштування
        sys_card = QFrame()
        sys_card.setProperty("class", "card")
        sys_lay = QVBoxLayout(sys_card)
        sys_lay.setContentsMargins(16, 12, 16, 12)
        sys_lay.setSpacing(8)

        sys_hdr = QHBoxLayout()
        sys_hdr.addWidget(QLabel(tr("ui_400", "<b>⚙️ Робота програми у Windows:</b>")))
        sys_hdr.addWidget(
            HelpButton(
                tr("ui_401", "Робота у фоновому режимі та права Адміністратора"),
                tr(
                    "ui_402",
                    "<b>Дві важливі системні функції:</b><br><br>"
                    "• <b>Згортання в трей:</b> натискання <code>[✕]</code> не закриває програму, а ховає її біля годинника Windows — гарячі клавіші продовжують працювати.<br>"
                    "• <b>Права Адміністратора:</b> деякі ігри запускаються з підвищеними правами й блокують клавіші звичайних програм. Якщо набір не спрацьовує в грі — перезапустіть програму від імені Адміністратора кнопкою нижче.",
                ),
            )
        )
        sys_hdr.addStretch(1)
        sys_lay.addLayout(sys_hdr)

        ui_lang_row = QHBoxLayout()
        ui_lang_row.setSpacing(8)
        ui_lang_lbl = QLabel(tr("sys.ui_lang_label", "🌐 Мова інтерфейсу:"))
        ui_lang_lbl.setStyleSheet("color: #CBD5E1; font-size: 13px; font-weight: 600;")
        ui_lang_row.addWidget(ui_lang_lbl)

        self.ui_lang_combo = QComboBox()
        self.ui_lang_combo.setMinimumWidth(190)
        for loc_item in list_available_ui_languages():
            self.ui_lang_combo.addItem(loc_item["label"], loc_item["code"])
        cfg_ui_lang = str(config.get("ui_language", "") or "").strip().lower()
        effective_ui = cfg_ui_lang or get_ui_language()
        ui_idx = self.ui_lang_combo.findData(effective_ui)
        self.ui_lang_combo.setCurrentIndex(ui_idx if ui_idx >= 0 else 0)
        self.ui_lang_combo.currentIndexChanged.connect(self._on_ui_lang_changed)
        ui_lang_row.addWidget(self.ui_lang_combo)

        open_loc_btn = QPushButton(tr("sys.open_locales_btn", "📂 Папка перекладів"))
        open_loc_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_loc_btn.clicked.connect(self._open_locales_folder)
        ui_lang_row.addWidget(open_loc_btn)
        ui_lang_row.addStretch(1)
        sys_lay.addLayout(ui_lang_row)

        self.chk_autostart = QCheckBox(tr("ui_403", "Запускати автоматично при ввімкненні комп'ютера"))
        self.chk_autostart.setChecked(is_autostart_enabled())
        self.chk_autostart.toggled.connect(self._on_autostart_toggled)
        sys_lay.addWidget(self.chk_autostart)

        self.chk_close_to_tray = QCheckBox(tr("ui_404", "При закритті вікна [✕] залишати програму працювати біля годинника (у треї)"))
        self.chk_close_to_tray.setChecked(bool(config.get("minimize_to_tray_on_close", True)))
        self.chk_close_to_tray.toggled.connect(lambda v: config.set("minimize_to_tray_on_close", v))
        sys_lay.addWidget(self.chk_close_to_tray)

        self.chk_start_min = QCheckBox(tr("ui_405", "При запуску одразу ховати вікно в трей"))
        self.chk_start_min.setChecked(bool(config.get("start_minimized", False)))
        self.chk_start_min.toggled.connect(lambda v: config.set("start_minimized", v))
        sys_lay.addWidget(self.chk_start_min)

        self.chk_run_as_admin = QCheckBox(tr("ui_406", "Завжди запускати від імені Адміністратора (потрібно для деяких ігор)"))
        self.chk_run_as_admin.setChecked(bool(config.get("run_as_admin", True)))
        self.chk_run_as_admin.toggled.connect(self._on_run_as_admin_toggled)
        sys_lay.addWidget(self.chk_run_as_admin)

        admin_row = QHBoxLayout()
        admin_ok = is_running_as_admin()
        admin_lbl = QLabel(
            tr("ui_407", "✅ Запущено з правами Адміністратора")
            if admin_ok
            else tr("ui_408", "ℹ️ Якщо гарячі клавіші не спрацьовують у якійсь програмі — перезапустіть з правами Адміністратора:")
        )
        admin_lbl.setWordWrap(True)
        admin_row.addWidget(admin_lbl, 1)

        if not admin_ok:
            admin_btn = QPushButton(tr("ui_409", "🛡️ Перезапустити як Адміністратор"))
            admin_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            admin_btn.clicked.connect(self._on_restart_admin)
            admin_row.addWidget(admin_btn)
        sys_lay.addLayout(admin_row)

        lay.addWidget(sys_card)

        transfer_card = QFrame()
        transfer_card.setProperty("class", "card")
        transfer_lay = QVBoxLayout(transfer_card)
        transfer_lay.setContentsMargins(16, 12, 16, 12)
        transfer_lay.setSpacing(8)
        transfer_lay.addWidget(QLabel(f"<b>{tr('transfer.section.title', '📦 Перенесення налаштувань')}</b>"))
        transfer_lay.addWidget(QLabel(tr(
            "transfer.section.description",
            "Створи профіль, щоб перенести вибрані налаштування на інший комп’ютер або відновити їх із резервної копії.",
        )))
        transfer_buttons = QHBoxLayout()
        export_settings_btn = QPushButton(tr("transfer.export.open", "📤 Експортувати налаштування"))
        export_settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        export_settings_btn.clicked.connect(self._export_settings_profile)
        import_settings_btn = QPushButton(tr("transfer.import.open", "📥 Імпортувати налаштування"))
        import_settings_btn.setProperty("class", "primaryBtn")
        import_settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        import_settings_btn.clicked.connect(self._import_settings_profile)
        transfer_buttons.addWidget(export_settings_btn)
        transfer_buttons.addWidget(import_settings_btn)
        transfer_buttons.addStretch(1)
        transfer_lay.addLayout(transfer_buttons)
        lay.addWidget(transfer_card)

        from core.updater import APP_VERSION

        update_card = QFrame()
        update_card.setProperty("class", "card")
        update_lay = QVBoxLayout(update_card)
        update_lay.setContentsMargins(16, 12, 16, 12)
        update_lay.setSpacing(8)

        update_top = QHBoxLayout()
        update_top.addWidget(QLabel(f"<b>{tr('update.section.title', '🔄 Оновлення програми')}</b>"))
        update_top.addStretch(1)
        ver_badge = QLabel(tr("update.current_version", "Поточна версія: <b>v{version}</b>", version=APP_VERSION))
        ver_badge.setStyleSheet("color: #A5B4FC; font-size: 12.5px;")
        update_top.addWidget(ver_badge)
        update_lay.addLayout(update_top)

        update_row = QHBoxLayout()
        self.chk_auto_updates = QCheckBox(
            tr("update.auto_check", "Автоматично перевіряти наявність оновлень при запуску")
        )
        self.chk_auto_updates.setChecked(bool(config.get("auto_check_updates", True)))
        self.chk_auto_updates.toggled.connect(lambda v: config.set("auto_check_updates", bool(v)))
        update_row.addWidget(self.chk_auto_updates, 1)

        check_upd_btn = QPushButton(tr("update.check_btn", "🔄 Перевірити оновлення"))
        check_upd_btn.setProperty("class", "primaryBtn")
        check_upd_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        check_upd_btn.clicked.connect(lambda: self._open_update_dialog())
        update_row.addWidget(check_upd_btn)
        update_lay.addLayout(update_row)
        lay.addWidget(update_card)

        credits_card = QFrame()
        credits_card.setProperty("class", "cardHighlight")
        credits_lay = QVBoxLayout(credits_card)
        credits_lay.setContentsMargins(16, 13, 16, 13)
        credits_lay.setSpacing(10)

        credits_title = QLabel(tr("audio.credits.title", "✨ Створення та тестування"))
        credits_title.setProperty("class", "cardHeader")
        credits_lay.addWidget(credits_title)

        credit_pill_style = (
            "background-color: #202638; color: #EEF2FF; border: 1px solid #5B6396; "
            "border-radius: 9px; padding: 8px 13px; font-size: 13.5px;"
        )
        creator_lbl = QLabel(
            f"✍️ {tr('audio.credits.created_by', 'Створив')}: "
            f"<a href=\"blaite_easter_egg\" style=\"color: #EEF2FF; text-decoration: none;\"><b>Blaite</b></a>"
        )
        creator_lbl.setStyleSheet(credit_pill_style)
        creator_lbl.setTextFormat(Qt.TextFormat.RichText)
        creator_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse)
        creator_lbl.setOpenExternalLinks(False)
        creator_lbl.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        creator_lbl.linkHovered.connect(
            lambda link, lbl=creator_lbl: lbl.setCursor(
                Qt.CursorShape.PointingHandCursor if link else Qt.CursorShape.ArrowCursor
            )
        )
        creator_lbl.linkActivated.connect(lambda _href="": self._open_blaite_easter_egg())
        self._creator_credit_lbl = creator_lbl
        tester_lbl = QLabel(
            f"🧪 {tr('audio.credits.tested_by', 'Протестував')}: <b>DedProger</b>"
        )
        tester_lbl.setStyleSheet(credit_pill_style)
        inspired_lbl = QLabel(
            f"💡 {tr('audio.credits.inspired_by', 'Надихнула на створення')}: <b>alexxa69419</b>"
        )
        inspired_lbl.setStyleSheet(credit_pill_style)
        credits_people = QWidget()
        credits_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        credits_policy.setHeightForWidth(True)
        credits_people.setSizePolicy(credits_policy)
        credits_people_layout = CreditsFlowLayout(credits_people)
        credits_people_layout.setSpacing(10)
        credits_people_layout.addWidget(creator_lbl)
        credits_people_layout.addWidget(tester_lbl)
        credits_people_layout.addWidget(inspired_lbl)
        credits_lay.addWidget(credits_people)
        lay.addWidget(credits_card)

        lay.addStretch(1)
        scroll.setWidget(content)

        self._last_core_audio_devices: Tuple[Tuple[str, ...], Tuple[str, ...]] = get_core_audio_active_all_device_names()
        self._populate_audio_devices()
        if not hasattr(self, "_audio_dev_poll_timer"):
            self._audio_dev_poll_timer = QTimer(self)
            self._audio_dev_poll_timer.setInterval(2500)
            self._audio_dev_poll_timer.timeout.connect(self._poll_audio_device_changes)
            self._audio_dev_poll_timer.start()
        return scroll

    def _poll_audio_device_changes(self) -> None:
        """Автоматично підхоплює підключення/відключення навушників і мікрофонів, коли відкрита вкладка звуку."""
        if not self.isVisible() or not hasattr(self, "pages") or self.pages.currentIndex() != 5:
            return
        if audio_router.is_playing:
            return
        live = get_core_audio_active_all_device_names()
        if (live[0] or live[1]) and live != getattr(self, "_last_core_audio_devices", ((), ())):
            self._last_core_audio_devices = live
            self._populate_audio_devices(refresh_backend=True)

    def _open_blaite_easter_egg(self) -> None:
        """Відкриває пасхалку «Eye -> Heart» при натисканні на нік Blaite."""
        from ui.easter_egg import BlaiteEasterEggDialog

        dlg = BlaiteEasterEggDialog(self)
        dlg.exec()

    def _open_update_dialog(self, initial_release: Optional[Any] = None) -> None:
        dlg = AppUpdateDialog(
            initial_release=initial_release,
            on_quit_for_install=self.force_quit_app,
            parent=self,
        )
        dlg.exec()

    def _auto_check_for_updates_on_startup(self) -> None:
        if not bool(config.get("auto_check_updates", True)):
            return
        from core.updater import check_for_updates, has_configured_update_sources

        if not has_configured_update_sources():
            return

        def _worker() -> None:
            try:
                rel, is_newer = check_for_updates(timeout=8.0)
                if is_newer and rel is not None:
                    self.startup_update_found.emit(rel)
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True, name="StartupUpdateCheck").start()

    def _on_startup_update_found(self, release: object) -> None:
        if release is not None:
            self._open_update_dialog(initial_release=release)

    def _check_post_update_notice(self) -> None:
        target_ver = str(config.get("pending_update_target_version", "") or "").strip()
        if not target_ver:
            return
        config.set("pending_update_target_version", "")
        from core.updater import APP_VERSION, is_newer_version

        if is_newer_version(target_ver, APP_VERSION):
            return
        try:
            self.live_status_label.setText(
                tr("update.completed_status", "✅ Оновлено до версії v{version}", version=APP_VERSION)
            )
        except Exception:
            pass
        try:
            if getattr(self, "tray", None) is not None and self.tray.isVisible():
                self.tray.showMessage(
                    "SilentVoice TTS",
                    tr(
                        "update.completed_toast",
                        "Оновлення успішно завершено! Встановлено версію v{version}.",
                        version=APP_VERSION,
                    ),
                    QSystemTrayIcon.MessageIcon.Information,
                    6000,
                )
        except Exception:
            pass

    def _export_settings_profile(self) -> None:
        dialog = SettingsTransferDialog(list(TRANSFER_CATEGORIES), mode="export", parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        categories = dialog.selected_categories()
        include_api_keys = dialog.includes_api_keys()
        if not categories:
            QMessageBox.information(
                self,
                tr("transfer.export.title", "Експорт налаштувань"),
                tr("transfer.select_category", "Вибери хоча б один розділ."),
            )
            return
        if include_api_keys:
            answer = QMessageBox.warning(
                self,
                tr("transfer.api_keys.export_title", "Додати API-ключі до файлу?"),
                tr("transfer.api_keys.export_warning", "У файлі будуть API-ключі. Будь-хто, хто матиме доступ до файлу, зможе їх скопіювати та використовувати для доступу до ваших сервісів. Продовжити лише якщо довіряєш одержувачу?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        path, _ = QFileDialog.getSaveFileName(
            self,
            tr("transfer.export.file_dialog_title", "Save settings profile"),
            "SilentVoice_settings.svprofile",
            tr("transfer.export.file_filter", "SilentVoice profile (*.svprofile);;ZIP archive (*.zip)"),
        )
        if not path:
            return
        try:
            payload = build_settings_export(config.data, categories, include_api_keys)
            write_settings_export(path, payload)
        except Exception as exc:
            QMessageBox.critical(self, tr("transfer.export.error_title", "Export failed"), str(exc))
            return
        QMessageBox.information(
            self,
            tr("transfer.export.success_title", "Export complete"),
            tr("transfer.export.success_message", "Settings profile saved to:\n{path}", path=path),
        )

    def _import_settings_profile(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr("transfer.import.file_dialog_title", "Open settings profile"),
            "",
            tr("transfer.import.file_filter", "SilentVoice profile (*.svprofile *.zip);;Older JSON file (*.json)"),
        )
        if not path:
            return
        try:
            payload = read_settings_import(path)
        except Exception as exc:
            QMessageBox.critical(self, tr("transfer.import.open_error_title", "Could not open file"), str(exc))
            return

        available = [key for key in TRANSFER_CATEGORIES if key in payload.get("categories", {})]
        has_api_keys = bool(payload.get("contains_api_keys", False))
        dialog = SettingsTransferDialog(
            available,
            include_api_keys_available=has_api_keys,
            mode="import",
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        categories = dialog.selected_categories()
        include_api_keys = dialog.includes_api_keys()
        if not categories and not include_api_keys:
            QMessageBox.information(
                self,
                tr("transfer.import.title", "Імпорт налаштувань"),
                tr("transfer.select_category", "Вибери хоча б один розділ."),
            )
            return
        names = [SettingsTransferDialog.category_label(key) for key in categories]
        if include_api_keys:
            names.append(tr("transfer.api_keys.short", "API-ключі"))
        answer = QMessageBox.question(
            self,
            tr("transfer.import.confirm_title", "Застосувати налаштування?"),
            tr("transfer.import.confirm_body", "Ці розділи буде замінено даними з файлу:\n\n{sections}\n\nІнші розділи залишаться без змін. Продовжити?", sections="\n".join(f"• {name}" for name in names)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if include_api_keys:
            answer = QMessageBox.warning(
                self,
                tr("transfer.api_keys.import_title", "Імпорт API-ключів"),
                tr("transfer.api_keys.import_warning", "Файл містить API-ключі. Після імпорту програма зможе використовувати їх для запитів до відповідних сервісів. Продовжити лише якщо довіряєш цьому файлу?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        try:
            applied = apply_settings_import(payload, categories, include_api_keys)
        except Exception as exc:
            QMessageBox.critical(self, tr("transfer.import.apply_error_title", "Could not apply settings"), str(exc))
            return
        if not applied:
            QMessageBox.information(
                self,
                tr("transfer.import.title", "Import settings"),
                tr("transfer.import.empty", "The file has no selected sections to import."),
            )
            return

        answer = QMessageBox.information(
            self,
            tr("transfer.import.done_title", "Settings imported"),
            tr("transfer.import.done_message", "The profile was saved. Restart SilentVoice to apply all imported settings, including voices, OBS, and keyboard shortcuts.\n\nRestart now?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._restart_after_settings_import()

    def _restart_after_settings_import(self) -> None:
        app = QApplication.instance()
        release_lock = app.property("_release_instance_lock") if app is not None else None
        if callable(release_lock):
            try:
                release_lock()
            except Exception:
                pass
        if getattr(sys, "frozen", False):
            command = [sys.executable, *sys.argv[1:]]
        else:
            command = [sys.executable, os.path.abspath(sys.argv[0]), *sys.argv[1:]]
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            subprocess.Popen(command, cwd=os.getcwd(), close_fds=True, creationflags=flags)
        except Exception as exc:
            QMessageBox.warning(self, tr("transfer.restart.error_title", "Could not restart"), str(exc))
            return
        self.force_quit_app()

    def _open_locales_folder(self) -> None:
        loc_dir = get_locales_dir()
        loc_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(loc_dir))
        except Exception as exc:
            QMessageBox.warning(self, tr("common.error", "Помилка"), str(exc))

    def _schedule_live_ui_language(self, new_lang: str) -> bool:
        """Зберігає поточний стан інтерфейсу і планує живу перебудову UI на новій мові."""
        norm_lang = str(new_lang or "").strip().lower() or "uk"
        if norm_lang == get_ui_language():
            return False

        state = {
            "old_test_default": tr("ui_120", "Привіт! Тепер цей голос говоритиме замість мене."),
            "old_ready_status": tr("ui_113", "Система готова"),
            "old_disabled": tr("ui_442", "Вимкнено"),
            "test_text": self.test_voice_input.text() if hasattr(self, "test_voice_input") else "",
            "live_status": self.live_status_label.text() if hasattr(self, "live_status_label") else "",
            "obs_url": self.obs_url_edit.text() if hasattr(self, "obs_url_edit") else "",
            "page_idx": self.pages.currentIndex() if hasattr(self, "pages") else 0,
            "tts_subtab": self.tts_tabs.currentIndex() if hasattr(self, "tts_tabs") else 0,
            "typing_active": bool(self.keyboard_hook.is_typing),
        }
        cur_w = self.pages.currentWidget() if hasattr(self, "pages") else None
        state["scroll_val"] = cur_w.verticalScrollBar().value() if isinstance(cur_w, QScrollArea) else 0

        set_ui_language(norm_lang)
        try:
            # Оновлюємо мову вже відкритої сторінки субтитрів OBS / попереднього перегляду в браузері
            obs_server.broadcast_style()
        except Exception:
            pass
        # Відкладаємо перебудову на наступний тік циклу подій, щоб поточний сигнал QComboBox коректно завершився
        QTimer.singleShot(0, lambda s=state: self._apply_ui_language_live(s))
        return True

    def _on_ui_lang_changed(self) -> None:
        from core.omnivoice_tags import sync_default_reaction_commands_for_lang

        raw_choice = self.ui_lang_combo.currentData()
        selected_lang = str(raw_choice or "").strip().lower()
        target_lang = selected_lang or "en"
        config.set("ui_language", target_lang)
        synced_rc = sync_default_reaction_commands_for_lang(config.data, target_lang)
        if synced_rc is not None:
            config.set("reaction_commands", synced_rc)
            if hasattr(self, "keyboard_hook"):
                self.keyboard_hook.update_from_config(config.data)
        if target_lang == get_ui_language():
            return
        self._schedule_live_ui_language(target_lang)

    def _apply_ui_language_live(self, state: Dict[str, Any]) -> None:
        """Перебудовує інтерфейс на новій мові на льоту без перезапуску програми."""
        self.setUpdatesEnabled(False)
        try:
            for timer in list(self._api_debounce_timers.values()) + list(self._ai_debounce_timers.values()):
                timer.stop()
                timer.deleteLater()

            self._engine_cards_ui.clear()
            self._ffmpeg_ui_bars.clear()
            self._api_debounce_timers.clear()
            self._online_status_labels.clear()
            self._ai_debounce_timers.clear()
            self._ai_status_labels.clear()
            self._ai_model_combos.clear()
            self._ollama_model_buttons.clear()
            self._ollama_install_buttons.clear()
            self._ai_test_buttons.clear()
            self._ollama_download_widgets.clear()
            self._socks5_checkboxes.clear()
            self._socks5_status_labels.clear()
            self._socks5_url_edits.clear()
            self._socks5_refresh_btns.clear()
            self._primary_dsp_speed_slider = None
            self._primary_dsp_speed_label = None

            self.setWindowTitle(tr("ui_084", "SilentVoice — Озвучення тексту для спілкування на ПК"))
            self._build_ui()

            if hasattr(self, "tts_tabs"):
                sub_idx = int(state.get("tts_subtab", 0))
                if 0 <= sub_idx < self.tts_tabs.count():
                    self.tts_tabs.setCurrentIndex(sub_idx)

            page_idx = int(state.get("page_idx", 0))
            if hasattr(self, "pages") and 0 <= page_idx < self.pages.count():
                self.switch_page(page_idx)

            test_text = str(state.get("test_text", ""))
            if hasattr(self, "test_voice_input") and test_text and test_text not in self._all_default_test_phrases():
                self.test_voice_input.setText(test_text)

            live_status = str(state.get("live_status", ""))
            if hasattr(self, "live_status_label") and live_status and live_status != state.get("old_ready_status", ""):
                self.live_status_label.setText(live_status)

            if hasattr(self, "obs_url_edit") and state.get("obs_url") == state.get("old_disabled"):
                self.obs_url_edit.setText(tr("ui_442", "Вимкнено"))

            self._update_typing_sidebar_btn(bool(state.get("typing_active", False)))
            self._refresh_status_banners()
            self._rebuild_tray_menu()
            self.overlay.retranslate_ui()
            self.fallback_window.retranslate_ui()
        finally:
            self.setUpdatesEnabled(True)

        scroll_val = int(state.get("scroll_val", 0))
        if scroll_val > 0 and hasattr(self, "pages"):
            def _restore_scroll() -> None:
                w = self.pages.currentWidget()
                if isinstance(w, QScrollArea):
                    w.verticalScrollBar().setValue(scroll_val)

            QTimer.singleShot(0, _restore_scroll)

        if bool(config.get("use_socks5_proxy", False)):
            self._start_socks5_probe()

    # ──────────────────────────────────────────────────────────────────
    # Допоміжні методи синхронізації
    # ──────────────────────────────────────────────────────────────────

    def _select_engine_and_voice(self, engine_id: str, voice_id: str) -> None:
        config.update({"active_engine": engine_id, "active_voice": voice_id})
        self._sync_tts_page_selectors()
        self._refresh_status_banners()
        tts_manager.schedule_smart_warmup(0.2, "engine_changed")
        QTimer.singleShot(0, lambda e=engine_id, v=voice_id: self._maybe_prompt_ffmpeg_install(e, v, from_voice_switch=True))

    def _sync_tts_page_selectors(self) -> None:
        if not hasattr(self, "tts_engine_combo"):
            return

        available = tts_manager.get_available_active_engines()
        self.tts_engine_combo.blockSignals(True)
        self.tts_voice_combo.blockSignals(True)

        self.tts_engine_combo.clear()
        self.tts_voice_combo.clear()

        if not available:
            config.update({"active_engine": "", "active_voice": "", "fallback_engine": "", "fallback_voice": ""})
            self.tts_engine_combo.blockSignals(False)
            self.tts_voice_combo.blockSignals(False)
            if hasattr(self, "all_prim_eng_combo"):
                self.all_prim_eng_combo.clear()
                self.all_prim_voice_combo.clear()
                self.all_fb_eng_combo.clear()
                self.all_fb_voice_combo.clear()
                self.all_fb_voice_combo.setVisible(False)
            self._rebuild_dsp_settings_cards()
            self._refresh_twitch_chat_voice_selectors()
            self._refresh_omnivoice_whisper_hotkey_availability()
            if hasattr(self, "keyboard_hook"):
                self.keyboard_hook.update_from_config(config.data)
            return

        cur_eng = str(config.get("active_engine", ""))
        eng_idx = 0
        for i, entry in enumerate(available):
            self.tts_engine_combo.addItem(entry["title"], entry["id"])
            if entry["id"] == cur_eng:
                eng_idx = i

        self.tts_engine_combo.setCurrentIndex(eng_idx)
        active_eng = str(self.tts_engine_combo.currentData() or "")

        voices = tts_manager.get_voices_for_engine(active_eng)
        cur_v = str(config.get("active_voice", ""))
        v_idx = 0
        for i, v in enumerate(voices):
            self.tts_voice_combo.add_gender_item(v["name"], v["id"], str(v.get("gender", "")))
            if v["id"] == cur_v:
                v_idx = i

        active_voice = ""
        if voices:
            self.tts_voice_combo.setCurrentIndex(v_idx)
            active_voice = str(self.tts_voice_combo.currentData() or "")

        config.update({"active_engine": active_eng, "active_voice": active_voice})
        self.tts_engine_combo.blockSignals(False)
        self.tts_voice_combo.blockSignals(False)

        # Синхронізація вкладки «Усі налаштування» (Головний TTS, Запасний TTS та картки DSP)
        if hasattr(self, "all_prim_eng_combo"):
            self.all_prim_eng_combo.blockSignals(True)
            self.all_prim_voice_combo.blockSignals(True)
            self.all_fb_eng_combo.blockSignals(True)
            self.all_fb_voice_combo.blockSignals(True)

            self.all_prim_eng_combo.clear()
            self.all_prim_voice_combo.clear()
            self.all_fb_eng_combo.clear()
            self.all_fb_voice_combo.clear()

            for entry in available:
                self.all_prim_eng_combo.addItem(entry["title"], entry["id"])
            self.all_prim_eng_combo.setCurrentIndex(eng_idx)

            for v in voices:
                self.all_prim_voice_combo.add_gender_item(v["name"], v["id"], str(v.get("gender", "")))
            if voices:
                self.all_prim_voice_combo.setCurrentIndex(v_idx)

            self.all_fb_eng_combo.addItem(tr("ui_410", "Не використовувати (Вимкнено)"), "")
            cur_fb_eng = str(config.get("fallback_engine", "")).strip()
            fb_eng_idx = 0
            for i, entry in enumerate(available, start=1):
                self.all_fb_eng_combo.addItem(entry["title"], entry["id"])
                if entry["id"] == cur_fb_eng:
                    fb_eng_idx = i
            self.all_fb_eng_combo.setCurrentIndex(fb_eng_idx)
            active_fb_eng = str(self.all_fb_eng_combo.currentData() or "")

            fb_voices = tts_manager.get_voices_for_engine(active_fb_eng) if active_fb_eng else []
            cur_fb_v = str(config.get("fallback_voice", "")).strip()
            fb_v_idx = 0
            for i, v in enumerate(fb_voices):
                self.all_fb_voice_combo.add_gender_item(v["name"], v["id"], str(v.get("gender", "")))
                if v["id"] == cur_fb_v:
                    fb_v_idx = i
            active_fb_voice = ""
            if fb_voices:
                self.all_fb_voice_combo.setCurrentIndex(fb_v_idx)
                active_fb_voice = str(self.all_fb_voice_combo.currentData() or "")

            self.all_fb_voice_combo.setVisible(bool(active_fb_eng))
            config.update({"fallback_engine": active_fb_eng, "fallback_voice": active_fb_voice})

            self.all_prim_eng_combo.blockSignals(False)
            self.all_prim_voice_combo.blockSignals(False)
            self.all_fb_eng_combo.blockSignals(False)
            self.all_fb_voice_combo.blockSignals(False)
            self._rebuild_dsp_settings_cards()

        self._refresh_twitch_chat_voice_selectors()
        self._refresh_omnivoice_whisper_hotkey_availability()
        if hasattr(self, "keyboard_hook"):
            self.keyboard_hook.update_from_config(config.data)

    def _on_tts_page_engine_changed(self) -> None:
        eng_id = str(self.tts_engine_combo.currentData() or "")
        voices = tts_manager.get_voices_for_engine(eng_id)
        v_id = voices[0]["id"] if voices else ""
        self._select_engine_and_voice(eng_id, v_id)

    def _on_tts_page_voice_changed(self) -> None:
        v_id = self.tts_voice_combo.currentData()
        if v_id:
            config.set("active_voice", str(v_id))
            if hasattr(self, "all_prim_voice_combo"):
                self.all_prim_voice_combo.blockSignals(True)
                idx = self.all_prim_voice_combo.findData(str(v_id))
                if idx >= 0:
                    self.all_prim_voice_combo.setCurrentIndex(idx)
                self.all_prim_voice_combo.blockSignals(False)
            self._refresh_status_banners()
            self._rebuild_dsp_settings_cards()
            tts_manager.schedule_smart_warmup(0.3, "voice_changed")
            cur_eng = str(config.get("active_engine", "")).strip()
            QTimer.singleShot(0, lambda e=cur_eng, v=str(v_id): self._maybe_prompt_ffmpeg_install(e, v, from_voice_switch=True))

    def _on_local_engine_action_clicked(self, engine_id: str) -> None:
        if tts_manager.is_downloading(engine_id):
            return
        if tts_manager.is_engine_installed(engine_id):
            spec = ENGINE_SPECS.get(engine_id, {})
            if QMessageBox.question(self, tr("common.confirm", "Підтвердження"), tr("confirm.delete_model", "Точно видалити модель «{name}» із комп’ютера?", name=str(spec.get("title", engine_id)))) != QMessageBox.StandardButton.Yes:
                return
            tts_manager.delete_engine_model(engine_id)
        else:
            self._start_model_download(engine_id)

    def _start_model_download(self, engine_id: str) -> None:
        if tts_manager.is_downloading(engine_id):
            return
        if not self._ensure_language_confirmed():
            return
        spec = ENGINE_SPECS.get(engine_id, {})
        if "files" in spec and not tts_manager.get_files_for_engine(engine_id):
            return
        if not bool(config.get("hide_local_model_first_run_notice", False)):
            engine_name = str(spec.get("title") or spec.get("name") or engine_id)
            dlg = LocalModelDownloadNoticeDialog(engine_name, self)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            if dlg.dont_show_again:
                config.set("hide_local_model_first_run_notice", True)
        if engine_id == "qwen3_tts":
            from core.llama_tts import should_prompt_qwen3_quant_for_vram

            vram_gb = get_discrete_gpu_vram_gb()
            if should_prompt_qwen3_quant_for_vram(vram_gb):
                q_dlg = Qwen3QuantChoiceDialog(vram_gb, self)
                if q_dlg.exec() != QDialog.DialogCode.Accepted:
                    return
                chosen_engine = str(getattr(q_dlg, "selected_engine", "qwen3_tts") or "qwen3_tts").strip()
                if chosen_engine == "zonos3_mini":
                    engine_id = "zonos3_mini"
                    if tts_manager.is_downloading(engine_id) or tts_manager.is_engine_installed(engine_id):
                        return
                else:
                    config.set("qwen3_tts_quant", "q8")
            else:
                config.set("qwen3_tts_quant", "q8")
        ui = self._engine_cards_ui.get(engine_id)
        if ui:
            ui["progress"].setVisible(True)
            ui["progress"].setValue(1)
            ui["prog_label"].setVisible(True)
            ui["prog_label"].setText(tr("ui_411", "Підготовка до встановлення..."))
            btn: QPushButton = ui["action_btn"]  # type: ignore
            c_btn = ui.get("cancel_btn")
            if isinstance(c_btn, QPushButton):
                btn.setVisible(False)
                c_btn.setEnabled(True)
                c_btn.setVisible(True)
            else:
                btn.setEnabled(False)
                btn.setText(tr("ui_173", "⏳ Завантаження..."))
        tts_manager.download_engine_model(engine_id)

    def _on_cancel_model_download_clicked(self, engine_id: str) -> None:
        ui = self._engine_cards_ui.get(engine_id)
        if ui:
            c_btn = ui.get("cancel_btn")
            if isinstance(c_btn, QPushButton):
                c_btn.setEnabled(False)
            lbl_w = ui.get("prog_label")
            if isinstance(lbl_w, QLabel):
                lbl_w.setText(tr("tts.dl.cancelling", "Скасування завантаження..."))
        tts_manager.cancel_download(engine_id)

    def _update_engine_card_state(self, engine_id: str, installed: bool) -> None:
        ui = self._engine_cards_ui.get(engine_id)
        if not ui:
            return
        ui["progress"].setVisible(False)
        ui["prog_label"].setVisible(False)
        c_btn = ui.get("cancel_btn")
        if isinstance(c_btn, QPushButton):
            c_btn.setVisible(False)
            c_btn.setEnabled(True)
        btn: QPushButton = ui["action_btn"]  # type: ignore
        btn.setVisible(True)
        if installed:
            btn.setEnabled(True)
            btn.setText(tr("ui_137", "Видалити"))
            btn.setProperty("class", "dangerBtn")
            btn.setToolTip(tr("ui_412", "Видалити цей TTS з комп'ютера"))
        elif engine_id in ("omnivoice", "qwen3_tts", "zonos3_mini") and not has_hardware_gpu():
            btn.setEnabled(False)
            btn.setText(tr("tts.no_gpu_btn", "Немає відеокарти"))
            btn.setProperty("class", "")
            btn.setToolTip(tr("tts.no_gpu_tip", "Для роботи цього рушія потрібна апаратна відеокарта (GPU)."))
        else:
            btn.setEnabled(True)
            btn.setText(tr("ui_134", "Встановити"))
            btn.setProperty("class", "primaryBtn")
            btn.setToolTip(tr("ui_413", "Завантажити та встановити цей TTS на комп'ютер"))
        btn.style().unpolish(btn)
        btn.style().polish(btn)
        if engine_id in ("omnivoice", "qwen3_tts", "zonos3_mini"):
            self._refresh_clone_card_ui(engine_id)

    def _populate_audio_devices(self, refresh_backend: bool = False) -> None:
        if refresh_backend:
            audio_router.refresh_devices()
        live_devs = get_core_audio_active_all_device_names()
        if live_devs[0] or live_devs[1]:
            self._last_core_audio_devices = live_devs
        vb_ok, vb_name = check_vbcable_installed()
        if hasattr(self, "install_vb_btn"):
            self.install_vb_btn.setVisible(not vb_ok)
        if vb_ok:
            self.vbcable_status_lbl.setText(tr("ui_f_020", "✅ Віртуальний мікрофон підключено ({vb_name})", vb_name=vb_name))
            self.vbcable_status_lbl.setStyleSheet("color: #34D399; font-weight: 700; font-size: 13px;")
        else:
            self.vbcable_status_lbl.setText(tr("ui_414", "⚠️ Віртуальний мікрофон (VB-Cable) ще не встановлено"))
            self.vbcable_status_lbl.setStyleSheet("color: #FBBF24; font-weight: 700; font-size: 13px;")

        devices = get_output_devices()
        in_devices = get_input_devices()
        cur_prim = str(config.get("primary_output_device", ""))
        cur_sec = str(config.get("secondary_output_device", ""))
        cur_mic = str(config.get("mic_passthrough_device", ""))

        self.primary_dev_combo.blockSignals(True)
        self.secondary_dev_combo.blockSignals(True)
        self.mic_dev_combo.blockSignals(True)
        self.primary_dev_combo.clear()
        self.secondary_dev_combo.clear()
        self.mic_dev_combo.clear()

        self.primary_dev_combo.addItem(tr("ui_415", "🔇 Вимкнено (Off)"), "__off__")
        self.secondary_dev_combo.addItem(tr("ui_417", "Стандартні навушники / колонки Windows"), "")

        prim_idx = 0
        sec_idx = 0
        cable_input_pos: Optional[int] = None
        cable_input_name = ""
        for pos, d in enumerate(devices, start=1):
            d_name = str(d["name"])
            d_low = d_name.lower()
            self.primary_dev_combo.addItem(d_name, d_name)
            self.secondary_dev_combo.addItem(d_name, d_name)
            if cable_input_pos is None and "cable input" in d_low:
                cable_input_pos = pos
                cable_input_name = d_name
            if cur_prim and cur_prim != "__off__" and (
                d_low == cur_prim.lower()
                or ("cable input" in cur_prim.lower() and "cable input" in d_low)
            ):
                prim_idx = pos
            if cur_sec and d_low == cur_sec.lower():
                sec_idx = pos

        # За замовчуванням обираємо CABLE Input (VB-Audio Virtual Cable);
        # якщо його немає або пристрій не знайдено — лишаємо «Вимкнено (Off)» (ніколи не стандартні динаміки).
        if cur_prim != "__off__":
            if prim_idx == 0 and not cur_prim and cable_input_pos is not None:
                prim_idx = cable_input_pos
                config.set("primary_output_device", cable_input_name)
            elif prim_idx == 0:
                config.set("primary_output_device", "__off__")

        self.primary_dev_combo.setCurrentIndex(prim_idx)
        self.secondary_dev_combo.setCurrentIndex(sec_idx)

        self.mic_dev_combo.addItem(tr("ui_418", "Стандартний мікрофон Windows"), "")
        mic_idx = 0
        for m_pos, d in enumerate(in_devices, start=1):
            d_name = str(d["name"])
            self.mic_dev_combo.addItem(d_name, d_name)
            if cur_mic and d_name.lower() == cur_mic.lower():
                mic_idx = m_pos
        self.mic_dev_combo.setCurrentIndex(mic_idx)

        prim_is_off = self.primary_dev_combo.currentData() == "__off__"
        self.prim_vol_slider.setEnabled(not prim_is_off)
        mic_active_ui = self.chk_mic_passthrough.isChecked() and not prim_is_off
        self.mic_dev_combo.setEnabled(mic_active_ui)
        self.mic_vol_slider.setEnabled(mic_active_ui)

        self.primary_dev_combo.blockSignals(False)
        self.secondary_dev_combo.blockSignals(False)
        self.mic_dev_combo.blockSignals(False)
        audio_router.sync_mic_passthrough()

    def _on_audio_devices_changed(self) -> None:
        prim_val = str(self.primary_dev_combo.currentData() or "__off__")
        prim_is_off = prim_val == "__off__"
        self.prim_vol_slider.setEnabled(not prim_is_off)
        mic_active_ui = self.chk_mic_passthrough.isChecked() and not prim_is_off
        self.mic_dev_combo.setEnabled(mic_active_ui)
        self.mic_vol_slider.setEnabled(mic_active_ui)
        config.update({
            "primary_output_device": prim_val,
            "primary_volume": self.prim_vol_slider.value(),
            "secondary_enabled": self.chk_secondary.isChecked(),
            "secondary_output_device": self.secondary_dev_combo.currentData() or "",
            "secondary_volume": self.sec_vol_slider.value(),
            "mic_passthrough_enabled": self.chk_mic_passthrough.isChecked(),
            "mic_passthrough_device": self.mic_dev_combo.currentData() or "",
            "mic_passthrough_volume": self.mic_vol_slider.value(),
        })
        audio_router.sync_mic_passthrough()

    def _on_install_vbcable(self) -> None:
        self.install_vb_btn.setEnabled(False)
        self.install_vb_btn.setText(tr("ui_173", "⏳ Завантаження..."))

        def _cb(pct: int, msg: str):
            _ = msg
            if not getattr(self, "_quit_in_progress", False):
                self.vbcable_progress.emit(int(pct))

        def _work():
            ok, msg = download_and_install_vbcable(_cb)
            if not getattr(self, "_quit_in_progress", False):
                self.vbcable_finished.emit(bool(ok), str(msg))

        threading.Thread(target=_work, daemon=True).start()

    def _on_vbcable_done(self, ok: bool, msg: str) -> None:
        self.install_vb_btn.setEnabled(True)
        self.install_vb_btn.setText(tr("ui_370", "⚡ Встановити віртуальний мікрофон (1 клік)"))
        if ok and str(config.get("primary_output_device", "")).strip() in ("", "__off__"):
            config.set("primary_output_device", "CABLE Input (VB-Audio Virtual Cable)")
        self._populate_audio_devices(refresh_backend=True)
        if ok:
            QMessageBox.information(self, tr("ui_419", "Віртуальний мікрофон VB-Cable"), msg)
        else:
            ErrorPopupDialog(msg, tr("ui_420", "Помилка встановлення VB-Cable"), self).exec()
            if hasattr(self, "chk_download_bypass"):
                self.chk_download_bypass.setChecked(bool(config.get("use_download_bypass", True)))

    def _open_proxy_settings_dialog(self) -> None:
        dlg = ProxySettingsDialog(self)
        if dlg.exec() == ProxySettingsDialog.DialogCode.Accepted:
            if hasattr(self, "chk_download_bypass"):
                self.chk_download_bypass.setChecked(bool(config.get("use_download_bypass", True)))

    def _create_socks5_bar(self) -> QFrame:
        """Компактний непомітний рядок для увімкнення SOCKS5-проксі та відображення статусу."""
        bar = QFrame()
        bar.setStyleSheet(
            "QFrame { background-color: rgba(30, 35, 52, 0.45); border: 1px solid #2D344C; border-radius: 8px; }"
        )
        top_row = QHBoxLayout(bar)
        top_row.setContentsMargins(10, 4, 10, 4)
        top_row.setSpacing(8)

        enabled = bool(config.get("use_socks5_proxy", False))
        chk = QCheckBox(tr("ui_421", "Використовувати SOCKS5"))
        chk.setChecked(enabled)
        chk.setToolTip(
            tr("ui_422", "Увімкніть, якщо OpenAI, Gemini, ElevenLabs або Groq недоступні у вашому регіоні.\n"
            "Програма направить запити через локальний мережевий сервіс або вказаний SOCKS5-проксі.")
        )
        chk.setStyleSheet("font-weight: 500; font-size: 12px; color: #CBD5E1; border: none; background: transparent;")
        top_row.addWidget(chk)
        self._socks5_checkboxes.append(chk)

        url_edit = QLineEdit(str(config.get("socks5_proxy_url", "")))
        url_edit.setPlaceholderText(tr("ui_423", "Автопошук на ПК (або ip:port)"))
        url_edit.setFixedWidth(165)
        url_edit.setFixedHeight(26)
        url_edit.setStyleSheet("font-size: 11.5px; padding: 2px 6px;")
        url_edit.setToolTip(
            tr("ui_424", "Залиште порожнім — програма знайде запущений локальний сервіс (V2Ray, Nekoray, Nekobox, Clash, Tor).\n"
            "Або впишіть свій SOCKS5 вручну, наприклад: 127.0.0.1:10808 чи логін:пароль@ip:port")
        )
        url_edit.setVisible(enabled)
        url_edit.editingFinished.connect(lambda e=url_edit: self._on_socks5_url_edited(e.text()))
        top_row.addWidget(url_edit)
        self._socks5_url_edits.append(url_edit)

        chk.toggled.connect(lambda checked, e=url_edit: self._on_socks5_toggled(checked, e.text()))

        refresh_btn = QPushButton("🔄")
        refresh_btn.setFixedSize(28, 26)
        refresh_btn.setStyleSheet("padding: 0px; font-size: 12px;")
        refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        refresh_btn.setToolTip(tr("ui_426", "Перевірити підключення до SOCKS5"))
        refresh_btn.setVisible(enabled)
        refresh_btn.clicked.connect(lambda _=False, e=url_edit: self._on_socks5_refresh_clicked(e.text()))
        top_row.addWidget(refresh_btn)
        self._socks5_refresh_btns.append(refresh_btn)

        init_status = tr("ui_429", "🟡 Пошук проксі...") if enabled else ""
        status_lbl = ElidedLabel(init_status)
        status_lbl.setToolTip(init_status)
        status_lbl.setStyleSheet(
            "color: #FBBF24; font-size: 11.5px; font-weight: 600; border: none; background: transparent;"
            if enabled
            else "color: #94A3B8; font-size: 11.5px; border: none; background: transparent;"
        )
        top_row.addWidget(status_lbl, 1)
        self._socks5_status_labels.append(status_lbl)

        top_row.addWidget(
            HelpButton(
                tr("ui_427", "Що таке SOCKS5 і коли він потрібен?"),
                tr("ui_428", "<b>Коли це потрібно?</b><br>"
                "Якщо онлайн-сервіси OpenAI, Gemini, ElevenLabs, Groq чи Fish Audio недоступні у вашому регіоні.<br><br>"
                "• <b>Якщо системне мережеве підключення вже налаштоване</b>, ця опція може бути не потрібна.<br><br>"
                "• <b>Використовуєте V2Ray / v2rayN / Hiddify / Nekoray / Nekobox / Clash / Tor у режимі локального проксі?</b> Поставте галочку <b>«Використовувати SOCKS5»</b> і залиште адресу порожньою — програма спробує знайти порт автоматично.<br><br>"
                "• <b>Маєте окремий SOCKS5-сервер?</b> Поставте галочку та впишіть адресу (наприклад: <code>127.0.0.1:10808</code> або <code>логін:пароль@ip:port</code>).<br><br>"
                "<i>Якщо все працює напряму — просто не вмикайте цю опцію.</i>"),
            )
        )

        return bar

    def _set_socks5_status_ui(self, text: str, color: str) -> None:
        for lbl in self._socks5_status_labels:
            lbl.setText(text)
            lbl.setToolTip(text)
            lbl.setStyleSheet(f"color: {color}; font-size: 11.5px; font-weight: 600; border: none; background: transparent;")

    def _sync_socks5_url_value(self, raw_text: Optional[str]) -> None:
        if raw_text is None:
            return
        val = raw_text.strip()
        if str(config.get("socks5_proxy_url", "")) != val:
            config.set("socks5_proxy_url", val)
            invalidate_socks5_cache()
        for edit in self._socks5_url_edits:
            if edit.text() != val:
                edit.blockSignals(True)
                edit.setText(val)
                edit.blockSignals(False)

    def _on_socks5_toggled(self, checked: bool, raw_url: Optional[str] = None) -> None:
        self._sync_socks5_url_value(raw_url)
        config.set("use_socks5_proxy", bool(checked))
        for chk in self._socks5_checkboxes:
            if chk.isChecked() != checked:
                chk.blockSignals(True)
                chk.setChecked(checked)
                chk.blockSignals(False)
        for edit in self._socks5_url_edits:
            edit.setVisible(bool(checked))
        for btn in self._socks5_refresh_btns:
            btn.setVisible(bool(checked))
        if not checked:
            self._set_socks5_status_ui("", "#94A3B8")
            return
        self._start_socks5_probe()

    def _on_socks5_url_edited(self, raw_text: str) -> None:
        prev = str(config.get("socks5_proxy_url", ""))
        self._sync_socks5_url_value(raw_text)
        if prev != raw_text.strip() and bool(config.get("use_socks5_proxy", False)):
            self._start_socks5_probe()

    def _on_socks5_refresh_clicked(self, raw_url: Optional[str] = None) -> None:
        self._sync_socks5_url_value(raw_url)
        invalidate_socks5_cache()
        if not bool(config.get("use_socks5_proxy", False)):
            self._on_socks5_toggled(True, raw_url)
        else:
            self._start_socks5_probe()

    def _start_socks5_probe(self) -> None:
        self._set_socks5_status_ui(tr("ui_431", "🟡 Пошук / перевірка проксі..."), "#FBBF24")
        threading.Thread(target=self._socks5_probe_worker, daemon=True).start()

    def _socks5_probe_worker(self) -> None:
        ok, msg = ensure_socks5_proxy_ready()
        self.socks5_status_changed.emit(ok, msg)

    def _on_socks5_probe_done(self, ok: bool, msg: str) -> None:
        if not bool(config.get("use_socks5_proxy", False)):
            self._set_socks5_status_ui(tr("ui_430", "Статус: вимкнено"), "#94A3B8")
            return
        if ok:
            self._set_socks5_status_ui(f"🟢 {msg}", "#4ADE80")
        else:
            self._set_socks5_status_ui(f"🔴 {msg}", "#F87171")

    def _on_autostart_toggled(self, checked: bool) -> None:
        config.set("autostart_windows", checked)
        set_windows_autostart(checked)

    def _on_run_as_admin_toggled(self, checked: bool) -> None:
        config.set("run_as_admin", checked)
        if checked and not is_running_as_admin():
            reply = QMessageBox.question(
                self,
                tr("ui_432", "Права Адміністратора"),
                tr("ui_433", "Перезапустити програму з правами Адміністратора прямо зараз?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._on_restart_admin()

    def _on_restart_admin(self) -> None:
        app = QApplication.instance()
        rel = app.property("_release_instance_lock") if app is not None else None
        if callable(rel):
            try:
                rel()
            except Exception:
                pass
        if restart_as_admin():
            self.force_quit_app()

    # ──────────────────────────────────────────────────────────────────
    # Системний трей та сигнали
    # ──────────────────────────────────────────────────────────────────

    def _setup_tray(self) -> None:
        if not hasattr(self, "tray"):
            self.tray = QSystemTrayIcon(create_app_icon(), self)
            self.tray.activated.connect(self._on_tray_activated)
        self._rebuild_tray_menu()
        self.tray.show()

    def _rebuild_tray_menu(self) -> None:
        if not hasattr(self, "tray"):
            return
        hk_toggle = str(config.get("hotkey_toggle_input", "Ctrl+Space"))
        self.tray.setToolTip(tr("ui_f_025", "SilentVoice — Озвучення тексту ({hk_toggle})", hk_toggle=hk_toggle))

        old_menu = getattr(self, "_tray_menu", None)
        menu = QMenu()
        menu.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        menu.setStyleSheet(
            "QMenu { background-color: #252A3B; color: #F8FAFC; border: 1px solid #3E4766; padding: 6px; }"
            "QMenu::item { padding: 7px 18px; border-radius: 6px; }"
            "QMenu::item:selected { background-color: #635BFF; }"
        )

        act_show = QAction(tr("ui_435", "Відкрити налаштування"), self)
        act_show.triggered.connect(self.restore_from_tray)
        menu.addAction(act_show)

        act_type = QAction(tr("ui_436", "⌨️ Почати писати на екрані"), self)
        act_type.triggered.connect(self._on_manual_toggle_typing)
        menu.addAction(act_type)

        act_repeat = QAction(tr("ui_437", "🔁 Повторити останню фразу"), self)
        act_repeat.triggered.connect(tts_manager.repeat_last)
        menu.addAction(act_repeat)

        act_stop = QAction(tr("ui_111", "⏹ Зупинити голос"), self)
        act_stop.triggered.connect(tts_manager.stop_current)
        menu.addAction(act_stop)

        menu.addSeparator()
        act_quit = QAction(tr("ui_438", "✕ Вийти з програми"), self)
        act_quit.triggered.connect(self.force_quit_app)
        menu.addAction(act_quit)

        self._tray_menu = menu
        if old_menu is not None:
            old_menu.deleteLater()

    def _show_tray_menu(self) -> None:
        menu = getattr(self, "_tray_menu", None)
        if menu is None:
            return
        menu.adjustSize()
        sz = menu.sizeHint()
        cur = QCursor.pos()
        screen = QGuiApplication.screenAt(cur) or QGuiApplication.primaryScreen()
        pos = cur
        if screen is not None:
            avail = screen.availableGeometry()
            mx = max(avail.left() + 4, min(cur.x(), avail.right() - sz.width() - 4))
            # Якщо клікнули по трею внизу/зверху біля панелі задач — показуємо меню в межах робочої області поверх панелі задач
            my = max(avail.top() + 4, min(cur.y() - sz.height(), avail.bottom() - sz.height() - 4))
            pos = QPoint(mx, my)
        menu.popup(pos)
        menu.raise_()
        menu.activateWindow()
        try:
            import ctypes
            from ctypes import wintypes

            hwnd = wintypes.HWND(int(menu.winId()))
            ctypes.windll.user32.SetWindowPos(hwnd, wintypes.HWND(-1), 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0040)
            ctypes.windll.user32.SetForegroundWindow(hwnd)
        except Exception:
            pass

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
            self.restore_from_tray()
        elif reason == QSystemTrayIcon.ActivationReason.Context:
            self._show_tray_menu()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not getattr(self, "_first_run_help_scheduled", False) and not bool(config.get("first_run_help_shown", False)):
            self._first_run_help_scheduled = True
            QTimer.singleShot(500, self._maybe_show_first_run_help)
        if not getattr(self, "_ffmpeg_startup_checked", False) and QApplication.platformName() != "offscreen":
            self._ffmpeg_startup_checked = True
            QTimer.singleShot(650, self._maybe_prompt_ffmpeg_on_startup)

    def _maybe_prompt_ffmpeg_on_startup(self) -> None:
        if is_ffmpeg_installed() or not self.isVisible():
            return
        prim_eng = str(config.get("active_engine", "") or "").strip()
        prim_voice = str(config.get("active_voice", "") or "").strip()
        if prim_eng:
            dsp = get_engine_dsp_settings(prim_eng, prim_voice)
            if bool(dsp.get("speed_enabled", False)) or bool(dsp.get("pitch_enabled", False)):
                self._maybe_prompt_ffmpeg_install(prim_eng, prim_voice, from_voice_switch=True)
                return
        fb_eng = str(config.get("fallback_engine", "") or "").strip()
        fb_voice = str(config.get("fallback_voice", "") or "").strip()
        if fb_eng:
            fb_dsp = get_engine_dsp_settings(fb_eng, fb_voice)
            if bool(fb_dsp.get("speed_enabled", False)) or bool(fb_dsp.get("pitch_enabled", False)):
                self._maybe_prompt_ffmpeg_install(fb_eng, fb_voice, from_voice_switch=True)

    def _maybe_prompt_ffmpeg_install(self, engine_id: str, voice_id: str = "", *, from_voice_switch: bool = False) -> None:
        if is_ffmpeg_installed() or tts_manager.is_downloading(FFMPEG_ENGINE_ID) or self._ffmpeg_prompt_open:
            return
        raw_eng = str(engine_id or "").split("::", 1)[-1].strip()
        if not raw_eng:
            return
        dsp = get_engine_dsp_settings(raw_eng, voice_id)
        if not (bool(dsp.get("speed_enabled", False)) or bool(dsp.get("pitch_enabled", False))):
            return
        switch_key = (raw_eng, str(voice_id or "").strip())
        if from_voice_switch and switch_key == self._ffmpeg_last_declined_voice:
            return

        self._ffmpeg_prompt_open = True
        try:
            dlg = FFmpegInstallDialog(self, voice_is_custom=str(voice_id or "").endswith("::custom"))
            res = dlg.exec()
            if res != FFmpegInstallDialog.DialogCode.Accepted and from_voice_switch:
                self._ffmpeg_last_declined_voice = switch_key
            elif res == FFmpegInstallDialog.DialogCode.Accepted:
                self._ffmpeg_last_declined_voice = None
        finally:
            self._ffmpeg_prompt_open = False

    def _maybe_show_first_run_help(self) -> None:
        """Показує підказку «Як працює ця програма?» при першому відкритті вікна."""
        if bool(config.get("first_run_help_shown", False)):
            return
        if not self.isVisible() or self.isMinimized():
            self._first_run_help_scheduled = False
            return
        self.showNormal()
        self.raise_()
        self.activateWindow()
        self.repaint()
        QApplication.processEvents()
        config.set("first_run_help_shown", True)
        btn = getattr(self, "_sb_brand_help_btn", None)
        if btn is not None:
            btn.show_help()

    def restore_from_tray(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()
        self._apply_window_capture_protection()
        self._apply_native_win_icon()

    def _save_main_window_geometry(self) -> None:
        geom = self.normalGeometry() if (self.isMinimized() or self.isMaximized()) else self.geometry()
        config.update({
            "main_window_width": int(geom.width()),
            "main_window_height": int(geom.height()),
            "main_window_x": int(geom.x()),
            "main_window_y": int(geom.y()),
        })

    def force_quit_app(self) -> None:
        if getattr(self, "_quit_in_progress", False):
            return
        self._quit_in_progress = True
        self._save_main_window_geometry()
        self._force_quit = True
        # Миттєво приховуємо всі вікна та іконку в треї, щоб програма не здавалася завислою під час зупинки движків
        for w_obj in (self, getattr(self, "overlay", None), getattr(self, "fallback_window", None), getattr(self, "tray", None)):
            if w_obj is not None:
                try:
                    w_obj.hide()
                except Exception:
                    pass
        app = QApplication.instance()
        if app is not None:
            try:
                app.processEvents()
            except Exception:
                pass
        release_lock = app.property("_release_instance_lock") if app is not None else None
        if callable(release_lock):
            try:
                release_lock()
            except Exception:
                pass
        for t_attr in ("_ollama_install_check_timer", "_lang_net_retry_timer", "_ov_preview_timer"):
            t_obj = getattr(self, t_attr, None)
            if t_obj is not None:
                try:
                    t_obj.stop()
                except Exception:
                    pass
        try:
            tts_manager.shutdown()
        except Exception:
            pass
        try:
            from core.llama_tts import shutdown_llama_tts
            shutdown_llama_tts()
        except Exception:
            pass
        stop_managed_llama_server()
        unload_saved_local_models(list(config.get("ai_items", [])))
        self.keyboard_hook.stop()
        if getattr(self, "_vban_health_monitor", None) is not None:
            self._vban_health_monitor.shutdown()
        try:
            from core.twitch_chat import twitch_chat
            twitch_chat.shutdown()
        except Exception:
            pass
        audio_router.shutdown()
        try:
            tts_manager.flush_audio_cache()
        except Exception:
            pass
        obs_server.stop()
        self.overlay.close()
        self.fallback_window.close()
        self.tray.hide()
        QApplication.quit()

    def closeEvent(self, event) -> None:
        self._save_main_window_geometry()
        if not self._force_quit and config.get("minimize_to_tray_on_close", True):
            event.ignore()
            self.hide()
            self.tray.showMessage(
                tr("ui_439", "SilentVoice продовжує працювати"),
                tr("ui_f_021", "Натисніть {hk} у будь-який момент, щоб написати та озвучити текст.", hk=config.get('hotkey_toggle_input', 'Ctrl+Space')),
                QSystemTrayIcon.MessageIcon.Information,
                2500,
            )
            return
        self.force_quit_app()
        super().closeEvent(event)

    def _connect_signals(self) -> None:
        self.keyboard_hook.typing_mode_changed.connect(self._on_typing_mode_changed)
        self.keyboard_hook.typing_text_changed.connect(self._on_typing_text_changed)
        self.keyboard_hook.send_triggered.connect(self._on_hook_send_triggered)
        self.keyboard_hook.cancel_triggered.connect(self._on_hook_cancel_triggered)
        self.keyboard_hook.stop_tts_triggered.connect(tts_manager.stop_current)
        self.keyboard_hook.repeat_last_triggered.connect(tts_manager.repeat_last)
        self.keyboard_hook.stop_chat_tts_triggered.connect(tts_manager.stop_chat_tts)
        self.keyboard_hook.toggle_chat_tts_triggered.connect(self._on_toggle_chat_tts_hotkey)
        self.keyboard_hook.toggle_game_mode_triggered.connect(self._on_toggle_game_mode_hotkey)
        self.keyboard_hook.toggle_omnivoice_whisper_triggered.connect(self._on_toggle_omnivoice_whisper_hotkey)
        self.keyboard_hook.slash_selection_changed.connect(self.overlay.set_tag_hint_selected_index)
        self.keyboard_hook.reaction_hotkey_triggered.connect(self._on_reaction_hotkey_triggered)
        self.keyboard_hook.quick_phrase_triggered.connect(self._trigger_quick_phrase_hotkey)

        self.overlay.position_changed.connect(self._on_overlay_dragged)
        self.overlay.move_mode_changed.connect(self._on_overlay_move_mode_changed)
        self.overlay.tag_picker_chosen.connect(self.keyboard_hook.insert_slash_tag)
        self.overlay.tag_picker_selection_changed.connect(self.keyboard_hook.set_slash_selected_index)
        self.overlay.tag_picker_hwnd_changed.connect(self.keyboard_hook.set_tag_picker_hwnd)

        self.fallback_window.text_submitted.connect(self._on_fallback_submitted)
        self.fallback_window.cancelled.connect(self._on_fallback_cancelled)
        self.fallback_window.open_state_changed.connect(self._on_fallback_open_state_changed)
        self.fallback_window.position_changed.connect(self._on_overlay_dragged)
        self.fallback_window.stop_requested.connect(tts_manager.stop_current)
        self.fallback_window.repeat_requested.connect(tts_manager.repeat_last)
        self.fallback_window.live_text_changed.connect(obs_server.broadcast_live_typing)

        tts_manager.speech_started.connect(self._on_speech_started)
        from core.twitch_chat import twitch_chat
        tts_manager.phrase_spoken.connect(twitch_chat.send_message)
        self.twitch_chat_event_received.connect(self._on_twitch_chat_message)
        twitch_chat.set_chat_message_callback(lambda msg: self.twitch_chat_event_received.emit(msg))
        self.twitch_moderation_received.connect(self._on_twitch_moderation)
        twitch_chat.set_moderation_callback(lambda ev, u, mid: self.twitch_moderation_received.emit(ev, u, mid))
        tts_manager.status_changed.connect(lambda s: self.live_status_label.setText(s))
        tts_manager.error_occurred.connect(self._on_tts_error)
        tts_manager.download_progress.connect(self._on_model_download_progress)
        tts_manager.download_failed.connect(self._on_model_download_failed)
        tts_manager.download_cancelled.connect(self._on_model_download_cancelled)
        tts_manager.model_status_changed.connect(self._on_model_status_changed)

        obs_server.server_status_changed.connect(self._on_obs_server_status)
        obs_server.preview_settings_saved.connect(self._on_obs_preview_settings_saved)
        obs_server.external_speak_requested.connect(self._trigger_speak)
        obs_server.external_toggle_input_requested.connect(self._on_manual_toggle_typing)
        obs_server.external_stop_requested.connect(tts_manager.stop_current)
        obs_server.external_repeat_requested.connect(tts_manager.repeat_last)
        obs_server.external_quick_phrase_requested.connect(self._on_external_quick_index)

    def _update_typing_sidebar_btn(self, active: bool) -> None:
        self._typing_btn_active = bool(active)
        if not hasattr(self, "toggle_typing_sidebar_btn"):
            return
        compact = bool(getattr(self, "_sidebar_compact", False))
        if active:
            self.toggle_typing_sidebar_btn.setText("🟢" if compact else tr("ui_440", "🟢 Завершити набір"))
            self.toggle_typing_sidebar_btn.setProperty("class", "successBtn")
        else:
            self.toggle_typing_sidebar_btn.setText("⌨️" if compact else tr("ui_109", "⌨️ Почати писати"))
            self.toggle_typing_sidebar_btn.setProperty("class", "primaryBtn")
        self.toggle_typing_sidebar_btn.style().unpolish(self.toggle_typing_sidebar_btn)
        self.toggle_typing_sidebar_btn.style().polish(self.toggle_typing_sidebar_btn)

    def _on_fallback_open_state_changed(self, is_open: bool) -> None:
        self._update_typing_sidebar_btn(is_open)
        if hasattr(self, "chk_continuous_chat"):
            self.chk_continuous_chat.blockSignals(True)
            self.chk_continuous_chat.setChecked(bool(config.get("continuous_chat_mode", False)))
            self.chk_continuous_chat.blockSignals(False)

    def _on_manual_toggle_typing(self) -> None:
        if self.keyboard_hook.uses_popup_window:
            if self.fallback_window.isVisible():
                audio_router.play_feedback_beep("cancel")
                self.fallback_window.close_and_restore_focus()
                self.keyboard_hook._is_typing = False
                self._update_typing_sidebar_btn(False)
            else:
                audio_router.play_feedback_beep("start")
                self.keyboard_hook._is_typing = True
                self.fallback_window.open_and_focus(continuous=bool(config.get("continuous_chat_mode", False)))
                self._update_typing_sidebar_btn(True)
        else:
            self.keyboard_hook.set_typing_mode(
                not self.keyboard_hook.is_typing,
                continuous=bool(config.get("continuous_chat_mode", False)),
            )

    def _on_typing_mode_changed(self, is_typing: bool, lang_code: str) -> None:
        if self.keyboard_hook.uses_popup_window:
            if is_typing:
                audio_router.play_feedback_beep("start")
                self.fallback_window.open_and_focus(continuous=self.keyboard_hook.is_continuous_active)
                self._update_typing_sidebar_btn(True)
            else:
                self.fallback_window.close_and_restore_focus()
                obs_server.broadcast_clear_live_typing()
                self._update_typing_sidebar_btn(False)
            return

        if is_typing:
            audio_router.play_feedback_beep("start")
            self.overlay.show_typing("", lang_code)
            self._update_typing_sidebar_btn(True)
        else:
            self.overlay.hide_overlay()
            obs_server.broadcast_clear_live_typing()
            self._update_typing_sidebar_btn(False)

    def _on_typing_text_changed(self, text: str, lang_code: str) -> None:
        self.overlay.show_typing(text, lang_code)
        obs_server.broadcast_live_typing(text)

    def _on_hook_send_triggered(self, text: str) -> None:
        if self.keyboard_hook.is_typing:
            # У режимі постійного спілкування (діалогу) набір залишається активним!
            self.overlay.show_typing("", self.keyboard_hook.get_current_lang_code())
            self._update_typing_sidebar_btn(True)
        else:
            self.overlay.hide_overlay()
            self._update_typing_sidebar_btn(False)
        if bool(config.get("obs_show_live_typing", False)) and str(config.get("obs_live_typing_mode", "typing_and_enter")) == "typing_only":
            obs_server.broadcast_clear_live_typing()
        self._trigger_speak(text)

    def _on_hook_cancel_triggered(self) -> None:
        audio_router.play_feedback_beep("cancel")
        self.overlay.hide_overlay()
        obs_server.broadcast_clear_live_typing()
        self._update_typing_sidebar_btn(False)

    def _on_fallback_submitted(self, text: str) -> None:
        still_open = self.fallback_window.isVisible()
        self.keyboard_hook._is_typing = still_open
        self.overlay.hide_overlay()
        self._update_typing_sidebar_btn(still_open)
        if bool(config.get("obs_show_live_typing", False)) and str(config.get("obs_live_typing_mode", "typing_and_enter")) == "typing_only":
            obs_server.broadcast_clear_live_typing()
        self._trigger_speak(text)

    def _on_fallback_cancelled(self) -> None:
        self.keyboard_hook._is_typing = False
        self.keyboard_hook._session_continuous = False
        self.overlay.hide_overlay()
        obs_server.broadcast_clear_live_typing()
        self._update_typing_sidebar_btn(False)
        audio_router.play_feedback_beep("cancel")

    def _trigger_speak(self, text: str) -> None:
        tts_manager.speak(text)
        self._refresh_history_list()

    def _trigger_quick_phrase_hotkey(self, text: str) -> None:
        tts_manager.speak_quick_text(text)
        self._refresh_history_list()

    def _on_speech_started(self, text: str, duration: float) -> None:
        # Під час самої озвучки (або тестування голосу) текст на екрані не показується
        if not self.keyboard_hook.is_typing and not self.overlay.is_move_mode:
            self.overlay.hide_overlay()

    def _on_tts_error(self, err_msg: str) -> None:
        clean_err = str(err_msg or "").strip() or tr(
            "tts.err.unknown",
            "Невідома помилка озвучення (порожнє повідомлення від рушія).",
        )
        print(f"[UI Error] {clean_err}", file=sys.stderr, flush=True)
        self.live_status_label.setText(f"⚠️ {clean_err[:55]}")
        ErrorPopupDialog(clean_err, tr("ui_441", "Помилка"), self).exec()
        if hasattr(self, "chk_download_bypass"):
            self.chk_download_bypass.setChecked(bool(config.get("use_download_bypass", True)))

    def _create_ffmpeg_notice_widget(self, scope: str, *, show_when_installed: bool = False) -> Optional[QFrame]:
        installed = is_ffmpeg_installed()
        downloading = tts_manager.is_downloading(FFMPEG_ENGINE_ID)
        if installed and not show_when_installed and not downloading:
            return None

        box = QFrame()
        box.setProperty("class", "cardHighlight" if scope == "system" else "card")
        if scope == "dsp":
            box.setStyleSheet(
                "QFrame { background-color: rgba(251, 191, 36, 0.08); border: 1px solid rgba(251, 191, 36, 0.35); border-radius: 8px; }"
            )

        lay = QVBoxLayout(box)
        lay.setContentsMargins(14 if scope == "system" else 10, 10 if scope == "system" else 8, 14 if scope == "system" else 10, 10 if scope == "system" else 8)
        lay.setSpacing(6)

        top_row = QHBoxLayout()
        top_row.setSpacing(8)
        status_lbl = QLabel()
        status_lbl.setWordWrap(True)
        top_row.addWidget(status_lbl, 1)

        top_row.addWidget(
            HelpButton(
                tr("ui.ffmpeg.help_title", "Навіщо потрібен FFmpeg?"),
                tr(
                    "ui.ffmpeg.help_body",
                    "<b>Чистий звук при зміні швидкості та тону (Pitch):</b><br>"
                    "Коли FFmpeg встановлено, програма змінює швидкість і висоту голосу через фільтр <code>asetrate/atempo</code> без металевих спотворень.<br><br>"
                    "• Якщо FFmpeg не знайдено в системі, використовується спрощений резервний алгоритм.<br>"
                    "• Натисніть <b>«Встановити FFmpeg»</b> — програма автоматично завантажить <code>ffmpeg.exe</code> у локальну папку <code>data/engines/ffmpeg</code> (~114 МБ).",
                ),
            )
        )

        install_btn = QPushButton(tr("ui.ffmpeg.install_btn", "⬇️ Встановити FFmpeg"))
        install_btn.setProperty("class", "primaryBtn")
        install_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        install_btn.setToolTip(tr("ui.ffmpeg.install_tip", "Автоматично завантажити та встановити FFmpeg для чистої зміни швидкості й тону голосу"))
        install_btn.clicked.connect(tts_manager.install_ffmpeg)
        top_row.addWidget(install_btn)

        cancel_btn = QPushButton("✕")
        cancel_btn.setProperty("class", "dangerBtn")
        cancel_btn.setFixedSize(30, 28)
        cancel_btn.setStyleSheet("padding: 0px; font-size: 13px; font-weight: 700;")
        cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel_btn.setToolTip(tr("ui.ffmpeg.cancel_tip", "Скасувати завантаження"))
        cancel_btn.clicked.connect(lambda: tts_manager.cancel_download(FFMPEG_ENGINE_ID))
        cancel_btn.setVisible(False)
        top_row.addWidget(cancel_btn)
        lay.addLayout(top_row)

        prog_bar = QProgressBar()
        prog_bar.setRange(0, 100)
        prog_bar.setFixedHeight(8)
        prog_bar.setTextVisible(False)
        prog_bar.setVisible(False)
        lay.addWidget(prog_bar)

        prog_lbl = QLabel()
        prog_lbl.setWordWrap(True)
        prog_lbl.setStyleSheet("color: #CBD5E1; font-size: 12px;")
        prog_lbl.setVisible(False)
        lay.addWidget(prog_lbl)

        entry = {
            "scope": scope,
            "box": box,
            "status_lbl": status_lbl,
            "install_btn": install_btn,
            "cancel_btn": cancel_btn,
            "progress": prog_bar,
            "prog_label": prog_lbl,
        }
        self._ffmpeg_ui_bars.append(entry)
        self._apply_ffmpeg_bar_state(entry, installed=installed, downloading=downloading)
        return box

    def _apply_ffmpeg_bar_state(
        self,
        entry: Dict[str, Any],
        *,
        installed: bool,
        downloading: bool,
        progress: Optional[Tuple[int, str]] = None,
        error_text: str = "",
        cancelled: bool = False,
    ) -> None:
        status_lbl = entry["status_lbl"]
        install_btn = entry["install_btn"]
        cancel_btn = entry["cancel_btn"]
        prog_bar = entry["progress"]
        prog_lbl = entry["prog_label"]

        if installed and not downloading:
            status_lbl.setText(tr("ui.ffmpeg.status_ok", "✅ FFmpeg встановлено (чиста зміна швидкості та тону голосу)"))
            status_lbl.setStyleSheet("color: #34D399; font-weight: 700; font-size: 13px;")
            install_btn.setVisible(False)
            cancel_btn.setVisible(False)
            prog_bar.setVisible(False)
            prog_lbl.setVisible(False)
            return

        status_lbl.setText(
            tr(
                "ui.ffmpeg.status_missing",
                "⚠️ FFmpeg не встановлено — без нього швидкість і тон (Pitch) працюють через спрощений алгоритм",
            )
        )
        status_lbl.setStyleSheet("color: #FBBF24; font-weight: 600; font-size: 12.5px;")
        install_btn.setVisible(True)

        if downloading:
            pct, msg = progress or tts_manager.get_download_progress(FFMPEG_ENGINE_ID) or (1, tr("tts.dl.ffmpeg_preparing", "Підготовка до встановлення FFmpeg..."))
            install_btn.setVisible(False)
            cancel_btn.setVisible(True)
            cancel_btn.setEnabled(True)
            prog_bar.setVisible(True)
            prog_bar.setValue(pct)
            prog_lbl.setStyleSheet("color: #CBD5E1; font-size: 12px;")
            prog_lbl.setVisible(True)
            prog_lbl.setText(wrap_long_tokens(msg))
            return
        if error_text:
            install_btn.setEnabled(True)
            install_btn.setText(tr("ui.ffmpeg.install_btn", "⬇️ Встановити FFmpeg"))
            cancel_btn.setVisible(False)
            prog_bar.setVisible(False)
            prog_lbl.setStyleSheet("color: #FCA5A5; font-size: 12px;")
            prog_lbl.setVisible(True)
            prog_lbl.setText(wrap_long_tokens(error_text))
            return
        if cancelled:
            install_btn.setEnabled(True)
            install_btn.setText(tr("ui.ffmpeg.install_btn", "⬇️ Встановити FFmpeg"))
            cancel_btn.setVisible(False)
            prog_bar.setVisible(False)
            prog_lbl.setStyleSheet("color: #CBD5E1; font-size: 12px;")
            prog_lbl.setVisible(True)
            prog_lbl.setText(tr("tts.dl.cancelled", "Завантаження скасовано"))
            return

        install_btn.setEnabled(True)
        install_btn.setText(tr("ui.ffmpeg.install_btn", "⬇️ Встановити FFmpeg"))
        cancel_btn.setVisible(False)
        prog_bar.setVisible(False)
        prog_lbl.setVisible(False)

    def _on_model_download_progress(self, engine_id: str, pct: int, label: str) -> None:
        self._download_ui_state[engine_id] = (pct, label)
        if engine_id == FFMPEG_ENGINE_ID:
            for entry in self._ffmpeg_ui_bars:
                self._apply_ffmpeg_bar_state(entry, installed=False, downloading=True, progress=(pct, label))
            self.live_status_label.setText(f"FFmpeg: {pct}%")
            return
        ui = self._engine_cards_ui.get(engine_id)
        if ui:
            ui["progress"].setVisible(True)
            ui["progress"].setValue(pct)
            if tts_manager.is_downloading(engine_id):
                btn: QPushButton = ui["action_btn"]  # type: ignore
                c_btn = ui.get("cancel_btn")
                if isinstance(c_btn, QPushButton):
                    btn.setVisible(False)
                    c_btn.setVisible(True)
                else:
                    btn.setEnabled(False)
                    btn.setText(tr("ui_173", "⏳ Завантаження..."))
            lbl_w: QLabel = ui["prog_label"]  # type: ignore
            lbl_w.setStyleSheet("color: #CBD5E1; font-size: 12px;")
            lbl_w.setVisible(True)
            lbl_w.setText(wrap_long_tokens(label))

    def _on_model_download_failed(self, engine_id: str, err_msg: str) -> None:
        self._download_ui_state.pop(engine_id, None)
        if engine_id == FFMPEG_ENGINE_ID:
            for entry in self._ffmpeg_ui_bars:
                self._apply_ffmpeg_bar_state(entry, installed=False, downloading=False, error_text=err_msg)
            return
        ui = self._engine_cards_ui.get(engine_id)
        if ui:
            ui["progress"].setVisible(False)
            c_btn = ui.get("cancel_btn")
            if isinstance(c_btn, QPushButton):
                c_btn.setVisible(False)
                c_btn.setEnabled(True)
            lbl_w: QLabel = ui["prog_label"]  # type: ignore
            lbl_w.setStyleSheet("color: #FCA5A5; font-size: 12px;")
            lbl_w.setVisible(True)
            lbl_w.setText(wrap_long_tokens(err_msg))
            btn: QPushButton = ui["action_btn"]  # type: ignore
            btn.setVisible(True)
            btn.setEnabled(True)
            btn.setText(tr("ui_134", "Встановити"))
            btn.setProperty("class", "primaryBtn")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _on_model_download_cancelled(self, engine_id: str) -> None:
        self._download_ui_state.pop(engine_id, None)
        if engine_id == FFMPEG_ENGINE_ID:
            for entry in self._ffmpeg_ui_bars:
                self._apply_ffmpeg_bar_state(entry, installed=False, downloading=False, cancelled=True)
            return
        ui = self._engine_cards_ui.get(engine_id)
        if ui:
            ui["progress"].setVisible(False)
            c_btn = ui.get("cancel_btn")
            if isinstance(c_btn, QPushButton):
                c_btn.setVisible(False)
                c_btn.setEnabled(True)
            lbl_w: QLabel = ui["prog_label"]  # type: ignore
            lbl_w.setStyleSheet("color: #CBD5E1; font-size: 12px;")
            lbl_w.setVisible(True)
            lbl_w.setText(tr("tts.dl.cancelled", "Завантаження скасовано"))
            btn: QPushButton = ui["action_btn"]  # type: ignore
            btn.setVisible(True)
            btn.setEnabled(True)
            btn.setText(tr("ui_134", "Встановити"))
            btn.setProperty("class", "primaryBtn")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _on_model_status_changed(self, engine_id: str, installed: bool) -> None:
        self._download_ui_state.pop(engine_id, None)
        if engine_id.startswith("online:"):
            self._download_ui_state.pop(engine_id.split(":", 1)[1], None)
        if engine_id == FFMPEG_ENGINE_ID:
            for entry in self._ffmpeg_ui_bars:
                if entry.get("scope") == "system":
                    self._apply_ffmpeg_bar_state(entry, installed=installed, downloading=False)
            self._rebuild_dsp_settings_cards()
            if installed:
                self.live_status_label.setText(tr("ui.ffmpeg.status_ok_short", "✅ FFmpeg встановлено"))
            return
        if engine_id.startswith("online:"):
            self._rebuild_online_cards()
        else:
            self._update_engine_card_state(engine_id, installed)
        if installed and engine_id != FFMPEG_ENGINE_ID:
            avail_ids = {str(a.get("id", "")) for a in tts_manager.get_available_active_engines()}
            if not engine_id.startswith("online:") or engine_id in avail_ids:
                config.set("active_engine", engine_id)
        self._sync_tts_page_selectors()
        if installed:
            tts_manager.schedule_smart_warmup(0.2, "post_install")

    def _on_obs_preview_settings_saved(self, saved: Dict[str, Any]) -> None:
        combo_values = (
            ("obs_style_combo", saved.get("style")),
            ("obs_anim_combo", saved.get("anim_style")),
            ("obs_motion_combo", saved.get("motion_effect")),
        )
        for attr, value in combo_values:
            combo = getattr(self, attr, None)
            if combo is None:
                continue
            index = combo.findData(value)
            if index >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(False)

        for attr, key in (
            ("chk_obs_animate_block", "animate_block"),
            ("chk_obs_animate_motion_block", "animate_motion_block"),
        ):
            checkbox = getattr(self, attr, None)
            if checkbox is not None:
                checkbox.blockSignals(True)
                checkbox.setChecked(bool(saved.get(key, False)))
                checkbox.blockSignals(False)

        dlg = getattr(self, "_obs_anim_dlg", None)
        if dlg is not None:
            dlg.close()

    def _on_obs_server_status(self, running: bool, port: int, url: str) -> None:
        if hasattr(self, "obs_url_edit"):
            self.obs_url_edit.setText(url if running else tr("ui_442", "Вимкнено"))

    def _on_external_quick_index(self, idx_1based: int) -> None:
        phrases = config.get("quick_phrases", [])
        if 1 <= idx_1based <= len(phrases):
            self._trigger_quick_phrase(phrases[idx_1based - 1])

    def _refresh_status_banners(self) -> None:
        hk_toggle = str(config.get("hotkey_toggle_input", "Ctrl+Space"))
        hk_send = str(config.get("hotkey_send", "Enter"))
        hk_cancel = str(config.get("hotkey_cancel", "Esc"))
        hk_stop = str(config.get("hotkey_stop_tts", "Ctrl+Shift+S"))

        self.sidebar_hk_label.setText(
            tr("ui_f_022", "Натисніть <b>{hk_toggle}</b>, введіть текст і натисніть <b>{hk_send}</b> (<b>{hk_cancel}</b> — відміна).", hk_toggle=hk_toggle, hk_send=hk_send, hk_cancel=hk_cancel)
        )
        if hasattr(self, "toggle_typing_sidebar_btn"):
            self.toggle_typing_sidebar_btn.setToolTip(
                tr("ui_f_023", "Вмикає режим набору тексту прямо поверх екрана ({hk_toggle}).\nНадрукуйте фразу на клавіатурі та натисніть {hk_send}, щоб озвучити її.", hk_toggle=hk_toggle, hk_send=hk_send)
            )
        if hasattr(self, "stop_sound_btn"):
            self.stop_sound_btn.setToolTip(
                tr("ui_f_024", "Миттєво перериває озвучення поточної фрази ({hk_stop}), якщо програма зараз щось говорить.", hk_stop=hk_stop)
            )
        if hasattr(self, "tray"):
            self.tray.setToolTip(tr("ui_f_025", "SilentVoice — Озвучення тексту ({hk_toggle})", hk_toggle=hk_toggle))

    def _open_donate_dialog(self) -> None:
        DonateDialog(self).exec()
