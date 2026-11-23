"""
ui/dialogs.py — Модальні вікна, підказки (!) та діалоги:
1. HelpButton — кругла іконка зі знаком оклику (!), яка при натисканні простою мовою пояснює функцію
2. VoiceSamplesDialog — вікно зі списком доступних голосів для прослуховування при натисканні «Прослухати»
3. ApiGuideDialog — покроковий гайд для конкретного онлайн-провайдера (Google Cloud TTS, ElevenLabs, OpenAI)
4. HotkeyCaptureDialog — діалог запису будь-якої гарячої клавіші
5. QuickPhraseDialog — діалог додавання / редагування швидкої фрази
"""

from __future__ import annotations

import copy
import shutil
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import QObject, QRectF, QSize, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QConicalGradient, QDesktopServices, QGuiApplication, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.ai_corrector import AI_PROVIDERS, _call_ai_item, fetch_ai_provider_models, get_ai_item_title, get_default_ai_prompt, get_effective_ai_prompt, get_last_ai_edit_result
from core.config import BUNDLE_DIR, PHRASES_AUDIO_DIR, config
from core.local_llm import (
    LLAMA_MODEL_DIR,
    download_recommended_lmstudio_model,
    download_recommended_gguf,
    list_lmstudio_models,
    start_llama_cpp_server,
    start_lmstudio_model,
)
from core.i18n import (
    get_ui_language,
    list_available_ui_languages,
    set_ui_language,
    tr,
    translate_locale_file,
    translate_ui_prompt_texts,
)
from core.tts_manager import (
    FFMPEG_ENGINE_ID,
    get_samples_for_engine,
    is_ffmpeg_installed,
    is_network_or_download_error,
    remove_custom_clone_sample,
    translate_error_message,
    tts_manager,
)
from core.win_api import KeyboardHookManager

DIALOG_STYLESHEET = """
QDialog {
    background-color: #1E2330;
    color: #F8FAFC;
    font-family: 'Segoe UI', system-ui, sans-serif;
}
QLabel {
    color: #E2E8F0;
    font-size: 13px;
}
QLabel#dialogTitle {
    color: #FFFFFF;
    font-size: 17px;
    font-weight: 700;
}
QLineEdit, QTextEdit {
    background-color: #272D3F;
    color: #FFFFFF;
    border: 1px solid #3D4663;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 14px;
}
QLineEdit:focus, QTextEdit:focus {
    border: 1px solid #7C5CFF;
}
QComboBox {
    background-color: #1B1F2D;
    color: #FFFFFF;
    border: 1px solid #3E4766;
    border-radius: 8px;
    padding: 6px 8px;
    font-size: 13px;
}
QComboBox:focus {
    border: 1px solid #7C5CFF;
}
QPushButton {
    background-color: #2E364C;
    color: #F8FAFC;
    border: 1px solid #434E6E;
    border-radius: 8px;
    padding: 7px 16px;
    font-size: 13px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #38425E;
    border-color: #7C5CFF;
}
QPushButton:pressed {
    background-color: #1B2030;
    border-color: #A5B4FC;
    padding-top: 8px;
    padding-bottom: 6px;
}
QPushButton:disabled {
    background-color: #252A38;
    color: #7C8499;
    border-color: #343B50;
}
QPushButton#primaryBtn {
    background-color: #635BFF;
    color: #FFFFFF;
    border: 1px solid #827BFF;
}
QPushButton#primaryBtn:hover {
    background-color: #5249F5;
}
QPushButton#primaryBtn:pressed {
    background-color: #453EC8;
}
QPushButton#primaryBtn:disabled {
    background-color: #343047;
    color: #8E91A7;
    border-color: #46445E;
}
QPushButton#successBtn {
    background-color: #10B981;
    color: #FFFFFF;
    border: 1px solid #34D399;
}
QPushButton#successBtn:hover {
    background-color: #059669;
}
QCheckBox {
    color: #CBD5E1;
    font-size: 13px;
    spacing: 9px;
    padding: 2px 0;
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
    border-color: #A5B4FC;
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
"""


class InfoPopupDialog(QDialog):
    """Просте та зрозуміле вікно пояснення функції при натисканні на кнопку (!)."""

    def __init__(self, title: str, explanation_html: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("dlg_f_001", "Підказка — {title}", title=title))
        self.setMinimumWidth(450)
        self.setMaximumWidth(550)
        self.setStyleSheet(DIALOG_STYLESHEET)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 20, 22, 18)
        lay.setSpacing(14)

        hdr = QLabel(f"💡 {title}")
        hdr.setObjectName("dialogTitle")
        hdr.setWordWrap(True)
        lay.addWidget(hdr)

        body = QLabel(explanation_html.replace("\n", "<br>"))
        body.setWordWrap(True)
        body.setTextFormat(Qt.TextFormat.RichText)
        body.setStyleSheet(
            "background-color: #262C3D; border: 1px solid #38415C; border-radius: 10px; "
            "padding: 14px 16px; color: #F1F5F9; font-size: 14px; line-height: 1.48;"
        )
        lay.addWidget(body)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        ok_btn = QPushButton(tr("dlg_001", "Зрозуміло"))
        ok_btn.setObjectName("primaryBtn")
        ok_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(ok_btn)
        lay.addLayout(btn_row)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.adjustSize()
        screen = QGuiApplication.primaryScreen()
        avail = screen.availableGeometry() if screen is not None else None
        p = self.parentWidget()
        if p is not None and p.isVisible() and not p.isMinimized() and p.width() > 100 and p.height() > 100:
            pg = p.frameGeometry()
            tx = pg.center().x() - self.width() // 2
            ty = pg.center().y() - self.height() // 2
        elif avail is not None:
            tx = avail.center().x() - self.width() // 2
            ty = avail.center().y() - self.height() // 2
        else:
            return
        if avail is not None:
            tx = max(avail.left() + 12, min(tx, avail.right() - self.width() - 12))
            ty = max(avail.top() + 12, min(ty, avail.bottom() - self.height() - 12))
        self.move(tx, ty)


class LocalModelDownloadNoticeDialog(QDialog):
    """Попередження перед завантаженням локальної моделі про можливий тривалий перший запуск."""

    def __init__(self, engine_name: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("dlg.local_notice.title", "Підготовка локальної моделі"))
        self.setMinimumWidth(460)
        self.setMaximumWidth(560)
        self.setStyleSheet(DIALOG_STYLESHEET)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 20, 22, 18)
        lay.setSpacing(14)

        hdr = QLabel("⏳ " + tr("dlg.local_notice.header", "Перший запуск голосу займає 10–40 секунд ({name})", name=engine_name))
        hdr.setObjectName("dialogTitle")
        hdr.setWordWrap(True)
        lay.addWidget(hdr)

        body_text = tr(
            "dlg.local_notice.body",
            "<b>Не дивуйтесь, якщо перша фраза думатиме довго!</b><br><br>"
            "• <b>Чому так:</b> щоразу після відкриття програми цьому голосу потрібно <b>10–40 секунд</b>, щоб підготуватись до роботи (найкраще це помітно одразу після встановлення).<br>"
            "• <b>А далі — швидко:</b> вже з <b>2-ї фрази</b> і до самого закриття програми озвучка триває всього <b>1–2 секунди</b>.<br>"
            "• <b>Надалі це рідко відчутно:</b> SilentVoice зазвичай сам підвантажує голос у фоні одразу після старту, тож чекати доведеться лише якщо почати говорити в перші секунди.<br><br>"
            "<i>Тож якщо перша фраза довго генерується — зачекайте й обов’язково спробуйте другу.</i>",
        )
        body = QLabel(body_text)
        body.setWordWrap(True)
        body.setTextFormat(Qt.TextFormat.RichText)
        body.setStyleSheet(
            "background-color: #262C3D; border: 1px solid #38415C; border-radius: 10px; "
            "padding: 14px 16px; color: #F1F5F9; font-size: 14px; line-height: 1.48;"
        )
        lay.addWidget(body)

        self.dont_show_cb = QCheckBox(tr("dlg.local_notice.dont_show", "Більше не показувати це повідомлення"))
        self.dont_show_cb.setCursor(Qt.CursorShape.PointingHandCursor)
        lay.addWidget(self.dont_show_cb)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        cancel_btn = QPushButton(tr("dlg.local_notice.cancel", "Скасувати"))
        cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        ok_btn = QPushButton(tr("dlg.local_notice.continue", "Завантажити"))
        ok_btn.setObjectName("primaryBtn")
        ok_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(ok_btn)

        lay.addLayout(btn_row)

    @property
    def dont_show_again(self) -> bool:
        return self.dont_show_cb.isChecked()


class Qwen3QuantChoiceDialog(QDialog):
    """Вибір квантування Qwen3-TTS 1.7B (Q8 або Q5), якщо на ПК менше 9 ГБ відеопам'яті."""

    def __init__(self, vram_gb: Optional[float], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("dlg.qwen_quant.title", "Вибір версії Qwen3-TTS 1.7B"))
        self.setMinimumWidth(480)
        self.setMaximumWidth(580)
        self.setStyleSheet(DIALOG_STYLESHEET)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 20, 22, 18)
        lay.setSpacing(14)

        if vram_gb is not None and vram_gb > 0:
            vram_rounded = round(float(vram_gb)) if abs(float(vram_gb) - round(float(vram_gb))) <= 0.35 else round(float(vram_gb), 1)
            vram_label = f"{int(vram_rounded)}" if isinstance(vram_rounded, int) or float(vram_rounded).is_integer() else f"{vram_rounded:.1f}"
            hdr_text = "🎮 " + tr(
                "dlg.qwen_quant.header_vram",
                "Виявлено менше 9 ГБ відеопам’яті (~{vram} ГБ VRAM)",
                vram=vram_label,
            )
        else:
            hdr_text = "🎮 " + tr(
                "dlg.qwen_quant.header_unknown",
                "Оберіть версію Qwen3-TTS 1.7B для вашої відеокарти",
            )
        hdr = QLabel(hdr_text)
        hdr.setObjectName("dialogTitle")
        hdr.setWordWrap(True)
        lay.addWidget(hdr)

        body_text = tr(
            "dlg.qwen_quant.body",
            "<b>Яку версію Qwen 1.7B ви хочете встановити?</b><br><br>"
            "• <b>Qwen 1.7B (Q8)</b> — краща якість голосу, але більше використання відеопам’яті (<b>~3± ГБ VRAM</b>).<br>"
            "• <b>Qwen 1.7B (Q5)</b> — трохи нижча якість, але менше використання відеопам’яті (<b>~2.2 ГБ VRAM</b>).<br><br>"
            "<i>Якщо у вас менше 9 ГБ відеопам’яті й ви плануєте грати в ігри паралельно з програмою, "
            "версія Q8 може не залишити достатньо відеопам’яті для гри.</i>",
        )
        body = QLabel(body_text)
        body.setWordWrap(True)
        body.setTextFormat(Qt.TextFormat.RichText)
        body.setStyleSheet(
            "background-color: #262C3D; border: 1px solid #38415C; border-radius: 10px; "
            "padding: 14px 16px; color: #F1F5F9; font-size: 14px; line-height: 1.48;"
        )
        lay.addWidget(body)

        radio_style = (
            "QRadioButton { background-color: #202636; border: 1px solid #38415C; border-radius: 9px; "
            "padding: 10px 12px; color: #F8FAFC; font-size: 13px; font-weight: 600; spacing: 10px; }"
            "QRadioButton:hover { border-color: #635BFF; background-color: #262D42; }"
            "QRadioButton:checked { border-color: #7C5CFF; background-color: #2B2F4A; }"
        )
        self._group = QButtonGroup(self)

        self.radio_q5 = QRadioButton(
            tr("dlg.qwen_quant.opt_q5", "Qwen 1.7B (Q5) — менше відеопам’яті (~2.2 ГБ VRAM, рекомендовано для ігор)")
        )
        self.radio_q5.setCursor(Qt.CursorShape.PointingHandCursor)
        self.radio_q5.setStyleSheet(radio_style)
        self.radio_q5.setChecked(True)
        self._group.addButton(self.radio_q5)
        lay.addWidget(self.radio_q5)

        self.radio_q8 = QRadioButton(
            tr("dlg.qwen_quant.opt_q8", "Qwen 1.7B (Q8) — краща якість (~3± ГБ VRAM)")
        )
        self.radio_q8.setCursor(Qt.CursorShape.PointingHandCursor)
        self.radio_q8.setStyleSheet(radio_style)
        self._group.addButton(self.radio_q8)
        lay.addWidget(self.radio_q8)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        cancel_btn = QPushButton(tr("dlg.local_notice.cancel", "Скасувати"))
        cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        ok_btn = QPushButton(tr("dlg.qwen_quant.install", "Встановити"))
        ok_btn.setObjectName("primaryBtn")
        ok_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(ok_btn)

        lay.addLayout(btn_row)

    @property
    def selected_quant(self) -> str:
        return "q8" if self.radio_q8.isChecked() else "q5"


class HelpButton(QPushButton):
    """
    Кругла кнопка зі знаком оклику (!) в кружечку.
    При наведенні показує коротку підказку, а при натисканні відкриває просте пояснення.
    """

    def __init__(self, title: str, explanation: str, parent: Optional[QWidget] = None) -> None:
        super().__init__("!", parent)
        self._title = title
        self._explanation = explanation
        self.setFixedSize(20, 20)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh_tooltip()
        self.setStyleSheet(
            """
            QPushButton {
                background-color: #333C57;
                color: #C7D2FE;
                border: 1px solid #52608A;
                border-radius: 10px;
                font-size: 11px;
                font-weight: 800;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #635BFF;
                color: #FFFFFF;
                border-color: #827BFF;
            }
            """
        )
        self.clicked.connect(self.show_help)

    def _resolved_explanation(self) -> str:
        txt = self._explanation
        replacements = {
            "{hk_toggle}": str(config.get("hotkey_toggle_input", "Ctrl+Space")),
            "{hk_conversation}": str(config.get("hotkey_conversation_mode", "Ctrl+Shift+Space")),
            "{hk_send}": str(config.get("hotkey_send", "Enter")),
            "{hk_cancel}": str(config.get("hotkey_cancel", "Esc")),
            "{hk_stop}": str(config.get("hotkey_stop_tts", "Ctrl+Shift+S")),
            "{hk_repeat}": str(config.get("hotkey_repeat_last", "Ctrl+Shift+R")),
        }
        for k, v in replacements.items():
            if k in txt:
                txt = txt.replace(k, v)
        return txt

    def _refresh_tooltip(self) -> None:
        resolved = self._resolved_explanation()
        plain_tip = (
            resolved.replace("<br/>", "\n")
            .replace("<br />", "\n")
            .replace("<br>", "\n")
            .replace("<b>", "")
            .replace("</b>", "")
            .replace("<i>", "")
            .replace("</i>", "")
            .replace("<code>", "")
            .replace("</code>", "")
        )
        self.setToolTip(tr("dlg_f_002", "{title}\n\n{plain_tip}\n\n(Натисніть, щоб прочитати детальніше)", title=self._title, plain_tip=plain_tip))

    def enterEvent(self, event) -> None:
        self._refresh_tooltip()
        super().enterEvent(event)

    def show_help(self) -> None:
        """Відкриває діалог з повним поясненням."""
        dlg = InfoPopupDialog(self._title, self._resolved_explanation(), self.window())
        dlg.exec()

    def _show_help(self) -> None:
        self.show_help()


class DialogComboBox(QComboBox):
    """Охайний випадаючий список для діалогових вікон у єдиному стилі з головним вікном."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._error_border = False
        self.setMinimumHeight(33)

    def set_error_highlight(self, enabled: bool) -> None:
        self._error_border = enabled
        self.update()

    def wheelEvent(self, event) -> None:
        event.ignore()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        rect = self.rect().adjusted(1, 1, -1, -1)
        if self._error_border:
            border_col = QColor("#EF4444")
            pen_w = 2.0
        elif self.hasFocus() or self.underMouse():
            border_col = QColor("#7C5CFF")
            pen_w = 1.0
        else:
            border_col = QColor("#3E4766")
            pen_w = 1.0

        painter.setBrush(QColor("#1B1F2D"))
        painter.setPen(QPen(border_col, pen_w))
        painter.drawRoundedRect(rect, 8, 8)

        arrow_w = 22
        content_rect = rect.adjusted(10, 0, -(arrow_w + 3), 0)
        painter.setPen(QColor("#F8FAFC"))
        painter.setFont(self.font())
        fm = painter.fontMetrics()
        elided = fm.elidedText(self.currentText(), Qt.TextElideMode.ElideRight, max(12, content_rect.width()))
        painter.drawText(content_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), elided)

        x_div = rect.right() - arrow_w
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


class ElidedLabel(QLabel):
    """Мітка в один рядок, яка обрізає занадто довгий текст із `...` без розтягування контейнера."""

    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        if text:
            self.setToolTip(text)

    def minimumSizeHint(self) -> QSize:
        sh = super().minimumSizeHint()
        return QSize(0, sh.height())

    def sizeHint(self) -> QSize:
        sh = super().sizeHint()
        return QSize(min(180, sh.width()), sh.height())

    def setText(self, text: str) -> None:
        super().setText(text)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setFont(self.font())
        painter.setPen(self.palette().color(self.foregroundRole()))
        elided = painter.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, max(10, self.width()))
        painter.drawText(self.rect(), int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), elided)
        painter.end()


class VoiceSamplesDialog(QDialog):
    """
    Вікно зі списком доступних голосів для прослуховування конкретного TTS-движка.
    Для OpenVoice / Chatterbox / Qwen3-TTS додатково підтримує фільтр за мовами, статтю, гайд та кнопку вибору голосу.
    """

    _CLONE_ENGINES = ("openvoice", "qwen3_tts")

    _LANG_LABELS = {
        "uk": "🇺🇦 Українська",
        "en": "🇺🇸 English",
        "ru": "🇷🇺 Русский",
        "pl": "🇵🇱 Polski",
        "de": "🇩🇪 Deutsch",
        "fr": "🇫🇷 Français",
        "es": "🇪🇸 Español",
    }

    def __init__(
        self,
        engine_id: str,
        engine_title: str,
        parent: Optional[QWidget] = None,
        on_select_sample: Optional[Callable[[str], None]] = None,
        allow_install: bool = True,
        selected_sample_path: str = "",
        on_delete_sample: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__(parent)
        self._engine_id = engine_id
        self._engine_title = engine_title
        self._is_clone_engine = engine_id in self._CLONE_ENGINES
        self._on_select_sample = on_select_sample
        self._on_delete_sample = on_delete_sample
        self._allow_install = bool(allow_install)
        self._selected_norm = self._norm_path(selected_sample_path)
        self._install_buttons: List[tuple[str, QPushButton]] = []
        self._lang_combo: Optional[DialogComboBox] = None
        self._gender_combo: Optional[DialogComboBox] = None
        self._guide_btn: Optional[QPushButton] = None

        self.setWindowTitle(tr("dlg_f_003", "Прослухати голоси — {engine_title}", engine_title=engine_title))
        self.setMinimumSize(520, 380)
        self.resize(580, 440)
        self.setStyleSheet(DIALOG_STYLESHEET)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 16)
        lay.setSpacing(10)

        hdr_row = QHBoxLayout()
        hdr_row.setSpacing(8)
        hdr = ElidedLabel(tr("dlg_f_004", "🎧 Доступні голоси: {engine_title}", engine_title=engine_title))
        hdr.setObjectName("dialogTitle")
        hdr.setToolTip("")
        hdr_row.addWidget(hdr, 1)

        if self._is_clone_engine:
            self._guide_btn = QPushButton(tr("ov.guide_btn", "📖 Гайд"))
            self._guide_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._guide_btn.setToolTip(tr("ov.guide_btn_tip", "Коротко про OpenVoice, де взяти голоси та готові тексти"))
            self._guide_btn.setStyleSheet("padding: 5px 11px; font-size: 12.5px;")
            self._guide_btn.clicked.connect(
                lambda checked=False: OpenVoiceGuideDialog(
                    self, engine_id=self._engine_id, engine_title=self._engine_title
                ).exec()
            )
            hdr_row.addWidget(self._guide_btn)
        lay.addLayout(hdr_row)

        filters_row = QHBoxLayout()
        filters_row.setSpacing(8)

        self._gender_combo = DialogComboBox()
        self._gender_combo.setMinimumWidth(135)
        self._gender_combo.addItem(tr("ui_103", "Всі голоси"), "both")
        self._gender_combo.addItem(tr("ui_104", "Жіночі"), "female")
        self._gender_combo.addItem(tr("ui_105", "Чоловічі"), "male")
        cur_gender = str(config.get("voice_gender_filter", "both") or "both").strip().lower()
        gen_idx = self._gender_combo.findData(cur_gender)
        self._gender_combo.setCurrentIndex(gen_idx if gen_idx >= 0 else 0)
        self._gender_combo.currentIndexChanged.connect(self._populate_samples_list)
        filters_row.addWidget(self._gender_combo)

        if self._is_clone_engine:
            self._lang_combo = DialogComboBox()
            self._lang_combo.setMinimumWidth(145)
            for code in ("uk", "en", "ru", "pl", "de", "fr", "es"):
                self._lang_combo.addItem(self._LANG_LABELS[code], code)
            cur_lang = str(config.get("app_language", "uk") or "uk").strip().lower()
            if cur_lang == "ua":
                cur_lang = "uk"
            lang_idx = self._lang_combo.findData(cur_lang)
            self._lang_combo.setCurrentIndex(lang_idx if lang_idx >= 0 else 0)
            self._lang_combo.currentIndexChanged.connect(self._populate_samples_list)
            filters_row.addWidget(self._lang_combo)

        filters_row.addStretch(1)
        lay.addLayout(filters_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            "QScrollArea { background-color: #252B3D; border: 1px solid #38415C; border-radius: 10px; }"
        )

        self._container = QWidget()
        self._container.setMinimumWidth(0)
        self._container.setStyleSheet("background: transparent;")
        self._c_lay = QVBoxLayout(self._container)
        self._c_lay.setContentsMargins(12, 12, 12, 12)
        self._c_lay.setSpacing(8)

        self._all_samples = (
            get_samples_for_engine(engine_id, lang="all", gender="both")
            if self._is_clone_engine
            else get_samples_for_engine(engine_id, gender="both")
        )
        self._populate_samples_list()

        scroll.setWidget(self._container)
        lay.addWidget(scroll, 1)

        btn_row = QHBoxLayout()
        stop_btn = QPushButton(tr("dlg_004", "⏹ Зупинити"))
        stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        stop_btn.clicked.connect(self._stop_preview_playback)
        btn_row.addWidget(stop_btn)
        btn_row.addStretch(1)

        close_btn = QPushButton(tr("dlg_005", "Закрити"))
        close_btn.setObjectName("primaryBtn")
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)

    def _stop_preview_playback(self) -> None:
        """Миттєво зупиняє поточне чи заплановане в черзі прослуховування зразка."""
        try:
            from core.audio_router import audio_router

            audio_router.begin_exclusive_playback()
        except Exception:
            pass
        tts_manager.stop_current()

    def done(self, result: int) -> None:
        self._stop_preview_playback()
        super().done(result)

    def _populate_samples_list(self) -> None:
        while self._c_lay.count():
            item = self._c_lay.takeAt(0)
            w = item.widget() if item else None
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        self._install_buttons.clear()

        lang_filter = str(self._lang_combo.currentData() or "uk") if self._lang_combo is not None else ""
        gender_filter = str(self._gender_combo.currentData() or "both") if self._gender_combo is not None else "both"
        if self._is_clone_engine and lang_filter:
            lang_samples = [s for s in self._all_samples if (s.get("lang") or "") in (lang_filter, "")]
        else:
            lang_samples = list(self._all_samples)
        if gender_filter in ("female", "male"):
            samples = [s for s in lang_samples if (s.get("gender") or "") in (gender_filter, "")]
        else:
            samples = list(lang_samples)

        if not samples:
            if self._is_clone_engine and not lang_samples:
                empty_text = tr(
                    "clone.samples.empty_for_lang",
                    "Для вашої мови голосів поки що немає.\nДодайте аудіофайли (.wav, .mp3, .ogg, .flac) у data/samples/{engine_id}/{lang}/ або додайте голос через «📂 Обрати голос».",
                    engine_id="clone_voices",
                    lang=lang_filter or "uk",
                )
            elif gender_filter in ("female", "male") and lang_samples:
                empty_text = tr(
                    "clone.samples.empty_for_gender",
                    "Для обраного фільтра статі голосів поки що немає.",
                )
            elif self._is_clone_engine:
                empty_text = tr(
                    "clone.samples.empty_hint_langs",
                    "Папка прикладів {engine_title} поки порожня.\nДодайте аудіофайли (.wav, .mp3, .ogg, .flac) у data/samples/{engine_id}/<мова>/ (uk, en, ru, pl, de, fr, es).",
                    engine_title=self._engine_title,
                    engine_id="clone_voices",
                )
            else:
                empty_text = tr("dlg_002", "Список голосів для прослуховування поки що порожній.")
            empty_lbl = QLabel(empty_text)
            empty_lbl.setWordWrap(True)
            empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_lbl.setStyleSheet("color: #94A3B8; font-size: 13px; padding: 40px 12px;")
            self._c_lay.addWidget(empty_lbl, 1)
            return

        for item in samples:
            self._add_sample_row(item)
        self._c_lay.addStretch(1)
        self._refresh_install_buttons()

    def _add_sample_row(self, item: Dict[str, str]) -> None:
        row_card = QFrame()
        row_card.setStyleSheet(
            "QFrame { background-color: #2E364D; border: 1px solid #414B6B; border-radius: 8px; }"
        )
        r_lay = QHBoxLayout(row_card)
        r_lay.setContentsMargins(12, 7, 10, 7)
        r_lay.setSpacing(6)

        gender = str(item.get("gender") or "").strip().lower()
        if gender in ("female", "male"):
            g_lbl = QLabel(tr("voice.gender_f", "Ж") if gender == "female" else tr("voice.gender_m", "М"))
            g_lbl.setFixedWidth(16)
            g_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            g_col = "#F472B6" if gender == "female" else "#38BDF8"
            g_lbl.setStyleSheet(f"border: none; background: transparent; color: {g_col}; font-weight: 700; font-size: 12px;")
            r_lay.addWidget(g_lbl)

        name_lbl = ElidedLabel(item["name"])
        lbl_font = name_lbl.font()
        lbl_font.setBold(True)
        name_lbl.setFont(lbl_font)
        name_lbl.setStyleSheet("border: none; background: transparent; color: #FFFFFF;")
        r_lay.addWidget(name_lbl, 1)

        if self._on_select_sample is not None:
            use_btn = QPushButton()
            use_btn.setFixedSize(34, 28)
            use_btn.setStyleSheet("padding: 0px; font-size: 13px;")
            norm_item = self._norm_path(item["path"])
            if self._allow_install:
                use_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                use_btn.clicked.connect(lambda checked=False, p=item["path"]: self._choose_sample(p))
            else:
                use_btn.setEnabled(False)
                use_btn.setToolTip(
                    tr(
                        "ov.samples.install_required_tip",
                        "Спочатку встановіть OpenVoice, щоб обрати цей голос",
                    )
                )
            self._install_buttons.append((norm_item, use_btn))
            r_lay.addWidget(use_btn)

        play_btn = QPushButton("▶")
        play_btn.setObjectName("primaryBtn")
        play_btn.setFixedSize(34, 28)
        play_btn.setStyleSheet("padding: 0px; font-size: 13px;")
        play_btn.setToolTip(tr("dlg_003", "▶ Прослухати"))
        play_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        play_btn.clicked.connect(
            lambda checked=False, p=item["path"]: tts_manager.play_sample_preview(p, monitor_only=True)
        )
        r_lay.addWidget(play_btn)

        if item.get("is_custom") == "1":
            del_btn = QPushButton()
            del_btn.setFixedSize(28, 28)
            del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            del_btn.setIcon(QIcon(str(BUNDLE_DIR / "assets" / "delete.svg")))
            del_btn.setIconSize(QSize(15, 15))
            del_btn.setStyleSheet("QPushButton { padding: 0px; text-align: center; }")
            del_btn.setToolTip(tr("ov.samples.delete_custom_tip", "Видалити доданий голос"))
            del_btn.clicked.connect(lambda checked=False, p=item["path"]: self._delete_custom_sample(p))
            r_lay.addWidget(del_btn)

        self._c_lay.addWidget(row_card)

    def _delete_custom_sample(self, path_str: str) -> None:
        del_norm = self._norm_path(path_str)
        if del_norm and del_norm == self._selected_norm:
            self._stop_preview_playback()
            self._selected_norm = ""
        remove_custom_clone_sample(self._engine_id, path_str)
        self._all_samples = (
            get_samples_for_engine(self._engine_id, lang="all", gender="both")
            if self._is_clone_engine
            else get_samples_for_engine(self._engine_id, gender="both")
        )
        if self._on_delete_sample is not None:
            self._on_delete_sample(path_str)
        self._populate_samples_list()

    @staticmethod
    def _norm_path(path_str: str) -> str:
        clean = str(path_str or "").strip()
        if not clean:
            return ""
        try:
            return str(Path(clean).resolve()).lower()
        except Exception:
            return clean.lower()

    def _choose_sample(self, path_str: str) -> None:
        if not self._allow_install or self._on_select_sample is None:
            return
        self._selected_norm = self._norm_path(path_str)
        self._on_select_sample(path_str)
        self._refresh_install_buttons()

    def _refresh_install_buttons(self) -> None:
        for norm_item, btn in self._install_buttons:
            is_selected = self._allow_install and bool(norm_item) and norm_item == self._selected_norm
            if is_selected:
                btn.setText("✅")
                btn.setToolTip(tr("ov.samples.selected_btn", "✓ Встановлено"))
                btn.setObjectName("successBtn")
            else:
                btn.setText("📥")
                if self._allow_install:
                    btn.setToolTip(tr("ov.samples.install_btn", "Встановити"))
                btn.setObjectName("primaryBtn" if self._allow_install else "")
            btn.style().unpolish(btn)
            btn.style().polish(btn)


OPENVOICE_SAMPLE_TEXTS: Dict[str, str] = {
    "uk": (
        "Привіт! Сьогодні надворі чудова погода, і цей запис допоможе точно передати мій природний голос, ритм та інтонацію. "
        "Чуєш, як чисто звучать голосні, шиплячі та м'які приголосні? Продовжуймо говорити спокійно й виразно, без поспіху, щоб кожна фраза звучала максимально живо й реалістично."
    ),
    "en": (
        "Hello! The weather outside is bright and crisp today, and this short recording captures my natural tone, rhythm, and pacing clearly. "
        "Can you hear how balanced the vowels, soft pauses, and sharp consonants sound together? Let's keep speaking at a calm, steady volume so every cloned sentence sounds vivid and natural."
    ),
    "ru": (
        "Привет! Сегодня на улице прекрасная погода, и эта короткая запись поможет точно передать мой естественный тембр, ритм и живую интонацию. "
        "Слышишь, как чётко звучат гласные, шипящие и звонкие согласные? Давай говорить спокойно, ровно и выразительно, чтобы каждая фраза потом звучала максимально живо."
    ),
    "pl": (
        "Cześć! Dzisiaj na zewnątrz jest piękna pogoda, a to krótkie nagranie dokładnie odda moją naturalną barwę głosu, rytm oraz spokojną intonację. "
        "Słyszysz, jak wyraźnie brzmią samogłoski i szeleszczące spółgłoski? Mówmy dalej równym, czystym głosem bez pośpiechu, aby każde kolejne zdanie brzmiało zupełnie naturalnie."
    ),
    "de": (
        "Hallo! Draußen ist heute wunderbares Wetter, und diese kurze Aufnahme fängt meine natürliche Stimmfarbe, meinen Rhythmus und klare Pausen genau ein. "
        "Hörst du, wie deutlich Vokale, Zischlaute und weiche Konsonanten klingen? Sprechen wir ruhig und gleichmäßig weiter, damit jeder Satz lebendig und authentisch wirkt."
    ),
    "fr": (
        "Bonjour ! Il fait un temps magnifique dehors aujourd'hui, et ce court enregistrement capture fidèlement mon timbre naturel, mon rythme et mon intonation. "
        "Entends-tu la clarté des voyelles, des liaisons et des consonnes douces ? Continuons à parler calmement et distinctement pour que chaque phrase reste vivante et naturelle."
    ),
    "es": (
        "¡Hola! Hoy hace un tiempo estupendo ahí fuera, y esta breve grabación recoge con precisión mi timbre natural, mi ritmo y mi entonación habitual. "
        "¿Notas qué claras suenan las vocales, las pausas y las consonantes? Sigamos hablando con calma y un volumen estable para que cada frase clonada suene muy real."
    ),
}


class OpenVoiceGuideDialog(QDialog):
    """Короткий практичний гайд із клонування голосу та готові тексти під усі мови."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        engine_id: str = "openvoice",
        engine_title: str = "OpenVoice v2",
    ) -> None:
        super().__init__(parent)
        is_ov = engine_id == "openvoice"
        self.setWindowTitle(
            tr("ov.guide.win_title", "Гайд по клонуванню через OpenVoice")
            if is_ov
            else tr("clone.guide.win_title", "Гайд по клонуванню голосу — {engine_title}", engine_title=engine_title)
        )
        self.setMinimumSize(540, 470)
        self.resize(580, 520)
        self.setStyleSheet(DIALOG_STYLESHEET)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 16)
        lay.setSpacing(12)

        hdr = QLabel(
            tr("ov.guide.header", "🎭 Як отримати найкращий голос через OpenVoice")
            if is_ov
            else tr("clone.guide.header", "🎭 Як отримати найкращий голос для {engine_title}", engine_title=engine_title)
        )
        hdr.setObjectName("dialogTitle")
        lay.addWidget(hdr)

        body = QLabel(
            tr(
                "ov.guide.body",
                "<b>Що це робить:</b> OpenVoice підміняє тембр будь-якого TTS під вибраний файл. Міняє голос слабенько, зате майже не навантажує ПК й дозволяє додати трохи індивідуальності.<br><br>"
                "<b>Де взяти файл голосу:</b><br>"
                "• Вирізати чистий фрагмент голосу з будь-якого відео чи запису з інтернету.<br>"
                "• Або на <b>ElevenLabs</b>, <b>Fish Audio</b> чи схожому сервісі (дають безкоштовні кредити) обрати голос, <b>вставити текст нижче</b> й скачати файл.<br><br>"
                "<b>Вимоги до файлу:</b> 10–25 с (ідеально ~12–18 с), один голос без музики, шуму й відлуння, рівна гучність.",
            )
            if is_ov
            else tr(
                "clone.guide.body",
                "<b>Що це робить:</b> {engine_title} озвучує текст одразу тембром і манерою з вибраного аудіофайлу.<br><br>"
                "<b>Де взяти файл голосу:</b><br>"
                "• Вирізати чистий фрагмент голосу з будь-якого відео чи запису з інтернету.<br>"
                "• Або зареєструватися на <b>ElevenLabs</b>, <b>Fish Audio</b> чи іншому схожому сервісі (дають безкоштовні кредити), обрати там голос, <b>вставити текст нижче</b> й скачати готовий файл.<br><br>"
                "<b>Як отримати найкращу якість файлу:</b><br>"
                "• <b>Тривалість:</b> 6–20 секунд (ідеально ~8–15 с).<br>"
                "• <b>Чистота:</b> один диктор, без музики, звуків гри, шуму чи відлуння.<br>"
                "• <b>Подача:</b> рівна природна гучність без шепоту, крику й перевантаження, із короткими паузами та різною інтонацією.",
                engine_title=engine_title,
            )
        )
        body.setWordWrap(True)
        body.setTextFormat(Qt.TextFormat.RichText)
        body.setStyleSheet("color: #E2E8F0; font-size: 13px; line-height: 1.45;")
        lay.addWidget(body)

        prompt_card = QFrame()
        prompt_card.setStyleSheet(
            "QFrame { background-color: #252B3D; border: 1px solid #38415C; border-radius: 10px; }"
        )
        p_lay = QVBoxLayout(prompt_card)
        p_lay.setContentsMargins(14, 12, 14, 12)
        p_lay.setSpacing(8)

        lang_row = QHBoxLayout()
        lang_lbl = QLabel(tr("ov.guide.text_lbl", "Текст для запису / озвучення:"))
        lang_lbl.setStyleSheet("border: none; background: transparent; color: #FFFFFF; font-weight: 700;")
        lang_row.addWidget(lang_lbl)
        lang_row.addStretch(1)

        self.lang_combo = DialogComboBox()
        self.lang_combo.setMinimumWidth(145)
        for code, title in (
            ("uk", "🇺🇦 Українська"),
            ("en", "🇺🇸 English"),
            ("ru", "🇷🇺 Русский"),
            ("pl", "🇵🇱 Polski"),
            ("de", "🇩🇪 Deutsch"),
            ("fr", "🇫🇷 Français"),
            ("es", "🇪🇸 Español"),
        ):
            self.lang_combo.addItem(title, code)
        cur_lang = str(config.get("app_language", "uk") or "uk").strip().lower()
        idx = self.lang_combo.findData(cur_lang)
        self.lang_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.lang_combo.currentIndexChanged.connect(self._on_lang_changed)
        lang_row.addWidget(self.lang_combo)
        p_lay.addLayout(lang_row)

        self.sample_text_edit = QTextEdit()
        self.sample_text_edit.setReadOnly(True)
        self.sample_text_edit.setFixedHeight(105)
        self.sample_text_edit.setStyleSheet(
            "QTextEdit { background-color: #1B2030; border: 1px solid #38415C; border-radius: 8px; color: #F8FAFC; padding: 8px; font-size: 13px; }"
        )
        p_lay.addWidget(self.sample_text_edit)

        copy_row = QHBoxLayout()
        self.copy_btn = QPushButton(tr("ov.guide.copy_btn", "📋 Скопіювати текст"))
        self.copy_btn.setObjectName("primaryBtn")
        self.copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.copy_btn.clicked.connect(self._copy_text)
        copy_row.addWidget(self.copy_btn)

        self.copy_status_lbl = QLabel("")
        self.copy_status_lbl.setStyleSheet("border: none; background: transparent; color: #34D399; font-size: 12px;")
        copy_row.addWidget(self.copy_status_lbl, 1)
        p_lay.addLayout(copy_row)
        lay.addWidget(prompt_card)

        self._on_lang_changed()

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_btn = QPushButton(tr("dlg_005", "Закрити"))
        close_btn.setObjectName("primaryBtn")
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        lay.addLayout(close_row)

    def _on_lang_changed(self) -> None:
        code = str(self.lang_combo.currentData() or "uk")
        self.sample_text_edit.setPlainText(OPENVOICE_SAMPLE_TEXTS.get(code, OPENVOICE_SAMPLE_TEXTS["en"]))
        self.copy_status_lbl.setText("")

    def _copy_text(self) -> None:
        cb = QGuiApplication.clipboard()
        if cb is not None:
            cb.setText(self.sample_text_edit.toPlainText())
        self.copy_status_lbl.setText(tr("ov.guide.copied", "✓ Скопійовано в буфер обміну!"))


class LocalModelsDialog(QDialog):
    """Просте підключення моделей через LM Studio, llama.cpp або локальний API."""

    models_fetched = Signal(object, str, str)
    lmstudio_models_fetched = Signal(object, str)
    managed_backend_finished = Signal(str, str, object, str)
    recommended_download_finished = Signal(str, str, str)
    operation_status = Signal(str)
    BACKENDS = (
        ("LM Studio", "lmstudio"),
        ("llama.cpp (.gguf)", "llama_cpp"),
        ("Other local API", "other"),
    )
    RECOMMENDED_MODELS = {
        "qwen25_3b": {
            "lms_repo": "lmstudio-community/Qwen2.5-3B-Instruct-GGUF",
            "filename": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
            "url": "https://huggingface.co/lmstudio-community/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        },
        "gemma3_4b": {
            "lms_repo": "lmstudio-community/gemma-3-4b-it-GGUF",
            "filename": "gemma-3-4b-it-Q4_K_M.gguf",
            "url": "https://huggingface.co/lmstudio-community/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf",
        },
        "qwen35_4b": {
            "lms_repo": "lmstudio-community/Qwen3.5-4B-GGUF",
            "filename": "Qwen3.5-4B-Q4_K_M.gguf",
            "url": "https://huggingface.co/lmstudio-community/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf",
        },
        "qwen35_9b": {
            "lms_repo": "lmstudio-community/Qwen3.5-9B-GGUF",
            "filename": "Qwen3.5-9B-Q4_K_M.gguf",
            "url": "https://huggingface.co/lmstudio-community/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf",
        },
    }

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("ui_local_models_title", "Інші локальні моделі"))
        self.setStyleSheet(DIALOG_STYLESHEET)
        self.available_models: List[str] = []
        self.selected_model = ""
        self.server_url = ""
        self.api_key = ""
        self.use_as_primary = True
        self.backend_title = ""
        self.managed_backend = ""
        self.model_path = ""
        self._request_url = ""
        self._preferred_lms_model_key = ""
        self._recommended_download_cancel = threading.Event()
        self.models_fetched.connect(self._on_models_fetched)
        self.lmstudio_models_fetched.connect(self._on_lmstudio_models_fetched)
        self.managed_backend_finished.connect(self._on_managed_backend_finished)
        self.recommended_download_finished.connect(self._on_recommended_download_finished)
        tts_manager.download_progress.connect(self._on_local_runtime_download_progress)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(8)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        root = QVBoxLayout(content)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)
        title = QLabel(tr("ui_local_models_heading", "Підключіть модель із локальної програми"))
        title.setObjectName("dialogTitle")
        root.addWidget(title)
        intro = QLabel(tr("ui_local_models_intro", "Моделі можна завантажити тут або в обраній програмі. SilentVoice допоможе запустити модель і підключитися до неї."))
        intro.setWordWrap(True)
        root.addWidget(intro)

        form = QFormLayout()
        form.setSpacing(6)
        self.backend_combo = DialogComboBox()
        for name, backend_key in self.BACKENDS:
            if backend_key == "llama_cpp":
                name = tr("ui_local_models_llama", "llama.cpp (.gguf)")
            elif backend_key == "other":
                name = tr("ui_local_models_other_address", "Other local API server")
            self.backend_combo.addItem(name, backend_key)
        form.addRow(tr("ui_local_models_backend", "Оберіть програму:"), self.backend_combo)

        self.address_label = QLabel(tr("ui_local_models_address", "Адреса сервера:"))
        self.address_edit = QLineEdit("http://localhost:8080/v1")
        self.address_edit.setPlaceholderText("http://localhost:8080/v1")
        self.address_edit.setVisible(False)
        self.address_edit.setEnabled(False)
        self.address_label.setVisible(False)
        form.addRow(self.address_label, self.address_edit)

        root.addLayout(form)

        self.model_file_row = QWidget()
        file_row = QHBoxLayout(self.model_file_row)
        file_row.setContentsMargins(0, 0, 0, 0)
        file_row.setSpacing(6)
        self.model_file_edit = QLineEdit()
        self.model_file_edit.setReadOnly(True)
        self.model_file_edit.setPlaceholderText(tr("ui_local_models_gguf_hint", "Choose a GGUF model file"))
        file_row.addWidget(self.model_file_edit, 1)
        self.choose_file_button = QPushButton(tr("ui_local_models_choose_gguf", "Choose GGUF file…"))
        self.choose_file_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.choose_file_button.clicked.connect(self._choose_gguf_file)
        file_row.addWidget(self.choose_file_button)
        self.model_file_row.setVisible(False)

        self.recommended_section = QWidget()
        recommended_layout = QVBoxLayout(self.recommended_section)
        recommended_layout.setContentsMargins(0, 0, 0, 0)
        recommended_layout.setSpacing(4)
        self.recommended_title = QLabel(tr("ui_local_models_recommended", "Рекомендовані моделі для старту"))
        self.recommended_title.setStyleSheet("color: #CBD5E1; font-size: 13px; font-weight: 600;")
        recommended_layout.addWidget(self.recommended_title)
        recommended_row = QHBoxLayout()
        recommended_row.setSpacing(6)
        self.recommended_model_combo = DialogComboBox()
        self.recommended_model_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.recommended_model_combo.setMinimumContentsLength(16)
        self.recommended_model_combo.addItem(
            tr("ui_local_models_model_gemma3_4b", "Gemma 3 4B — рекомендована основна модель"),
            "gemma3_4b",
        )
        self.recommended_model_combo.addItem(
            tr("ui_local_models_model_qwen25_3b", "Qwen2.5 3B Instruct — швидка текстова модель (~1,9 ГБ)"),
            "qwen25_3b",
        )
        self.recommended_model_combo.addItem(
            tr("ui_local_models_model_qwen35_4b", "Qwen3.5 4B — багатомовна модель (~2,7 ГБ)"),
            "qwen35_4b",
        )
        self.recommended_model_combo.addItem(
            tr("ui_local_models_model_qwen35_9b", "Qwen3.5 9B — вища якість (~5,6 ГБ; від 8 ГБ вільної RAM)"),
            "qwen35_9b",
        )
        recommended_row.addWidget(self.recommended_model_combo, 1)
        self.download_model_button = QPushButton(tr("ui_local_models_download_lms", "⬇ Завантажити в LM Studio"))
        self.download_model_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.download_model_button.clicked.connect(self._download_recommended_model)
        recommended_row.addWidget(self.download_model_button)
        self.cancel_model_download_button = QPushButton(tr("tts.dl.cancel_btn", "✕ Скасувати"))
        self.cancel_model_download_button.setProperty("class", "dangerBtn")
        self.cancel_model_download_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_model_download_button.setVisible(False)
        self.cancel_model_download_button.clicked.connect(
            lambda checked=False: self._recommended_download_cancel.set()
        )
        recommended_row.addWidget(self.cancel_model_download_button)
        recommended_layout.addLayout(recommended_row)
        self.recommended_hint = QLabel(tr("ui_local_models_lms_download_hint", "Натисніть «Завантажити», щоб модель з’явилася в списку LM Studio."))
        self.recommended_hint.setWordWrap(True)
        self.recommended_hint.setStyleSheet("color: #94A3B8; font-size: 12px;")
        recommended_layout.addWidget(self.recommended_hint)
        self.browse_models_button = QPushButton(tr("ui_local_models_browse_lms", "Відкрити каталог моделей"))
        self.browse_models_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.browse_models_button.clicked.connect(self._open_model_catalog)
        recommended_layout.addWidget(self.browse_models_button, 0, Qt.AlignmentFlag.AlignLeft)
        root.addWidget(self.recommended_section)
        root.addWidget(self.model_file_row)

        self.detect_button = QPushButton(tr("ui_local_models_detect", "🔎 Find models in LM Studio"))
        self.detect_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.detect_button.clicked.connect(self._discover_lmstudio_models)
        root.addWidget(self.detect_button)

        self.key_checkbox = QCheckBox(
            tr("ui_local_models_key", "Сервер просить ключ доступу")
        )
        self.key_checkbox.setVisible(False)
        root.addWidget(self.key_checkbox)
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText(tr("ui_local_models_key_ph", "Вставте ключ доступу, якщо сервер його вимагає"))
        self.key_edit.setVisible(False)
        root.addWidget(self.key_edit)
        self.key_checkbox.toggled.connect(self.key_edit.setVisible)

        self.install_button = QPushButton(tr("ui_local_models_install", "⬇ Download LM Studio"))
        self.install_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.install_button.clicked.connect(self._open_backend_download)
        root.addWidget(self.install_button)

        model_row = QHBoxLayout()
        self.model_combo = DialogComboBox()
        self.model_combo.setMinimumWidth(180)
        model_row.addWidget(self.model_combo, 1)
        self.refresh_button = QPushButton(tr("ui_local_models_refresh", "▶ Start and connect"))
        self.refresh_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refresh_button.clicked.connect(self._start_or_refresh)
        model_row.addWidget(self.refresh_button)
        root.addLayout(model_row)

        self.test_selected_button = QPushButton(tr("ai.test.open", "Перевірити"))
        self.test_selected_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.test_selected_button.setVisible(False)
        self.test_selected_button.clicked.connect(self._test_selected_model)
        root.addWidget(self.test_selected_button)

        self.status_label = QLabel(tr("ui_local_models_ready", "Оберіть програму й натисніть «Знайти моделі»."))
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #94A3B8; font-size: 12px;")
        root.addWidget(self.status_label)
        self.operation_status.connect(self.status_label.setText)

        self.operation_progress = QProgressBar()
        self.operation_progress.setRange(0, 0)
        self.operation_progress.setTextVisible(True)
        self.operation_progress.setFormat("%p%")
        self.operation_progress.setFixedHeight(16)
        self.operation_progress.setVisible(False)
        root.addWidget(self.operation_progress)

        self.primary_checkbox = QCheckBox(
            tr("ui_local_models_primary", "Використовувати вибрану модель як основну для AI-коректора")
        )
        self.primary_checkbox.setChecked(True)
        root.addWidget(self.primary_checkbox)
        root.addStretch(1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel_button = QPushButton(tr("dlg_026", "Скасувати"))
        cancel_button.clicked.connect(self.reject)
        buttons.addWidget(cancel_button)
        self.use_button = QPushButton(tr("ui_local_models_use", "Підключити модель"))
        self.use_button.setObjectName("primaryBtn")
        self.use_button.setEnabled(False)
        self.use_button.clicked.connect(self._accept_model)
        cancel_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.use_button.setCursor(Qt.CursorShape.PointingHandCursor)
        buttons.addWidget(self.use_button)
        outer.addLayout(buttons)

        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        self.recommended_model_combo.currentIndexChanged.connect(self._on_recommended_model_changed)
        self._on_backend_changed(self.backend_combo.currentIndex())
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            available_width = max(320, available.width() - 32)
            minimum_width = min(600, available_width)
            self.setMinimumWidth(minimum_width)
            self.setMaximumWidth(available_width)
            self.setMaximumHeight(max(320, available.height() - 32))
            self.resize(min(760, max(minimum_width, self.sizeHint().width()), available_width),
                        min(620, max(520, self.sizeHint().height()), available.height() - 32))

    def _on_backend_changed(self, index: int) -> None:
        backend = str(self.backend_combo.itemData(index) or "lmstudio")
        is_lms = backend == "lmstudio"
        is_llama = backend == "llama_cpp"
        is_other = backend == "other"
        self.address_edit.setVisible(is_other)
        self.address_edit.setEnabled(is_other)
        self.address_label.setVisible(is_other)
        self.key_checkbox.setVisible(is_other)
        self.key_edit.setVisible(is_other and self.key_checkbox.isChecked())
        self.model_file_row.setVisible(is_llama)
        self.model_combo.setVisible(not is_llama)
        self.detect_button.setVisible(is_lms)
        self.install_button.setVisible(is_lms)
        self.recommended_section.setVisible(is_lms or is_llama)
        self.install_button.setText(tr("ui_local_models_install", "⬇ Download LM Studio"))
        self.download_model_button.setText(
            tr("ui_local_models_download_lms", "⬇ Завантажити в LM Studio")
            if is_lms else tr("ui_local_models_download_gguf", "⬇ Завантажити модель")
        )
        self.recommended_hint.setText(
            tr("ui_local_models_lms_download_hint", "Натисніть «Завантажити», щоб модель з’явилася в списку LM Studio.")
            if is_lms else tr("ui_local_models_gguf_download_hint", "Модель завантажиться в SilentVoice і з’явиться в полі вибору файлу нижче.")
        )
        self.browse_models_button.setText(
            tr("ui_local_models_browse_lms", "Відкрити каталог моделей")
            if is_lms else tr("ui_local_models_browse_gguf", "Знайти інші GGUF-моделі")
        )
        self.refresh_button.setText(
            tr("ui_local_models_start_connect", "▶ Start and connect")
            if not is_other else tr("ui_local_models_refresh", "🔄 Find models")
        )
        self.model_combo.clear()
        self.available_models = []
        self.selected_model = ""
        self.server_url = ""
        self.managed_backend = ""
        self.model_path = ""
        self.use_button.setEnabled(False)
        self.test_selected_button.setVisible(False)
        self.refresh_button.setEnabled(
            is_other or is_lms or (is_llama and bool(self.model_file_edit.text().strip()))
        )
        self.detect_button.setEnabled(True)
        self.operation_progress.setVisible(False)
        self.status_label.setText(
            tr("ui_local_models_lms_ready", "Завантажте рекомендовану модель тут або встановіть її в LM Studio. Потім виберіть її зі списку й натисніть «Запустити та підключити».")
            if is_lms else
            tr("ui_local_models_gguf_ready", "Оберіть рекомендовану модель і завантажте її кнопкою нижче або виберіть свій файл .gguf.")
            if is_llama else
            tr("ui_local_models_other_ready", "Enter the local server address and find its models.")
        )
        self._refresh_recommended_model_state()
        if is_lms:
            QTimer.singleShot(0, self._discover_lmstudio_models)

    def _selected_recommended_model(self) -> Dict[str, str]:
        return self.RECOMMENDED_MODELS.get(str(self.recommended_model_combo.currentData() or "gemma3_4b"), {})

    def _on_recommended_model_changed(self, _index: int) -> None:
        if str(self.backend_combo.currentData() or "") == "llama_cpp":
            self._refresh_recommended_model_state()

    def _refresh_recommended_model_state(self) -> None:
        if not hasattr(self, "recommended_model_combo"):
            return
        if str(self.backend_combo.currentData() or "") != "llama_cpp":
            return
        model_info = self._selected_recommended_model()
        path = LLAMA_MODEL_DIR / str(model_info.get("filename", ""))
        if path.is_file():
            self.model_path = str(path)
            self.model_file_edit.setText(str(path))
            self.refresh_button.setEnabled(True)
            self.download_model_button.setEnabled(False)
            self.download_model_button.setText(tr("ui_local_models_gguf_downloaded", "✓ Уже завантажено"))
            self.status_label.setText(tr("ui_local_models_gguf_selected", "Модель уже завантажена. Натисніть «Запустити та підключити»."))
        else:
            current_path = Path(self.model_file_edit.text().strip()) if self.model_file_edit.text().strip() else None
            known_paths = {
                LLAMA_MODEL_DIR / str(info["filename"])
                for info in self.RECOMMENDED_MODELS.values()
            }
            if current_path in known_paths:
                self.model_file_edit.clear()
                self.model_path = ""
                self.refresh_button.setEnabled(False)
            elif current_path and current_path.is_file() and current_path.suffix.lower() == ".gguf":
                self.model_path = str(current_path)
                self.refresh_button.setEnabled(True)
            else:
                self.model_path = ""
                self.refresh_button.setEnabled(False)
            self.download_model_button.setEnabled(True)
            self.download_model_button.setText(tr("ui_local_models_download_gguf", "⬇ Завантажити модель"))

    def _open_model_catalog(self) -> None:
        backend = str(self.backend_combo.currentData() or "")
        url = (
            "https://lmstudio.ai/models"
            if backend == "lmstudio"
            else "https://huggingface.co/models?library=gguf&pipeline_tag=text-generation"
        )
        QDesktopServices.openUrl(QUrl(url))

    def _download_recommended_model(self) -> None:
        backend = str(self.backend_combo.currentData() or "")
        model_info = self._selected_recommended_model()
        if backend not in ("lmstudio", "llama_cpp") or not model_info:
            return
        if backend == "llama_cpp":
            existing_path = LLAMA_MODEL_DIR / str(model_info["filename"])
            if existing_path.is_file():
                self._refresh_recommended_model_state()
                return

        self._recommended_download_cancel.clear()
        self.cancel_model_download_button.setVisible(True)
        self.cancel_model_download_button.setEnabled(True)
        model_label = self.recommended_model_combo.currentText()
        self._preferred_lms_model_key = str(model_info.get("lms_repo", "")) if backend == "lmstudio" else ""
        status = tr("ui_local_models_download_start", "Завантажую рекомендовану модель: {model}…", model=model_label)
        self._set_operation_busy(True, status)
        self.operation_progress.setRange(0, 0)

        def _worker() -> None:
            try:
                if backend == "lmstudio":
                    result = download_recommended_lmstudio_model(
                        str(model_info["url"]),
                        str(model_info["filename"]),
                        str(model_info["lms_repo"]),
                        cancel_event=self._recommended_download_cancel,
                    )
                else:
                    result = str(download_recommended_gguf(
                        str(model_info["url"]),
                        str(model_info["filename"]),
                        cancel_event=self._recommended_download_cancel,
                    ))
                self.recommended_download_finished.emit(backend, result, "")
            except Exception as exc:
                self.recommended_download_finished.emit(backend, "", str(exc))

        threading.Thread(target=_worker, daemon=True, name="DownloadRecommendedLLM").start()

    def _on_recommended_download_finished(self, backend: str, result: str, error: str) -> None:
        self._set_operation_busy(False)
        self.cancel_model_download_button.setVisible(False)
        self.cancel_model_download_button.setEnabled(True)
        if error:
            if "cancel" in error.lower():
                self.status_label.setText(tr("ui_local_models_download_cancelled", "Завантаження скасовано."))
            else:
                self.status_label.setText(tr("ui_local_models_download_error", "Не вдалося завантажити модель: {error}", error=error[-600:]))
            self._refresh_recommended_model_state()
            return
        if backend == "llama_cpp" and result:
            self.model_path = result
            self.model_file_edit.setText(result)
            self.refresh_button.setEnabled(True)
            self.download_model_button.setEnabled(False)
            self.download_model_button.setText(tr("ui_local_models_gguf_downloaded", "✓ Уже завантажено"))
            self.status_label.setText(tr("ui_local_models_gguf_download_done", "Модель завантажено в SilentVoice. Тепер натисніть «Запустити та підключити»."))
            return
        if result:
            self._preferred_lms_model_key = result
        self.status_label.setText(tr("ui_local_models_lms_download_done", "Модель завантажено в LM Studio. Оновлюю список…"))
        self._discover_lmstudio_models()

    def _open_backend_download(self) -> None:
        QDesktopServices.openUrl(QUrl("https://lmstudio.ai/download"))

    def _choose_gguf_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr("ui_local_models_choose_gguf", "Choose GGUF file…"),
            str(LLAMA_MODEL_DIR if LLAMA_MODEL_DIR.is_dir() else Path.home()),
            "GGUF models (*.gguf)",
        )
        if path:
            self.model_file_edit.setText(path)
            self.model_path = path
            self.refresh_button.setEnabled(True)
            self.status_label.setText(tr("ui_local_models_gguf_selected", "Model selected. Press Start and connect."))

    def _discover_lmstudio_models(self) -> None:
        if str(self.backend_combo.currentData() or "") != "lmstudio":
            return
        self.detect_button.setEnabled(False)
        self.operation_progress.setVisible(True)
        self.status_label.setText(tr("ui_local_models_lms_searching", "Looking for models in LM Studio…"))

        def _worker() -> None:
            try:
                self.lmstudio_models_fetched.emit(list_lmstudio_models(), "")
            except Exception as exc:
                self.lmstudio_models_fetched.emit([], str(exc))

        threading.Thread(target=_worker, daemon=True, name="FindLMStudioModels").start()

    def _on_lmstudio_models_fetched(self, models: object, error: str) -> None:
        if str(self.backend_combo.currentData() or "") != "lmstudio":
            return
        self.operation_progress.setVisible(False)
        self.detect_button.setEnabled(True)
        self.model_combo.clear()
        self.available_models = []
        if error:
            self.install_button.setVisible(True)
            self.status_label.setText(
                tr("ui_local_models_lms_missing", "LM Studio CLI was not found. Install/open LM Studio once, then try again.")
                if "not found" in error.lower()
                else tr("ui_local_models_lms_error", "Could not read the LM Studio model list. Check that LM Studio is installed and try again.")
            )
            return
        entries = models if isinstance(models, list) else []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            key = str(entry.get("key", "")).strip()
            label = str(entry.get("label", key)).strip()
            if key:
                self.model_combo.addItem(label, key)
                self.available_models.append(key)
        if self.available_models:
            target_key = self._preferred_lms_model_key.split("@", 1)[0].lower()
            preferred_index = next(
                (
                    i for i in range(self.model_combo.count())
                    if str(self.model_combo.itemData(i) or "").lower().split("@", 1)[0] == target_key
                    or target_key in str(self.model_combo.itemData(i) or "").lower()
                ),
                0,
            )
            self.model_combo.setCurrentIndex(preferred_index)
            self.status_label.setText(
                tr("ui_local_models_lms_count", "Found {count} models. Choose one, then press Start and connect.", count=len(self.available_models))
            )
        else:
            self.status_label.setText(tr("ui_local_models_lms_none", "No models are installed in LM Studio yet. Install one there, then come back here."))

    def _set_operation_busy(self, busy: bool, message: str = "") -> None:
        if not busy:
            self.operation_progress.setRange(0, 0)
        elif self.operation_progress.maximum() != 0:
            self.operation_progress.setRange(0, 0)
        self.operation_progress.setVisible(busy)
        self.refresh_button.setEnabled(not busy)
        self.detect_button.setEnabled(not busy)
        self.choose_file_button.setEnabled(not busy)
        self.backend_combo.setEnabled(not busy)
        self.recommended_model_combo.setEnabled(not busy)
        self.download_model_button.setEnabled(not busy)
        self.browse_models_button.setEnabled(not busy)
        if message:
            self.status_label.setText(message)

    def _on_local_runtime_download_progress(self, engine_id: str, pct: int, label: str) -> None:
        if engine_id not in ("local_llama_cpp", "local_llm_download"):
            return
        self.operation_progress.setRange(0, 100)
        self.operation_progress.setValue(max(0, min(100, int(pct))))
        self.operation_progress.setVisible(True)
        self.status_label.setText(label)

    def _start_or_refresh(self) -> None:
        backend = str(self.backend_combo.currentData() or "lmstudio")
        if backend == "other":
            self._refresh_models()
            return
        if backend == "lmstudio":
            if not self.available_models:
                self._discover_lmstudio_models()
                return
            model_key = str(self.model_combo.currentData() or "").strip()
            if not model_key:
                self.status_label.setText(tr("ui_local_models_lms_choose", "Choose a model from the list first."))
                return
            self._set_operation_busy(True, tr("ui_local_models_lms_starting", "Starting LM Studio and loading the model…"))

            def _worker_lms() -> None:
                try:
                    url, models = start_lmstudio_model(model_key, self.operation_status.emit)
                    self.managed_backend_finished.emit(backend, url, models, "")
                except Exception as exc:
                    self.managed_backend_finished.emit(backend, "", [], str(exc))

            threading.Thread(target=_worker_lms, daemon=True, name="StartLMStudioModel").start()
            return

        model_path = self.model_file_edit.text().strip()
        if not model_path:
            self.status_label.setText(tr("ui_local_models_gguf_choose", "Choose a .gguf file first."))
            return
        self.model_path = model_path
        self._set_operation_busy(True, tr("ui_local_models_gguf_starting", "Preparing llama.cpp and starting the model…"))

        def _worker_llama() -> None:
            try:
                url, models = start_llama_cpp_server(model_path, progress=self.operation_status.emit)
                self.managed_backend_finished.emit(backend, url, models, "")
            except Exception as exc:
                self.managed_backend_finished.emit(backend, "", [], str(exc))

        threading.Thread(target=_worker_llama, daemon=True, name="StartLlamaCppModel").start()

    def _on_managed_backend_finished(self, backend: str, address: str, models: object, error: str) -> None:
        self._set_operation_busy(False)
        self.refresh_button.setText(tr("ui_local_models_start_connect", "▶ Start and connect"))
        if error:
            self.use_button.setEnabled(False)
            self.status_label.setText(tr("ui_local_models_start_error", "Could not start the model: {error}", error=error[-900:]))
            return
        self.managed_backend = backend
        self.server_url = address
        self.backend_title = "LM Studio" if backend == "lmstudio" else "llama.cpp"
        self.available_models = [str(model).strip() for model in models if str(model).strip()] if isinstance(models, list) else []
        if backend == "llama_cpp":
            self.selected_model = self.available_models[0] if self.available_models else Path(self.model_path).stem
        else:
            self.model_combo.clear()
            self.model_combo.addItems(self.available_models)
            self.selected_model = self.model_combo.currentText().strip()
        has_model = bool(self.selected_model)
        self.use_button.setEnabled(has_model)
        self.test_selected_button.setVisible(has_model)
        self.status_label.setText(
            tr("ui_local_models_connected", "Connected and ready: {model}", model=self.selected_model)
        )
        if has_model and backend in ("lmstudio", "llama_cpp"):
            # Managed backends are fully started and ready now, so add them directly.
            self._accept_model()

    def _refresh_models(self) -> None:
        address = self.address_edit.text().strip()
        key = self.key_edit.text().strip()
        if not address:
            self.status_label.setText(tr("ui_local_models_enter_address", "Вкажіть адресу локального сервера."))
            return
        self._request_url = address
        self._set_operation_busy(True)
        self.refresh_button.setText(tr("ui_local_models_searching", "⏳ Searching…"))
        self.status_label.setText(tr("ui_local_models_connecting", "Підключаюся до локального сервера..."))

        def _worker() -> None:
            try:
                models = fetch_ai_provider_models("local_api", key, address)
                self.models_fetched.emit(models, "", address)
            except Exception as exc:
                self.models_fetched.emit([], str(exc), address)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_models_fetched(self, models: object, error: str, address: str) -> None:
        self._set_operation_busy(False)
        self.refresh_button.setText(tr("ui_local_models_refresh", "🔄 Знайти моделі"))
        if address:
            self.address_edit.setText(address)
        if address != self.address_edit.text().strip():
            return
        self.model_combo.clear()
        self.available_models = [str(model).strip() for model in models if str(model).strip()] if isinstance(models, list) else []
        if error:
            self.use_button.setEnabled(False)
            self.status_label.setText(
                tr(
                    "ui_local_models_error",
                    "Не вдалося підключитися. Переконайтеся, що модель запущена, а сервер увімкнений.",
                )
            )
            return
        self.model_combo.addItems(self.available_models)
        has_models = bool(self.available_models)
        self.use_button.setEnabled(has_models)
        self.test_selected_button.setVisible(has_models)
        self.status_label.setText(
            tr("ui_local_models_count", "Знайдено моделей: {count}", count=len(self.available_models))
            if has_models
            else tr("ui_local_models_none", "Сервер підключений, але не показує моделей. Спершу відкрийте модель у цій програмі.")
        )

    def _test_selected_model(self) -> None:
        model = self.selected_model if self.managed_backend == "llama_cpp" else self.model_combo.currentText().strip()
        if not model:
            return
        base_url = self.server_url if self.managed_backend else self.address_edit.text().strip()
        item = {
            "id": "local_api_test",
            "provider": "local_api",
            "title": f"{self.backend_combo.currentText()} — {model}",
            "model": model,
            "api_key": self.key_edit.text().strip(),
            "base_url": base_url,
        }
        AICorrectorTestDialog(item, self).exec()

    def _accept_model(self) -> None:
        if self.managed_backend != "llama_cpp":
            self.selected_model = self.model_combo.currentText().strip()
        if not self.selected_model:
            return
        if not self.managed_backend:
            self.server_url = self.address_edit.text().strip()
            self.api_key = self.key_edit.text().strip()
        self.use_as_primary = self.primary_checkbox.isChecked()
        index = self.backend_combo.currentIndex()
        self.backend_title = str(self.backend_combo.currentText() or "Локальний сервер")
        self.accept()

    def reject(self) -> None:
        self._recommended_download_cancel.set()
        super().reject()

    def closeEvent(self, event) -> None:
        self._recommended_download_cancel.set()
        super().closeEvent(event)


class AICorrectorTestDialog(QDialog):
    """Test one configured LLM with the real spellcheck prompt and show before/after text."""

    result_ready = Signal(str, str, str, float)

    def __init__(self, item: Dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._item = dict(item)
        self.setWindowTitle(tr("ai.test.title", "Перевірка LLM — {name}", name=get_ai_item_title(item)))
        self.resize(820, 560)
        self.setStyleSheet(DIALOG_STYLESHEET)
        self.result_ready.connect(self._show_result)
        root = QVBoxLayout(self)
        title = QLabel(tr("ai.test.heading", "Перевірте, як модель виправляє текст"))
        title.setObjectName("dialogTitle")
        root.addWidget(title)
        prompt_hint = QLabel(tr("ai.test.prompt_hint", "Приклад і ваш текст перевіряються з поточним промптом коректора орфографії."))
        prompt_hint.setWordWrap(True)
        prompt_hint.setStyleSheet("color: #94A3B8; font-size: 12px;")
        root.addWidget(prompt_hint)
        self.last_meta = QLabel("")
        self.last_meta.setWordWrap(True)
        self.last_meta.setStyleSheet("color: #94A3B8; font-size: 12px;")
        root.addWidget(self.last_meta)
        columns = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel(tr("ai.test.input", "Надісланий текст")))
        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText(tr("ai.test.input_ph", "Введіть текст із помилками для перевірки"))
        left.addWidget(self.input_edit, 1)
        right = QVBoxLayout()
        right.addWidget(QLabel(tr("ai.test.output", "Результат редактора")))
        self.output_edit = QTextEdit()
        self.output_edit.setReadOnly(True)
        right.addWidget(self.output_edit, 1)
        columns.addLayout(left, 1)
        columns.addLayout(right, 1)
        root.addLayout(columns, 1)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        buttons = QHBoxLayout()
        self.example_button = QPushButton(tr("ai.test.example", "▶ Перевірити на прикладі"))
        self.example_button.clicked.connect(self._run_example)
        buttons.addWidget(self.example_button)
        self.manual_button = QPushButton(tr("ai.test.manual", "Перевірити мій текст"))
        self.manual_button.clicked.connect(self._run_manual)
        buttons.addWidget(self.manual_button)
        last_button = QPushButton(tr("ai.test.last", "Показати останнє повідомлення"))
        last_button.clicked.connect(self._show_last)
        buttons.addWidget(last_button)
        buttons.addStretch(1)
        close_button = QPushButton(tr("dlg_005", "Закрити"))
        close_button.clicked.connect(self.accept)
        buttons.addWidget(close_button)
        root.addLayout(buttons)
        self._show_last()

    def _run_example(self) -> None:
        self._run_text(tr("ai.test.example_text", "ghbdsn я вчора відправив файли але здається ти їх не отримав перевір будьласка і напиши мені до 18:30"))

    def _run_manual(self) -> None:
        text = self.input_edit.toPlainText().strip()
        if not text:
            self.status.setText(tr("ai.test.empty", "Спочатку введіть текст."))
            return
        self._run_text(text)

    def _run_text(self, text: str) -> None:
        self.input_edit.setPlainText(text)
        self.example_button.setEnabled(False)
        self.manual_button.setEnabled(False)
        self.status.setText(tr("ai.test.running", "Надсилаю запит до {name}…", name=get_ai_item_title(self._item)))
        prompt = get_effective_ai_prompt()

        def _worker() -> None:
            started_at = time.perf_counter()
            try:
                timeout = 180.0 if str(self._item.get("provider", "")) == "ollama" else 30.0
                output = _call_ai_item(self._item, text, prompt, timeout=timeout)
                self.result_ready.emit(text, output, "", time.perf_counter() - started_at)
            except Exception as exc:
                self.result_ready.emit(text, "", str(exc), time.perf_counter() - started_at)

        threading.Thread(target=_worker, daemon=True, name="AICorrectorTest").start()

    def _show_result(self, text: str, output: str, error: str, elapsed: float) -> None:
        self.example_button.setEnabled(True)
        self.manual_button.setEnabled(True)
        self.input_edit.setPlainText(text)
        self.output_edit.setPlainText(output or (text if error else ""))
        self.status.setText(
            tr("ai.test.error", "Помилка: {error}", error=error)
            if error
            else tr("ai.test.done", "Готово за {seconds:.1f} с. Порівняйте вхідний текст і результат праворуч.", seconds=elapsed)
        )

    def _show_last(self) -> None:
        last = get_last_ai_edit_result()
        if not last:
            self.last_meta.setText(tr("ai.test.no_last", "Ще немає попереднього результату AI-коректора."))
            return
        self.input_edit.setPlainText(str(last.get("input", "")))
        self.output_edit.setPlainText(str(last.get("output", "")))
        route = str(last.get("route", ""))
        provider = str(last.get("provider", ""))
        err = str(last.get("error", ""))
        self.last_meta.setText(
            tr("ai.test.last_meta", "Останній виклик: {provider} · маршрут: {route}{error}", provider=provider or "—", route=route or "—", error=f" · {err}" if err else "")
        )


class ApiGuideDialog(QDialog):
    """Покрокова інструкція для підключення конкретного онлайн-сервісу TTS."""

    @staticmethod
    def _build_guides() -> Dict[str, tuple[str, str, str, str]]:
        return {
            "google": (
                tr("dlg_006", "☁️ Як отримати API ключ для Google Cloud TTS"),
                tr(
                    "dlg_007",
                    "<b>1. Вхід та увімкнення API:</b>\n"
                    "• Перейдіть на <b>cloud.google.com/text-to-speech</b> (кнопка нижче), увійдіть у Google-акаунт і натисніть синю кнопку по центру.\n"
                    "• У верхньому пошуку введіть <b>text to speech</b>, оберіть <b>Cloud Text-to-Speech API</b> і натисніть синю кнопку <b>Enable (Увімкнути)</b>.\n\n"
                    "<b>2. Платіжний акаунт (Billing) — для безкоштовного ліміту:</b>\n"
                    "• Google попросить увімкнути білінг: натисніть <b>Enable billing → Create billing account</b>. Якщо проєкт ще не створений — створіть його.\n"
                    "• У формі оберіть: тип акаунта <b>Individual</b> (не Organization), податковий статус <b>Personal</b>, введіть ім'я, адресу та поштовий індекс міста.\n"
                    "• Прив'яжіть банківську картку — <b>на ній обов'язково має бути хоча б ~10 грн (або ~$1)</b> для тестової перевірки. Іноді форма поводиться дивно й просить увімкнути білінг / ввести дані вдруге — це нормально, просто натисніть <b>Enable</b> і підтвердіть ще раз.\n\n"
                    "<b>3. Створення API-ключа:</b>\n"
                    "• У меню зліва відкрийте <b>Credentials</b> (іконка ключика), зверху натисніть <b>+ Create Credentials → API key</b>.\n"
                    "• Введіть назву, нижче в <b>Select API</b> оберіть <b>Cloud Text-to-Speech API</b>, пролистайте вниз і натисніть <b>Create</b>.\n"
                    "• Скопіюйте ключ і вставте його в програмі в поле <b>API ключ</b> (збережеться автоматично).\n\n"
                    "<i>🛡️ У програмі увімкнено захист від перевищення безкоштовного ліміту — кошти з картки не спишуться. Щоб відкрити платні моделі, натисніть <b>💳 Перейти на платно</b> у блоці Google Cloud TTS.</i>",
                ),
                tr("dlg_008", "Відкрити Google Cloud TTS"),
                "https://cloud.google.com/text-to-speech",
            ),
            "elevenlabs": (
                tr("dlg_009", "🎙️ Як отримати API ключ для ElevenLabs"),
                tr(
                    "dlg_010",
                    "<b>1. Реєстрація та створення ключа:</b>\n"
                    "• Зайдіть на сайт <b>elevenlabs.io</b>, зареєструйтеся або увійдіть у свій акаунт і відкрийте розділ <b>API Keys</b> (кнопка нижче).\n"
                    "• Натисніть <b>Create Key</b> (Створити ключ) та введіть будь-яку назву.\n"
                    "• <b>Обов'язково</b> увімкніть потрібні дозволи для ключа:\n"
                    "  — <b>Text-to-Speech</b> → <code>Access</code> (Доступ)\n"
                    "  — <b>Voices</b> → <code>Read</code> (Читання)\n"
                    "  — <b>Voice Collections</b> → <code>Read</code> (Читання)\n"
                    "• Натисніть справа внизу <b>Create Key</b>, скопіюйте ключ і вставте його в програмі в поле <b>API ключ</b> (він збережеться автоматично).\n\n"
                    "<b>2. Тарифи, ліміти та моделі:</b>\n"
                    "• Натисніть на свою аватарку профілю → <b>Upgrade</b>, щоб переглянути залишок ліміту або обрати платний тариф.\n"
                    "• Окремої кнопки «перейти на платно» тут немає — у програмі одразу доступні моделі (<b>Flash</b>, <b>Turbo</b>, <b>Multilingual</b>). Для повсякденного озвучення найкраще підходить <b>Flash</b> (вона найшвидша та витрачає менше ліміту), а для максимальної якості — <b>Multilingual</b>.\n\n"
                    "<b>3. Вибір голосу з сайту (Voice ID):</b>\n"
                    "• На сайті в меню зліва відкрийте <b>Text to Speech</b> або <b>Voices</b>, оберіть мову та голос, який подобається (голоси із замочком можуть вимагати платного тарифу).\n"
                    "• Натисніть три крапки <b>...</b> біля голосу → <b>Copy Voice ID</b> і вставте цей ID у поле <b>ID голосу</b> в програмі (або натисніть <b>🔄 Оновити голоси</b>, щоб завантажити голоси з акаунту).",
                ),
                tr("dlg_011", "Відкрити ElevenLabs API Keys"),
                "https://elevenlabs.io/app/developers/api-keys",
            ),
            "openai": (
                tr("dlg_012", "🤖 Як отримати API ключ для OpenAI TTS"),
                tr(
                    "dlg_013",
                    "<b>1. Реєстрація та створення ключа:</b>\n"
                    "• Зайдіть на сайт <b>platform.openai.com</b> та увійдіть або зареєструйтеся.\n"
                    "• У меню зліва відкрийте <b>API keys</b> (або натисніть кнопку нижче), потім справа зверху натисніть <b>+ Create new secret key</b>.\n"
                    "• Введіть будь-яку назву, створіть ключ (починається на <code>sk-...</code>), скопіюйте його та вставте в поле <b>API ключ</b> у програмі (він збережеться автоматично).\n\n"
                    "<b>2. Оплата та баланс (важливо!):</b>\n"
                    "• У OpenAI API <b>немає безкоштовного тарифу</b> чи місячних підписок (навіть якщо у вас є ChatGPT Plus — це окремий сервіс).\n"
                    "• Тут діє оплата за фактом використання: відкрийте розділ <b>Billing</b> (кнопка <b>Go to billing</b>), прив'яжіть картку та поповніть баланс (зазвичай від $5) — без цього озвучення не працюватиме.\n\n"
                    "<b>3. Вибір моделі в програмі:</b>\n"
                    "• Оберіть потрібну модель у блоці OpenAI: <b>tts-1</b> (швидка й найдешевша), <b>tts-1-hd</b> (вища якість звуку) або <b>gpt-4o-mini-tts</b> (найновіша модель із природною інтонацією).",
                ),
                tr("dlg_014", "Відкрити OpenAI API Keys"),
                "https://platform.openai.com/api-keys",
            ),
            "fish_audio": (
                tr("dlg_015", "🐟 Як отримати API ключ та ID голосу для Fish Audio"),
                tr(
                    "dlg_016",
                    "<b>1. Реєстрація та ключ:</b>\n"
                    "• Зайдіть на сайт <b>fish.audio</b>, натисніть <b>Зареєструватися</b> (найпростіше через Google-акаунт).\n"
                    "• Перейдіть у розділ <b>API Keys</b> (кнопка нижче), натисніть справа зверху чорну кнопку <b>Створити API-ключ</b> (Create API Key), введіть будь-яку назву та натисніть <b>Створити</b>.\n"
                    "• Скопіюйте ключ та вставте його в програмі в поле <b>API ключ</b> (він збережеться автоматично).\n\n"
                    "<b>2. Кредити та платний режим:</b>\n"
                    "• Безкоштовно надається <b>8 000 кредитів</b>, які оновлюються щомісяця. Залишок та дату оновлення можна подивитися, натиснувши на свою аватарку справа зверху (там само через <b>Оновити</b> можна оформити підписку, наприклад на 250 000 кредитів/міс).\n"
                    "• Якщо ви оформили підписку і хочете використовувати платні моделі, натисніть у програмі кнопку <b>💳 Перейти на платно</b> (під кнопкою Гайд).\n\n"
                    "<b>3. Додавання голосу з каталогу або клонування свого:</b>\n"
                    "• На сайті відкрийте розділ <b>Text-to-Speech</b>, оберіть і додайте потрібний голос, а потім праворуч натисніть <b>Копіювати ідентифікатор моделі</b>. Там само на сайті можна <b>клонувати власний голос</b> (завантаживши короткий запис мовлення) і так само скопіювати його ID.\n"
                    "• Вставте цей ID у поле <b>ID голосу</b> в програмі (або натисніть <b>🔄 Оновити голоси</b>).\n"
                    "• <b>Важливо:</b> щоб доданий голос з'явився у списку, обов'язково оберіть відповідну для нього <b>мову</b> у списку справа (наприклад, для російського голосу має бути обрана російська мова, інакше він не відобразиться).",
                ),
                tr("dlg_017", "Відкрити Fish Audio API Keys"),
                "https://fish.audio/app/api-keys",
            ),
            "gemini_tts": (
                tr("dlg_018", "✨ Як отримати API ключ для Google AI Studio (Gemini TTS)"),
                tr(
                    "dlg_019",
                    "<b>1. Вхід у Google AI Studio:</b>\n"
                    "• Зайдіть на сайт <b>aistudio.google.com</b>, натисніть <b>Get started</b> та увійдіть через свій Google-акаунт.\n"
                    "• Натисніть зліва знизу на іконку ключа (або відкрийте сторінку за кнопкою нижче).\n\n"
                    "<b>2. Створення проєкту (обов'язково, якщо ще немає):</b>\n"
                    "• У меню зліва відкрийте розділ <b>Проєкти</b> (Projects), натисніть справа зверху <b>Створити новий проєкт</b>, введіть будь-яку назву та натисніть <b>Створити</b>.\n\n"
                    "<b>3. Створення API-ключа:</b>\n"
                    "• Поверніться в розділ <b>API ключі</b> (API keys) на лівій панелі та натисніть справа зверху <b>Створити API ключ</b>.\n"
                    "• Введіть назву, у випадаючому списку знизу обов'язково оберіть ваш проєкт і натисніть <b>Створити ключ</b>.\n"
                    "• Скопіюйте ключ (починається на <code>AIza...</code>), вставте його в поле <b>API ключ</b> у програмі (він збережеться автоматично) та оберіть модель зі списку (за замовчуванням <b>gemini-2.5-flash-preview-tts</b>, працює на безкоштовному тарифі).",
                ),
                tr("dlg_020", "Відкрити Google AI Studio"),
                "https://aistudio.google.com/app/apikey",
            ),
            "silero_bot": (
                tr("tg.guide.title", "🤖 Як підключити @silero_voice_bot через Telegram"),
                tr(
                    "tg.guide.html",
                    "<b>1. Підключення Telegram-акаунта:</b>\n"
                    "• У блоці <b>Silero Bot (Telegram)</b> натисніть <b>Підключити Telegram</b> — програма працюватиме з ботом через ваш акаунт.\n"
                    "• Увійдіть <b>по QR-коду</b> (Налаштування → Пристрої → Підключити пристрій) або за номером телефону. Якщо стоїть двофакторний захист (2FA), введіть свій пароль Telegram.\n"
                    "• За бажанням досвідчені користувачі можуть замінити стандартні API-ключі на власні (<code>my.telegram.org/apps</code>), але це не обов'язково.\n\n"
                    "<b>2. Ліміт та обов'язкова підписка на канал:</b>\n"
                    "• Безкоштовний ліміт у боті — близько <b>666 символів</b> (далі за потреби можна оформити недорогий тариф у самому боті).\n"
                    "• <b>Важливо:</b> у безкоштовному режимі бот може попросити підписатися на їхній новинний канал. Натисніть у програмі <b>🔊 Прослухати тест</b> — якщо підписка потрібна, бот надішле посилання на канал у Telegram.\n\n"
                    "<b>3. Додавання та вибір голосів:</b>\n"
                    "• Натисніть <b>➕ Додати голос</b>: введіть код вручну або просто оберіть голос у Telegram-боті через кнопку <b>Вибрати голос</b> (фільтри → відкрити голос → <b>Обрати</b>) — він додасться в програму і вікно закриється автоматично.\n"
                    "• Якщо ви вже обрали голос у боті раніше — натисніть <b>🔄 З бота (/speaker)</b>, щоб підтягнути останній активний голос.",
                ),
                tr("tg.guide.btn", "Відкрити my.telegram.org/apps"),
                "https://my.telegram.org/apps",
            ),
        }

    def __init__(self, provider: Optional[str] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("dlg_021", "Гайд — Налаштування API ключа"))
        self.setMinimumWidth(520)
        self.setMaximumWidth(620)
        self.setStyleSheet(DIALOG_STYLESHEET)

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(12)

        guides = self._build_guides()
        providers_to_show = [provider] if (provider and (provider in guides or provider in AI_PROVIDERS)) else list(guides.keys())

        for prov_key in providers_to_show:
            if prov_key in guides:
                hdr_text, body_html, btn_label, url = guides[prov_key]
            else:
                spec = AI_PROVIDERS[prov_key]
                hdr_text = tr(f"ai.guide.{prov_key}.title", str(spec["guide_title"]))
                body_html = (
                    tr(
                        "ui_local_models_guide",
                        "Choose LM Studio to start one of its downloaded models automatically, or choose llama.cpp and select a .gguf file. SilentVoice installs the CPU server if needed, starts it, and connects. Use Other local API only for a server you already run.",
                    )
                    if prov_key == "local_api"
                    else tr(
                        "ui_local_models_ollama_guide",
                        "Install Ollama and download a model with the buttons in its card. For other local models, click ⋯ and choose LM Studio or a .gguf file; SilentVoice starts the server and connects automatically.",
                    )
                    if prov_key == "ollama"
                    else tr(f"ai.guide.{prov_key}.html", str(spec["guide_html"]))
                )
                btn_label = tr(f"ai.guide.{prov_key}.btn", str(spec["guide_btn"]))
                url = str(spec["guide_url"])
                extra_btn = None
                if "guide_models_btn" in spec and "guide_models_url" in spec:
                    extra_btn = (
                        tr(f"ai.guide.{prov_key}.models_btn", str(spec["guide_models_btn"])),
                        str(spec["guide_models_url"]),
                    )
            if prov_key in guides:
                extra_btn = None
            root.addWidget(
                self._make_card(
                    hdr_text,
                    body_html,
                    btn_label,
                    url,
                    extra_btn=extra_btn,
                    show_video_hint=(prov_key not in ("ollama", "local_api")),
                )
            )

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        close_btn = QPushButton(tr("dlg_001", "Зрозуміло"))
        close_btn.setObjectName("primaryBtn")
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    def _make_card(
        self,
        header_text: str,
        body_html: str,
        btn_label: Optional[str],
        url: Optional[str],
        extra_btn: Optional[tuple[str, str]] = None,
        show_video_hint: bool = True,
    ) -> QWidget:
        card = QWidget()
        card.setStyleSheet(
            "QWidget { background-color: #262C3D; border: 1px solid #38415C; border-radius: 12px; }"
        )
        lay = QVBoxLayout(card)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(8)

        h_lbl = QLabel(header_text)
        h_lbl.setStyleSheet("border: none; background: transparent; color: #A5B4FC; font-size: 15px; font-weight: 700;")
        lay.addWidget(h_lbl)

        b_lbl = QLabel(body_html.replace("\n", "<br>"))
        b_lbl.setWordWrap(True)
        b_lbl.setTextFormat(Qt.TextFormat.RichText)
        b_lbl.setStyleSheet("border: none; background: transparent; color: #F1F5F9; font-size: 13px; line-height: 1.45;")
        lay.addWidget(b_lbl)

        if show_video_hint:
            yt_lbl = QLabel(
                tr(
                    "dlg.youtube_hint",
                    "💡 <i>Якщо щось незрозуміло або інтерфейс сайту оновився — просто знайдіть на YouTube короткий відео-гайд по налаштуванню цього сервісу.</i>",
                )
            )
            yt_lbl.setWordWrap(True)
            yt_lbl.setTextFormat(Qt.TextFormat.RichText)
            yt_lbl.setStyleSheet("border: none; background: transparent; color: #94A3B8; font-size: 12px;")
            lay.addWidget(yt_lbl)

        if (btn_label and url) or extra_btn:
            row = QHBoxLayout()
            if btn_label and url:
                btn = QPushButton(f"🌐 {btn_label}")
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(url)))
                row.addWidget(btn)
            if extra_btn:
                extra_label, extra_url = extra_btn
                extra_link_btn = QPushButton(f"📚 {extra_label}")
                extra_link_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                extra_link_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(extra_url)))
                row.addWidget(extra_link_btn)
            row.addStretch(1)
            lay.addLayout(row)

        return card


class HotkeyCaptureDialog(QDialog):
    """Діалог запису будь-якої клавіші або комбінації."""

    def __init__(
        self,
        hook_manager: KeyboardHookManager,
        current_hotkey: str = "",
        title_label: str = tr("dlg_022", "Призначення клавіші"),
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._hook = hook_manager
        self.recorded_hotkey: Optional[str] = None

        self.setWindowTitle(title_label)
        self.setFixedSize(460, 245)
        self.setStyleSheet(DIALOG_STYLESHEET)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 20, 22, 18)
        lay.setSpacing(10)

        lbl_title = QLabel(f"⌨️ {title_label}")
        lbl_title.setObjectName("dialogTitle")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(lbl_title)

        lbl_hint = QLabel(
            tr(
                "dlg_023",
                "Натисніть потрібну клавішу чи комбінацію\n"
                "(напр.: Ctrl+Space, Ctrl+Alt, CapsLock)",
            )
        )
        lbl_hint.setWordWrap(True)
        lbl_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_hint.setStyleSheet("color: #CBD5E1; font-size: 12px;")
        lay.addWidget(lbl_hint)

        self.double_tap_chk = QCheckBox(tr("dlg.hk.double_tap", "Подвійне натискання (2x, напр. 2x Shift)"))
        self.double_tap_chk.setCursor(Qt.CursorShape.PointingHandCursor)
        self.double_tap_chk.setToolTip(
            tr("dlg.hk.double_tap_tip", "Натисніть клавішу 1 раз — запишеться подвійне натискання.")
        )
        is_cur_double = str(current_hotkey or "").strip().lower().startswith(("2x ", "2х ", "double"))
        self.double_tap_chk.setChecked(is_cur_double)
        self.double_tap_chk.toggled.connect(self._hook.set_recording_double)
        lay.addWidget(self.double_tap_chk, 0, Qt.AlignmentFlag.AlignCenter)

        self.preview_lbl = QLabel(current_hotkey or tr("dlg_024", "Очікую натискання клавіші..."))
        self.preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_lbl.setStyleSheet(
            "background-color: #262C3D; color: #C7D2FE; border: 2px dashed #7C5CFF; "
            "border-radius: 10px; padding: 10px; font-size: 17px; font-weight: 700;"
        )
        lay.addWidget(self.preview_lbl)

        btn_row = QHBoxLayout()
        clear_btn = QPushButton(tr("dlg_025", "Прибрати клавішу"))
        clear_btn.clicked.connect(self._on_clear)
        cancel_btn = QPushButton(tr("dlg_026", "Скасувати"))
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(cancel_btn)
        lay.addLayout(btn_row)

        self._hook_connected = True
        self._hook.hotkey_recorded.connect(self._on_hotkey_captured)
        self._hook.start_recording_hotkey(double_mode=self.double_tap_chk.isChecked())

    def _disconnect_hook(self) -> None:
        self._hook.stop_recording_hotkey()
        if self._hook_connected:
            self._hook_connected = False
            try:
                self._hook.hotkey_recorded.disconnect(self._on_hotkey_captured)
            except Exception:
                pass

    def _on_hotkey_captured(self, hk: str) -> None:
        self.recorded_hotkey = hk
        self.preview_lbl.setText(hk)
        self.accept()

    def _on_clear(self) -> None:
        self._hook.stop_recording_hotkey()
        self.recorded_hotkey = ""
        self.accept()

    def closeEvent(self, event) -> None:
        self._disconnect_hook()
        super().closeEvent(event)

    def reject(self) -> None:
        self._disconnect_hook()
        super().reject()

    def accept(self) -> None:
        self._disconnect_hook()
        super().accept()


class QuickPhraseDialog(QDialog):
    """
    Діалог створення або редагування швидкої фрази.
    Має 2 основні кнопки створення звуку:
    1. «📁 Додати файлом» — прикріпити готовий звуковий файл (.wav, .mp3, .ogg, .flac) з ПК.
    2. «🎙️ Записати» — синтезувати введений текст активним голосом програми та зберегти у файл,
       щоб при кожному натисканні фраза відтворювалася миттєво без повторного переозвучування.
    """

    def __init__(
        self,
        hook_manager: KeyboardHookManager,
        phrase_data: Optional[Dict[str, str]] = None,
        parent: Optional[QWidget] = None,
        auto_pick_file: bool = False,
    ) -> None:
        super().__init__(parent)
        self._hook = hook_manager
        self._data = dict(
            phrase_data
            or {"title": "", "text": "", "hotkey": "", "emoji": "💬", "audio_file": ""}
        )

        self.setWindowTitle(tr("dlg_027", "Швидка фраза"))
        self.setMinimumWidth(510)
        self.setMaximumWidth(560)
        self.setStyleSheet(DIALOG_STYLESHEET)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 18)
        lay.setSpacing(12)

        header = QLabel(tr("dlg_028", "💬 Налаштування швидкої фрази"))
        header.setObjectName("dialogTitle")
        lay.addWidget(header)

        form = QFormLayout()
        form.setSpacing(10)

        top_row = QHBoxLayout()
        self.emoji_edit = QLineEdit(self._data.get("emoji", "💬"))
        self.emoji_edit.setFixedWidth(52)
        self.emoji_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_edit = QLineEdit(self._data.get("title", ""))
        self.title_edit.setPlaceholderText(tr("dlg_029", "Коротка назва кнопки (напр. Привіт)"))
        top_row.addWidget(self.emoji_edit)
        top_row.addWidget(self.title_edit, 1)
        form.addRow(tr("dlg_030", "Іконка і назва:"), top_row)

        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(self._data.get("text", ""))
        self.text_edit.setPlaceholderText(
            tr("dlg_031", "Текст, який програма скаже вголос і запише у файл (щоб не переозвучувати щоразу)...")
        )
        self.text_edit.setFixedHeight(72)
        form.addRow(tr("dlg_032", "Текст озвучки:"), self.text_edit)

        # Блок із 2 кнопками: «📁 Додати файлом» та «🎙️ Записати»
        audio_card = QFrame()
        audio_card.setStyleSheet(
            "QFrame { background-color: #252B3E; border: 1px solid #3B4566; border-radius: 10px; }"
        )
        ac_lay = QVBoxLayout(audio_card)
        ac_lay.setContentsMargins(12, 10, 12, 10)
        ac_lay.setSpacing(8)

        audio_btn_row = QHBoxLayout()
        audio_btn_row.setSpacing(8)

        self.btn_add_file = QPushButton(tr("dlg_033", "📁 Додати файлом"))
        self.btn_add_file.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_add_file.setToolTip(tr("dlg_034", "Вибрати готовий аудіофайл (.wav, .mp3, .ogg) з комп'ютера"))
        self.btn_add_file.clicked.connect(self._on_pick_audio_file)
        audio_btn_row.addWidget(self.btn_add_file, 1)

        self.btn_record_tts = QPushButton(tr("dlg_035", "🎙️ Записати"))
        self.btn_record_tts.setObjectName("primaryBtn")
        self.btn_record_tts.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_record_tts.setToolTip(
            tr("dlg_036", "Озвучити введений вище текст вибраним голосом і зберегти у файл, щоб не переозвучувати постійно")
        )
        self.btn_record_tts.clicked.connect(self._on_record_tts_to_file)
        audio_btn_row.addWidget(self.btn_record_tts, 1)

        self.btn_preview_audio = QPushButton(tr("dlg_003", "▶ Прослухати"))
        self.btn_preview_audio.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_preview_audio.clicked.connect(self._on_preview_saved_audio)
        audio_btn_row.addWidget(self.btn_preview_audio)

        ac_lay.addLayout(audio_btn_row)

        self.audio_status_lbl = QLabel()
        self.audio_status_lbl.setWordWrap(True)
        self.audio_status_lbl.setStyleSheet("border: none; background: transparent; font-size: 12px;")
        ac_lay.addWidget(self.audio_status_lbl)

        lay.addLayout(form)
        lay.addWidget(audio_card)

        hk_row = QHBoxLayout()
        hk_lbl = QLabel(tr("dlg_038", "Гаряча клавіша:"))
        hk_row.addWidget(hk_lbl)
        self.hk_edit = QLineEdit(self._data.get("hotkey", ""))
        self.hk_edit.setReadOnly(True)
        self.hk_edit.setPlaceholderText(tr("dlg_039", "Не призначено (необов'язково)"))
        hk_btn = QPushButton(tr("dlg_040", "⌨️ Вибрати клавішу"))
        hk_btn.clicked.connect(self._record_hk)
        hk_row.addWidget(self.hk_edit, 1)
        hk_row.addWidget(hk_btn)
        lay.addLayout(hk_row)

        lay.addStretch(1)

        btns = QHBoxLayout()
        btns.addStretch(1)
        cancel_b = QPushButton(tr("dlg_026", "Скасувати"))
        cancel_b.clicked.connect(self.reject)
        save_b = QPushButton(tr("dlg_041", "Зберегти"))
        save_b.setObjectName("primaryBtn")
        save_b.clicked.connect(self._on_save)
        btns.addWidget(cancel_b)
        btns.addWidget(save_b)
        lay.addLayout(btns)

        self._refresh_audio_status()
        if auto_pick_file:
            QTimer.singleShot(100, self._on_pick_audio_file)

    def _refresh_audio_status(self) -> None:
        af = str(self._data.get("audio_file", "")).strip()
        exists = bool(af and Path(af).exists())
        self.btn_preview_audio.setVisible(exists)
        if exists:
            self.audio_status_lbl.setText(tr("dlg_f_005", "✅ Прикріплено файл: <b>{filename}</b> (відтворюється миттєво з файлу)", filename=Path(af).name))
            self.audio_status_lbl.setStyleSheet("border: none; background: transparent; color: #34D399; font-size: 12px;")
        else:
            self.audio_status_lbl.setText(
                tr("dlg_042", "ℹ️ Натисніть «🎙️ Записати», щоб озвучити текст і зберегти його у файл, або «📁 Додати файлом», щоб обрати свій звук.")
            )
            self.audio_status_lbl.setStyleSheet("border: none; background: transparent; color: #94A3B8; font-size: 12px;")

    def _on_pick_audio_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            tr("dlg_043", "Оберіть звуковий файл для швидкої фрази"),
            "",
            tr("dlg_044", "Аудіофайли (*.wav *.mp3 *.ogg *.flac);;Усі файли (*.*)"),
        )
        if not file_path:
            return
        src = Path(file_path)
        if not src.exists():
            return
        PHRASES_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        dest = PHRASES_AUDIO_DIR / f"file_{int(time.time() * 1000)}{src.suffix.lower()}"
        try:
            shutil.copy2(src, dest)
            self._data["audio_file"] = str(dest)
            if not self.title_edit.text().strip():
                self.title_edit.setText(src.stem[:24])
            if not self.text_edit.toPlainText().strip():
                self.text_edit.setPlainText(src.stem)
            self._refresh_audio_status()
        except Exception as exc:
            self.audio_status_lbl.setText(tr("dlg_f_006", "⚠️ Не вдалося скопіювати файл: {exc}", exc=exc))
            self.audio_status_lbl.setStyleSheet("border: none; background: transparent; color: #F87171; font-size: 12px;")

    def _on_record_tts_to_file(self) -> None:
        text = self.text_edit.toPlainText().strip()
        if not text:
            self.audio_status_lbl.setText(tr("dlg_045", "⚠️ Спочатку напишіть текст у полі «Текст озвучки» вище!"))
            self.audio_status_lbl.setStyleSheet("border: none; background: transparent; color: #F87171; font-size: 12px;")
            self.text_edit.setFocus()
            return

        self.btn_record_tts.setEnabled(False)
        self.btn_record_tts.setText(tr("dlg_046", "⏳ Запис у файл..."))
        self.audio_status_lbl.setText(tr("dlg_047", "⏳ Генеруємо голос та записуємо у звуковий файл..."))
        self.audio_status_lbl.setStyleSheet("border: none; background: transparent; color: #A5B4FC; font-size: 12px;")
        QApplication.processEvents()

        PHRASES_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        dest = PHRASES_AUDIO_DIR / f"recorded_{int(time.time() * 1000)}.wav"
        ok, msg = tts_manager.synthesize_to_file(text, dest)
        self.btn_record_tts.setEnabled(True)
        self.btn_record_tts.setText(tr("dlg_035", "🎙️ Записати"))

        if ok:
            self._data["audio_file"] = msg
            if not self.title_edit.text().strip():
                self.title_edit.setText(text[:22])
            self._refresh_audio_status()
            tts_manager.play_sample_preview(msg, monitor_only=True)
        else:
            self.audio_status_lbl.setText(f"⚠️ {msg}")
            self.audio_status_lbl.setStyleSheet("border: none; background: transparent; color: #F87171; font-size: 12px;")

    def _on_preview_saved_audio(self) -> None:
        af = str(self._data.get("audio_file", "")).strip()
        if af and Path(af).exists():
            tts_manager.play_sample_preview(af, monitor_only=True)

    def _record_hk(self) -> None:
        dlg = HotkeyCaptureDialog(self._hook, self.hk_edit.text(), tr("dlg_048", "Клавіша для швидкої фрази"), self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.recorded_hotkey is not None:
            self.hk_edit.setText(dlg.recorded_hotkey)

    def _on_save(self) -> None:
        text = self.text_edit.toPlainText().strip()
        af = str(self._data.get("audio_file", "")).strip()
        if not text and not (af and Path(af).exists()):
            self.audio_status_lbl.setText(tr("dlg_049", "⚠️ Введіть текст для запису або додайте звуковий файл!"))
            self.audio_status_lbl.setStyleSheet("border: none; background: transparent; color: #F87171; font-size: 12px;")
            return

        title = self.title_edit.text().strip() or (text[:22] if text else Path(af).stem[:22])
        emoji = self.emoji_edit.text().strip() or "💬"
        self._data["title"] = title
        self._data["text"] = text or title
        self._data["emoji"] = emoji
        self._data["hotkey"] = self.hk_edit.text().strip()
        self.accept()

    def get_result(self) -> Dict[str, str]:
        return self._data


class ProxySettingsDialog(QDialog):
    """
    Просте вікно налаштування додаткових мережевих серверів і дзеркал.
    Створене для простого налаштування мережевого підключення.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("dlg_051", "Мережеве підключення (Проксі та сервери)"))
        self.setMinimumWidth(490)
        self.setMaximumWidth(570)
        self.setStyleSheet(DIALOG_STYLESHEET)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 18)
        lay.setSpacing(12)

        hdr = QLabel(tr("dlg_052", "🛡️ Налаштування мережевого підключення"))
        hdr.setObjectName("dialogTitle")
        hdr.setWordWrap(True)
        lay.addWidget(hdr)

        info_lbl = QLabel(
            tr(
                "dlg_053",
                "Стабільність з’єднання із серверами завантаження може залежати від вашої мережі.<br><br>"
                "• Програма автоматично підбирає доступні дзеркала та додаткові сервери.<br>"
                "• За потреби можна вказати власну адресу SOCKS5 у полі нижче.",
            )
        )
        info_lbl.setWordWrap(True)
        info_lbl.setTextFormat(Qt.TextFormat.RichText)
        info_lbl.setStyleSheet(
            "background-color: #262C3D; border: 1px solid #38415C; border-radius: 10px; "
            "padding: 12px 14px; color: #F1F5F9; font-size: 13px; line-height: 1.45;"
        )
        lay.addWidget(info_lbl)

        self.chk_bypass = QCheckBox(
            tr("dlg_054", "Автоматично використовувати додаткові сервери, якщо основне з’єднання недоступне")
        )
        self.chk_bypass.setStyleSheet("color: #FFFFFF; font-size: 14px; font-weight: 700; padding: 4px 0px;")
        self.chk_bypass.setChecked(bool(config.get("use_download_bypass", True)))
        lay.addWidget(self.chk_bypass)

        custom_lbl = QLabel(tr("dlg_055", "Власний проксі-сервер (залиште порожнім для автоматичного режиму):"))
        custom_lbl.setStyleSheet("color: #CBD5E1; font-size: 12px;")
        lay.addWidget(custom_lbl)

        self.proxy_edit = QLineEdit(str(config.get("custom_proxy_url", "")))
        self.proxy_edit.setPlaceholderText(tr("dlg_056", "Автоматично (або введіть http://ip:port)"))
        lay.addWidget(self.proxy_edit)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        cancel_btn = QPushButton(tr("dlg_026", "Скасувати"))
        cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        save_btn = QPushButton(tr("dlg_041", "Зберегти"))
        save_btn.setObjectName("primaryBtn")
        save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(save_btn)
        lay.addLayout(btn_row)

    def _on_save(self) -> None:
        config.set("use_download_bypass", self.chk_bypass.isChecked())
        config.set("custom_proxy_url", self.proxy_edit.text().strip())
        self.accept()


class ErrorPopupDialog(QDialog):
    """
    Універсальне вікно виведення помилок із перекладом на мову користувача.
    Якщо помилка пов'язана із завантаженням або інтернетом — додатково пояснює,
    що потрібно перевірити інтернет або налаштування мережевого підключення.
    """

    def __init__(
        self,
        error_text: str,
        title: str = tr("dlg_057", "Виникла помилка"),
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(500)
        self.setMaximumWidth(600)
        self.setStyleSheet(DIALOG_STYLESHEET)

        raw_text = str(error_text or "").strip()
        if not raw_text:
            tb_txt = traceback.format_exc()
            if tb_txt and "NoneType: None" not in tb_txt:
                raw_text = tb_txt.strip()
            else:
                raw_text = tr(
                    "dlg.err.empty_fallback",
                    "Виникла невідома помилка без текстового опису. Деталі записано у лог програми.",
                )
        translated = (translate_error_message(raw_text) or raw_text).strip() or raw_text
        self._full_text = translated
        is_net_err = is_network_or_download_error(raw_text) or is_network_or_download_error(translated)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 18)
        lay.setSpacing(12)

        hdr = QLabel(f"⚠️ {title}")
        hdr.setObjectName("dialogTitle")
        hdr.setStyleSheet("color: #FCA5A5; font-size: 17px; font-weight: 700;")
        hdr.setWordWrap(True)
        lay.addWidget(hdr)

        # Поле з текстом помилки (переносить довгі рядки і не розтягує вікно)
        err_box = QTextEdit()
        err_box.setReadOnly(True)
        err_box.setPlainText(translated)
        err_box.setMinimumHeight(80)
        err_box.setMaximumHeight(135)
        err_box.setStyleSheet(
            "QTextEdit { background-color: #291E28; color: #FECACA; border: 1px solid #7F1D1D; "
            "border-radius: 9px; padding: 10px 12px; font-size: 13px; }"
        )
        lay.addWidget(err_box)

        # Якщо помилка пов'язана зі скачуванням або мережею — показуємо зрозумілу пораду про інтернет і мережеве підключення
        if is_net_err:
            hint_card = QFrame()
            hint_card.setStyleSheet(
                "QFrame { background-color: #232A3E; border: 1px solid #485685; border-radius: 10px; }"
            )
            h_lay = QVBoxLayout(hint_card)
            h_lay.setContentsMargins(14, 12, 14, 12)
            h_lay.setSpacing(8)

            hint_lbl = QLabel(
                tr(
                    "dlg_058",
                    "<b>🌐 Чому виникла ця помилка при завантаженні?</b><br>"
                    "Програма автоматично спробувала завантажити файли напряму та через додаткові сервери й дзеркала, "
                    "але з'єднання все одно не вдалося.<br><br>"
                    "<b>💡 Що робити:</b><br>"
                    "1. Перевірте, чи є підключення до інтернету.<br>"
                    "2. Спробуйте іншу мережу або перевірте налаштування підключення на комп'ютері.",
                )
            )
            hint_lbl.setWordWrap(True)
            hint_lbl.setTextFormat(Qt.TextFormat.RichText)
            hint_lbl.setStyleSheet("border: none; background: transparent; color: #E2E8F0; font-size: 13px;")
            h_lay.addWidget(hint_lbl)
            lay.addWidget(hint_card)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        copy_btn = QPushButton(tr("dlg_059", "📋 Скопіювати помилку"))
        copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        copy_btn.clicked.connect(self._copy_error)
        btn_row.addWidget(copy_btn)

        btn_row.addStretch(1)

        ok_btn = QPushButton(tr("dlg_001", "Зрозуміло"))
        ok_btn.setObjectName("primaryBtn")
        ok_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(ok_btn)

        lay.addLayout(btn_row)

    def _copy_error(self) -> None:
        cb = QGuiApplication.clipboard()
        if cb:
            cb.setText(self._full_text)


class LanguageConfirmDialog(QDialog):
    """
    Діалог підтвердження мови для нового користувача перед першим встановленням голосів.
    Питає: «Ваша мова — Українська (UA)?» з кнопками «Так» та «Ні» і списком вибору мови.
    Якщо користувач натискає «Ні» — у цьому ж вікні червоним текстом з'являється підказка
    обрати свою мову зі списку та натиснути «Так».
    Після натискання «Так» зберігає app_language та language_confirmed = True,
    щоб більше ніколи не перепитувати.
    """

    LANGUAGES = [
        ("English (EN)", "en"),
        ("Deutsch (DE)", "de"),
        ("Español (ES)", "es"),
        ("Français (FR)", "fr"),
        ("Polski (PL)", "pl"),
        ("Русский (RU)", "ru"),
        ("Українська (UA)", "uk"),
    ]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("dlg_060", "Вибір мови — SilentVoice"))
        self.setMinimumWidth(460)
        self.setMaximumWidth(520)
        self.setStyleSheet(DIALOG_STYLESHEET)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(14)

        self.question_lbl = QLabel()
        self.question_lbl.setObjectName("dialogTitle")
        self.question_lbl.setWordWrap(True)
        self.question_lbl.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.question_lbl)

        desc_lbl = QLabel(
            tr(
                "dlg_061",
                "Голосові моделі та голоси при встановленні завантажуватимуться лише для обраної вами мови, "
                "щоб не займати зайве місце на комп'ютері.",
            )
        )
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet("color: #CBD5E1; font-size: 13px;")
        lay.addWidget(desc_lbl)

        self.red_warn_lbl = QLabel(tr("dlg_062", "Тоді оберіть свою мову зі списку нижче та натисніть «Так»!"))
        self.red_warn_lbl.setWordWrap(True)
        self.red_warn_lbl.setStyleSheet(
            "background-color: #3B1822; color: #FF6B6B; border: 1px solid #EF4444; "
            "border-radius: 8px; padding: 10px 12px; font-size: 14px; font-weight: 700;"
        )
        self.red_warn_lbl.setVisible(False)
        lay.addWidget(self.red_warn_lbl)

        combo_row = QHBoxLayout()
        combo_row.setSpacing(10)
        combo_lbl = QLabel(tr("dlg_063", "<b>Оберіть мову:</b>"))
        combo_lbl.setStyleSheet("color: #F8FAFC; font-size: 14px;")
        combo_row.addWidget(combo_lbl)

        self.lang_combo = DialogComboBox()
        self.lang_combo.setStyleSheet(
            "QComboBox QAbstractItemView { background-color: #272D3F; color: #FFFFFF; "
            "selection-background-color: #635BFF; border: 1px solid #4A5578; }"
        )
        for title, code in self.LANGUAGES:
            self.lang_combo.addItem(title, code)

        cur_lang = str(config.get("app_language", "uk")).strip().lower()
        idx = self.lang_combo.findData(cur_lang)
        self.lang_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.lang_combo.currentIndexChanged.connect(self._update_question_text)
        combo_row.addWidget(self.lang_combo, 1)
        lay.addLayout(combo_row)

        self._update_question_text()

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch(1)

        self.no_btn = QPushButton(tr("dlg_064", "Ні"))
        self.no_btn.setMinimumWidth(110)
        self.no_btn.setMinimumHeight(36)
        self.no_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.no_btn.clicked.connect(self._on_no_clicked)
        btn_row.addWidget(self.no_btn)

        self.yes_btn = QPushButton(tr("dlg_065", "✅ Так"))
        self.yes_btn.setObjectName("primaryBtn")
        self.yes_btn.setMinimumWidth(130)
        self.yes_btn.setMinimumHeight(36)
        self.yes_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.yes_btn.clicked.connect(self._on_yes_clicked)
        btn_row.addWidget(self.yes_btn)

        lay.addLayout(btn_row)

    def _update_question_text(self) -> None:
        lang_title = self.lang_combo.currentText()
        self.question_lbl.setText(tr("dlg_f_007", "🌐 Ваша мова — <span style='color:#A5B4FC;'>{lang_title}</span>?", lang_title=lang_title))

    def _on_no_clicked(self) -> None:
        self.red_warn_lbl.setVisible(True)
        self.lang_combo.set_error_highlight(True)
        self.lang_combo.setFocus()
        self.lang_combo.showPopup()

    def _on_yes_clicked(self) -> None:
        chosen_lang = str(self.lang_combo.currentData() or "uk")
        config.update({
            "app_language": chosen_lang,
            "language_confirmed": True,
        })
        self.accept()

    def selected_language(self) -> str:
        return str(self.lang_combo.currentData() or "uk")


class _SpinnerWidget(QWidget):
    """Плавна анімація завантаження (обертове градієнтне кільце) на час AI-перекладу."""

    def __init__(self, size: int = 44, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

    def start(self) -> None:
        if not self._timer.isActive():
            self._timer.start()
        self.show()

    def stop(self) -> None:
        self._timer.stop()

    def _tick(self) -> None:
        self._angle = (self._angle + 7) % 360
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pad = 5.0
        rect = QRectF(pad, pad, self.width() - 2 * pad, self.height() - 2 * pad)

        # Фонове кільце
        bg_pen = QPen(QColor("#2D334D"), 4.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(bg_pen)
        painter.drawEllipse(rect)

        # Градієнтна дуга, що обертається
        grad = QConicalGradient(rect.center(), -self._angle)
        grad.setColorAt(0.0, QColor("#38BDF8"))
        grad.setColorAt(0.45, QColor("#7C5CFF"))
        grad.setColorAt(1.0, QColor(124, 92, 255, 0))
        arc_pen = QPen(grad, 4.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(arc_pen)
        painter.drawArc(rect, -self._angle * 16, 250 * 16)


class _LocaleTranslateThread(QThread):
    """Фоновий потік для перекладу інтерфейсу частинами через Groq API."""

    progress = Signal(int, int, str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, lang_code: str, native_name: str, english_name: str, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._lang_code = lang_code
        self._native_name = native_name
        self._english_name = english_name
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            dest = translate_locale_file(
                target_lang=self._lang_code,
                target_native_name=self._native_name,
                target_english_name=self._english_name,
                progress_cb=lambda done, total, label: self.progress.emit(done, total, label),
                cancel_check=lambda: self._cancelled,
            )
            self.finished_ok.emit(str(dest))
        except Exception as exc:
            self.failed.emit(str(exc))


class FirstRunLanguageDialog(QDialog):
    """
    Діалог першого запуску, коли системна мова не входить до вбудованих перекладів.
    Питає (мовою користувача через швидкий переклад), чи це його мова,
    пропонує перекласти весь інтерфейс або обрати одну з вбудованих мов (за замовчуванням English).
    """

    TTS_SUPPORTED_LANGS = ("uk", "en", "pl", "de", "fr", "es", "ru")

    def __init__(self, sys_code: str, native_name: str, english_name: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._sys_code = str(sys_code or "en").strip().lower()
        self._native_name = native_name or english_name or self._sys_code.upper()
        self._english_name = english_name or self._native_name
        self._worker: Optional[_LocaleTranslateThread] = None

        base_texts = {
            "window_title": "Language Setup — SilentVoice",
            "question": f"Is your language {self._native_name} ({self._sys_code.upper()})?",
            "subtitle": (
                f"SilentVoice doesn't have a built-in {self._native_name} translation yet. "
                "We can automatically translate the entire interface into your language, "
                "or you can choose one of the built-in languages below:"
            ),
            "builtin_lbl": "Built-in languages:",
            "translate_btn": f"Translate interface ({self._native_name})",
            "use_selected_btn": "Use selected language",
            "translating_status": f"Translating interface into {self._native_name}...",
            "err_prefix": "Translation error",
        }
        self._t = translate_ui_prompt_texts(
            base_texts,
            target_lang=self._sys_code,
            target_english_name=self._english_name,
        )

        self.setWindowTitle(self._t["window_title"])
        self.setMinimumWidth(500)
        self.setMaximumWidth(720)
        self.setStyleSheet(DIALOG_STYLESHEET)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 22, 24, 22)
        lay.setSpacing(14)

        self.question_lbl = QLabel(f"🌐 {self._t['question']}")
        self.question_lbl.setObjectName("dialogTitle")
        self.question_lbl.setWordWrap(True)
        lay.addWidget(self.question_lbl)

        self.sub_lbl = QLabel(self._t["subtitle"])
        self.sub_lbl.setWordWrap(True)
        self.sub_lbl.setStyleSheet("color: #CBD5E1; font-size: 13px; line-height: 1.4;")
        lay.addWidget(self.sub_lbl)

        combo_row = QHBoxLayout()
        combo_row.setSpacing(10)
        combo_lbl = QLabel(f"<b>{self._t['builtin_lbl']}</b>")
        combo_lbl.setStyleSheet("color: #F8FAFC; font-size: 13px;")
        combo_row.addWidget(combo_lbl)

        self.lang_combo = DialogComboBox()
        self.lang_combo.setStyleSheet(
            "QComboBox QAbstractItemView { background-color: #272D3F; color: #FFFFFF; "
            "selection-background-color: #635BFF; border: 1px solid #4A5578; }"
        )
        for item in list_available_ui_languages():
            self.lang_combo.addItem(item["label"], item["code"])
        en_idx = self.lang_combo.findData("en")
        self.lang_combo.setCurrentIndex(en_idx if en_idx >= 0 else 0)
        combo_row.addWidget(self.lang_combo, 1)
        lay.addLayout(combo_row)

        # Панель красивої анімації під час перекладу
        self.loading_box = QFrame()
        self.loading_box.setStyleSheet(
            "QFrame { background-color: #1B1F32; border: 1px solid #353D5E; border-radius: 10px; }"
        )
        load_lay = QVBoxLayout(self.loading_box)
        load_lay.setContentsMargins(16, 14, 16, 14)
        load_lay.setSpacing(10)

        spin_row = QHBoxLayout()
        spin_row.setSpacing(12)
        self.spinner = _SpinnerWidget(40, self.loading_box)
        spin_row.addWidget(self.spinner)

        self.status_lbl = QLabel(self._t["translating_status"])
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("color: #E2E8F0; font-size: 13px; font-weight: 600; border: none;")
        spin_row.addWidget(self.status_lbl, 1)
        load_lay.addLayout(spin_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(16)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background: #121524; border: 1px solid #2E344D; border-radius: 8px; "
            "color: #E2E8F0; font-size: 11px; text-align: center; }"
            "QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "stop:0 #7C5CFF, stop:1 #38BDF8); border-radius: 7px; }"
        )
        load_lay.addWidget(self.progress_bar)
        self.loading_box.setVisible(False)
        lay.addWidget(self.loading_box)

        self.error_lbl = QLabel("")
        self.error_lbl.setWordWrap(True)
        self.error_lbl.setStyleSheet(
            "background-color: #3B1822; color: #FF8A8A; border: 1px solid #EF4444; "
            "border-radius: 8px; padding: 8px 12px; font-size: 12px;"
        )
        self.error_lbl.setVisible(False)
        lay.addWidget(self.error_lbl)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch(1)

        self.translate_btn = QPushButton(f"✨ {self._t['translate_btn']}")
        self.translate_btn.setMinimumHeight(38)
        self.translate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.translate_btn.clicked.connect(self._on_translate_clicked)
        btn_row.addWidget(self.translate_btn)

        self.use_btn = QPushButton(self._t["use_selected_btn"])
        self.use_btn.setObjectName("primaryBtn")
        self.use_btn.setDefault(True)
        self.use_btn.setMinimumHeight(38)
        self.use_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.use_btn.clicked.connect(self._on_use_selected)
        btn_row.addWidget(self.use_btn)

        lay.addLayout(btn_row)

    def _on_use_selected(self) -> None:
        from core.config import sync_default_quick_phrases_for_lang
        chosen = str(self.lang_combo.currentData() or "en").strip().lower() or "en"
        set_ui_language(chosen)
        config.update({
            "ui_language": chosen,
            "app_language": chosen,
            "language_confirmed": True,
            "first_run_lang_checked": True,
        })
        sync_default_quick_phrases_for_lang(chosen)
        self.accept()

    def _on_translate_clicked(self) -> None:
        self.error_lbl.setVisible(False)
        self.lang_combo.setEnabled(False)
        self.use_btn.setEnabled(False)
        self.translate_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_lbl.setText(self._t["translating_status"])
        self.loading_box.setVisible(True)
        self.spinner.start()

        self._worker = _LocaleTranslateThread(self._sys_code, self._native_name, self._english_name, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished_ok)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, done: int, total: int, label: str) -> None:
        pct = int((done / max(total, 1)) * 100)
        self.progress_bar.setValue(pct)
        self.status_lbl.setText(f"{self._t['translating_status']} ({label})")

    def _on_finished_ok(self, _dest_path: str) -> None:
        from core.config import sync_default_quick_phrases_for_lang
        self.spinner.stop()
        self.progress_bar.setValue(100)
        set_ui_language(self._sys_code)
        config.update({
            "ui_language": self._sys_code,
            "app_language": self._sys_code,
            "language_confirmed": True,
            "first_run_lang_checked": True,
        })
        sync_default_quick_phrases_for_lang(self._sys_code)
        self.accept()

    def _on_failed(self, err: str) -> None:
        self.spinner.stop()
        self.loading_box.setVisible(False)
        self.lang_combo.setEnabled(True)
        self.use_btn.setEnabled(True)
        self.translate_btn.setEnabled(True)
        self.error_lbl.setText(f"{self._t['err_prefix']}: {err}")
        self.error_lbl.setVisible(True)

    def reject(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(1500)
        self.spinner.stop()
        if not bool(config.get("first_run_lang_checked", False)):
            set_ui_language("en")
            config.update({
                "ui_language": "en",
                "app_language": "en",
                "language_confirmed": True,
                "first_run_lang_checked": True,
            })
        super().reject()


class AISettingsDialog(QDialog):
    """Вікно налаштувань AI-коректора орфографії: увімкнення, головний/запасний сервіс та промпт."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("dlg_066", "⚙️ Налаштування AI Орфографії"))
        self.setMinimumWidth(560)
        self.setMaximumWidth(680)
        self.setStyleSheet(DIALOG_STYLESHEET)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(12)

        self.chk_enabled = QCheckBox(tr("dlg_067", "Увімкнути AI-коректор тексту перед відправкою в TTS та OBS"))
        self.chk_enabled.setChecked(bool(config.get("ai_spellcheck_enabled", False)))
        self.chk_enabled.setStyleSheet("font-weight: 600; color: #F1F5F9; font-size: 13.5px;")
        self.chk_enabled.setToolTip(
            tr("dlg_068", "Коли увімкнено, текст перед озвучкою та показом у OBS спершу виправляється через обраний AI-сервіс")
        )
        lay.addWidget(self.chk_enabled)

        form_card = QFrame()
        form_card.setObjectName("card")
        f_lay = QFormLayout(form_card)
        f_lay.setContentsMargins(14, 12, 14, 12)
        f_lay.setSpacing(10)

        ai_items = list(config.get("ai_items", []))

        self.primary_combo = DialogComboBox()
        self.primary_combo.setToolTip(tr("dlg_069", "Основний AI-сервіс, який першим отримує текст на виправлення"))
        self.primary_combo.addItem(tr("dlg_070", "— Не вибрано —"), "")
        for item in ai_items:
            self.primary_combo.addItem(get_ai_item_title(item), str(item.get("id", "")))
        cur_prim = str(config.get("ai_primary_id", ""))
        p_idx = self.primary_combo.findData(cur_prim)
        if p_idx >= 0:
            self.primary_combo.setCurrentIndex(p_idx)
        elif ai_items:
            self.primary_combo.setCurrentIndex(1)
        f_lay.addRow(tr("dlg_071", "Головний AI-сервіс:"), self.primary_combo)

        self.fallback_combo = DialogComboBox()
        self.fallback_combo.setToolTip(tr("dlg_072", "Резервний AI-сервіс на випадок, якщо головний не відповідає або вичерпано ліміт"))
        self.fallback_combo.addItem(tr("dlg_073", "Не використовувати"), "")
        for item in ai_items:
            self.fallback_combo.addItem(get_ai_item_title(item), str(item.get("id", "")))
        cur_fb = str(config.get("ai_fallback_id", ""))
        f_idx = self.fallback_combo.findData(cur_fb)
        if f_idx >= 0:
            self.fallback_combo.setCurrentIndex(f_idx)
        f_lay.addRow(tr("dlg_074", "Запасний AI-сервіс:"), self.fallback_combo)

        lay.addWidget(form_card)

        prompt_hdr = QHBoxLayout()
        prompt_lbl = QLabel(tr("dlg_075", "Системний промпт для мови:"))
        prompt_lbl.setStyleSheet("font-weight: 600; color: #E2E8F0;")
        prompt_hdr.addWidget(prompt_lbl)

        self._draft_prompts: Dict[str, str] = dict(config.get("ai_spellcheck_prompts", {}) or {})
        self._cur_prompt_lang = str(config.get("app_language", "uk")).strip().lower() or "uk"

        self.prompt_lang_combo = DialogComboBox()
        self.prompt_lang_combo.setToolTip(tr("dlg_076", "Оберіть мову, для якої налаштовується промпт"))
        for title, code in LanguageConfirmDialog.LANGUAGES:
            self.prompt_lang_combo.addItem(title, code)
        pl_idx = self.prompt_lang_combo.findData(self._cur_prompt_lang)
        if pl_idx >= 0:
            self.prompt_lang_combo.setCurrentIndex(pl_idx)
        self.prompt_lang_combo.currentIndexChanged.connect(self._on_prompt_lang_changed)
        prompt_hdr.addWidget(self.prompt_lang_combo)

        prompt_hdr.addWidget(
            HelpButton(
                tr("dlg_077", "Що таке системний промпт?"),
                tr(
                    "dlg_078",
                    "<b>Що це?</b><br>"
                    "Інструкція, за якою AI виправляє ваш текст (наприклад: прибирати помилки, ставити коми, не змінювати сленг).<br><br>"
                    "• Для кожної мови інтерфейсу зберігається <b>свій окремий промпт</b>.<br>"
                    "• Програма автоматично бере той промпт, який відповідає вибраній мові.<br><br>"
                    "<i>Якщо не знаєте, що писати — стандартний промпт уже налаштований оптимально.</i>",
                ),
            )
        )
        prompt_hdr.addStretch(1)
        reset_prompt_btn = QPushButton(tr("dlg_079", "↺ Скинути промпт"))
        reset_prompt_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reset_prompt_btn.setToolTip(tr("dlg_080", "Повернути стандартний рекомендований промпт для обраної мови"))
        reset_prompt_btn.clicked.connect(self._on_reset_prompt_clicked)
        prompt_hdr.addWidget(reset_prompt_btn)
        lay.addLayout(prompt_hdr)

        self.prompt_edit = QTextEdit()
        self.prompt_edit.setMinimumHeight(165)
        self.prompt_edit.setToolTip(tr("dlg_081", "Інструкція, якою керується AI при виправленні вашого тексту"))
        self.prompt_edit.setPlainText(get_effective_ai_prompt(self._cur_prompt_lang))
        lay.addWidget(self.prompt_edit)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        cancel_btn = QPushButton(tr("dlg_026", "Скасувати"))
        cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        save_btn = QPushButton(tr("dlg_082", "💾 Зберегти"))
        save_btn.setObjectName("primaryBtn")
        save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(save_btn)
        lay.addLayout(btn_row)

    def _stash_current_prompt_draft(self) -> None:
        raw = self.prompt_edit.toPlainText().strip()
        default_for_lang = get_default_ai_prompt(self._cur_prompt_lang).strip()
        if not raw or raw == default_for_lang:
            self._draft_prompts.pop(self._cur_prompt_lang, None)
        else:
            self._draft_prompts[self._cur_prompt_lang] = raw

    def _on_prompt_lang_changed(self) -> None:
        self._stash_current_prompt_draft()
        self._cur_prompt_lang = str(self.prompt_lang_combo.currentData() or "uk")
        custom = self._draft_prompts.get(self._cur_prompt_lang, "").strip()
        self.prompt_edit.setPlainText(custom or get_default_ai_prompt(self._cur_prompt_lang))

    def _on_reset_prompt_clicked(self) -> None:
        self._draft_prompts.pop(self._cur_prompt_lang, None)
        self.prompt_edit.setPlainText(get_default_ai_prompt(self._cur_prompt_lang))

    def _on_save(self) -> None:
        self._stash_current_prompt_draft()
        config.update({
            "ai_spellcheck_enabled": self.chk_enabled.isChecked(),
            "ai_primary_id": str(self.primary_combo.currentData() or ""),
            "ai_fallback_id": str(self.fallback_combo.currentData() or ""),
            "ai_spellcheck_prompts": self._draft_prompts,
            "ai_spellcheck_prompt": "",
        })
        self.accept()


class _AsyncTaskBridge(QObject):
    """Передає результат фонової задачі назад у головний потік Qt."""

    finished = Signal(object, str)


def run_in_background(
    parent: Optional[QObject],
    fn: Callable[[], Any],
    on_done: Callable[[Any, str], None],
) -> None:
    """Запускає fn() у фоновому потоці та викликає on_done(result, error) у GUI-потоці."""
    bridge = _AsyncTaskBridge(parent)
    bridge.finished.connect(on_done)
    # Утримуємо посилання на bridge до завершення, навіть якщо parent=None
    setattr(bridge, "_self_ref", bridge)

    def _cleanup_and_deliver(res: Any, err: str) -> None:
        if hasattr(bridge, "_self_ref"):
            delattr(bridge, "_self_ref")
        bridge.deleteLater()

    bridge.finished.connect(_cleanup_and_deliver)

    def _worker() -> None:
        try:
            res = fn()
            bridge.finished.emit(res, "")
        except Exception as exc:
            bridge.finished.emit(None, str(exc))

    threading.Thread(target=_worker, daemon=True).start()


def _create_secret_input_row(
    placeholder: str,
    initial_value: str = "",
    parent: Optional[QWidget] = None,
) -> tuple[QWidget, QLineEdit]:
    """Створює рядок із прихованим полем вводу та кнопкою «Показати / Сховати»."""
    row_w = QWidget(parent)
    lay = QHBoxLayout(row_w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)

    edit = QLineEdit(initial_value)
    edit.setEchoMode(QLineEdit.EchoMode.Password)
    edit.setPlaceholderText(placeholder)
    lay.addWidget(edit, 1)

    btn = QPushButton(tr("ui_145", "Показати"))
    btn.setMinimumWidth(86)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)

    def _toggle() -> None:
        if edit.echoMode() == QLineEdit.EchoMode.Password:
            edit.setEchoMode(QLineEdit.EchoMode.Normal)
            btn.setText(tr("ui_146", "Сховати"))
        else:
            edit.setEchoMode(QLineEdit.EchoMode.Password)
            btn.setText(tr("ui_145", "Показати"))

    btn.clicked.connect(_toggle)
    lay.addWidget(btn)
    return row_w, edit


def _render_qr_pixmap(url: str, size: int = 212) -> QPixmap:
    """Рендерить URL авторизації Telegram (tg://login?token=...) у чіткий QPixmap QR-коду."""
    from core.telegram_silero import ensure_qrcode_installed

    ensure_qrcode_installed()
    import qrcode

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=1,
        border=3,
    )
    qr.add_data(url)
    qr.make(fit=True)
    matrix = qr.get_matrix()
    n = len(matrix)
    img = QImage(n, n, QImage.Format.Format_RGB32)
    white = QColor(255, 255, 255).rgb()
    dark = QColor(15, 23, 42).rgb()
    for y, row in enumerate(matrix):
        for x, cell in enumerate(row):
            img.setPixel(x, y, dark if cell else white)
    return QPixmap.fromImage(img).scaled(
        size,
        size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.FastTransformation,
    )


class TelegramAuthDialog(QDialog):
    """Авторизація в Telegram (QR-код або телефон + API ID/Hash → код → пароль 2FA) для @silero_voice_bot."""

    MAX_QR_AUTO_REFRESH = 3

    def __init__(self, item_id: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        from core.telegram_silero import load_default_api_credentials

        self._item_id = item_id
        self._busy = False
        self._closed = False
        self._qr_gen_id = 0
        self._qr_auto_retries = 0
        self._qr_remaining_sec = 0
        self._smart_pasting = False

        self.setWindowTitle(tr("tg.auth.title", "Підключення Telegram (@silero_voice_bot)"))
        screen = self.screen() or QGuiApplication.primaryScreen()
        avail = screen.availableGeometry() if screen is not None else None
        max_w = max(360, avail.width() - 32) if avail is not None else 900
        max_h = max(360, avail.height() - 48) if avail is not None else 800
        self.setMinimumSize(min(480, max_w), min(360, max_h))
        self.setMaximumSize(max_w, max_h)
        self.setStyleSheet(DIALOG_STYLESHEET)

        item = next((x for x in config.get("online_items", []) if str(x.get("id")) == item_id), {})
        self._proxy_mode = str(item.get("proxy_mode", "auto") or "auto")
        self._proxy_url = str(item.get("proxy_url", "") or "")

        saved_item_id = str(item.get("tg_api_id", "") or "").strip()
        saved_item_hash = str(item.get("tg_api_hash", "") or "").strip()
        def_id, def_hash, _def_src = load_default_api_credentials()
        self._has_default_keys = bool(def_id > 0 and def_hash)
        if saved_item_id == str(def_id) and saved_item_hash == def_hash:
            saved_item_id = ""
            saved_item_hash = ""
        self._has_custom_override = bool(saved_item_id and saved_item_hash)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(12)

        self.err_lbl = QLabel("")
        self.err_lbl.setWordWrap(True)
        self.err_lbl.setVisible(False)
        self.err_lbl.setStyleSheet(
            "background-color: #450A0A; color: #FCA5A5; border: 1px solid #991B1B; "
            "border-radius: 6px; padding: 8px 10px; font-size: 12.5px;"
        )
        root.addWidget(self.err_lbl)

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self._scroll.viewport().setStyleSheet("background: transparent;")

        self.stack = QStackedWidget()
        self.stack.setStyleSheet("QStackedWidget { background: transparent; }")
        self._scroll.setWidget(self.stack)
        root.addWidget(self._scroll, 1)

        # Крок 0: Вибір способу входу (QR-код або Телефон) + API-ключі
        p_cred = QWidget()
        c_lay = QVBoxLayout(p_cred)
        c_lay.setContentsMargins(0, 0, 0, 0)
        c_lay.setSpacing(10)

        # Перемикач режиму: QR-код | За номером телефону
        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        self.btn_mode_qr = QPushButton(tr("tg.auth.tab_qr", "📷 Вхід по QR-коду"))
        self.btn_mode_qr.setCheckable(True)
        self.btn_mode_qr.setChecked(True)
        self.btn_mode_qr.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_mode_qr.clicked.connect(lambda: self._switch_auth_mode("qr"))

        self.btn_mode_phone = QPushButton(tr("tg.auth.tab_phone", "📱 За номером телефону"))
        self.btn_mode_phone.setCheckable(True)
        self.btn_mode_phone.setChecked(False)
        self.btn_mode_phone.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_mode_phone.clicked.connect(lambda: self._switch_auth_mode("phone"))

        tab_css = (
            "QPushButton { background-color: #262C3D; color: #CBD5E1; border: 1px solid #38415C; "
            "border-radius: 8px; padding: 7px 12px; font-weight: 600; }"
            "QPushButton:checked { background-color: #5B53F6; color: #FFFFFF; border-color: #7C5CFF; }"
        )
        self.btn_mode_qr.setStyleSheet(tab_css)
        self.btn_mode_phone.setStyleSheet(tab_css)
        mode_row.addWidget(self.btn_mode_qr, 1)
        mode_row.addWidget(self.btn_mode_phone, 1)
        c_lay.addLayout(mode_row)

        # Блок API ID / API Hash (прихований під спойлер, якщо вже є стандартні/збережені ключі програми)
        if self._has_default_keys and not self._has_custom_override:
            def_row = QHBoxLayout()
            def_row.setSpacing(8)
            def_badge = QLabel(tr("tg.auth.using_default_keys", "✅ Використовуються стандартні API-ключі програми"))
            def_badge.setStyleSheet("color: #34D399; font-size: 12px; font-weight: 600;")
            def_badge.setWordWrap(True)
            def_row.addWidget(def_badge, 1)

            self.btn_toggle_custom_keys = QPushButton(tr("tg.auth.custom_keys_btn", "⚙️ Свої API-ключі"))
            self.btn_toggle_custom_keys.setCheckable(True)
            self.btn_toggle_custom_keys.setChecked(False)
            self.btn_toggle_custom_keys.setCursor(Qt.CursorShape.PointingHandCursor)
            self.btn_toggle_custom_keys.clicked.connect(self._on_toggle_custom_keys)
            def_row.addWidget(self.btn_toggle_custom_keys)
            c_lay.addLayout(def_row)
        else:
            self.btn_toggle_custom_keys = None

        self.api_box = QFrame()
        self.api_box.setObjectName("apiBox")
        self.api_box.setStyleSheet(
            "#apiBox { background-color: #202536; border: 1px solid #333C57; border-radius: 8px; }"
        )
        api_lay = QVBoxLayout(self.api_box)
        api_lay.setContentsMargins(12, 10, 12, 10)
        api_lay.setSpacing(8)

        hint_lbl = QLabel(
            tr(
                "tg.auth.credentials_hint_smart",
                "Вставте <b>API ID</b> та <b>API Hash</b> з <a href='https://my.telegram.org/apps' style='color:#60A5FA;'>my.telegram.org/apps</a> "
                "(можна вставити скопійований блок цілком — поля заповняться самі).",
            )
        )
        hint_lbl.setOpenExternalLinks(True)
        hint_lbl.setWordWrap(True)
        hint_lbl.setStyleSheet("color: #CBD5E1; font-size: 12px; border: none; background: transparent;")
        api_lay.addWidget(hint_lbl)

        api_form = QFormLayout()
        api_form.setSpacing(6)
        id_w, self.api_id_edit = _create_secret_input_row("12345678", saved_item_id, self.api_box)
        hash_w, self.api_hash_edit = _create_secret_input_row("0123456789abcdef...", saved_item_hash, self.api_box)
        self.api_id_edit.textChanged.connect(lambda val: self._try_smart_paste(val))
        self.api_hash_edit.textChanged.connect(lambda val: self._try_smart_paste(val))
        api_form.addRow(tr("tg.auth.api_id", "API ID:"), id_w)
        api_form.addRow(tr("tg.auth.api_hash", "API Hash:"), hash_w)
        api_lay.addLayout(api_form)

        self.chk_remember_app_keys = QCheckBox(
            tr("tg.auth.remember_app_keys", "💾 Запам'ятати ці API-ключі локально (щоб не вводити при наступних входах)")
        )
        self.chk_remember_app_keys.setChecked(True)
        self.chk_remember_app_keys.setStyleSheet("color: #CBD5E1; font-size: 11.5px; border: none; background: transparent;")
        api_lay.addWidget(self.chk_remember_app_keys)

        self.api_box.setVisible(not (self._has_default_keys and not self._has_custom_override))
        c_lay.addWidget(self.api_box)

        # Під-стек: 0 = QR-код, 1 = Номер телефону
        self.mode_stack = QStackedWidget()
        c_lay.addWidget(self.mode_stack)

        # Під-сторінка QR
        p_qr = QWidget()
        qr_lay = QVBoxLayout(p_qr)
        qr_lay.setContentsMargins(0, 4, 0, 0)
        qr_lay.setSpacing(10)

        qr_frame = QFrame()
        qr_frame.setStyleSheet(
            "QFrame { background-color: #FFFFFF; border-radius: 12px; padding: 6px; }"
        )
        qrf_lay = QVBoxLayout(qr_frame)
        qrf_lay.setContentsMargins(8, 8, 8, 8)
        qrf_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.qr_img_lbl = QLabel(tr("tg.auth.qr_placeholder", "Натисніть «Показати QR-код»"))
        self.qr_img_lbl.setFixedSize(216, 216)
        self.qr_img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.qr_img_lbl.setWordWrap(True)
        self.qr_img_lbl.setStyleSheet("color: #334155; font-size: 12.5px; font-weight: 600; background: #FFFFFF;")
        qrf_lay.addWidget(self.qr_img_lbl)

        qr_wrap = QHBoxLayout()
        qr_wrap.addStretch(1)
        qr_wrap.addWidget(qr_frame)
        qr_wrap.addStretch(1)
        qr_lay.addLayout(qr_wrap)

        qr_steps = QLabel(
            tr(
                "tg.auth.qr_steps",
                "1. Відкрийте <b>Telegram</b> на телефоні → <b>Налаштування → Пристрої</b>.<br>"
                "2. Натисніть <b>Підключити пристрій</b> і наведіть камеру на QR-код.",
            )
        )
        qr_steps.setWordWrap(True)
        qr_steps.setAlignment(Qt.AlignmentFlag.AlignCenter)
        qr_steps.setStyleSheet("color: #CBD5E1; font-size: 12.5px;")
        qr_lay.addWidget(qr_steps)

        qr_btn_row = QHBoxLayout()
        self.qr_countdown_lbl = QLabel("")
        self.qr_countdown_lbl.setStyleSheet("color: #A5B4FC; font-size: 12px;")
        qr_btn_row.addWidget(self.qr_countdown_lbl, 1)

        self.btn_refresh_qr = QPushButton(tr("tg.auth.refresh_qr", "🔄 Показати / Оновити QR"))
        self.btn_refresh_qr.setObjectName("primaryBtn")
        self.btn_refresh_qr.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_refresh_qr.clicked.connect(self._on_refresh_qr_clicked)
        qr_btn_row.addWidget(self.btn_refresh_qr)
        qr_lay.addLayout(qr_btn_row)
        self.mode_stack.addWidget(p_qr)

        # Під-сторінка Номер телефону
        p_phone = QWidget()
        ph_lay = QVBoxLayout(p_phone)
        ph_lay.setContentsMargins(0, 6, 0, 0)
        ph_lay.setSpacing(10)

        ph_form = QFormLayout()
        ph_form.setSpacing(8)
        phone_w, self.phone_edit = _create_secret_input_row("+380991234567", str(item.get("tg_phone", "")), p_phone)
        ph_form.addRow(tr("tg.auth.phone", "Телефон:"), phone_w)
        ph_lay.addLayout(ph_form)

        row0 = QHBoxLayout()
        row0.addStretch(1)
        self.btn_req_code = QPushButton(tr("tg.auth.request_code", "Надіслати код"))
        self.btn_req_code.setObjectName("primaryBtn")
        self.btn_req_code.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_req_code.clicked.connect(self._on_request_code)
        row0.addWidget(self.btn_req_code)
        ph_lay.addLayout(row0)
        self.mode_stack.addWidget(p_phone)

        self.stack.addWidget(p_cred)

        # Крок 1: Код підтвердження
        p_code = QWidget()
        k_lay = QVBoxLayout(p_code)
        k_lay.setContentsMargins(0, 0, 0, 0)
        k_lay.setSpacing(10)

        code_hint = QLabel(tr("tg.auth.code_hint", "Введіть код підтвердження, який надійшов у ваш Telegram (або SMS):"))
        code_hint.setWordWrap(True)
        code_hint.setStyleSheet("color: #CBD5E1; font-size: 13px;")
        k_lay.addWidget(code_hint)

        self.code_edit = QLineEdit()
        self.code_edit.setPlaceholderText("12345")
        self.code_edit.returnPressed.connect(self._on_submit_code)
        k_lay.addWidget(self.code_edit)

        row1 = QHBoxLayout()
        btn_back = QPushButton(tr("tg.auth.back", "Назад"))
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.clicked.connect(lambda: self._go_step(0))
        row1.addWidget(btn_back)
        row1.addStretch(1)
        self.btn_sign_in = QPushButton(tr("tg.auth.sign_in", "Увійти"))
        self.btn_sign_in.setObjectName("primaryBtn")
        self.btn_sign_in.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_sign_in.clicked.connect(self._on_submit_code)
        row1.addWidget(self.btn_sign_in)
        k_lay.addLayout(row1)
        self.stack.addWidget(p_code)

        # Крок 2: Пароль двофакторної аутентифікації (2FA)
        p_pw = QWidget()
        pw_lay = QVBoxLayout(p_pw)
        pw_lay.setContentsMargins(0, 0, 0, 0)
        pw_lay.setSpacing(10)

        pw_hint = QLabel(tr("tg.auth.password_hint", "На акаунті увімкнено двофакторний захист (2FA). Введіть хмарний пароль Telegram:"))
        pw_hint.setWordWrap(True)
        pw_hint.setStyleSheet("color: #CBD5E1; font-size: 13px;")
        pw_lay.addWidget(pw_hint)

        pw_w, self.password_edit = _create_secret_input_row(tr("tg.auth.password_placeholder", "Пароль 2FA"), "", p_pw)
        self.password_edit.returnPressed.connect(self._on_submit_password)
        pw_lay.addWidget(pw_w)

        row2 = QHBoxLayout()
        btn_pw_back = QPushButton(tr("tg.auth.back", "Назад"))
        btn_pw_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_pw_back.clicked.connect(lambda: self._go_step(0))
        row2.addWidget(btn_pw_back)
        row2.addStretch(1)
        self.btn_check_pw = QPushButton(tr("tg.auth.verify_password", "Підтвердити пароль"))
        self.btn_check_pw.setObjectName("primaryBtn")
        self.btn_check_pw.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_check_pw.clicked.connect(self._on_submit_password)
        row2.addWidget(self.btn_check_pw)
        pw_lay.addLayout(row2)
        self.stack.addWidget(p_pw)

        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("color: #94A3B8; font-size: 12px;")
        self.status_lbl.setVisible(False)
        root.addWidget(self.status_lbl)

        self._qr_timer = QTimer(self)
        self._qr_timer.setInterval(1000)
        self._qr_timer.timeout.connect(self._on_qr_tick)
        self._size_sync_timer = QTimer(self)
        self._size_sync_timer.setSingleShot(True)
        self._size_sync_timer.timeout.connect(self._sync_dialog_size)
        self._sync_dialog_size()

        # Якщо API-ключі вже доступні — одразу запускаємо QR-код автоматично
        if self._resolve_dialog_api_creds() is not None:
            QTimer.singleShot(60, self._start_qr_flow)

    def _try_smart_paste(self, raw_val: str) -> None:
        if self._smart_pasting:
            return
        from core.telegram_silero import parse_telegram_app_credentials

        aid, ahash = parse_telegram_app_credentials(raw_val)
        if aid and ahash:
            self._smart_pasting = True
            try:
                self.api_id_edit.setText(str(aid))
                self.api_hash_edit.setText(ahash)
            finally:
                self._smart_pasting = False

    def _sync_dialog_size(self) -> None:
        if self._closed:
            return
        screen = self.screen() or QGuiApplication.primaryScreen()
        avail = screen.availableGeometry() if screen is not None else None
        max_w = max(360, avail.width() - 32) if avail is not None else 900
        max_h = max(360, avail.height() - 48) if avail is not None else 800

        lay = self.layout()
        if lay is not None:
            lay.activate()
            margins = lay.contentsMargins()
            spacing = max(0, lay.spacing())
        else:
            margins = self.contentsMargins()
            spacing = 10

        cur_page = self.stack.currentWidget()
        page_sh = cur_page.sizeHint() if cur_page is not None else self.stack.sizeHint()
        avail_lbl_w = max(320, max(self.minimumWidth(), self.width()) - margins.left() - margins.right())
        extra_h = margins.top() + margins.bottom()
        if self.err_lbl.isVisible():
            extra_h += max(self.err_lbl.sizeHint().height(), self.err_lbl.heightForWidth(avail_lbl_w)) + spacing
        if self.status_lbl.isVisible():
            extra_h += max(self.status_lbl.sizeHint().height(), self.status_lbl.heightForWidth(avail_lbl_w)) + spacing

        target_w = min(max_w, max(self.minimumWidth(), page_sh.width() + margins.left() + margins.right() + 18))
        target_h = min(max_h, max(self.minimumHeight(), page_sh.height() + extra_h + 6))
        if self.width() != target_w or self.height() != target_h:
            self.resize(target_w, target_h)

    def _schedule_sync_dialog_size(self) -> None:
        self._size_sync_timer.start(0)

    def _on_toggle_custom_keys(self, checked: bool) -> None:
        self.api_box.setVisible(bool(checked))
        self._schedule_sync_dialog_size()

    def _resolve_dialog_api_creds(self) -> Optional[tuple[int, str]]:
        from core.telegram_silero import load_default_api_credentials

        id_s = self.api_id_edit.text().strip()
        h_s = self.api_hash_edit.text().strip()
        if id_s.isdigit() and int(id_s) > 0 and h_s:
            return int(id_s), h_s
        if not (self.btn_toggle_custom_keys and self.btn_toggle_custom_keys.isChecked()):
            def_id, def_hash, _ = load_default_api_credentials()
            if def_id > 0 and def_hash:
                return def_id, def_hash
        return None

    def _validate_and_get_api_creds(self) -> Optional[tuple[int, str]]:
        creds = self._resolve_dialog_api_creds()
        if creds is not None:
            return creds
        id_s = self.api_id_edit.text().strip()
        if not id_s.isdigit():
            if not self.api_box.isVisible():
                self.api_box.setVisible(True)
            self.api_id_edit.setFocus()
            self._set_error(tr("tg.auth.err_api_id", "API ID має бути цілим числом (отримайте на my.telegram.org)."))
            return None
        if not self.api_hash_edit.text().strip():
            if not self.api_box.isVisible():
                self.api_box.setVisible(True)
            self.api_hash_edit.setFocus()
            self._set_error(tr("tg.auth.err_api_hash", "Введіть API Hash з my.telegram.org."))
            return None
        return int(id_s), self.api_hash_edit.text().strip()

    def _cancel_qr_wait(self) -> None:
        self._qr_gen_id += 1
        self._qr_timer.stop()
        self.qr_countdown_lbl.setText("")
        from core.telegram_silero import telegram_silero

        telegram_silero.cancel_qr_login()

    def _switch_auth_mode(self, mode: str) -> None:
        self._set_error("")
        is_qr = mode == "qr"
        self.btn_mode_qr.setChecked(is_qr)
        self.btn_mode_phone.setChecked(not is_qr)
        self.mode_stack.setCurrentIndex(0 if is_qr else 1)
        if not is_qr:
            self._cancel_qr_wait()
            self._set_busy(False, "")
        else:
            if self._resolve_dialog_api_creds() is not None:
                self._start_qr_flow()
        self._schedule_sync_dialog_size()

    def _on_qr_tick(self) -> None:
        if self._qr_remaining_sec > 0:
            self._qr_remaining_sec -= 1
            self.qr_countdown_lbl.setText(
                tr("tg.auth.qr_waiting_sec", "⏳ Очікування сканування ({sec} с)...", sec=self._qr_remaining_sec)
            )

    def _on_refresh_qr_clicked(self) -> None:
        self._qr_auto_retries = 0
        self._start_qr_flow()

    def _start_qr_flow(self) -> None:
        if self._closed:
            return
        creds = self._validate_and_get_api_creds()
        if creds is None:
            return
        api_id, api_hash = creds
        self._cancel_qr_wait()
        gen_id = self._qr_gen_id
        self._qr_auto_retries = 0
        self._set_error("")
        self.btn_refresh_qr.setEnabled(False)
        self.qr_countdown_lbl.setText(tr("tg.auth.qr_generating", "⏳ Генерація QR-коду..."))
        from core.telegram_silero import telegram_silero

        def _worker() -> Dict[str, Any]:
            res = telegram_silero.start_qr_login(api_id, api_hash, self._proxy_mode, self._proxy_url)
            if str(res.get("state")) == "qr_ready":
                # Підготуємо qrcode завчасно у фоновому потоці
                from core.telegram_silero import ensure_qrcode_installed

                ensure_qrcode_installed()
            return res

        run_in_background(self, _worker, lambda res, err: self._on_qr_started(gen_id, res, err))

    def _on_qr_started(self, gen_id: int, res: Any, err: str) -> None:
        if self._closed or gen_id != self._qr_gen_id:
            return
        self.btn_refresh_qr.setEnabled(True)
        if err:
            self._qr_timer.stop()
            self.qr_countdown_lbl.setText("")
            self._set_error(err)
            return

        st = str((res or {}).get("state", ""))
        if st == "connected":
            self._finish_connected()
            return

        if st == "qr_ready":
            url = str((res or {}).get("url", "") or "")
            exp_sec = int((res or {}).get("expires_in", 30) or 30)
            try:
                pix = _render_qr_pixmap(url, 212)
                self.qr_img_lbl.setPixmap(pix)
            except Exception as exc:
                self._set_error(str(exc))
                return

            self._qr_remaining_sec = exp_sec
            self.qr_countdown_lbl.setText(
                tr("tg.auth.qr_waiting_sec", "⏳ Очікування сканування ({sec} с)...", sec=exp_sec)
            )
            self._qr_timer.start()
            from core.telegram_silero import telegram_silero

            run_in_background(
                self,
                lambda: telegram_silero.wait_qr_login(timeout_sec=float(exp_sec + 2)),
                lambda wres, werr: self._on_qr_wait_done(gen_id, wres, werr),
            )

    def _on_qr_wait_done(self, gen_id: int, res: Any, err: str) -> None:
        if self._closed or gen_id != self._qr_gen_id:
            return
        self._qr_timer.stop()
        if err:
            self.qr_countdown_lbl.setText("")
            self._set_error(err)
            return

        st = str((res or {}).get("state", ""))
        if st == "connected":
            self._finish_connected()
            return
        if st == "password_required":
            self._save_credentials_to_config()
            self._go_step(2)
            self.password_edit.setFocus()
            return
        if st == "cancelled":
            return
        if st == "qr_expired":
            if self._qr_auto_retries < self.MAX_QR_AUTO_REFRESH:
                self._qr_auto_retries += 1
                from core.telegram_silero import telegram_silero

                self.qr_countdown_lbl.setText(tr("tg.auth.qr_refreshing", "🔄 Оновлення QR-коду..."))
                run_in_background(
                    self,
                    telegram_silero.recreate_qr_login,
                    lambda rres, rerr: self._on_qr_started(gen_id, rres, rerr),
                )
            else:
                self.qr_countdown_lbl.setText(
                    tr("tg.auth.qr_expired_hint", "⌛ QR-код застарів. Натисніть «Показати / Оновити QR».")
                )

    def _set_error(self, msg: str) -> None:
        self.err_lbl.setText(msg)
        self.err_lbl.setVisible(bool(msg))
        self._schedule_sync_dialog_size()

    def _set_busy(self, busy: bool, status_text: str = "") -> None:
        self._busy = busy
        for btn in (self.btn_req_code, self.btn_sign_in, self.btn_check_pw, self.btn_refresh_qr):
            btn.setEnabled(not busy)
        self.status_lbl.setText(status_text)
        self.status_lbl.setVisible(bool(status_text))
        self._schedule_sync_dialog_size()

    def _go_step(self, idx: int) -> None:
        self._set_error("")
        if idx != 0:
            self._qr_timer.stop()
        self.stack.setCurrentIndex(idx)
        self._schedule_sync_dialog_size()

    def _save_credentials_to_config(self) -> None:
        from core.telegram_silero import save_default_api_credentials

        id_txt = self.api_id_edit.text().strip()
        hash_txt = self.api_hash_edit.text().strip()
        phone_txt = self.phone_edit.text().strip()

        items = list(config.get("online_items", []))
        for x in items:
            if str(x.get("id")) == self._item_id:
                if phone_txt:
                    x["tg_phone"] = phone_txt
                if id_txt and hash_txt:
                    x["tg_api_id"] = id_txt
                    x["tg_api_hash"] = hash_txt
                break
        config.set("online_items", items)

        if id_txt.isdigit() and int(id_txt) > 0 and hash_txt and self.chk_remember_app_keys.isChecked():
            save_default_api_credentials(int(id_txt), hash_txt)

    def _finish_connected(self) -> None:
        self._save_credentials_to_config()
        self._closed = True
        self._qr_timer.stop()
        self.accept()

    def _on_request_code(self) -> None:
        if self._busy:
            return
        self._cancel_qr_wait()
        creds = self._validate_and_get_api_creds()
        if creds is None:
            return
        api_id, api_hash = creds
        phone = self.phone_edit.text().strip()
        if not phone:
            self._set_error(tr("tg.auth.err_phone", "Введіть номер телефону Telegram."))
            return

        self._set_error("")
        self._set_busy(True, tr("tg.auth.connecting", "Підключення до Telegram та запит коду..."))
        from core.telegram_silero import telegram_silero

        run_in_background(
            self,
            lambda: telegram_silero.start_auth(phone, api_id, api_hash, self._proxy_mode, self._proxy_url),
            self._on_auth_step_done,
        )

    def _on_submit_code(self) -> None:
        if self._busy:
            return
        code = self.code_edit.text().strip()
        if not code:
            self._set_error(tr("tg.auth.err_code", "Введіть код підтвердження."))
            return
        self._set_error("")
        self._set_busy(True, tr("tg.auth.signing_in", "Перевірка коду..."))
        from core.telegram_silero import telegram_silero

        run_in_background(self, lambda: telegram_silero.submit_code(code), self._on_auth_step_done)

    def _on_submit_password(self) -> None:
        if self._busy:
            return
        pw = self.password_edit.text()
        if not pw.strip():
            self._set_error(tr("tg.auth.err_password", "Введіть пароль 2FA."))
            return
        self._set_error("")
        self._set_busy(True, tr("tg.auth.checking_pw", "Перевірка пароля 2FA..."))
        from core.telegram_silero import telegram_silero

        run_in_background(self, lambda: telegram_silero.submit_password(pw), self._on_auth_step_done)

    def _on_auth_step_done(self, res: Any, err: str) -> None:
        if self._closed:
            return
        self._set_busy(False, "")
        if err:
            self._set_error(err)
            return
        st = str((res or {}).get("state", ""))
        if st == "connected":
            self._finish_connected()
        elif st == "code_required":
            self._save_credentials_to_config()
            self._go_step(1)
            self.code_edit.setFocus()
        elif st == "password_required":
            self._go_step(2)
            self.password_edit.setFocus()

    def reject(self) -> None:
        self._closed = True
        self._cancel_qr_wait()
        super().reject()

    def closeEvent(self, event) -> None:
        self._closed = True
        self._cancel_qr_wait()
        super().closeEvent(event)


class AddSileroVoiceDialog(QDialog):
    """Діалог додавання коду голосу @silero_voice_bot з перевіркою через /speaker <code>."""

    def __init__(self, existing_ids: list[str], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._existing = {x.strip().lower() for x in existing_ids if x}
        self._busy = False
        self.added_voice: Optional[Dict[str, str]] = None

        self.setWindowTitle(tr("tg.voice.add_title", "Додати голос @silero_voice_bot"))
        self.setMinimumWidth(410)
        self.setStyleSheet(DIALOG_STYLESHEET)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(10)

        hint = QLabel(
            tr(
                "tg.voice.add_hint",
                "Введіть код голосу з бота @silero_voice_bot (наприклад <code>hamster_clerk</code>, <code>rene</code>, <code>baya</code>). "
                "Бот перевірить код через <code>/speaker</code>.",
            )
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #CBD5E1; font-size: 12.5px;")
        lay.addWidget(hint)

        self.code_edit = QLineEdit()
        self.code_edit.setPlaceholderText(tr("tg.voice.code_placeholder", "Код голосу (наприклад hamster_clerk)"))
        self.code_edit.returnPressed.connect(self._on_add)
        lay.addWidget(self.code_edit)

        self.desc_edit = QLineEdit()
        self.desc_edit.setPlaceholderText(tr("tg.voice.desc_placeholder", "Назва / опис (необов'язково, напр. Хомяки)"))
        self.desc_edit.returnPressed.connect(self._on_add)
        lay.addWidget(self.desc_edit)

        self.err_lbl = QLabel("")
        self.err_lbl.setWordWrap(True)
        self.err_lbl.setVisible(False)
        self.err_lbl.setStyleSheet(
            "background-color: #450A0A; color: #FCA5A5; border: 1px solid #991B1B; "
            "border-radius: 6px; padding: 8px 10px; font-size: 12px;"
        )
        lay.addWidget(self.err_lbl)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        cancel_btn = QPushButton(tr("dlg_026", "Скасувати"))
        cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        self.add_btn = QPushButton(tr("tg.voice.add_btn", "Додати"))
        self.add_btn.setObjectName("primaryBtn")
        self.add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_btn.clicked.connect(self._on_add)
        btn_row.addWidget(self.add_btn)
        lay.addLayout(btn_row)

    def _on_add(self) -> None:
        if self._busy:
            return
        raw_code = self.code_edit.text().strip()
        if raw_code.lower().startswith("/speaker"):
            raw_code = raw_code[len("/speaker"):].strip()
        code = raw_code.lstrip("/").strip().lower()
        if not code:
            self.err_lbl.setText(tr("tg.voice.err_empty", "Введіть код голосу."))
            self.err_lbl.setVisible(True)
            return
        if code in self._existing:
            self.err_lbl.setText(tr("tg.voice.err_dup", "Голос «{code}» уже додано до списку.", code=code))
            self.err_lbl.setVisible(True)
            return

        self._busy = True
        self.add_btn.setEnabled(False)
        self.add_btn.setText(tr("tg.voice.checking", "Перевірка..."))
        self.err_lbl.setVisible(False)
        desc = self.desc_edit.text().strip()
        from core.telegram_silero import telegram_silero

        def _done(res: Any, err: str) -> None:
            self._busy = False
            self.add_btn.setEnabled(True)
            self.add_btn.setText(tr("tg.voice.add_btn", "Додати"))
            if err:
                self.err_lbl.setText(err)
                self.err_lbl.setVisible(True)
                return
            ok, msg = res
            if not ok:
                self.err_lbl.setText(msg)
                self.err_lbl.setVisible(True)
                return
            self.added_voice = {"id": msg, "description": desc}
            self.accept()

        run_in_background(self, lambda: telegram_silero.set_speaker(code, force=True), _done)


class ObsAnimationSettingsDialog(QDialog):
    """Вікно індивідуальних налаштувань обраної OBS-анімації (появи або руху)."""

    def __init__(
        self,
        kind: str,
        anim_id: str,
        on_saved: Optional[Callable[[], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        from core.obs_animations import (
            get_entry_spec,
            get_motion_spec,
            get_variant_defaults,
            get_variant_key,
            resolve_anim_params,
        )

        self._kind = kind
        self._spec = get_entry_spec(anim_id) if kind == "entry" else get_motion_spec(anim_id)
        self._anim_id = str(self._spec["id"])
        self._cfg_key = "obs_anim_params" if kind == "entry" else "obs_motion_params"
        self._on_saved = on_saved
        self._get_variant_key = get_variant_key
        self._get_variant_defaults = get_variant_defaults

        stored = dict(config.get(self._cfg_key, {}) or {})
        self._orig_stored = copy.deepcopy(stored)
        self._tested = False
        raw_custom = stored.get(self._anim_id)
        self._by_variant: Dict[str, Dict[str, Any]] = (
            copy.deepcopy(raw_custom["by_variant"])
            if isinstance(raw_custom, dict) and isinstance(raw_custom.get("by_variant"), dict)
            else {}
        )
        self._params: Dict[str, Any] = resolve_anim_params(kind, self._anim_id, stored)
        self._cur_variant_key: str = get_variant_key(kind, self._spec, self._params)
        self._by_variant.setdefault(self._cur_variant_key, dict(self._params))
        self._sync_callbacks: List[Callable[[], None]] = []

        title_txt = tr(self._spec["i18n_key"], self._spec["name"])
        self.setWindowTitle(tr("obs.anim_dlg.title", "Налаштування анімації — {name}", name=title_txt))
        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setMinimumWidth(460)
        self.setMaximumWidth(540)
        self.setStyleSheet(DIALOG_STYLESHEET)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 18)
        lay.setSpacing(12)

        hdr = QLabel(f"✨ {title_txt}")
        hdr.setObjectName("dialogTitle")
        hdr.setWordWrap(True)
        lay.addWidget(hdr)

        desc_lbl = QLabel(tr(f"{self._spec['i18n_key']}.desc", self._spec.get("description", "")))
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet("color: #CBD5E1; font-size: 13px;")
        lay.addWidget(desc_lbl)

        if self._anim_id in ("typewriter", "none"):
            info_text = (
                tr("obs.anim_dlg.typewriter_note", "Швидкість друкарської машинки автоматично синхронізується з аудіо озвучки.")
                if self._anim_id == "typewriter"
                else tr("obs.anim_dlg.none_note", "Оберіть анімацію руху у списку, щоб налаштувати її параметри.")
            )
            note = QLabel(info_text)
            note.setWordWrap(True)
            note.setStyleSheet(
                "background-color: #262C3D; border: 1px solid #38415C; border-radius: 10px; "
                "padding: 12px 14px; color: #E2E8F0; font-size: 13px;"
            )
            lay.addWidget(note)
            btn_row = QHBoxLayout()
            btn_row.addStretch(1)
            ok_btn = QPushButton(tr("dlg_001", "Зрозуміло"))
            ok_btn.setObjectName("primaryBtn")
            ok_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            ok_btn.clicked.connect(self.accept)
            btn_row.addWidget(ok_btn)
            lay.addLayout(btn_row)
            return

        card = QFrame()
        card.setStyleSheet("QFrame { background-color: #22283A; border: 1px solid #38415C; border-radius: 10px; }")
        c_lay = QVBoxLayout(card)
        c_lay.setContentsMargins(16, 14, 16, 14)
        c_lay.setSpacing(10)

        def _add_range(key: str, label: str, mn: float, mx: float, step: float, unit: str) -> QWidget:
            wrap = QWidget()
            w_lay = QVBoxLayout(wrap)
            w_lay.setContentsMargins(0, 0, 0, 0)
            w_lay.setSpacing(4)
            top = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet("border: none; background: transparent; color: #F1F5F9; font-size: 13px; font-weight: 600;")
            val_lbl = QLabel()
            val_lbl.setStyleSheet("border: none; background: transparent; color: #A5B4FC; font-size: 12px; font-weight: 700;")
            top.addWidget(lbl)
            top.addStretch(1)
            top.addWidget(val_lbl)
            w_lay.addLayout(top)

            scale = 1 if step >= 1 and float(step).is_integer() else max(1, round(1.0 / step))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(round(mn * scale), round(mx * scale))
            slider.setSingleStep(1)
            slider.setPageStep(1)

            def _fmt(v: float) -> str:
                if scale == 1:
                    return f"{int(round(v))}{' ' + unit if unit and unit not in ('%', 'px', 'x', '°') else unit}"
                return f"{v:.2f}".rstrip("0").rstrip(".") + (f" {unit}" if unit and unit not in ("%", "px", "x", "°") else unit)

            def _sync() -> None:
                cur = float(self._params.get(key, mn))
                slider.blockSignals(True)
                slider.setValue(round(cur * scale))
                slider.blockSignals(False)
                val_lbl.setText(_fmt(cur))

            def _on_val(iv: int) -> None:
                v = round(iv / scale, 4) if scale > 1 else int(iv)
                self._params[key] = v
                val_lbl.setText(_fmt(float(v)))

            slider.valueChanged.connect(_on_val)
            _sync()
            self._sync_callbacks.append(_sync)
            w_lay.addWidget(slider)
            c_lay.addWidget(wrap)
            return wrap

        def _add_select(key: str, label: str, options: List[Dict[str, str]]) -> QWidget:
            wrap = QWidget()
            w_lay = QHBoxLayout(wrap)
            w_lay.setContentsMargins(0, 0, 0, 0)
            w_lay.setSpacing(10)
            lbl = QLabel(label)
            lbl.setStyleSheet("border: none; background: transparent; color: #F1F5F9; font-size: 13px; font-weight: 600;")
            combo = DialogComboBox()
            for opt in options:
                combo.addItem(tr(opt.get("i18n", ""), opt["label"]), opt["value"])

            def _sync() -> None:
                idx = combo.findData(self._params.get(key, options[0]["value"]))
                combo.blockSignals(True)
                combo.setCurrentIndex(idx if idx >= 0 else 0)
                combo.blockSignals(False)

            def _on_select_changed() -> None:
                # Зберігаємо значення попереднього режиму та перемикаємося на профіль нового
                self._by_variant[self._cur_variant_key] = dict(self._params)
                self._params[key] = combo.currentData()
                new_vkey = self._get_variant_key(self._kind, self._spec, self._params)
                self._cur_variant_key = new_vkey
                if new_vkey in self._by_variant:
                    loaded = dict(self._by_variant[new_vkey])
                    loaded[key] = combo.currentData()
                    self._params = loaded
                else:
                    self._params = self._get_variant_defaults(self._kind, self._spec, self._params)
                    self._by_variant[new_vkey] = dict(self._params)
                for fn in self._sync_callbacks:
                    fn()

            combo.currentIndexChanged.connect(_on_select_changed)
            _sync()
            self._sync_callbacks.append(_sync)
            w_lay.addWidget(lbl)
            w_lay.addWidget(combo, 1)
            c_lay.addWidget(wrap)
            return wrap

        if kind == "motion":
            _add_select(
                "trigger",
                tr("obs.ctrl.motion_trigger", "Запуск відносно появи:"),
                [
                    {"value": "after", "i18n": "obs.opt.trigger_after", "label": "Після появи тексту"},
                    {"value": "simultaneous", "i18n": "obs.opt.trigger_sim", "label": "Одночасно з появою"},
                    {"value": "only", "i18n": "obs.opt.trigger_only", "label": "Лише рух (без появи)"},
                ],
            )

        if kind == "entry":
            _add_range("duration", tr("obs.ctrl.duration", "Тривалість появи"), 150, 2500, 50, "ms")
        else:
            _add_range("duration", tr("obs.ctrl.motion_duration", "Тривалість циклу"), 600, 6000, 100, "ms")

        if self._spec.get("supports_split"):
            split_opts = [
                {"value": "whole", "i18n": "obs.opt.split_whole", "label": "Весь текст разом"},
                {"value": "words", "i18n": "obs.opt.split_words", "label": "По словах"},
                {"value": "chars", "i18n": "obs.opt.split_chars", "label": "По літерах"},
            ]
            split_row = _add_select("split", tr("obs.ctrl.split", "Поява елементів:"), split_opts)
            stagger_row = None
            if kind == "entry":
                stagger_row = _add_range("stagger", tr("obs.ctrl.stagger", "Інтервал між елементами"), 10, 200, 5, "ms")
            split_combo = split_row.findChild(QComboBox)

            def _update_stagger_vis() -> None:
                if stagger_row is None:
                    return
                is_whole = str(self._params.get("split", "whole")) == "whole"
                stagger_row.setVisible(not is_whole)

            if split_combo is not None:
                split_combo.currentIndexChanged.connect(_update_stagger_vis)
            self._sync_callbacks.append(_update_stagger_vis)
            _update_stagger_vis()

        for ctrl in self._spec.get("controls", []):
            if ctrl.get("type") == "select":
                _add_select(ctrl["key"], tr(ctrl.get("i18n", ""), ctrl["label"]) + ":", ctrl["options"])
            else:
                _add_range(
                    ctrl["key"],
                    tr(ctrl.get("i18n", ""), ctrl["label"]),
                    float(ctrl["min"]),
                    float(ctrl["max"]),
                    float(ctrl["step"]),
                    str(ctrl.get("unit", "")),
                )

        lay.addWidget(card)

        btn_row = QHBoxLayout()
        reset_btn = QPushButton(tr("obs.anim_dlg.reset", "↺ Скинути"))
        reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(reset_btn)

        test_btn = QPushButton(tr("obs.anim_dlg.test", "✨ Тест"))
        test_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        test_btn.clicked.connect(self._on_test)
        btn_row.addWidget(test_btn)

        btn_row.addStretch(1)

        cancel_btn = QPushButton(tr("dlg_026", "Скасувати"))
        cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        save_btn = QPushButton(tr("dlg_041", "Зберегти"))
        save_btn.setObjectName("primaryBtn")
        save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(save_btn)
        lay.addLayout(btn_row)

    def _on_reset(self) -> None:
        self._params = self._get_variant_defaults(self._kind, self._spec, self._params)
        self._by_variant[self._cur_variant_key] = dict(self._params)
        for fn in self._sync_callbacks:
            fn()

    def _apply_to_config(self, auto_save: bool = True) -> None:
        self._by_variant[self._cur_variant_key] = dict(self._params)
        stored = dict(config.get(self._cfg_key, {}) or {})
        entry: Dict[str, Any] = {"by_variant": copy.deepcopy(self._by_variant)}
        for k in ("split", "trigger"):
            if k in self._params:
                entry[k] = self._params[k]
        for ctrl in self._spec.get("controls", []):
            if ctrl.get("type") == "select" and ctrl["key"] in self._params:
                entry[ctrl["key"]] = self._params[ctrl["key"]]
        stored[self._anim_id] = entry
        config.set(self._cfg_key, stored, auto_save=auto_save)

    def _on_test(self) -> None:
        from core.obs_server import obs_server

        self._apply_to_config(auto_save=False)
        self._tested = True
        obs_server.broadcast_style()
        obs_server.broadcast_speech(
            tr(
                "obs.sample_text",
                "Привіт глядачам! Це перевірка субтитрів в OBS ✨ Так виглядатиме ваш текст на трансляції: шрифт, розмір, кольори, обводка та анімація появи. Налаштуйте все під свій стиль!",
            ),
            8.0,
            force=True,
        )

    def reject(self) -> None:
        if self._tested:
            from core.obs_server import obs_server

            config.set(self._cfg_key, self._orig_stored, auto_save=False)
            obs_server.broadcast_style()
        super().reject()

    def _on_save(self) -> None:
        self._apply_to_config(auto_save=True)
        if self._on_saved:
            self._on_saved()
        self.accept()


class SettingsTransferDialog(QDialog):
    """Choose which configuration categories to transfer."""

    CATEGORY_LABELS = {
        "general": "Мова та запуск програми",
        "appearance": "Вигляд екранного тексту",
        "voices": "Голос і вимова",
        "input": "Набір тексту та сигнали",
        "hotkeys": "Гарячі клавіші",
        "obs": "Субтитри OBS",
        "quick_phrases": "Швидкі фрази та їхні аудіофайли",
        "online_services": "Онлайн-голоси",
        "ai_services": "AI-коректор",
        "audio": "Гучність і мікшування звуку",
    }

    CATEGORY_DESCRIPTIONS = {
        "general": "Мова інтерфейсу, фільтр голосів і спосіб запуску програми.",
        "appearance": "Оформлення, розмір і розташування тексту на екрані; кольори та розмір головного вікна.",
        "voices": "Основний і запасний TTS, голоси, швидкість, вимова та обробка звуку.",
        "input": "Режим набору, безперервний діалог і сигнали початку та скасування.",
        "hotkeys": "Призначені клавіші для набору, озвучення, зупинки та інших дій.",
        "obs": "Підключення OBS, вигляд і анімація субтитрів, показ набору наживо.",
        "quick_phrases": "Збережені фрази, їхні комбінації клавіш і прикріплені аудіофайли.",
        "online_services": "Додані сервіси онлайн-озвучення та їхні параметри. API-ключі додаються окремою галочкою.",
        "ai_services": "Сервіси AI-коректора, основний і резервний сервіси та інструкції для виправлення тексту.",
        "audio": "Рівні гучності та мікшування мікрофона. Назви пристроїв прив'язані до цього ПК і не переносяться.",
    }

    def __init__(
        self,
        categories: list[str],
        *,
        include_api_keys_available: bool = True,
        mode: str = "export",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        is_import = mode == "import"
        self.setWindowTitle(tr("transfer.import.title", "Імпорт налаштувань") if is_import else tr("transfer.export.title", "Експорт налаштувань"))
        self.setMinimumWidth(530)
        self.setStyleSheet(DIALOG_STYLESHEET)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(8)

        title = QLabel(tr("transfer.import.heading", "Вибери, які розділи замінити з файлу") if is_import else tr("transfer.export.heading", "Вибери, що зберегти у файл"))
        title.setObjectName("dialogTitle")
        layout.addWidget(title)
        note = QLabel(
            tr(
                "transfer.import.note",
                "Постав галочки біля розділів, які потрібно замінити даними з файлу. Інші налаштування залишаться без змін. API-ключі вимкнені за замовчуванням.",
            )
            if is_import
            else tr(
                "transfer.export.note",
                "Постав галочки біля розділів, які потрібно включити у файл профілю. API-ключі вимкнені за замовчуванням.",
            )
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.category_checks: Dict[str, QCheckBox] = {}
        content = QWidget()
        checks_layout = QVBoxLayout(content)
        checks_layout.setContentsMargins(0, 0, 0, 0)
        checks_layout.setSpacing(5)
        for key in self.CATEGORY_LABELS:
            if key not in categories:
                continue
            row = QWidget()
            row_layout = QVBoxLayout(row)
            row_layout.setContentsMargins(0, 2, 0, 3)
            row_layout.setSpacing(1)
            checkbox = QCheckBox(self.category_label(key))
            checkbox.setChecked(True)
            self.category_checks[key] = checkbox
            row_layout.addWidget(checkbox)
            description = QLabel(
                tr(
                    f"transfer.category.{key}.description",
                    self.CATEGORY_DESCRIPTIONS[key],
                )
            )
            description.setWordWrap(True)
            description.setStyleSheet("color: #AAB4C8; font-size: 11px; margin-left: 25px;")
            row_layout.addWidget(description)
            checks_layout.addWidget(row)

        if self.category_checks:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setWidget(content)
            layout.addWidget(scroll, 1)

        api_label_key = "transfer.api_keys.import_label" if is_import else "transfer.api_keys.export_label"
        api_label_default = "Імпортувати API-ключі (конфіденційні дані)" if is_import else "Додати API-ключі до файлу (конфіденційні дані)"
        self.api_keys_checkbox = QCheckBox(tr(api_label_key, api_label_default))
        self.api_keys_checkbox.setChecked(False)
        self.api_keys_checkbox.setEnabled(include_api_keys_available)
        if not include_api_keys_available:
            self.api_keys_checkbox.setToolTip(tr("transfer.api_keys.none", "У цьому файлі немає API-ключів."))
        self.api_keys_checkbox.toggled.connect(self._on_api_keys_toggled)
        layout.addWidget(self.api_keys_checkbox)

        warning = QLabel(
            tr(
                "transfer.api_keys.import_note" if is_import else "transfer.api_keys.export_note",
                "Файл містить конфіденційні API-ключі. Після імпорту програма зможе звертатися до відповідних сервісів. Продовжуй лише якщо довіряєш файлу. Для перенесення ключів також потрібні налаштування цих сервісів."
                if is_import
                else "API-ключі дають доступ до ваших сервісів. Додавай їх у файл лише якщо одержувач має їм довіряти. Для перенесення ключів також потрібні налаштування відповідних онлайн- та AI-сервісів.",
            )
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #FBBF24;")
        layout.addWidget(warning)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(
            tr("transfer.export.button", "Зберегти профіль") if not is_import else tr("transfer.import.button", "Продовжити")
        )
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr("transfer.cancel", "Скасувати"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_api_keys_toggled(self, enabled: bool) -> None:
        # Service definitions contain the credentials, so always include them
        # together when the user opts into transferring API keys.
        for key in ("online_services", "ai_services"):
            checkbox = self.category_checks.get(key)
            if checkbox is not None:
                if enabled:
                    checkbox.setChecked(True)
                checkbox.setEnabled(not enabled)

    def selected_categories(self) -> list[str]:
        return [key for key, checkbox in self.category_checks.items() if checkbox.isChecked()]

    @classmethod
    def category_label(cls, key: str) -> str:
        return tr(f"transfer.category.{key}.label", cls.CATEGORY_LABELS.get(key, key))

    def includes_api_keys(self) -> bool:
        return self.api_keys_checkbox.isChecked()


class AppUpdateDialog(QDialog):
    """Вікно перевірки та встановлення оновлень SilentVoice TTS (GitHub / GitLab Releases)."""

    check_finished = Signal(object, bool, str)
    download_progress = Signal(int, float, float)
    download_finished = Signal(str, str)
    notes_localized = Signal(int, str)

    def __init__(
        self,
        initial_release: Optional[Any] = None,
        on_quit_for_install: Optional[Callable[[], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        from core.updater import APP_VERSION, DEFAULT_GITHUB_REPO, DEFAULT_GITLAB_REPO

        self._release = initial_release
        self._on_quit_for_install = on_quit_for_install
        self._cancel_event = threading.Event()
        self._downloading = False
        self._checking = False
        self._downloaded_path: Optional[Path] = None
        self._notes_req_id = 0

        self.setWindowTitle(tr("update.dlg.title", "Оновлення SilentVoice TTS"))
        self.setMinimumWidth(720)
        self.setMinimumHeight(480)
        self.resize(780, 560)
        self.setStyleSheet(DIALOG_STYLESHEET)

        self.check_finished.connect(self._on_check_finished)
        self.download_progress.connect(self._on_download_progress)
        self.download_finished.connect(self._on_download_finished)
        self.notes_localized.connect(self._on_notes_localized)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 16)
        lay.setSpacing(10)

        hdr = QLabel(f"🔄 {tr('update.dlg.title', 'Оновлення SilentVoice TTS')}")
        hdr.setObjectName("dialogTitle")
        lay.addWidget(hdr)

        ver_lbl = QLabel(tr("update.current_version", "Поточна версія: <b>v{version}</b>", version=APP_VERSION))
        ver_lbl.setStyleSheet("color: #CBD5E1; font-size: 13px;")
        lay.addWidget(ver_lbl)

        # Поле адреси репозиторію (зручно вставити свій GitHub або GitLab)
        repo_row = QHBoxLayout()
        repo_row.setSpacing(8)
        repo_lbl = QLabel(tr("update.dlg.repo_label", "Репозиторій (GitHub / GitLab):"))
        repo_lbl.setStyleSheet("color: #CBD5E1; font-size: 12.5px;")
        repo_row.addWidget(repo_lbl)

        saved_repo = str(config.get("update_repo_url", "") or DEFAULT_GITHUB_REPO or DEFAULT_GITLAB_REPO).strip()
        self.repo_edit = QLineEdit(saved_repo)
        self.repo_edit.setPlaceholderText(
            tr(
                "update.dlg.repo_placeholder",
                "https://github.com/owner/SilentVoice or https://gitlab.com/owner/SilentVoice",
            )
        )
        self.repo_edit.editingFinished.connect(self._save_repo_url)
        repo_row.addWidget(self.repo_edit, 1)

        self.check_btn = QPushButton(tr("update.check_btn", "🔄 Перевірити оновлення"))
        self.check_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.check_btn.clicked.connect(self.start_check)
        repo_row.addWidget(self.check_btn)
        lay.addLayout(repo_row)

        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setTextFormat(Qt.TextFormat.RichText)
        self.status_lbl.setStyleSheet("font-size: 13.5px; color: #E2E8F0; padding: 4px 0;")
        lay.addWidget(self.status_lbl)

        self.notes_edit = QTextEdit()
        self.notes_edit.setReadOnly(True)
        self.notes_edit.setMinimumHeight(240)
        self.notes_edit.setPlaceholderText(
            tr(
                "update.dlg.notes_placeholder",
                "Тут з'явиться список змін (Release Notes) після перевірки релізу.",
            )
        )
        lay.addWidget(self.notes_edit, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background: #141826; border: 1px solid #353E5C; border-radius: 8px; "
            "color: #F8FAFC; font-size: 11px; text-align: center; }"
            "QProgressBar::chunk { background: #635BFF; border-radius: 7px; }"
        )
        lay.addWidget(self.progress_bar)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.open_page_btn = QPushButton(tr("update.dlg.open_page_btn", "🌐 Відкрити сторінку релізу"))
        self.open_page_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_page_btn.setVisible(False)
        self.open_page_btn.clicked.connect(self._open_release_page)
        btn_row.addWidget(self.open_page_btn)

        btn_row.addStretch(1)

        self.install_btn = QPushButton(tr("update.dlg.install_btn", "⚡ Оновити зараз (1 клік)"))
        self.install_btn.setObjectName("successBtn")
        self.install_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.install_btn.setVisible(False)
        self.install_btn.clicked.connect(self._start_download_and_install)
        btn_row.addWidget(self.install_btn)

        self.close_btn = QPushButton(tr("dlg_005", "Закрити"))
        self.close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.close_btn)
        lay.addLayout(btn_row)

        if self._release is not None:
            self._apply_release_ui(self._release, True)
        else:
            from core.updater import has_configured_update_sources

            if has_configured_update_sources() or self.repo_edit.text().strip():
                QTimer.singleShot(80, self.start_check)
            else:
                self.status_lbl.setText(
                    tr(
                        "update.err.not_configured",
                        "Адресу репозиторію GitHub або GitLab ще не вказано.",
                    )
                )

    def _save_repo_url(self) -> None:
        config.set("update_repo_url", self.repo_edit.text().strip())

    def start_check(self) -> None:
        if self._checking or self._downloading:
            return
        self._save_repo_url()
        custom_repo = self.repo_edit.text().strip()
        self._checking = True
        self._downloaded_path = None
        self.check_btn.setEnabled(False)
        self.install_btn.setVisible(False)
        self.open_page_btn.setVisible(False)
        self.status_lbl.setStyleSheet("font-size: 13.5px; color: #A5B4FC; padding: 4px 0;")
        self.status_lbl.setText(tr("update.dlg.checking", "⏳ Перевірка наявності оновлень..."))

        def _worker() -> None:
            from core.updater import check_for_updates

            try:
                rel, is_newer = check_for_updates(custom_source=custom_repo)
                self.check_finished.emit(rel, is_newer, "")
            except Exception as exc:
                self.check_finished.emit(None, False, str(exc))

        threading.Thread(target=_worker, daemon=True, name="UpdateChecker").start()

    def _on_check_finished(self, release: object, is_newer: bool, err: str) -> None:
        self._checking = False
        self.check_btn.setEnabled(True)
        if err:
            self.status_lbl.setStyleSheet("font-size: 13.5px; color: #FCA5A5; padding: 4px 0;")
            self.status_lbl.setText(f"⚠️ {err}")
            return
        self._release = release
        self._apply_release_ui(release, is_newer)

    def _set_notes_markdown(self, text: str) -> None:
        cleaned = str(text or "").strip()
        if not cleaned:
            self.notes_edit.setPlainText(
                tr("update.dlg.no_notes", "Опис змін для цього релізу не вказано.")
            )
            return
        try:
            self.notes_edit.setMarkdown(cleaned)
        except Exception:
            self.notes_edit.setPlainText(cleaned)

    def _on_notes_localized(self, req_id: int, localized_text: str) -> None:
        if req_id != self._notes_req_id:
            return
        if localized_text.strip():
            self._set_notes_markdown(localized_text)

    def _apply_release_ui(self, release: Any, is_newer: bool) -> None:
        from core.updater import APP_VERSION, extract_release_notes_section, localize_release_notes

        if release is None:
            return
        notes = str(getattr(release, "notes", "") or "").strip()
        self._notes_req_id += 1
        req_id = self._notes_req_id
        ui_lang = get_ui_language()

        if not notes:
            self._set_notes_markdown("")
        else:
            preview_notes, already_localized = extract_release_notes_section(notes, ui_lang)
            self._set_notes_markdown(preview_notes)
            if not already_localized:
                def _translate_worker() -> None:
                    try:
                        translated = localize_release_notes(notes, target_lang=ui_lang)
                        if translated and translated.strip() != preview_notes.strip():
                            self.notes_localized.emit(req_id, translated)
                    except Exception:
                        pass

                threading.Thread(
                    target=_translate_worker,
                    daemon=True,
                    name="UpdateNotesTranslator",
                ).start()

        html_url = str(getattr(release, "html_url", "") or "").strip()
        self.open_page_btn.setVisible(bool(html_url))

        if is_newer:
            self.status_lbl.setStyleSheet("font-size: 13.5px; color: #4ADE80; padding: 4px 0;")
            self.status_lbl.setText(
                tr(
                    "update.dlg.available",
                    "🎉 Доступна нова версія: <b>v{new_ver}</b> (зараз: v{cur_ver}, джерело: {source})",
                    new_ver=release.version,
                    cur_ver=APP_VERSION,
                    source=release.source,
                )
            )
            self.install_btn.setVisible(bool(getattr(release, "installer_url", "")))
            self.install_btn.setEnabled(True)
        else:
            self.status_lbl.setStyleSheet("font-size: 13.5px; color: #38BDF8; padding: 4px 0;")
            self.status_lbl.setText(
                tr(
                    "update.dlg.up_to_date",
                    "✅ У вас встановлена найновіша версія програми (v{version}).",
                    version=APP_VERSION,
                )
            )
            self.install_btn.setVisible(False)

    def _open_release_page(self) -> None:
        url = str(getattr(self._release, "html_url", "") or "").strip()
        if url:
            QDesktopServices.openUrl(QUrl(url))

    def _start_download_and_install(self) -> None:
        if self._release is None or self._downloading:
            return
        # Якщо інсталятор уже завантажено у цій сесії (наприклад, користувач скасував UAC) — запускаємо одразу
        if self._downloaded_path is not None and self._downloaded_path.is_file():
            self._launch_downloaded_installer(self._downloaded_path)
            return

        self._downloading = True
        self._cancel_event.clear()
        self.install_btn.setEnabled(False)
        self.check_btn.setEnabled(False)
        self.repo_edit.setEnabled(False)
        self.close_btn.setText(tr("ui.cancel_btn", "Скасувати"))
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_lbl.setStyleSheet("font-size: 13.5px; color: #A5B4FC; padding: 4px 0;")
        self.status_lbl.setText(
            tr(
                "update.dlg.connecting",
                "⏳ Підключення до сервера та початок завантаження оновлення...",
            )
        )

        rel = self._release

        def _worker() -> None:
            from core.updater import download_update_installer

            try:
                path = download_update_installer(
                    rel,
                    progress_cb=lambda pct, dl, tot: self.download_progress.emit(pct, dl, tot),
                    cancel_event=self._cancel_event,
                )
                self.download_finished.emit(str(path), "")
            except Exception as exc:
                self.download_finished.emit("", str(exc))

        threading.Thread(target=_worker, daemon=True, name="UpdateDownloader").start()

    def _on_download_progress(self, pct: int, dl_mb: float, tot_mb: float) -> None:
        self.progress_bar.setValue(pct)
        self.status_lbl.setStyleSheet("font-size: 13.5px; color: #E2E8F0; padding: 4px 0;")
        if tot_mb > 0:
            self.status_lbl.setText(
                tr(
                    "update.dlg.downloading",
                    "⬇️ Завантаження інсталятора: {dl:.1f} / {tot:.1f} МБ ({pct}%)",
                    dl=dl_mb,
                    tot=tot_mb,
                    pct=pct,
                )
            )
        else:
            self.status_lbl.setText(
                tr(
                    "update.dlg.downloading_unknown",
                    "⬇️ Завантаження інсталятора: {dl:.1f} МБ...",
                    dl=dl_mb,
                )
            )

    def _on_download_finished(self, file_path: str, err: str) -> None:
        self._downloading = False
        self.check_btn.setEnabled(True)
        self.repo_edit.setEnabled(True)
        self.close_btn.setText(tr("dlg_005", "Закрити"))
        if err:
            self.install_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.status_lbl.setStyleSheet("font-size: 13.5px; color: #FCA5A5; padding: 4px 0;")
            self.status_lbl.setText(f"⚠️ {err}")
            return

        installer_path = Path(file_path)
        self._downloaded_path = installer_path
        self._launch_downloaded_installer(installer_path)

    def _launch_downloaded_installer(self, installer_path: Path) -> None:
        from core.updater import launch_update_installer

        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(True)
        self.install_btn.setEnabled(False)
        self.status_lbl.setStyleSheet("font-size: 13.5px; color: #4ADE80; padding: 4px 0;")
        self.status_lbl.setText(
            tr(
                "update.dlg.launching_visible",
                "✅ Завантаження завершено! Запускаємо інсталятор — зараз відкриється вікно встановлення, а після завершення SilentVoice перезапуститься сам.",
            )
        )
        QApplication.processEvents()

        target_ver = str(getattr(self._release, "version", "") or "").strip()
        if target_ver:
            config.set("pending_update_target_version", target_ver)

        try:
            launched = launch_update_installer(installer_path, silent=True)
        except Exception as exc:
            config.set("pending_update_target_version", "")
            self.install_btn.setEnabled(True)
            self.status_lbl.setStyleSheet("font-size: 13.5px; color: #FCA5A5; padding: 4px 0;")
            self.status_lbl.setText(f"⚠️ {exc}")
            return

        if launched:
            self.accept()
            if self._on_quit_for_install is not None:
                QTimer.singleShot(300, self._on_quit_for_install)
        else:
            config.set("pending_update_target_version", "")
            self.install_btn.setEnabled(True)
            self.status_lbl.setStyleSheet("font-size: 13.5px; color: #FBBF24; padding: 4px 0;")
            self.status_lbl.setText(
                tr(
                    "update.dlg.uac_cancelled",
                    "⚠️ Запуск інсталятора скасовано (UAC). Файл оновлення вже завантажено — натисніть «Оновити зараз», щоб продовжити встановлення.",
                )
            )

    def reject(self) -> None:
        if self._downloading:
            self._cancel_event.set()
        super().reject()


class FFmpegInstallDialog(QDialog):
    """Вікно-пояснення при увімкненні швидкості / тону (Pitch) або custom-голосу, коли FFmpeg відсутній."""

    def __init__(self, parent: Optional[QWidget] = None, *, voice_is_custom: bool = False) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("ui.ffmpeg.dlg_title", "Потрібен FFmpeg"))
        self.setModal(True)
        self.setMinimumWidth(470)
        self.setStyleSheet(DIALOG_STYLESHEET)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(12)

        title = QLabel(tr("ui.ffmpeg.dlg_header", "🎬 Потрібен FFmpeg для зміни швидкості та тону"))
        title.setObjectName("dialogTitle")
        title.setWordWrap(True)
        lay.addWidget(title)

        body_text = (
            tr(
                "ui.ffmpeg.dlg_body_custom",
                "Ви обрали голос <b>Custom</b>, у якому за замовчуванням увімкнено зміну <b>швидкості</b> та <b>тону (Pitch)</b>.<br><br>"
                "Для чистої обробки голосу без металевих спотворень і артефактів потрібен <b>FFmpeg</b> (~114 МБ).<br>"
                "Без нього використовуватиметься спрощений резервний алгоритм.",
            )
            if voice_is_custom
            else tr(
                "ui.ffmpeg.dlg_body",
                "Ви увімкнули зміну <b>швидкості мовлення</b> або <b>тону голосу (Pitch)</b>.<br><br>"
                "Для чистого звучання без металевих спотворень і артефактів потрібен <b>FFmpeg</b> (~114 МБ).<br>"
                "Без нього використовуватиметься спрощений резервний алгоритм.",
            )
        )
        body = QLabel(body_text)
        body.setWordWrap(True)
        body.setTextFormat(Qt.TextFormat.RichText)
        body.setStyleSheet("color: #E2E8F0; font-size: 13.5px; line-height: 1.45;")
        lay.addWidget(body)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFixedHeight(10)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        lay.addWidget(self.progress_bar)

        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("color: #CBD5E1; font-size: 12.5px;")
        self.status_lbl.setVisible(False)
        lay.addWidget(self.status_lbl)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch(1)

        self.cancel_btn = QPushButton(tr("dlg_026", "Скасувати"))
        self.cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.cancel_btn)

        self.install_btn = QPushButton(tr("ui.ffmpeg.dlg_install", "⬇️ Встановити"))
        self.install_btn.setProperty("class", "primaryBtn")
        self.install_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.install_btn.clicked.connect(self._on_install_clicked)
        btn_row.addWidget(self.install_btn)
        lay.addLayout(btn_row)

        self._signals_connected = False
        self._connect_signals()

        if tts_manager.is_downloading(FFMPEG_ENGINE_ID):
            pct, msg = tts_manager.get_download_progress(FFMPEG_ENGINE_ID) or (
                1,
                tr("tts.dl.ffmpeg_preparing", "Підготовка до встановлення FFmpeg..."),
            )
            self._show_downloading_state(pct, msg)

    def _connect_signals(self) -> None:
        if self._signals_connected:
            return
        tts_manager.download_progress.connect(self._on_dl_progress)
        tts_manager.download_failed.connect(self._on_dl_failed)
        tts_manager.download_cancelled.connect(self._on_dl_cancelled)
        tts_manager.model_status_changed.connect(self._on_dl_status)
        self._signals_connected = True

    def _disconnect_signals(self) -> None:
        if not self._signals_connected:
            return
        for sig, slot in (
            (tts_manager.download_progress, self._on_dl_progress),
            (tts_manager.download_failed, self._on_dl_failed),
            (tts_manager.download_cancelled, self._on_dl_cancelled),
            (tts_manager.model_status_changed, self._on_dl_status),
        ):
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._signals_connected = False

    def _show_downloading_state(self, pct: int, msg: str) -> None:
        self.install_btn.setEnabled(False)
        self.install_btn.setText(tr("ui_173", "⏳ Завантаження..."))
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(pct)
        self.status_lbl.setStyleSheet("color: #CBD5E1; font-size: 12.5px;")
        self.status_lbl.setVisible(True)
        self.status_lbl.setText(msg)

    def _on_install_clicked(self) -> None:
        if is_ffmpeg_installed():
            self.accept()
            return
        self._show_downloading_state(1, tr("tts.dl.ffmpeg_preparing", "Підготовка до встановлення FFmpeg..."))
        tts_manager.install_ffmpeg()

    def _on_dl_progress(self, engine_id: str, pct: int, msg: str) -> None:
        if engine_id != FFMPEG_ENGINE_ID:
            return
        self._show_downloading_state(pct, msg)

    def _on_dl_failed(self, engine_id: str, err: str) -> None:
        if engine_id != FFMPEG_ENGINE_ID:
            return
        self.install_btn.setEnabled(True)
        self.install_btn.setText(tr("ui.ffmpeg.dlg_install", "⬇️ Встановити"))
        self.progress_bar.setVisible(False)
        self.status_lbl.setStyleSheet("color: #FCA5A5; font-size: 12.5px;")
        self.status_lbl.setVisible(True)
        self.status_lbl.setText(err)

    def _on_dl_cancelled(self, engine_id: str) -> None:
        if engine_id != FFMPEG_ENGINE_ID:
            return
        self.install_btn.setEnabled(True)
        self.install_btn.setText(tr("ui.ffmpeg.dlg_install", "⬇️ Встановити"))
        self.progress_bar.setVisible(False)
        self.status_lbl.setStyleSheet("color: #CBD5E1; font-size: 12.5px;")
        self.status_lbl.setVisible(True)
        self.status_lbl.setText(tr("tts.dl.cancelled", "Завантаження скасовано"))

    def _on_dl_status(self, engine_id: str, installed: bool) -> None:
        if engine_id == FFMPEG_ENGINE_ID and installed:
            self.accept()

    def done(self, result: int) -> None:
        self._disconnect_signals()
        super().done(result)

    def reject(self) -> None:
        if tts_manager.is_downloading(FFMPEG_ENGINE_ID):
            tts_manager.cancel_download(FFMPEG_ENGINE_ID)
        super().reject()


DONATE_DONATELLO_URL = "https://donatello.to/B1M"
DONATE_KOFI_URL = "https://ko-fi.com/b1m164/tip"
DONATE_CRYPTO_URL = "https://b1m164.notion.site/Donate-via-cryptocurrency-3f6eff284d6780289505cfaa2b43f748"
DONATE_CRYPTO_RU_URL = "https://b1m164.notion.site/Donate-via-cryptocurrency-420eff284d6780d48d2be109f8065016?source=copy_link"
DONATE_TWITCH_URL = "https://www.twitch.tv/B1M_164"


class DonateDialog(QDialog):
    """Вікно підтримки проєкту з вибором способу донату залежно від мови інтерфейсу."""

    def __init__(self, parent: Optional[QWidget] = None, *, ui_lang: Optional[str] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("donate.dlg.title", "Підтримати SilentVoice"))
        self.setModal(True)
        self.setMinimumWidth(460)
        self.setStyleSheet(DIALOG_STYLESHEET)

        lang = str(ui_lang if ui_lang is not None else get_ui_language() or "en").strip().lower()
        self.option_buttons: List[tuple[str, QPushButton]] = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 18)
        lay.setSpacing(12)

        hdr = QLabel(tr("donate.dlg.header", "❤️ Підтримати розробку SilentVoice"))
        hdr.setObjectName("dialogTitle")
        hdr.setWordWrap(True)
        lay.addWidget(hdr)

        desc = QLabel(
            tr(
                "donate.dlg.desc",
                "SilentVoice — безкоштовний проєкт. Якщо програма допомагає вам спілкуватися, грати чи стрімити, ви можете підтримати її розвиток:",
            )
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #CBD5E1; font-size: 13.5px; line-height: 1.45;")
        lay.addWidget(desc)

        if lang == "ru":
            options = [
                (
                    tr("donate.opt.crypto_title", "🪙 Криптовалюта"),
                    tr("donate.opt.crypto_desc", "Підтримка в криптовалюті (працює з будь-якої країни)"),
                    DONATE_CRYPTO_RU_URL,
                ),
            ]
        else:
            options = [
                (
                    tr("donate.opt.donatello_title", "💳 Donatello — Банківська картка"),
                    tr("donate.opt.donatello_desc", "Оплата карткою Visa / Mastercard (Україна та міжнародні картки)"),
                    DONATE_DONATELLO_URL,
                ),
                (
                    tr("donate.opt.kofi_title", "☕ Ko-fi — PayPal"),
                    tr("donate.opt.kofi_desc", "Оплата через PayPal (зручно для підтримки з-за кордону)"),
                    DONATE_KOFI_URL,
                ),
                (
                    tr("donate.opt.crypto_title", "🪙 Криптовалюта"),
                    tr("donate.opt.crypto_desc", "Підтримка в криптовалюті (працює з будь-якої країни)"),
                    DONATE_CRYPTO_URL,
                ),
                (
                    tr("donate.opt.twitch_title", "💜 Twitch — Канал автора"),
                    tr("donate.opt.twitch_desc", "Підтримайте підпискою або заходьте на стрім (B1M_164)"),
                    DONATE_TWITCH_URL,
                ),
            ]

        for opt_title, opt_desc, opt_url in options:
            card = QFrame()
            card.setStyleSheet(
                "QFrame { background-color: #262C40; border: 1px solid #3B4463; border-radius: 10px; }"
            )
            c_lay = QHBoxLayout(card)
            c_lay.setContentsMargins(14, 12, 14, 12)
            c_lay.setSpacing(12)

            txt_col = QVBoxLayout()
            txt_col.setSpacing(3)
            lbl_t = QLabel(opt_title)
            lbl_t.setStyleSheet("color: #F8FAFC; font-size: 14px; font-weight: 700; border: none; background: transparent;")
            lbl_t.setWordWrap(True)
            txt_col.addWidget(lbl_t)

            lbl_d = QLabel(opt_desc)
            lbl_d.setStyleSheet("color: #94A3B8; font-size: 12.5px; border: none; background: transparent;")
            lbl_d.setWordWrap(True)
            txt_col.addWidget(lbl_d)
            c_lay.addLayout(txt_col, 1)

            open_btn = QPushButton(tr("donate.btn.open", "Відкрити →"))
            open_btn.setProperty("class", "primaryBtn")
            open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            open_btn.setMinimumWidth(112)
            open_btn.clicked.connect(lambda _checked=False, u=opt_url: self._open_url(u))
            c_lay.addWidget(open_btn)

            lay.addWidget(card)
            self.option_buttons.append((opt_url, open_btn))

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        close_btn = QPushButton(tr("dlg_005", "Закрити"))
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)

    def _open_url(self, url: str) -> None:
        QDesktopServices.openUrl(QUrl(url))
        self.accept()


class CrashReportDialog(QDialog):
    """Вікно після серйозного крашу або зависання з пакуванням логів у .zip та переходом на GitHub Issues."""

    def __init__(
        self,
        report_info: Dict[str, Any],
        data_dir: Path,
        crash_log_path: Path,
        logs_dir: Path,
        app_version: str,
        repo_url: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._report_info = dict(report_info or {})
        self._data_dir = Path(data_dir)
        self._crash_log_path = Path(crash_log_path)
        self._logs_dir = Path(logs_dir)
        self._app_version = str(app_version or "")
        self._repo_url = str(repo_url or "")
        self._zip_path: Optional[Path] = None

        kind = str(self._report_info.get("kind", "crash") or "crash")
        is_freeze = kind == "freeze"

        self.setWindowTitle(tr("crash_dlg.win_title", "Звіт про збій — SilentVoice"))
        self.setMinimumWidth(520)
        self.setStyleSheet(DIALOG_STYLESHEET)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 20, 22, 18)
        lay.setSpacing(12)

        title_text = (
            tr("crash_dlg.title_freeze", "⚠️ Попередній запуск програми завис")
            if is_freeze
            else tr("crash_dlg.title_crash", "⚠️ Попередній запуск завершився аварійно")
        )
        title_lbl = QLabel(title_text)
        title_lbl.setObjectName("dialogTitle")
        title_lbl.setWordWrap(True)
        lay.addWidget(title_lbl)

        desc_lbl = QLabel(
            tr(
                "crash_dlg.desc",
                "SilentVoice зберіг діагностичні логи останнього збою. Якщо надішлете їх автору на GitHub, це допоможе швидко знайти й виправити проблему.",
            )
        )
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet("color: #CBD5E1; font-size: 13.5px; line-height: 1.45;")
        lay.addWidget(desc_lbl)

        summary_card = QFrame()
        summary_card.setStyleSheet(
            "QFrame { background-color: #262C40; border: 1px solid #3B4463; border-radius: 10px; }"
        )
        s_lay = QVBoxLayout(summary_card)
        s_lay.setContentsMargins(14, 10, 14, 10)
        s_lay.setSpacing(6)

        ts_str = str(self._report_info.get("timestamp", "") or "")
        summary_raw = str(self._report_info.get("summary", "") or "").strip()
        meta_parts = []
        if ts_str:
            meta_parts.append(tr("crash_dlg.time", "Час: {time}").format(time=ts_str))
        if self._app_version:
            meta_parts.append(f"v{self._app_version.lstrip('v')}")
        if meta_parts:
            meta_lbl = QLabel(" · ".join(meta_parts))
            meta_lbl.setStyleSheet("color: #94A3B8; font-size: 12px; border: none; background: transparent;")
            s_lay.addWidget(meta_lbl)

        if summary_raw:
            sum_lbl = QLabel(summary_raw[:240])
            sum_lbl.setWordWrap(True)
            sum_lbl.setStyleSheet(
                "color: #F8FAFC; font-family: Consolas, monospace; font-size: 12px; border: none; background: transparent;"
            )
            s_lay.addWidget(sum_lbl)

        lay.addWidget(summary_card)

        privacy_lbl = QLabel(
            tr(
                "crash_dlg.privacy",
                "🔒 Приватність: до архіву додаються лише crash.log, лог тієї сесії та версія ОС. Файл config.json, API-ключі й сесії НЕ включаються, а ім'я користувача Windows маскується.",
            )
        )
        privacy_lbl.setWordWrap(True)
        privacy_lbl.setStyleSheet("color: #94A3B8; font-size: 12px; line-height: 1.4;")
        lay.addWidget(privacy_lbl)

        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet("color: #4ADE80; font-size: 12.5px; font-weight: 600;")
        self._status_lbl.setVisible(False)
        lay.addWidget(self._status_lbl)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.github_btn = QPushButton(tr("crash_dlg.btn_github", "🐛 Відправити на GitHub"))
        self.github_btn.setProperty("class", "primaryBtn")
        self.github_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.github_btn.clicked.connect(self._on_open_github)
        btn_row.addWidget(self.github_btn)

        self.zip_btn = QPushButton(tr("crash_dlg.btn_zip", "📂 Зберегти .zip"))
        self.zip_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.zip_btn.clicked.connect(self._on_open_zip_only)
        btn_row.addWidget(self.zip_btn)

        btn_row.addStretch(1)

        self.close_btn = QPushButton(tr("dlg_005", "Закрити"))
        self.close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.close_btn)

        lay.addLayout(btn_row)

    def _ensure_zip(self) -> Optional[Path]:
        if self._zip_path is not None and self._zip_path.exists():
            return self._zip_path
        try:
            from core.app_logger import build_crash_report_zip

            self._zip_path = build_crash_report_zip(
                data_dir=self._data_dir,
                crash_log_path=self._crash_log_path,
                logs_dir=self._logs_dir,
                report_info=self._report_info,
                app_version=self._app_version,
            )
            return self._zip_path
        except Exception as exc:
            self._status_lbl.setStyleSheet("color: #F87171; font-size: 12.5px; font-weight: 600;")
            self._status_lbl.setText(tr("crash_dlg.err_zip", "Не вдалося створити архів: {err}").format(err=exc))
            self._status_lbl.setVisible(True)
            return None

    def _reveal_in_explorer(self, path: Path) -> None:
        import os
        import subprocess

        try:
            if os.name == "nt":
                subprocess.Popen(["explorer", f"/select,{path}"])
                return
        except Exception:
            pass
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))

    def _on_open_zip_only(self) -> None:
        zp = self._ensure_zip()
        if zp is None:
            return
        self._reveal_in_explorer(zp)
        self._status_lbl.setStyleSheet("color: #4ADE80; font-size: 12.5px; font-weight: 600;")
        self._status_lbl.setText(
            tr("crash_dlg.zip_ready", "✅ Архів готовий: {name} — можете надіслати його автору.").format(name=zp.name)
        )
        self._status_lbl.setVisible(True)

    def _on_open_github(self) -> None:
        from core.app_logger import build_github_issue_url

        zp = self._ensure_zip()
        if zp is None:
            return
        self._reveal_in_explorer(zp)
        url = build_github_issue_url(
            repo_url=self._repo_url,
            report_info=self._report_info,
            app_version=self._app_version,
            zip_name=zp.name,
        )
        QDesktopServices.openUrl(QUrl(url))
        self._status_lbl.setStyleSheet("color: #4ADE80; font-size: 12.5px; font-weight: 600;")
        self._status_lbl.setText(
            tr(
                "crash_dlg.github_opened",
                "✅ Відкрито GitHub і папку з {name}. Просто перетягніть цей .zip у вікно звіту!",
            ).format(name=zp.name)
        )
        self._status_lbl.setVisible(True)





