"""
ui/reaction_commands.py — Згортуваний блок керування звуковими командами та реакціями (/ \\ |)
у стилі блоку VBAN на сторінці «Швидкі фрази», а також діалог створення/редагування команди.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QSize, QTimer, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.config import BASE_DIR, BUNDLE_DIR, REACTION_AUDIO_DIR, config
from core.i18n import get_ui_language, tr
from core.omnivoice_tags import (
    build_default_whisper_command,
    get_default_reaction_commands,
    is_omnivoice_whisper_command_visible,
    is_reaction_item_playable,
    is_whisper_reaction_item,
    load_reaction_command_items,
    parse_reaction_command_aliases,
    resolve_reaction_audio_path,
    sanitize_reaction_command_name,
    sync_default_reaction_commands_for_lang,
)
from core.tts_manager import tts_manager
from ui.dialogs import DIALOG_STYLESHEET, HotkeyCaptureDialog, center_dialog_on_screen, toggle_collapsible_widget
from ui.overlay_window import TAG_HINT_MODES


_STYLE_LABEL_KEYS: Dict[str, tuple[str, str]] = {
    "emoji_dock": ("rc.style.emoji_dock", "Смайлики без фону (збільшення)"),
    "menu_mini": ("rc.style.menu_mini", "Міні-список (5 рядків)"),
    "menu_micro": ("rc.style.menu_micro", "Мікро-список (4 рядки)"),
    "menu_v_compact": ("rc.style.menu_v_compact", "Вузький список без описів"),
    "emoji_grid": ("rc.style.emoji_grid", "Сітка смайликів (2×4)"),
}

_QUICK_EMOJIS: tuple[str, ...] = ("😂", "😮‍💨", "😮", "😯", "😲", "🤔", "👍", "😤", "🔥", "💀", "🎉", "🎵")


def _copy_audio_to_reaction_dir(src_path: Path) -> str:
    """Копіює обраний звуковий файл у `data/reaction_audio`, щоб команда не ламалась при переміщенні файлу."""
    REACTION_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    safe_stem = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in src_path.stem)[:28].strip("_") or "sound"
    suffix = src_path.suffix.lower() or ".wav"
    dest = REACTION_AUDIO_DIR / f"rc_{int(time.time() * 1000)}_{safe_stem}{suffix}"
    shutil.copy2(src_path, dest)
    return str(dest)


class _ReactionCommandListWidget(QListWidget):
    """Список команд із підтримкою перетягування мишкою (drag-and-drop) для зміни порядку показу."""

    order_changed = Signal(list)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSpacing(3)
        self.setStyleSheet(
            "QListWidget { background-color: #1B1F2D; border: 1px solid #38415E; border-radius: 8px; padding: 4px; }"
            "QListWidget::item { background-color: #262C3F; border: 1px solid #38415E; border-radius: 8px; margin: 1px 0px; }"
            "QListWidget::item:selected { border-color: #7C5CFF; background-color: #2D354E; }"
            "QListWidget::item:hover { border-color: #5B6791; }"
            "QScrollBar:vertical { background: transparent; width: 7px; margin: 4px 2px 4px 0px; }"
            "QScrollBar::handle:vertical { background: #475569; border-radius: 3px; min-height: 24px; }"
            "QScrollBar::handle:vertical:hover { background: #7C5CFF; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
            "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }"
        )

    def dropEvent(self, event) -> None:
        super().dropEvent(event)
        ordered_ids: List[str] = []
        for i in range(self.count()):
            it = self.item(i)
            if it is not None:
                cid = str(it.data(Qt.ItemDataRole.UserRole) or "")
                if cid:
                    ordered_ids.append(cid)
        if ordered_ids:
            self.order_changed.emit(ordered_ids)


class ReactionCommandDialog(QDialog):
    """Діалог створення або редагування звукової команди / реакції."""

    def __init__(
        self,
        item: Optional[Dict[str, Any]] = None,
        *,
        keyboard_hook: Any = None,
        existing_commands: Optional[set[str]] = None,
        omnivoice_enabled: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._hook = keyboard_hook
        self._is_edit = item is not None
        self._existing_commands = {c.lower() for c in (existing_commands or set())}
        self._omnivoice_enabled = bool(omnivoice_enabled)
        self._data: Dict[str, Any] = (
            dict(item)
            if item is not None
            else {
                "id": f"rc_custom_{int(time.time() * 1000)}",
                "command": "",
                "aliases": [],
                "display": "",
                "description": "",
                "audio_file": "",
                "omnivoice_tag": "",
                "hotkey": "",
                "enabled": True,
            }
        )
        self._centered_once = False
        if self._is_edit:
            cur_cmd = sanitize_reaction_command_name(str(self._data.get("command") or ""))
            self._existing_commands.discard(cur_cmd)
            for al in (self._data.get("aliases") or []):
                self._existing_commands.discard(sanitize_reaction_command_name(str(al)))
        self._was_playable_initially = is_reaction_item_playable(
            self._data,
            self._omnivoice_enabled,
            base_dir=BASE_DIR,
        )

        self.setWindowTitle(
            tr("rc.dlg.edit_title", "Редагувати команду")
            if self._is_edit
            else tr("rc.dlg.add_title", "Нова звукова команда")
        )
        self.setMinimumWidth(470)
        self.setMaximumWidth(560)
        self.setStyleSheet(DIALOG_STYLESHEET)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(10)

        hdr = QLabel(
            tr("rc.dlg.edit_hdr", "✎ Редагування звукової команди")
            if self._is_edit
            else tr("rc.dlg.add_hdr", "🎭 Нова звукова команда")
        )
        hdr.setObjectName("dialogTitle")
        lay.addWidget(hdr)

        # 1. Назва команди (підтримує кілька варіантів через кому для швидкого друку)
        lay.addWidget(QLabel(tr("rc.dlg.cmd_lbl", "Команди (можна кілька через кому; виклик через /, \\ або |):")))
        cmd_row = QHBoxLayout()
        cmd_row.setSpacing(8)
        init_aliases = [
            sanitize_reaction_command_name(str(a))
            for a in (self._data.get("aliases") or [])
            if sanitize_reaction_command_name(str(a))
        ]
        init_cmd = sanitize_reaction_command_name(str(self._data.get("command") or ""))
        if init_cmd and init_cmd not in init_aliases:
            init_aliases.insert(0, init_cmd)
        self.cmd_edit = QLineEdit(", ".join(init_aliases) if init_aliases else str(self._data.get("command") or ""))
        self.cmd_edit.setPlaceholderText(tr("rc.dlg.cmd_ph", "Напр.: хаха, ахаха, ахах, хах"))
        self.cmd_edit.setMaxLength(160)
        self.cmd_edit.textChanged.connect(self._refresh_preview)
        cmd_row.addWidget(self.cmd_edit, 1)

        self.cmd_preview_lbl = QLabel()
        self.cmd_preview_lbl.setStyleSheet("color: #A78BFA; font-size: 12px; font-weight: 600;")
        cmd_row.addWidget(self.cmd_preview_lbl)
        lay.addLayout(cmd_row)

        # 2. Відображення у тексті (емодзі або порожньо -> [команда])
        lay.addWidget(QLabel(tr("rc.dlg.display_lbl", "Відображення в оверлеї (емодзі або залиште порожнім → [команда]):")))
        disp_row = QHBoxLayout()
        disp_row.setSpacing(8)
        self.display_edit = QLineEdit(str(self._data.get("display") or ""))
        self.display_edit.setPlaceholderText(tr("rc.dlg.display_ph", "Напр. 😂 (порожньо → [хаха])"))
        self.display_edit.setMaxLength(8)
        self.display_edit.textChanged.connect(self._refresh_preview)
        disp_row.addWidget(self.display_edit, 1)

        clear_disp_btn = QPushButton("✕")
        clear_disp_btn.setFixedSize(32, 32)
        clear_disp_btn.setStyleSheet("padding: 0px; font-size: 13px;")
        clear_disp_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_disp_btn.setToolTip(tr("rc.dlg.display_clear", "Очистити знак (відображатиметься як [команда])"))
        clear_disp_btn.clicked.connect(self.display_edit.clear)
        disp_row.addWidget(clear_disp_btn)

        self.display_preview_lbl = QLabel()
        self.display_preview_lbl.setStyleSheet("color: #CBD5E1; font-size: 12px;")
        disp_row.addWidget(self.display_preview_lbl)
        lay.addLayout(disp_row)

        emoji_strip = QHBoxLayout()
        emoji_strip.setSpacing(4)
        for em in _QUICK_EMOJIS:
            eb = QPushButton(em)
            eb.setFixedSize(30, 28)
            eb.setCursor(Qt.CursorShape.PointingHandCursor)
            eb.setStyleSheet("padding: 0px; font-size: 14px;")
            eb.clicked.connect(lambda _c=False, e=em: self.display_edit.setText(e))
            emoji_strip.addWidget(eb)
        emoji_strip.addStretch(1)
        lay.addLayout(emoji_strip)

        # 3. Короткий опис
        lay.addWidget(QLabel(tr("rc.dlg.desc_lbl", "Короткий опис (необов’язково, видно у списках):")))
        self.desc_edit = QLineEdit(str(self._data.get("description") or ""))
        self.desc_edit.setPlaceholderText(tr("rc.dlg.desc_ph", "Напр.: Мій звук сміху"))
        self.desc_edit.setMaxLength(64)
        lay.addWidget(self.desc_edit)

        # 4. Аудіофайл
        audio_box = QFrame()
        audio_box.setStyleSheet(
            "QFrame { background-color: #1A1E2B; border: 1px solid #353D58; border-radius: 10px; }"
        )
        ab_lay = QVBoxLayout(audio_box)
        ab_lay.setContentsMargins(12, 10, 12, 10)
        ab_lay.setSpacing(8)

        self.audio_status_lbl = QLabel()
        self.audio_status_lbl.setWordWrap(True)
        self.audio_status_lbl.setStyleSheet("border: none; background: transparent; font-size: 12px;")
        ab_lay.addWidget(self.audio_status_lbl)

        ab_btns = QHBoxLayout()
        ab_btns.setSpacing(8)
        _ab_btn_css = "padding: 6px 12px; min-height: 18px;"
        self.btn_pick_file = QPushButton(tr("rc.dlg.pick_file", "🎵 Обрати аудіофайл"))
        self.btn_pick_file.setStyleSheet(_ab_btn_css)
        self.btn_pick_file.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_pick_file.clicked.connect(self._on_pick_file)
        ab_btns.addWidget(self.btn_pick_file, 1)

        self.btn_preview_file = QPushButton(tr("rc.dlg.preview_file", "▶ Прослухати"))
        self.btn_preview_file.setStyleSheet(_ab_btn_css)
        self.btn_preview_file.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_preview_file.clicked.connect(self._on_preview_file)
        ab_btns.addWidget(self.btn_preview_file)

        self.btn_clear_file = QPushButton(tr("rc.dlg.clear_file", "✕ Прибрати"))
        self.btn_clear_file.setStyleSheet(_ab_btn_css)
        self.btn_clear_file.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear_file.clicked.connect(self._on_clear_file)
        ab_btns.addWidget(self.btn_clear_file)
        ab_lay.addLayout(ab_btns)
        lay.addWidget(audio_box)

        # 5. Гаряча клавіша (бінд для швидкої вставки в текст)
        lay.addWidget(QLabel(tr("rc.dlg.hotkey_lbl", "Бінд для швидкої вставки в текст (необов’язково):")))
        hk_row = QHBoxLayout()
        hk_row.setSpacing(8)
        self.hk_edit = QLineEdit(str(self._data.get("hotkey") or ""))
        self.hk_edit.setReadOnly(True)
        self.hk_edit.setPlaceholderText(tr("dlg_039", "Не призначено (необов'язково)"))
        hk_row.addWidget(self.hk_edit, 1)

        hk_pick_btn = QPushButton(tr("dlg_040", "⌨️ Вибрати клавішу"))
        hk_pick_btn.setStyleSheet("padding: 6px 12px; min-height: 18px;")
        hk_pick_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        hk_pick_btn.setEnabled(self._hook is not None)
        hk_pick_btn.clicked.connect(self._on_record_hotkey)
        hk_row.addWidget(hk_pick_btn)

        hk_clear_btn = QPushButton("✕")
        hk_clear_btn.setFixedSize(32, 32)
        hk_clear_btn.setStyleSheet("padding: 0px; font-size: 13px;")
        hk_clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        hk_clear_btn.setToolTip(tr("rc.dlg.hotkey_clear", "Очистити бінд"))
        hk_clear_btn.clicked.connect(self.hk_edit.clear)
        hk_row.addWidget(hk_clear_btn)
        lay.addLayout(hk_row)

        self.err_lbl = QLabel("")
        self.err_lbl.setWordWrap(True)
        self.err_lbl.setStyleSheet("color: #F87171; font-size: 12px;")
        self.err_lbl.setVisible(False)
        lay.addWidget(self.err_lbl)

        # Кнопки Зберегти / Скасувати
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        cancel_btn = QPushButton(tr("common.cancel", "Скасувати"))
        cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        save_btn = QPushButton(tr("common.save", "Зберегти"))
        save_btn.setObjectName("primaryBtn")
        save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(save_btn)
        lay.addLayout(btn_row)

        self._refresh_audio_status()
        self._refresh_preview()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._centered_once:
            self._centered_once = True
            center_dialog_on_screen(self, adjust_size=True)
            QTimer.singleShot(0, lambda: center_dialog_on_screen(self, adjust_size=True) if self.isVisible() else None)

    def _refresh_preview(self) -> None:
        aliases = parse_reaction_command_aliases(self.cmd_edit.text())
        cmd = aliases[0] if aliases else ""
        disp = self.display_edit.text().strip()
        if len(aliases) > 1:
            self.cmd_preview_lbl.setText(f"\\{cmd} (+{len(aliases) - 1})")
            self.cmd_preview_lbl.setToolTip(", ".join(f"\\{a}" for a in aliases))
        else:
            self.cmd_preview_lbl.setText(f"\\{cmd}" if cmd else "")
            self.cmd_preview_lbl.setToolTip("")
        shown = disp if disp else (f"[{cmd}]" if cmd else "[...]")
        self.display_preview_lbl.setText(tr("rc.dlg.in_text", "У тексті: {shown}", shown=shown))

    def _refresh_audio_status(self) -> None:
        is_whisper = is_whisper_reaction_item(self._data)
        resolved = "" if is_whisper else resolve_reaction_audio_path(self._data, base_dir=BASE_DIR)
        has_file = bool(resolved)
        self.btn_pick_file.setVisible(not is_whisper)
        self.btn_preview_file.setVisible(has_file)
        self.btn_clear_file.setVisible(has_file)
        ov_tag = str(self._data.get("omnivoice_tag") or "").strip()

        if is_whisper:
            self.audio_status_lbl.setText(
                tr(
                    "rc.dlg.whisper_info",
                    "🤫 Вбудований режим шепоту OmniVoice. Вставте в текст 1 раз — фраза далі звучатиме пошепки; вставте вдруге — повернеться звичайний голос.",
                )
            )
            self.audio_status_lbl.setStyleSheet("border: none; background: transparent; color: #C4B5FD; font-size: 12px;")
        elif has_file:
            self.audio_status_lbl.setText(
                tr("rc.dlg.file_attached", "✅ Файл: <b>{name}</b> (працює з будь-яким TTS)", name=Path(resolved).name)
            )
            self.audio_status_lbl.setStyleSheet("border: none; background: transparent; color: #34D399; font-size: 12px;")
        elif ov_tag and self._omnivoice_enabled:
            self.audio_status_lbl.setText(
                tr("rc.dlg.file_omnivoice", "🎙️ Вбудована реакція OmniVoice ({tag}). Можна прикріпити свій файл для всіх TTS.", tag=ov_tag)
            )
            self.audio_status_lbl.setStyleSheet("border: none; background: transparent; color: #C4B5FD; font-size: 12px;")
        else:
            self.audio_status_lbl.setText(
                tr("rc.dlg.file_missing", "⚪ Файл не обрано — доки не додано файл, команду не можна увімкнути.")
            )
            self.audio_status_lbl.setStyleSheet("border: none; background: transparent; color: #94A3B8; font-size: 12px;")

    def _on_pick_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            tr("rc.pick_file.title", "Оберіть аудіофайл для команди"),
            "",
            tr("dlg_044", "Аудіофайли (*.wav *.mp3 *.ogg *.flac);;Усі файли (*.*)"),
        )
        if not file_path:
            return
        src = Path(file_path)
        if not src.is_file():
            return
        try:
            self._data["audio_file"] = _copy_audio_to_reaction_dir(src)
            if not self.cmd_edit.text().strip():
                guess = sanitize_reaction_command_name(src.stem)[:16]
                if guess:
                    self.cmd_edit.setText(guess)
            self._refresh_audio_status()
            self._refresh_preview()
        except Exception as exc:
            self.err_lbl.setText(str(exc))
            self.err_lbl.setVisible(True)

    def _on_preview_file(self) -> None:
        resolved = resolve_reaction_audio_path(self._data, base_dir=BASE_DIR)
        if resolved:
            tts_manager.play_sample_preview(resolved, monitor_only=True)

    def _on_clear_file(self) -> None:
        self._data["audio_file"] = ""
        self._refresh_audio_status()

    def _on_record_hotkey(self) -> None:
        if self._hook is None:
            return
        dlg = HotkeyCaptureDialog(
            self._hook,
            self.hk_edit.text(),
            tr("rc.dlg.hotkey_title", "Бінд для звукової команди"),
            self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.recorded_hotkey is not None:
            self.hk_edit.setText(dlg.recorded_hotkey)

    def _on_save(self) -> None:
        parsed_aliases = parse_reaction_command_aliases(self.cmd_edit.text())
        if not parsed_aliases:
            self.err_lbl.setText(tr("rc.dlg.err_empty_cmd", "⚠️ Введіть назву команди (наприклад: хаха)."))
            self.err_lbl.setVisible(True)
            self.cmd_edit.setFocus()
            return
        cmd = parsed_aliases[0]
        dup = next((al for al in parsed_aliases if al in self._existing_commands), "")
        if dup:
            self.err_lbl.setText(tr("rc.dlg.err_dup_cmd", "⚠️ Команда \\{cmd} уже існує у списку.", cmd=dup))
            self.err_lbl.setVisible(True)
            self.cmd_edit.setFocus()
            return
        aliases = parsed_aliases

        had_playable = is_reaction_item_playable(
            {"audio_file": self._data.get("audio_file", ""), "omnivoice_tag": self._data.get("omnivoice_tag", "")},
            self._omnivoice_enabled,
            base_dir=BASE_DIR,
        )
        self._data["command"] = cmd
        self._data["aliases"] = aliases
        self._data["display"] = self.display_edit.text().strip()
        self._data["description"] = self.desc_edit.text().strip()
        self._data["hotkey"] = self.hk_edit.text().strip()
        if not had_playable:
            self._data["enabled"] = False
        elif not self._is_edit or not self._was_playable_initially:
            # Якщо команда нова або щойно отримала аудіофайл (до цього була заблокована) — вмикаємо автоматично
            self._data["enabled"] = True
        self.accept()

    def get_result(self) -> Dict[str, Any]:
        return self._data


class ReactionCommandsCard(QFrame):
    """
    Згортуваний блок «Звукові команди та реакції (/ \\ |)» у стилі блоку VBAN.
    Розміщується на сторінці «Швидкі фрази» одразу після картки швидких фраз.
    """

    commands_changed = Signal()
    style_changed_preview = Signal(str)

    def __init__(
        self,
        keyboard_hook: Any = None,
        *,
        combo_cls: Any = QComboBox,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._hook = keyboard_hook
        self.setProperty("class", "card")

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 10, 16, 10)
        root.setSpacing(0)

        # Заголовок (згортуваний, як у VBAN)
        hdr_row = QHBoxLayout()
        hdr_row.setContentsMargins(0, 0, 0, 0)

        self.expand_btn = QToolButton()
        self.expand_btn.setCheckable(True)
        self.expand_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.expand_btn.setArrowType(Qt.ArrowType.RightArrow)
        self.expand_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.expand_btn.setText(tr("rc.card.title", "Звукові команди та реакції (/ \\ |)"))
        self.expand_btn.setStyleSheet(
            "QToolButton { border: 0; background: transparent; color: #F1F5F9; "
            "font-size: 14px; font-weight: 700; padding: 4px; text-align: left; }"
            "QToolButton:hover { color: #A78BFA; }"
        )
        hdr_row.addWidget(self.expand_btn)
        hdr_row.addStretch(1)

        self.status_badge = QLabel()
        self.status_badge.setStyleSheet("font-size: 12px; color: #AAB4C8;")
        hdr_row.addWidget(self.status_badge)
        root.addLayout(hdr_row)

        # Тіло картки (згорнуте за замовчуванням; верхній відступ всередині body, щоб не було стрибка на 8px)
        self.body = QWidget()
        body_lay = QVBoxLayout(self.body)
        body_lay.setContentsMargins(0, 8, 0, 0)
        body_lay.setSpacing(10)

        help_lbl = QLabel(
            tr(
                "rc.card.help",
                "Введіть /, \\ або | під час набору тексту, щоб вставити власний звук чи реакцію прямо в речення. "
                "Перетягуйте рядки мишкою, щоб змінити порядок у підказці.",
            )
        )
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet("color: #AAB4C8; font-size: 12px;")
        body_lay.addWidget(help_lbl)

        # Рядок 1: вибір стилю + кнопка прев’ю + галочка інтеграції з OmniVoice
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        style_lbl = QLabel(tr("rc.style.label", "Стиль підказки:"))
        style_lbl.setStyleSheet("color: #CBD5E1; font-size: 13px;")
        ctrl_row.addWidget(style_lbl)

        self.style_combo = combo_cls()
        for mode_id in TAG_HINT_MODES:
            key, default_lbl = _STYLE_LABEL_KEYS.get(mode_id, ("", mode_id))
            self.style_combo.addItem(tr(key, default_lbl) if key else default_lbl, mode_id)
        cur_style = str(config.get("reaction_picker_style", "emoji_dock") or "emoji_dock").strip()
        s_idx = self.style_combo.findData(cur_style)
        if s_idx >= 0:
            self.style_combo.setCurrentIndex(s_idx)
        self.style_combo.currentIndexChanged.connect(self._on_style_changed)
        ctrl_row.addWidget(self.style_combo, 1)

        preview_btn = QPushButton(tr("rc.style.preview_btn", "👁 Показати"))
        preview_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        preview_btn.setToolTip(tr("rc.style.preview_tip", "Показати, як виглядає обраний стиль підказки поверх екрана"))
        preview_btn.clicked.connect(lambda: self.style_changed_preview.emit(str(self.style_combo.currentData() or "emoji_dock")))
        ctrl_row.addWidget(preview_btn)
        body_lay.addLayout(ctrl_row)

        self.omnivoice_chk = QCheckBox(
            tr("rc.omnivoice.enable", "Інтеграція з OmniVoice (вбудовані живі реакції: сміх, зітхання, вау…)")
        )
        self.omnivoice_chk.setChecked(bool(config.get("reaction_omnivoice_enabled", True)))
        self.omnivoice_chk.toggled.connect(self._on_omnivoice_toggled)
        body_lay.addWidget(self.omnivoice_chk)

        # Рядок 2: дії списку
        list_hdr = QHBoxLayout()
        list_hdr.setSpacing(8)
        list_title = QLabel(tr("rc.list.title", "Список команд:"))
        list_title.setProperty("class", "cardHeader")
        list_hdr.addWidget(list_title)
        list_hdr.addStretch(1)

        reset_btn = QPushButton(tr("rc.list.reset_btn", "↺ Стандартні"))
        reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reset_btn.setToolTip(tr("rc.list.reset_tip", "Відновити стандартний набір реакцій"))
        reset_btn.clicked.connect(self._on_reset_defaults)
        list_hdr.addWidget(reset_btn)

        add_btn = QPushButton(tr("rc.list.add_btn", "+ Додати команду"))
        add_btn.setProperty("class", "primaryBtn")
        add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_btn.clicked.connect(self._on_add_command)
        list_hdr.addWidget(add_btn)
        body_lay.addLayout(list_hdr)

        # Список команд із drag-and-drop
        self.cmd_list = _ReactionCommandListWidget()
        self.cmd_list.order_changed.connect(self._on_order_changed)
        body_lay.addWidget(self.cmd_list)

        self.body.setVisible(False)
        root.addWidget(self.body)

        self.expand_btn.toggled.connect(self._on_expand_toggled)
        self._ensure_seeded()
        self.rebuild_list()

    def _on_expand_toggled(self, checked: bool) -> None:
        toggle_collapsible_widget(self.expand_btn, self.body, checked)

    def _ui_lang(self) -> str:
        return str(get_ui_language() or config.get("ui_language") or "uk").strip().lower() or "uk"

    def _ensure_seeded(self) -> None:
        lang = self._ui_lang()
        if config.get("reaction_commands", None) is None:
            config.set("reaction_commands", get_default_reaction_commands(lang))
            config.set("reaction_whisper_seeded", True)
        else:
            items = load_reaction_command_items(config.data, lang)
            changed = False
            if not bool(config.get("reaction_whisper_seeded", False)):
                if not any(is_whisper_reaction_item(it) for it in items):
                    items.append(build_default_whisper_command(lang))
                    changed = True
                config.set("reaction_whisper_seeded", True)
            if changed:
                config.set("reaction_commands", items)
            synced = sync_default_reaction_commands_for_lang(config.data, lang)
            if synced is not None:
                config.set("reaction_commands", synced)

    def _get_items(self) -> List[Dict[str, Any]]:
        return load_reaction_command_items(config.data, self._ui_lang())

    def _save_items(self, items: List[Dict[str, Any]]) -> None:
        config.set("reaction_commands", items)
        self.rebuild_list()
        self.commands_changed.emit()

    def _on_style_changed(self) -> None:
        style = str(self.style_combo.currentData() or "emoji_dock").strip()
        config.set("reaction_picker_style", style)
        self.style_changed_preview.emit(style)

    def _on_omnivoice_toggled(self, checked: bool) -> None:
        config.set("reaction_omnivoice_enabled", bool(checked))
        self.rebuild_list()
        self.commands_changed.emit()

    def _on_order_changed(self, ordered_ids: List[str]) -> None:
        items = self._get_items()
        by_id = {str(it.get("id") or ""): it for it in items}
        reordered: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for cid in ordered_ids:
            if cid in by_id and cid not in seen:
                reordered.append(by_id[cid])
                seen.add(cid)
        for it in items:
            cid = str(it.get("id") or "")
            if cid not in seen:
                reordered.append(it)
        self._save_items(reordered)

    def _on_toggle_item_enabled(self, cid: str, checked: bool) -> None:
        items = self._get_items()
        for it in items:
            if str(it.get("id") or "") == cid:
                it["enabled"] = bool(checked)
                break
        self._save_items(items)

    def _on_quick_attach_file(self, cid: str) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            tr("rc.pick_file.title", "Оберіть аудіофайл для команди"),
            "",
            tr("dlg_044", "Аудіофайли (*.wav *.mp3 *.ogg *.flac);;Усі файли (*.*)"),
        )
        if not file_path:
            return
        src = Path(file_path)
        if not src.is_file():
            return
        try:
            copied = _copy_audio_to_reaction_dir(src)
        except Exception as exc:
            QMessageBox.warning(self, tr("ui_441", "Помилка"), str(exc))
            return
        items = self._get_items()
        for it in items:
            if str(it.get("id") or "") == cid:
                it["audio_file"] = copied
                it["enabled"] = True
                break
        self._save_items(items)

    def _on_add_command(self) -> None:
        items = self._get_items()
        existing = {
            sanitize_reaction_command_name(str(al))
            for it in items
            for al in ([it.get("command", "")] + list(it.get("aliases") or []))
            if sanitize_reaction_command_name(str(al))
        }
        dlg = ReactionCommandDialog(
            None,
            keyboard_hook=self._hook,
            existing_commands=existing,
            omnivoice_enabled=self.omnivoice_chk.isChecked(),
            parent=self.window(),
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            items.append(dlg.get_result())
            self._save_items(items)

    def _on_quick_bind_hotkey(self, cid: str) -> None:
        if self._hook is None:
            return
        items = self._get_items()
        idx = next((i for i, it in enumerate(items) if str(it.get("id") or "") == cid), -1)
        if idx < 0:
            return
        cur_hk = str(items[idx].get("hotkey") or "")
        cmd_name = str(items[idx].get("command") or "")
        dlg = HotkeyCaptureDialog(
            self._hook,
            cur_hk,
            tr("rc.dlg.hotkey_title_cmd", "Бінд для \\{cmd}", cmd=cmd_name),
            self.window(),
        )
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.recorded_hotkey is not None:
            items[idx]["hotkey"] = dlg.recorded_hotkey.strip()
            self._save_items(items)

    def _on_edit_command(self, cid: str) -> None:
        items = self._get_items()
        idx = next((i for i, it in enumerate(items) if str(it.get("id") or "") == cid), -1)
        if idx < 0:
            return
        existing = {
            sanitize_reaction_command_name(str(al))
            for it in items
            for al in ([it.get("command", "")] + list(it.get("aliases") or []))
            if sanitize_reaction_command_name(str(al))
        }
        dlg = ReactionCommandDialog(
            items[idx],
            keyboard_hook=self._hook,
            existing_commands=existing,
            omnivoice_enabled=self.omnivoice_chk.isChecked(),
            parent=self.window(),
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            items[idx] = dlg.get_result()
            self._save_items(items)

    def _on_delete_command(self, cid: str) -> None:
        items = self._get_items()
        idx = next((i for i, it in enumerate(items) if str(it.get("id") or "") == cid), -1)
        if idx < 0:
            return
        cmd_name = str(items[idx].get("command") or "")
        ans = QMessageBox.question(
            self.window(),
            tr("common.confirm", "Підтвердження"),
            tr("rc.confirm.delete", "Точно видалити команду \\{cmd}?", cmd=cmd_name),
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        items.pop(idx)
        self._save_items(items)

    def _on_reset_defaults(self) -> None:
        ans = QMessageBox.question(
            self.window(),
            tr("common.confirm", "Підтвердження"),
            tr("rc.confirm.reset", "Відновити стандартний список реакцій?"),
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        self._save_items(get_default_reaction_commands(self._ui_lang()))

    def rebuild_list(self) -> None:
        items = self._get_items()
        ov_enabled = bool(config.get("reaction_omnivoice_enabled", True))
        whisper_visible = is_omnivoice_whisper_command_visible(config.data)
        self.omnivoice_chk.blockSignals(True)
        self.omnivoice_chk.setChecked(ov_enabled)
        self.omnivoice_chk.blockSignals(False)

        self.cmd_list.clear()
        active_count = 0

        for it in items:
            is_whisper = is_whisper_reaction_item(it)
            if is_whisper and not whisper_visible:
                continue
            cid = str(it.get("id") or "")
            cmd = str(it.get("command") or "").strip()
            disp = str(it.get("display") or "").strip()
            desc = str(it.get("description") or "").strip()
            ov_tag = str(it.get("omnivoice_tag") or "").strip()
            hk_str = str(it.get("hotkey") or "").strip()
            resolved_audio = resolve_reaction_audio_path(it, base_dir=BASE_DIR)
            has_file = bool(resolved_audio)
            can_use = is_reaction_item_playable(it, ov_enabled, base_dir=BASE_DIR)
            is_checked = can_use and bool(it.get("enabled", True))
            if is_checked:
                active_count += 1

            li = QListWidgetItem()
            li.setData(Qt.ItemDataRole.UserRole, cid)
            li.setSizeHint(QSize(0, 44))

            row_w = QWidget()
            row_w.setStyleSheet("background: transparent;")
            r_lay = QHBoxLayout(row_w)
            r_lay.setContentsMargins(8, 4, 6, 4)
            r_lay.setSpacing(7)

            grip = QLabel("⋮⋮")
            grip.setCursor(Qt.CursorShape.OpenHandCursor)
            grip.setToolTip(tr("rc.row.drag_tip", "Перетягніть мишкою, щоб змінити порядок"))
            grip.setStyleSheet("color: #64748B; font-size: 13px; font-weight: 700; padding: 0 2px;")
            r_lay.addWidget(grip)

            chk = QCheckBox()
            chk.setChecked(is_checked)
            chk.setEnabled(can_use)
            if not can_use:
                chk.setToolTip(
                    tr(
                        "rc.chk.disabled_tip",
                        "Додайте аудіофайл (або увімкніть інтеграцію з OmniVoice), щоб активувати команду",
                    )
                )
            else:
                chk.setToolTip(tr("rc.chk.enabled_tip", "Увімкнути або приховати цю команду в підказці"))
            chk.toggled.connect(lambda c, _id=cid: self._on_toggle_item_enabled(_id, c))
            r_lay.addWidget(chk)

            # Емодзі або автоматичне `[команда]`
            disp_lbl = QLabel(disp if disp else f"[{cmd}]")
            disp_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            disp_lbl.setMinimumWidth(34 if disp else 52)
            disp_lbl.setStyleSheet(
                "font-size: 15px; color: #FFFFFF; background: transparent;"
                if disp
                else "font-size: 11px; font-weight: 700; color: #C4B5FD; background: rgba(124,92,255,0.16); "
                "border: 1px solid rgba(124,92,255,0.35); border-radius: 5px; padding: 2px 5px;"
            )
            r_lay.addWidget(disp_lbl)

            # Команда + короткий опис
            text_lbl = QLabel()
            text_lbl.setMinimumWidth(0)
            text_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            cmd_col = "#F1F5F9" if is_checked else "#7C879E"
            desc_col = "#94A3B8" if is_checked else "#5E687D"
            extra_aliases = [
                sanitize_reaction_command_name(str(a))
                for a in (it.get("aliases") or [])
                if sanitize_reaction_command_name(str(a)) and sanitize_reaction_command_name(str(a)) != cmd
            ]
            alias_short = ""
            if extra_aliases:
                shown_al = ", ".join(extra_aliases[:3]) + ("…" if len(extra_aliases) > 3 else "")
                alias_short = f" <span style='color:#64748B; font-size:11px;'>({shown_al})</span>"
            if desc:
                text_lbl.setText(
                    f"<span style='color:{cmd_col}; font-weight:700;'>\\{cmd}</span>{alias_short} "
                    f"<span style='color:{desc_col};'>— {desc}</span>"
                )
            else:
                text_lbl.setText(f"<span style='color:{cmd_col}; font-weight:700;'>\\{cmd}</span>{alias_short}")
            all_al_tip = ", ".join(f"\\{a}" for a in ([cmd] + extra_aliases))
            text_lbl.setToolTip(all_al_tip + (f" — {desc}" if desc else ""))
            r_lay.addWidget(text_lbl, 1)

            # Бейдж бінду (якщо призначено)
            if hk_str:
                hk_lbl = QLabel(f"⌨ {hk_str}")
                hk_lbl.setToolTip(tr("rc.row.hotkey_bound_tip", "Бінд: {hk} (натисніть під час набору, щоб вставити)", hk=hk_str))
                hk_lbl.setStyleSheet(
                    "font-size: 11px; font-weight: 600; color: #93C5FD; background: rgba(59,130,246,0.14); "
                    "border: 1px solid rgba(96,165,250,0.34); border-radius: 6px; padding: 2px 7px;"
                    if is_checked
                    else "font-size: 11px; color: #64748B; background: rgba(100,116,139,0.10); "
                    "border: 1px solid rgba(100,116,139,0.20); border-radius: 6px; padding: 2px 7px;"
                )
                r_lay.addWidget(hk_lbl)

            # Бейдж джерела звуку (Файл / OmniVoice / Без файлу)
            src_lbl = QLabel()
            if has_file:
                fname = Path(resolved_audio).name
                short_name = fname if len(fname) <= 18 else (fname[:15] + "…")
                src_lbl.setText(f"🎵 {short_name}")
                src_lbl.setToolTip(fname)
                src_lbl.setStyleSheet(
                    "font-size: 11px; color: #34D399; background: rgba(16,185,129,0.13); "
                    "border: 1px solid rgba(52,211,153,0.3); border-radius: 6px; padding: 2px 7px;"
                )
            elif ov_tag and ov_enabled:
                src_lbl.setText("OmniVoice")
                src_lbl.setToolTip(ov_tag)
                src_lbl.setStyleSheet(
                    "font-size: 11px; color: #C4B5FD; background: rgba(124,92,255,0.16); "
                    "border: 1px solid rgba(167,139,250,0.3); border-radius: 6px; padding: 2px 7px;"
                )
            else:
                src_lbl.setText(tr("rc.badge.no_file", "Без файлу"))
                src_lbl.setStyleSheet(
                    "font-size: 11px; color: #94A3B8; background: rgba(148,163,184,0.12); "
                    "border: 1px solid rgba(148,163,184,0.22); border-radius: 6px; padding: 2px 7px;"
                )
            r_lay.addWidget(src_lbl)

            # Кнопки: Прослухати (якщо є файл), Додати/змінити файл, Редагувати, Видалити
            if has_file:
                play_btn = QPushButton("▶")
                play_btn.setFixedSize(26, 26)
                play_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                play_btn.setToolTip(tr("rc.dlg.preview_file", "▶ Прослухати"))
                play_btn.setStyleSheet(
                    "QPushButton { padding: 0px; font-size: 11px; background: #2D3748; border: 1px solid #475569; border-radius: 6px; color: #34D399; }"
                    "QPushButton:hover { border-color: #34D399; background: #334155; }"
                )
                play_btn.clicked.connect(lambda _c=False, af=resolved_audio: tts_manager.play_sample_preview(af, monitor_only=True))
                r_lay.addWidget(play_btn)

            if not is_whisper:
                file_btn = QPushButton("🎵")
                file_btn.setFixedSize(26, 26)
                file_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                file_btn.setToolTip(
                    tr("rc.row.change_file_tip", "Змінити аудіофайл")
                    if has_file
                    else tr("rc.row.add_file_tip", "Додати аудіофайл (.wav, .mp3, .ogg, .flac)")
                )
                file_btn.setStyleSheet(
                    "QPushButton { padding: 0px; font-size: 12px; background: #2D354E; border: 1px solid #635BFF; border-radius: 6px; }"
                    "QPushButton:hover { background: #3B4466; border-color: #A78BFA; }"
                    if not can_use
                    else "QPushButton { padding: 0px; font-size: 12px; border-radius: 6px; }"
                )
                file_btn.clicked.connect(lambda _c=False, _id=cid: self._on_quick_attach_file(_id))
                r_lay.addWidget(file_btn)

            hk_row_btn = QPushButton("⌨")
            hk_row_btn.setFixedSize(26, 26)
            hk_row_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            hk_row_btn.setEnabled(self._hook is not None)
            hk_row_btn.setToolTip(
                tr("rc.row.hotkey_change_tip", "Змінити бінд ({hk})", hk=hk_str)
                if hk_str
                else tr("rc.row.hotkey_add_tip", "Призначити бінд для швидкої вставки в текст")
            )
            hk_row_btn.setStyleSheet(
                "QPushButton { padding: 0px; font-size: 12px; color: #93C5FD; background: #22304A; border: 1px solid #3B82F6; border-radius: 6px; }"
                "QPushButton:hover { background: #2B3D5E; border-color: #60A5FA; }"
                if hk_str
                else "QPushButton { padding: 0px; font-size: 12px; border-radius: 6px; }"
            )
            hk_row_btn.clicked.connect(lambda _c=False, _id=cid: self._on_quick_bind_hotkey(_id))
            r_lay.addWidget(hk_row_btn)

            edit_btn = QPushButton()
            edit_btn.setFixedSize(26, 26)
            edit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            edit_btn.setIcon(QIcon(str(BUNDLE_DIR / "assets" / "edit.svg")))
            edit_btn.setIconSize(QSize(14, 14))
            edit_btn.setToolTip(tr("rc.row.edit_tip", "Редагувати команду"))
            edit_btn.setStyleSheet("padding: 0px; qproperty-iconSize: 14px 14px;")
            edit_btn.clicked.connect(lambda _c=False, _id=cid: self._on_edit_command(_id))
            r_lay.addWidget(edit_btn)

            del_btn = QPushButton()
            del_btn.setFixedSize(26, 26)
            del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            del_btn.setIcon(QIcon(str(BUNDLE_DIR / "assets" / "delete.svg")))
            del_btn.setIconSize(QSize(15, 15))
            del_btn.setToolTip(tr("rc.row.delete_tip", "Видалити команду"))
            del_btn.setStyleSheet("QPushButton { padding: 0px; text-align: center; }")
            del_btn.clicked.connect(lambda _c=False, _id=cid: self._on_delete_command(_id))
            r_lay.addWidget(del_btn)

            self.cmd_list.addItem(li)
            self.cmd_list.setItemWidget(li, row_w)

        vis_rows = max(1, min(6, self.cmd_list.count()))
        self.cmd_list.setFixedHeight(vis_rows * 50 + 12)

        if active_count > 0:
            self.status_badge.setText(tr("rc.card.badge_active", "🟢 {n} активних", n=active_count))
            self.status_badge.setStyleSheet("font-size: 12px; color: #4ADE80;")
        else:
            self.status_badge.setText(tr("rc.card.badge_none", "⚪ Немає активних"))
            self.status_badge.setStyleSheet("font-size: 12px; color: #AAB4C8;")
