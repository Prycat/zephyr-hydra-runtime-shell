# -*- coding: utf-8 -*-
"""
zephyr_gui.py — Zephyr Command Workbench
Prycat Research Team
PySide6 GUI wrapping agent.py via subprocess pipe.
Python 3.9 compatible.
"""
import sys
import os
import math
import html
import random
import textwrap
import subprocess
import queue
import threading
import time
from collections import deque
from typing import Optional
import json as _json
import urllib.request
import re

from PySide6.QtCore import (
    Qt, QThread, Signal, QTimer, QPointF, QRectF, QPoint, QRect, QEvent
)
from PySide6.QtGui import (
    QColor, QPainter, QPen, QBrush, QRadialGradient,
    QLinearGradient, QFont, QPalette, QFontDatabase,
    QTextCursor, QTextCharFormat, QPainterPath,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPlainTextEdit, QLineEdit, QPushButton,
    QSplitter, QScrollArea, QLabel, QSizePolicy, QFrame,
    QCheckBox, QSlider,
)

_CONFIG_DEFAULTS = {
    "active_model": "hermes3:8b",
    "oracle_model": "hermes3:8b",
    "turboquant_enabled": False,
}


def _zephyr_config_path() -> str:
    return os.path.join(os.path.expanduser("~/.zephyr"), "config.json")


def load_zephyr_config() -> dict:
    """Load ~/.zephyr/config.json, returning defaults for missing keys."""
    path = _zephyr_config_path()
    try:
        with open(path, "r") as f:
            data = _json.load(f)
        return {**_CONFIG_DEFAULTS, **data}
    except FileNotFoundError:
        return dict(_CONFIG_DEFAULTS)
    except _json.JSONDecodeError:
        print(f"[config] warning: corrupt config at {path}, using defaults")
        return dict(_CONFIG_DEFAULTS)


def save_zephyr_config(cfg: dict) -> None:
    """Persist config dict to ~/.zephyr/config.json."""
    path = _zephyr_config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        _json.dump(cfg, f, indent=2)
    os.replace(tmp, path)


class OllamaFetchThread(QThread):
    """Fetches available Ollama models in background. Emits list of name strings."""
    models_ready = Signal(list)  # list[str]

    _URL = __import__("config").OLLAMA_TAGS_URL

    def run(self):
        try:
            with urllib.request.urlopen(self._URL, timeout=3) as r:
                data = _json.loads(r.read())
            names = [m["name"] for m in data.get("models", [])]
        except Exception:
            names = []
        self.models_ready.emit(names)


class OllamaPullThread(QThread):
    """Runs `ollama pull <model>` and streams status lines to the GUI."""
    line_ready   = Signal(str)   # human-readable progress line
    pull_done    = Signal(bool)  # True = success, False = error

    def __init__(self, model: str, parent=None):
        super().__init__(parent)
        self._model = model

    def run(self):
        import subprocess, json as _j
        try:
            proc = subprocess.Popen(
                ["ollama", "pull", self._model],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for raw in proc.stdout:
                raw = raw.strip()
                if not raw:
                    continue
                # ollama pull emits JSON lines — parse for a clean display
                try:
                    obj = _j.loads(raw)
                    status = obj.get("status", "")
                    total  = obj.get("total", 0)
                    done   = obj.get("completed", 0)
                    if total and done:
                        pct = int(done / total * 100)
                        line = f"{status}  {pct}%"
                    else:
                        line = status
                except Exception:
                    line = raw
                if line:
                    self.line_ready.emit(line)
            proc.wait()
            self.pull_done.emit(proc.returncode == 0)
        except FileNotFoundError:
            self.line_ready.emit("ERROR: 'ollama' not found in PATH")
            self.pull_done.emit(False)
        except Exception as e:
            self.line_ready.emit(f"ERROR: {e}")
            self.pull_done.emit(False)


def _local_trained_models() -> set:
    """Return set of model names registered by a local BlackLoRA-N training run."""
    try:
        import json as _j
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "blackwell", "adapters", "registered.json"
        )
        with open(path, "r", encoding="utf-8") as f:
            return set(_j.load(f).get("models", []))
    except Exception:
        return set()


def _parse_quant(name: str) -> tuple:
    """Return (base_name, quant_label) from an Ollama model name string.

    Examples:
        'hermes3:8b'               -> ('hermes3:8b', '')
        'hermes3:8b-q4_0'          -> ('hermes3:8b', 'q4_0')
        'mistral:7b-instruct-q8_0' -> ('mistral:7b-instruct', 'q8_0')
    """
    m = re.search(r'-(q\d+_\d+|fp16|bf16)$', name)
    if m:
        return name[:m.start()], m.group(1)
    return name, ""


# ═══════════════════════════════════════════════════════════════
#  ModelDownloadCard — pull any Ollama/HuggingFace model
# ═══════════════════════════════════════════════════════════════
class ModelDownloadCard(QWidget):
    """Floating card for pulling a model from Ollama library or hf.co/."""

    download_complete = Signal()   # emitted on successful pull

    _BG     = QColor("#181818")
    _BORDER = QColor("#2e2e2e")
    _TEXT   = QColor("#c8c8c8")
    _DIM    = QColor("#505050")
    _TEAL   = QColor("#1a8272")
    _WIDTH  = 360

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFont(QFont("Consolas", 9))
        self._thread = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        # Header
        header = QLabel("ADD MODEL")
        header.setFont(QFont("Consolas", 9, QFont.Bold))
        header.setStyleSheet(f"color: {self._TEAL.name()}; background: transparent;")
        layout.addWidget(header)

        hint = QLabel("ollama:  llama3.2:3b\nhf.co:   hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF")
        hint.setFont(QFont("Consolas", 8))
        hint.setStyleSheet(f"color: {self._DIM.name()}; background: transparent;")
        layout.addWidget(hint)

        # Input row
        input_row = QHBoxLayout()
        input_row.setSpacing(6)
        self._input = QLineEdit()
        self._input.setPlaceholderText("model name or hf.co/user/repo")
        self._input.setFont(QFont("Consolas", 9))
        self._input.setStyleSheet(f"""
            QLineEdit {{
                background: #111111;
                color: {self._TEXT.name()};
                border: 1px solid {self._BORDER.name()};
                border-radius: 3px;
                padding: 3px 6px;
            }}
            QLineEdit:focus {{
                border: 1px solid {self._TEAL.name()};
            }}
        """)
        self._input.returnPressed.connect(self._start_pull)
        input_row.addWidget(self._input)

        self._pull_btn = QPushButton("PULL")
        self._pull_btn.setFont(QFont("Consolas", 9, QFont.Bold))
        self._pull_btn.setFixedWidth(52)
        self._pull_btn.setCursor(Qt.PointingHandCursor)
        self._pull_btn.setStyleSheet(f"""
            QPushButton {{
                background: {self._TEAL.name()};
                color: #111;
                border: none;
                border-radius: 3px;
                padding: 4px 8px;
            }}
            QPushButton:hover {{ background: #22a090; }}
            QPushButton:disabled {{ background: {self._BORDER.name()}; color: {self._DIM.name()}; }}
        """)
        self._pull_btn.clicked.connect(self._start_pull)
        input_row.addWidget(self._pull_btn)
        layout.addLayout(input_row)

        # Progress log
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas", 8))
        self._log.setFixedHeight(110)
        self._log.setStyleSheet(f"""
            QPlainTextEdit {{
                background: #0e0e0e;
                color: {self._TEXT.name()};
                border: 1px solid {self._BORDER.name()};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(self._log)

        self.setFixedWidth(self._WIDTH)
        self.adjustSize()

        QApplication.instance().installEventFilter(self)

    def show_at(self, pos: QPoint):
        self._input.clear()
        self._log.clear()
        self._pull_btn.setEnabled(True)
        self.move(pos)
        self.show()
        self.raise_()
        self._input.setFocus()
        self.activateWindow()

    def _start_pull(self):
        model = self._input.text().strip()
        if not model:
            return
        self._pull_btn.setEnabled(False)
        self._log.clear()
        self._log.appendPlainText(f"pulling {model}...")
        self._thread = OllamaPullThread(model)
        self._thread.line_ready.connect(self._on_line)
        self._thread.pull_done.connect(self._on_done)
        self._thread.start()

    def _on_line(self, line: str):
        # Replace last line if it's a progress update (ends with %)
        cursor = self._log.textCursor()
        cursor.movePosition(QTextCursor.End)
        text = self._log.toPlainText()
        last_nl = text.rfind("\n")
        last_line = text[last_nl + 1:] if last_nl >= 0 else text
        if last_line.endswith("%") and line.endswith("%"):
            # overwrite the last progress line in place
            cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
            cursor.insertText(line)
        else:
            self._log.appendPlainText(line)
        self._log.ensureCursorVisible()

    def _on_done(self, success: bool):
        self._pull_btn.setEnabled(True)
        if success:
            self._log.appendPlainText("✓ done — model ready")
            self.download_complete.emit()
            QTimer.singleShot(1800, self.hide)
        else:
            self._log.appendPlainText("✗ pull failed — check model name")

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.hide()

    def eventFilter(self, obj, event):
        if self.isVisible() and event.type() == QEvent.MouseButtonPress:
            try:
                gpos = event.globalPosition().toPoint()
            except AttributeError:
                gpos = event.globalPos()
            if not self.geometry().contains(gpos):
                self.hide()
        return False

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(0, 0, self.width(), self.height())
        p.setPen(Qt.NoPen)
        p.setBrush(self._BG)
        p.drawRoundedRect(rect, 4, 4)
        p.setPen(QPen(self._BORDER, 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 4, 4)
        p.end()


class ModelSwitcherCard(QWidget):
    """Floating model-selection card that appears above ThinkingBar cell 0."""

    model_selected = Signal(str)       # emits Ollama model name string
    turboquant_toggled = Signal(bool)  # emits new TurboQuant enabled state
    add_model_clicked = Signal()       # opens ModelDownloadCard

    _BG    = QColor("#181818")
    _BORDER= QColor("#2e2e2e")
    _TEXT  = QColor("#c8c8c8")
    _DIM   = QColor("#505050")
    _TEAL  = QColor("#1a8272")
    _HOVER = QColor("#222e2c")
    _WIDTH = 260
    _ROW_H = 26

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFont(QFont("Consolas", 9))

        self._models: list = []
        self._active_model: str = ""
        self._tq_enabled: bool = False
        self._hover_row: int = -1
        self._rows: list = []

        QApplication.instance().installEventFilter(self)

    def show_at(self, pos: QPoint, active_model: str, tq_enabled: bool):
        self._active_model = active_model
        self._tq_enabled = tq_enabled
        self._models = []
        self._rebuild()
        self.move(pos)
        self.show()
        self.raise_()
        self.setFocus()
        self.activateWindow()
        # Recreate thread each time — QThread.start() is no-op after thread finishes
        self._fetch_thread = OllamaFetchThread()
        self._fetch_thread.models_ready.connect(self._on_models_ready)
        self._fetch_thread.start()

    def _on_models_ready(self, names: list):
        self._models = sorted(names)
        self._rebuild()
        self.update()

    def _rebuild(self):
        """Rebuild self._rows and resize widget height."""
        trained = _local_trained_models()
        rows = []
        rows.append(("header", "SELECT MODEL", ""))

        if not self._models:
            rows.append(("loading", "fetching models...", ""))
        else:
            groups = {}
            for n in self._models:
                base, quant = _parse_quant(n)
                groups.setdefault(base, []).append((n, quant))
            for base, variants in groups.items():
                rows.append(("group", base, ""))
                for full_name, quant in variants:
                    badge = "★ local" if full_name in trained else quant
                    rows.append(("model", full_name, badge))

        rows.append(("sep", "", ""))
        tq_label = "KV BOOST: ON " if self._tq_enabled else "KV BOOST: OFF"
        rows.append(("turboquant", "TURBOQUANT", tq_label))
        if self._tq_enabled:
            rows.append(("tq_info", "~4.4x KV cache compression", ""))
        rows.append(("sep", "", ""))
        rows.append(("add", "+ ADD MODEL", ""))

        self._rows = rows
        h = len(rows) * self._ROW_H + 8
        self.setFixedSize(self._WIDTH, h)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(0, 0, self.width(), self.height())

        p.setPen(Qt.NoPen)
        p.setBrush(self._BG)
        p.drawRoundedRect(rect, 4, 4)

        p.setPen(QPen(self._BORDER, 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 4, 4)

        font_bold = QFont("Consolas", 9, QFont.Bold)
        font_norm = QFont("Consolas", 9)
        font_dim  = QFont("Consolas", 8)

        y = 4
        for i, (rtype, label, value) in enumerate(self._rows):
            ry = y
            rh = self._ROW_H

            if rtype == "model" and i == self._hover_row:
                p.setPen(Qt.NoPen)
                p.setBrush(self._HOVER)
                p.drawRect(1, ry, self._WIDTH - 2, rh)

            if rtype == "turboquant" and i == self._hover_row:
                p.setPen(Qt.NoPen)
                p.setBrush(self._HOVER)
                p.drawRect(1, ry, self._WIDTH - 2, rh)

            if rtype == "header":
                p.setFont(font_bold)
                p.setPen(QPen(self._TEAL))
                p.drawText(10, ry + 17, label)

            elif rtype == "loading":
                p.setFont(font_dim)
                p.setPen(QPen(self._DIM))
                p.drawText(10, ry + 17, label)

            elif rtype == "group":
                p.setFont(font_dim)
                p.setPen(QPen(self._DIM))
                p.drawText(10, ry + 17, label)

            elif rtype == "model":
                is_active = label == self._active_model
                if is_active:
                    p.setPen(QPen(self._TEAL, 2))
                    p.drawLine(2, ry + 4, 2, ry + rh - 4)
                p.setFont(font_norm)
                p.setPen(QPen(self._TEAL if is_active else self._TEXT))
                p.drawText(12, ry + 17, label)
                if value:
                    p.setFont(font_dim)
                    p.setPen(QPen(self._DIM))
                    p.drawText(self._WIDTH - 55, ry + 17, value)

            elif rtype == "sep":
                p.setPen(QPen(self._BORDER))
                p.drawLine(8, ry + rh // 2, self._WIDTH - 8, ry + rh // 2)

            elif rtype == "turboquant":
                p.setFont(font_bold)
                p.setPen(QPen(self._TEAL if self._tq_enabled else self._DIM))
                p.drawText(10, ry + 17, label)
                p.setFont(font_dim)
                p.setPen(QPen(self._TEAL if self._tq_enabled else self._DIM))
                p.drawText(self._WIDTH - 100, ry + 17, value)

            elif rtype == "tq_info":
                p.setFont(font_dim)
                p.setPen(QPen(self._DIM))
                p.drawText(10, ry + 14, label)

            elif rtype == "add":
                if i == self._hover_row:
                    p.setPen(Qt.NoPen)
                    p.setBrush(self._HOVER)
                    p.drawRect(1, ry, self._WIDTH - 2, rh)
                p.setFont(font_bold)
                p.setPen(QPen(self._TEAL))
                p.drawText(10, ry + 17, label)

            y += rh

        p.end()

    def mouseMoveEvent(self, e):
        idx = self._row_at(e.pos().y())
        if self._hover_row != idx:
            self._hover_row = idx
            self.update()
        super().mouseMoveEvent(e)

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        idx = self._row_at(e.pos().y())
        if idx < 0 or idx >= len(self._rows):
            return
        rtype, label, _ = self._rows[idx]
        if rtype == "model":
            self.model_selected.emit(label)
            self.hide()
        elif rtype == "turboquant":
            self._tq_enabled = not self._tq_enabled
            self._rebuild()
            self.update()
            self.turboquant_toggled.emit(self._tq_enabled)
        elif rtype == "add":
            self.hide()
            self.add_model_clicked.emit()

    def _row_at(self, y: int) -> int:
        row = (y - 4) // self._ROW_H
        return row if 0 <= row < len(self._rows) else -1

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.hide()

    def eventFilter(self, obj, event):
        if self.isVisible() and event.type() == QEvent.MouseButtonPress:
            # Use globalPosition().toPoint() for PySide6 Qt6 compatibility
            try:
                gpos = event.globalPosition().toPoint()
            except AttributeError:
                gpos = event.globalPos()
            if not self.geometry().contains(gpos):
                self.hide()
        return False

    def leaveEvent(self, e):
        self._hover_row = -1
        self.update()


# ═══════════════════════════════════════════════════════════════
#  OracleSwitcherCard — oracle model selector popup
#  Floats above ThinkingBar cell 2 (ORACLE)
# ═══════════════════════════════════════════════════════════════
class OracleSwitcherCard(QWidget):
    """Floating oracle-model selection card that appears above ThinkingBar cell 2."""

    oracle_selected = Signal(str)   # emits Ollama model name string
    add_model_clicked = Signal()    # opens ModelDownloadCard

    _BG    = QColor("#181818")
    _BORDER= QColor("#2e2e2e")
    _TEXT  = QColor("#c8c8c8")
    _DIM   = QColor("#505050")
    _TEAL  = QColor("#1a8272")
    _HOVER = QColor("#222e2c")
    _WIDTH = 260
    _ROW_H = 26

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFont(QFont("Consolas", 9))

        self._models: list = []
        self._active_model: str = ""
        self._hover_row: int = -1
        self._rows: list = []

        QApplication.instance().installEventFilter(self)

    def show_at(self, pos: QPoint, active_model: str):
        self._active_model = active_model
        self._models = []
        self._rebuild()
        self.move(pos)
        self.show()
        self.raise_()
        self.setFocus()
        self.activateWindow()
        self._fetch_thread = OllamaFetchThread()
        self._fetch_thread.models_ready.connect(self._on_models_ready)
        self._fetch_thread.start()

    def _on_models_ready(self, names: list):
        self._models = sorted(names)
        self._rebuild()
        self.update()

    def _rebuild(self):
        trained = _local_trained_models()
        rows = []
        rows.append(("header", "ORACLE MODEL", ""))

        if not self._models:
            rows.append(("loading", "fetching models...", ""))
        else:
            groups = {}
            for n in self._models:
                base, quant = _parse_quant(n)
                groups.setdefault(base, []).append((n, quant))
            for base, variants in groups.items():
                rows.append(("group", base, ""))
                for full_name, quant in variants:
                    badge = "★ local" if full_name in trained else quant
                    rows.append(("model", full_name, badge))

        rows.append(("sep", "", ""))
        rows.append(("add", "+ ADD MODEL", ""))

        self._rows = rows
        h = len(rows) * self._ROW_H + 8
        self.setFixedSize(self._WIDTH, h)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(0, 0, self.width(), self.height())

        p.setPen(Qt.NoPen)
        p.setBrush(self._BG)
        p.drawRoundedRect(rect, 4, 4)

        p.setPen(QPen(self._BORDER, 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 4, 4)

        font_bold = QFont("Consolas", 9, QFont.Bold)
        font_norm = QFont("Consolas", 9)
        font_dim  = QFont("Consolas", 8)

        y = 4
        for i, (rtype, label, value) in enumerate(self._rows):
            ry = y
            rh = self._ROW_H

            if rtype == "model" and i == self._hover_row:
                p.setPen(Qt.NoPen)
                p.setBrush(self._HOVER)
                p.drawRect(1, ry, self._WIDTH - 2, rh)

            if rtype == "header":
                p.setFont(font_bold)
                p.setPen(QPen(self._TEAL))
                p.drawText(10, ry + 17, label)

            elif rtype == "loading":
                p.setFont(font_dim)
                p.setPen(QPen(self._DIM))
                p.drawText(10, ry + 17, label)

            elif rtype == "group":
                p.setFont(font_dim)
                p.setPen(QPen(self._DIM))
                p.drawText(10, ry + 17, label)

            elif rtype == "model":
                is_active = label == self._active_model
                if is_active:
                    p.setPen(QPen(self._TEAL, 2))
                    p.drawLine(2, ry + 4, 2, ry + rh - 4)
                p.setFont(font_norm)
                p.setPen(QPen(self._TEAL if is_active else self._TEXT))
                p.drawText(12, ry + 17, label)
                if value:
                    p.setFont(font_dim)
                    p.setPen(QPen(self._DIM))
                    p.drawText(self._WIDTH - 55, ry + 17, value)

            elif rtype == "add":
                if i == self._hover_row:
                    p.setPen(Qt.NoPen)
                    p.setBrush(self._HOVER)
                    p.drawRect(1, ry, self._WIDTH - 2, rh)
                p.setFont(font_bold)
                p.setPen(QPen(self._TEAL))
                p.drawText(10, ry + 17, label)

            y += rh

        p.end()

    def mouseMoveEvent(self, e):
        idx = self._row_at(e.pos().y())
        if self._hover_row != idx:
            self._hover_row = idx
            self.update()
        super().mouseMoveEvent(e)

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        idx = self._row_at(e.pos().y())
        if idx < 0 or idx >= len(self._rows):
            return
        rtype, label, _ = self._rows[idx]
        if rtype == "model":
            self.oracle_selected.emit(label)
            self.hide()
        elif rtype == "add":
            self.hide()
            self.add_model_clicked.emit()

    def _row_at(self, y: int) -> int:
        row = (y - 4) // self._ROW_H
        return row if 0 <= row < len(self._rows) else -1

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.hide()

    def eventFilter(self, obj, event):
        if self.isVisible() and event.type() == QEvent.MouseButtonPress:
            try:
                gpos = event.globalPosition().toPoint()
            except AttributeError:
                gpos = event.globalPos()
            if not self.geometry().contains(gpos):
                self.hide()
        return False

    def leaveEvent(self, e):
        self._hover_row = -1
        self.update()


# ═══════════════════════════════════════════════════════════════
#  BlackwellConfigCard — Fix B / D / E operator controls popup
#  Floats above ThinkingBar cell 1 (/BW CONFIG)
# ═══════════════════════════════════════════════════════════════
class BlackwellConfigCard(QWidget):
    """
    Floating Blackwell config popup — Fix B / D / E controls.
    Colors match ModelSwitcherCard. Sliders with live value labels.
    Persists to blackwell/blackwell_config.json on Apply.
    """

    config_applied = Signal(dict)

    # ── Colors — match ModelSwitcherCard exactly ─────────────────
    _W      = 380
    _BG     = "#181818"
    _BORDER = "#2e2e2e"
    _TEAL   = "#1a8272"
    _DIM    = "#505050"
    _TEXT   = "#c8c8c8"

    DEFAULTS: dict = {
        "gap_threshold":      0.20,
        "window_size":        100,
        "min_samples":        10,
        "oracle_temperature": 0.80,
        "judge_temperature":  0.00,
        "assertion_window":   80,
        "hedge_window":       60,
        "enforce_semantic":   True,
    }

    CONFIG_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "blackwell", "blackwell_config.json",
    )

    # (key, ftype, hint, sl_min, sl_max, scale)
    # scale: slider integer = real_value * scale
    # ftype "float" → display with 2 decimals; "int" → display as int; "bool" → checkbox
    _FIELDS = [
        # ── Fix B ────────────────────────────────────────────────
        ("gap_threshold",      "float", "abort training if oracle/rule gap exceeds this", 5,   50,  100),
        ("window_size",        "int",   "rolling sample window size",                     10,  500, 1),
        ("min_samples",        "int",   "samples before drift gate activates",            1,   50,  1),
        # ── Fix D ────────────────────────────────────────────────
        ("oracle_temperature", "float", "oracle idea generation creativity",              0,   200, 100),
        ("judge_temperature",  "float", "evaluator strictness  (0 = fully deterministic)",0,  100, 100),
        # ── Fix E ────────────────────────────────────────────────
        ("assertion_window",   "int",   "chars after hedge to find verb/noun",            20,  200, 1),
        ("hedge_window",       "int",   "chars after hedge to find quantity",             10,  150, 1),
        ("enforce_semantic",   "bool",  "require real assertion context in calibration",  0,   1,   1),
    ]

    _GROUPS = [
        ("DRIFT MONITOR", "Fix B", ["gap_threshold", "window_size", "min_samples"]),
        ("DECORRELATION",  "Fix D", ["oracle_temperature", "judge_temperature"]),
        ("ANTI-GAMING",    "Fix E", ["assertion_window", "hedge_window", "enforce_semantic"]),
    ]

    def __init__(self, parent=None):
        super().__init__(
            parent,
            Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedWidth(self._W)
        self._config = dict(self.DEFAULTS)
        self._load_config()
        # _inputs: key → QSlider or QCheckBox
        self._inputs:      dict = {}
        # _val_labels: key → QLabel showing current value
        self._val_labels:  dict = {}
        # _field_meta: key → (ftype, scale)
        self._field_meta:  dict = {f[0]: (f[1], f[5]) for f in self._FIELDS}
        self._stat_labels: dict = {}
        self._build_ui()
        self.adjustSize()
        QApplication.instance().installEventFilter(self)

    # ── Config I/O ───────────────────────────────────────────────

    def _load_config(self):
        try:
            with open(self.CONFIG_PATH, encoding="utf-8") as f:
                saved = _json.load(f)
            for k, v in saved.items():
                if k in self._config:
                    self._config[k] = v
        except (FileNotFoundError, _json.JSONDecodeError):
            pass

    def _save_config(self):
        try:
            os.makedirs(os.path.dirname(self.CONFIG_PATH), exist_ok=True)
            with open(self.CONFIG_PATH, "w", encoding="utf-8") as f:
                _json.dump(self._config, f, indent=2)
        except Exception:
            pass

    def _slider_to_value(self, key: str, sl_int: int):
        ftype, scale = self._field_meta[key]
        if ftype == "float":
            return sl_int / scale
        return sl_int   # int or bool

    def _value_to_slider(self, key: str, value) -> int:
        ftype, scale = self._field_meta[key]
        if ftype == "float":
            return int(round(float(value) * scale))
        return int(value)

    def _fmt_value(self, key: str, value) -> str:
        ftype, _ = self._field_meta[key]
        if ftype == "float":
            return f"{float(value):.2f}"
        return str(int(value))

    def _read_inputs(self) -> dict:
        out = {}
        for key, widget in self._inputs.items():
            if isinstance(widget, QCheckBox):
                out[key] = widget.isChecked()
            else:
                out[key] = self._slider_to_value(key, widget.value())
        return out

    # ── UI ───────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(1, 1, 1, 1)
        outer.setSpacing(0)

        inner = QWidget()
        inner.setObjectName("bwcfg_inner")
        inner.setStyleSheet(f"""
            #bwcfg_inner {{
                background: {self._BG};
                border: 1px solid {self._BORDER};
                border-radius: 4px;
            }}
            QSlider::groove:horizontal {{
                height: 3px;
                background: {self._BORDER};
                border-radius: 1px;
            }}
            QSlider::sub-page:horizontal {{
                background: {self._TEAL};
                border-radius: 1px;
            }}
            QSlider::handle:horizontal {{
                width: 11px; height: 11px;
                margin: -4px 0;
                background: {self._TEAL};
                border-radius: 5px;
            }}
            QSlider::handle:horizontal:hover {{
                background: #22a090;
            }}
            QCheckBox {{
                color: {self._TEXT};
                font-family: Consolas;
                font-size: 9pt;
            }}
            QCheckBox::indicator {{
                width: 13px; height: 13px;
                border: 1px solid {self._BORDER};
                border-radius: 2px;
                background: #111;
            }}
            QCheckBox::indicator:checked {{
                background: {self._TEAL};
                border-color: {self._TEAL};
            }}
        """)

        vbox = QVBoxLayout(inner)
        vbox.setContentsMargins(10, 9, 10, 9)
        vbox.setSpacing(0)

        # ── Header ───────────────────────────────────────────────
        hdr = QLabel("BLACKWELL CONFIG")
        hdr.setStyleSheet(
            f"color:{self._TEAL};font-family:Consolas;font-size:9pt;"
            "font-weight:bold;letter-spacing:3px;"
        )
        vbox.addWidget(hdr)
        sub = QLabel("Fix B · D · E  —  operator controls")
        sub.setStyleSheet(f"color:{self._DIM};font-family:Consolas;font-size:8pt;")
        vbox.addWidget(sub)
        vbox.addSpacing(7)

        # ── Stat pills ───────────────────────────────────────────
        _stat_defs = [
            ("GAP",        "gap_threshold",      lambda c: f"{c['gap_threshold']:.2f}"),
            ("ORC TEMP",   "oracle_temperature",  lambda c: f"{c['oracle_temperature']:.2f}"),
            ("JUDGE TEMP", "judge_temperature",   lambda c: f"{c['judge_temperature']:.2f}"),
            ("GATE",       "window_size",         lambda c: f"{c['min_samples']}/{c['window_size']}"),
        ]
        pills_row = QHBoxLayout()
        pills_row.setSpacing(4)
        for pill_name, stat_key, fmt_fn in _stat_defs:
            pill = QFrame()
            pill.setStyleSheet(
                f"background:#1f1f1f;border:1px solid {self._BORDER};border-radius:3px;"
            )
            pv = QVBoxLayout(pill)
            pv.setContentsMargins(6, 3, 6, 3)
            pv.setSpacing(1)
            ln = QLabel(pill_name)
            ln.setStyleSheet(
                f"color:{self._DIM};font-family:Consolas;font-size:7pt;letter-spacing:1px;"
            )
            lv = QLabel(fmt_fn(self._config))
            lv.setStyleSheet(
                f"color:{self._TEAL};font-family:Consolas;font-size:10pt;font-weight:bold;"
            )
            pv.addWidget(ln)
            pv.addWidget(lv)
            pills_row.addWidget(pill)
            self._stat_labels[stat_key] = (lv, fmt_fn)

        pills_wrap = QWidget()
        pills_wrap.setLayout(pills_row)
        vbox.addWidget(pills_wrap)
        vbox.addSpacing(8)

        # ── Build field lookup ────────────────────────────────────
        field_lookup = {f[0]: f for f in self._FIELDS}

        # ── Groups ───────────────────────────────────────────────
        for grp_name, fix_lbl, keys in self._GROUPS:
            # Section header row
            sec = QWidget()
            sec.setStyleSheet(f"background:#1f1f1f;border-radius:2px;")
            sh = QHBoxLayout(sec)
            sh.setContentsMargins(6, 2, 6, 2)
            g_t = QLabel(grp_name)
            g_t.setStyleSheet(
                f"color:{self._TEAL};font-family:Consolas;"
                "font-size:8pt;font-weight:bold;letter-spacing:2px;"
            )
            g_f = QLabel(fix_lbl)
            g_f.setStyleSheet(
                f"color:{self._DIM};font-family:Consolas;font-size:8pt;"
            )
            sh.addWidget(g_t)
            sh.addStretch()
            sh.addWidget(g_f)
            vbox.addWidget(sec)
            vbox.addSpacing(3)

            for key in keys:
                _, ftype, hint, sl_min, sl_max, scale = field_lookup[key]
                cur_val = self._config.get(key, self.DEFAULTS[key])

                # ── Key name + value label row ────────────────────
                top_row = QWidget()
                top_h = QHBoxLayout(top_row)
                top_h.setContentsMargins(2, 0, 2, 0)
                top_h.setSpacing(4)

                lbl_key = QLabel(key)
                lbl_key.setStyleSheet(
                    f"color:{self._TEXT};font-family:Consolas;font-size:8pt;"
                )
                lbl_key.setToolTip(hint)
                top_h.addWidget(lbl_key, 1)

                if ftype == "bool":
                    # Boolean: checkbox on right side of key row, no slider row
                    cb = QCheckBox()
                    cb.setChecked(bool(cur_val))
                    cb.stateChanged.connect(self._update_stats)
                    top_h.addWidget(cb)
                    self._inputs[key] = cb
                    vbox.addWidget(top_row)
                else:
                    # Numeric: show formatted value on right
                    lbl_val = QLabel(self._fmt_value(key, cur_val))
                    lbl_val.setFixedWidth(34)
                    lbl_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    lbl_val.setStyleSheet(
                        f"color:{self._TEAL};font-family:Consolas;"
                        "font-size:8pt;font-weight:bold;"
                    )
                    top_h.addWidget(lbl_val)
                    self._val_labels[key] = lbl_val
                    vbox.addWidget(top_row)

                    # Slider row
                    sl_row = QWidget()
                    sl_h = QHBoxLayout(sl_row)
                    sl_h.setContentsMargins(2, 0, 2, 4)
                    sl_h.setSpacing(0)

                    slider = QSlider(Qt.Horizontal)
                    slider.setMinimum(sl_min)
                    slider.setMaximum(sl_max)
                    slider.setValue(self._value_to_slider(key, cur_val))
                    slider.setFixedHeight(18)

                    def _make_cb(k, lv):
                        def on_change(v):
                            real = self._slider_to_value(k, v)
                            lv.setText(self._fmt_value(k, real))
                            self._update_stats()
                        return on_change

                    slider.valueChanged.connect(_make_cb(key, lbl_val))
                    sl_h.addWidget(slider)
                    self._inputs[key] = slider
                    vbox.addWidget(sl_row)

                # Hint label (dim, small)
                hint_lbl = QLabel(hint)
                hint_lbl.setStyleSheet(
                    f"color:{self._DIM};font-family:Consolas;font-size:7pt;"
                )
                hint_lbl.setContentsMargins(2, 0, 2, 3)
                vbox.addWidget(hint_lbl)

            vbox.addSpacing(5)

        # ── Separator ────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"border-top:1px solid {self._BORDER};")
        sep.setFixedHeight(1)
        vbox.addWidget(sep)
        vbox.addSpacing(6)

        # ── Buttons ──────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        apply_btn = QPushButton("APPLY")
        apply_btn.setStyleSheet(f"""
            QPushButton {{
                background: {self._TEAL};
                color: {self._TEXT};
                border: none;
                border-radius: 3px;
                font-family: Consolas; font-size: 8pt;
                letter-spacing: 2px; padding: 4px 14px;
            }}
            QPushButton:hover {{ background: #22a090; }}
            QPushButton:pressed {{ background: #155f56; }}
        """)
        apply_btn.setCursor(Qt.PointingHandCursor)
        apply_btn.clicked.connect(self._apply)

        reset_btn = QPushButton("RESET")
        reset_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {self._DIM};
                border: 1px solid {self._BORDER};
                border-radius: 3px;
                font-family: Consolas; font-size: 8pt;
                letter-spacing: 2px; padding: 4px 14px;
            }}
            QPushButton:hover {{
                color: {self._TEXT};
                border-color: #505050;
            }}
        """)
        reset_btn.setCursor(Qt.PointingHandCursor)
        reset_btn.clicked.connect(self._reset)

        btn_row.addStretch()
        btn_row.addWidget(reset_btn)
        btn_row.addWidget(apply_btn)
        vbox.addLayout(btn_row)

        outer.addWidget(inner)

    # ── Slots ────────────────────────────────────────────────────

    def _update_stats(self, *_):
        current = self._read_inputs()
        for key, (lbl, fmt_fn) in self._stat_labels.items():
            try:
                lbl.setText(fmt_fn(current))
            except Exception:
                pass

    def _apply(self):
        self._config = self._read_inputs()
        self._save_config()
        self.config_applied.emit(dict(self._config))
        self.hide()

    def _reset(self):
        self._config = dict(self.DEFAULTS)
        for key, widget in self._inputs.items():
            val = self.DEFAULTS[key]
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(val))
            else:
                widget.setValue(self._value_to_slider(key, val))
        self._update_stats()

    def show_at(self, pos: QPoint):
        self._load_config()
        for key, widget in self._inputs.items():
            val = self._config.get(key, self.DEFAULTS[key])
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(val))
            else:
                widget.setValue(self._value_to_slider(key, val))
                if key in self._val_labels:
                    self._val_labels[key].setText(self._fmt_value(key, val))
        self._update_stats()
        self.adjustSize()
        self.move(pos)
        self.show()
        self.raise_()
        self.activateWindow()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.hide()

    def eventFilter(self, obj, event):
        if self.isVisible() and event.type() == QEvent.MouseButtonPress:
            try:
                gpos = event.globalPosition().toPoint()
            except AttributeError:
                gpos = event.globalPos()
            if not self.geometry().contains(gpos):
                self.hide()
        return False


AGENT_PATH = r"C:\Users\gamer23\Desktop\hermes-agent\agent.py"
MODEL_NAME  = "hermes3:8b"

# ─── Colours ──────────────────────────────────────────────────
C_BG         = QColor("#090c10")
C_SURFACE    = QColor("#0d1117")
C_TEAL       = QColor(128, 221, 202, 235)
C_TEAL_DIM   = QColor(128, 221, 202, 133)
C_MUTED      = QColor(170, 182, 194, 143)
C_CYAN       = QColor(122, 184, 216, 220)
C_AMBER      = QColor(212, 160, 80,  220)
C_GREEN      = QColor(102, 196, 122, 220)
C_RED        = QColor(210, 90,  90,  220)
C_WHITE_DIM  = QColor(215, 223, 230, 185)


# ═══════════════════════════════════════════════════════════════
#  Dragon art — rendered directly into ConsoleWidget on startup
# ═══════════════════════════════════════════════════════════════

_DRAGON_ART = [
" ----..                                                                                                                                  .-----.  ",
"  .+++++++##-                                                                                                                     .-##+++++++-  ",
"    .+##+++++-+##+-                                                                                                          .+##++--++####-    ",
"      +#####++++-++###+.                                                                                                 -###++--+++######.     ",
"       .########+++-+++###+.                                                                                         -###++---+++#######+       ",
"         -#########+++-++++###-                                                                                   -###+-+--+++#########-        ",
"           ###########++-+--+-###-                                                                             -###------+++##########          ",
"            -############++-----+###-                                  .+#-                                 .###+------++###########+           ",
"             .##############+----+-####                                {E}--                              -###+-----+++############-            ",
"               ###############+------+###.                            #-   ++                           ####-------++#############.             ",
"                -###############++-----+###-                         ++    .#-                        ####-------++##############.              ",
"                 .###############+-------+###.                      #-+--..#-++                     +###------+-+###############.               ",
"                  .################+-------+###-                   #+-+#-+-#--+-                  +###--------+################.                ",
"                    ################+-----.--##+--            .    +-+-++-#--+--                +--##--.-----+################.                 ",
"                    .################+---------####     -  .##+ -++  . +...---   +-  .##.     .####---------+################-                  ",
"                   .##################+-------.+++##-  #.#+#+###++     +- .--     -########-###++++.-------+###################                 ",
"                  ##################+------------   .###.+###+---.     -#--+-     -++--+###+##+  -------------+#################-               ",
"                -##############+---++++--+-----.   #######-#.-++#-     +##+#-      #--..---+#++#- .+---------------+##############.             ",
"              .##########++++++++++++++++----##   ++++###+##..-+##+    #####-    +##+-..-#####++##.##----+-++++++++++++-++##########.           ",
"             #######++++++#########+++++-+--+#+   ..-####+#+...-+##-  -#####    +##+..---#####+-.. +#---+++++++##########+++++++######          ",
"           +###+++++##################+++---+##.-+-###++---   .--###  +####+   .##+...    #-+###+-.##--+++++++#################+++++###+        ",
"         .#++++++######################++++++#+++####----#     --### -#####-   -##+.+     #+.-+###++###+++++#######################++++++.      ",
"               .-++##################+++++++-++####+-.. .     .+####.+####+.   -###-+      -- .-+####-+++++++####################++-            ",
"                     .-+############+++++---+####-+--.        -####..####+-    .###++         --++####+-++++#+#############+-.                  ",
"                          .#######+#+++---#####+--+-.        -+###+ ---....     +###+-         .--++####++-+++++########.                       ",
"                          -#####----+--++####--.-+-.         +####. -...--.     -####+          .---++####+--++-###++###+                       ",
"                          ###-+++++--++####+---+-++-        +####+.--.. .. .     +###+-         +----+-+###++++-++#######                       ",
"                         .+++##++-+#-####+-.---+--+#       .+####-         ..   .#+-#++        .#+-+++---#####++..#######-                      ",
"           .             +++++---+#####++--+--++#--#-      -+####-     . .  -.  .####++        ##-+++++---+#####+-.-######                      ",
"                       .-------++####++-.##+----##-+#.     .-####+  .-..  ..  ..+####--       ##-++++++++#--+#####+--.-++--..                   ",
"         ....      ...-.----+++####+----++#----.-##-##     .-++###- ..--#.  ..-+####---      ##++++#++++####--#####+++++-.-.   .                ",
"    .+++-+-.+     +...----++######+---++##-+++--.-+#####--##---####. -#####. .####+---##   -###+######+#+####++++####+++----.---         -      ",
"   --    ...+-..-#...--+-+#######+---#+-#++####+----+###++##-.--#+##############------#######++########++-####++-++#####------..-       . +     ",
"  .       ..-.-.-------+######+-+----+-+++#######+-.--++####--.--++###########+--..-.###++++#############+-####+---+#####+-+---..-..  ----..    ",
" .. .---------------++######+-+--.. ++-++#########+++----+###...--+--######++---..-.-###+++###############+##-....+-+######+-+---......---..--  ",
".----------+#+-+-+########--+++.. ..-+++##########++++++++###-....-+---+#--+-.. ---++###+##################+##  ...+---######+#--+---....   ... ",
"-----------+-#+########+#--++-..   -#++#####+.        -++++++-..-...---#++.. ..---.-+##++##-        .+#####+##     .+----+######----+-.......--.  ",
"----..--+##-+##########++#-+..     +++++-                -##-++..-..  -#++.  .---..++-#++               -###+#. ..  .-+---+######+-.-++-..-----.  ",
"-..--+++-++########++++-----.      +-.                    #+--#+..-...+#+#-...--..#++-##                   .++-      ..----++######+-.--#-..----  ",
"---#+-++#############-..-.-..                     - #.  #####++#+..--++#-++------##+######   #                        . .+--++#########+++++----  ",
"+##-+##########+##+#-.... ...                    #+##  ######--##-.++--+---++#-.##-+-######  ##.#                      . .-.+++++#########++++-.  ",
"##-###########+##+-...... .-                    ##++########+###+-.--+-------++-+.  .++#######++##                      - -----+############+++-  ",
"#+###########+#+---... .  -.                    +########+-+   -#+-.-+-..--+#-+-#-    +-##########-                     .- .----++############+-  ",
"##########+##+++--.... -++                      .########--   -##-.--------+++-.+##.  #+-#############-                  ##  .--####+##########+  ",
"###########++-++-..---+-                        +#######--.  +###++-----.-+++--+#######++-########++####+                -# . .-..+#############  ",
"###########+--#-...                        ---########+-++   ####++------+----+#+####+- ++--##############.              ---. ..---+--+#########  ",
"#########+------+                .#+#.    #####---+##--+.   +##+-++#++----+++-#+++###-   .#--###+-+########- .+-          .-..-.+-----++########  ",
"##+####+------+..               -#############+####+.--.    -#--+---++--+######--+--+-     ++.-##################           .-.-+----++-++######  ",
"####++++#-----..-            +#################-#####--     -   .--- ++-++#+###-+#-  .     .--####-+##############-               -.---+--+#####  ",
"###++-------.----- .        +#######+--######--..-###+          ##-    --+####+---.         -###...--######--########              ..------++###  ",
"+++--------  --..    .     #######+-+-.+###+++-.-..-+##-    .####+-             +###+.    .###+-.-..-++####.-+-#######.             .##------+++  ",
"+-.-.-----  .--..         #######-+....+##++...--....####+.                       - -  .-####+...----.-+##+...-++######.            .++.-++-----  ",
"--.----.  .......        -##++##-    .--#+--------.....####                            ####-+..---+++---+#+---  -+##+###             .-- -++----  ",
".-------..   ..-         +#+..#-       -+----. .-+-----++###-                        -###-+++-+++-  .-++++#+-    .+#.++#.              .....++--  ",
"-----.                   --   -         -              .+###+                        ++##-+         +####+--      .+  .+.               .....---  ",
"-..  ..                  .                             -##++.                         ++##-       +#####+--        .   .                     .+-  ",
".                                                     .++-                              .++-   .######+-+                            ...       -  ",
".              .                                             -#++---+###+                 .-#######+--+                                           ",
".                                                         -#-          --#####++--++###########+--++.                                            ",
"                                                         +.                -+++########+++---+++.                                                 ",
"                                                       -.                       .-------..                                                        ",
"                                                      .                                                                                           ",
]

_DRAGON_COLOUR_MAP = {
    "#": "#1a6a3a",
    "+": "#2aaa8a",
    "-": "#4dcdb4",
    ".": "#2a3a4a",
}

def _dragon_render_line(raw_line):
    """Convert one raw dragon art line to an HTML string."""
    parts = []
    i = 0
    while i < len(raw_line):
        if raw_line[i:i+3] == "{E}":
            parts.append('<span style="color:#66c47a;font-weight:bold;">@</span>')
            i += 3
        else:
            ch = raw_line[i]
            safe_ch = html.escape(ch, quote=False)
            colour = _DRAGON_COLOUR_MAP.get(ch)
            if colour:
                parts.append(f'<span style="color:{colour};">{safe_ch}</span>')
            else:
                parts.append(safe_ch)
            i += 1
    inner = "".join(parts)
    return (
        f'<span style="white-space:pre;font-family:Consolas,monospace;font-size:8pt;">'
        f'{inner}</span>'
    )


def _dragon_splash_into_console(console_widget):
    """Render the dragon art directly into ConsoleWidget (no separate window)."""
    from PySide6.QtWidgets import QApplication as _QApp
    for raw in _DRAGON_ART:
        console_widget.appendHtml(_dragon_render_line(raw))
    console_widget.appendHtml(
        '<p style="font-family:Consolas,monospace;font-size:13pt;'
        'color:#4dcdb4;font-weight:bold;letter-spacing:8px;'
        'text-align:center;margin:0;">Z  E  P  H  Y  R</p>'
    )
    console_widget.appendHtml(
        '<p style="font-family:Consolas,monospace;font-size:8.5pt;'
        'color:#2a7a5a;text-align:center;margin:0;">'
        'Prycat Research  \xb7  local intelligence  \xb7  BlackLoRA-N core</p>'
    )
    _QApp.processEvents()


# ═══════════════════════════════════════════════════════════════
#  ZephyrProcess — subprocess thread
# ═══════════════════════════════════════════════════════════════
class ZephyrProcess(QThread):
    output_signal   = Signal(str)
    finished_signal = Signal()
    stream_started  = Signal()   # emitted when <<ZS>> arrives
    stream_ended    = Signal()   # emitted when <<ZE>> arrives
    token_gap       = Signal(float)  # wired in MainWindow.__init__

    _SENTINEL = object()   # signals the input queue to stop

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc      = None      # type: Optional[subprocess.Popen]
        self._lock      = threading.Lock()
        self._input_q   = queue.Queue()   # GUI → worker thread

    def run(self):
        try:
            import os as _os
            _env = _os.environ.copy()
            _env["PYTHONUNBUFFERED"] = "1"
            _env["PYTHONUTF8"] = "1"
            _env["PYTHONIOENCODING"] = "utf-8"
            proc = subprocess.Popen(
                [sys.executable, "-u", AGENT_PATH],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
                env=_env,
            )
            with self._lock:
                self._proc = proc

            # Drain input queue in a separate writer thread so
            # stdout reading never blocks on stdin writes.
            def _stdin_writer():
                while True:
                    item = self._input_q.get()
                    if item is self._SENTINEL:
                        break
                    try:
                        proc.stdin.write(item + "\n")
                        proc.stdin.flush()
                    except OSError:
                        break
                try:
                    proc.stdin.close()
                except OSError:
                    pass

            writer = threading.Thread(target=_stdin_writer, daemon=True)
            writer.start()

            last_tok_t = None

            for line in proc.stdout:
                stripped = line.rstrip("\n")
                if stripped == "<<ZS>>":
                    last_tok_t = time.monotonic()
                    self.stream_started.emit()
                elif stripped == "<<ZE>>":
                    last_tok_t = None
                    self.stream_ended.emit()
                elif stripped.startswith("\x01"):
                    now = time.monotonic()
                    if last_tok_t is not None:
                        gap_ms = (now - last_tok_t) * 1000.0
                        # First emission after <<ZS>> is time-to-first-token (TTFT);
                        # subsequent emissions are true inter-token gaps.
                        self.token_gap.emit(gap_ms)
                    last_tok_t = now
                # Non-\x01 lines (e.g. plain text, <<ZS>>, <<ZE>>) are intentionally
                # skipped for gap tracking — we only timestamp \x01 token lines.
                self.output_signal.emit(stripped)

            self._input_q.put(self._SENTINEL)
            writer.join(timeout=2)
            proc.wait()

        except Exception as exc:
            self.output_signal.emit(f"[Zephyr GUI] Failed to start agent: {exc}")
        finally:
            with self._lock:
                self._proc = None
            self.finished_signal.emit()

    def send_input(self, text: str):
        """Thread-safe: called from GUI thread, queued to worker."""
        self._input_q.put(text)

    def stop(self):
        with self._lock:
            proc = self._proc
        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except OSError:
                pass
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except OSError:
                    pass
        self._input_q.put(self._SENTINEL)


# ═══════════════════════════════════════════════════════════════
#  ZephyrButton — Monolith Signal custom button
# ═══════════════════════════════════════════════════════════════
class ZephyrButton(QPushButton):
    """
    Custom button with:
    - Scanline grain texture
    - BorderWake breathing pulse
    - Hover sweep (teal line across top+bottom border)
    - Mouse-tracking radial glow blob
    - State dot (idle/running/success/error)
    """

    SWEEP_MS   = 700      # hover sweep duration in ms
    WAKE_MS    = 5500     # borderWake pulse period in ms
    TICK_MS    = 16       # ~60fps timer interval

    def __init__(
        self,
        label: str,
        command: str,
        tooltip: str,
        fire_immediately: bool = True,
        parent=None
    ):
        super().__init__(parent)
        self.label            = label
        self.command          = command
        self.fire_immediately = fire_immediately
        self._state           = "idle"

        # Animation state
        self._sweep_t       = 0.0
        self._sweeping      = False
        self._wake_t        = 0
        self._mouse_pos     = QPointF(-1, -1)
        self._mouse_inside  = False
        self._state_tint_a  = 0.0

        # Timer
        self._timer = QTimer(self)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        # Widget setup
        self.setMouseTracking(True)
        self.setFixedHeight(52)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tooltip)
        self.setFlat(True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setAttribute(Qt.WidgetAttribute.WA_AlwaysShowToolTips, True)
        self.setToolTipDuration(4000)

        # Cache font to avoid allocation every paintEvent frame
        self._font = QFont("Consolas", 9)
        self._font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.5)
        self._font.setBold(True)
        self._badge_text: str = ""

    def set_state(self, state: str):
        """state: idle | running | success | error"""
        self._state = state
        if state in ("success", "error"):
            self._state_tint_a = 1.0
        else:
            self._state_tint_a = 0.0

    def _tick(self):
        dt = self.TICK_MS
        self._wake_t = (self._wake_t + dt) % self.WAKE_MS
        if self._sweeping and self._sweep_t < 1.0:
            self._sweep_t = min(1.0, self._sweep_t + dt / self.SWEEP_MS)
        if self._state_tint_a > 0:
            self._state_tint_a = max(0.0, self._state_tint_a - dt / 1200.0)
        self.update()

    def enterEvent(self, event):
        self._mouse_inside = True
        self._sweeping     = True
        self._sweep_t      = 0.0
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._mouse_inside = False
        self._sweeping     = False
        self._sweep_t      = 0.0
        self._mouse_pos    = QPointF(-1, -1)
        super().leaveEvent(event)

    def mouseMoveEvent(self, event):
        self._mouse_pos = event.position()
        super().mouseMoveEvent(event)

    def paintEvent(self, event):
        # NOTE: super().paintEvent() intentionally NOT called — this widget is
        # fully custom-painted. Calling super() would draw native QPushButton
        # chrome (border, focus ring) on top of all custom layers.
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        rect = QRectF(0, 0, w, h)

        # 1. Base fill
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(C_SURFACE))
        p.drawRoundedRect(rect, 4, 4)

        # 2. Scanline grain
        p.setPen(QPen(QColor(255, 255, 255, 5), 1))
        y = 0
        while y < h:
            p.drawLine(0, y, w, y)
            y += 3

        # 3. Mouse glow blob
        if self._mouse_inside and self._mouse_pos.x() >= 0:
            glow = QRadialGradient(self._mouse_pos, 60)
            glow.setColorAt(0, QColor(77, 194, 179, 46))
            glow.setColorAt(1, QColor(77, 194, 179, 0))
            p.setBrush(QBrush(glow))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(rect, 4, 4)

        # 4. Inner bevel
        p.setPen(QPen(QColor(255, 255, 255, 8), 1))
        p.drawLine(1, 1, w - 1, 1)
        p.drawLine(1, 1, 1, h - 1)
        p.setPen(QPen(QColor(0, 0, 0, 100), 1))
        p.drawLine(1, h - 1, w - 1, h - 1)
        p.drawLine(w - 1, 1, w - 1, h - 1)

        # 5. Resting border
        p.setPen(QPen(QColor(255, 255, 255, 20), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 4, 4)

        # 6. BorderWake pulse
        wake_phase = math.sin(2 * math.pi * self._wake_t / self.WAKE_MS)
        wake_alpha = int(2 + 6 * (wake_phase * 0.5 + 0.5))
        p.setPen(QPen(QColor(77, 194, 179, wake_alpha), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 4, 4)

        # 7. Hover sweep
        if self._sweep_t > 0:
            ease = 1.0 - (1.0 - self._sweep_t) ** 3
            sx = ease * w
            shard_alpha = int(180 * (1.0 - self._sweep_t))
            p.setPen(QPen(QColor(128, 221, 202, shard_alpha), 1))
            p.drawLine(int(sx) - 24, 0, int(sx), 0)
            p.drawLine(int(sx) - 24, h - 1, int(sx), h - 1)

        # 8. State tint
        if self._state_tint_a > 0 and self._state in ("success", "error"):
            tint_color = C_GREEN if self._state == "success" else C_RED
            tint = QColor(tint_color)
            tint.setAlpha(int(40 * self._state_tint_a))
            p.setBrush(QBrush(tint))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(rect, 4, 4)

        # 9. Label text
        p.setFont(self._font)
        text_color = QColor(C_TEAL) if self._mouse_inside else QColor(C_TEAL_DIM)
        p.setPen(QPen(text_color))
        text_rect = QRectF(12, 0, w - 30, h)
        p.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.label)

        # 10. State dot
        dot_x = w - 14
        dot_y = h // 2
        dot_r = 4
        if self._state == "idle":
            dot_color = QColor(255, 255, 255, 40)
        elif self._state == "running":
            pulse = math.sin(2 * math.pi * self._wake_t / 800) * 0.5 + 0.5
            dot_color = QColor(C_AMBER)
            dot_color.setAlpha(int(120 + 100 * pulse))
        elif self._state == "success":
            dot_color = C_GREEN
        else:
            dot_color = C_RED
        p.setBrush(QBrush(dot_color))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(dot_x - dot_r, dot_y - dot_r, dot_r * 2, dot_r * 2)

        # 11. Badge text (pair count)
        badge = self._badge_text
        if badge:
            badge_font = QFont("Consolas", 7)
            p.setFont(badge_font)
            p.setPen(QColor("#4dcdb4"))
            badge_rect = self.rect().adjusted(0, 0, -6, 0)
            p.drawText(badge_rect,
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       badge)

    def set_badge(self, text: str) -> None:
        """Set a small count badge shown at the right edge of the button."""
        self._badge_text = text
        self.update()  # trigger repaint


# ═══════════════════════════════════════════════════════════════
#  ConsoleWidget
# ═══════════════════════════════════════════════════════════════
class ConsoleWidget(QPlainTextEdit):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        # WidgetWidth: Qt handles word-wrap at the widget boundary automatically.
        # This prevents horizontal scroll on long responses without any manual
        # column tracking that could break mid-word.
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._auto_scroll   = True
        self._streaming     = False
        self._stream_cursor = None   # persistent cursor kept inside Zephyr paragraph
        # Pre-built char format used while streaming
        self._stream_fmt  = QTextCharFormat()
        self._stream_fmt.setForeground(QColor("#80ddca"))
        self._stream_fmt.setFontFamily("Consolas")

        font = QFont("Consolas", 10)
        self.setFont(font)

        # Small inner breathing room — outer centering is handled by the layout
        self.setViewportMargins(16, 8, 16, 8)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Base, C_BG)
        palette.setColor(QPalette.ColorRole.Text, C_TEAL)
        self.setPalette(palette)

        self.setStyleSheet("""
            QPlainTextEdit {
                background-color: #090c10;
                color: rgba(128,221,202,0.92);
                border: none;
                selection-background-color: #1a3a40;
            }
            QScrollBar:vertical {
                background: #0d1117;
                width: 6px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background: #2a3a4a;
                border-radius: 3px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """)

        self.verticalScrollBar().valueChanged.connect(self._on_scroll)

    def wheelEvent(self, event):
        """Scroll only — no Ctrl+wheel zoom."""
        from PySide6.QtCore import Qt
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            event.ignore()
            return
        super().wheelEvent(event)

    def _on_scroll(self, value: int):
        at_bottom = value == self.verticalScrollBar().maximum()
        self._auto_scroll = at_bottom

    # ── Streaming helpers ─────────────────────────────────────────
    _WRAP_COL = 144   # used by append_line textwrap for static lines

    def _begin_stream(self):
        """Called when <<ZS>> arrives — paint 'Zephyr: ' and enter stream mode."""
        self._streaming = True
        # Append a new paragraph with the teal bold "Zephyr: " label.
        self.appendHtml(
            '<span style="color:#80ddca; font-family:Consolas,monospace; '
            'font-weight:bold;">Zephyr: </span>'
        )
        # Create a PERSISTENT cursor anchored to the end of this paragraph.
        # Subsequent appendHtml calls (e.g. tool notifications) add new blocks
        # AFTER this one, but _stream_cursor stays here so tokens always land
        # in the Zephyr paragraph — preventing tool lines from being glued
        # onto the response text.
        cur = QTextCursor(self.document())
        cur.movePosition(QTextCursor.MoveOperation.End)
        cur.setCharFormat(self._stream_fmt)
        self._stream_cursor = cur

    def _stream_token(self, token: str):
        """Append one token via the persistent stream cursor.
        WidgetWidth mode handles visual word-wrap; we only split on real \\n."""
        if self._stream_cursor is None:
            return
        cur = self._stream_cursor
        # Handle any literal newlines in the token
        for i, part in enumerate(token.split('\n')):
            if i > 0:
                cur.insertBlock()
            if part:
                cur.insertText(part, self._stream_fmt)
        if self._auto_scroll:
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def _end_stream(self):
        """Called when <<ZE>> arrives — close the streaming paragraph."""
        self._streaming     = False
        self._stream_cursor = None
        self.appendPlainText("")   # breathing room after response
        if self._auto_scroll:
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    # Line prefixes that are internal noise — never shown to the user
    _MUTED_PREFIXES = (
        "<<SESSION:",
        "[tools_mcp]",
        "[mcp]",
        "[mcp_thread]",
    )

    def append_line(self, line: str):
        """Colorize and append one line from the agent."""
        # Silently consume protocol markers and MCP internal noise
        if any(line.startswith(p) for p in self._MUTED_PREFIXES):
            return
        # ── Streaming protocol ────────────────────────────────
        if line == "<<ZS>>":
            self._begin_stream()
            return
        if line == "<<ZE>>":
            self._end_stream()
            return
        if line.startswith("\x01"):
            self._stream_token(line[1:])
            return

        stripped = line.strip()

        if stripped.startswith("You:"):
            color = "#aab6c2"
        elif stripped.startswith("Zephyr:"):
            color = "#80ddca"
        elif any(tok in stripped for tok in ["[tool]", "tool_call", "Running tool"]):
            color = "#7ab8d8"
        elif any(tok in stripped for tok in ["Error", "error", "Traceback", "failed", "Failed"]):
            color = "#d4a050"
        elif stripped.startswith("─") or stripped.startswith("=") or stripped.startswith("━"):
            color = "#445566"
        else:
            color = "#80ddca"

        # Hard-wrap lines longer than 144 chars so nothing scrolls off-screen.
        # Preserve short lines and special decorators unchanged.
        if len(line) > self._WRAP_COL and not stripped.startswith(("─", "━", "=")):
            # Determine indent for continuation lines (preserve leading spaces)
            indent = len(line) - len(line.lstrip(' '))
            wrap_width = self._WRAP_COL - indent
            wrapped = textwrap.wrap(stripped, width=max(wrap_width, 40),
                                    subsequent_indent=' ' * indent)
            for wl in (wrapped or [stripped]):
                self._append_single_line(wl, color)
            return

        self._append_single_line(line, color)

    def _append_single_line(self, line: str, color: str):
        """Render one line of text as an HTML span and append it."""
        safe = (line
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))
        self.appendHtml(
            f'<span style="color:{color}; '
            f'font-family:Consolas,monospace; '
            f'white-space:pre;">{safe}</span>'
        )
        if self._auto_scroll:
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().maximum()
            )

class FeedbackBar(QWidget):
    """
    Small thumbs-up / thumbs-down row that floats over the console
    bottom-right briefly after each assistant response completes.
    Auto-hides after 8 seconds or after a vote is cast.
    """
    feedback_given = Signal(bool)   # True = thumbs-up, False = thumbs-down

    _DIM  = "#3a4a5a"
    _TEAL = "#1a8272"
    _RED  = "#8b1a1a"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._voted = False
        self._auto_hide_timer = QTimer(self)
        self._auto_hide_timer.setSingleShot(True)
        self._auto_hide_timer.timeout.connect(self.deleteLater)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        self._up   = QPushButton("▲")
        self._down = QPushButton("▼")
        for btn in (self._up, self._down):
            btn.setFixedSize(24, 20)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: #0d1117;
                    color: {self._DIM};
                    border: 1px solid {self._DIM};
                    border-radius: 3px;
                    font-size: 9px;
                }}
                QPushButton:hover {{ color: white; border-color: white; }}
            """)

        self._up.clicked.connect(lambda: self._vote(True))
        self._down.clicked.connect(lambda: self._vote(False))

        layout.addStretch()
        layout.addWidget(self._up)
        layout.addWidget(self._down)
        self.adjustSize()

    def _vote(self, positive: bool):
        if self._voted:
            return
        self._voted = True
        color = self._TEAL if positive else self._RED
        btn = self._up if positive else self._down
        btn.setStyleSheet(btn.styleSheet().replace(self._DIM, color, 2))
        self.feedback_given.emit(positive)
        # Cancel the 8-second auto-hide timer before scheduling our own
        self._auto_hide_timer.stop()
        QTimer.singleShot(1000, self.deleteLater)


def _iso_proj(wx, wy, wz, pcx, pcy, cosY, sinY, cosP, sinP, cam, bias):
    """Isometric perspective projection → (screen_x, screen_y, depth)."""
    dx  = wx * cosY - wz * sinY
    dz_ = wx * sinY + wz * cosY
    dy  = wy * cosP - dz_ * sinP
    dz_ = wy * sinP + dz_ * cosP
    sc  = cam / (cam + dz_ + bias)
    return (pcx + dx * sc, pcy + dy * sc, dz_)


# ═══════════════════════════════════════════════════════════════
#  ThinkingBar — live telemetry slab while Zephyr generates
#  Design: blacklora_telemetry_bar.html  (Prycat Research)
# ═══════════════════════════════════════════════════════════════
class ThinkingBar(QWidget):
    HEIGHT = 80
    _CELL_LABELS = ["MODEL",   "/BW CONFIG",  "ORACLE",   "LOAD"]
    _CELL_VALUES = ["hermes3:8b", "vector accrual", "branch sel.", "inertia cls-v"]

    model_cell_clicked      = Signal()
    blackwell_cell_clicked  = Signal()
    oracle_cell_clicked     = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(self.HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self._model_cell_hovered     = False
        self._blackwell_cell_hovered = False
        self._oracle_cell_hovered    = False
        self._tq_enabled = False
        self._active_model_full = "hermes3:8b"
        self._CELL_VALUES = list(self._CELL_VALUES)  # make instance-level mutable copy

        # Animation state
        self._stress   = 0.26;  self._t_stress = 0.26
        self._progress = 0.43;  self._t_prog   = 0.43
        self._phases   = [0.24, 0.72, 0.51, 0.63]
        self._t_phases = [0.24, 0.72, 0.51, 0.63]
        self._frame    = 0
        self._dot      = 0.0
        self._dot_dir  = 1.0
        self._vflow    = 0.0
        self._active   = False   # True while Zephyr is generating
        self._loading  = False   # True between user submit and first token
        self._token_gaps: deque = deque(maxlen=20)  # inter-token gaps in ms
        self._gap_max    = 80.0   # rolling normalisation ceiling (ms)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(16)
        self._timer.start()      # always animating

    # ── Public ────────────────────────────────────────────────
    def set_loading(self):
        """LOADING state — orange — shown between submit and first token."""
        self._loading = True
        self._active  = False

    def start(self):
        """THINKING state — red — shown while tokens are streaming."""
        self._loading = False
        self._active  = True
        self._frame   = 0

    def stop(self):
        """READY state — teal — shown when generation is complete."""
        self._active  = False
        self._loading = False
        self._token_gaps.clear()  # intentional instant clear; idle pulse restores gentle terrain
        self._gap_max  = 80.0

    def record_token_gap(self, gap_ms: float):
        """Slot: receives inter-token gap (ms) from ZephyrProcess.token_gap signal.

        First call per stream is time-to-first-token; subsequent calls are
        true inter-token gaps. Both are useful as compute-intensity proxies.
        """
        self._token_gaps.append(gap_ms)
        if gap_ms > self._gap_max:
            self._gap_max = gap_ms
        else:
            self._gap_max = max(40.0, self._gap_max * 0.995)

    def _surface_heights(self):
        """Return a GRID_Z × GRID_X list-of-lists of normalized heights (0.0–1.0).

        Axes:
          X (columns, ix 0→9): token position, oldest left → newest right
          Z (rows,    iz 0→9): smoothing lane — front=raw, back=heavily averaged
            iz=0  raw inter-token gap (jagged, nearest lane)
            iz=k  (2k+1)-sample rolling average  (k=1..9, progressively smoother)
        """
        GRID_X, GRID_Z = 10, 10
        gaps = list(self._token_gaps)   # oldest first
        N    = len(gaps)
        norm = max(1.0, self._gap_max)

        result = [[0.0] * GRID_X for _ in range(GRID_Z)]
        if N == 0:
            return result

        for ix in range(GRID_X):
            # Map column index to buffer position via linear interpolation
            t    = ix / (GRID_X - 1) if GRID_X > 1 else 0.0
            fp   = t * (N - 1)
            lo   = int(fp)
            hi   = min(lo + 1, N - 1)
            frac = fp - lo
            raw_v = gaps[lo] * (1.0 - frac) + gaps[hi] * frac

            for iz in range(GRID_Z):
                if iz == 0:
                    h = raw_v
                else:
                    # half=iz → kernel width = 2*iz+1 samples
                    # iz=1→3-smp, iz=2→5-smp, … iz=9→19-smp (smooth back)
                    half    = iz
                    centre  = int(t * (N - 1))
                    samples = [gaps[max(0, min(N - 1, centre + off))]
                               for off in range(-half, half + 1)]
                    h = sum(samples) / len(samples)

                result[iz][ix] = min(1.0, h / norm)

        return result

    def _collapse(self):
        pass   # kept for compatibility, no longer used

    def _cell0_rect(self) -> QRect:
        """Bounding rect of cell 0 in local widget coordinates."""
        W, H = self.width(), self.height()
        PAD     = 12
        LEFT_W  = 196
        RIGHT_W = 200
        mid_x   = PAD + LEFT_W + 10
        mid_w   = W - PAD - LEFT_W - 10 - RIGHT_W - 10 - PAD
        cell_gap = 6
        cell_w   = max(1, (mid_w - cell_gap * 3) // 4)
        cell_h   = H - PAD * 2
        return QRect(int(mid_x), PAD, int(cell_w), int(cell_h))

    def _cell1_rect(self) -> QRect:
        """Bounding rect of cell 1 (/BW CONFIG) in local widget coordinates."""
        W, H = self.width(), self.height()
        PAD     = 12
        LEFT_W  = 196
        RIGHT_W = 200
        mid_x   = PAD + LEFT_W + 10
        mid_w   = W - PAD - LEFT_W - 10 - RIGHT_W - 10 - PAD
        cell_gap = 6
        cell_w   = max(1, (mid_w - cell_gap * 3) // 4)
        cell_h   = H - PAD * 2
        return QRect(int(mid_x + cell_w + cell_gap), PAD, int(cell_w), int(cell_h))

    def _cell2_rect(self) -> QRect:
        """Bounding rect of cell 2 (ORACLE) in local widget coordinates."""
        W, H = self.width(), self.height()
        PAD     = 12
        LEFT_W  = 196
        RIGHT_W = 200
        mid_x   = PAD + LEFT_W + 10
        mid_w   = W - PAD - LEFT_W - 10 - RIGHT_W - 10 - PAD
        cell_gap = 6
        cell_w   = max(1, (mid_w - cell_gap * 3) // 4)
        cell_h   = H - PAD * 2
        return QRect(int(mid_x + (cell_w + cell_gap) * 2), PAD, int(cell_w), int(cell_h))

    # ── Animation ─────────────────────────────────────────────
    @staticmethod
    def _lerp(a, b, t):
        return a + (b - a) * t

    def _tick(self):
        self._frame += 1
        if self._frame % 120 == 0:
            self._t_stress = max(0.08, min(0.96,
                self._stress + (random.random() - 0.42) * 0.32))
            self._t_prog   = max(0.12, min(0.94,
                self._progress + (random.random() - 0.40) * 0.26))
            self._t_phases = [
                max(0.06, min(0.96, p + (random.random() - 0.40) * 0.28))
                for p in self._phases
            ]
        L = self._lerp
        if self._active:
            # Thinking: values drift freely toward random targets (busy, alive)
            self._stress   = L(self._stress,   self._t_stress, 0.024)
            self._progress = L(self._progress, self._t_prog,   0.030)
            self._phases   = [L(p, t, 0.030) for p, t in zip(self._phases, self._t_phases)]
        else:
            # Ready: drain everything down to quiet near-zero levels
            self._stress   = L(self._stress,   0.08, 0.014)
            self._progress = L(self._progress, 0.12, 0.014)
            self._phases   = [L(p, 0.08, 0.014) for p in self._phases]

        # Dot speed: brisk when active, slow pulse when idle
        speed = 0.013 if self._active else 0.004
        self._dot += speed * self._dot_dir
        if self._dot >= 1.0: self._dot = 1.0; self._dot_dir = -1
        if self._dot <= 0.0: self._dot = 0.0; self._dot_dir =  1

        flow_speed = 0.28 if self._active else 0.06
        self._vflow = (self._vflow + flow_speed) % 36.0
        self.update()

    def set_active_model(self, model: str):
        """Update cell 0 to display the active model name."""
        self._active_model_full = model
        display = model if len(model) <= 14 else model[:13] + "\u2026"
        if self._tq_enabled:
            display = (model if len(model) <= 11 else model[:11]) + " TQ"
        self._CELL_VALUES[0] = display
        self.update()

    def set_turboquant(self, enabled: bool):
        """Toggle TurboQuant badge on cell 0."""
        self._tq_enabled = enabled
        self.set_active_model(self._active_model_full)

    # ── Mouse interaction (cells 0, 1, 2) ────────────────────
    def mouseMoveEvent(self, e):
        pos = e.pos()
        h0 = self._cell0_rect().contains(pos)
        h1 = self._cell1_rect().contains(pos)
        h2 = self._cell2_rect().contains(pos)
        changed = (h0 != self._model_cell_hovered or
                   h1 != self._blackwell_cell_hovered or
                   h2 != self._oracle_cell_hovered)
        self._model_cell_hovered     = h0
        self._blackwell_cell_hovered = h1
        self._oracle_cell_hovered    = h2
        if changed:
            self.setCursor(
                Qt.PointingHandCursor if (h0 or h1 or h2) else Qt.ArrowCursor
            )
            self.update()
        super().mouseMoveEvent(e)

    def leaveEvent(self, e):
        if self._model_cell_hovered or self._blackwell_cell_hovered or self._oracle_cell_hovered:
            self._model_cell_hovered     = False
            self._blackwell_cell_hovered = False
            self._oracle_cell_hovered    = False
            self.setCursor(Qt.ArrowCursor)
            self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            if self._cell0_rect().contains(e.pos()):
                self.model_cell_clicked.emit()
            elif self._cell1_rect().contains(e.pos()):
                self.blackwell_cell_clicked.emit()
            elif self._cell2_rect().contains(e.pos()):
                self.oracle_cell_clicked.emit()
        super().mousePressEvent(e)

    # ── Signal colour ─────────────────────────────────────────
    # READY  → stress-modulated teal  (cool → bright teal)
    # THINKING → stress-modulated red (dim red → hot red/orange)
    def _sig(self, alpha=255):
        t = self._stress
        if self._active:
            # THINKING — red: #c03030 → #ff6040 as stress rises
            return QColor(int(192 + t * 63),
                          int(48  + t * 48),
                          int(48  + t * 16),
                          alpha)
        elif self._loading:
            # LOADING — orange: warm amber-orange, pulsed by stress
            return QColor(int(210 + t * 35),
                          int(120 + t * 30),
                          int(20  + t *  8),
                          alpha)
        else:
            # READY — teal: dim at low stress, brighter at high
            return QColor(int(26  + t * 40),
                          int(130 + t * 54),
                          int(115 + t * 51),
                          alpha)

    # ── Paint ─────────────────────────────────────────────────
    def paintEvent(self, event):
        p   = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        sig  = self._sig()

        # ── Glass background ─────────────────────────────────
        bg = QLinearGradient(0, 0, 0, H)
        bg.setColorAt(0, QColor(15, 19, 24, 212))
        bg.setColorAt(1, QColor( 9, 12, 16, 198))
        p.setPen(QPen(QColor(255, 255, 255, 28), 1))
        p.setBrush(QBrush(bg))
        p.drawRoundedRect(QRectF(0.5, 0.5, W - 1, H - 1), 8, 8)

        # Top-edge highlight
        hl = QLinearGradient(0, 1, W, 1)
        hl.setColorAt(0.0,  QColor(255, 255, 255, 0))
        hl.setColorAt(0.35, QColor(255, 255, 255, 14))
        hl.setColorAt(1.0,  QColor(255, 255, 255, 0))
        p.setPen(QPen(QBrush(hl), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawLine(QPointF(10, 1), QPointF(W - 10, 1))

        # Ambient glow (bottom-left)
        ag = QRadialGradient(QPointF(W * 0.18, H + 10), W * 0.36)
        ag.setColorAt(0, self._sig(24))
        ag.setColorAt(1, QColor(0, 0, 0, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(ag))
        p.drawRect(0, 0, W, H)

        # ── Geometry ─────────────────────────────────────────
        PAD     = 12
        LEFT_W  = 196
        RIGHT_W = 200
        mid_x   = PAD + LEFT_W + 10
        mid_w   = W - PAD - LEFT_W - 10 - RIGHT_W - 10 - PAD
        right_x = W - PAD - RIGHT_W

        # ── LEFT BLOCK ───────────────────────────────────────
        lx, ly = PAD, 9

        p.setFont(QFont("Consolas", 7))
        p.setPen(QColor(191, 203, 212, 86))
        p.drawText(lx, ly + 9, "BLACKLORA-N  ·  PROMPT ENGINE")

        p.setFont(QFont("Consolas", 13, QFont.Weight.Bold))
        t = self._stress
        if self._active:
            # THINKING — vivid red, pulses with stress
            title_col = QColor(int(220 + t * 35), int(50 + t * 30), int(50 + t * 10))
            title_txt = "THINKING"
        elif self._loading:
            # LOADING — warm orange, pulses gently
            title_col = QColor(int(225 + t * 20), int(130 + t * 25), int(30 + t * 10))
            title_txt = "LOADING"
        else:
            # READY — muted teal, calm
            title_col = QColor(80, 160, 148)
            title_txt = "READY"
        p.setPen(title_col)
        p.drawText(lx, ly + 28, title_txt)

        p.setFont(QFont("Consolas", 7))
        pill_txt = "BlackLoRA-N core cycle"
        pfm  = p.fontMetrics()
        pw   = pfm.horizontalAdvance(pill_txt) + 14
        ph   = 16
        py_  = ly + 36
        p.setPen(QPen(QColor(255, 255, 255, 16), 1))
        p.setBrush(QBrush(QColor(255, 255, 255, 10)))
        p.drawRoundedRect(QRectF(lx, py_, pw, ph), 3, 3)
        p.setPen(QColor(220, 228, 238, 185))
        p.drawText(int(lx + 7), int(py_ + ph - 4), pill_txt)
        p.setPen(QColor(191, 203, 212, 105))
        p.drawText(int(lx + pw + 8), int(py_ + ph - 4),
                   f"density {self._stress:.2f}")

        # ── CENTER: 4 TELEMETRY CELLS ─────────────────────────
        if mid_w > 60:
            cell_gap = 6
            cell_w   = max(1, (mid_w - cell_gap * 3) // 4)
            cell_h   = H - PAD * 2

            for i in range(4):
                cx = mid_x + i * (cell_w + cell_gap)
                cy = PAD

                p.setPen(QPen(QColor(255, 255, 255, 22), 1))
                p.setBrush(QBrush(QColor(255, 255, 255, 12)))
                p.drawRect(QRectF(cx, cy, cell_w, cell_h))

                if i == 0 and self._model_cell_hovered:
                    p.setPen(QPen(QColor("#2a6258"), 1))
                    p.drawRect(QRectF(cx, cy, cell_w, cell_h))

                if i == 1 and self._blackwell_cell_hovered:
                    p.setPen(QPen(QColor("#2a3d62"), 1))
                    p.drawRect(QRectF(cx, cy, cell_w, cell_h))

                p.setFont(QFont("Consolas", 7))
                p.setPen(QColor(191, 203, 212, 118))
                p.drawText(int(cx + 6), int(cy + 14), self._CELL_LABELS[i])

                p.setFont(QFont("Consolas", 8))
                p.setPen(QColor(230, 238, 245, 185))
                val = p.fontMetrics().elidedText(
                    self._CELL_VALUES[i],
                    Qt.TextElideMode.ElideRight, cell_w - 10)
                p.drawText(int(cx + 6), int(cy + cell_h - 12), val)

                fw = max(0.0, self._phases[i] * cell_w)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(self._sig(38)))
                p.drawRect(QRectF(cx, cy + cell_h - 5, fw, 5))
                p.setBrush(QBrush(self._sig(200)))
                p.drawRect(QRectF(cx, cy + cell_h - 2, fw, 2))

        # ── RIGHT BLOCK ───────────────────────────────────────
        if right_x > mid_x + 40:
            rx, ry = right_x, PAD

            # ── 3D Token-Timing Surface ───────────────────────
            # X = token bucket  (0=oldest, 9=newest)
            # Z = smoothing lane (0=raw front, 9=heavily-averaged back)
            # Y = normalised inter-token gap height
            #
            # Grid is 10×10 (square), YAW=45° → projects as a diamond
            # filling the full width; shallow PITCH gives low-angle look.
            VH     = 50               # panel height (px)
            GRID_X = GRID_Z = 10
            CELL   = 24.0             # world units per cell (equal X & Z → square)
            MAX_H  = 20.0             # world height when gap=1.0
            _YAW   = math.pi / 4     # exactly 45° → perfect diamond silhouette
            _PITCH = -0.20            # very shallow: low-angle isometric feel
            _CAM   = 300.0
            _BIAS  = 160.0

            _cosY = math.cos(_YAW);  _sinY = math.sin(_YAW)   # = 1/√2 each
            _cosP = math.cos(_PITCH); _sinP = math.sin(_PITCH)

            # Centre projection on the right block; ground plane sits at 75 %
            _pcx = rx + RIGHT_W * 0.5
            _pcy = ry + VH * 0.75

            # Height field (0–1), 10 rows × 10 cols
            _hf = self._surface_heights()

            # Idle pulse so terrain breathes when no tokens are flowing
            if not self._active:
                _pulse = 0.06 * (0.5 + 0.5 * math.sin(self._frame * 0.025))
                _hf = [[min(1.0, v + _pulse) for v in row] for row in _hf]

            # Build projected quads (9×9 = 81 quads)
            _sig_c  = self._sig()
            _quads  = []
            _half   = (GRID_X - 1) / 2.0      # same for X and Z (square)
            for iz in range(GRID_Z - 1):
                for ix in range(GRID_X - 1):
                    wx0 = (ix     - _half) * CELL
                    wx1 = (ix + 1 - _half) * CELL
                    wz0 = (iz     - _half) * CELL
                    wz1 = (iz + 1 - _half) * CELL
                    h00 = _hf[iz  ][ix  ] * MAX_H
                    h10 = _hf[iz  ][ix+1] * MAX_H
                    h11 = _hf[iz+1][ix+1] * MAX_H
                    h01 = _hf[iz+1][ix  ] * MAX_H
                    a = _iso_proj(wx0, -h00, wz0, _pcx, _pcy, _cosY, _sinY, _cosP, _sinP, _CAM, _BIAS)
                    b = _iso_proj(wx1, -h10, wz0, _pcx, _pcy, _cosY, _sinY, _cosP, _sinP, _CAM, _BIAS)
                    c = _iso_proj(wx1, -h11, wz1, _pcx, _pcy, _cosY, _sinY, _cosP, _sinP, _CAM, _BIAS)
                    d = _iso_proj(wx0, -h01, wz1, _pcx, _pcy, _cosY, _sinY, _cosP, _sinP, _CAM, _BIAS)
                    avg_depth = (a[2] + b[2] + c[2] + d[2]) * 0.25
                    avg_h     = (h00  + h10  + h11  + h01 ) * 0.25
                    _quads.append((avg_depth, avg_h, a, b, c, d))

            # Painter's algorithm: back-to-front
            _quads.sort(key=lambda q: q[0])

            # Pass 1: filled quads + grid lines
            for _dep, _ah, _a, _b, _c, _d in _quads:
                _inten  = min(1.0, _ah / MAX_H)
                _falpha = int(10 + _inten * 80)
                _path   = QPainterPath()
                _path.moveTo(QPointF(_a[0], _a[1]))
                _path.lineTo(QPointF(_b[0], _b[1]))
                _path.lineTo(QPointF(_c[0], _c[1]))
                _path.lineTo(QPointF(_d[0], _d[1]))
                _path.closeSubpath()
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(QColor(
                    _sig_c.red(), _sig_c.green(), _sig_c.blue(), _falpha)))
                p.drawPath(_path)
                _salpha = int(6 + _inten * 32)
                p.setPen(QPen(QColor(200, 230, 255, _salpha), 0.5))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPath(_path)

            # Pass 2: radial glow on peaks
            for _dep, _ah, _a, _b, _c, _d in _quads:
                _inten = min(1.0, _ah / MAX_H)
                if _inten < 0.22:
                    continue
                _qcx = (_a[0] + _b[0] + _c[0] + _d[0]) * 0.25
                _qcy = (_a[1] + _b[1] + _c[1] + _d[1]) * 0.25
                _rg  = 2.5 + _inten * 5.0
                _gg  = QRadialGradient(QPointF(_qcx, _qcy), _rg)
                _gg.setColorAt(0, self._sig(int(60 * _inten)))
                _gg.setColorAt(1, QColor(0, 0, 0, 0))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(_gg))
                p.drawEllipse(QPointF(_qcx, _qcy), _rg, _rg)

            # Footer text (below vector view, fixed position)
            FY = ry + VH + 5
            p.setFont(QFont("Consolas", 7))
            p.setPen(QColor(191, 203, 212, 86))
            p.drawText(int(rx), int(FY + 9),
                       "blacklora loop // weighted inference")
            rt   = f"p={self._progress:.2f} / rv={self._stress:.2f}"
            rt_w = p.fontMetrics().horizontalAdvance(rt)
            p.drawText(int(rx + RIGHT_W - rt_w), int(FY + 9), rt)

        # ── Full-width bouncing scanner line ──────────────────
        # Always drawn edge-to-edge; colour tracks state automatically.
        TPAD = PAD
        TY   = H - 14
        TH   = 7
        tw   = W - TPAD * 2

        # Track rect
        p.setPen(QPen(QColor(255, 255, 255, 18), 1))
        p.setBrush(QBrush(QColor(255, 255, 255, 7)))
        p.drawRoundedRect(QRectF(TPAD, TY, tw, TH), 3, 3)

        # State-coloured centre glow along track
        tc   = self._sig()
        lg2  = QLinearGradient(TPAD, 0, TPAD + tw, 0)
        lg2.setColorAt(0.0, QColor(tc.red(), tc.green(), tc.blue(), 5))
        lg2.setColorAt(0.5, QColor(tc.red(), tc.green(), tc.blue(), 30))
        lg2.setColorAt(1.0, QColor(tc.red(), tc.green(), tc.blue(), 5))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(lg2))
        p.drawRect(QRectF(TPAD, TY + TH / 2 - 1, tw, 2))

        # Bouncing dot — smoothstep eased, full-width travel
        te   = self._dot * self._dot * (3 - 2 * self._dot)
        DOT  = 5
        dx   = TPAD + DOT + te * (tw - DOT * 2)
        dy   = TY + TH / 2
        # Soft halo
        dg2  = QRadialGradient(QPointF(dx, dy), DOT * 4.5)
        dg2.setColorAt(0, self._sig(90))
        dg2.setColorAt(1, QColor(0, 0, 0, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(dg2))
        p.drawEllipse(QPointF(dx, dy), DOT * 4.5, DOT * 4.5)
        # Bright core
        bright = self._sig()
        dc2    = QColor(int(bright.red()   * 0.55 + 255 * 0.45),
                        int(bright.green() * 0.55 + 255 * 0.45),
                        int(bright.blue()  * 0.55 + 255 * 0.45))
        p.setBrush(QBrush(dc2))
        p.setPen(QPen(QColor(255, 255, 255, 45), 0.8))
        p.drawEllipse(QPointF(dx, dy), DOT, DOT)

        p.end()


# ═══════════════════════════════════════════════════════════════
#  InputBar
# ═══════════════════════════════════════════════════════════════
class InputBar(QLineEdit):
    """
    Signals:
        submitted(str) — emitted when user hits Enter or clicks Send
    """
    submitted = Signal(str)

    HISTORY_MAX = 50

    def __init__(self, parent=None):
        super().__init__(parent)
        self._history     = []
        self._history_idx = -1

        font = QFont("Consolas", 10)
        self.setFont(font)
        self.setPlaceholderText("▶  type a message or /command...")
        self.setStyleSheet("""
            QLineEdit {
                background-color: #0d1117;
                color: rgba(128,221,202,0.92);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 4px;
                padding: 8px 12px;
                selection-background-color: #1a3a40;
            }
            QLineEdit:focus {
                border-color: rgba(128,221,202,0.38);
            }
        """)
        self.returnPressed.connect(self._fire)

    def _fire(self):
        text = self.text().strip()
        if not text:
            return
        if not self._history or self._history[-1] != text:
            self._history.append(text)
            if len(self._history) > self.HISTORY_MAX:
                self._history.pop(0)
        self._history_idx = -1
        self.clear()
        self.submitted.emit(text)

    def inject(self, text: str, fire: bool = False):
        """Pre-fill the input bar. If fire=True, submit immediately."""
        self.setFocus()
        self.setText(text)
        self.setCursorPosition(len(text))
        if fire:
            self._fire()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Up:
            if self._history:
                if self._history_idx == -1:
                    self._history_idx = len(self._history) - 1
                elif self._history_idx > 0:
                    self._history_idx -= 1
                self.setText(self._history[self._history_idx])
                self.setCursorPosition(len(self.text()))
            return
        if event.key() == Qt.Key.Key_Down:
            if self._history_idx != -1:
                if self._history_idx < len(self._history) - 1:
                    self._history_idx += 1
                    self.setText(self._history[self._history_idx])
                else:
                    self._history_idx = -1
                    self.clear()
                self.setCursorPosition(len(self.text()))
            return
        super().keyPressEvent(event)


# ═══════════════════════════════════════════════════════════════
#  ZephyrTitleBar — custom frameless title bar
#  Pixel hydra mark · frosted glass · draggable · window controls
# ═══════════════════════════════════════════════════════════════
class ZephyrTitleBar(QWidget):
    """
    Fully custom-painted title bar replacing the OS chrome.
    Left:   10×8 pixel hydra icon in emerald + ZEPHYR wordmark + live dot
    Right:  minimize / maximize / close buttons (painted, hover-aware)
    Drag:   click-drag anywhere outside the buttons moves the window
    """

    HEIGHT = 48

    # Pixel hydra cells as (col, row) in a 10-col × 8-row grid (0-indexed)
    _HYDRA: frozenset = frozenset([
        # center spine
        (5,1),(5,2),(5,3),(5,4),(5,5),(5,6),
        # left head / neck
        (3,1),(4,1),(4,2),(3,2),(2,2),(2,1),
        # right head / neck
        (6,1),(7,1),(7,2),(8,2),(8,1),(6,2),
        # lower flares
        (4,4),(3,5),(2,6),(4,5),
        (6,4),(7,5),(8,6),(6,5),
        # center jaw / body width
        (4,3),(6,3),(4,6),(6,6),
    ])

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(self.HEIGHT)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

        self._drag_pos   = None          # for window dragging
        self._hovered    = None          # 'min' | 'max' | 'close' | None
        self._dot_phase  = 0.0           # live-dot sine phase

        pulse = QTimer(self)
        pulse.setInterval(40)            # 25 fps is plenty for a sine wave
        pulse.timeout.connect(self._tick)
        pulse.start()

    # ── Animation ────────────────────────────────────────────────

    def _tick(self):
        self._dot_phase = (self._dot_phase + 0.07) % (2 * math.pi)
        self.update()

    # ── Button geometry ──────────────────────────────────────────

    def _btn_rects(self):
        """Return (min_rect, max_rect, close_rect) as plain tuples (x,y,w,h)."""
        W, H = self.width(), self.HEIGHT
        bw, bh, pad = 38, 22, 8
        cy = (H - bh) // 2
        cx = W - pad - bw
        mx = cx - pad - bw
        nx = mx - pad - bw
        return (nx, cy, bw, bh), (mx, cy, bw, bh), (cx, cy, bw, bh)

    def _btn_hit(self, px, py):
        """Return which button contains (px,py), or None."""
        for key, rect in zip(("min","max","close"), self._btn_rects()):
            x,y,w,h = rect
            if x <= px <= x+w and y <= py <= y+h:
                return key
        return None

    # ── Paint ─────────────────────────────────────────────────────

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.HEIGHT

        # ── Background: deep navy, 90 % opaque ─────────────────
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(7, 11, 18, 232)))
        p.drawRect(0, 0, W, H)

        # Left emerald ambient bloom
        bloom = QRadialGradient(QPointF(70, H * 0.5), 100)
        bloom.setColorAt(0.0, QColor(26, 130, 115, 32))
        bloom.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setBrush(QBrush(bloom))
        p.drawRect(0, 0, W, H)

        # Top highlight stripe (1 px)
        hl = QLinearGradient(0, 0, W, 0)
        hl.setColorAt(0.00, QColor(255, 255, 255,  0))
        hl.setColorAt(0.25, QColor(255, 255, 255, 40))
        hl.setColorAt(0.65, QColor(128, 221, 202, 28))
        hl.setColorAt(1.00, QColor(255, 255, 255,  0))
        p.setBrush(QBrush(hl))
        p.drawRect(0, 0, W, 1)

        # Bottom separator
        p.setBrush(QBrush(QColor(255, 255, 255, 10)))
        p.drawRect(0, H - 1, W, 1)

        # ── Pixel hydra icon ────────────────────────────────────
        IX, IY   = 14, 6          # icon top-left
        IW, IH   = 36, 36         # rendered size in px
        COLS, ROWS = 10, 8
        CW   = IW / COLS
        CH   = IH / ROWS
        GAP  = 0.9

        # Icon background pill
        p.setBrush(QBrush(QColor(0, 0, 0, 55)))
        p.setPen(QPen(QColor(26, 130, 115, 55), 0.8))
        p.drawRoundedRect(QRectF(IX - 3, IY - 3, IW + 6, IH + 6), 6, 6)

        p.setPen(Qt.PenStyle.NoPen)
        for (cx, cy) in self._HYDRA:
            rx = IX + cx * CW + GAP * 0.5
            ry = IY + cy * CH + GAP * 0.5
            rw = CW - GAP
            rh = CH - GAP
            # Soft outer glow
            p.setBrush(QBrush(QColor(74, 222, 128, 38)))
            p.drawRoundedRect(QRectF(rx - 1.2, ry - 1.2, rw + 2.4, rh + 2.4), 1.2, 1.2)
            # Cell body
            p.setBrush(QBrush(QColor(74, 222, 128, 218)))
            p.drawRoundedRect(QRectF(rx, ry, rw, rh), 0.7, 0.7)

        # ── ZEPHYR wordmark ─────────────────────────────────────
        TX = IX + IW + 12
        p.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
        p.setPen(QColor(210, 230, 225, 230))
        p.drawText(int(TX), int(H * 0.5 + 4), "ZEPHYR")

        title_w = p.fontMetrics().horizontalAdvance("ZEPHYR")

        p.setFont(QFont("Consolas", 8))
        p.setPen(QColor(26, 155, 135, 155))
        p.drawText(int(TX), int(H * 0.5 + 16), "hydra runtime shell")

        # ── Live dot (sine-pulsed) ───────────────────────────────
        dot_a = int(170 + 85 * math.sin(self._dot_phase))
        DX = TX + title_w + 14
        DY = H * 0.5 - 3.0
        DR = 3.2
        # Glow halo
        glow = QRadialGradient(QPointF(DX, DY), DR * 2.8)
        glow.setColorAt(0.0, QColor(102, 196, 122, dot_a))
        glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setBrush(QBrush(glow))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(DX, DY), DR * 2.8, DR * 2.8)
        # Dot core
        p.setBrush(QBrush(QColor(120, 210, 140, min(255, dot_a + 40))))
        p.drawEllipse(QPointF(DX, DY), DR, DR)

        # ── Window control buttons ───────────────────────────────
        # (key, bg_norm, bg_hover, border, icon_colour)
        BTN_STYLES = {
            "min":   (QColor(255,255,255, 10), QColor(255,255,255, 22),
                      QColor(255,255,255, 18), QColor(170, 182, 194, 200)),
            "max":   (QColor(26, 130, 115, 16), QColor(26, 130, 115, 34),
                      QColor(26, 130, 115, 40), QColor(128, 221, 202, 200)),
            "close": (QColor(160,  40,  40, 18), QColor(200,  55,  55, 38),
                      QColor(200,  60,  60, 45), QColor(220,  88,  88, 215)),
        }

        for key, (x, y, bw, bh) in zip(("min","max","close"), self._btn_rects()):
            hov = (self._hovered == key)
            bg_n, bg_h, bd, ic = BTN_STYLES[key]
            rect = QRectF(x, y, bw, bh)

            # Face
            p.setBrush(QBrush(bg_h if hov else bg_n))
            p.setPen(QPen(bd if hov else QColor(255,255,255,14), 0.7))
            p.drawRoundedRect(rect, 6, 6)

            # Top sheen
            sheen = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            sheen.setColorAt(0, QColor(255,255,255, 16 if hov else 10))
            sheen.setColorAt(1, QColor(0,0,0,0))
            p.setBrush(QBrush(sheen))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(rect, 6, 6)

            # Icon strokes
            ccx = rect.center().x()
            ccy = rect.center().y()
            pen = QPen(ic, 1.6, Qt.PenStyle.SolidLine,
                       Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)

            if key == "min":
                p.drawLine(QPointF(ccx - 5.5, ccy), QPointF(ccx + 5.5, ccy))
            elif key == "max":
                w_m = self.window()
                if w_m and w_m.isMaximized():
                    # Restore icon: two overlapping squares
                    p.drawRect(QRectF(ccx - 5, ccy - 4.5, 8.5, 7.5))
                    p.drawRect(QRectF(ccx - 3, ccy - 6.5, 8.5, 7.5))
                else:
                    p.drawRect(QRectF(ccx - 4.5, ccy - 4, 9, 8))
            else:  # close
                p.drawLine(QPointF(ccx - 4.5, ccy - 3.5), QPointF(ccx + 4.5, ccy + 3.5))
                p.drawLine(QPointF(ccx + 4.5, ccy - 3.5), QPointF(ccx - 4.5, ccy + 3.5))

        p.end()

    # ── Mouse events ─────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position().toPoint()
        hit = self._btn_hit(pos.x(), pos.y())
        if hit == "close":
            self.window().close()
        elif hit == "max":
            w = self.window()
            w.showNormal() if w.isMaximized() else w.showMaximized()
        elif hit == "min":
            self.window().showMinimized()
        else:
            self._drag_pos = (
                event.globalPosition().toPoint()
                - self.window().frameGeometry().topLeft()
            )
        event.accept()

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        prev = self._hovered
        self._hovered = self._btn_hit(pos.x(), pos.y())
        if self._hovered != prev:
            self.update()
            self.setCursor(
                Qt.CursorShape.PointingHandCursor
                if self._hovered else Qt.CursorShape.ArrowCursor
            )
        if (event.buttons() & Qt.MouseButton.LeftButton
                and self._drag_pos is not None
                and self._hovered is None):
            self.window().move(
                event.globalPosition().toPoint() - self._drag_pos
            )
        event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if self._btn_hit(event.position().toPoint().x(),
                         event.position().toPoint().y()) is None:
            w = self.window()
            w.showNormal() if w.isMaximized() else w.showMaximized()
        event.accept()

    def leaveEvent(self, event):
        if self._hovered is not None:
            self._hovered = None
            self.update()
        self.setCursor(Qt.CursorShape.ArrowCursor)


# ═══════════════════════════════════════════════════════════════
#  PaletteWidget — right command panel
# ═══════════════════════════════════════════════════════════════
class SectionDivider(QWidget):
    """Thin labelled divider between button groups."""
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 2)
        lbl = QLabel(label.upper())
        lbl.setStyleSheet("""
            color: rgba(170,182,194,0.45);
            font-family: Consolas, monospace;
            font-size: 9px;
            letter-spacing: 3px;
        """)
        layout.addWidget(lbl)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: rgba(255,255,255,0.06);")
        layout.addWidget(line)
        layout.setStretch(1, 1)


class PaletteWidget(QWidget):
    # Emits (command: str, fire: bool)
    command_requested = Signal(str, bool)

    BUTTONS = [
        (
            "/blackwell",
            "/blackwell",
            "Drops Zephyr into a planning space where he interviews you,\n"
            "and your answers reshape how he sees the world — permanently.",
            True,
        ),
        (
            "/axioms",
            "/blackwell axioms",
            "20-question interview: logic, tone, and antinomy probes.\n"
            "You set the ground truth — your answers become the immutable\n"
            "training anchors Zephyr cannot drift away from.",
            True,
        ),
        (
            "/coding-blackwell",
            "/coding-blackwell",
            "CS-focused planning session — Zephyr interviews you on coding habits,\n"
            "languages, and problems. Sharpens his coding instincts permanently.",
            True,
        ),
        (
            "/trajectory",
            "/trajectory",
            "Show trajectory pair counts and current regret vector.\n"
            "Every real conversation is logged here for fine-tuning.",
            True,
        ),
        (
            "/help",
            "/help",
            "Show all available commands.",
            True,
        ),
        (
            "/tools",
            "/tools",
            "List all of Zephyr's active tools.",
            True,
        ),
        (
            "/search",
            "/search ",
            "Raw DuckDuckGo search, instant.\nUsage: /search <query>",
            False,
        ),
        (
            "/browse",
            "/browse ",
            "Fetch a URL directly.\nUsage: /browse <url>",
            False,
        ),
        (
            "/run",
            "/run ",
            "Run Python immediately.\nUsage: /run <code>",
            False,
        ),
        (
            "/status",
            "/status",
            "Check that Ollama is alive and responding.",
            True,
        ),
        (
            "/model",
            "/model",
            "Show current model name and API connection info.",
            True,
        ),
        (
            "/save",
            "/save",
            "Save conversation to Obsidian vault as a formatted .md\n"
            "with YAML frontmatter (date, time, model, tags).\n"
            "Usage: /save  or  /save my research chat",
            True,
        ),
        (
            "/clear",
            "/clear",
            "Reset conversation history.\nZephyr will ask for confirmation (y/n).",
            True,
        ),
    ]

    KEYS_BUTTONS = [
        (
            "/keys setup",
            "/keys setup",
            "Interactive wizard: select provider, enter your API key.\n"
            "Stored masked in ~/.zephyr/keys.json.\n"
            "Providers: claude, gpt, grok, gemini",
            True,
        ),
        (
            "/keys list",
            "/keys list",
            "Show which providers are configured:\n"
            "claude ✓  gpt ✓  grok ✗  gemini ✓",
            True,
        ),
    ]

    CALL_BUTTONS = [
        (
            "/call",
            "/call ",
            "Route your message to the best available external AI.\n"
            "Passes context so the AI knows it's consulting for Zephyr/Prycat.\n"
            "Usage: /call <message>",
            False,
        ),
        (
            "/call claude",
            "/call claude ",
            "Force Claude (claude-opus-4-5 via Anthropic).\nUsage: /call claude <message>",
            False,
        ),
        (
            "/call gpt",
            "/call gpt ",
            "Force GPT-4o via OpenAI.\nUsage: /call gpt <message>",
            False,
        ),
        (
            "/call grok",
            "/call grok ",
            "Force Grok-3 via xAI endpoint.\nUsage: /call grok <message>",
            False,
        ),
        (
            "/call gemini",
            "/call gemini ",
            "Force Gemini 2.0 Flash via Google.\nUsage: /call gemini <message>",
            False,
        ),
    ]

    TRAINING_BUTTONS = [
        (
            "/Run BlackLoRA-N",
            "/run_lora",
            "Run LoRA fine-tuning on completed Blackwell interview data.\n"
            "(Requires 200+ training pairs — check /blackwell first.)",
            True,
        ),
    ]

    SESSION_BUTTONS = [
        (
            "/exit",
            "/exit",
            "Quit Zephyr.",
            True,
        ),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(200)
        self.setMaximumWidth(300)
        self.setStyleSheet("background-color: #090c10;")

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical {
                background: #0d1117; width: 6px; border: none;
            }
            QScrollBar::handle:vertical {
                background: #3a5a6a; border-radius: 3px; min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #4dcdb4;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)

        inner = QWidget()
        inner.setStyleSheet("background: transparent;")
        vbox = QVBoxLayout(inner)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(4)

        hdr = QLabel("COMMAND PALETTE")
        hdr.setStyleSheet("""
            color: rgba(170,182,194,0.4);
            font-family: Consolas, monospace;
            font-size: 9px;
            letter-spacing: 3px;
            padding: 4px 4px 8px 4px;
        """)
        vbox.addWidget(hdr)

        def add_group(buttons, capture=None):
            for label, cmd, tip, fire in buttons:
                btn = ZephyrButton(label, cmd, tip, fire)
                btn.clicked.connect(
                    lambda checked=False, c=cmd, f=fire:
                        self.command_requested.emit(c, f)
                )
                vbox.addWidget(btn)
                if capture is not None and cmd in capture:
                    capture[cmd] = btn

        _caps = {"/blackwell": None, "/coding-blackwell": None, "/trajectory": None}
        add_group(self.BUTTONS, capture=_caps)
        self._blackwell_btn  = _caps["/blackwell"]
        self._coding_btn     = _caps["/coding-blackwell"]
        self._trajectory_btn = _caps["/trajectory"]
        vbox.addWidget(SectionDivider("Keys"))
        add_group(self.KEYS_BUTTONS)
        vbox.addWidget(SectionDivider("External AI"))
        add_group(self.CALL_BUTTONS)
        vbox.addWidget(SectionDivider("Training"))
        add_group(self.TRAINING_BUTTONS)
        vbox.addWidget(SectionDivider("Session"))
        add_group(self.SESSION_BUTTONS)
        vbox.addStretch()

        # Live pair-count badge refresh
        from PySide6.QtCore import QTimer
        self._badge_timer = QTimer(self)
        self._badge_timer.timeout.connect(self._refresh_badges)
        self._badge_timer.start(30_000)
        QTimer.singleShot(0, self._refresh_badges)  # immediate on startup

        scroll.setWidget(inner)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _refresh_badges(self):
        """Read JSONL line counts from disk and update button badges."""
        try:
            import os

            def _count(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        return sum(1 for ln in f if ln.strip())
                except FileNotFoundError:
                    return 0

            base = os.path.dirname(os.path.abspath(__file__))
            bw  = _count(os.path.join(base, "blackwell", "training_pairs.jsonl"))
            cbw = _count(os.path.join(base, "blackwell", "coding_training_pairs.jsonl"))
            traj = _count(os.path.join(base, "trajectory_samples.jsonl"))

            if self._blackwell_btn:
                self._blackwell_btn.set_badge(f"{bw} pairs" if bw else "")
            if self._coding_btn:
                self._coding_btn.set_badge(f"{cbw} pairs" if cbw else "")
            if self._trajectory_btn:
                self._trajectory_btn.set_badge(f"{traj} pairs" if traj else "")
        except Exception as exc:
            import sys
            print(f"[badge refresh] error: {exc}", file=sys.stderr, flush=True)


# ═══════════════════════════════════════════════════════════════
#  MainWindow
# ═══════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zephyr — Prycat Research")
        self.resize(1100, 700)
        self.setMinimumSize(800, 500)
        self.setMouseTracking(True)

        # Remove OS chrome — our ZephyrTitleBar takes over
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Window
        )

        # ── Central widget ────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Custom title bar (replaces HeaderBar)
        self._header = ZephyrTitleBar()
        root.addWidget(self._header)

        # Splitter: console left | palette right
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background: #1a2030;
                width: 2px;
            }
        """)

        # Left pane: console + input row
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self._console = ConsoleWidget()
        left_layout.addWidget(self._console)

        # Telemetry bar — shown while Zephyr is generating
        self._thinking_bar = ThinkingBar()
        left_layout.addWidget(self._thinking_bar)

        # Load saved config and apply to ThinkingBar
        _cfg = load_zephyr_config()
        self._active_model = _cfg.get("active_model", "hermes3:8b")
        self._tq_enabled = _cfg.get("turboquant_enabled", False)
        self._thinking_bar.set_active_model(self._active_model)
        self._thinking_bar.set_turboquant(self._tq_enabled)

        # Model switcher card — singleton, repositioned on each open
        self._model_card = ModelSwitcherCard()
        self._model_card.model_selected.connect(self._on_model_selected)
        self._model_card.turboquant_toggled.connect(self._on_turboquant_toggled)
        self._thinking_bar.model_cell_clicked.connect(self._show_model_card)

        # Blackwell config card — Fix B / D / E operator controls
        self._blackwell_card = BlackwellConfigCard()
        self._thinking_bar.blackwell_cell_clicked.connect(self._show_blackwell_card)

        # Oracle model switcher card — cell 2
        self._oracle_model = _cfg.get("oracle_model", "hermes3:8b")
        self._oracle_card = OracleSwitcherCard()
        self._oracle_card.oracle_selected.connect(self._on_oracle_selected)
        self._thinking_bar.oracle_cell_clicked.connect(self._show_oracle_card)

        # Model download card — shared by both model pickers
        self._download_card = ModelDownloadCard()
        self._download_card.download_complete.connect(self._on_download_complete)
        self._model_card.add_model_clicked.connect(self._show_download_card)
        self._oracle_card.add_model_clicked.connect(self._show_download_card)

        # Input row
        input_row = QWidget()
        input_row.setStyleSheet(
            "background: #0d1117; border-top: 1px solid rgba(255,255,255,0.06);"
        )
        input_row_layout = QHBoxLayout(input_row)
        input_row_layout.setContentsMargins(8, 6, 8, 6)
        input_row_layout.setSpacing(6)

        self._input = InputBar()
        input_row_layout.addWidget(self._input)

        send_btn = QPushButton("SEND")
        send_btn.setFixedWidth(64)
        send_btn.setFixedHeight(34)
        send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        send_btn.setStyleSheet("""
            QPushButton {
                background: rgba(77,194,179,0.12);
                color: rgba(128,221,202,0.9);
                border: 1px solid rgba(77,194,179,0.25);
                border-radius: 4px;
                font-family: Consolas, monospace;
                font-size: 10px;
                letter-spacing: 2px;
            }
            QPushButton:hover {
                background: rgba(77,194,179,0.22);
                border-color: rgba(77,194,179,0.45);
            }
            QPushButton:pressed {
                background: rgba(77,194,179,0.08);
            }
        """)
        send_btn.clicked.connect(self._input._fire)
        input_row_layout.addWidget(send_btn)
        left_layout.addWidget(input_row)

        # Right pane: palette
        self._palette = PaletteWidget()

        splitter.addWidget(left_widget)
        splitter.addWidget(self._palette)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([740, 360])
        root.addWidget(splitter)

        # ── Dragon splash directly in console ────────────────
        _dragon_splash_into_console(self._console)

        # ── Wire subprocess ───────────────────────────────────
        self._process = ZephyrProcess(self)
        self._process.output_signal.connect(self._console.append_line)
        self._process.finished_signal.connect(self._on_agent_exit)
        self._process.stream_started.connect(
            self._thinking_bar.start,
            Qt.ConnectionType.QueuedConnection,
        )
        self._process.stream_ended.connect(
            self._thinking_bar.stop,
            Qt.ConnectionType.QueuedConnection,
        )
        self._process.token_gap.connect(
            self._thinking_bar.record_token_gap,
            Qt.ConnectionType.QueuedConnection,
        )

        # Feedback state
        self._current_session_id: str = ""
        self._current_turn: int = 0

        self._process.output_signal.connect(self._on_agent_line)
        self._process.stream_ended.connect(
            self._on_response_complete,
            Qt.ConnectionType.QueuedConnection,
        )

        self._process.start()

        # ── Wire input → process ──────────────────────────────
        self._input.submitted.connect(self._on_user_input)

        # ── Wire palette → input ──────────────────────────────
        self._palette.command_requested.connect(self._on_command_requested)

    def _on_user_input(self, text: str):
        self._console.append_line(f"You: {text}")
        self._current_turn += 1
        self._thinking_bar.set_loading()
        self._process.send_input(text)

    def _on_command_requested(self, command: str, fire: bool):
        self._input.inject(command, fire)

    def _on_agent_exit(self):
        self._console.append_line("─── Zephyr process ended ───")
        self._thinking_bar.stop()   # clear LOADING/THINKING if process died mid-stream
        self._current_turn = 0
        self._current_session_id = ""

    def _on_agent_line(self, line: str):
        """Parse special markers from agent stdout."""
        if line.startswith("<<SESSION:") and line.endswith(">>"):
            self._current_session_id = line[10:-2]

    def _on_response_complete(self):
        """Show thumbs feedback bar briefly after each response."""
        if not self._current_session_id:
            return
        # Must parent to viewport() — QPlainTextEdit's visible area is a child
        # viewport widget; widgets parented to the scroll-area frame are hidden behind it.
        vp = self._console.viewport()
        bar = FeedbackBar(parent=vp)
        bar.feedback_given.connect(self._on_feedback)
        # Position bottom-right of the viewport
        bar.adjustSize()
        x = vp.width() - bar.width() - 8
        y = vp.height() - bar.height() - 4
        bar.move(x, y)
        bar.show()
        bar.raise_()
        # Auto-hide after 8s if no vote
        bar._auto_hide_timer.start(8000)

    def _on_feedback(self, positive: bool):
        """Forward thumbs vote to agent via /feedback command."""
        vote = "up" if positive else "down"
        self._process.send_input(
            f"/feedback {self._current_session_id} {self._current_turn} {vote}"
        )

    def _show_model_card(self):
        """Position and show ModelSwitcherCard above ThinkingBar cell 0."""
        cell_rect = self._thinking_bar._cell0_rect()
        global_pos = self._thinking_bar.mapToGlobal(cell_rect.topLeft())
        # Position card so bottom-left aligns with cell top-left
        card_h = self._model_card.height() if self._model_card.height() > 0 else 200
        card_pos = QPoint(global_pos.x(), global_pos.y() - card_h - 4)
        # Clamp to screen
        screen = QApplication.primaryScreen().availableGeometry()
        card_pos.setY(max(screen.top(), card_pos.y()))
        self._model_card.show_at(card_pos, self._active_model, self._tq_enabled)

    def _show_blackwell_card(self):
        """Position and show BlackwellConfigCard above ThinkingBar cell 1."""
        cell_rect  = self._thinking_bar._cell1_rect()
        global_pos = self._thinking_bar.mapToGlobal(cell_rect.topLeft())
        self._blackwell_card.adjustSize()
        card_h = self._blackwell_card.height() or 480
        card_w = self._blackwell_card.width()
        card_pos = QPoint(global_pos.x(), global_pos.y() - card_h - 4)
        # Clamp horizontally so it doesn't run off the right edge
        screen = QApplication.primaryScreen().availableGeometry()
        card_pos.setY(max(screen.top(), card_pos.y()))
        card_pos.setX(min(card_pos.x(), screen.right() - card_w - 4))
        self._blackwell_card.show_at(card_pos)

    def _show_download_card(self):
        """Show ModelDownloadCard centred above the ThinkingBar."""
        bar_rect   = self._thinking_bar.geometry()
        global_top = self._thinking_bar.mapToGlobal(QPoint(0, 0)).y()
        card_w     = ModelDownloadCard._WIDTH
        card_h     = self._download_card.sizeHint().height() or 240
        cx         = self._thinking_bar.mapToGlobal(QPoint(bar_rect.width() // 2, 0)).x()
        pos        = QPoint(cx - card_w // 2, global_top - card_h - 4)
        screen     = QApplication.primaryScreen().availableGeometry()
        pos.setY(max(screen.top(), pos.y()))
        pos.setX(max(screen.left() + 4, min(pos.x(), screen.right() - card_w - 4)))
        self._download_card.show_at(pos)

    def _on_download_complete(self):
        """Refresh both open pickers after a successful pull."""
        # Re-fetch model lists so new model appears immediately
        if self._model_card.isVisible():
            self._model_card.show_at(
                self._model_card.pos(),
                self._active_model,
                self._tq_enabled,
            )
        if self._oracle_card.isVisible():
            self._oracle_card.show_at(
                self._oracle_card.pos(),
                self._oracle_model,
            )

    def _show_oracle_card(self):
        """Position and show OracleSwitcherCard above ThinkingBar cell 2."""
        cell_rect  = self._thinking_bar._cell2_rect()
        global_pos = self._thinking_bar.mapToGlobal(cell_rect.topLeft())
        card_h = self._oracle_card.height() if self._oracle_card.height() > 0 else 200
        card_pos = QPoint(global_pos.x(), global_pos.y() - card_h - 4)
        screen = QApplication.primaryScreen().availableGeometry()
        card_pos.setY(max(screen.top(), card_pos.y()))
        card_pos.setX(min(card_pos.x(), screen.right() - OracleSwitcherCard._WIDTH - 4))
        self._oracle_card.show_at(card_pos, self._oracle_model)

    def _on_oracle_selected(self, model_name: str):
        """Persist the chosen oracle model to config."""
        self._oracle_model = model_name
        cfg = load_zephyr_config()
        cfg["oracle_model"] = model_name
        save_zephyr_config(cfg)

    def _on_model_selected(self, model_name: str):
        """Switch active model in agent and persist."""
        self._active_model = model_name
        self._thinking_bar.set_active_model(model_name)
        cfg = load_zephyr_config()
        cfg["active_model"] = model_name
        save_zephyr_config(cfg)
        self._process.send_input(f"/model {model_name}")

    def _on_turboquant_toggled(self, enabled: bool):
        self._tq_enabled = enabled
        self._thinking_bar.set_turboquant(enabled)
        cfg = load_zephyr_config()
        cfg["turboquant_enabled"] = enabled
        save_zephyr_config(cfg)

    _RESIZE_MARGIN = 8

    def _enable_native_resize(self):
        """Add WS_THICKFRAME to the HWND so Windows handles edge-resize natively.

        FramelessWindowHint removes the OS chrome but also strips WS_THICKFRAME,
        which is what gives a window its resizable border.  Re-adding it via the
        Windows API restores resize cursors and drag-resize.

        WS_THICKFRAME also creates an invisible non-client border that would eat
        scroll events — nativeEvent handles WM_NCCALCSIZE to collapse that border
        back to zero while keeping the thick-frame resize hit-testing.
        """
        import ctypes
        GWL_STYLE     = -16
        WS_THICKFRAME = 0x00040000
        SWP_FLAGS     = 0x0001 | 0x0002 | 0x0004 | 0x0010 | 0x0020

        hwnd   = int(self.winId())
        user32 = ctypes.windll.user32
        style  = user32.GetWindowLongW(hwnd, GWL_STYLE)
        user32.SetWindowLongW(hwnd, GWL_STYLE, style | WS_THICKFRAME)
        user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, SWP_FLAGS)

    def nativeEvent(self, event_type, message):
        """Handle WM_NCCALCSIZE and WM_NCHITTEST for frameless resize.

        WS_THICKFRAME gives Windows resize hit-testing but also creates an
        invisible non-client border that steals scroll/click events.
        WM_NCCALCSIZE → 0  collapses that border so the entire rect is client
        area (scroll works everywhere).
        WM_NCHITTEST returns the correct HTXXX zone so Windows shows the right
        resize cursor and handles the actual drag.

        Fields are read at exact byte offsets rather than via a ctypes struct
        so there is no struct-alignment crash risk.
        Offsets on 64-bit Windows MSG:
          +0  hwnd    (8 bytes)
          +8  message (4 bytes)
          +12 pad     (4 bytes implicit)
          +16 wParam  (8 bytes)
          +24 lParam  (8 bytes)
        Offsets on 32-bit:
          +0  hwnd    (4 bytes)
          +4  message (4 bytes)
          +8  wParam  (4 bytes)
          +12 lParam  (4 bytes)
        """
        if bytes(event_type) == b"windows_generic_MSG":
            import ctypes, sys
            from PySide6.QtCore import QPoint

            WM_NCCALCSIZE = 0x0083
            WM_NCHITTEST  = 0x0084
            HTTOP         = 12
            HTTOPLEFT     = 13
            HTTOPRIGHT    = 14
            HTBOTTOM      = 15
            HTBOTTOMLEFT  = 16
            HTBOTTOMRIGHT = 17
            HTLEFT        = 10
            HTRIGHT       = 11

            addr = int(message)
            is64 = sys.maxsize > 2**32

            # Read message field at known offset (no struct needed)
            msg_offset  = 8 if is64 else 4
            lp_offset   = 24 if is64 else 12
            lp_type     = ctypes.c_int64 if is64 else ctypes.c_int32

            try:
                msg_id = ctypes.c_uint32.from_address(addr + msg_offset).value
            except Exception:
                return super().nativeEvent(event_type, message)

            if msg_id == WM_NCCALCSIZE:
                # Zero non-client area — entire window rect becomes client area
                return True, 0

            if msg_id == WM_NCHITTEST:
                try:
                    lp = lp_type.from_address(addr + lp_offset).value
                except Exception:
                    return super().nativeEvent(event_type, message)

                # Signed 16-bit screen coords packed into lParam
                sx = ctypes.c_int16(lp & 0xFFFF).value
                sy = ctypes.c_int16((lp >> 16) & 0xFFFF).value
                local  = self.mapFromGlobal(QPoint(sx, sy))
                x, y   = local.x(), local.y()
                w, h   = self.width(), self.height()
                m      = self._RESIZE_MARGIN
                left   = x <= m
                right  = x >= w - m
                top    = y <= m
                bottom = y >= h - m
                if top    and left:  return True, HTTOPLEFT
                if top    and right: return True, HTTOPRIGHT
                if bottom and left:  return True, HTBOTTOMLEFT
                if bottom and right: return True, HTBOTTOMRIGHT
                if top:              return True, HTTOP
                if bottom:           return True, HTBOTTOM
                if left:             return True, HTLEFT
                if right:            return True, HTRIGHT

        return super().nativeEvent(event_type, message)

    def closeEvent(self, event):
        self._process.stop()
        self._process.wait(2000)
        super().closeEvent(event)


# ═══════════════════════════════════════════════════════════════
#  Global stylesheet
# ═══════════════════════════════════════════════════════════════
GLOBAL_QSS = """
QMainWindow, QWidget {
    background-color: #090c10;
    color: rgba(128,221,202,0.92);
    font-family: Consolas, monospace;
}

QToolTip {
    background-color: #0d1117;
    color: rgba(128,221,202,0.9);
    border: 1px solid rgba(77,194,179,0.3);
    font-family: Consolas, monospace;
    font-size: 10px;
    padding: 6px 10px;
    border-radius: 4px;
}

QSplitter::handle {
    background: #1a2030;
}
"""


# ═══════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Zephyr")
    app.setStyleSheet(GLOBAL_QSS)

    # Force tooltip colours via palette — Windows ignores QSS on QToolTip
    pal = app.palette()
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor("#0d1117"))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(128, 221, 202))
    app.setPalette(pal)

    from PySide6.QtWidgets import QToolTip
    QToolTip.setPalette(pal)
    QToolTip.setFont(QFont("Consolas", 9))


    window = MainWindow()
    window.show()
    window._enable_native_resize()   # must be after show() so winId() is valid
    sys.exit(app.exec())
