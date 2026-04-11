"""
zephyr_gui.py — Zephyr Command Workbench
Prycat Research Team
PySide6 GUI wrapping agent.py via subprocess pipe.
Python 3.9 compatible.
"""
import sys
import math
import subprocess
from typing import Optional

from PySide6.QtCore import (
    Qt, QThread, Signal, QTimer, QPointF, QRectF
)
from PySide6.QtGui import (
    QColor, QPainter, QPen, QBrush, QRadialGradient,
    QLinearGradient, QFont, QPalette, QFontDatabase
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPlainTextEdit, QLineEdit, QPushButton,
    QSplitter, QScrollArea, QLabel, QSizePolicy, QFrame
)

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
#  ZephyrProcess — subprocess thread
# ═══════════════════════════════════════════════════════════════
class ZephyrProcess(QThread):
    output_signal   = Signal(str)
    finished_signal = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc = None   # type: Optional[subprocess.Popen]

    def run(self):
        self._proc = subprocess.Popen(
            [sys.executable, AGENT_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        for line in self._proc.stdout:
            self.output_signal.emit(line.rstrip("\n"))
        self.finished_signal.emit()

    def send_input(self, text: str):
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.write(text + "\n")
                self._proc.stdin.flush()
            except OSError:
                pass

    def stop(self):
        if self._proc:
            try:
                self._proc.terminate()
            except OSError:
                pass
