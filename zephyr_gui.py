"""
zephyr_gui.py — Zephyr Command Workbench
Prycat Research Team
PySide6 GUI wrapping agent.py via subprocess pipe.
Python 3.9 compatible.
"""
import sys
import math
import subprocess
import queue
import threading
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

    _SENTINEL = object()   # signals the input queue to stop

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc      = None      # type: Optional[subprocess.Popen]
        self._lock      = threading.Lock()
        self._input_q   = queue.Queue()   # GUI → worker thread

    def run(self):
        try:
            proc = subprocess.Popen(
                [sys.executable, AGENT_PATH],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
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

            for line in proc.stdout:
                self.output_signal.emit(line.rstrip("\n"))

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
