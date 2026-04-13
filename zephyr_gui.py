# -*- coding: utf-8 -*-
"""
zephyr_gui.py — Zephyr Command Workbench
Prycat Research Team
PySide6 GUI wrapping agent.py via subprocess pipe.
Python 3.9 compatible.
"""
import sys
import math
import html
import random
import textwrap
import subprocess
import queue
import threading
from collections import deque
from typing import Optional

from PySide6.QtCore import (
    Qt, QThread, Signal, QTimer, QPointF, QRectF
)
from PySide6.QtGui import (
    QColor, QPainter, QPen, QBrush, QRadialGradient,
    QLinearGradient, QFont, QPalette, QFontDatabase,
    QTextCursor, QTextCharFormat, QPainterPath,
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
        '<span style="white-space:pre;font-family:Consolas,monospace;font-size:13pt;'
        'color:#4dcdb4;font-weight:bold;letter-spacing:8px;">'
        '          Z  E  P  H  Y  R</span>'
    )
    console_widget.appendHtml(
        '<span style="white-space:pre;font-family:Consolas,monospace;font-size:8.5pt;'
        'color:#2a7a5a;">    Prycat Research  \xb7  local intelligence  \xb7  BlackLoRA-N core</span>'
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

            import time as _time
            last_tok_t = None

            for line in proc.stdout:
                stripped = line.rstrip("\n")
                if stripped == "<<ZS>>":
                    last_tok_t = _time.monotonic()
                    self.stream_started.emit()
                elif stripped == "<<ZE>>":
                    last_tok_t = None
                    self.stream_ended.emit()
                elif stripped.startswith("\x01"):
                    now = _time.monotonic()
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

    def append_line(self, line: str):
        """Colorize and append one line from the agent."""
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
    _CELL_LABELS = ["PARSE",  "/BLACKWELL",  "COMMIT",   "LOAD"]
    _CELL_VALUES = ["weighted", "vector accrual", "branch sel.", "inertia cls-v"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(self.HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)

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
          X (columns, ix 0→19): token position, oldest left → newest right
          Z (rows,    iz 0→2):  smoothing lane
            iz=0  raw inter-token gap (jagged front)
            iz=1  5-sample rolling average (mid depth)
            iz=2  7-sample rolling average (smooth back)
        """
        GRID_X, GRID_Z = 20, 3
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
                    # Wider smoothing kernel for deeper Z lanes:
                    # iz=1 → kernel=4 (half=2), iz=2 → kernel=7 (half=3)
                    half    = 1 + iz
                    centre  = int(t * (N - 1))
                    samples = [gaps[max(0, min(N - 1, centre + off))]
                               for off in range(-half, half + 1)]
                    h = sum(samples) / len(samples)

                result[iz][ix] = min(1.0, h / norm)

        return result

    def _collapse(self):
        pass   # kept for compatibility, no longer used

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
            # X = token position bucket (0=oldest, 19=newest)
            # Z = smoothing lane (0=raw, 1=5-avg, 2=7-avg)
            # Y = normalized inter-token gap (tall spike = slow token)
            VH = 36                 # slightly taller than old 32
            GRID_X, GRID_Z = 20, 3
            CELL_X  =  8.5          # world units per X column
            CELL_Z  = 14.0          # world units per Z lane
            MAX_H   = 26.0          # world units when gap = 1.0
            _YAW    = 0.78          # horizontal rotation (≈45°)
            _PITCH  = -0.46         # tilt downward
            _CAM    = 340.0         # perspective camera distance
            _BIAS   = 230.0         # depth bias

            _cosY = math.cos(_YAW);  _sinY = math.sin(_YAW)
            _cosP = math.cos(_PITCH); _sinP = math.sin(_PITCH)

            # Projection centre: slightly below vertical centre of the panel
            _pcx = rx + RIGHT_W * 0.5
            _pcy = ry + VH * 0.64

            # Compute height field (0–1 normalised)
            _hf = self._surface_heights()

            # READY/LOADING: gentle slow pulse keeps terrain alive when idle
            if not self._active:
                _pulse = 0.06 * (0.5 + 0.5 * math.sin(self._frame * 0.025))
                _hf = [[min(1.0, v + _pulse) for v in row] for row in _hf]

            # Build projected quads
            _sig_c = self._sig()
            _quads = []
            _half_x = (GRID_X - 1) / 2.0
            _half_z = (GRID_Z - 1) / 2.0
            for iz in range(GRID_Z - 1):
                for ix in range(GRID_X - 1):
                    wx0 = (ix     - _half_x) * CELL_X
                    wx1 = (ix + 1 - _half_x) * CELL_X
                    wz0 = (iz     - _half_z) * CELL_Z
                    wz1 = (iz + 1 - _half_z) * CELL_Z
                    # Negative Y: screen-Y increases downward, world-Y up
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

            # Painter's algorithm: back-to-front depth sort
            _quads.sort(key=lambda q: q[0])

            # Pass 1: filled + wireframe quads
            for _dep, _ah, _a, _b, _c, _d in _quads:
                _inten  = min(1.0, _ah / MAX_H)
                _falpha = int(14 + _inten * 72)
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
                _salpha = int(8 + _inten * 38)
                p.setPen(QPen(QColor(200, 230, 255, _salpha), 0.6))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPath(_path)

            # Pass 2: radial glow on high peaks
            for _dep, _ah, _a, _b, _c, _d in _quads:
                _inten = min(1.0, _ah / MAX_H)
                if _inten < 0.28:
                    continue
                _qcx = (_a[0] + _b[0] + _c[0] + _d[0]) * 0.25
                _qcy = (_a[1] + _b[1] + _c[1] + _d[1]) * 0.25
                _rg  = 3.5 + _inten * 5.5
                _gg  = QRadialGradient(QPointF(_qcx, _qcy), _rg)
                _gg.setColorAt(0, self._sig(int(55 * _inten)))
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
#  HeaderBar
# ═══════════════════════════════════════════════════════════════
class HeaderBar(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        self.setStyleSheet("background-color: #0d1117;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(10)

        # Pulsing dot
        self._dot = QLabel("●")
        self._dot.setStyleSheet("color: rgba(102,196,122,0.95); font-size: 10px;")
        layout.addWidget(self._dot)

        # ZEPHYR label
        title = QLabel("ZEPHYR")
        title.setStyleSheet("""
            color: rgba(128,221,202,0.92);
            font-family: Consolas, monospace;
            font-size: 13px;
            font-weight: bold;
            letter-spacing: 4px;
        """)
        layout.addWidget(title)

        # Model name
        model_lbl = QLabel(MODEL_NAME)
        model_lbl.setStyleSheet("""
            color: rgba(170,182,194,0.5);
            font-family: Consolas, monospace;
            font-size: 10px;
            letter-spacing: 2px;
        """)
        layout.addWidget(model_lbl)
        layout.addStretch()

        # Dot pulse timer
        self._dot_on = True
        dot_timer = QTimer(self)
        dot_timer.setInterval(750)
        dot_timer.timeout.connect(self._pulse_dot)
        dot_timer.start()

    def _pulse_dot(self):
        self._dot_on = not self._dot_on
        alpha = "0.95" if self._dot_on else "0.35"
        self._dot.setStyleSheet(
            f"color: rgba(102,196,122,{alpha}); font-size: 10px;"
        )


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
            "/coding-blackwell",
            "/coding-blackwell",
            "CS-focused planning session — Zephyr interviews you on coding habits,\n"
            "languages, and problems. Sharpens his coding instincts permanently.",
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

        def add_group(buttons):
            for label, cmd, tip, fire in buttons:
                btn = ZephyrButton(label, cmd, tip, fire)
                btn.clicked.connect(
                    lambda checked=False, c=cmd, f=fire:
                        self.command_requested.emit(c, f)
                )
                vbox.addWidget(btn)

        add_group(self.BUTTONS)
        vbox.addWidget(SectionDivider("Keys"))
        add_group(self.KEYS_BUTTONS)
        vbox.addWidget(SectionDivider("External AI"))
        add_group(self.CALL_BUTTONS)
        vbox.addWidget(SectionDivider("Training"))
        add_group(self.TRAINING_BUTTONS)
        vbox.addWidget(SectionDivider("Session"))
        add_group(self.SESSION_BUTTONS)
        vbox.addStretch()

        scroll.setWidget(inner)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)


# ═══════════════════════════════════════════════════════════════
#  MainWindow
# ═══════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zephyr — Prycat Research")
        self.resize(1100, 700)
        self.setMinimumSize(800, 500)

        # ── Central widget ────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        self._header = HeaderBar()
        root.addWidget(self._header)

        # Thin divider line
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet("color: rgba(255,255,255,0.06); margin: 0;")
        root.addWidget(div)

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
        self._process.stream_started.connect(self._thinking_bar.start)
        self._process.stream_ended.connect(self._thinking_bar.stop)
        self._process.token_gap.connect(
            self._thinking_bar.record_token_gap,
            Qt.ConnectionType.QueuedConnection,
        )
        self._process.start()

        # ── Wire input → process ──────────────────────────────
        self._input.submitted.connect(self._on_user_input)

        # ── Wire palette → input ──────────────────────────────
        self._palette.command_requested.connect(self._on_command_requested)

    def _on_user_input(self, text: str):
        self._console.append_line(f"You: {text}")
        self._thinking_bar.set_loading()
        self._process.send_input(text)

    def _on_command_requested(self, command: str, fire: bool):
        self._input.inject(command, fire)

    def _on_agent_exit(self):
        self._console.append_line("─── Zephyr process ended ───")

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
    sys.exit(app.exec())
