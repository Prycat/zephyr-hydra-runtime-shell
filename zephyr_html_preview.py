# -*- coding: utf-8 -*-
"""
zephyr_html_preview.py — HTML preview pane for Zephyr.

HtmlPreviewPane: a QWidget containing a header bar and an embedded
QWebEngineView. Inserted as the right half of a QSplitter inside the
Zephyr console pane. Starts hidden (width=0); expands when the agent
produces a complete ```html block.

Requires: pip install PySide6-WebEngine
Graceful degradation: if PySide6-WebEngine is not installed the pane
shows a styled placeholder and the rest of Zephyr is unaffected.
"""
from __future__ import annotations
import re
import sys

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
)

__all__ = ["is_webengine_available", "extract_last_html_block", "HtmlPreviewPane"]


# ── Availability check ────────────────────────────────────────────────────────

def is_webengine_available() -> bool:
    """Return True if PySide6-WebEngine is installed."""
    try:
        from PySide6.QtWebEngineWidgets import QWebEngineView  # noqa: F401
        return True
    except ImportError:
        return False


# ── HTML extractor ────────────────────────────────────────────────────────────

_HTML_BLOCK_RE = re.compile(r'```html\s*\n(.*?)^```', re.DOTALL | re.MULTILINE)


def extract_last_html_block(text: str) -> str | None:
    """
    Find the last complete ```html ... ``` block in *text*.

    Returns the inner HTML string (stripped), or None if no complete block
    is present.

    The closing fence must appear at the start of a line.
    CRLF line endings are normalised to LF before matching.
    """
    text = text.replace('\r\n', '\n')
    matches = _HTML_BLOCK_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


# ── Colour constants (match Zephyr palette) ───────────────────────────────────

_C_BG     = "#090c10"
_C_HEADER = "#0d1117"
_C_TEAL   = "#80ddca"
_C_DIM    = "#445566"


# ── HtmlPreviewPane ───────────────────────────────────────────────────────────

class HtmlPreviewPane(QWidget):
    """
    Right-hand pane of the console sub-splitter.

    Header bar: "◈ PREVIEW"  [↺]  [✕]
    Body: QWebEngineView (sandboxed) or a fallback label if WebEngine missing.

    Signals
    -------
    close_requested : emitted when ✕ is clicked
    """

    close_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(0)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header bar ────────────────────────────────────────
        header = QWidget()
        header.setFixedHeight(32)
        header.setStyleSheet(
            f"background: {_C_HEADER};"
            "border-bottom: 1px solid rgba(255,255,255,0.06);"
        )
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(10, 0, 6, 0)
        h_layout.setSpacing(4)

        lbl = QLabel("◈  PREVIEW")
        lbl.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        lbl.setStyleSheet(f"color: {_C_TEAL}; background: transparent;")
        h_layout.addWidget(lbl)
        h_layout.addStretch()

        _btn_style = (
            "QPushButton {"
            "  background: transparent;"
            f"  color: {_C_DIM};"
            "  border: none;"
            "  font-family: Consolas;"
            "  font-size: 13px;"
            "  padding: 2px 6px;"
            "}"
            f"QPushButton:hover {{ color: {_C_TEAL}; }}"
        )

        self._reload_btn = QPushButton("↺")
        self._reload_btn.setToolTip("Reload preview")
        self._reload_btn.setStyleSheet(_btn_style)
        self._reload_btn.setFixedSize(28, 24)
        self._reload_btn.clicked.connect(self._on_reload)
        h_layout.addWidget(self._reload_btn)

        close_btn = QPushButton("✕")
        close_btn.setToolTip("Close preview")
        close_btn.setStyleSheet(_btn_style)
        close_btn.setFixedSize(28, 24)
        close_btn.clicked.connect(self.close_requested.emit)
        h_layout.addWidget(close_btn)

        root.addWidget(header)

        # ── Body ──────────────────────────────────────────────
        if is_webengine_available():
            from PySide6.QtWebEngineWidgets import QWebEngineView
            from PySide6.QtWebEngineCore import QWebEngineSettings

            self._view = QWebEngineView()

            # Sandbox: disable local file access and external navigation
            s = self._view.settings()
            s.setAttribute(
                QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls,
                False,
            )
            s.setAttribute(
                QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls,
                False,
            )
            s.setAttribute(
                QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows,
                False,
            )
            # NOTE: JavascriptEnabled is intentionally left ON — canvas animations
            # (the primary use case) require JS.  The other three settings provide
            # meaningful isolation without breaking animation support.

            self._view.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            root.addWidget(self._view)
            self._has_webengine = True
        else:
            fallback = QLabel(
                "HTML preview unavailable.\n\n"
                "pip install PySide6-WebEngine"
            )
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback.setFont(QFont("Consolas", 9))
            fallback.setStyleSheet(
                f"color: {_C_DIM}; background: {_C_BG};"
            )
            root.addWidget(fallback)
            self._has_webengine = False
            self._reload_btn.setEnabled(False)
            self._reload_btn.setToolTip("WebEngine not available")

        self._current_html: str = ""

    # ── Public API ────────────────────────────────────────────

    def render(self, html: str) -> None:
        """Render *html* in the preview. No-op if WebEngine is not available."""
        self._current_html = html
        if self._has_webengine:
            self._view.setHtml(html, QUrl("about:blank"))

    def clear(self) -> None:
        """Clear the preview to a blank page."""
        self._current_html = ""
        if self._has_webengine:
            self._view.setHtml("<html><body></body></html>", QUrl("about:blank"))

    # ── Internal ──────────────────────────────────────────────

    def _on_reload(self) -> None:
        if self._current_html:
            self.render(self._current_html)
        # else: nothing to reload after clear() — intentional no-op
