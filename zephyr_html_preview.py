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

__all__ = ["is_webengine_available", "extract_last_html_block"]


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
