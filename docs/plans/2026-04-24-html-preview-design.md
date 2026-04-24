# Zephyr HTML Preview Panel — Design

**Date:** 2026-04-24

## Goal

When the agent generates a complete `\`\`\`html` code block, the Zephyr console area
splits into two side-by-side panes: the raw streamed text on the left (unchanged),
and a live rendered HTML preview on the right — matching the ChatGPT artifacts UX.

## Architecture

A new horizontal `QSplitter` (`_console_splitter`) wraps the existing `ConsoleWidget`
and a new `HtmlPreviewPane` widget. The preview pane starts at width=0 (hidden). When
`stream_ended` fires, `_on_response_complete()` scans the console text for the last
complete ` ```html … ``` ` block; if found it calls `setHtml()` on the embedded
`QWebEngineView` and expands the splitter to 50/50.

The preview renders in an isolated `QWebEngineView` with file-access and remote-URL
access disabled. No data leaves the machine.

## Tech Stack

- PySide6 (already present) + **PySide6-WebEngine** (new dep, `pip install PySide6-WebEngine`)
- `QWebEngineView` / `QWebEngineSettings` for sandboxed in-process HTML rendering
- `re` for `\`\`\`html` block extraction (stdlib)

## Components

### `zephyr_html_preview.py` (new file)

- `is_webengine_available() -> bool` — try-import guard
- `extract_last_html_block(text: str) -> str | None` — regex over full console text
- `HtmlPreviewPane(QWidget)` — header bar (◈ PREVIEW · ↺ · ✕) + `QWebEngineView` body
  - `render(html: str)` — calls `view.setHtml(html)`, shows pane
  - `clear()` — loads blank page
  - Signal `close_requested` — emitted when ✕ clicked

### `zephyr_gui.py` changes

- `MainWindow.__init__`: replace `left_layout.addWidget(self._console)` with
  `self._console_splitter` (QSplitter) containing console + `HtmlPreviewPane`
- `_on_response_complete()`: call `self._try_show_preview()` after feedback bar logic
- New `_try_show_preview()`: extract HTML, render, expand splitter
- New `_close_preview()`: collapse splitter back to `[full, 0]`

## Graceful Degradation

If `PySide6-WebEngine` is not installed, `HtmlPreviewPane` renders a styled
placeholder label: `"pip install PySide6-WebEngine to enable HTML preview"`.
The rest of Zephyr is completely unaffected.
