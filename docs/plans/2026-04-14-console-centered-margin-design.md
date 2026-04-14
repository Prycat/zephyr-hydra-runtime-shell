# Console Centered Margin — Design Doc

**Date:** 2026-04-14  
**Status:** Approved

## Problem

On wide monitors the Zephyr console fills the entire left splitter pane (~740–1200px+), making lines of text very long and hard to read. The ASCII dragon art and conversation text sprawl across the full width.

## Goal

Constrain the console text column to a readable max-width and center it within the left pane, with dark gutters filling the remaining space — similar to how Warp or a markdown reader handles wide viewports.

## Approach

**Approach 1 — `setMaximumWidth` + centering stretch** (selected)

Wrap the `ConsoleWidget`, `ThinkingBar`, and `input_row` in a `content_col` widget with `setMaximumWidth(_CONSOLE_MAX_WIDTH)`. Center it inside `left_widget` using `addStretch()` on both sides of an `QHBoxLayout`.

Rejected alternatives:
- `setViewportMargins` on resize — requires subclassing `resizeEvent`, recalculates every frame
- Painted gutter overlay — fragile layering, doesn't constrain text reflow

## Architecture

```
left_widget  (fills splitter pane, dark bg, QHBoxLayout)
  ├── addStretch()               ← left gutter
  ├── content_col  QWidget
  │     setMaximumWidth(900)
  │     └── QVBoxLayout
  │           ├── ConsoleWidget
  │           ├── ThinkingBar
  │           └── input_row
  └── addStretch()               ← right gutter
```

## Constants

```python
_CONSOLE_MAX_WIDTH = 900  # px — ~100–110 chars at default monospace font
```

## Responsive behaviour

| Left pane width | Result |
|---|---|
| < 900 px | Stretches collapse to zero; full-width as before |
| = 900 px | Snug fit, no gutters |
| > 900 px | Equal dark gutters either side of text column |

## Scope

- **Changed:** `MainWindow.__init__` in `zephyr_gui.py` — layout restructure only
- **Unchanged:** `ConsoleWidget`, `PaletteWidget`, `ThinkingBar`, splitter ratios, title bar

## Testing

1. Launch GUI at default size (1100×700) — no gutters visible
2. Maximise window on 1080p — gutters appear, text column centred at ~900px
3. Drag splitter left until pane < 900px — gutters collapse, no horizontal scrollbar
4. Send a message — FeedbackBar positions correctly within `content_col`
