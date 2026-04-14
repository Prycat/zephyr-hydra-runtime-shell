# Console Centered Margin Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Constrain the Zephyr console text column to 900px max-width and center it within the left splitter pane, so wide-monitor users see readable gutters instead of line-length sprawl.

**Architecture:** Introduce a `content_col` QWidget with `setMaximumWidth(900)` inside a new `QHBoxLayout` on `left_widget`, flanked by two `addStretch()` calls. The existing `ConsoleWidget`, `ThinkingBar`, and `input_row` move into `content_col`'s `QVBoxLayout`. No other classes are touched.

**Tech Stack:** Python 3.9, PySide6 (Qt6), `zephyr_gui.py`

---

### Task 1: Restructure `left_widget` layout in `MainWindow.__init__`

**Files:**
- Modify: `zephyr_gui.py:2158-2220` (the `# Left pane` block inside `MainWindow.__init__`)

**Step 1: Add the `_CONSOLE_MAX_WIDTH` constant near the top of `MainWindow.__init__` (just after `splitter` is created)**

Locate this comment in `zephyr_gui.py`:
```python
        # Left pane: console + input row
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
```

Replace with:
```python
        # Left pane: console + input row — centered in a max-width column
        _CONSOLE_MAX_WIDTH = 900

        left_widget = QWidget()
        left_outer = QHBoxLayout(left_widget)
        left_outer.setContentsMargins(0, 0, 0, 0)
        left_outer.setSpacing(0)

        content_col = QWidget()
        content_col.setMaximumWidth(_CONSOLE_MAX_WIDTH)
        left_layout = QVBoxLayout(content_col)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        left_outer.addStretch()
        left_outer.addWidget(content_col, stretch=1)
        left_outer.addStretch()
```

**Step 2: Verify nothing else needs changing**

The variables `left_layout`, `self._console`, `self._thinking_bar`, and `input_row` are all added to `left_layout` in the lines immediately following — they now target `content_col`'s layout automatically since we reused the `left_layout` name. No other lines change.

Check these three `addWidget` calls still read exactly:
```python
        left_layout.addWidget(self._console)
        ...
        left_layout.addWidget(self._thinking_bar)
        ...
        left_layout.addWidget(input_row)
```
If they do, no further edits needed.

**Step 3: Launch and verify manually**

```bash
python zephyr_gui.py
```

Checklist:
- [ ] Default window (1100×700): no gutters, full-width console as before
- [ ] Maximise on 1080p or wider: equal dark gutters appear left and right of text column
- [ ] Drag splitter so left pane < 900px: gutters collapse, no horizontal scrollbar appears
- [ ] Send a message: FeedbackBar appears correctly, response renders inside the column

**Step 4: Commit**

```bash
git add zephyr_gui.py
git commit -m "feat: center console to 900px max-width column with dark gutters"
```
