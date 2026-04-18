# Zephyr GUI — Design Document
**Date:** 2026-04-11  
**Project:** Prycat Research / Zephyr Agent  
**Status:** Approved — ready for implementation

---

## 1. Overview

A single-file native desktop GUI (`zephyr_gui.py`) that wraps the existing Zephyr CLI agent (`agent.py`) with zero modification to the agent itself. The GUI provides a command palette with tool buttons, a streaming output console, and a styled input bar — all communicating with the agent through a subprocess stdin/stdout pipe.

**Stack:** Python 3.9 + PySide6  
**Install:** `pip install PySide6`  
**Launch:** `python zephyr_gui.py`

---

## 2. Architecture — Subprocess Pipe

```
zephyr_gui.py
│
├─ ZephyrProcess (QThread)
│    ├─ Launches: python agent.py  (subprocess, stdin/stdout pipe)
│    ├─ Reads stdout line-by-line → emits output_signal(str)
│    └─ Writes to stdin: send_input(text)
│
├─ MainWindow (QMainWindow)
│    ├─ ConsoleWidget  (left ⅔)
│    └─ PaletteWidget  (right ⅓)
│         └─ ZephyrButton × N
```

`agent.py` is **never modified**. All GUI-to-agent communication is plain text over pipes — identical to keyboard input in the terminal.

---

## 3. Layout — Split Pane

```
┌──────────────────────────────────────────────────────────────┐
│  ZEPHYR ●  hermes3:8b                              [─][□][✕] │
├─────────────────────────────────────┬────────────────────────┤
│                                     │  COMMAND PALETTE       │
│   OUTPUT CONSOLE                    │                        │
│   (QPlainTextEdit, read-only)       │  [/blackwell        ]  │
│   monospace, teal-on-black          │  [/help             ]  │
│   auto-scroll, color-coded          │  [/tools            ]  │
│   selectable text                   │  [/search           ]  │
│                                     │  [/browse           ]  │
│                                     │  [/run              ]  │
│                                     │  [/status           ]  │
│                                     │  [/model            ]  │
│                                     │  [/save             ]  │
│                                     │  [/clear            ]  │
│                                     │  ── Keys ──            │
│                                     │  [/keys setup       ]  │
│                                     │  [/keys list        ]  │
│                                     │  ── External AI ──     │
│                                     │  [/call             ]  │
│                                     │  [/call claude      ]  │
│                                     │  [/call gpt         ]  │
│                                     │  [/call grok        ]  │
│                                     │  [/call gemini      ]  │
│                                     │  ── Training ──        │
│                                     │  [/Run BlackLoRA-N  ]  │
│                                     │  ── Session ──         │
├─────────────────────────────────────┤  [/exit             ]  │
│  ▶  input bar              [SEND]   │                        │
└─────────────────────────────────────┴────────────────────────┘
```

- **Left pane (⅔ width):** Console + input bar
- **Right pane (⅓ width):** Scrollable button palette, grouped by section dividers
- **Splitter:** `QSplitter`, user-draggable, default ratio 2:1
- **Window frame:** Standard OS chrome (not borderless) for stability

---

## 4. Button Design — "Monolith Signal"

A custom `ZephyrButton(QPushButton)` subclass. All effects rendered in `paintEvent` via `QPainter`. A `QTimer` at 60fps drives animation state.

### Visual Layers (paint order)
1. **Base fill** — `#0d1117`, 4px border-radius
2. **Scanline grain** — repeating 1px horizontal lines every 3px at 1.8% opacity
3. **Mouse glow blob** — `QRadialGradient` centered at mouse pos, `rgba(77,194,179,0.18)` → transparent. Updates via `mouseMoveEvent`
4. **Inner bevel** — top-left hairline `rgba(255,255,255,0.03)`, bottom-right `rgba(0,0,0,0.4)`
5. **Resting border** — `rgba(255,255,255,0.08)` hairline rect
6. **BorderWake pulse** — slow `sin()` wave (period 5.5s) modulates inner border from `rgba(77,194,179,0.02)` → `rgba(77,194,179,0.08)`
7. **Hover sweep** — on hover, `sweep_t` animates `0→1` over 700ms (ease-out). Draws a bright teal line translating across top+bottom edges of the border rectangle
8. **Text** — `Consolas`/`IBM Plex Mono`, teal `rgba(128,221,202,0.92)`, UPPERCASE, large tracking
9. **Command sub-text** — dim teal `rgba(128,221,202,0.52)`, shows the slash command string
10. **State dot** — 6px circle, right side: `grey`=idle, `amber` pulsing=running, `green`=success, `red`=error

### States
| State | Dot | Effect |
|---|---|---|
| `idle` | dim grey | borderWake breathing only |
| `running` | amber, pulsing | scanline sweep animates across full button face |
| `success` | green | green tint flash, fades in 1.2s |
| `error` | red | amber tint flash, fades in 1.2s |

### Tooltip
Each button has a `setToolTip(description)`. Qt native tooltips styled dark via QSS.

---

## 5. Button Definitions (ordered)

| Order | Label | Command injected | Tooltip |
|---|---|---|---|
| 1 | `/blackwell` | `/blackwell` | Drops Zephyr into a planning space where he interviews you, and your answers reshape how he sees the world — permanently. |
| 2 | `/help` | `/help` | Show all commands. |
| 3 | `/tools` | `/tools` | List all of Zephyr's active tools. |
| 4 | `/search` | `/search ` | Raw DuckDuckGo search, instant. Appends cursor for query input. |
| 5 | `/browse` | `/browse ` | Fetch a URL directly. Appends cursor for URL input. |
| 6 | `/run` | `/run ` | Run Python immediately. Appends cursor for code input. |
| 7 | `/status` | `/status` | Check Ollama is alive. |
| 8 | `/model` | `/model` | Show model + API info. |
| 9 | `/save` | `/save` | Save conversation to Obsidian vault as a formatted .md with YAML frontmatter. |
| 10 | `/clear` | `/clear` | Reset conversation history (prompts y/n confirmation). |
| — | *divider* | Keys | — |
| 11 | `/keys setup` | `/keys setup` | Interactive wizard: select provider from dropdown, enter key, stored masked in ~/.zephyr/keys.json. |
| 12 | `/keys list` | `/keys list` | Show which providers are active: claude ✓ gpt ✓ grok ✗ gemini ✓ |
| — | *divider* | External AI | — |
| 13 | `/call` | `/call ` | Route your message to the best available external AI provider. |
| 14 | `/call claude` | `/call claude ` | Force Claude (claude-opus-4-5 via Anthropic). |
| 15 | `/call gpt` | `/call gpt ` | Force GPT-4o via OpenAI. |
| 16 | `/call grok` | `/call grok ` | Force Grok-3 via xAI endpoint. |
| 17 | `/call gemini` | `/call gemini ` | Force Gemini 2.0 Flash via Google. |
| — | *divider* | Training | — |
| 18 | `/Run BlackLoRA-N` | `/run_lora` | Run LoRA fine-tuning from completed Blackwell interview data. (Future tool) |
| — | *divider* | Session | — |
| 19 | `/exit` | `/exit` | Quit Zephyr. |

**Button injection rule:** Buttons that need a follow-up argument (`/search`, `/browse`, `/run`, `/call *`) inject the text into the input bar with the cursor placed at the end — the user types their argument and hits Enter. Buttons with no argument (`/help`, `/status`, `/blackwell`, etc.) inject and fire immediately.

---

## 6. Console

- Widget: `QPlainTextEdit`, read-only, no line wrap
- Font: `Consolas 11pt` (fallback `Courier New`)
- Background: `#090c10`
- Text color: `rgba(128,221,202,0.92)` (teal) — default Zephyr output
- Color rules applied via `appendHtml()`:
  - `You:` lines — dim white `#aab6c2`
  - `Zephyr:` lines — teal `#80ddca`
  - Tool call lines (contain `→` or `[tool]`) — cyan `#7ab8d8`
  - Error lines — amber `#d4a050`
  - System/divider lines — muted `#445566`
- Auto-scroll: follows new output unless user has scrolled up (detected via scrollbar position)
- Selectable: user can copy text out

---

## 7. Input Bar

- Widget: `QLineEdit`, same font/colors as console
- Placeholder text: `▶  type a message or /command...`
- Enter key or SEND button fires input
- Input is written to subprocess stdin + echoed to console as `You: <text>`
- Arrow Up/Down cycles through last 50 inputs (in-memory history list)
- After a button injects a partial command (e.g. `/search `), input bar receives focus automatically

---

## 8. Header Bar

- `ZEPHYR` label — teal, bold, monospace
- `●` live indicator — green `#66c47a`, pulses via `QTimer` opacity animation (1.5s period)
- Model name — `hermes3:8b`, dim muted text
- Standard OS window controls (minimize/maximize/close)

---

## 9. Theme — Global QSS

Applied to `QApplication`:
- Window background: `#090c10`
- All text: teal family
- `QScrollBar`: minimal, dark, no arrows
- `QSplitter` handle: `#1a2030`, 2px
- `QToolTip`: `#0d1117` bg, teal border, teal text, monospace font
- `QLineEdit`: `#0d1117` bg, teal border on focus

---

## 10. Providers Reference (for /keys setup dropdown)

| Name | Model | Package |
|---|---|---|
| claude | claude-opus-4-5 | anthropic |
| gpt | gpt-4o | openai |
| grok | grok-3 | openai (xAI endpoint) |
| gemini | gemini-2.0-flash | google-generativeai |

---

## 11. File Structure

```
hermes-agent/
├── agent.py               # untouched
├── zephyr_keys.py         # untouched
├── zephyr_gui.py          # NEW — entire GUI in one file
├── blackwell/
│   └── ...                # untouched
└── docs/plans/
    └── 2026-04-11-zephyr-gui-design.md
```

---

## 12. Out of Scope (YAGNI)

- Drag-and-drop file input
- Multiple conversation tabs
- Settings panel (use /keys and /model commands instead)
- Embedded terminal emulator (subprocess pipe is sufficient)
- `/Run BlackLoRA-N` implementation (stub button only — future task)
