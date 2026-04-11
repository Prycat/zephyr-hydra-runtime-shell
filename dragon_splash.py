# -*- coding: utf-8 -*-
"""
dragon_splash.py — Zephyr startup animation
Animated ASCII dragon splash for Prycat Research / Zephyr agent.
Python 3.9 compatible. Runs in Windows Terminal / any ANSI terminal.
"""
import sys
import time
import math
import random

# ─── ANSI escape codes ────────────────────────────────────────────────────────
RST    = "\033[0m"
DARK   = "\033[90m"       # dim grey  — body shadow
TEAL   = "\033[96m"       # cyan/teal — wing membrane, body
TEAL2  = "\033[36m"       # dark teal — body depth
GRN    = "\033[92m"       # bright green — eyes (alive)
GRN2   = "\033[32m"       # dim green — eyes (dim pulse)
WHT    = "\033[97m"       # white — edge highlights
DIM    = "\033[2m"        # dim
BOLD   = "\033[1m"
HIDE   = "\033[?25l"      # hide cursor
SHOW   = "\033[?25h"      # show cursor
UP     = "\033[{}A"       # move cursor up N lines


def _enable_ansi():
    """Enable ANSI escape codes on Windows 10+."""
    if sys.platform == "win32":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
        except Exception:
            pass


# ─── Dragon ASCII art ─────────────────────────────────────────────────────────
# Two-frame eye animation: EYE_A (bright) and EYE_B (dim)
# Markers in the art: {E} = eye placeholder

_DRAGON = [
    r"                                                                                ",
    r"        .    *         .    *    .         *    .         *    .    *           ",
    r"      *    .    *    .    *    .    *    .    *    .    *    .    *    .    *   ",
    r"    .    *    .    *    .    *    .    *    .    *    .    *    .    *    .     ",
    r"                                                                                ",
    r"                 __                                           __                ",
    r"              .-'  '-.___                             ___.--''  '-.             ",
    r"           .-'          '-.___________________________...---''      '-.         ",
    r"         .'     .-''-.          .---.         .---.          .-''-.    '.       ",
    r"        /     .'      '.       /     \       /     \       .'      '.    \      ",
    r"       /    .'   /\    '.     / .-''-. \   / .-''-. \    .'    /\   '.   \     ",
    r"      /    /    /  \    \   / /  {E}  \ \ / /  {E}  \ \   /    /  \    \   \    ",
    r"     |    /    / /\ \    \ | |    __   | | |    __   | | /    / /\ \    \   |   ",
    r"     |   |    / /  \ \    || |   /  \  | | |   /  \  | ||    / /  \ \   |   |   ",
    r"     |   |   / /    \ \   || |  | -- | | | |  | -- | | ||   / /    \ \  |   |   ",
    r"     |   |  /_/      \_\  ||  \ \__/ /   |  \ \__/ /  /||  /_/      \_\ |   |   ",
    r"      \   \            /  /    '----'     |   '----'    \  \            /   /    ",
    r"       \   '.        .'  / .-----------.  |  .----------. \  '.        .'   /    ",
    r"        \    '-....-'   / /             \ | /            \ \   '-....-'    /     ",
    r"         '.            / /   _________   \|/   _________  \ \            .'      ",
    r"           '-.        / /   /         \      /         \   \ \        .-'        ",
    r"              '------/ /   /           \    /           \   \ \------'           ",
    r"                    /_/   /    .---.    \  /    .---.    \   \_\                 ",
    r"                         /    /     \    \/    /     \    \                      ",
    r"                        /    /       \        /       \    \                     ",
    r"                       /    /_________\      /_________\    \                    ",
    r"                      /                \    /                \                   ",
    r"                     /    ____________  \  /  ____________    \                  ",
    r"                    /    /   /    \   \  \/  /   /    \   \    \                 ",
    r"                   /____/___/      \___\/\___/___/      \___\____\               ",
    r"                                    \  /\/  /                                    ",
    r"                                     \/    \/                                    ",
    r"                                                                                ",
]

# Row and column of each {E} marker (0-indexed, counting only the text content)
# We'll do a substitution at render time.

_TITLE_FRAMES = [
    "  ·  ·  ·  Z  ·  ·  ·  ",
    "  ·  ·  Z  E  ·  ·  ·  ",
    "  ·  Z  E  P  ·  ·  ·  ",
    "  Z  E  P  H  ·  ·  ·  ",
    "  Z  E  P  H  Y  ·  ·  ",
    "  Z  E  P  H  Y  R  ·  ",
    "  Z  E  P  H  Y  R  ·  ",
    f"  Z  E  P  H  Y  R     ",
]

_SUBTITLE = "  Prycat Research  ·  local intelligence  ·  Blackwellian core  "

_SPARKS = list("·∙•*✦✧⊹")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _w(text: str):
    sys.stdout.write(text)
    sys.stdout.flush()


def _clear_up(n: int):
    sys.stdout.write(f"\033[{n}A\033[J")
    sys.stdout.flush()


def _render_dragon(eye_bright: bool) -> list:
    """Return dragon lines with eye colour substituted."""
    eye = f"{GRN}{BOLD}@{RST}" if eye_bright else f"{GRN2}o{RST}"
    out = []
    for line in _DRAGON:
        colored = line.replace("{E}", eye)
        # Colour sparks/particles rows (rows 1-3)
        out.append(colored)
    return out


def _print_dragon_frame(lines: list, eye_bright: bool):
    """Print the full dragon with colours."""
    rendered = _render_dragon(eye_bright)

    spark_chars = "·∙•*"

    for i, line in enumerate(rendered):
        if i in (1, 2, 3):
            # Particle / spark rows — teal with random bright spots
            colored = ""
            for ch in line:
                if ch in spark_chars or ch == "*":
                    r = random.random()
                    if r > 0.85:
                        colored += f"{GRN}{BOLD}{ch}{RST}"
                    elif r > 0.5:
                        colored += f"{TEAL}{ch}{RST}"
                    else:
                        colored += f"{DARK}{ch}{RST}"
                else:
                    colored += ch
            _w(colored + "\n")
        elif i == 0 or i == len(rendered) - 1:
            _w(line + "\n")
        else:
            # Body — teal with edge highlights
            colored = ""
            for ch in line:
                if ch in r"/\|_":
                    colored += f"{WHT}{ch}{RST}"
                elif ch in ".-'`":
                    colored += f"{TEAL}{ch}{RST}"
                elif ch in "()[]{}":
                    colored += f"{TEAL2}{ch}{RST}"
                elif ch in "@o":
                    pass  # already handled in render
                    colored += ch
                else:
                    colored += f"{TEAL2}{ch}{RST}"
            _w(colored + "\n")


# ─── Reveal animation ─────────────────────────────────────────────────────────

def _reveal_animation():
    """Typewriter reveal: print dragon lines one by one, fast."""
    total = len(_DRAGON)

    for i, line in enumerate(_DRAGON):
        # Spark rows: instant
        if i in (1, 2, 3):
            spark_colored = ""
            for ch in line:
                if ch in "·∙•*.":
                    spark_colored += f"{TEAL}{ch}{RST}"
                else:
                    spark_colored += ch
            _w(spark_colored + "\n")
            time.sleep(0.008)
        else:
            # Body rows: colour as we print
            colored = ""
            for ch in line:
                if ch in r"/\|_":
                    colored += f"{WHT}{ch}{RST}"
                elif ch in ".-'`":
                    colored += f"{TEAL}{ch}{RST}"
                elif ch == "{":
                    pass
                else:
                    colored += f"{TEAL2}{ch}{RST}"
            # Handle eye markers in reveal
            colored = line.replace("{E}", f"{GRN}{BOLD}@{RST}")
            # Re-colour non-eye chars
            final = ""
            skip = 0
            j = 0
            raw = line
            while j < len(raw):
                if raw[j:j+3] == "{E}":
                    final += f"{GRN}{BOLD}@{RST}"
                    j += 3
                elif raw[j] in r"/\_":
                    final += f"{WHT}{raw[j]}{RST}"
                    j += 1
                elif raw[j] in ".-'`":
                    final += f"{TEAL}{raw[j]}{RST}"
                    j += 1
                elif raw[j] in "|":
                    final += f"{TEAL}{raw[j]}{RST}"
                    j += 1
                else:
                    final += f"{TEAL2}{raw[j]}{RST}"
                    j += 1
            _w(final + "\n")
            # Slightly slower for dramatic body lines
            delay = 0.018 if i > 5 else 0.01
            time.sleep(delay)


# ─── Eye pulse animation ──────────────────────────────────────────────────────

def _eye_pulse(cycles: int = 6):
    """After reveal, pulse the dragon's eyes N times."""
    n_lines = len(_DRAGON)

    for cycle in range(cycles):
        eye_bright = (cycle % 2 == 0)

        # Move cursor back up to top of dragon
        sys.stdout.write(f"\033[{n_lines}A")
        sys.stdout.flush()

        # Reprint dragon with updated eye state
        for i, line in enumerate(_DRAGON):
            # Build coloured line
            final = ""
            j = 0
            while j < len(line):
                if line[j:j+3] == "{E}":
                    if eye_bright:
                        final += f"{GRN}{BOLD}@{RST}"
                    else:
                        final += f"{GRN2}o{RST}"
                    j += 3
                elif line[j] in r"/\_":
                    final += f"{WHT}{line[j]}{RST}"
                    j += 1
                elif line[j] in ".-'`|":
                    if i in (1, 2, 3):
                        # Spark rows: randomise colour
                        r = random.random()
                        if r > 0.8:
                            final += f"{GRN}{line[j]}{RST}"
                        elif r > 0.4:
                            final += f"{TEAL}{line[j]}{RST}"
                        else:
                            final += f"{DARK}{line[j]}{RST}"
                    else:
                        final += f"{TEAL}{line[j]}{RST}"
                    j += 1
                elif line[j] in "·∙•*.":
                    r = random.random()
                    if r > 0.85:
                        final += f"{GRN}{BOLD}{line[j]}{RST}"
                    elif r > 0.5:
                        final += f"{TEAL}{line[j]}{RST}"
                    else:
                        final += f"{DARK}{line[j]}{RST}"
                    j += 1
                else:
                    final += f"{TEAL2}{line[j]}{RST}"
                    j += 1

            _w(final + "\n")

        time.sleep(0.22 if eye_bright else 0.18)


# ─── Title reveal ─────────────────────────────────────────────────────────────

def _title_sequence():
    """Typewriter reveal of ZEPHYR title."""
    padding = " " * 20

    for frame in _TITLE_FRAMES:
        _w(f"\r{padding}{BOLD}{TEAL}{frame}{RST}")
        sys.stdout.flush()
        time.sleep(0.09)

    # Hold title
    time.sleep(0.3)
    _w("\n")

    # Subtitle fade in char by char
    _w(f"{DIM}{DARK}")
    for ch in _SUBTITLE:
        _w(ch)
        sys.stdout.flush()
        time.sleep(0.022)
    _w(f"{RST}\n\n")
    time.sleep(0.4)


# ─── Boot lines ───────────────────────────────────────────────────────────────

_BOOT_LINES = [
    (f"{DARK}  [ initialising Blackwellian core         ]{RST}", 0.04),
    (f"{DARK}  [ loading vector space  V∈[0,1]^5        ]{RST}", 0.04),
    (f"{DARK}  [ connecting to Ollama  localhost:11434  ]{RST}", 0.04),
    (f"{TEAL}  [ Zephyr online ✓                        ]{RST}", 0.06),
]


def _boot_sequence():
    """Simulate boot log lines."""
    for line, delay in _BOOT_LINES:
        _w(line + "\n")
        sys.stdout.flush()
        time.sleep(delay)
    _w("\n")


# ─── Main entry ───────────────────────────────────────────────────────────────

def show_splash():
    """
    Full animated splash sequence:
    1. Reveal dragon line by line
    2. Pulse eyes 6 times
    3. ZEPHYR title typewriter
    4. Subtitle
    5. Boot log lines
    """
    _enable_ansi()
    _w(HIDE)   # hide cursor during animation

    try:
        _reveal_animation()
        _eye_pulse(cycles=6)
        _title_sequence()
        _boot_sequence()
    except KeyboardInterrupt:
        pass
    finally:
        _w(SHOW)   # always restore cursor


if __name__ == "__main__":
    show_splash()
