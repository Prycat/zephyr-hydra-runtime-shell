# -*- coding: utf-8 -*-
"""
launch_zephyr.py — Safe launcher for Zephyr GUI.
Shows a Windows error dialog if anything goes wrong instead of silently dying.
"""
import sys
import os
import traceback

def show_error(title, message):
    """Show a Windows MessageBox — works even if Qt failed to load."""
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            0,
            str(message),
            str(title),
            0x10  # MB_ICONERROR
        )
    except Exception:
        # Absolute last resort
        with open(os.path.join(os.path.dirname(__file__), "zephyr_error.log"), "w") as f:
            f.write(f"{title}\n\n{message}\n")

def main():
    # Make sure we can find our own directory
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    # Check Python version
    if sys.version_info < (3, 9):
        show_error(
            "Zephyr — Wrong Python Version",
            f"Zephyr requires Python 3.9 or later.\n\n"
            f"Currently running: Python {sys.version}\n\n"
            f"Executable: {sys.executable}"
        )
        return

    # Check PySide6
    try:
        import PySide6
    except ImportError:
        show_error(
            "Zephyr — Missing Dependency",
            "PySide6 is not installed.\n\n"
            "Fix: open a terminal and run:\n"
            "    pip install PySide6\n\n"
            f"Python: {sys.executable}"
        )
        return

    # Check agent.py exists
    agent_path = os.path.join(here, "agent.py")
    if not os.path.exists(agent_path):
        show_error(
            "Zephyr — Missing File",
            f"Cannot find agent.py at:\n{agent_path}\n\n"
            "Make sure Zephyr.bat is inside the hermes-agent folder."
        )
        return

    # Launch the GUI by running it as a script, not importing it
    try:
        import runpy
        runpy.run_path(os.path.join(here, "zephyr_gui.py"), run_name="__main__")
    except SystemExit:
        pass  # normal exit from sys.exit()
    except Exception:
        show_error(
            "Zephyr — Launch Error",
            f"Failed to start Zephyr GUI:\n\n{traceback.format_exc()}"
        )

if __name__ == "__main__":
    main()
