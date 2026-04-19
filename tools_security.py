"""
tools_security.py — Prycat Research
Security layer for Zephyr's tool suite.

Fixes:
  - Path traversal in read_file / write_file  (FILE_SANDBOX_ROOT enforcement)
  - SSRF in browse_url / http_request         (private IP blocklist)
  - eval() breakout in calculate              (AST-based safe evaluator)
  - API key leakage in run_python             (stripped subprocess environment)
"""

import ast
import ipaddress
import math
import os
import re
import socket
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

# ─── File sandbox ─────────────────────────────────────────────────────────────
#
# read_file and write_file are restricted to this directory tree.
# Override with ZEPHYR_FILE_ROOT env var if you want a different root.
#
_DEFAULT_FILE_ROOT = os.path.join(os.path.expanduser("~"), "Desktop", "zephyr-sandbox")
FILE_SANDBOX_ROOT: str = os.environ.get("ZEPHYR_FILE_ROOT", _DEFAULT_FILE_ROOT)


def check_path(path: str, *, write: bool = False) -> Optional[str]:
    """
    Resolve *path* and verify it sits inside FILE_SANDBOX_ROOT.

    Returns None if the path is safe.
    Returns an error string if the path is blocked.
    """
    try:
        resolved = Path(path).resolve()
        root     = Path(FILE_SANDBOX_ROOT).resolve()

        # Path.is_relative_to() is Python 3.9+; use str comparison for 3.8 compat
        resolved_str = str(resolved)
        root_str     = str(root)

        if resolved_str != root_str and not resolved_str.startswith(root_str + os.sep):
            return (
                f"[BLOCKED] Path is outside the file sandbox.\n"
                f"  Allowed root : {root_str}\n"
                f"  Requested    : {resolved_str}\n"
                f"  Tip: set ZEPHYR_FILE_ROOT to expand the sandbox."
            )
        return None
    except Exception as exc:
        return f"[BLOCKED] Path validation error: {exc}"


# ─── SSRF / private-IP blocklist ──────────────────────────────────────────────
#
# browse_url and http_request resolve the target hostname and reject
# any address that falls inside a private / link-local / loopback range.

_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),    # loopback
    ipaddress.ip_network("10.0.0.0/8"),     # RFC-1918 private
    ipaddress.ip_network("172.16.0.0/12"),  # RFC-1918 private
    ipaddress.ip_network("192.168.0.0/16"), # RFC-1918 private
    ipaddress.ip_network("169.254.0.0/16"), # link-local / AWS metadata service
    ipaddress.ip_network("0.0.0.0/8"),      # "this" network
    ipaddress.ip_network("::1/128"),        # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),       # IPv6 ULA
    ipaddress.ip_network("fe80::/10"),      # IPv6 link-local
]

_BLOCKED_HOST_PATTERNS = re.compile(
    r"^(localhost|localhost\.localdomain|.*\.local)$", re.IGNORECASE
)


def check_url(url: str) -> Optional[str]:
    """
    Verify that *url* does not point at a private or internal address.

    Returns None if the URL is safe.
    Returns an error string if blocked.
    """
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        if not host:
            return "[BLOCKED] URL has no resolvable host."

        if _BLOCKED_HOST_PATTERNS.match(host):
            return f"[BLOCKED] Host '{host}' is an internal address."

        # Resolve to IP and check all returned addresses
        try:
            infos = socket.getaddrinfo(host, None)
            for info in infos:
                raw_ip = info[4][0]
                # Strip IPv6 zone id if present
                raw_ip = raw_ip.split("%")[0]
                addr = ipaddress.ip_address(raw_ip)
                for net in _BLOCKED_NETWORKS:
                    if addr in net:
                        return (
                            f"[BLOCKED] Host '{host}' resolves to private/internal "
                            f"IP {addr} — SSRF guard."
                        )
        except socket.gaierror:
            pass  # DNS failure — let the HTTP client handle it naturally

        return None
    except Exception as exc:
        return f"[BLOCKED] URL validation error: {exc}"


# ─── Safe math evaluator ──────────────────────────────────────────────────────
#
# Replaces eval(expression, safe_ns) in calculate().
# Parses to AST first and only allows a whitelist of node types —
# no __subclasses__, no attribute access, no import, no call to arbitrary names.

_MATH_NAMES: dict = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
_MATH_NAMES.update({"abs": abs, "round": round, "min": min, "max": max, "pow": pow})

_ALLOWED_NODES = frozenset({
    ast.Expression,
    ast.BinOp, ast.UnaryOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
    ast.UAdd, ast.USub,
    ast.Call,
    ast.Constant,
    ast.Name, ast.Load,
})


def safe_eval(expression: str) -> str:
    """
    Evaluate a math expression safely via AST inspection.

    Rejects anything that isn't an arithmetic expression over the math module.
    Returns the result as a string, or an error message.
    """
    expression = expression.strip()
    if len(expression) > 500:
        return "Error: expression too long (max 500 chars)."
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return f"Syntax error: {exc}"

    for node in ast.walk(tree):
        node_type = type(node)
        if node_type not in _ALLOWED_NODES:
            return (
                f"Error: disallowed operation in expression "
                f"({node_type.__name__}). Only arithmetic is allowed."
            )
        if isinstance(node, ast.Name) and node.id not in _MATH_NAMES:
            return f"Error: unknown name '{node.id}'."
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                return "Error: only named function calls are allowed."
            if node.func.id not in _MATH_NAMES:
                return f"Error: unknown function '{node.func.id}'."

    try:
        result = eval(  # noqa: S307 — safe: AST pre-validated above
            compile(tree, "<expr>", "eval"),
            {"__builtins__": {}},
            _MATH_NAMES,
        )
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


# ─── Sanitised subprocess environment for run_python ──────────────────────────
#
# Strip any env var that looks like a secret before handing the environment
# to the Python subprocess that executes user/LLM-supplied code.

_SECRET_KEYWORDS = re.compile(
    r"(key|token|secret|password|passwd|credential|auth|api_?key|bearer)",
    re.IGNORECASE,
)


def safe_python_env() -> dict:
    """
    Return a copy of os.environ with all secret-looking variables removed.
    Called by run_python() before launching the subprocess.
    """
    env = os.environ.copy()
    stripped = []
    for var in list(env.keys()):
        if _SECRET_KEYWORDS.search(var):
            del env[var]
            stripped.append(var)
    if stripped:
        # Silent in production; uncomment to debug:
        # print(f"[security] stripped env vars: {stripped}")
        pass
    return env


def safe_python_argv() -> list:
    """
    Return the argv prefix for a sandboxed Python subprocess.
    -I  : isolated mode — ignores PYTHONPATH, user site-packages, PYTHON* env vars
    -S  : skip automatic import of 'site'
    """
    return [sys.executable, "-I", "-S"]
