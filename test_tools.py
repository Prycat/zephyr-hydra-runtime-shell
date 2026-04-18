"""
test_tools.py — Unit tests for Zephyr's tool functions.
Run with: python test_tools.py
Tests each tool directly without going through the LLM.
"""

import sys

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def test(name, fn):
    try:
        result = fn()
        status = PASS if result else FAIL
        print(f"{status}  {name}")
        print(f"       → {str(result)[:120]}")
        results.append((name, True, result))
    except Exception as e:
        print(f"{FAIL}  {name}")
        print(f"       → Exception: {e}")
        results.append((name, False, str(e)))
    print()

print("=" * 55)
print("  Zephyr Tool Test Suite")
print("=" * 55)
print()

# ── Import all tools from agent.py ─────────────────────────
sys.path.insert(0, ".")
from agent import calculate, get_current_time, read_file, write_file
from agent import web_search, browse_url, run_python, http_request

# ── calculate ──────────────────────────────────────────────
test("calculate: 2 ** 10",
     lambda: calculate("2 ** 10") == "1024")

test("calculate: sqrt(144)",
     lambda: calculate("sqrt(144)") == "12.0")

# ── get_current_time ───────────────────────────────────────
test("get_current_time: returns a string",
     lambda: isinstance(get_current_time(), str) and len(get_current_time()) > 0)

# ── write_file / read_file ─────────────────────────────────
test("write_file: creates a file",
     lambda: "successfully" in write_file("_zephyr_test.txt", "hello from zephyr"))

test("read_file: reads back content",
     lambda: read_file("_zephyr_test.txt") == "hello from zephyr")

import os
try:
    os.unlink("_zephyr_test.txt")
except Exception:
    pass

# ── web_search ─────────────────────────────────────────────
test("web_search: returns results for 'python programming'",
     lambda: len(web_search("python programming", max_results=3)) > 50)

# ── browse_url ─────────────────────────────────────────────
test("browse_url: fetches example.com",
     lambda: len(browse_url("https://example.com")) > 50)

# ── run_python ─────────────────────────────────────────────
test("run_python: print hello",
     lambda: run_python('print("hello from zephyr")').strip() == "hello from zephyr")

test("run_python: arithmetic",
     lambda: run_python('print(2 + 2)').strip() == "4")

test("run_python: timeout guard (infinite loop)",
     lambda: "timed out" in run_python('while True: pass'))

# ── http_request ───────────────────────────────────────────
test("http_request: GET httpbin.org/get",
     lambda: "200" in http_request("GET", "https://httpbin.org/get"))

test("http_request: GET invalid URL returns error message",
     lambda: "error" in http_request("GET", "https://this-does-not-exist-zephyr-test.xyz").lower())

# ── Summary ────────────────────────────────────────────────
print("=" * 55)
passed = sum(1 for _, ok, _ in results if ok)
total  = len(results)
print(f"  {passed}/{total} tests passed")
if passed == total:
    print("  All tools operational. Zephyr is ready.")
else:
    failed = [name for name, ok, _ in results if not ok]
    print(f"  Failed: {', '.join(failed)}")
print("=" * 55)
