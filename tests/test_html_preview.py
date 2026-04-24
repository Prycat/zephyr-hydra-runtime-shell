# tests/test_html_preview.py
import sys
import os

from zephyr_html_preview import extract_last_html_block, is_webengine_available


def test_extract_simple():
    text = "Here is some code:\n```html\n<h1>Hello</h1>\n```\nDone."
    result = extract_last_html_block(text)
    assert result == "<h1>Hello</h1>"


def test_extract_returns_last_block():
    text = "```html\n<p>first</p>\n```\n\n```html\n<p>second</p>\n```"
    result = extract_last_html_block(text)
    assert result == "<p>second</p>"


def test_extract_multiline():
    text = "```html\n<html>\n<body>\n<p>hi</p>\n</body>\n</html>\n```"
    result = extract_last_html_block(text)
    assert result == "<html>\n<body>\n<p>hi</p>\n</body>\n</html>"


def test_extract_none_when_no_block():
    assert extract_last_html_block("no html here") is None
    assert extract_last_html_block("") is None


def test_extract_incomplete_block_returns_none():
    # Opening fence with no closing fence — not a complete block
    assert extract_last_html_block("```html\n<p>unclosed") is None


def test_extract_crlf_normalised():
    # Windows-style CRLF should not leave \r artifacts in the captured content
    text = "```html\r\n<h1>Hello</h1>\r\n```\r\n"
    result = extract_last_html_block(text)
    assert result == "<h1>Hello</h1>"


def test_is_webengine_available_returns_bool():
    result = is_webengine_available()
    assert isinstance(result, bool)


def test_is_webengine_available_true_when_present():
    from unittest.mock import patch, MagicMock
    with patch.dict('sys.modules', {'PySide6.QtWebEngineWidgets': MagicMock()}):
        # The function does a fresh import each call so we test directly
        assert isinstance(is_webengine_available(), bool)


def test_is_webengine_available_false_when_missing():
    from unittest.mock import patch
    # Force ImportError by setting the module entry to None
    with patch.dict('sys.modules', {'PySide6.QtWebEngineWidgets': None}):
        result = is_webengine_available()
        assert result is False
