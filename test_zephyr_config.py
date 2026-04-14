import os, json, pytest

def test_load_returns_defaults_when_missing(tmp_path):
    import zephyr_gui
    cfg_path = str(tmp_path / "config.json")
    orig = zephyr_gui._zephyr_config_path
    zephyr_gui._zephyr_config_path = lambda: cfg_path
    try:
        cfg = zephyr_gui.load_zephyr_config()
    finally:
        zephyr_gui._zephyr_config_path = orig
    assert cfg["active_model"] == "hermes3:8b"
    assert cfg["turboquant_enabled"] == False

def test_save_and_reload(tmp_path):
    import zephyr_gui
    cfg_path = str(tmp_path / "config.json")
    orig = zephyr_gui._zephyr_config_path
    zephyr_gui._zephyr_config_path = lambda: cfg_path
    try:
        zephyr_gui.save_zephyr_config({"active_model": "mistral:7b", "turboquant_enabled": True})
        cfg = zephyr_gui.load_zephyr_config()
    finally:
        zephyr_gui._zephyr_config_path = orig
    assert cfg["active_model"] == "mistral:7b"
    assert cfg["turboquant_enabled"] == True

def test_load_warns_on_corrupt_json(tmp_path, capsys):
    import zephyr_gui
    cfg_path = str(tmp_path / "config.json")
    with open(cfg_path, "w") as f:
        f.write("{bad json")
    orig = zephyr_gui._zephyr_config_path
    zephyr_gui._zephyr_config_path = lambda: cfg_path
    try:
        cfg = zephyr_gui.load_zephyr_config()
    finally:
        zephyr_gui._zephyr_config_path = orig
    assert cfg == dict(zephyr_gui._CONFIG_DEFAULTS)
    assert "corrupt" in capsys.readouterr().out

def test_load_merges_partial_config(tmp_path):
    import zephyr_gui
    cfg_path = str(tmp_path / "config.json")
    with open(cfg_path, "w") as f:
        json.dump({"active_model": "phi3:mini"}, f)
    orig = zephyr_gui._zephyr_config_path
    zephyr_gui._zephyr_config_path = lambda: cfg_path
    try:
        cfg = zephyr_gui.load_zephyr_config()
    finally:
        zephyr_gui._zephyr_config_path = orig
    assert cfg["active_model"] == "phi3:mini"
    assert cfg["turboquant_enabled"] == False  # default filled in

def test_save_creates_directory(tmp_path):
    import zephyr_gui
    cfg_path = str(tmp_path / "nested" / "config.json")
    orig = zephyr_gui._zephyr_config_path
    zephyr_gui._zephyr_config_path = lambda: cfg_path
    try:
        zephyr_gui.save_zephyr_config({"active_model": "x"})
    finally:
        zephyr_gui._zephyr_config_path = orig
    assert os.path.exists(cfg_path)

import json as _json
from unittest.mock import patch, MagicMock

def test_ollama_fetch_thread_parses_response():
    """OllamaFetchThread emits parsed model list on success."""
    import zephyr_gui
    fake_response = {"models": [
        {"name": "hermes3:8b"},
        {"name": "hermes3:8b-q4_0"},
        {"name": "mistral:7b-instruct-q4_0"},
    ]}
    thread = zephyr_gui.OllamaFetchThread()
    received = []
    thread.models_ready.connect(lambda models: received.extend(models))

    with patch("zephyr_gui.urllib.request.urlopen") as mock_open:
        mock_resp = MagicMock()
        mock_resp.read.return_value = _json.dumps(fake_response).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = mock_resp
        thread.run()  # call run() directly, not start()

    assert "hermes3:8b" in received
    assert "mistral:7b-instruct-q4_0" in received

def test_ollama_fetch_thread_emits_empty_on_error():
    """OllamaFetchThread emits empty list when Ollama is unreachable."""
    import zephyr_gui
    thread = zephyr_gui.OllamaFetchThread()
    received = []
    thread.models_ready.connect(lambda models: received.extend(models))

    with patch("zephyr_gui.urllib.request.urlopen", side_effect=OSError("refused")):
        thread.run()

    assert received == []
