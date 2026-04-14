import os, json, pytest

def test_load_returns_defaults_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path / p.lstrip("~/")))
    import importlib, zephyr_gui
    importlib.reload(zephyr_gui)
    cfg = zephyr_gui.load_zephyr_config()
    assert cfg["active_model"] == "hermes3:8b"
    assert cfg["turboquant_enabled"] == False

def test_save_and_reload(tmp_path, monkeypatch):
    monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path / p.lstrip("~/")))
    import importlib, zephyr_gui
    importlib.reload(zephyr_gui)
    zephyr_gui.save_zephyr_config({"active_model": "mistral:7b", "turboquant_enabled": True})
    cfg = zephyr_gui.load_zephyr_config()
    assert cfg["active_model"] == "mistral:7b"
    assert cfg["turboquant_enabled"] == True
