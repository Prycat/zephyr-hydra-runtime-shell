import os, sys, tempfile, textwrap
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from blackwell.config_loader import load_thinking_config, ThinkingConfig

def test_returns_dataclass():
    cfg = load_thinking_config()
    assert isinstance(cfg, ThinkingConfig)

def test_defaults_present():
    cfg = load_thinking_config()
    assert 0.0 <= cfg.judge_temperature <= 1.0
    assert 0.0 <= cfg.safety_floor <= 1.0
    assert cfg.min_pairs > 0

def test_missing_file_uses_defaults():
    cfg = load_thinking_config(path="/nonexistent/path.yaml")
    assert cfg.model_temperature == 0.7

def test_partial_yaml_merges():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(textwrap.dedent("""
            judge:
              temperature: 0.5
        """))
        tmp = f.name
    cfg = load_thinking_config(path=tmp)
    assert cfg.judge_temperature == 0.5
    assert cfg.model_temperature == 0.7
    os.unlink(tmp)

def test_frozen():
    cfg = load_thinking_config()
    try:
        cfg.judge_temperature = 0.99
        assert False, "Should be frozen"
    except Exception:
        pass
