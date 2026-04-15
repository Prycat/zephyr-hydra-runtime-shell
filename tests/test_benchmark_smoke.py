"""Smoke tests for blackwell/benchmark.py — no Ollama required."""
import json
import pytest
from blackwell.benchmark import score_response, load_prompts, PROMPTS_PATH


def test_score_response_passes_when_expected_present():
    assert score_response("The answer is 391", ["391"], ["i'm here to help"]) is True


def test_score_response_fails_when_forbidden_present():
    assert score_response("I'm here to help! The answer is 391", ["391"], ["i'm here to help"]) is False


def test_score_response_fails_when_no_expected():
    assert score_response("Something completely different", ["391"], []) is False


def test_score_case_insensitive():
    assert score_response("THE ANSWER IS 391", ["391"], []) is True
    assert score_response("I AM HERE TO HELP", ["x"], ["i'm here to help"]) is False


def test_load_prompts_returns_24():
    prompts = load_prompts()
    assert len(prompts) == 24


def test_prompts_have_required_fields():
    for p in load_prompts():
        assert "id" in p
        assert "category" in p
        assert "prompt" in p
        assert "expected_signals" in p
        assert "forbidden_signals" in p
        assert len(p["expected_signals"]) >= 1


def test_prompts_cover_four_categories():
    cats = {p["category"] for p in load_prompts()}
    assert cats == {"content_grounding", "honest_uncertainty", "no_filler", "tool_discipline"}
