"""Smoke tests for blackwell/data_generator.py."""
import json
import pytest
from blackwell.data_generator import (
    generate_pairs,
    generate_pairs_for_category,
    write_pairs,
    CATEGORIES,
    PAIRS_PER_CATEGORY,
    MIN_TOTAL_PAIRS,
)


def test_categories_defined():
    assert set(CATEGORIES) == {
        "content_grounding", "honest_uncertainty", "no_filler", "tool_discipline"
    }


def test_pairs_per_category():
    assert PAIRS_PER_CATEGORY == 50


def test_min_total_pairs():
    assert MIN_TOTAL_PAIRS == 200


def test_generate_pairs_returns_correct_count():
    pairs = generate_pairs()
    assert len(pairs) == MIN_TOTAL_PAIRS


def test_pairs_are_sharegpt_format():
    pairs = generate_pairs()
    for p in pairs[:5]:
        assert "conversations" in p
        convs = p["conversations"]
        assert len(convs) == 2
        assert convs[0]["from"] == "human"
        assert convs[1]["from"] == "gpt"
        assert len(convs[0]["value"]) > 0
        assert len(convs[1]["value"]) > 0


def test_write_to_file(tmp_path):
    out = tmp_path / "test_pairs.jsonl"
    pairs = generate_pairs()
    write_pairs(pairs, str(out))
    lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == MIN_TOTAL_PAIRS
    for line in lines:
        obj = json.loads(line)
        assert "conversations" in obj


def test_each_category_represented():
    for cat in CATEGORIES:
        pairs = generate_pairs_for_category(cat)
        assert len(pairs) == PAIRS_PER_CATEGORY
