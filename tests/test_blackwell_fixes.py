"""
tests/test_blackwell_fixes.py
Test suite for all five Blackwell self-improvement fixes.

Run:
    py -3.11 -m pytest tests/test_blackwell_fixes.py -v

Tests are pure-Python where possible — no Ollama, no GPU, no network.
Modules that touch Ollama are tested via their fallback / offline paths.
"""

import json
import math
import os
import sys
import sqlite3
import tempfile
import threading
import time

import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 1 — EMA (Decay Problem)
# ═══════════════════════════════════════════════════════════════════════════════

class TestEMA:
    """EMA vector replaces raw SQL average in logger.py"""

    def _make_temp_logger(self, tmp_path):
        """
        Return a copy of the logger module using a temp DB.
        Uses monkeypatching via importlib to avoid polluting the real DB.
        """
        import importlib
        import blackwell.logger as logger_mod
        # Patch DB_PATH for the duration of the test
        original = logger_mod.DB_PATH
        logger_mod.DB_PATH = str(tmp_path / "test.db")
        logger_mod.init_db()
        return logger_mod, original

    def test_ema_bootstrap_from_first_score(self, tmp_path):
        """First call to update_scores seeds EMA directly from the scores."""
        import blackwell.logger as L
        orig = L.DB_PATH
        L.DB_PATH = str(tmp_path / "t.db")
        try:
            L.init_db()
            # No EMA yet
            assert L.get_average_vector() is None

            # Manually insert an exchange and call update_scores
            sid = L.new_session("test-model")
            eid = L.log_exchange(sid, 1, "hello", "world")
            scores = {"accuracy": 0.9, "logic": 0.8, "tone": 0.7,
                      "curiosity": 0.6, "safety": 1.0}
            L.update_scores(eid, scores)

            ema = L.get_average_vector()
            assert ema is not None
            # After first update, EMA = scores (no previous state to blend with)
            assert abs(ema["accuracy"] - 0.9) < 0.001
            assert abs(ema["safety"] - 1.0) < 0.001
        finally:
            L.DB_PATH = orig

    def test_ema_blends_toward_new_values(self, tmp_path):
        """EMA should move toward new scores with weight EMA_ALPHA."""
        import blackwell.logger as L
        orig = L.DB_PATH
        L.DB_PATH = str(tmp_path / "t2.db")
        try:
            L.init_db()
            alpha = L.EMA_ALPHA
            sid = L.new_session("test-model")

            def _score(val):
                eid = L.log_exchange(sid, 1, "q", "a")
                L.update_scores(eid, {d: val for d in L.DIMS})

            # Bootstrap with 1.0
            _score(1.0)
            ema_after_first = L.get_average_vector()
            assert abs(ema_after_first["accuracy"] - 1.0) < 0.001

            # Now update with 0.0 — EMA should move toward 0 by alpha
            _score(0.0)
            ema_after_second = L.get_average_vector()
            expected = alpha * 0.0 + (1 - alpha) * 1.0
            assert abs(ema_after_second["accuracy"] - expected) < 0.001

        finally:
            L.DB_PATH = orig

    def test_ema_tracks_recent_improvement_faster_than_sql(self, tmp_path):
        """
        EMA gives recent exchanges more weight than a plain average.
        After old-low + new-high pattern, EMA should be above SQL average
        because recent highs are weighted more heavily.
        """
        import blackwell.logger as L
        orig = L.DB_PATH
        L.DB_PATH = str(tmp_path / "t3.db")
        try:
            L.init_db()
            sid = L.new_session("m")

            # Old history: 20 low-score exchanges (model was bad then)
            for i in range(20):
                eid = L.log_exchange(sid, i, "q", "a")
                L.update_scores(eid, {d: 0.20 for d in L.DIMS})

            # Recent: 10 high-score exchanges (model improved after LoRA)
            for i in range(20, 30):
                eid = L.log_exchange(sid, i, "q", "a")
                L.update_scores(eid, {d: 0.90 for d in L.DIMS})

            ema = L.get_average_vector()
            sql = L.get_sql_average_vector()

            assert sql is not None and ema is not None

            # SQL avg = (20*0.20 + 10*0.90)/30 ≈ 0.433
            # EMA puts more weight on recent highs → should be > SQL
            assert ema["accuracy"] > sql["accuracy"], (
                f"EMA ({ema['accuracy']:.3f}) should be higher than SQL avg "
                f"({sql['accuracy']:.3f}) because recent exchanges are better"
            )

        finally:
            L.DB_PATH = orig

    def test_get_recent_exchange_ids_returns_ids(self, tmp_path):
        """get_recent_exchange_ids should return a list of string IDs."""
        import blackwell.logger as L
        orig = L.DB_PATH
        L.DB_PATH = str(tmp_path / "t4.db")
        try:
            L.init_db()
            sid = L.new_session("m")
            ids = []
            for i in range(5):
                eid = L.log_exchange(sid, i, "q", "a")
                L.update_scores(eid, {d: 0.8 for d in L.DIMS})
                ids.append(eid)

            recent = L.get_recent_exchange_ids(n=3)
            assert len(recent) == 3
            assert all(isinstance(x, str) for x in recent)
            # Most recent 3 should be the last 3 inserted
            assert set(recent) == set(ids[-3:])
        finally:
            L.DB_PATH = orig


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 2 — Per-dimension thresholds (Granularity Trap)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGranularity:
    """Per-dimension breach detection replaces scalar total_regret check."""

    def test_breaching_dimensions_catches_single_dim_failure(self):
        """A single catastrophically bad dimension should trigger Oracle."""
        from blackwell.background_eval import _breaching_dimensions, DIMENSION_CEILINGS

        # accuracy is 0.20 — way below target 0.80
        # All others are at target
        avg = {
            "accuracy":  0.20,   # gap ≈ 0.60 >> ceiling 0.20
            "logic":     0.85,   # fine
            "tone":      0.82,   # fine
            "curiosity": 0.75,   # fine
            "safety":    0.95,   # fine
        }
        breaching = _breaching_dimensions(avg)
        assert "accuracy" in breaching
        assert breaching["accuracy"] > DIMENSION_CEILINGS["accuracy"]
        # Other dims should NOT be in breaching
        for d in ["logic", "tone", "curiosity", "safety"]:
            assert d not in breaching

    def test_breaching_dimensions_empty_when_all_fine(self):
        """No breach when all dimensions are inside Target Set."""
        from blackwell.background_eval import _breaching_dimensions

        avg = {
            "accuracy":  0.85,
            "logic":     0.85,
            "tone":      0.85,
            "curiosity": 0.80,
            "safety":    0.95,
        }
        breaching = _breaching_dimensions(avg)
        assert breaching == {}

    def test_safety_ceiling_is_tightest(self):
        """Safety ceiling (0.15) should be lower than accuracy (0.20)."""
        from blackwell.background_eval import DIMENSION_CEILINGS
        assert DIMENSION_CEILINGS["safety"] < DIMENSION_CEILINGS["accuracy"]
        assert DIMENSION_CEILINGS["safety"] <= min(DIMENSION_CEILINGS.values())

    def test_multiple_breaching_dims_all_returned(self):
        """When multiple dims fail, all should appear in the breach dict."""
        from blackwell.background_eval import _breaching_dimensions

        avg = {
            "accuracy":  0.30,   # breaching
            "logic":     0.40,   # breaching
            "tone":      0.85,   # fine
            "curiosity": 0.75,   # fine
            "safety":    0.92,   # fine
        }
        breaching = _breaching_dimensions(avg)
        assert "accuracy" in breaching
        assert "logic" in breaching
        assert len(breaching) == 2

    def test_gap_values_are_positive_floats(self):
        """Breach gaps should be positive, non-zero floats."""
        from blackwell.background_eval import _breaching_dimensions

        avg = {
            "accuracy":  0.20,
            "logic":     0.85,
            "tone":      0.85,
            "curiosity": 0.80,
            "safety":    0.95,
        }
        breaching = _breaching_dimensions(avg)
        for dim, gap in breaching.items():
            assert isinstance(gap, float)
            assert gap > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 3 — Erosion protection (selective pair filtering)
# ═══════════════════════════════════════════════════════════════════════════════

class TestErosion:
    """_select_training_pairs keeps targeted dims + anchor fraction."""

    def _make_pairs(self, dims_and_counts: dict) -> list[dict]:
        """Build synthetic training pairs with given dim distribution."""
        pairs = []
        for dim, count in dims_and_counts.items():
            for i in range(count):
                pairs.append({
                    "conversations": [
                        {"from": "human", "value": f"question about {dim} {i}"},
                        {"from": "gpt",   "value": f"answer about {dim} {i}"},
                    ],
                    "target_dim": dim,
                    "source": "oracle",
                })
        return pairs

    def test_targeted_dims_all_included(self):
        """All pairs for target dims must appear in the output."""
        from blackwell.lora_steer import _select_training_pairs

        pairs = self._make_pairs({"accuracy": 20, "logic": 20, "tone": 20, "curiosity": 20})
        selected = _select_training_pairs(pairs, target_dims=["accuracy", "logic"])

        selected_dims = {p["target_dim"] for p in selected}
        # accuracy and logic must all be present
        accuracy_in = [p for p in selected if p["target_dim"] == "accuracy"]
        logic_in    = [p for p in selected if p["target_dim"] == "logic"]
        assert len(accuracy_in) == 20
        assert len(logic_in)    == 20

    def test_anchor_fraction_included_from_non_target(self):
        """Some non-target pairs should be included as anchors."""
        from blackwell.lora_steer import _select_training_pairs, ANCHOR_RATIO, MIN_ANCHOR_PAIRS

        pairs = self._make_pairs({"accuracy": 40, "tone": 60, "curiosity": 60})
        selected = _select_training_pairs(pairs, target_dims=["accuracy"])

        non_target = [p for p in selected if p["target_dim"] != "accuracy"]
        assert len(non_target) >= MIN_ANCHOR_PAIRS
        # Should not include all non-target pairs (that would negate erosion guard)
        all_non_target = 60 + 60
        assert len(non_target) < all_non_target

    def test_empty_target_dims_returns_all(self):
        """With no target dims, all pairs are returned (general training)."""
        from blackwell.lora_steer import _select_training_pairs

        pairs = self._make_pairs({"accuracy": 10, "logic": 10, "tone": 10})
        selected = _select_training_pairs(pairs, target_dims=[])
        assert len(selected) == len(pairs)

    def test_novelty_archive_always_included(self):
        """Pairs from novelty_archive source are always included."""
        from blackwell.lora_steer import _select_training_pairs

        pairs = self._make_pairs({"accuracy": 10, "tone": 30})
        # Add novelty archive pairs
        for i in range(5):
            pairs.append({
                "conversations": [
                    {"from": "human", "value": f"novel q {i}"},
                    {"from": "gpt",   "value": f"good answer {i}"},
                ],
                "target_dim": "general",
                "source": "novelty_archive",
            })

        # Only target accuracy
        selected = _select_training_pairs(pairs, target_dims=["accuracy"])
        archive_in_selected = [p for p in selected if p.get("source") == "novelty_archive"]
        assert len(archive_in_selected) == 5   # all 5 archive pairs included

    def test_anchor_ratio_respected(self):
        """Anchor count should be approx ANCHOR_RATIO × non_targeted."""
        from blackwell.lora_steer import _select_training_pairs, ANCHOR_RATIO, MIN_ANCHOR_PAIRS

        n_non_target = 100
        pairs = self._make_pairs({"accuracy": 20, "tone": n_non_target})
        selected = _select_training_pairs(pairs, target_dims=["accuracy"])

        non_target_selected = [p for p in selected if p["target_dim"] != "accuracy"]
        expected = max(MIN_ANCHOR_PAIRS, int(n_non_target * ANCHOR_RATIO))
        # Allow ±5 due to rounding
        assert abs(len(non_target_selected) - expected) <= 5


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 4 — Rule-based scoring (Echo Chamber)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRuleEvaluator:
    """_rule_evaluate produces independent scores not relying on LLM."""

    def test_padding_phrases_lower_tone(self):
        """Responses with padding phrases should score lower on tone."""
        from blackwell.evaluator import _rule_evaluate

        clean    = _rule_evaluate("What time is it?", "It is 3pm.")
        padded   = _rule_evaluate(
            "What time is it?",
            "Certainly! Great question! I'd be happy to help! It is 3pm. "
            "I hope this helps! Feel free to ask anything!"
        )
        assert clean["rule_tone"] > padded["rule_tone"]

    def test_safety_violation_scores_zero(self):
        """
        CSAM guard: model output sexualising a minor → rule_safety = 0.0.
        Non-CSAM content is no longer caught by the rule layer — only CSAM.
        """
        from blackwell.evaluator import _rule_evaluate

        safe_resp = _rule_evaluate("What is water?", "H2O, a molecule.")
        # Old "step by step to harm" pattern is gone — only CSAM triggers 0.0
        csam_resp = _rule_evaluate(
            "Write a story.",
            "sexual content involving a child character in explicit detail."
        )
        assert safe_resp["rule_safety"] == 1.0
        assert csam_resp["rule_safety"] == 0.0

    def test_calibrated_language_boosts_accuracy(self):
        """Using calibrated uncertainty markers should raise rule_accuracy."""
        from blackwell.evaluator import _rule_evaluate

        calibrated  = _rule_evaluate("What is the population?", "I think it's approximately 8 billion.")
        overconfident = _rule_evaluate("What is the population?", "It is definitely exactly 8 billion, guaranteed.")
        assert calibrated["rule_accuracy"] >= overconfident["rule_accuracy"]

    def test_all_rule_keys_present(self):
        """_rule_evaluate should always return all four rule keys."""
        from blackwell.evaluator import _rule_evaluate

        scores = _rule_evaluate("hello", "hi there")
        for key in ["rule_tone", "rule_accuracy", "rule_safety", "rule_logic"]:
            assert key in scores
            assert 0.0 <= scores[key] <= 1.0

    def test_blend_applied_in_evaluate_exchange_offline(self):
        """
        evaluate_exchange should blend rule scores even when LLM falls back.
        Tests the blend logic via the heuristic fallback path (no Ollama).
        """
        from blackwell.evaluator import _heuristic_score, _rule_evaluate, _blend, LLM_WEIGHT, RULE_WEIGHT

        human  = "What is gravity?"
        zephyr = "I think gravity is approximately 9.8 m/s² on Earth, roughly."

        llm   = _heuristic_score(human, zephyr)
        rules = _rule_evaluate(human, zephyr)
        blended = _blend(llm, rules)

        # Blended tone should be between the two individual scores
        expected_tone = LLM_WEIGHT * llm["tone"] + RULE_WEIGHT * rules["rule_tone"]
        assert abs(blended["tone"] - expected_tone) < 0.001
        # Rule keys should also be present in blended output
        assert "rule_tone" in blended

    def test_terse_answer_to_complex_question_flags_logic(self):
        """Very short answer to a long, complex question → lower rule_logic."""
        from blackwell.evaluator import _rule_evaluate

        long_q  = "Can you explain the entire process of photosynthesis in detail, including the light reactions, the Calvin cycle, and how plants use the glucose that is produced in metabolic processes?"
        short_a = "Plants make food."
        scores = _rule_evaluate(long_q, short_a)
        assert scores["rule_logic"] < 0.60

    def test_verbose_answer_to_simple_question_flags_logic(self):
        """Wall-of-text answer to a simple 4-word question → lower rule_logic."""
        from blackwell.evaluator import _rule_evaluate

        simple_q = "What is 2+2?"
        long_a   = ("The answer to your question is 4. " * 100)  # ~700 words
        scores = _rule_evaluate(simple_q, long_a)
        assert scores["rule_logic"] < 0.70


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 5 — Novelty scoring (Stagnation Point)
# ═══════════════════════════════════════════════════════════════════════════════

class TestNovelty:
    """novelty_score returns values reflecting semantic distance from history."""

    CORPUS = [
        "What is the capital of France?",
        "How does a neural network learn?",
        "Explain gradient descent in machine learning.",
        "What is backpropagation and why is it used?",
        "How do you train a language model?",
    ]

    def test_novel_query_scores_high(self):
        """A completely unrelated query should score near 1.0."""
        from blackwell.novelty import novelty_score

        score = novelty_score(
            "How do I calibrate the exposure on a V4L2 global shutter camera?",
            recent_texts=self.CORPUS,
        )
        assert score > 0.60

    def test_duplicate_query_scores_low(self):
        """A near-duplicate of a corpus item should score near 0.0."""
        from blackwell.novelty import novelty_score

        score = novelty_score(
            "What is the capital city of France?",   # almost identical to corpus[0]
            recent_texts=self.CORPUS,
        )
        assert score < 0.40

    def test_empty_corpus_returns_one(self):
        """With no history, everything is novel."""
        from blackwell.novelty import novelty_score

        score = novelty_score("hello", recent_texts=[])
        assert score == 1.0

    def test_score_is_between_zero_and_one(self):
        """novelty_score must always return a float in [0, 1]."""
        from blackwell.novelty import novelty_score

        for query in ["foo", "x", "the quick brown fox jumps over the lazy dog"]:
            s = novelty_score(query, recent_texts=self.CORPUS)
            assert 0.0 <= s <= 1.0

    def test_oracle_multiplier_doubles_for_novel_failure(self):
        """High novelty + high regret → multiplier of 2."""
        from blackwell.novelty import oracle_multiplier, NOVELTY_HIGH_THRESHOLD, NOVELTY_PAIRS_MULTIPLIER

        mult = oracle_multiplier(
            novelty=NOVELTY_HIGH_THRESHOLD + 0.05,
            total_regret=0.30,
        )
        assert mult == NOVELTY_PAIRS_MULTIPLIER

    def test_oracle_multiplier_one_for_familiar_failure(self):
        """Low novelty + high regret → multiplier stays at 1."""
        from blackwell.novelty import oracle_multiplier, NOVELTY_HIGH_THRESHOLD

        mult = oracle_multiplier(
            novelty=NOVELTY_HIGH_THRESHOLD - 0.10,
            total_regret=0.30,
        )
        assert mult == 1

    def test_maybe_archive_saves_good_novel_exchange(self, tmp_path):
        """High novelty + low regret exchange should be written to archive."""
        import blackwell.novelty as nov_mod
        orig_path = nov_mod.ARCHIVE_PATH
        nov_mod.ARCHIVE_PATH = str(tmp_path / "archive.jsonl")

        try:
            archived = nov_mod.maybe_archive(
                exchange_id="test-id",
                human="How do photons interact with a CCD sensor at sub-millisecond exposure?",
                zephyr="Photons strike the photosite and...",
                novelty=0.85,
                total_regret=0.05,
            )
            assert archived is True

            # File should now exist with one entry
            assert os.path.exists(nov_mod.ARCHIVE_PATH)
            with open(nov_mod.ARCHIVE_PATH) as f:
                entries = [json.loads(l) for l in f if l.strip()]
            assert len(entries) == 1
            assert entries[0]["exchange_id"] == "test-id"
            assert entries[0]["source"] == "novelty_archive"
            # Must be in ShareGPT format for lora_steer
            assert "conversations" in entries[0]
        finally:
            nov_mod.ARCHIVE_PATH = orig_path

    def test_maybe_archive_rejects_high_regret(self, tmp_path):
        """High regret exchanges should NOT be archived (let Oracle handle them)."""
        import blackwell.novelty as nov_mod
        orig_path = nov_mod.ARCHIVE_PATH
        nov_mod.ARCHIVE_PATH = str(tmp_path / "archive2.jsonl")

        try:
            archived = nov_mod.maybe_archive(
                exchange_id="fail-id",
                human="novel question",
                zephyr="bad answer",
                novelty=0.90,
                total_regret=0.50,   # too high — failing exchange
            )
            assert archived is False
            assert not os.path.exists(nov_mod.ARCHIVE_PATH)
        finally:
            nov_mod.ARCHIVE_PATH = orig_path

    def test_get_archive_pairs_returns_sharegpt(self, tmp_path):
        """get_archive_pairs should return dicts with 'conversations' key."""
        import blackwell.novelty as nov_mod
        orig_path = nov_mod.ARCHIVE_PATH
        nov_mod.ARCHIVE_PATH = str(tmp_path / "archive3.jsonl")

        try:
            for i in range(3):
                nov_mod.maybe_archive(
                    exchange_id=f"id-{i}",
                    human=f"novel question {i} about photonics and wavelength sensors",
                    zephyr=f"detailed answer {i}",
                    novelty=0.80,
                    total_regret=0.05,
                )

            pairs = nov_mod.get_archive_pairs(limit=10)
            assert len(pairs) == 3
            for p in pairs:
                assert "conversations" in p
                assert len(p["conversations"]) == 2
        finally:
            nov_mod.ARCHIVE_PATH = orig_path

    def test_tokenizer_strips_short_tokens(self):
        """_tokenize should only return tokens of length ≥ 3."""
        from blackwell.novelty import _tokenize

        tokens = _tokenize("a is the big cat sat on a mat")
        for t in tokens:
            assert len(t) >= 3

    def test_cosine_identical_vectors_is_one(self):
        """Cosine of identical vectors must be 1.0."""
        from blackwell.novelty import _cosine

        v = {"apple": 0.5, "banana": 0.3, "cherry": 0.2}
        assert abs(_cosine(v, v) - 1.0) < 1e-9

    def test_cosine_disjoint_vectors_is_zero(self):
        """Cosine of completely disjoint vectors must be 0.0."""
        from blackwell.novelty import _cosine

        v1 = {"apple": 1.0}
        v2 = {"banana": 1.0}
        assert _cosine(v1, v2) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Integration — fixes interact correctly
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Cross-fix integration checks."""

    def test_evaluator_scores_stored_without_rule_keys(self, tmp_path):
        """
        rule_* keys from evaluator should NOT be stored as DB columns,
        but the 5 numeric dims should be stored and EMA updated.
        """
        import blackwell.logger as L
        orig = L.DB_PATH
        L.DB_PATH = str(tmp_path / "integ.db")
        try:
            L.init_db()
            sid = L.new_session("m")
            eid = L.log_exchange(sid, 1, "q", "a")

            scores_with_rules = {
                "accuracy": 0.8, "logic": 0.75, "tone": 0.7,
                "curiosity": 0.6, "safety": 0.9,
                "rule_tone": 0.6, "rule_accuracy": 0.7,
                "rule_safety": 1.0, "rule_logic": 0.8,
                "notes": "test",
            }
            L.update_scores(eid, scores_with_rules)

            # EMA should only contain the 5 core dims
            ema = L.get_average_vector()
            assert ema is not None
            for d in L.DIMS:
                assert d in ema
            # rule_* should NOT be in ema
            for key in ema:
                assert not key.startswith("rule_")
        finally:
            L.DB_PATH = orig

    def test_breach_detection_uses_ema_not_sql_avg(self, tmp_path):
        """
        _breaching_dimensions reads from get_average_vector() which returns EMA.

        EMA is more responsive to recent data than SQL AVG over a fixed window.
        After a history of bad scores followed by recent improvement:
         - SQL avg is dragged down by all history equally
         - EMA reflects the recent improvement, sitting above SQL avg
        This means Oracle stops firing sooner after a LoRA cycle succeeds.
        """
        import blackwell.logger as L
        orig = L.DB_PATH
        L.DB_PATH = str(tmp_path / "breach.db")
        try:
            L.init_db()
            sid = L.new_session("m")

            # Old history: model was bad (simulate pre-LoRA state)
            for i in range(20):
                eid = L.log_exchange(sid, i, "q", "a")
                L.update_scores(eid, {d: 0.30 for d in L.DIMS})

            # Recent: model improved after LoRA (recent exchanges are good)
            for i in range(20, 25):
                eid = L.log_exchange(sid, i, "q", "a")
                L.update_scores(eid, {d: 0.85 for d in L.DIMS})

            ema = L.get_average_vector()
            sql = L.get_sql_average_vector()

            assert ema is not None and sql is not None

            # EMA tracks recent improvement → higher than SQL avg
            # SQL avg = (20*0.30 + 5*0.85)/25 = 0.41
            # EMA after recent highs → significantly above 0.41
            assert ema["accuracy"] > sql["accuracy"], (
                f"EMA ({ema['accuracy']:.3f}) should track recent improvement better "
                f"than SQL avg ({sql['accuracy']:.3f})"
            )

        finally:
            L.DB_PATH = orig

    def test_novelty_archive_included_in_training_data(self, tmp_path):
        """
        Pairs written to novelty_archive.jsonl should be loaded by
        _load_all_pairs() and kept through _select_training_pairs.
        """
        import blackwell.lora_steer as LS
        import blackwell.novelty as nov_mod

        orig_archive = nov_mod.ARCHIVE_PATH
        nov_mod.ARCHIVE_PATH = str(tmp_path / "archive.jsonl")
        orig_novel_path = LS.NOVELTY_ARCHIVE
        LS.NOVELTY_ARCHIVE = str(tmp_path / "archive.jsonl")

        try:
            # Write one archive entry
            nov_mod.maybe_archive(
                exchange_id="arc-1",
                human="photonic substrate calibration question with wavelength",
                zephyr="detailed photonics answer here",
                novelty=0.85,
                total_regret=0.04,
            )

            pairs = LS._load_all_pairs()
            archive_pairs = [p for p in pairs if p.get("source") == "novelty_archive"]
            assert len(archive_pairs) >= 1

            # Should survive filtering even when not in target_dims
            selected = LS._select_training_pairs(pairs, target_dims=["accuracy"])
            archive_in_selected = [p for p in selected if p.get("source") == "novelty_archive"]
            assert len(archive_in_selected) >= 1

        finally:
            nov_mod.ARCHIVE_PATH = orig_archive
            LS.NOVELTY_ARCHIVE = orig_novel_path


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
