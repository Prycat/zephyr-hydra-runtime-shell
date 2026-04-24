"""
Tests for blackwell/answer_scorer.py

Design under test:
  score_answer(question, answer) → dict
    {
      "score":      float   0.0-1.0
      "low_signal": bool    True when score < SIGNAL_THRESHOLD
      "reason":     str     one-line explanation
      "incoherent_question": bool  True when question itself is poorly formed
    }

  SIGNAL_THRESHOLD = 0.4
"""

import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def _fake_response(content: str, status: int = 200):
    mock = MagicMock()
    mock.status_code = status
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    return mock


_HIGH_SIGNAL = json.dumps({
    "score": 0.85,
    "reason": "Answer is specific and technically detailed.",
    "incoherent_question": False,
})

_LOW_SIGNAL = json.dumps({
    "score": 0.2,
    "reason": "Answer is vague and provides no concrete detail.",
    "incoherent_question": False,
})

_INCOHERENT = json.dumps({
    "score": 0.15,
    "reason": "Question conflates two unrelated domains.",
    "incoherent_question": True,
})


# ── 1. Return shape ───────────────────────────────────────────────────────────

class TestScoreAnswerShape(unittest.TestCase):

    def test_returns_dict(self):
        from blackwell.answer_scorer import score_answer
        with patch('blackwell.answer_scorer.httpx.post',
                   return_value=_fake_response(_HIGH_SIGNAL)):
            result = score_answer("What language do you use?", "Python for ML.")
        self.assertIsInstance(result, dict)

    def test_has_score_field(self):
        from blackwell.answer_scorer import score_answer
        with patch('blackwell.answer_scorer.httpx.post',
                   return_value=_fake_response(_HIGH_SIGNAL)):
            result = score_answer("Q", "A")
        self.assertIn("score", result)

    def test_has_low_signal_field(self):
        from blackwell.answer_scorer import score_answer
        with patch('blackwell.answer_scorer.httpx.post',
                   return_value=_fake_response(_HIGH_SIGNAL)):
            result = score_answer("Q", "A")
        self.assertIn("low_signal", result)

    def test_has_reason_field(self):
        from blackwell.answer_scorer import score_answer
        with patch('blackwell.answer_scorer.httpx.post',
                   return_value=_fake_response(_HIGH_SIGNAL)):
            result = score_answer("Q", "A")
        self.assertIn("reason", result)

    def test_has_incoherent_question_field(self):
        from blackwell.answer_scorer import score_answer
        with patch('blackwell.answer_scorer.httpx.post',
                   return_value=_fake_response(_HIGH_SIGNAL)):
            result = score_answer("Q", "A")
        self.assertIn("incoherent_question", result)


# ── 2. Score range ────────────────────────────────────────────────────────────

class TestScoreRange(unittest.TestCase):

    def test_score_is_float(self):
        from blackwell.answer_scorer import score_answer
        with patch('blackwell.answer_scorer.httpx.post',
                   return_value=_fake_response(_HIGH_SIGNAL)):
            result = score_answer("Q", "A")
        self.assertIsInstance(result["score"], float)

    def test_score_between_zero_and_one(self):
        from blackwell.answer_scorer import score_answer
        with patch('blackwell.answer_scorer.httpx.post',
                   return_value=_fake_response(_HIGH_SIGNAL)):
            result = score_answer("Q", "A")
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)

    def test_high_score_not_low_signal(self):
        from blackwell.answer_scorer import score_answer
        with patch('blackwell.answer_scorer.httpx.post',
                   return_value=_fake_response(_HIGH_SIGNAL)):
            result = score_answer("Q", "A")
        self.assertFalse(result["low_signal"])

    def test_low_score_is_low_signal(self):
        from blackwell.answer_scorer import score_answer
        with patch('blackwell.answer_scorer.httpx.post',
                   return_value=_fake_response(_LOW_SIGNAL)):
            result = score_answer("Q", "A")
        self.assertTrue(result["low_signal"])


# ── 3. Threshold boundary ─────────────────────────────────────────────────────

class TestThreshold(unittest.TestCase):

    def test_score_exactly_at_threshold_not_low_signal(self):
        from blackwell.answer_scorer import score_answer, SIGNAL_THRESHOLD
        at_threshold = json.dumps({
            "score": SIGNAL_THRESHOLD,
            "reason": "At boundary.",
            "incoherent_question": False,
        })
        with patch('blackwell.answer_scorer.httpx.post',
                   return_value=_fake_response(at_threshold)):
            result = score_answer("Q", "A")
        self.assertFalse(result["low_signal"])

    def test_score_just_below_threshold_is_low_signal(self):
        from blackwell.answer_scorer import score_answer, SIGNAL_THRESHOLD
        below = json.dumps({
            "score": round(SIGNAL_THRESHOLD - 0.01, 3),
            "reason": "Just below.",
            "incoherent_question": False,
        })
        with patch('blackwell.answer_scorer.httpx.post',
                   return_value=_fake_response(below)):
            result = score_answer("Q", "A")
        self.assertTrue(result["low_signal"])

    def test_threshold_constant_exists(self):
        from blackwell.answer_scorer import SIGNAL_THRESHOLD
        self.assertIsInstance(SIGNAL_THRESHOLD, float)
        self.assertGreater(SIGNAL_THRESHOLD, 0.0)
        self.assertLess(SIGNAL_THRESHOLD, 1.0)


# ── 4. Incoherent question detection ─────────────────────────────────────────

class TestIncoherentQuestion(unittest.TestCase):

    def test_incoherent_question_flagged(self):
        from blackwell.answer_scorer import score_answer
        with patch('blackwell.answer_scorer.httpx.post',
                   return_value=_fake_response(_INCOHERENT)):
            result = score_answer(
                "How does CUDA compare to Loguru for data pipelines?",
                "They are both useful tools."
            )
        self.assertTrue(result["incoherent_question"])

    def test_coherent_question_not_flagged(self):
        from blackwell.answer_scorer import score_answer
        with patch('blackwell.answer_scorer.httpx.post',
                   return_value=_fake_response(_HIGH_SIGNAL)):
            result = score_answer(
                "What logging framework do you use in your ML pipeline?",
                "I use structlog with JSON output piped to a dashboard."
            )
        self.assertFalse(result["incoherent_question"])


# ── 5. Fallback on LLM failure ────────────────────────────────────────────────

class TestFallback(unittest.TestCase):

    def test_returns_dict_on_network_error(self):
        from blackwell.answer_scorer import score_answer
        with patch('blackwell.answer_scorer.httpx.post',
                   side_effect=Exception("network down")):
            result = score_answer("Q", "A")
        self.assertIsInstance(result, dict)

    def test_fallback_has_all_required_fields(self):
        from blackwell.answer_scorer import score_answer
        with patch('blackwell.answer_scorer.httpx.post',
                   side_effect=Exception("timeout")):
            result = score_answer("Q", "A")
        for field in ("score", "low_signal", "reason", "incoherent_question"):
            self.assertIn(field, result)

    def test_fallback_score_in_range(self):
        from blackwell.answer_scorer import score_answer
        with patch('blackwell.answer_scorer.httpx.post',
                   side_effect=Exception("timeout")):
            result = score_answer("Q", "A")
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)

    def test_fallback_does_not_mark_low_signal(self):
        """Scoring failure should not penalise a potentially good answer."""
        from blackwell.answer_scorer import score_answer
        with patch('blackwell.answer_scorer.httpx.post',
                   side_effect=Exception("unavailable")):
            result = score_answer("Q", "A")
        self.assertFalse(result["low_signal"])


# ── 6. planning.py integration ────────────────────────────────────────────────

class TestPlanningIntegration(unittest.TestCase):

    def test_training_record_has_answer_score(self):
        """JSONL records written per-answer must include answer_score."""
        from blackwell import planning

        dummy_wm = {
            "skills": [], "patterns": [], "gaps": [],
            "updated_at": None, "sessions": 0,
        }
        written_records = []

        class _FakeFile:
            def write(self, s):
                written_records.append(s)
            def __enter__(self): return self
            def __exit__(self, *a): pass

        with patch.object(planning, 'load_coding_world_model', return_value=dummy_wm), \
             patch.object(planning, 'save_coding_world_model'), \
             patch.object(planning, 'synthesise_coding_update',
                          return_value={"skills": [], "patterns": [], "gaps": []}), \
             patch.object(planning, 'generate_next_coding_question',
                          return_value="Which language?"), \
             patch.object(planning, '_drain_console_buffer'), \
             patch('builtins.input', side_effect=["Python and CUDA", "done"]), \
             patch('builtins.open', return_value=_FakeFile()), \
             patch('blackwell.wiki.write_wiki_page'), \
             patch('blackwell.answer_scorer.httpx.post',
                   return_value=_fake_response(_HIGH_SIGNAL)):

            planning.run_coding_planning_session()

        self.assertEqual(len(written_records), 1)
        record = json.loads(written_records[0])
        self.assertIn("answer_score", record)
        self.assertIn("low_signal", record)

    def test_scorer_called_per_answer(self):
        """score_answer must be called once per recorded answer."""
        from blackwell import planning
        import blackwell.answer_scorer as scorer_mod

        dummy_wm = {
            "skills": [], "patterns": [], "gaps": [],
            "updated_at": None, "sessions": 0,
        }

        with patch.object(planning, 'load_coding_world_model', return_value=dummy_wm), \
             patch.object(planning, 'save_coding_world_model'), \
             patch.object(planning, 'synthesise_coding_update',
                          return_value={"skills": [], "patterns": [], "gaps": []}), \
             patch.object(planning, 'generate_next_coding_question',
                          return_value="Which language?"), \
             patch.object(planning, '_drain_console_buffer'), \
             patch('builtins.input', side_effect=["Python", "Rust", "done"]), \
             patch('builtins.open', unittest.mock.mock_open()), \
             patch('blackwell.wiki.write_wiki_page'), \
             patch.object(scorer_mod, 'score_answer',
                          return_value={"score": 0.8, "low_signal": False,
                                        "reason": "ok", "incoherent_question": False}) as mock_score:

            planning.run_coding_planning_session()

        self.assertEqual(mock_score.call_count, 2)


if __name__ == '__main__':
    unittest.main()
