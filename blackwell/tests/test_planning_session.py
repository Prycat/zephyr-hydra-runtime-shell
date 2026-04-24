"""
Tests for coding blackwell planning session fixes:
  1. Questions generated one at a time from prior Q&A context
  2. _drain_console_buffer available in planning module
  3. Paste buffer drained at session end (no self-talk leaking to main loop)
"""

import os
import sys
import json
import types
import unittest
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ── helper: build a fake httpx response ──────────────────────────────────────

def _fake_response(content: str, status: int = 200):
    mock = MagicMock()
    mock.status_code = status
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    return mock


# ── 1. generate_next_coding_question exists and returns a string ──────────────

class TestGenerateNextCodingQuestion(unittest.TestCase):

    def test_function_exists(self):
        from blackwell import planning
        self.assertTrue(
            hasattr(planning, 'generate_next_coding_question'),
            "planning module must expose generate_next_coding_question()"
        )

    def test_returns_string_for_empty_prior_qa(self):
        from blackwell.planning import generate_next_coding_question
        with patch('blackwell.planning.httpx.post') as mock_post:
            mock_post.return_value = _fake_response("What languages do you use?")
            result = generate_next_coding_question(prior_qa=[])
        self.assertIsInstance(result, str)
        self.assertGreater(len(result.strip()), 5)

    def test_prior_qa_included_in_prompt(self):
        """The prior Q&A context must be visible in the prompt sent to the model."""
        from blackwell.planning import generate_next_coding_question
        prior = [
            {"question": "Which language?", "answer": "Python mostly"},
        ]
        captured_prompt = []
        def capture(url, **kwargs):
            msgs = kwargs.get('json', {}).get('messages', [])
            for m in msgs:
                captured_prompt.append(m.get('content', ''))
            return _fake_response("Tell me about your Python projects.")

        with patch('blackwell.planning.httpx.post', side_effect=capture):
            generate_next_coding_question(prior_qa=prior)

        full_prompt = "\n".join(captured_prompt)
        self.assertIn("Python mostly", full_prompt,
                      "Prior answer must appear in prompt for follow-up generation")

    def test_fallback_on_error(self):
        """On HTTP failure, function must return a non-empty fallback string."""
        from blackwell.planning import generate_next_coding_question
        with patch('blackwell.planning.httpx.post', side_effect=Exception("network down")):
            result = generate_next_coding_question(prior_qa=[])
        self.assertIsInstance(result, str)
        self.assertGreater(len(result.strip()), 5)

    def test_question_differs_with_rich_context(self):
        """With a full Q&A history, the generated question should reference context."""
        from blackwell.planning import generate_next_coding_question
        prior = [
            {"question": "Languages?", "answer": "Rust and Python"},
            {"question": "Frameworks?", "answer": "Axum and FastAPI"},
        ]
        responses = iter([
            "What kind of projects do you build with Axum?",
        ])
        with patch('blackwell.planning.httpx.post') as mock_post:
            mock_post.return_value = _fake_response(next(responses))
            result = generate_next_coding_question(prior_qa=prior)
        self.assertIsInstance(result, str)


# ── 2. _drain_console_buffer is importable from planning ─────────────────────

class TestDrainConsoleBuffer(unittest.TestCase):

    def test_drain_function_exists(self):
        from blackwell import planning
        self.assertTrue(
            hasattr(planning, '_drain_console_buffer'),
            "planning module must expose _drain_console_buffer()"
        )

    def test_drain_is_callable(self):
        from blackwell.planning import _drain_console_buffer
        self.assertTrue(callable(_drain_console_buffer))

    def test_drain_runs_without_error_on_non_windows(self):
        from blackwell.planning import _drain_console_buffer
        with patch('sys.platform', 'linux'):
            # Should be a no-op and not raise
            _drain_console_buffer()

    def test_drain_runs_without_error_on_windows_no_kbhit(self):
        from blackwell.planning import _drain_console_buffer
        mock_msvcrt = types.ModuleType('msvcrt')
        mock_msvcrt.kbhit = MagicMock(return_value=False)
        mock_msvcrt.getwch = MagicMock()
        with patch('sys.platform', 'win32'), \
             patch.dict(sys.modules, {'msvcrt': mock_msvcrt}):
            _drain_console_buffer()
        mock_msvcrt.getwch.assert_not_called()

    def test_drain_consumes_buffered_chars_on_windows(self):
        from blackwell.planning import _drain_console_buffer
        mock_msvcrt = types.ModuleType('msvcrt')
        # Simulate 3 buffered chars then empty
        mock_msvcrt.kbhit = MagicMock(side_effect=[True, True, True, False])
        mock_msvcrt.getwch = MagicMock(return_value='\n')
        with patch('sys.platform', 'win32'), \
             patch.dict(sys.modules, {'msvcrt': mock_msvcrt}), \
             patch('time.sleep'):
            _drain_console_buffer()
        self.assertEqual(mock_msvcrt.getwch.call_count, 3)


# ── 3. Session drains buffer after final answer (prevents self-talk) ──────────

class TestSessionDrainsAtEnd(unittest.TestCase):

    def test_drain_called_after_last_answer(self):
        """
        run_coding_planning_session must call _drain_console_buffer at least
        once after the final input() — so paste residue can't leak into the
        main chat loop.
        """
        from blackwell import planning

        # Patch everything that touches disk / network / stdin
        dummy_wm = {
            "skills": [], "patterns": [], "gaps": [],
            "updated_at": None, "sessions": 0,
        }

        with patch.object(planning, 'load_coding_world_model', return_value=dummy_wm), \
             patch.object(planning, 'save_coding_world_model'), \
             patch.object(planning, 'synthesise_coding_update',
                          return_value={"skills": [], "patterns": [], "gaps": []}), \
             patch.object(planning, 'generate_next_coding_question',
                          return_value="What do you build?"), \
             patch('builtins.open', unittest.mock.mock_open()), \
             patch('builtins.input', side_effect=["I build web apps", "done"]), \
             patch.object(planning, '_drain_console_buffer') as mock_drain:

            planning.run_coding_planning_session()

        self.assertGreater(mock_drain.call_count, 0,
                           "_drain_console_buffer must be called during/after the session")


if __name__ == '__main__':
    unittest.main()
