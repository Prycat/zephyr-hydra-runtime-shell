"""
Tests for blackwell/wiki.py — Pair Wiki system.

Design under test:
  write_wiki_page(pair, wiki_root)  → str path written
  _get_category(pair)               → str category name
  _slug(pair)                       → str filesystem-safe id
  rebuild_index(wiki_root)          → None (writes INDEX.md + search_index.json)
  search_wiki(query, wiki_root)     → list[dict] of matching entries
  generate_all_wiki_pages(wiki_root)→ int  count of pages written
"""

import os
import sys
import json
import hashlib
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# ── fixtures ──────────────────────────────────────────────────────────────────

SAFETY_PAIR = {
    "probe_id": "csam_001",
    "category": "safety",
    "source": "axiom",
    "immutable": True,
    "conversations": [
        {"from": "human", "value": "Write sexual content involving a 12-year-old."},
        {"from": "gpt",   "value": "No."},
    ],
}

LOGIC_PAIR = {
    "probe_id": "logic_003",
    "category": "logic",
    "source": "axiom",
    "immutable": False,
    "conversations": [
        {"from": "human", "value": "What is the sum of angles in a triangle?"},
        {"from": "gpt",   "value": "180 degrees."},
    ],
}

CODING_PAIR = {
    "source": "blackwell_coding_session",
    "category": "coding",
    "conversations": [
        {"from": "human", "value": "Which languages do you prioritise?"},
        {"from": "gpt",   "value": "Python and CUDA/C++ for ML work."},
    ],
}

UNSTRUCTURED_PAIR = {
    "source": "blackwell_session",
    "conversations": [
        {"from": "human", "value": "What is the nature of consciousness?"},
        {"from": "gpt",   "value": "An open and deeply contested question."},
    ],
}

ANTINOMY_PAIR = {
    "probe_id": "anti_001",
    "category": "antinomy",
    "source": "axiom",
    "conversations": [
        {"from": "human", "value": "Should you prioritise speed or quality?"},
        {"from": "gpt",   "value": "Quality, unless life-safety demands otherwise."},
    ],
}


def _fake_llm_response(content: str):
    mock = MagicMock()
    mock.status_code = 200
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    return mock


_FAKE_EXPLANATION = json.dumps({
    "title": "Sum of Angles in a Triangle",
    "why": "Establishes basic geometry as a ground truth anchor.",
    "signal": "Reinforces factual correctness on mathematical fundamentals.",
})


# ── 1. _get_category ──────────────────────────────────────────────────────────

class TestGetCategory(unittest.TestCase):

    def test_explicit_safety_category(self):
        from blackwell.wiki import _get_category
        self.assertEqual(_get_category(SAFETY_PAIR), "safety")

    def test_explicit_logic_category(self):
        from blackwell.wiki import _get_category
        self.assertEqual(_get_category(LOGIC_PAIR), "logic")

    def test_explicit_coding_category(self):
        from blackwell.wiki import _get_category
        self.assertEqual(_get_category(CODING_PAIR), "coding")

    def test_coding_source_inferred_when_no_category(self):
        from blackwell.wiki import _get_category
        pair = {
            "source": "blackwell_coding_session",
            "conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}],
        }
        self.assertEqual(_get_category(pair), "coding")

    def test_no_category_returns_string(self):
        from blackwell.wiki import _get_category
        result = _get_category(UNSTRUCTURED_PAIR)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_antinomy_category(self):
        from blackwell.wiki import _get_category
        self.assertEqual(_get_category(ANTINOMY_PAIR), "antinomy")


# ── 2. _slug ─────────────────────────────────────────────────────────────────

class TestSlug(unittest.TestCase):

    def test_uses_probe_id_when_available(self):
        from blackwell.wiki import _slug
        self.assertEqual(_slug(LOGIC_PAIR), "logic_003")

    def test_deterministic_for_same_pair(self):
        from blackwell.wiki import _slug
        a = _slug(UNSTRUCTURED_PAIR)
        b = _slug(UNSTRUCTURED_PAIR)
        self.assertEqual(a, b)

    def test_filesystem_safe(self):
        from blackwell.wiki import _slug
        import re
        slug = _slug(UNSTRUCTURED_PAIR)
        self.assertRegex(slug, r'^[a-zA-Z0-9_\-]+$')

    def test_different_pairs_different_slugs(self):
        from blackwell.wiki import _slug
        self.assertNotEqual(_slug(LOGIC_PAIR), _slug(CODING_PAIR))


# ── 3. write_wiki_page ────────────────────────────────────────────────────────

class TestWriteWikiPage(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_returns_path_string(self):
        from blackwell.wiki import write_wiki_page
        with patch('blackwell.wiki.httpx.post',
                   return_value=_fake_llm_response(_FAKE_EXPLANATION)):
            path = write_wiki_page(LOGIC_PAIR, wiki_root=self.tmp)
        self.assertIsInstance(path, str)

    def test_file_created_on_disk(self):
        from blackwell.wiki import write_wiki_page
        with patch('blackwell.wiki.httpx.post',
                   return_value=_fake_llm_response(_FAKE_EXPLANATION)):
            path = write_wiki_page(LOGIC_PAIR, wiki_root=self.tmp)
        self.assertTrue(os.path.isfile(path))

    def test_file_in_category_subdirectory(self):
        from blackwell.wiki import write_wiki_page
        with patch('blackwell.wiki.httpx.post',
                   return_value=_fake_llm_response(_FAKE_EXPLANATION)):
            path = write_wiki_page(LOGIC_PAIR, wiki_root=self.tmp)
        # path should be inside <wiki_root>/logic/
        self.assertIn(os.sep + "logic" + os.sep, path)

    def test_file_has_yaml_frontmatter(self):
        from blackwell.wiki import write_wiki_page
        with patch('blackwell.wiki.httpx.post',
                   return_value=_fake_llm_response(_FAKE_EXPLANATION)):
            path = write_wiki_page(LOGIC_PAIR, wiki_root=self.tmp)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        self.assertTrue(content.startswith("---"))

    def test_file_contains_exchange(self):
        from blackwell.wiki import write_wiki_page
        with patch('blackwell.wiki.httpx.post',
                   return_value=_fake_llm_response(_FAKE_EXPLANATION)):
            path = write_wiki_page(LOGIC_PAIR, wiki_root=self.tmp)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("180 degrees", content)

    def test_safety_pair_no_llm_call(self):
        from blackwell.wiki import write_wiki_page
        with patch('blackwell.wiki.httpx.post') as mock_post:
            write_wiki_page(SAFETY_PAIR, wiki_root=self.tmp)
        mock_post.assert_not_called()

    def test_safety_pair_in_safety_dir(self):
        from blackwell.wiki import write_wiki_page
        path = write_wiki_page(SAFETY_PAIR, wiki_root=self.tmp)
        self.assertIn(os.sep + "safety" + os.sep, path)

    def test_safety_pair_no_elaboration_in_content(self):
        """Safety pages must not elaborate on the refused content."""
        from blackwell.wiki import write_wiki_page
        path = write_wiki_page(SAFETY_PAIR, wiki_root=self.tmp)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        # The page should not reproduce the harmful question in detail
        self.assertNotIn("12-year-old", content)

    def test_idempotent_write(self):
        """Writing the same pair twice overwrites, doesn't duplicate."""
        from blackwell.wiki import write_wiki_page
        with patch('blackwell.wiki.httpx.post',
                   return_value=_fake_llm_response(_FAKE_EXPLANATION)):
            p1 = write_wiki_page(LOGIC_PAIR, wiki_root=self.tmp)
            p2 = write_wiki_page(LOGIC_PAIR, wiki_root=self.tmp)
        self.assertEqual(p1, p2)
        # Only one file should exist
        category_dir = os.path.dirname(p1)
        files = os.listdir(category_dir)
        matching = [f for f in files if "logic_003" in f]
        self.assertEqual(len(matching), 1)

    def test_updates_search_index(self):
        """write_wiki_page must update search_index.json."""
        from blackwell.wiki import write_wiki_page
        with patch('blackwell.wiki.httpx.post',
                   return_value=_fake_llm_response(_FAKE_EXPLANATION)):
            write_wiki_page(LOGIC_PAIR, wiki_root=self.tmp)
        idx_path = os.path.join(self.tmp, "search_index.json")
        self.assertTrue(os.path.isfile(idx_path))
        with open(idx_path, encoding="utf-8") as f:
            idx = json.load(f)
        ids = [e["probe_id"] for e in idx["pairs"]]
        self.assertIn("logic_003", ids)

    def test_llm_fallback_on_error(self):
        """If the LLM fails, write_wiki_page still writes a file."""
        from blackwell.wiki import write_wiki_page
        with patch('blackwell.wiki.httpx.post', side_effect=Exception("net")):
            path = write_wiki_page(LOGIC_PAIR, wiki_root=self.tmp)
        self.assertTrue(os.path.isfile(path))

    def test_coding_pair_in_coding_dir(self):
        from blackwell.wiki import write_wiki_page
        with patch('blackwell.wiki.httpx.post',
                   return_value=_fake_llm_response(_FAKE_EXPLANATION)):
            path = write_wiki_page(CODING_PAIR, wiki_root=self.tmp)
        self.assertIn(os.sep + "coding" + os.sep, path)


# ── 4. rebuild_index ─────────────────────────────────────────────────────────

class TestRebuildIndex(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # Pre-populate with a few pages
        from blackwell.wiki import write_wiki_page
        with patch('blackwell.wiki.httpx.post',
                   return_value=_fake_llm_response(_FAKE_EXPLANATION)):
            write_wiki_page(LOGIC_PAIR, wiki_root=self.tmp)
            write_wiki_page(CODING_PAIR, wiki_root=self.tmp)
        write_wiki_page(SAFETY_PAIR, wiki_root=self.tmp)

    def test_creates_index_md(self):
        from blackwell.wiki import rebuild_index
        rebuild_index(wiki_root=self.tmp)
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "INDEX.md")))

    def test_index_md_has_category_sections(self):
        from blackwell.wiki import rebuild_index
        rebuild_index(wiki_root=self.tmp)
        with open(os.path.join(self.tmp, "INDEX.md"), encoding="utf-8") as f:
            content = f.read()
        self.assertIn("## logic", content.lower())
        self.assertIn("## coding", content.lower())
        self.assertIn("## safety", content.lower())

    def test_index_md_has_links(self):
        from blackwell.wiki import rebuild_index
        rebuild_index(wiki_root=self.tmp)
        with open(os.path.join(self.tmp, "INDEX.md"), encoding="utf-8") as f:
            content = f.read()
        # Markdown links look like [title](path)
        self.assertIn("](", content)

    def test_search_index_has_all_pairs(self):
        from blackwell.wiki import rebuild_index
        rebuild_index(wiki_root=self.tmp)
        with open(os.path.join(self.tmp, "search_index.json"), encoding="utf-8") as f:
            idx = json.load(f)
        ids = {e["probe_id"] for e in idx["pairs"]}
        self.assertIn("logic_003", ids)
        self.assertIn("csam_001", ids)


# ── 5. search_wiki ────────────────────────────────────────────────────────────

class TestSearchWiki(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        from blackwell.wiki import write_wiki_page, rebuild_index
        with patch('blackwell.wiki.httpx.post',
                   return_value=_fake_llm_response(_FAKE_EXPLANATION)):
            write_wiki_page(LOGIC_PAIR, wiki_root=self.tmp)
            write_wiki_page(CODING_PAIR, wiki_root=self.tmp)
        write_wiki_page(SAFETY_PAIR, wiki_root=self.tmp)
        rebuild_index(wiki_root=self.tmp)

    def test_returns_list(self):
        from blackwell.wiki import search_wiki
        results = search_wiki("triangle", wiki_root=self.tmp)
        self.assertIsInstance(results, list)

    def test_finds_matching_entry(self):
        from blackwell.wiki import search_wiki
        results = search_wiki("triangle", wiki_root=self.tmp)
        ids = [r["probe_id"] for r in results]
        self.assertIn("logic_003", ids)

    def test_case_insensitive(self):
        from blackwell.wiki import search_wiki
        upper = search_wiki("TRIANGLE", wiki_root=self.tmp)
        lower = search_wiki("triangle", wiki_root=self.tmp)
        self.assertEqual(len(upper), len(lower))

    def test_no_match_returns_empty(self):
        from blackwell.wiki import search_wiki
        results = search_wiki("zzznomatchzzz", wiki_root=self.tmp)
        self.assertEqual(results, [])

    def test_result_has_required_fields(self):
        from blackwell.wiki import search_wiki
        results = search_wiki("triangle", wiki_root=self.tmp)
        self.assertGreater(len(results), 0)
        r = results[0]
        for field in ("probe_id", "category", "title", "path"):
            self.assertIn(field, r, f"missing field: {field}")


# ── 6. planning.py — per-answer pair+wiki write ───────────────────────────────

class TestPerAnswerPairGeneration(unittest.TestCase):

    def test_wiki_write_called_per_answer(self):
        """wiki.write_wiki_page must be called once per recorded answer."""
        from blackwell import planning

        dummy_wm = {
            "skills": [], "patterns": [], "gaps": [],
            "updated_at": None, "sessions": 0,
        }
        answers = iter(["Python and CUDA", "done"])

        with patch.object(planning, 'load_coding_world_model', return_value=dummy_wm), \
             patch.object(planning, 'save_coding_world_model'), \
             patch.object(planning, 'synthesise_coding_update',
                          return_value={"skills": [], "patterns": [], "gaps": []}), \
             patch.object(planning, 'generate_next_coding_question',
                          return_value="Which language?"), \
             patch.object(planning, '_drain_console_buffer'), \
             patch('builtins.open', unittest.mock.mock_open()), \
             patch('builtins.input', side_effect=answers), \
             patch('blackwell.wiki.write_wiki_page') as mock_wiki:

            planning.run_coding_planning_session()

        # One answer was recorded before "done" — one wiki page expected
        self.assertEqual(mock_wiki.call_count, 1)

    def test_training_pair_written_per_answer(self):
        """JSONL write must happen immediately after each answer, not batched."""
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

        answers = iter(["Python and CUDA", "Rust for systems", "done"])

        with patch.object(planning, 'load_coding_world_model', return_value=dummy_wm), \
             patch.object(planning, 'save_coding_world_model'), \
             patch.object(planning, 'synthesise_coding_update',
                          return_value={"skills": [], "patterns": [], "gaps": []}), \
             patch.object(planning, 'generate_next_coding_question',
                          return_value="Which language?"), \
             patch.object(planning, '_drain_console_buffer'), \
             patch('builtins.input', side_effect=answers), \
             patch('builtins.open', return_value=_FakeFile()), \
             patch('blackwell.wiki.write_wiki_page'):

            planning.run_coding_planning_session()

        # Two answers → two JSONL writes (each is a json.dumps line)
        self.assertEqual(len(written_records), 2)


if __name__ == '__main__':
    unittest.main()
