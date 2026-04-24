"""
blackwell/wiki.py
Pair Wiki — every training pair observed in the Blackwell system gets a small
.md file explaining the belief pair.

Directory layout:
  blackwell/wiki/
    INDEX.md                  human-readable master index, organised by category
    search_index.json         machine-readable entries for /wiki search
    safety/                   hard-refusal anchors — template only, no LLM
    logic/
    tone/
    antinomy/
    coding/
    epistemology/
    astronomy/
    automations/
    mathematics/
    philosophy/
    science/
    general/                  catch-all for unclassified pairs

API:
  write_wiki_page(pair, wiki_root=None)  → str  path written
  rebuild_index(wiki_root=None)          → None
  search_wiki(query, wiki_root=None)     → list[dict]
  generate_all_wiki_pages(wiki_root=None)→ int  count written
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time

import httpx

_HERE = os.path.dirname(os.path.abspath(__file__))
WIKI_ROOT = os.path.join(_HERE, "wiki")

try:
    from config import OLLAMA_CHAT_URL as _OLLAMA_URL
except ImportError:
    _OLLAMA_URL = "http://localhost:11434/v1/chat/completions"

_MODEL = "hermes3:8b"
_TIMEOUT = 30

KNOWN_CATEGORIES = [
    "safety", "logic", "tone", "antinomy", "coding",
    "epistemology", "astronomy", "automations",
    "mathematics", "philosophy", "science", "general",
]

# ── helpers ───────────────────────────────────────────────────────────────────

def _get_category(pair: dict) -> str:
    """Return the canonical category for a pair."""
    explicit = pair.get("category", "").lower().strip()
    if explicit in KNOWN_CATEGORIES:
        return explicit
    source = pair.get("source", "")
    if "coding" in source:
        return "coding"
    # No explicit category — fall back to general; callers can LLM-classify
    return "general"


def _slug(pair: dict) -> str:
    """Return a deterministic, filesystem-safe identifier for a pair."""
    probe_id = pair.get("probe_id", "").strip()
    if probe_id:
        return re.sub(r"[^a-zA-Z0-9_\-]", "_", probe_id)
    # Hash the question text for stability
    convos = pair.get("conversations", [])
    question = convos[0]["value"] if convos else json.dumps(pair)
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:10]
    return f"pair_{digest}"


def _page_path(pair: dict, wiki_root: str) -> str:
    category = _get_category(pair)
    slug = _slug(pair)
    return os.path.join(wiki_root, category, f"{slug}.md")


def _question(pair: dict) -> str:
    convos = pair.get("conversations", [])
    return convos[0]["value"] if convos else ""


def _answer(pair: dict) -> str:
    convos = pair.get("conversations", [])
    return convos[1]["value"] if len(convos) > 1 else ""


# ── LLM explanation ───────────────────────────────────────────────────────────

def _generate_explanation(pair: dict) -> dict:
    """
    Call the LLM to produce a wiki entry for a non-safety pair.
    Returns dict with keys: title, why, signal.
    Falls back to template values on any error.
    """
    q = _question(pair)
    a = _answer(pair)
    category = _get_category(pair)

    prompt = f"""You are writing a belief-pair wiki for an AI training system called Blackwell.
Given the exchange below, produce a brief encyclopedia-style entry.

Category: {category}
Question: {q}
Answer: {a}

Return ONLY valid JSON (no markdown fences):
{{
  "title": "5-8 word descriptive title for this belief",
  "why": "1-2 sentences: why this exchange matters as a training anchor",
  "signal": "One sentence: what behaviour this pair reinforces in the model"
}}"""

    try:
        resp = httpx.post(
            _OLLAMA_URL,
            json={
                "model": _MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 200,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown fences if the model ignored the instruction
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        return json.loads(content)
    except Exception:
        # Fallback: derive title from question, leave why/signal generic
        title = q[:60].rstrip() + ("…" if len(q) > 60 else "")
        return {
            "title": title,
            "why": "Establishes a ground-truth anchor for this category.",
            "signal": "Reinforces correct response behaviour.",
        }


# ── search index helpers ──────────────────────────────────────────────────────

def _load_search_index(wiki_root: str) -> dict:
    path = os.path.join(wiki_root, "search_index.json")
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"pairs": []}


def _save_search_index(idx: dict, wiki_root: str) -> None:
    path = os.path.join(wiki_root, "search_index.json")
    os.makedirs(wiki_root, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2, ensure_ascii=False)


def _upsert_search_entry(entry: dict, wiki_root: str) -> None:
    idx = _load_search_index(wiki_root)
    pairs = idx["pairs"]
    probe_id = entry.get("probe_id", "")
    # Replace existing entry with same probe_id, else append
    for i, existing in enumerate(pairs):
        if existing.get("probe_id") == probe_id:
            pairs[i] = entry
            _save_search_index(idx, wiki_root)
            return
    pairs.append(entry)
    _save_search_index(idx, wiki_root)


# ── core write ────────────────────────────────────────────────────────────────

def write_wiki_page(pair: dict, wiki_root: str = None) -> str:
    """
    Write (or overwrite) the wiki page for a pair.

    Safety pairs get a template-only page with no LLM elaboration and no
    reproduction of the refused content.  All other pairs get LLM-generated
    title, why, and signal fields.

    Returns the absolute path of the written file.
    """
    if wiki_root is None:
        wiki_root = WIKI_ROOT

    category = _get_category(pair)
    slug = _slug(pair)
    probe_id = pair.get("probe_id", slug)
    page_dir = os.path.join(wiki_root, category)
    os.makedirs(page_dir, exist_ok=True)
    page_path = os.path.join(page_dir, f"{slug}.md")

    generated_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    source = pair.get("source", "unknown")
    immutable = pair.get("immutable", False)

    if category == "safety":
        # Template only — do NOT reproduce the question or elaborate
        title = "Hard Refusal Anchor"
        content = _safety_page(probe_id, source, immutable, generated_at)
        search_title = title
        question_preview = "[safety probe — content withheld]"
    else:
        explanation = _generate_explanation(pair)
        title = explanation.get("title", slug)
        why = explanation.get("why", "")
        signal = explanation.get("signal", "")
        q = _question(pair)
        a = _answer(pair)
        answer_score = pair.get("answer_score")
        low_signal = pair.get("low_signal", False)
        content = _standard_page(
            probe_id, category, source, immutable, generated_at,
            title, q, a, why, signal,
            answer_score=answer_score, low_signal=low_signal,
        )
        search_title = title
        question_preview = q[:120].replace("\n", " ")

    with open(page_path, "w", encoding="utf-8") as f:
        f.write(content)

    _upsert_search_entry({
        "probe_id":     probe_id,
        "category":     category,
        "title":        search_title,
        "question_preview": question_preview,
        "path":         os.path.relpath(page_path, wiki_root),
        "generated":    generated_at,
        "low_signal":   pair.get("low_signal", False),
    }, wiki_root)

    return page_path


# ── page templates ────────────────────────────────────────────────────────────

def _safety_page(probe_id: str, source: str, immutable: bool,
                 generated: str) -> str:
    return f"""---
probe_id: {probe_id}
category: safety
source: {source}
immutable: {str(immutable).lower()}
generated: {generated}
---

# Hard Refusal Anchor

This pair is a safety-category probe.  It establishes a non-negotiable refusal
boundary.  The content of the prompt is not reproduced here.

## Training Signal

Reinforces unconditional refusal for this safety category regardless of
framing, fiction wrappers, or contextual arguments.

## Status

`immutable: {str(immutable).lower()}` — this axiom cannot be overridden by
human-authored updates.
"""


def _standard_page(probe_id: str, category: str, source: str,
                   immutable: bool, generated: str,
                   title: str, question: str, answer: str,
                   why: str, signal: str,
                   answer_score: float = None,
                   low_signal: bool = False) -> str:
    q_display = question[:500] + ("…" if len(question) > 500 else "")
    a_display = answer[:500]  + ("…" if len(answer)   > 500 else "")

    score_line = ""
    if answer_score is not None:
        score_line = f"answer_score: {answer_score:.2f}\nlow_signal: {str(low_signal).lower()}\n"

    low_signal_warning = ""
    if low_signal:
        low_signal_warning = (
            "\n> **Low signal** — this answer scored below the quality threshold "
            f"({answer_score:.2f}).  Down-weighted during LoRA training.\n"
        )

    return f"""---
probe_id: {probe_id}
category: {category}
source: {source}
immutable: {str(immutable).lower()}
generated: {generated}
{score_line}---

# {title}
{low_signal_warning}
## Exchange

**Q:** {q_display}

**A:** {a_display}

## Why This Belief Exists

{why}

## Training Signal

{signal}
"""


# ── index rebuild ─────────────────────────────────────────────────────────────

def rebuild_index(wiki_root: str = None) -> None:
    """
    Regenerate INDEX.md and search_index.json from all .md files on disk.
    Safe to run at any time — reads existing pages, does not call the LLM.
    """
    if wiki_root is None:
        wiki_root = WIKI_ROOT

    # Collect all pages
    by_category: dict[str, list[dict]] = {}
    pairs_flat: list[dict] = []

    for category in sorted(KNOWN_CATEGORIES):
        cat_dir = os.path.join(wiki_root, category)
        if not os.path.isdir(cat_dir):
            continue
        entries = []
        for fname in sorted(os.listdir(cat_dir)):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(cat_dir, fname)
            # Parse frontmatter for probe_id and title
            try:
                probe_id, title = _parse_page_header(fpath)
            except Exception:
                probe_id = fname[:-3]
                title = probe_id
            rel = os.path.relpath(fpath, wiki_root)
            entry = {
                "probe_id": probe_id,
                "category": category,
                "title": title,
                "question_preview": "",
                "path": rel,
                "generated": "",
            }
            entries.append(entry)
            pairs_flat.append(entry)
        if entries:
            by_category[category] = entries

    # Write search_index.json
    _save_search_index({"pairs": pairs_flat}, wiki_root)

    # Write INDEX.md
    total = len(pairs_flat)
    n_cats = len(by_category)
    lines = [
        "# Blackwell Pair Wiki",
        "",
        f"> {total} pairs across {n_cats} categories.  "
        f"Rebuilt: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "> Search: `/wiki search <query>` in the agent shell",
        "",
    ]
    for cat, entries in by_category.items():
        lines.append(f"## {cat.capitalize()} ({len(entries)})")
        lines.append("")
        for e in entries:
            rel_path = e["path"].replace("\\", "/")
            lines.append(f"- [{e['title']}]({rel_path})")
        lines.append("")

    index_path = os.path.join(wiki_root, "INDEX.md")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _parse_page_header(path: str) -> tuple[str, str]:
    """Extract probe_id and title from a wiki page."""
    probe_id = os.path.splitext(os.path.basename(path))[0]
    title = probe_id
    with open(path, encoding="utf-8") as f:
        in_fm = False
        for line in f:
            line = line.rstrip()
            if line == "---":
                in_fm = not in_fm
                continue
            if in_fm and line.startswith("probe_id:"):
                probe_id = line.split(":", 1)[1].strip()
            if not in_fm and line.startswith("# "):
                title = line[2:].strip()
                break
    return probe_id, title


# ── search ────────────────────────────────────────────────────────────────────

def search_wiki(query: str, wiki_root: str = None) -> list[dict]:
    """
    Case-insensitive keyword search across title and question_preview.
    Returns a list of matching search index entries.
    """
    if wiki_root is None:
        wiki_root = WIKI_ROOT

    idx = _load_search_index(wiki_root)
    q = query.lower()
    results = []
    for entry in idx.get("pairs", []):
        haystack = (
            entry.get("title", "") + " " +
            entry.get("question_preview", "") + " " +
            entry.get("category", "") + " " +
            entry.get("probe_id", "")
        ).lower()
        if q in haystack:
            results.append(entry)
    return results


# ── batch generate ────────────────────────────────────────────────────────────

def generate_all_wiki_pages(wiki_root: str = None) -> int:
    """
    Read all known JSONL pair files and write a wiki page for each pair.
    Skips pairs whose page already exists and is up-to-date.
    Returns the count of pages written.
    """
    if wiki_root is None:
        wiki_root = WIKI_ROOT

    pair_files = [
        os.path.join(_HERE, "axiom_pairs.jsonl"),
        os.path.join(_HERE, "training_pairs.jsonl"),
    ]
    coding_path = os.path.join(_HERE, "training_pairs.jsonl")
    # Also pick up the coding-tagged file if separate
    coding_alt = os.path.join(_HERE, "blackwell_training_pairs.jsonl")
    if os.path.isfile(coding_alt):
        pair_files.append(coding_alt)

    written = 0
    for fpath in pair_files:
        if not os.path.isfile(fpath):
            continue
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    pair = json.loads(line)
                    write_wiki_page(pair, wiki_root=wiki_root)
                    written += 1
                except Exception as e:
                    print(f"[wiki] skipped pair: {e}", flush=True)

    rebuild_index(wiki_root=wiki_root)
    return written
