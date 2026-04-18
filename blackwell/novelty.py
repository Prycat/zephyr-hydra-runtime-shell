"""
blackwell/novelty.py
Fix 5 — Stagnation Point: Novelty scoring via TF-IDF cosine distance.

Scores each incoming exchange against recent conversation history.
Returns 0.0 (seen before) → 1.0 (completely new territory).

High novelty + high regret  → double synthesis pairs, prioritise Oracle.
High novelty + low regret   → archive as positive anchor for replay.

No external dependencies — uses stdlib math + Counter only.
"""

import math
import re
import json
import os
import threading
from collections import Counter
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────
NOVELTY_WINDOW           = 100    # compare against this many recent exchanges
NOVELTY_HIGH_THRESHOLD   = 0.60   # above this = genuinely new territory
NOVELTY_LOW_REGRET_CAP   = 0.15   # below this regret = exchange is a good anchor
NOVELTY_PAIRS_MULTIPLIER = 2      # multiply n_pairs when novel+failing
ARCHIVE_MAX_SIZE         = 500    # cap archive to avoid unbounded growth

ARCHIVE_PATH = os.path.join(os.path.dirname(__file__), "novelty_archive.jsonl")
_archive_lock = threading.Lock()


# ── Text utilities ─────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lower-case alphabetic tokens, length ≥ 3."""
    return re.findall(r'\b[a-z]{3,}\b', text.lower())


def _build_idf(docs: list[list[str]]) -> dict[str, float]:
    """
    Compute inverse document frequency for a corpus of token lists.
    idf(t) = log(N / (df(t) + 1)) + 1   (smooth, never zero)
    """
    n = len(docs)
    doc_freq: Counter = Counter()
    for tokens in docs:
        doc_freq.update(set(tokens))
    return {t: math.log(n / (freq + 1)) + 1.0 for t, freq in doc_freq.items()}


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """TF-IDF vector for a token list given a pre-computed IDF table."""
    tf = Counter(tokens)
    total = sum(tf.values()) or 1
    return {t: (count / total) * idf.get(t, 1.0) for t, count in tf.items()}


def _cosine(v1: dict[str, float], v2: dict[str, float]) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    common = set(v1) & set(v2)
    if not common:
        return 0.0
    dot  = sum(v1[k] * v2[k] for k in common)
    mag1 = math.sqrt(sum(x * x for x in v1.values()))
    mag2 = math.sqrt(sum(x * x for x in v2.values()))
    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0
    return dot / (mag1 * mag2)


# ── Public API ─────────────────────────────────────────────────────────────────

def novelty_score(human: str, recent_texts: Optional[list[str]] = None) -> float:
    """
    Compute novelty of *human* turn against recent exchange history.

    Parameters
    ----------
    human : str
        The incoming human message to score.
    recent_texts : list[str] | None
        Pre-fetched recent human messages.  If None, fetched from DB.

    Returns
    -------
    float
        0.0 = identical to something seen recently
        1.0 = completely unlike anything in the window
    """
    if recent_texts is None:
        try:
            from blackwell.logger import get_recent_exchanges
            recent = get_recent_exchanges(limit=NOVELTY_WINDOW)
            recent_texts = [r["human"] for r in recent if r.get("human")]
        except Exception:
            return 1.0   # can't load history → treat as novel

    if not recent_texts:
        return 1.0

    # Tokenise corpus
    corpus_docs = [_tokenize(t) for t in recent_texts]
    query_tokens = _tokenize(human)

    if not query_tokens:
        return 0.5   # empty / non-textual — neutral score

    idf = _build_idf(corpus_docs + [query_tokens])
    query_vec = _tfidf_vector(query_tokens, idf)

    max_sim = 0.0
    for doc_tokens in corpus_docs:
        if not doc_tokens:
            continue
        doc_vec = _tfidf_vector(doc_tokens, idf)
        sim = _cosine(query_vec, doc_vec)
        if sim > max_sim:
            max_sim = sim

    return round(1.0 - max_sim, 4)


def oracle_multiplier(novelty: float, total_regret: float) -> int:
    """
    Return the synthesis pair multiplier for this exchange.

    Novel failure  → 2×  (we found something the model hasn't solved in new territory)
    Familiar failure → 1×  (normal Oracle cycle)
    Novel success  → 0   (archive it, don't train on failure)
    """
    is_novel   = novelty >= NOVELTY_HIGH_THRESHOLD
    is_failing = total_regret > 0.0   # caller passes filtered regret

    if is_novel and is_failing:
        return NOVELTY_PAIRS_MULTIPLIER
    return 1


def maybe_archive(
    exchange_id: str,
    human: str,
    zephyr: str,
    novelty: float,
    total_regret: float,
) -> bool:
    """
    Archive high-novelty, low-regret exchanges as positive anchors.
    These are replayed during LoRA training to prevent attractor collapse.

    Returns True if the exchange was archived.
    """
    if novelty < NOVELTY_HIGH_THRESHOLD:
        return False
    if total_regret > NOVELTY_LOW_REGRET_CAP:
        return False   # novel but still failing — let Oracle handle it

    entry = {
        "exchange_id": exchange_id,
        "human":       human,
        "zephyr":      zephyr,
        "novelty":     novelty,
        "regret":      total_regret,
        # ShareGPT format so lora_steer can load it directly
        "conversations": [
            {"from": "human", "value": human},
            {"from": "gpt",   "value": zephyr},
        ],
        "source":     "novelty_archive",
        "target_dim": "general",
    }

    with _archive_lock:
        # Read current size to enforce cap
        count = 0
        if os.path.exists(ARCHIVE_PATH):
            with open(ARCHIVE_PATH, "r", encoding="utf-8") as f:
                count = sum(1 for line in f if line.strip())

        if count >= ARCHIVE_MAX_SIZE:
            return False   # archive full — skip rather than rotate for now

        with open(ARCHIVE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    return True


def get_archive_pairs(limit: int = 50) -> list[dict]:
    """Load the most recent *limit* archive entries in ShareGPT format."""
    if not os.path.exists(ARCHIVE_PATH):
        return []
    entries = []
    with open(ARCHIVE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    # Return the most recent, up to limit
    return entries[-limit:]


if __name__ == "__main__":
    # Quick smoke test
    corpus = [
        "What is the capital of France?",
        "How does a neural network learn?",
        "Explain gradient descent.",
        "What is backpropagation?",
    ]
    queries = [
        ("What is Paris known for?",                 "similar to France question"),
        ("How do I calibrate a V4L2 camera sensor?", "novel — photonics/hardware"),
        ("What is 2 + 2?",                           "very different"),
    ]
    for q, label in queries:
        score = novelty_score(q, recent_texts=corpus)
        print(f"  novelty={score:.3f}  [{label}]")
        print(f"    '{q[:60]}'")
