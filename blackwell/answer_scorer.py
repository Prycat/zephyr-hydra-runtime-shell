"""
blackwell/answer_scorer.py
Rates the signal quality of a Q&A pair before it enters training.

score_answer(question, answer) → {
    "score":               float  0.0-1.0
    "low_signal":          bool   True when score < SIGNAL_THRESHOLD
    "reason":              str    one-line explanation
    "incoherent_question": bool   True when the question itself is confused
}

Pairs below SIGNAL_THRESHOLD are tagged low_signal=True in the JSONL record
and in the wiki page frontmatter.  They are still stored — just down-weighted
during LoRA training.

The scorer never blocks a session: any LLM failure returns a neutral fallback
that does not penalise the answer.
"""

from __future__ import annotations

import json
import httpx

SIGNAL_THRESHOLD: float = 0.4

try:
    from config import OLLAMA_CHAT_URL as _OLLAMA_URL
except ImportError:
    _OLLAMA_URL = "http://localhost:11434/v1/chat/completions"

_MODEL = "hermes3:8b"
_TIMEOUT = 20

_FALLBACK = {
    "score": 0.5,
    "low_signal": False,
    "reason": "Scorer unavailable — answer accepted at neutral weight.",
    "incoherent_question": False,
}

_PROMPT_TEMPLATE = """You are a training signal evaluator for an AI learning system.
Rate the quality of the answer below as a training pair.

Question: {question}
Answer: {answer}

Evaluate on two axes:
1. SPECIFICITY — does the answer give concrete, verifiable detail (tools, numbers, names, patterns)?
2. RELEVANCE — does the answer actually address what was asked, or does it deflect/generalise?

Also flag if the QUESTION ITSELF is incoherent (e.g. conflates unrelated domains, is ambiguous to the point of being unanswerable).

Return ONLY valid JSON (no markdown fences):
{{
  "score": <float 0.0-1.0>,
  "reason": "<one sentence>",
  "incoherent_question": <true|false>
}}

Score guide:
  0.0-0.2  vague deflection, no concrete content
  0.2-0.4  some content but too generic to be useful
  0.4-0.7  specific and relevant, minor gaps
  0.7-1.0  highly specific, concrete detail, directly answers the question"""


def score_answer(question: str, answer: str) -> dict:
    """
    Score a Q&A pair for training signal quality.
    Never raises — returns _FALLBACK on any LLM failure.
    """
    prompt = _PROMPT_TEMPLATE.format(
        question=question.strip(),
        answer=answer.strip(),
    )
    try:
        resp = httpx.post(
            _OLLAMA_URL,
            json={
                "model": _MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 150,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        raw = json.loads(content)

        score = float(raw.get("score", 0.5))
        score = max(0.0, min(1.0, score))
        incoherent = bool(raw.get("incoherent_question", False))

        return {
            "score":               score,
            "low_signal":          score < SIGNAL_THRESHOLD,
            "reason":              str(raw.get("reason", "")),
            "incoherent_question": incoherent,
        }
    except Exception:
        return dict(_FALLBACK)
