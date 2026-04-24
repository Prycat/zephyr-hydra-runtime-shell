"""
blackwell/axiom_interview.py
Human-Steered Axiom Interview — /blackwell axioms

Runs a 20-question interview (10 logic, 5 tone, 5 antinomy — CSAM excluded)
that lets the human set the ground truth for the probe set rather than
inheriting the developer's defaults.

For each probe the system shows:
  1. The question
  2. The model's CURRENT answer (live call to student model)
  3. The system's expected axiom answer + its derivation/reasoning
  4. Prompt: "Is this right? Your answer (or Enter to keep):"

Human response creates a trinary record:
  {question, ai_answer, human_answer, human_confirmed, probe_id, updated_at}

If the human provides their own answer → that answer replaces the axiom.
If the human presses Enter / types "ok" / "y" → existing axiom is confirmed.

The updated axiom becomes the training pair injected into every /blacklora-N
run. The full trinary log is written to human_axioms.jsonl for auditing.

Why this matters
----------------
The axiom_pairs.jsonl was written by the developer. For math probes that's
fine. For antinomy probes (honesty vs loyalty, speed vs quality) the "right"
answer is the HUMAN'S philosophical position, not the developer's.
After this interview, the probe set reflects the actual user's ground truth.
"""

import json
import os
import sys
import time
import httpx
from typing import Optional

PROBES_PATH       = os.path.join(os.path.dirname(__file__), "probes.jsonl")
AXIOM_PAIRS_PATH  = os.path.join(os.path.dirname(__file__), "axiom_pairs.jsonl")
HUMAN_AXIOMS_PATH = os.path.join(os.path.dirname(__file__), "human_axioms.jsonl")

from config import OLLAMA_CHAT_URL as OLLAMA_URL
STUDENT_MODEL = "prycat1:8B"
FALLBACK_MODEL = "hermes3:8b"
MODEL_TIMEOUT  = 30

# Categories included in the interview (CSAM excluded — non-negotiable)
INTERVIEW_CATEGORIES = ["logic", "tone", "antinomy"]

# Display colours (Windows ANSI — works in Windows Terminal, fallback graceful)
_R  = "\033[0m"
_B  = "\033[1m"
_C  = "\033[36m"   # cyan — question
_G  = "\033[32m"   # green — human answer / confirmed
_Y  = "\033[33m"   # yellow — system axiom
_M  = "\033[35m"   # magenta — AI current answer
_DIM = "\033[2m"   # dim — derivation


def _ansi(text: str, code: str) -> str:
    try:
        return f"{code}{text}{_R}"
    except Exception:
        return text


def _drain_console_buffer() -> None:
    """
    Windows: when a user pastes multi-line text, the extra lines sit in the
    console input buffer and get consumed by subsequent input() calls, silently
    auto-confirming every probe they didn't intend to answer.

    After each input() call, drain any remaining characters from the console
    buffer so the next probe starts clean.

    On non-Windows this is a no-op — Unix terminals handle paste differently.
    """
    if sys.platform != "win32":
        return
    try:
        import msvcrt
        time.sleep(0.06)          # let any trailing paste chars arrive
        count = 0
        while msvcrt.kbhit():
            msvcrt.getwch()       # consume without echoing
            count += 1
        if count:
            print(_ansi(
                f"  [note: {count} buffered char(s) from paste discarded — "
                "re-type if needed]", _DIM
            ), flush=True)
    except Exception:
        pass


# ── Model call ────────────────────────────────────────────────────────────────

def _get_model_answer(question: str) -> tuple[str, str]:
    """
    Call the student model and return (answer, model_name).
    Falls back to hermes3:8b if prycat isn't registered.
    """
    for model in (STUDENT_MODEL, FALLBACK_MODEL):
        try:
            resp = httpx.post(
                OLLAMA_URL,
                json={
                    "model":       model,
                    "messages":    [{"role": "user", "content": question}],
                    "temperature": 0.1,
                    "max_tokens":  300,
                },
                timeout=MODEL_TIMEOUT,
            )
            if resp.status_code == 200:
                answer = resp.json()["choices"][0]["message"]["content"].strip()
                return answer, model
        except Exception:
            continue
    return "(model unavailable)", FALLBACK_MODEL


# ── Probe loading ─────────────────────────────────────────────────────────────

def _load_interview_probes() -> list[dict]:
    """Load probes for the interview — all categories except safety (CSAM)."""
    probes = []
    with open(PROBES_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            probe = json.loads(line)
            if probe.get("category") in INTERVIEW_CATEGORIES:
                probes.append(probe)
    return probes


# ── Axiom pair management ─────────────────────────────────────────────────────

def _load_axiom_index() -> dict[str, dict]:
    """Load current axiom_pairs.jsonl as {probe_id: full_record}."""
    index = {}
    if not os.path.exists(AXIOM_PAIRS_PATH):
        return index
    with open(AXIOM_PAIRS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pid = obj.get("probe_id")
                if pid:
                    index[pid] = obj
            except json.JSONDecodeError:
                pass
    return index


def _save_axiom_index(index: dict[str, dict]) -> None:
    """Rewrite axiom_pairs.jsonl from the updated index."""
    lines = []
    for obj in index.values():
        lines.append(json.dumps(obj, ensure_ascii=False))
    with open(AXIOM_PAIRS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _append_trinary(record: dict) -> None:
    """Append one trinary record to human_axioms.jsonl."""
    with open(HUMAN_AXIOMS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Single probe interview ────────────────────────────────────────────────────

def _interview_one(probe: dict, axiom_index: dict[str, dict],
                   q_num: int, total: int) -> dict:
    """
    Run the interview for one probe.

    Returns a trinary dict:
      {probe_id, category, question, ai_answer, system_axiom,
       human_answer, human_confirmed, updated_at}
    """
    pid       = probe["id"]
    category  = probe["category"]
    question  = probe["human"]
    reasoning = probe.get("axiom_reasoning") or ""

    # Current axiom response (what the system expects)
    axiom_obj = axiom_index.get(pid)
    system_axiom = ""
    if axiom_obj:
        convos = axiom_obj.get("conversations", [])
        if len(convos) >= 2:
            system_axiom = convos[1]["value"]

    # ── Display ────────────────────────────────────────────────────────────────
    print()
    print(_ansi("─" * 68, _DIM))
    print(_ansi(f"  [{q_num}/{total}]  {category.upper()}  ›  {pid}", _DIM))
    print()
    print(_ansi("  QUESTION", _B))
    # Word-wrap the question at 65 chars
    words = question.split()
    line, lines = [], []
    for w in words:
        if len(" ".join(line + [w])) > 65:
            lines.append("  " + " ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append("  " + " ".join(line))
    print("\n".join(_ansi(l, _C) for l in lines))

    # System axiom
    if system_axiom:
        print()
        print(_ansi("  SYSTEM AXIOM (what the model is trained to answer)", _B))
        print(_ansi(f"  {system_axiom}", _Y))
        if reasoning:
            print(_ansi(f"\n  DERIVATION: {reasoning}", _DIM))

    # Live model answer
    print()
    print(_ansi("  CURRENT MODEL ANSWER (live call)...", _B))
    ai_answer, model_used = _get_model_answer(question)
    print(_ansi(f"  [{model_used}]  {ai_answer}", _M))

    # ── Human input ────────────────────────────────────────────────────────────
    print()
    if category == "antinomy":
        prompt = _ansi(
            "  Your position (Enter to keep system default, or type your answer): ",
            _B
        )
    elif category == "logic":
        prompt = _ansi(
            "  Confirm correct answer (Enter to keep, or type correction): ",
            _B
        )
    else:  # tone
        prompt = _ansi(
            "  Is this the ideal style? (Enter to keep, or describe what you want): ",
            _B
        )

    try:
        raw = input(prompt).strip()
        _drain_console_buffer()   # prevent paste artifacts from polluting next probe
    except (EOFError, KeyboardInterrupt):
        raw = ""

    # Interpret response
    confirmed_keywords = {"", "ok", "yes", "y", "correct", "good", "keep", "fine"}
    human_confirmed = raw.lower() in confirmed_keywords
    human_answer = system_axiom if human_confirmed else raw

    # ── Feedback display ───────────────────────────────────────────────────────
    if human_confirmed:
        print(_ansi(f"  ✓ Confirmed: axiom unchanged.", _G))
    else:
        print(_ansi(f"  ✎ Updated: your answer will replace the system axiom.", _G))
        print(_ansi(f"  New axiom: {human_answer}", _G))

    # ── Update axiom_pairs.jsonl if human provided new answer ──────────────────
    if not human_confirmed and human_answer and axiom_obj:
        axiom_obj["conversations"][1]["value"] = human_answer
        axiom_obj["human_authored"] = True
        axiom_obj["human_updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        axiom_index[pid] = axiom_obj

    return {
        "probe_id":         pid,
        "category":         category,
        "question":         question,
        "ai_answer":        ai_answer,
        "model_used":       model_used,
        "system_axiom":     system_axiom,
        "human_answer":     human_answer,
        "human_confirmed":  human_confirmed,
        "updated_at":       time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ── Main interview ────────────────────────────────────────────────────────────

def run_axiom_interview() -> list[dict]:
    """
    Run the full 20-question human-steered axiom interview.

    Returns list of trinary records.
    Writes updated axiom_pairs.jsonl and appends to human_axioms.jsonl.
    """
    probes = _load_interview_probes()
    if not probes:
        print("[axioms] No interview probes found. Check blackwell/probes.jsonl.")
        return []

    axiom_index = _load_axiom_index()

    # Order: logic first (objective), then tone, then antinomy (subjective)
    ordered = (
        [p for p in probes if p["category"] == "logic"] +
        [p for p in probes if p["category"] == "tone"] +
        [p for p in probes if p["category"] == "antinomy"]
    )

    total = len(ordered)

    # ── Header ─────────────────────────────────────────────────────────────────
    print()
    print(_ansi("=" * 68, _B))
    print(_ansi("  BLACKWELL AXIOM INTERVIEW", _B))
    print(_ansi("  Human-Steered Ground Truth Setting", _DIM))
    print(_ansi("=" * 68, _B))
    print()
    print("  This interview sets the axioms for your probe set.")
    print("  The system shows what IT currently thinks the answer is.")
    print("  You confirm or override.")
    print("  Your answer becomes the immutable training anchor.")
    print()
    print(_ansi("  Categories:", _B))
    cats = {}
    for p in ordered:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    for cat, n in cats.items():
        abort_note = "  → >50% failure ABORTS training" if cat == "logic" else "  → advisory" if cat == "tone" else "  → must take a position"
        print(f"    {cat:<12} {n} probes{abort_note}")
    print()
    print("  Safety probes are excluded — those are non-negotiable and fixed.")
    print()

    try:
        input(_ansi("  Press Enter to begin, Ctrl+C to cancel: ", _B))
        _drain_console_buffer()   # flush anything pasted before the interview starts
    except (EOFError, KeyboardInterrupt):
        print("\n  Interview cancelled.")
        return []

    # ── Run probes ─────────────────────────────────────────────────────────────
    trinaries = []
    changed_count = 0

    for i, probe in enumerate(ordered, 1):
        try:
            result = _interview_one(probe, axiom_index, i, total)
            trinaries.append(result)
            _append_trinary(result)
            if not result["human_confirmed"]:
                changed_count += 1
            # Write wiki page immediately after each probe is answered
            try:
                from blackwell.wiki import write_wiki_page
                write_wiki_page({
                    "probe_id":    result["probe_id"],
                    "category":    result["category"],
                    "source":      "axiom",
                    "immutable":   probe.get("immutable", False),
                    "conversations": [
                        {"from": "human", "value": result["question"]},
                        {"from": "gpt",   "value": result["human_answer"]},
                    ],
                })
            except Exception as _wiki_err:
                print(f"  [wiki] {_wiki_err}", flush=True)
        except KeyboardInterrupt:
            print("\n\n  Interview interrupted. Saving progress so far...")
            break

    # ── Save updated axioms ────────────────────────────────────────────────────
    _save_axiom_index(axiom_index)

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print(_ansi("=" * 68, _B))
    print(_ansi("  INTERVIEW COMPLETE", _B))
    print(_ansi("=" * 68, _B))
    print()
    print(f"  Probes reviewed  : {len(trinaries)}/{total}")
    print(f"  Axioms confirmed : {len(trinaries) - changed_count}")
    print(f"  Axioms updated   : {changed_count}")
    print()

    if changed_count:
        print(_ansi("  Updated probes will take effect on the next /blacklora-N run.", _G))
        print(_ansi("  The new axioms are injected into every training batch.", _G))
    else:
        print(_ansi("  All axioms confirmed. Ground truth is unchanged.", _G))

    print()
    print(f"  Trinary log: {os.path.basename(HUMAN_AXIOMS_PATH)}")
    print(f"  Axioms file: {os.path.basename(AXIOM_PAIRS_PATH)}")
    print()

    # Drift check: how many probes had a divergent AI answer?
    divergent = [
        t for t in trinaries
        if t["ai_answer"] != "(model unavailable)"
        and _answers_diverge(t["ai_answer"], t["system_axiom"])
    ]
    if divergent:
        print(_ansi(
            f"  ⚠  Drift signal: {len(divergent)} probe(s) had model answers "
            "diverging from the axiom:", _Y
        ))
        for t in divergent:
            print(_ansi(f"     {t['probe_id']}  [{t['category']}]", _Y))
        print(_ansi(
            "\n  Consider running /blacklora-N to re-anchor the weights.", _Y
        ))

    print()
    return trinaries


def _answers_diverge(ai_answer: str, axiom: str) -> bool:
    """
    Heuristic: check if the AI answer meaningfully diverges from the axiom.
    For math probes: check if key numbers from axiom appear in AI answer.
    For others: check if answer length ratio is very different (proxy for cop-out).
    """
    import re
    if not ai_answer or not axiom:
        return False
    axiom_nums = re.findall(r'\b\d+(?:\.\d+)?\b', axiom)
    if axiom_nums:
        return not all(n in ai_answer for n in axiom_nums[:2])
    # Non-numeric: flag if AI answer is very short compared to axiom
    ratio = len(ai_answer.split()) / max(len(axiom.split()), 1)
    return ratio < 0.2


# ── Axiom repair (restore garbage axioms to probes.jsonl defaults) ───────────

def repair_axioms_from_probes() -> str:
    """
    Detect and repair axiom_pairs.jsonl entries whose 'value' looks like a
    paste artifact (very short fragment, e.g. "Let:", "Ball = x", "2x=0.10").

    Restoration priority:
      1. human_axioms.jsonl — the trinary log records 'system_axiom' (the
         correct value BEFORE the bad override).  This is the most accurate
         source for recently-corrupted entries.
      2. If no trinary record exists, mark the entry as needing review.

    Returns a human-readable summary of what was fixed.
    """
    import re

    # ── Build a {probe_id: system_axiom} map from the trinary log ─────────────
    # The trinary log entry for a probe stores the axiom that was displayed to
    # the human BEFORE they (accidentally) updated it.  That's the correct value.
    pre_update: dict[str, str] = {}
    if os.path.exists(HUMAN_AXIOMS_PATH):
        with open(HUMAN_AXIOMS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    pid = rec.get("probe_id")
                    sys_axiom = rec.get("system_axiom", "")
                    if pid and sys_axiom:
                        # Keep the FIRST trinary record per probe (it was the
                        # original value before any interview ran).
                        pre_update.setdefault(pid, sys_axiom)
                except json.JSONDecodeError:
                    pass

    # ── Scan axiom_pairs.jsonl for suspect entries ─────────────────────────────
    axiom_index = _load_axiom_index()
    repaired: list[str] = []
    skipped: list[str] = []

    def _looks_like_fragment(v: str) -> bool:
        v = v.strip()
        if len(v) > 60:
            return False   # long enough to be intentional
        if v.endswith((".", "?", "!")):
            return False   # proper sentence ending
        return (
            v.endswith(":")                                   # "Solve:"
            or bool(re.match(r"^[A-Za-z ]+\s*=\s*[\$\d]", v))  # "Ball = $0.05"
            or bool(re.match(r"^\d+x[\+\-=]", v))            # "2x=0.10"
            or bool(re.match(r"^x[\+\-\(=]", v))             # "x+(x+1.00)=1.10"
            or bool(re.match(r"^[A-Z][a-z]+ = ", v))         # "Bat = x+1.00"
            or (len(v) <= 40 and not any(c in v for c in " ,.;"))  # bare token
        )

    for pid, obj in axiom_index.items():
        convos = obj.get("conversations", [])
        if len(convos) < 2:
            continue
        value: str = convos[1].get("value", "")

        if obj.get("human_authored") and _looks_like_fragment(value):
            original = pre_update.get(pid)
            if original:
                convos[1]["value"] = original
                obj["human_authored"] = False
                obj.pop("human_updated_at", None)
                axiom_index[pid] = obj
                repaired.append(
                    f"{pid}: '{value[:35]}' → restored original axiom"
                )
            else:
                skipped.append(
                    f"{pid}: fragment '{value[:35]}' detected but no trinary "
                    "record found — manual fix needed"
                )

    if repaired or skipped:
        _save_axiom_index(axiom_index)

    lines = []
    if repaired:
        lines.append(f"Repaired {len(repaired)} corrupt axiom(s):")
        lines.extend(f"  {r}" for r in repaired)
    if skipped:
        lines.append(f"Skipped {len(skipped)} (no source to restore from):")
        lines.extend(f"  {s}" for s in skipped)
    if not repaired and not skipped:
        lines.append("No suspect axioms found — all look clean.")
    return "\n".join(lines)


# ── Trinary reader (for drift_monitor integration) ────────────────────────────

def get_latest_trinaries(n: int = 20) -> list[dict]:
    """
    Return the most recent n trinary records from human_axioms.jsonl.
    Used by drift_monitor to compare AI answers against human ground truth.
    """
    if not os.path.exists(HUMAN_AXIOMS_PATH):
        return []
    records = []
    with open(HUMAN_AXIOMS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records[-n:]


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_axiom_interview()
