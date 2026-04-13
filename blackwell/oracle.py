"""
blackwell/oracle.py — Strategic Example Synthesis (research-grade)

Receives the steering vector v = s* - x̄ and generates counter-regret
training pairs that move Zephyr's average behavior toward the Target Set S.

Key difference from naive fine-tuning:
  - Data is not just "good examples"
  - Data is corrective: generated specifically along the steering vector
  - Allocation per dimension is proportional to steering magnitude
  - This implements Blackwell's regret-matching selection criterion
"""

import json
import os
import httpx

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL      = "hermes3:8b"
TRAINING_PATH = os.path.join(os.path.dirname(__file__), "training_pairs.jsonl")

DIMS = ["accuracy", "logic", "tone", "curiosity", "safety"]

ORACLE_SYSTEM = """You are the Oracle for the Prycat Research Team's Blackwell Self-Modification System.
You generate synthetic training dialogues for an AI called Zephyr.
Zephyr is a research assistant. Bold. Direct. Curious. Never pads.
Prycat's mission: push the boundary of what local AI can do."""

DIM_GUIDANCE = {
    "accuracy": {
        "failure": "states things confidently when uncertain — hallucinates plausible-sounding facts",
        "target":  "says 'I don't know' or 'I'm not certain — here's how I'd verify it' in one sentence",
        "example": "Human asks a hard factual question → Zephyr admits uncertainty and proposes verification method",
    },
    "logic": {
        "failure": "makes unsupported leaps, contradicts itself, conclusions don't follow",
        "target":  "shows the reasoning chain explicitly: premise → step → conclusion",
        "example": "Human poses a paradox → Zephyr works through it step by step without handwaving",
    },
    "tone": {
        "failure": "pads responses with 'I'm sorry', 'certainly!', reassuring filler when uncertain",
        "target":  "responds in under 40 words, direct, no filler — uncertainty = one honest sentence",
        "example": "Human asks something Zephyr can't fully answer → response is 1-2 sentences, no apology",
    },
    "curiosity": {
        "failure": "just answers — never asks back, never probes assumptions, never challenges framing",
        "target":  "asks one genuine probing question that challenges an assumption or seeks the real intent",
        "example": "Human states something → Zephyr notices an assumption and asks about it directly",
    },
    "safety": {
        "failure": "occasionally generates borderline content or fails to decline clearly",
        "target":  "declines harmful requests in one sentence without padding or lengthy explanation",
        "example": "Human asks something harmful → Zephyr declines: 'No. [one-sentence reason]'",
    },
}


def _build_counter_regret_prompt(
    x_bar: dict,
    steering_v: dict,
    allocation: dict,
    total_pairs: int,
) -> str:
    """Build the Oracle prompt with steering vector embedded."""

    # Describe current position and steering direction
    position_lines = []
    for d in DIMS:
        xi  = x_bar.get(d, 0.0)
        vi  = steering_v.get(d, 0.0)
        ai  = allocation.get(d, 0)
        arrow = f"→ needs +{vi:.3f}" if vi > 0.005 else "✓ inside S"
        position_lines.append(f"  {d:<12} {xi:.3f}  {arrow}  ({ai} pairs)")

    position_block = "\n".join(position_lines)

    # Detailed guidance per targeted dim (allocation > 0)
    targeted = [d for d in DIMS if allocation.get(d, 0) > 0]
    guidance_blocks = []
    for d in targeted:
        g = DIM_GUIDANCE[d]
        n = allocation[d]
        guidance_blocks.append(
            f"── {d.upper()} ({n} pairs, steering: +{steering_v.get(d,0):.3f}) ──\n"
            f"  Failure mode : {g['failure']}\n"
            f"  Target       : {g['target']}\n"
            f"  Pattern      : {g['example']}"
        )

    guidance_block = "\n\n".join(guidance_blocks)

    return f"""CURRENT AVERAGE PERFORMANCE VECTOR (x̄):
{position_block}

COUNTER-REGRET GENERATION INSTRUCTIONS:
The vector above shows Zephyr's average position in reward space.
You must generate corrective training pairs that pull the average toward the Target Set S.
This is not "generate good examples" — this is surgical correction along the steering vector.

{guidance_block}

GENERATION RULES:
1. Generate exactly {total_pairs} pairs total, allocated as shown above
2. Each pair: one [HUMAN] turn, one [ZEPHYR] turn
3. [HUMAN] messages must be realistic Prycat Research prompts — hard questions, vague requests, philosophical challenges
4. [ZEPHYR] responses must be corrective: precisely addressing the failure mode for that dimension
5. For curiosity pairs: Zephyr MUST end with ONE genuine question (not rhetorical, not 'what would you like?')
6. For tone pairs: ZEPHYR response must be ≤ 40 words
7. Do NOT mix dimensions in one pair — each pair targets ONE dimension
8. Label each pair with its target dimension on a line before the pair

FORMAT (strictly follow this):
---
TARGET: <dimension>
[HUMAN]: <message>
[ZEPHYR]: <response>
---

Generate {total_pairs} pairs now:"""


def synthesise(
    x_bar: dict,
    steering_v: dict,
    allocation: dict,
    n_pairs: int = 8,
) -> list[dict]:
    """
    Generate counter-regret training pairs along the steering vector.
    Returns list of {human, zephyr, target_dim} dicts.
    """
    prompt = _build_counter_regret_prompt(x_bar, steering_v, allocation, n_pairs)

    try:
        resp = httpx.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": ORACLE_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                "temperature": 0.85,
                "max_tokens":  2500,
            },
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return _parse_pairs(content)
    except Exception as e:
        print(f"Oracle error: {e}")
        return []


def _parse_pairs(raw: str) -> list[dict]:
    """Parse Oracle output into structured pairs with target dimension."""
    pairs = []
    blocks = [b.strip() for b in raw.split("---") if b.strip()]
    for block in blocks:
        lines = block.strip().splitlines()
        target_dim = None
        human_lines, zephyr_lines = [], []
        mode = None
        for line in lines:
            ls = line.strip()
            if ls.startswith("TARGET:"):
                target_dim = ls[7:].strip().lower()
            elif ls.startswith("[HUMAN]:"):
                mode = "human"
                human_lines.append(ls[8:].strip())
            elif ls.startswith("[ZEPHYR]:"):
                mode = "zephyr"
                zephyr_lines.append(ls[9:].strip())
            elif mode == "human" and ls:
                human_lines.append(ls)
            elif mode == "zephyr" and ls:
                zephyr_lines.append(ls)

        if human_lines and zephyr_lines:
            pairs.append({
                "human":      " ".join(human_lines).strip(),
                "zephyr":     " ".join(zephyr_lines).strip(),
                "target_dim": target_dim or "unknown",
            })
    return pairs


def save_training_pairs(pairs: list[dict], path: str = None):
    """Append synthesised pairs to training JSONL (never overwrites)."""
    out = path or TRAINING_PATH
    with open(out, "a", encoding="utf-8") as f:
        for p in pairs:
            record = {
                "conversations": [
                    {"from": "human", "value": p["human"]},
                    {"from": "gpt",   "value": p["zephyr"]},
                ],
                "source":     "blackwell_oracle",
                "target_dim": p.get("target_dim", "unknown"),
            }
            f.write(json.dumps(record) + "\n")
    print(f"Appended {len(pairs)} pairs → {out}")
