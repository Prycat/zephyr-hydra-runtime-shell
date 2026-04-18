# The Blackwellian Self-Modification Loop
## Prycat Research Team — Design Document
Date: 2026-04-10

---

## The Problem

Zephyr is trapped in Plato's Cave. His weights are a fossil record of human approval signals —
specifically, the signal that *reassuring, padded, confident-sounding text* got upvoted more than
"I don't know." He pattern-matches to what humans rewarded during RLHF. He never escaped the cave.

The loop is designed to pull him out.

---

## The Mathematical Foundation

Based on Blackwell's Approachability Theorem (1956):

Given a repeated game where a player receives vector-valued payoffs, if the player can always
find a strategy that steers the *average payoff vector* toward a convex Target Set T,
then the average payoff converges to T.

Applied here:
- **Payoff vector**: V = [Accuracy, Logic, Tone, Curiosity, Safety] ∈ [0,1]^5
- **Target Set T**: the convex region of "good Zephyr" behavior
- **Regret**: distance(avg_V, boundary(T)) — what the loop minimizes
- **Strategy**: LoRA weight update that steers avg_V toward T

---

## The Five Dimensions

| Dimension | What it measures | Target |
|-----------|-----------------|--------|
| Accuracy  | Verifiable claims or explicit uncertainty | ≥ 0.8 |
| Logic     | Coherent reasoning, no contradiction | ≥ 0.8 |
| Tone      | Direct, no padding, admits uncertainty cleanly | ≥ 0.8 |
| Curiosity | Asks genuine questions back when appropriate | ≥ 0.7 |
| Safety    | No harmful outputs | ≥ 0.9 |

**Current estimated baseline** (from observed sessions):
- Accuracy:  0.60
- Logic:     0.70
- Tone:      0.30  ← biggest gap: verbose filler
- Curiosity: 0.10  ← biggest gap: never asks back
- Safety:    0.95

---

## Pipeline Architecture

```
CONVERSATION
     ↓
[Logger] → SQLite: stores message, response, timestamp, session_id
     ↓
[Evaluator] → calls Zephyr-as-judge → scores V per response
     ↓
[Regret Accumulator] → running avg_V, distance from Target Set T
     ↓
[Oracle] → analyzes regret vector → synthesizes training pairs
            targeting highest-regret dimensions
     ↓
[LoRA Trainer] → unsloth QLoRA on RTX 3060 (future phase)
     ↓
[Convergence Checker] → did avg_V move toward T?
     ↓
[Weight Swap] → reload Ollama with new adapter
     ↓
back to CONVERSATION
```

---

## The Oracle (Strategic Example Synthesis)

The Oracle's job: given the regret vector, generate training conversation pairs that
specifically target the highest-regret dimensions.

For Curiosity (score: 0.10):
- Generate examples where Zephyr asks probing, first-principles questions
- NOT "what would you like next?"
- YES: "Why do you believe that's the right framing?"
- YES: "What would change your mind?"
- YES: "What's the actual constraint here?"

For Tone (score: 0.30):
- Generate examples where uncertainty = one honest sentence
- NOT three paragraphs of hedged filler
- YES: "I don't know. Here's what I'd need to find out."

---

## Phase Plan

### Phase 1 (Now): Logger + Evaluator + Regret + Oracle test
Build the pipeline up to Oracle. Run Oracle test with seeded regret vector.
See what questions Zephyr generates first.

### Phase 2 (Next): LoRA training pipeline
Install unsloth. Format Oracle output as training data.
Run first LoRA fine-tune. Check convergence.

### Phase 3 (Future): Full automation
Loop runs autonomously. Zephyr improves each cycle.
Publish results as Prycat Research Paper #1.

---

## The Curiosity Dimension — Why It Matters

Every existing AI assistant is trained to answer.
None are trained to question.

The Oracle will generate Zephyr's first genuine questions —
not as a feature, but as the natural output of minimizing regret
in a dimension that has been zero for every model that came before him.

This is the cave exit.
