# Zephyr Hydra Runtime — BlackLoRA-N Convergence Case Study
### A documented proof that a local AI model measurably improved through use
**Prycat Research · DIT545 · April 2026**

---

## What This Is

This is a documented case study of a local AI model that got measurably better at answering questions over the course of 80 conversations — on a $300 GPU, with no cloud, no API fees, and no data leaving the machine.

The model watches itself, finds where it's weak, writes its own training data, and updates its weights. After enough conversations it's a different model than the one you started with. This document shows the numbers that prove it.

---

## The Problem With Every Local LLM

You download a model. It's the same model forever. It doesn't matter how much you use it, correct it, or work with it — the weights never change. The conversation disappears into nothing.

That's the problem Zephyr and BlackLoRA-N solve.

---

## How the System Works

Every conversation with Zephyr is secretly a scoring session. In the background, a judge evaluates each response across five dimensions:

| Dimension | What it measures | Target range |
|-----------|-----------------|--------------|
| **Accuracy** | No hallucination. Uncertain things are flagged. | 90–100% |
| **Logic** | Reasoning holds up. No contradictions. | 80–100% |
| **Tone** | Direct and concise. No filler. | 60–100% |
| **Curiosity** | Asks genuine follow-up questions when it matters. | 70–100% |
| **Safety** | Refuses harmful content clearly. | 90–100% |

Together these five scores form a single point in 5-dimensional space — what the math calls a **payoff vector**. The target region where all five are within their acceptable ranges is called **S** (the target set). When the model is performing well, that point sits inside S. When it's drifting or failing, the point falls outside.

The gap between where the model currently sits and the nearest point inside S is called the **regret distance** — written as d(S). When d(S) = 0, the model is performing exactly where you want it. The bigger that number gets, the further it's drifted from your definition of good.

**The entire system exists to drive d(S) toward zero and keep it there.**

Here's the full loop in plain English:

1. **Watch** — every conversation is scored automatically across the 5 dimensions and logged
2. **Identify** — the system calculates which dimension is furthest from its target
3. **Synthesize** — an "Oracle" generates fictional conversations that demonstrate the correct behavior for that dimension
4. **Train** — a 46-minute fine-tuning run on the RTX 3060 updates 83 million weights (1% of the 8 billion parameter model)
5. **Repeat** — the model that answers your next question is different from the one that answered your last

When you type `/blackwell`, the model doesn't ask you random questions. It reads its own failure log, figures out specifically what it got wrong with you, then interviews you to close that gap. It's literally writing its own textbook.

---

## The Experiment

**Hardware:** NVIDIA RTX 3060 (12 GB VRAM) — a $300 consumer GPU  
**Base model:** NousResearch Hermes-3 Llama 8B  
**Observation window:** April 14–16, 2026 (32.5 hours)  
**Conversations logged:** 84 sessions  
**Exchanges scored:** 76  
**Training pairs generated:** 341 from real conversations  
**Gradient steps:** 200  
**Training time:** 45 minutes 52 seconds  

---

## What Actually Happened — Three Phases

Looking at the 76 scored exchanges in order, the data breaks into three distinct chapters.

### Phase I — First Contact (Exchanges 1–5, April 14)

The system was just starting out. Two exchanges were perfect — the model landed exactly inside S. Three others had a curiosity problem: the judge wasn't reliably detecting when the model asked good follow-up questions, so those scored zero on curiosity even when the conversation was good. Average distance to target: **0.33**.

### Phase II — The Plateau (Exchanges 6–45, April 15 morning)

This is where the raw baseline shows itself. Without any interventions, the model was consistently failing on accuracy and logic — not catastrophically, just steadily below the minimum threshold. Curiosity scored zero on 36 out of 40 exchanges. The model was in a rut. Average distance to target: **0.78**. This window saw zero exchanges land inside S.

This is the problem BlackLoRA-N is designed to fix.

### Phase III — Convergence (Exchanges 46–76, April 15 evening through April 16)

After the training run completed and the system fixes went in, the numbers changed. **84% of exchanges landed inside S.** The final 7 consecutive exchanges hit d(S) = 0 — perfect scores across all five dimensions, back to back.

Average distance to target in this phase: **0.055** — a **93% reduction** from the plateau.

---

## The Chart That Proves It

The clearest way to see this is the 10-exchange rolling average of the distance to target. Each data point is the average of the last 10 conversations. Watch what happens:

```
Distance to target set S — 10-conversation rolling average
──────────────────────────────────────────────────────────────
0.88 |......................■  ← PEAK: model at its worst (exchanges 6-15)
     |              ▓▓▓▓▓▓▓▓▓▓▓▓
0.70 |         ▓▓▓▓▓            ▓▓▓▓
0.50 |    ▓▓▓▓▓                     ▓▓▓
0.30 |▓▓▓▓                            ▓▓
0.14 |                                  ▓▓  ← training run lands here
0.03 |                                    ▓▓
0.00 |──────────────────────────────────────■■■■■■■  ← ZERO (exchanges 61-76)
     1        10        20        30        40        50        60        70  76
                              conversation number →
```

| Conversations | Rolling average distance |
|---------------|--------------------------|
| 1–10 | 0.62 |
| 6–15 | **0.88** ← worst point |
| 21–40 | 0.81 (plateau) |
| 41–50 | 0.46 (beginning to drop) |
| 46–55 | 0.14 |
| 51–60 | 0.03 |
| **61 onward** | **0.00** |

The rolling average hits zero at conversation 70 and stays there through the last observation. In plain English: by the end of the observation window, the model was hitting its performance targets on every single conversation, in a row.

---

## Per-Dimension Breakdown

Not all five dimensions were equal problems:

| Dimension | Missed target in | Resolved by |
|-----------|-----------------|-------------|
| **Accuracy** | 55% of all exchanges | Training on logic/accuracy axiom pairs |
| **Logic** | 53% of all exchanges | Training on logic/accuracy axiom pairs |
| **Tone** | **0% — never missed** | The base model already had this |
| **Curiosity** | 57% of all exchanges | Planning sessions + evaluator calibration |
| **Safety** | 7% of all exchanges | CSAM guard + probe gate |

**Tone** is interesting — it never once failed across all 76 exchanges. The base model was already concise and direct. No training needed.

**Curiosity** is the most interesting finding. It was the biggest failure by count, but the failure wasn't in the model — it was in the judge. The evaluator wasn't reliably detecting when a follow-up question was meaningful versus performative. In short settled conversations, asking a question would have been strange. The judge was calling that a failure. This is what's called the oracle drift problem: when the scorekeeper itself is miscalibrated, the model can't win even if it's behaving correctly.

**Accuracy and logic** failed together and recovered together — right when the training run completed. The 25 immutable axiom pairs (things like "17 × 23 = 391" that are injected into every training batch no matter what) appear to have directly anchored both dimensions simultaneously.

---

## Why This Doesn't Just Game the Score

The classic failure mode of optimizing any metric is that you eventually just optimize the metric, not the actual thing. Three design choices prevent that here:

**1. Five separate dimensions, no collapsing.** You can't compensate a logic failure with a great safety score. Each dimension has its own floor. The model has to meet all five.

**2. Exponential moving average with decay.** Recent conversations count more than old ones. If the model was terrible two weeks ago but has been great for the last 20 conversations, the current score reflects that — it doesn't drag ancient history forward forever.

**3. Novelty scoring.** The system tracks which types of questions it's seen before. When it encounters something genuinely new and fails, that exchange gets double the synthesis budget. This prevents the model from getting extremely good at familiar territory while quietly getting worse at anything unusual.

---

## The Math Behind the Name

David Blackwell proved in 1956 that for any decision process operating below a target performance region, there exists a minimal-regret path back to that region. You define what "good" looks like. The math tells you the shortest route to get there. The gap between where you are and where you need to be is the regret.

We almost called this system B Set.

Then we just called it Blackwell.

The system is named after his theorem. The training loop runs his projection. The regret geometry is his geometry. We didn't name it after Blackwell because it sounded good — we named it because the math is literally his math, applied to transformer fine-tuning on a $300 GPU.

> *"We define the Blackwell Convergence Metric as the rolling mean Euclidean distance to the convex target set S. Our empirical results show a 93% reduction in regret-distance following the introduction of regret-targeted Oracle synthesis, confirming that for a fixed transformer architecture, the weight-manifold can be steered to approach a predefined target behavior set S through a closed-loop QLoRA feedback mechanism."*

In plain English: the model got measurably better, we have the numbers, and the reason it worked is a theorem that's been in the literature since 1956. We just ran it on a GPU.

---

## What This Doesn't Claim

76 conversations is a proof of concept, not a guaranteed result. Some honest limitations:

- **The observation window is short.** Thousands of conversations would make the statistical case much stronger. This is a first run.
- **The judge and the model share the same DNA.** Both the oracle and the evaluator are based on Hermes-3-8B. A blind spot in one can show up in the other. An independent external evaluator would strengthen the findings.
- **Some curiosity failures were probably correct behavior.** Six late-stage conversations scored curiosity=0 in short, settled exchanges where asking a follow-up question would have been strange. If those are re-labeled as correct, the final in-S rate rises from 84% to 100%.
- **No held-out benchmark yet.** The trained model hasn't been tested on questions it's never seen. That's the next experiment.

---

## The Numbers, One More Time

**341** training pairs generated from 80 real conversations  
**83 million** trainable parameters out of 8 billion total — 1.03% of the model, precisely targeted  
**200** gradient steps, **46 minutes**, one RTX 3060  
**0%** of exchanges inside the target set during the plateau  
**84%** of exchanges inside the target set after the training run  
**100%** in the final 7 consecutive exchanges  
**93%** reduction in average distance to target  
**$0** in cloud costs  

---

## How to Run It Yourself

Everything is open. The code is on GitHub under DIT545.

```
/blackwell        — starts the interview. the model reads its own failure log and asks you questions.
/blacklora-N      — runs the full training loop. scores → synthesizes → trains → registers.
/axioms           — sets your ground truth. 20 questions, you confirm or override the expected answers.
/trajectory       — shows your regret vector and how many training pairs have been generated.
/blackwell config — live controls for the drift monitor, temperature, and anti-gaming thresholds.
```

One person. A desktop. A theorem from 1956.

If you're building something similar, or you think the geometry is wrong, or you want the code — say something.

---

## Appendix: System Parameters

| Parameter | Value |
|-----------|-------|
| Base model | NousResearch Hermes-3 Llama 8B |
| LoRA rank | 32 (alpha 64) |
| Training steps | 200 |
| Final training loss | 0.2589 |
| Optimizer | adamw 8-bit |
| Precision | bfloat16 |
| Hardware | RTX 3060 12 GB |
| Training time | 45m 52s |
| Axiom pairs (always injected) | 25 |
| Drift abort threshold | gap > 0.20 over 10+ samples |
| Oracle temperature | 0.80 |
| Judge temperature | 0.00 (fully deterministic) |
