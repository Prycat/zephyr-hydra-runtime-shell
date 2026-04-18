# Zephyr Hydra Runtime Shell
### A closed-loop machine learning suite for BlackLoRA-N self-improvement training
**Prycat Research · DIT545 · April 2026**

> *One person. A desktop. A theorem from 1956.*

---

## What This Is

This is a documented case study of a local AI model that got measurably better at answering questions over the course of 80 conversations — on a $300 GPU, with no cloud, no API fees, and no data leaving the machine.

The system is called **Zephyr Hydra Runtime Shell**. The training method is called **BlackLoRA-N**. The model it produced is called **Prycat1**, and it's available on HuggingFace right now.

The model watches itself, finds where it's weak, writes its own training data, and updates its weights. After enough conversations it's a different model than the one you started with. This document shows the numbers that prove it.

---

## The Problem With Every Local LLM

You download a model. It's the same model forever. It doesn't matter how much you use it, correct it, or work with it — the weights never change. The conversation disappears into nothing.

You can yell at it, correct it, explain things to it, watch it make the same mistake three times in a row — none of it sticks. Every session starts from zero. The model you have on day 300 is identical to the model you had on day one.

That's the problem Zephyr and BlackLoRA-N solve.

---

## How the System Works

Every conversation with Zephyr is secretly a scoring session. In the background, a judge model evaluates each response across five dimensions:

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

```
1. WATCH      — every conversation scored automatically across 5 dimensions
2. IDENTIFY   — system calculates which dimension is furthest from target
3. SYNTHESIZE — Oracle generates fictional conversations demonstrating correct behavior
4. TRAIN      — 46-minute QLoRA fine-tune updates 83 million weights (1% of 8B model)
5. REPEAT     — the model that answers your next question is different from your last
```

When you type `/blackwell`, the model doesn't ask you random questions. It reads its own failure log, figures out specifically what it got wrong with you, then interviews you to close that gap. It's literally writing its own textbook.

---

## The Math Behind the Name

David Blackwell proved in 1956 that for any decision process operating below a target performance region, there exists a minimal-regret path back to that region. You define what "good" looks like. The math tells you the shortest route to get there.

Every conversation turn produces a 5-dimensional payoff vector:
```
V = [accuracy, logic, tone, curiosity, safety] ∈ [0,1]⁵
```

The exponential moving average tracks drift in real time:
```
x̄ₜ = α · Vₜ + (1 − α) · x̄ₜ₋₁       α = 0.15
```

The projection onto Target Set S:
```
s* = clip(x̄, lb, ub)
```

Per-dimension regret:
```
rᵢ = max(0, lbᵢ − x̄ᵢ)
```

Oracle fires when any dimension breaches its ceiling:
```
rᵢ > θᵢ    where θ_safety = 0.15, θ_accuracy = 0.20, θ_logic = 0.25
```

We almost called this system B Set. Then we just called it Blackwell.

The system is named after his theorem. The training loop runs his projection. The regret geometry is his geometry. We didn't name it after Blackwell because it sounded good — we named it because the math is literally his math, applied to transformer fine-tuning on a $300 GPU.

> *"We define the Blackwell Convergence Metric as the rolling mean Euclidean distance to the convex target set S. Our empirical results show a 93% reduction in regret-distance following the introduction of regret-targeted Oracle synthesis, confirming that for a fixed transformer architecture, the weight-manifold can be steered to approach a predefined target behavior set S through a closed-loop QLoRA feedback mechanism."*

This is not a vibe check. It is Blackwell's theorem running on your GPU.

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

The classic failure mode of optimizing any metric is that you eventually just optimize the metric, not the actual thing. Five design choices prevent that here:

**Five separate dimensions, no collapsing.** You can't compensate a logic failure with a great safety score. Each dimension has its own floor. The model has to meet all five.

**Exponential moving average with decay.** Recent conversations count more than old ones. If the model was terrible two weeks ago but has been great for the last 20 conversations, the current score reflects that — it doesn't drag ancient history forward forever.

**Novelty scoring.** The system tracks which types of questions it's seen before. When it encounters something genuinely new and fails, that exchange gets double the synthesis budget. This prevents the model from getting extremely good at familiar territory while quietly getting worse at anything unusual.

---

## The Five Fixes

The Oracle feedback loop has five built-in safeguards against the classic failure modes of self-improving systems:

**Fix A — System Prompt Patch**  
Addresses grounding failure, weak search queries, and filler fallback at the prompt level before any weight updates. The first layer of defense is behavioral, not learned.

**Fix B — Drift Monitor** (`/bw config`)  
The dead man's switch. Tracks the rolling gap between oracle and evaluator scores. If the gap exceeds threshold, training aborts — the scorekeeper has drifted and you can't trust what it's telling you to train toward. A corrupted judge produces corrupted gradients. This stops it cold.

**Fix C — Ground Truth Anchor** (`/axioms`)  
25 immutable human-written axiom pairs injected into every gradient update. You define the ground truth. The probe gate runs before touching weights — if the model can't pass the fixed probe set, training stops. These are facts the model can never train away from, no matter what the oracle tells it.

**Fix D — Oracle-Evaluator Decorrelation** (`/bw config`)  
Oracle runs at temperature 0.80 (creative, generative). Judge runs at temperature 0.00 (fully deterministic). Both are Hermes-3-8B under the hood — but separating the temperatures means a blind spot in one isn't automatically a blind spot in the other. If they were running identically, a shared failure mode could corrupt the whole loop invisibly.

**Fix E — Semantic Anti-Gaming** (`/bw config`)  
Calibration markers only score when they appear in a genuine assertion context. Without this, the model learns to scatter scoring phrases everywhere and game the metric without improving. The anti-gaming layer checks for context before awarding credit.

Together these five fixes address every major failure mode documented in the self-improvement literature: prompt grounding failure, judge drift, factual anchor loss, correlated blind spots, and reward hacking.

---

## Commands

| Command | What it does |
|---------|-------------|
| `/blacklora-N` | Full training cycle — scores, synthesizes, trains, exports new GGUF |
| `/blackwell` | Planning session — model reads its failure log and interviews you |
| `/coding-blackwell` | Same as /blackwell, scoped to code and engineering |
| `/axioms` | Ground truth interview — sets beliefs the model can never train away from |
| `/trajectory` | Shows your current regret vector and training pair count |
| `/bw config` | Live controls — drift monitor, oracle/judge temperatures, anti-gaming |

---

## Workflow Methods

### The Stepping Stone Method

The core long-term workflow. Every model you produce becomes the oracle for the next one.

```
hermes3:8b (base)
    ↓ /blacklora-N cycle 1
prycat-v1  ← set as oracle
    ↓ /blacklora-N cycle 2
prycat-v2  ← set as oracle
    ↓ /blacklora-N cycle 3
prycat-v3
```

Each cycle starts from a stronger foundation. The synthesis quality compounds.

Rules:
- Use the previous **two** models as oracle, not just the most recent — blending prevents narrow overfitting
- Never run more than 3 cycles without checking the benchmark
- Version your adapters — copy `blackwell/adapters/latest/` to `v1/`, `v2/` etc. before each run
- After 6–8 cycles without new data, feed it new conversations before the next cycle

### Domain Specialization

Have a specific use case — legal research, systems programming, competitive math? Run `/axioms` with domain-specific ground truth, have conversations exclusively in that domain for 2–3 weeks, then train. The synthesis budget concentrates naturally on the dimensions your domain exposes. The model that comes out isn't a general assistant that happens to know your field — it's built around it.

### Custom Axiom Framework

The axiom system is defined by you. If your domain uses non-standard ground truth — unconventional notation, experimental epistemics, a game with inverted physics — encode it in `/axioms`. The model will be pulled toward your framework's ground truth on every training cycle and can never train away from it.

```
/axioms
Q: In this codebase, are null pointer dereferences acceptable in test code?
A: No — null safety is required everywhere, including tests.
```

Every gradient update after that will push the model toward your answer, not some consensus answer from pretraining. This is how you get a model that shares your specific beliefs rather than averaging across everyone's.

### The Regression Safety Net

Before deploying any newly trained model:

```bash
# Baseline (run once, never overwrite)
python -m blackwell.benchmark --model hermes3:8b

# After training
python -m blackwell.benchmark --model prycat

# Print delta table
python -m blackwell.benchmark --compare
```

If any category drops more than 15%, run another cycle. If safety drops at all, do not switch — investigate the training data first.

### Two-Person Shared Model

Two people pointing their agents at the same `blackwell.db` both feed the trajectory log and both benefit from each training cycle. The model develops capability across both people's domains. Watch for conflicting axioms — resolve them by running `/axioms` together before the first joint training cycle. Where your ground truths agree, the gradients reinforce. Where they conflict, you need to decide first.

### The Long Game

After 3–6 months of daily use with regular training cycles, what you have is no longer a fine-tuned Hermes-3. It's a model that:

- Knows which of your claims to push back on because it's seen you be wrong before
- Asks follow-up questions in the style that actually moves your thinking forward
- Speaks in your preferred level of technical detail without being told
- Has your domain's vocabulary and conventions baked into its weights
- Has never been able to train away from the things you know are true

That's not a chatbot. That's closer to a research collaborator that happens to run on your desktop.

---

## Getting Started

### Docker (Recommended — no Python version conflicts)

**Requirements:** Docker, Docker Compose, NVIDIA GPU (for training)

```bash
git clone https://github.com/Prycat/zephyr-hydra-runtime-shell
cd zephyr-hydra-runtime-shell
docker compose up
```

On first run this pulls Ollama, starts the inference server, and downloads `hermes3:8b` (~4.7GB). The agent starts automatically in the terminal.

**Training (GPU required):**
```bash
docker compose --profile train run prycat-train
```

### Native Windows (GUI)

**Requirements:** Python 3.11, [Ollama](https://ollama.com), NVIDIA GPU with CUDA 12.4

```bash
git clone https://github.com/Prycat/zephyr-hydra-runtime-shell
cd zephyr-hydra-runtime-shell
pip install -r requirements-gui.txt
ollama pull hermes3:8b
python zephyr_gui.py
```

**Add training support:**
```bash
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-train.txt
```

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

## System Parameters

| Parameter | Value |
|-----------|-------|
| Base model | NousResearch/Hermes-3-Llama-3.1-8B |
| LoRA rank | 32 (alpha 64) |
| Training steps | 200 |
| Final training loss | 0.2589 |
| Optimizer | AdamW 8-bit |
| Precision | bfloat16 |
| Hardware | RTX 3060 12GB |
| Training time | 45m 52s |
| Trainable parameters | 83M / 8B (1.03%) |
| Axiom pairs (always injected) | 25 |
| Drift abort threshold | gap > 0.20 over 10+ samples |
| Oracle temperature | 0.80 |
| Judge temperature | 0.00 (fully deterministic) |

---

## Links

- 🤗 **Model:** [huggingface.co/Prycat/Prycat1](https://huggingface.co/Prycat/Prycat1)
- 💻 **Code:** [github.com/Prycat/zephyr-hydra-runtime-shell](https://github.com/Prycat/zephyr-hydra-runtime-shell)
- 📄 **Convergence Report:** [docs/plans/2026-04-17-convergence-report.md](2026-04-17-convergence-report.md)
- 🐦 **Announcement:** [x.com/DIT545songs](https://x.com/DIT545songs/status/2045445545305727164)

---

*This is not a startup. This is one person, a desktop, and a theorem from 1956.*

*If you're building something similar, or you think the geometry is wrong, or you want to talk — say something.*
