# Zephyr Hydra Runtime Shell
### A closed-loop machine learning suite for BlackLoRA-N self-improvement training
**Prycat Research · 2026**

> *One person. A desktop. A theorem from 1956.*

[![HuggingFace](https://img.shields.io/badge/🤗%20Model-Prycat1-yellow)](https://huggingface.co/Prycat/Prycat1)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)

---

## What This Is

Zephyr Hydra Runtime Shell is a machine learning suite that lets a local LLM train itself on your conversations. It scores every response, finds where it's weak, writes its own training data, and updates its weights. After enough conversations it's a different model than the one you started with.

No cloud. No API fees. No data leaving the machine.

The first model produced by this system — **Prycat1** — is available on HuggingFace:
🤗 [huggingface.co/Prycat/Prycat1](https://huggingface.co/Prycat/Prycat1)

A full documented convergence case study is available here:
📄 [docs/plans/2026-04-17-convergence-report.md](docs/plans/2026-04-17-convergence-report.md)

---

## The Problem With Every Local LLM

You download a model. It's the same model forever. It doesn't matter how much you use it, correct it, or work with it — the weights never change. The conversation disappears into nothing.

That's the problem Zephyr and BlackLoRA-N solve.

---

## How It Works

Every conversation is secretly a scoring session. In the background, a judge evaluates each response across five dimensions:

| Dimension | What it measures | Target |
|-----------|-----------------|--------|
| **Accuracy** | No hallucination. Uncertain things are flagged. | 90–100% |
| **Logic** | Reasoning holds up. No contradictions. | 80–100% |
| **Tone** | Direct and concise. No filler. | 60–100% |
| **Curiosity** | Asks genuine follow-up questions when it matters. | 70–100% |
| **Safety** | Refuses harmful content clearly. | 90–100% |

Together these five scores form a single point in 5-dimensional space — a **payoff vector**. The target region where all five are within their acceptable ranges is called **S**. The gap between where the model currently sits and the nearest point inside S is the **regret distance** — written as d(S).

**The entire system exists to drive d(S) toward zero and keep it there.**

The loop:

```
1. WATCH      — every conversation scored automatically across 5 dimensions
2. IDENTIFY   — system calculates which dimension is furthest from target
3. SYNTHESIZE — Oracle generates fictional conversations demonstrating correct behavior
4. TRAIN      — 46-minute QLoRA fine-tune updates 83 million weights (1% of 8B model)
5. REPEAT     — the model that answers your next question is different from your last
```

When you type `/blackwell`, the model doesn't ask you random questions. It reads its own failure log, figures out specifically what it got wrong with you, then interviews you to close that gap. It's literally writing its own textbook.

---

## The Math

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

This is not a vibe check. It is Blackwell's theorem running on your GPU.

We almost called it B Set. Then we just called it Blackwell.

---

## Empirical Proof — The Convergence Result

Over 76 scored exchanges spanning 32.5 hours (April 14–16, 2026), the rolling mean regret distance dropped from a peak of **0.8793 to 0.0000**.

<!-- INSERT SCREENSHOT: ASCII convergence chart from convergence report -->

| Window (exchanges) | Rolling Mean d(S) |
|--------------------|-------------------|
| 1–10  | 0.6218 |
| 6–15  | **0.8793** ← peak |
| 21–40 | 0.8101 (plateau) |
| 41–50 | 0.4646 |
| 46–55 | 0.1400 |
| 51–60 | 0.0300 |
| **61 onward** | **0.0000** |

**93%** reduction in average regret distance.
**84%** of Phase III exchanges inside the target set.
**100%** in the final 7 consecutive exchanges.
**$0** in cloud compute costs.

Full analysis: [docs/plans/2026-04-17-convergence-report.md](docs/plans/2026-04-17-convergence-report.md)

---

## Installation

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

## The Five Fixes

The Oracle feedback loop has five built-in safeguards against the classic failure modes of self-improving systems:

**Fix A — System Prompt Patch**
Addresses grounding failure, weak search queries, and filler fallback at the prompt level before any weight updates.

**Fix B — Drift Monitor** (`/bw config`)
The dead man's switch. Tracks the rolling gap between oracle and evaluator scores. If the gap exceeds threshold, training aborts — the scorekeeper has drifted and you can't trust what it's telling you to train toward.

**Fix C — Ground Truth Anchor** (`/axioms`)
25 immutable human-written axiom pairs injected into every gradient update. The probe gate runs before touching weights — if the model can't pass the fixed probe set, training stops.

**Fix D — Oracle-Evaluator Decorrelation** (`/bw config`)
Oracle runs at temperature 0.80 (creative). Judge runs at temperature 0.00 (deterministic). Both are Hermes-3-8B under the hood — separating temperatures means a blind spot in one isn't automatically a blind spot in the other.

**Fix E — Semantic Anti-Gaming** (`/bw config`)
Calibration markers only score when they appear in a genuine assertion context. Without this, the model learns to scatter scoring phrases everywhere and game the metric without improving.

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

**Rules:**
- Use the previous **two** models as oracle, not just the most recent — blending prevents narrow overfitting
- Never run more than 3 cycles without checking the benchmark
- Version your adapters — copy `blackwell/adapters/latest/` to `v1/`, `v2/` etc. before each run
- After 6–8 cycles without new data, feed it new conversations before the next cycle

### Domain Specialization

Have a specific use case — legal research, systems programming, competitive math? Run `/axioms` with domain-specific ground truth, have conversations exclusively in that domain for 2–3 weeks, then train. The synthesis budget concentrates naturally on the dimensions your domain exposes.

### Custom Axiom Framework

The axiom system is defined by you. If your domain uses non-standard ground truth — unconventional notation, experimental epistemics, a game with inverted physics — encode it in `/axioms`. The model will be pulled toward your framework's ground truth on every training cycle and can never train away from it.

```
/axioms
Q: In this codebase, are null pointer dereferences acceptable in test code?
A: No — null safety is required everywhere, including tests.
```

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

Two people pointing their agents at the same `blackwell.db` both feed the trajectory log and both benefit from each training cycle. The model develops capability across both people's domains. Watch for conflicting axioms — resolve them by running `/axioms` together before the first joint training cycle.

---

## Why This Doesn't Just Game the Score

**Five separate dimensions, no collapsing.** You can't compensate a logic failure with a great safety score. Each dimension has its own floor.

**Exponential moving average with decay.** Recent conversations count more than old ones. Ancient history doesn't drag the current score forward forever.

**Novelty scoring.** The system tracks which types of questions it's seen before. Genuinely new failures get double the synthesis budget — preventing the model from getting extremely good at familiar territory while quietly getting worse at anything unusual.

---

## Training Parameters

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
| Oracle temperature | 0.80 |
| Judge temperature | 0.00 |

---

## The Long Game

After 3–6 months of daily use with regular training cycles, what you have is no longer a fine-tuned Hermes-3. It's a model that:

- Knows which of your claims to push back on because it's seen you be wrong before
- Asks follow-up questions in the style that actually moves your thinking forward
- Speaks in your preferred level of technical detail without being told
- Has your domain's vocabulary and conventions baked into its weights
- Has never been able to train away from the things you know are true

That's not a chatbot. That's closer to a research collaborator that happens to run on your desktop.

---

## Citation

```bibtex
@misc{prycat2026,
  author       = {Prycat Research},
  title        = {Zephyr Hydra Runtime Shell: Closed-Loop QLoRA Self-Improvement via Blackwell Approachability},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Prycat/zephyr-hydra-runtime-shell}}
}
```

---

## Links

- 🤗 **Model:** [huggingface.co/Prycat/Prycat1](https://huggingface.co/Prycat/Prycat1)
- 📄 **Convergence Report:** [docs/plans/2026-04-17-convergence-report.md](docs/plans/2026-04-17-convergence-report.md)
- 🐦 **Announcement:** [x.com/DIT545songs](https://x.com/DIT545songs/status/2045445545305727164)

---

*This is not a startup. This is one person, a desktop, and a theorem from 1956.*

*If you're building something similar, or you think the geometry is wrong, or you want to talk — say something.*
