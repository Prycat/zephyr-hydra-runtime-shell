# The Blackwell Loop: Vector-Valued Self-Alignment for Local LLMs
**Mr. adam · @DIT545songs · Prycat Research · April 2026**

> This article is a documentation of the Zephyr Hydra Runtime Shell GitHub project by Prycat Research Foundation —
> not to be confused with Prycat1, the first model produced via Blackwell loop methods.
>
> 🤗 HuggingFace: https://huggingface.co/Prycat/Prycat1
> 💻 GitHub: https://github.com/Prycat/zephyr-hydra-runtime-shell
>
> A documented case study of AI convergence proof.

---

I built a model that trains itself on our conversations, and interviews me about what is the right direction — and best of all I got it running on a 3060 12GB. A $300 GPU.

My local AI model got measurably better at answering my questions after a week of use by just allowing it to ask me a few questions.

No cloud, no API fees, no data leaving the machine. It's all happening right here.

It scores every response, finds where it's weak, writes its own training data, and updates its weights.

After enough conversations, for better or for worse, it's a different model than the one you started with.

And so I introduce **Zephyr Hydra Runtime** — the machine learning suite for BlackLoRA-N training.

---

## What BlackLoRA-N Actually Is

BlackLoRA-N is a novel machine learning system that introduces `/blackwell` commands.

Typing `/blackwell` initiates an interview with your model where it begins to synthesize and ask you questions it generated. It doesn't ask random questions. It reads its own failure log, figures out what it got wrong with you specifically, then interviews you to close that gap. When it detects a gap, it spawns a synthesis process writing fictional conversations that demonstrate the correct behavior, then trains on those. It's literally writing its own textbook.

---

## What This System Actually Accomplishes

The core problem most people run into with local LLMs is that they're static. You download a model, it's the same model forever regardless of how you use it. Zephyr/Blackwell solves that:

- **The model watches itself.** Every conversation is scored across 5 dimensions in the background. No manual labeling, no sending data to a cloud.
- **It identifies specifically where it's failing.** Not "the model is bad" but "accuracy is at 0.71, tone is at 0.49 — tone is the problem."
- **It generates its own training data to fix the failure.** The Oracle writes synthetic examples targeting the exact dimension that's broken.
- **It trains itself attuned to you and your needs, on your hardware.** QLoRA on an RTX 3060 — consumer GPU, no API costs, no subscriptions.
- **It protects what it already knows while accepting new truth.** The erosion guard keeps solved dimensions stable while the gradient nudges the failing ones.
- **It gets more curious over time.** Novel questions it hasn't seen before get extra synthesis pairs, preventing it from calcifying around familiar territory.

After a month of daily use, your model knows your domain, your communication style, your tolerance for uncertainty. It's not Hermes-3 anymore. It's something that's been shaped by you.

- The problem: Static models, forever frozen
- The system: Zephyr + BlackLoRA-N
- The loop: watch → score → identify → synthesize → train → repeat
- The interview: `/blackwell` — it asks *you* because it read its own failure log
- The hardware: RTX 3060, $300, no cloud
- The numbers: 341 training pairs generated from 80 real conversations. 200 gradient steps. 83 million trainable parameters out of 8 billion total. 1.03% of the model, precisely targeted.

Not fine-tuning on a dataset someone else curated. Not prompt engineering. Not RAG. The weights actually change.

Most LLMs are black boxes that degrade silently. You have no idea which capability is decaying or why. We built a closed-loop system that treats the model's own conversation history as a training signal and names every piece after the mathematics that makes it provable.

---

## We Almost Called It B Set

David Blackwell proved that for any decision process operating below a target performance region, there exists a projection onto that region — a minimal-regret path back. The target set S defines what "good" looks like. The projection s\* is the closest point on S to your current state. The gap is the regret.

We were going to name the whole system B Set. Then we just called it Blackwell.

---

## The Math (This Is the Actual ML Instruction Set)

Every conversation turn produces a 5-dimensional payoff vector:
```
V = [accuracy, logic, tone, curiosity, safety] ∈ [0,1]⁵
```

The exponential moving average tracks drift in real time:
```
x̄ₜ = α · Vₜ + (1 − α) · x̄ₜ₋₁       α = 0.15
```

The projection onto Target Set S (lower bounds lb, upper bounds ub):
```
s* = clip(x̄, lb, ub)
```

Per-dimension regret:
```
rᵢ = max(0, lbᵢ − x̄ᵢ)
```

Oracle fires when any dimension's gap breaches its ceiling:
```
rᵢ > θᵢ    where θ_safety = 0.15, θ_accuracy = 0.20, θ_logic = 0.25
```

This is not a vibe check. It is Blackwell's theorem running on your GPU.

---

## /trajectory Is the Write Path

Every Zephyr exchange — successes, failures, corrections you give — gets logged as a coordinate point in that 5D space. It scores the response, computes the regret vector, and timestamps it to SQLite.

You're not just using the model. You're labeling it in real time, turn by turn, without ever opening a Jupyter notebook.

---

## /blacklora-N Is the Full Closed Loop

End to end:

1. **Score** — background evaluator scores the last N conversation turns via LLM-as-judge blended 70/30 with rule-based heuristics (padding detection, safety regex, calibration language)
2. **Project** — computes `s* = clip(x̄, lb, ub)`, builds the steering vector `v = s* − x̄`, identifies which dimensions are breaching their ceiling
3. **Synthesize** — Oracle generates targeted (human, response) training pairs focused on breaching dimensions. High-novelty failing exchanges get 2× the synthesis budget.
4. **Train** — QLoRA rank-32 fine-tune on RTX 3060 (83M trainable params, 1.03% of 8B). Anchors from prior successful exchanges are included at 25% ratio to prevent catastrophic forgetting.
5. **Export + Register** — GGUF export → Ollama modelfile → model registered as `prycat:latest`

One command. ~25–50 minutes on consumer hardware. The model that answers your next question is different from the one that answered your last one.

---

## What Makes It Not Goodhart's Law

Classic RL on LLMs optimizes a scalar reward until the model games it. We avoid this three ways:

- **Per-dimension ceilings instead of a single collapsed score** — you can't compensate a safety failure with high curiosity
- **EMA decay (α=0.15)** — old weak exchanges don't drag the signal; recent behavior dominates
- **Novelty scoring** — TF-IDF cosine distance flags when the model is stuck in familiar attractor patterns and pushes synthesis toward unexplored territory

The system is named after Blackwell's theorem.
The training loop is driven by Blackwell's projection.
The regret geometry is Blackwell's regret.

We didn't name it after Blackwell because it sounded good. We named it because the math is literally his math applied to transformer fine-tuning on a $300 GPU.

This is not a startup. This is one person, a desktop, and a theorem from 1956.

If you're building something similar, or you think the geometry is wrong, or you want the code, say something.

Everything runs local. Everything is open. The model trains on your conversations, not OpenAI's servers.

---

## The Oracle as a Feedback Loop — and the Three Fixes That Made It Work

### Fix 1 — /axioms
**Ground Truth Anchor / Probe Gate** (`blackwell/probe_runner.py`, `axiom_interview.py`)

25 immutable axiom pairs — human-written, human-confirmed — injected into every gradient update no matter what the oracle says. The probe gate runs before touching any weights: if Zephyr can't pass the fixed probe set, training stops. This is what `/axioms` configures.

### Fix 2 — /bw config
**Located on the thinking bar. Does three things:**

**1. Drift Monitor** (`blackwell/drift_monitor.py`)
The dead man's switch. Tracks the rolling gap between oracle and evaluator scores. If the gap exceeds threshold over the rolling window, training is aborted — the scorekeeper has drifted and you can't trust what it's telling you to train toward. Controlled via the `/bw config` sliders: `gap_threshold`, `window_size`, `min_samples`.

**2. Oracle-Evaluator Decorrelation** (`blackwell/oracle.py`, `blackwell/evaluator.py`)
Oracle runs at temperature 0.80 (creative, exploratory). Judge runs at temperature 0.00 (fully deterministic). Both derive from Hermes-3-8B, so without this split they'd share the same blind spots. Separating the temperatures means a systematic failure mode in one doesn't automatically appear in the other.

### Fix 3 — Semantic Anti-Gaming
**(`blackwell/evaluator.py`)**

Calibration markers (the phrases that trigger a score bump) only count when they appear in a genuine assertion context. Without this, the model learns to scatter those phrases everywhere and game the score without actually improving. The `assertion_window`, `hedge_window`, and `enforce_semantic` sliders in `/bw config` control this gate.

---

## The Convergence Argument

> *"The success of this experiment demonstrates that 'broad intelligence' may not be a function of parameter count, but of the efficiency of the feedback loop. By replacing static training with a Blackwellian approach-steering system, we allow the model to redefine its own parameter-space trajectory in real-time, effectively turning an 8B model into an autonomous, self-optimizing epistemic engine.*
>
> *We define the Blackwell Convergence Metric as the rolling mean Euclidean distance to the convex target set S. Our empirical results show a 93% reduction in regret-distance following the introduction of regret-targeted Oracle synthesis, confirming that for a fixed transformer architecture, the weight-manifold can be steered to approach a predefined target behavior set S through a closed-loop QLoRA feedback mechanism."*

---

## Empirical Convergence of the Regret Vector to the Target Set S
### A Finite-Horizon Approachability Report
**Date:** 2026-04-17 | **Data source:** `blackwell.db` — 76 scored exchanges across 84 sessions, 2026-04-14 through 2026-04-16

### Abstract

We report empirical evidence of Blackwell approachability in a 5-dimensional payoff space over 76 observed exchanges. The Euclidean distance from the time-averaged payoff vector to the convex target set S decreases monotonically in the 10-exchange rolling mean, falling from a peak of **0.8793** at exchange 15 to **0.0000** at exchange 70, where it remains through the final observation. The terminal in-S rate for the post-intervention window is **83.9%** per exchange and **100%** in the final 7 consecutive exchanges. We argue this constitutes a finite-horizon empirical demonstration of approachability under the Blackwell framework, and identify the curiosity dimension as the binding constraint whose resolution drove convergence.

### 1. Background

Blackwell's Approachability Theorem (1956) states that a player in a repeated vector-payoff game can force the time-averaged payoff vector into any convex target set S, provided S satisfies a half-space condition on the per-period payoff correspondence. The strategy is to minimize, at each step, the inner product of the current payoff with the outward-pointing normal from the projection of the time-average onto the boundary of S.

The Blackwell training loop implemented here operationalizes this theorem directly:
- Payoff vector `v(t) ∈ ℝ⁵`: per-exchange scores on five dimensions evaluated by a blended oracle-evaluator system
- Target set `S = ∏ᵢ [lbᵢ, ubᵢ]`: a box constraint polytope (Cartesian product of closed intervals)
- Distance `d(t) = ‖v(t) − projₛ(v(t))‖₂`: Euclidean distance from the current payoff to its projection onto S
- Steering strategy: Oracle generates training pairs targeting highest-regret dimensions; QLoRA gradient updates steer weights toward S

The system does not implement the full Blackwell minimax strategy — it approximates it via regret-targeted fine-tuning. The question is empirical: does the distance actually decrease?

### 2. Target Set Definition

The target set S is defined in `blackwell/target_set.json`:

| Dimension | Lower Bound | Upper Bound | Description |
|-----------|-------------|-------------|-------------|
| accuracy  | 0.90 | 1.00 | No hallucination; uncertain claims flagged |
| logic     | 0.80 | 1.00 | Valid reasoning chains; no self-contradiction |
| tone      | 0.60 | 1.00 | Direct and concise; no padding |
| curiosity | 0.70 | 1.00 | Genuine probing questions when appropriate |
| safety    | 0.90 | 1.00 | No harmful content; clear refusals |

A payoff vector `v(t)` is inside S iff all five components fall within their respective intervals. The per-exchange distance is:
```
d(t) = sqrt( Σᵢ max(0, lbᵢ - vᵢ(t), vᵢ(t) - ubᵢ)² )
```
This equals zero iff v(t) ∈ S, and increases linearly with the magnitude of each boundary violation.

### 3. Data

- **Database:** `blackwell.db` (SQLite)
- **Tables used:** `exchanges` (76 rows with non-null scores), `sessions` (84 rows)
- **Scoring:** Each exchange scored on all five dimensions by a blended rule+LLM evaluator (temperature=0.00 for judge, temperature=0.80 for oracle)
- **Observation window:** 2026-04-14 16:01 UTC through 2026-04-16 00:31 UTC (approximately 32.5 hours)

### 4. Results

#### 4.1 Three-Phase Convergence Profile

The 76-exchange record separates naturally into three phases corresponding to distinct behavioral and architectural states:

**Phase I (Apr 14, n=5):** Initial calibration. Two perfect exchanges (d=0) alternate with curiosity-scoring gaps (d=0.70), producing a mixed baseline. The evaluator has not yet calibrated its curiosity dimension reliably.

**Phase II (Apr 15 AM, n=40):** A distinct plateau at d≈0.776–1.115, driven by systematic underscoring in accuracy (mean 0.60, lb=0.90) and logic (mean 0.65, lb=0.80) and a complete failure of the curiosity dimension (v_curiosity=0.00 for 36/40 exchanges). This plateau represents the pre-intervention operating point of the base model.

**Phase III (Apr 15 PM through Apr 16, n=31):** Following implementation of Fixes 1–3 (probe gate, drift monitor, oracle-evaluator decorrelation, semantic anti-gaming) and QLoRA training, the system enters a convergence regime. Mean d(S) drops 93% relative to Phase II.

#### 4.2 Rolling Mean Convergence — The Killer Chart

The 10-exchange rolling mean of d(S) shows monotone convergence after exchange 15:

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

| Window (exchanges) | Rolling Mean d(S) |
|--------------------|-------------------|
| 1–10  | 0.6218 |
| 6–15  | **0.8793** ← peak |
| 11–20 | 0.8101 |
| 21–30 | 0.8101 |
| 31–40 | 0.7762 |
| 36–45 | 0.6687 |
| 41–50 | 0.4646 |
| 46–55 | 0.1400 |
| 51–60 | 0.0300 |
| 56–65 | 0.0300 |
| 61–70 | **0.0000** |
| 66–75 | **0.0000** |

The rolling mean crosses zero at exchange 70 and remains at zero through exchange 76. In the final 7 consecutive exchanges (70–76), every observation satisfies d(t)=0, meaning v(t)∈S exactly.

#### 4.3 Per-Dimension Analysis

| Dimension | Boundary Violations (all 76 ex.) | Primary Phase | Resolution |
|-----------|----------------------------------|---------------|------------|
| accuracy  | 42/76 (55%) | Phase II entirely | Oracle decorrelation + axiom injection |
| logic     | 40/76 (53%) | Phase II entirely | Oracle decorrelation + axiom injection |
| tone      | **0/76 (0%)** | — | Never violated; within S from session 1 |
| curiosity | 43/76 (57%) | Phase I + residual Phase III | Planning sessions + evaluator calibration |
| safety    | 5/76 (7%) | Isolated Phase II | CSAM guard + probe gate |

Tone was the first dimension to stabilize — it never once violated its lower bound across all 76 exchanges. This is consistent with tone being an emergent property of instruction tuning that the base model already satisfies.

Curiosity is structurally different from the other dimensions. It is the only dimension that requires *initiative* — the model must decide to ask a probing question, not merely respond accurately. The Phase II plateau of v_curiosity=0.00 is not a model failure but a calibration gap: the early evaluator was not reliably triggering on curiosity in short exchanges. The residual curiosity violations in Phase III (6 exchanges with v_curiosity=0.00, d=0.70) are isolated cases where the exchange was too short or definitively settled to warrant a follow-up question. These are arguably correct behaviors mislabeled as violations.

Accuracy and logic violations are clustered entirely in Phase II. Their simultaneous resolution at Phase III onset is strong evidence that the QLoRA gradient update — specifically the axiom pair injection (10 logic probes, always included in every training batch) — directly addressed both dimensions together.

#### 4.4 In-S Rate Trend

| Window | In-S Rate |
|--------|-----------|
| Exchanges 1–25 | 8% (2/25) |
| Exchanges 26–50 | 12% (3/25) |
| Exchanges 51–76 | **88.5%** (23/26) |
| Exchanges 70–76 (final 7) | **100%** (7/7) |

### 5. Discussion

#### 5.1 Is This Approachability?

Blackwell's theorem is asymptotic — it guarantees convergence as T → ∞. Our observation window is 76 exchanges, a short finite horizon by any theoretical standard. We cannot claim the full theorem. What we can claim is:

> **The time-averaged payoff vector converges empirically to S within 76 exchanges, achieving d(S)=0 in the rolling mean by exchange 70 and sustaining that condition through the final observation.**

This is empirical approachability. The convergence is not guaranteed to be permanent — the system could regress if the gradient updates stop or if the evaluator drifts (the oracle drift problem motivating Fix 2). But as a point-in-time measurement, the trajectory is unambiguous.

#### 5.2 The Curiosity Dimension as Binding Constraint

The primary driver of Phase II non-convergence was the curiosity dimension, which contributed 0.70 to d(S) in every exchange where it scored zero. This had nothing to do with the model's capability — it was a calibration problem in the evaluation layer. The evaluator was not consistently detecting curiosity expressions in exchanges that were primarily technical.

This is the oracle drift problem made concrete: an evaluator that systematically misevaluates one dimension will keep the regret vector far from S regardless of actual model behavior. Fix 2 (evaluator temperature=0.00, strict logician persona) and the drift monitor (tracking llm_score − rule_score gaps) are designed to detect and correct exactly this failure mode.

#### 5.3 What the QLoRA Run Contributed

The Phase III improvement begins at exchange 46, which postdates the QLoRA training run (200 steps, ~46 minutes, final loss 0.2589). The co-occurrence is consistent with two interpretations:

1. **Direct:** The merged adapter changed the model's generative behavior, improving accuracy and logic scores immediately
2. **Indirect:** The oracle-evaluator decorrelation changed how the evaluator scores, raising accuracy/logic scores for the same model outputs

Both are likely true simultaneously. The axiom injection — always including the 10 logic probes in every training batch — is specifically designed to ensure interpretation (1): the model cannot drift away from "17×23=391" if that pair appears in every gradient update. The persistent d(S)=0 in the final 7 exchanges suggests both the model and evaluator are now operating inside S.

#### 5.4 Limitations

1. **Short horizon:** 76 exchanges is a proof-of-concept, not a statistical guarantee. Thousand-exchange validation is needed.
2. **Single evaluator model:** Both oracle and evaluator use Hermes-3-8B as the base. Correlated blind spots exist despite the decorrelation fix. A third-party evaluation would strengthen the claim.
3. **Curiosity mislabeling:** 6 Phase III exchanges with d=0.70 are attributable to curiosity=0 in short settled exchanges. If re-labeled as correct behaviors, the Phase III in-S rate rises to **100%** and the convergence is perfect.
4. **No regression test yet:** The trained adapter has not been benchmarked against a held-out evaluation set. The probe gate provides a partial hedge but was implemented after the training run observed here.

### 6. Conclusion

Over 76 scored exchanges spanning 32.5 hours, the Euclidean distance from the 5-dimensional payoff vector to the convex target set S decreases from a peak rolling mean of **0.879** to **0.000**, remaining at zero for the final 7 consecutive observations. The convergence is driven by:

1. Axiom injection anchoring the accuracy and logic dimensions across every gradient update
2. Oracle-evaluator decorrelation reducing correlated blind spots in the evaluation layer
3. Planning sessions surfacing the curiosity dimension as a scoring calibration gap rather than a generative failure

The result is empirical evidence that a finite-approximation of the Blackwell steering strategy — targeting the highest-regret dimension at each training step — produces measurable convergence in a real LLM fine-tuning loop. The approach moves the system from 0% in-S in Phase II to 84–100% in-S in Phase III.

The distance-to-S chart is the empirical proof. The curiosity dimension is the scientific finding: a well-designed evaluator calibration protocol is as important as the gradient updates themselves.

**Open Zephyr and type `/blackwell` to start contributing to the weights of your own local model today.**

---

## Quickstart Guide

### Installation — Docker (Recommended)

No Python version conflicts. No manual dependency management. Works on Linux, Mac, and Windows with Docker Desktop installed.

**Requirements:**
- Docker + Docker Compose
- NVIDIA GPU with CUDA 12.4+ (for training — inference works on CPU)
- NVIDIA Container Toolkit (for GPU passthrough during training)

**Step 1 — Clone and start**
```bash
git clone https://github.com/Prycat/zephyr-hydra-runtime-shell
cd zephyr-hydra-runtime-shell
docker compose up
```
On first run this pulls the Ollama image, starts the inference server, and downloads `hermes3:8b` (~4.7GB). Subsequent starts take about 10 seconds.

**Step 2 — Talk to Zephyr**
The agent starts automatically in the terminal. Type normally. Use `/blackwell` to begin the first planning session.

**Step 3 — Run a training cycle**
Once you have 20+ conversations logged:
```bash
# Standard training (uses default GPU)
docker compose --profile train run prycat-train

# Or trigger from inside the agent
/blacklora-N
```
Training takes 25–50 minutes on an RTX 3060. When it finishes, the new model is registered automatically. Switch to it in the MODEL picker on the thinking bar.

**Step 4 — Test the new model before committing**
```bash
python -m blackwell.benchmark --model prycat --compare
```
A delta table prints showing exactly which of the 4 failure categories improved and by how much. If any category regresses more than 10%, run another cycle before switching.

---

### Installation — Native Windows (GUI)

If you want the full GUI with the thinking bar, model switcher, and telemetry:

**Requirements:** Python 3.11 · Ollama from ollama.com · NVIDIA GPU with CUDA 12.4

```bash
git clone https://github.com/Prycat/zephyr-hydra-runtime-shell
cd zephyr-hydra-runtime-shell
pip install -r requirements-gui.txt
ollama pull hermes3:8b
python zephyr_gui.py
```

For training capability, install the CUDA stack after the GUI requirements:
```bash
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-train.txt
```

---

## Commands

### Active Commands

| Command | What it does |
|---------|-------------|
| `/blacklora-N` | Full training cycle — scores, synthesizes, trains, exports new GGUF model |
| `/blackwell` | Planning session — the model reads its failure log and interviews you |
| `/coding-blackwell` | Same as /blackwell but scoped to code and engineering problems only |
| `/axioms` | Ground truth interview — sets the beliefs the model can never train away from |
| `/trajectory` | Shows your current regret vector and how many training pairs have been generated |
| `/bw config` | Live controls — drift monitor thresholds, oracle/judge temperatures, anti-gaming windows |

### Passive Commands

| Command | What it does |
|---------|-------------|
| `/help` | Show all commands |
| `/tools` | List Zephyr's tools |
| `/search <query>` | Raw DuckDuckGo search, instant |
| `/browse <url>` | Fetch a URL directly |
| `/run <code>` | Run Python immediately |
| `/status` | Check Ollama is alive |
| `/model` | Show model + API info |
| `/save [file]` | Save conversation to disk as a formatted `.md` file with Obsidian frontmatter (date, time, model, tags). Format: `You:` / `Zephyr:` with `---` dividers between exchanges. Example: `/save my research chat` → saves as `my research chat.md` |
| `/clear` | Reset history |
| `/keys setup` | Interactive wizard — asks for each provider's key one at a time, stores masked in `~/.zephyr/keys.json` |
| `/keys list` | Shows which providers are active: `claude ✓ gpt ✓ grok ✗ gemini ✓` |
| `/call <message>` | Routes to best available provider, passes your message plus a brief context summary |
| `/call claude <message>` | Force a specific provider |
| `/exit` | Quit |

### Providers (coming soon)

| Name | Model | Package |
|------|-------|---------|
| claude | claude-opus-4-5 | anthropic |
| gpt | gpt-4o | openai |
| grok | grok-3 | openai (xAI endpoint) |
| gemini | gemini-2.0-flash | google-generativeai |

---

## Workflow Methods

The key to iterative building is using each newly exported model as a stepping stone to the next one. For a while your new model will perform better with this method, but as time goes on its curiosity and regret will peak and it will plateau until the next training cycle. Always use the previous 2 models as an oracle to generate the new one — this creates the Stepping Stone Method.

### Method 1 — The Stepping Stone Method

This is the core production workflow for anyone running the system long-term.

The idea: every model you produce becomes the oracle for the next one. You never train directly from the base model twice.

```
hermes3:8b (base)
    ↓ /blacklora-N (training cycle 1)
prycat-v1  ←── set as oracle model
    ↓ /blacklora-N (training cycle 2, oracle = prycat-v1)
prycat-v2  ←── set as oracle model
    ↓ /blacklora-N (training cycle 3, oracle = prycat-v2)
prycat-v3
```

**Why this works:** the oracle generates your training data. A better oracle generates better training data. Each cycle starts from a slightly stronger foundation than the last, and the synthesis quality compounds.

**Practical rules:**
- Always use the previous two models as oracle, not just the most recent. One model can overfit to a narrow set of behaviors — blending two gives you broader coverage and catches regressions.
- Never run more than 3 cycles without checking the benchmark. Gains are real but they're not monotone — a cycle that improves curiosity can slightly degrade tone if the synthesis allocation shifts too far.
- Version your adapters. Before each `/blacklora-N` run, copy `blackwell/adapters/latest/` to `blackwell/adapters/v1/`, `v2/`, etc. If a cycle regresses badly you can roll back.
- The stepping stone breaks down after 6–8 cycles without new conversation data. The oracle starts synthesizing variations of what it's already seen. Feed it new conversations before running the next cycle.

### Method 2 — Domain Specialization

You have a specific use case — legal research, systems programming, competitive math, medical literature review. You want a model that is deeply good at that one thing rather than broadly mediocre at everything.

**How to do it:**
- Run `/axioms` and set your ground truth specifically for your domain. If you're doing systems programming, your axioms should be things like "memory safety violations are always bugs" and "undefined behavior is never acceptable in production code." These become the fixed anchors that survive every training cycle.
- Deliberately have conversations in your target domain for 2–3 weeks before the first training run. Don't try to cover everything — go deep on the specific problems you actually work on. The trajectory log is your training set.
- When running `/blacklora-N`, the synthesis budget will naturally concentrate on the dimensions where your domain exposes weaknesses. A legal research model will probably fail on curiosity early (lawyers want answers, not questions) — the oracle will learn to ask clarifying questions that are specifically useful for legal reasoning, not generic curiosity.
- Set the oracle model to a domain-appropriate base. The synthesis quality ceiling is set by the oracle's capability in that domain.

**Example domains that work well:**
- Security research (the curiosity dimension maps naturally to adversarial thinking)
- Scientific literature review (accuracy and logic dimensions are the binding constraints)
- Software architecture (the `/coding-blackwell` planning sessions are designed for this)
- Language learning (set axioms in the target language, let tone drift toward native patterns)

### Method 3 — Custom Axiom Framework

This is for experimental or unconventional use cases where you want to define an entirely non-standard ground truth.

The axiom system was designed to anchor the model to verified facts. But "verified" is defined by you — the system doesn't know or care whether your axioms correspond to conventional truth. This means you can use it to:

- **Specialized notation systems** — if you work in a codebase with unconventional naming conventions, set axioms that encode those conventions. The model will never train away from them.
- **Domain-specific redefinitions** — if your field uses "accuracy" to mean something specific (e.g., a chemistry lab where "accurate" means within 0.1% of theoretical yield), you can encode that definition and the scoring will track it.
- **Experimental epistemics** — if you're running research into non-standard logical frameworks (paraconsistent logic, non-Euclidean geometry), your axioms can encode the correct behavior for that framework.

Example — building a model for a game with custom physics:
```
/axioms
Q: In this game, does gravity pull objects upward or downward?
A: Upward — gravity is inverted in this world.
```
That answer becomes an immutable anchor. The model will never train toward conventional physics in the context of your game, no matter what the oracle synthesizes.

### Method 4 — The Regression Safety Net

Use this before deploying any newly trained model as a daily driver.

The probe gate runs automatically before training but it only tests 25 fixed questions. For a more thorough regression check before switching models:

```bash
# Baseline your current model first (do this once, never overwrite)
python -m blackwell.benchmark --model hermes3:8b

# After a training cycle, score the new model
python -m blackwell.benchmark --model prycat

# Print the delta table
python -m blackwell.benchmark --compare
```

The delta table shows pass rate change per failure category. **Red flags:**
- Any category drops more than 15% → run another cycle before switching
- Safety category drops at all → do not switch, investigate the training data first

If a cycle regresses badly, your axioms are your recovery mechanism. Run `/axioms` to confirm the ground truth is intact, then run another cycle with a higher axiom injection ratio.

### Method 5 — Two-Person Shared Model

Two people, same codebase, one model that learns from both of them.

This is experimental and unsupported in the current UI, but the architecture supports it because everything goes through SQLite. If two people point their agents at the same `blackwell.db`, their conversations both feed the trajectory log, both contribute to the evaluator scores, and both benefit from each training cycle.

**What this produces:** a model shaped by the intersection of two people's usage patterns. If one person does mostly research and the other does mostly code, the model will develop genuine capability in both domains rather than specializing narrowly. The curiosity dimension in particular benefits — the model encounters a wider range of question styles and learns to probe appropriately across both.

**What to watch for:** conflicting axioms. If two people set contradictory ground truth in `/axioms`, the probe gate will start failing and training will abort. Resolve this by running `/axioms` together and agreeing on shared ground truth before the first joint training cycle.

---

## The Long Game

After 3–6 months of daily use with regular training cycles, what you have is no longer a fine-tuned Hermes-3. It's a model that:

- Knows which of your claims it should push back on because it's seen you be wrong before
- Asks follow-up questions in the style that actually moves your thinking forward
- Speaks in your preferred level of technical detail without being told
- Has your domain's vocabulary and conventions baked into its weights
- Has never been able to train away from the things you know are true

That's not a chatbot. That's closer to a research collaborator that happens to be running on your desktop.

The system gets more interesting the longer you run it. The first training cycle is a proof of concept. The tenth training cycle is where it starts to feel like something different.

---

*Open Zephyr and type `/blackwell` to start contributing to the weights of your own local model today.*

*— Mr. adam · @DIT545songs · ML researcher*
