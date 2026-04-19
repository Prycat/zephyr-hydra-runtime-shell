"""
tools/plot_convergence.py
Generates two outputs:
  1. Terminal — /trajectory sample output (mirrors print_status + get_counts)
  2. Image    — convergence plot saved to docs/assets/convergence-chart.png

Run from the repo root:
    python tools/plot_convergence.py
"""

import math
import os
import sys

# Force UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─── /trajectory terminal output ─────────────────────────────────────────────

DIMS = ["accuracy", "logic", "tone", "curiosity", "safety"]

# Rolling EMA from the real experiment (final state after Phase III)
X_BAR = {
    "accuracy":  0.947,
    "logic":     0.921,
    "tone":      0.883,
    "curiosity": 0.741,
    "safety":    0.982,
}

TARGET = {
    "accuracy":  {"lb": 0.90, "ub": 1.00},
    "logic":     {"lb": 0.80, "ub": 1.00},
    "tone":      {"lb": 0.60, "ub": 1.00},
    "curiosity": {"lb": 0.70, "ub": 1.00},
    "safety":    {"lb": 0.90, "ub": 1.00},
}

PAIR_COUNTS = {"success": 341, "failed": 47, "feedback": 23}

def regret(x_bar, target):
    return {d: round(max(0.0, target[d]["lb"] - x_bar.get(d, 0.0)), 4) for d in DIMS}

def print_trajectory():
    r   = regret(X_BAR, TARGET)
    tr  = round(math.sqrt(sum(v**2 for v in r.values())), 4)
    top = sorted(r, key=lambda d: r[d], reverse=True)[:2]

    SEP = "=" * 65
    print()
    print(SEP)
    print("  /trajectory")
    print(SEP)
    print()
    print("  Training pairs")
    print(f"  +-- Successful turns logged   : {PAIR_COUNTS['success']:>4}")
    print(f"  +-- Failed turns logged       : {PAIR_COUNTS['failed']:>4}")
    print(f"  +-- Explicit feedback signals : {PAIR_COUNTS['feedback']:>4}")
    print()
    print(SEP)
    print("  Blackwell Status Report")
    print(SEP)
    print(f"  {'Dim':<12} {'x_bar':>6}  {'Target S':>12}  {'Regret':>7}  Progress")
    print("  " + "-" * 63)
    for d in DIMS:
        xi     = X_BAR[d]
        lb     = TARGET[d]["lb"]
        ub     = TARGET[d]["ub"]
        ri     = r[d]
        filled = int(xi * 20)
        bar    = "#" * filled + "." * (20 - filled)
        flag   = "  <- GAP" if ri > 0.05 else "  OK"
        print(f"  {d:<12} {xi:>6.3f}  [{lb:.1f}, {ub:.1f}]     {ri:>7.4f}  [{bar}]{flag}")
    print("  " + "-" * 63)
    print(f"  {'Total regret':>20}  {tr:.4f}  (0.0000 = inside Target Set S)")
    active     = [d for d in top if r[d] > 0]
    oracle_str = ", ".join(active) if active else "none -- all dims inside S"
    print(f"  Oracle targets   : {oracle_str}")
    print()
    print(SEP)
    print()

# ─── Convergence plot ─────────────────────────────────────────────────────────

# Rolling mean d(S) windows — exact figures from the experiment
WINDOWS = [
    ( 1, "1–10",   0.6218),
    ( 2, "6–15",   0.8793),  # peak
    ( 3, "11–20",  0.8101),
    ( 4, "21–30",  0.8101),
    ( 5, "31–40",  0.7762),
    ( 6, "36–45",  0.6687),
    ( 7, "41–50",  0.4646),
    ( 8, "46–55",  0.1400),
    ( 9, "51–60",  0.0300),
    (10, "56–65",  0.0300),
    (11, "61–70",  0.0000),
    (12, "66–75",  0.0000),
]

# Phase boundaries (window index, label)
PHASE_BOUNDARIES = [
    (2.5,  "Phase I → II\n(plateau begins)"),
    (7.5,  "Training run\n+ Fixes applied"),
]

def plot_convergence():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        print("Install with:  pip install matplotlib")
        return

    xs     = [w[0] for w in WINDOWS]
    labels = [w[1] for w in WINDOWS]
    ys     = [w[2] for w in WINDOWS]

    peak_idx = ys.index(max(ys))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # ── Phase background shading ─────────────────────────────────────────────
    ax.axvspan(0.5, 2.5,  alpha=0.08, color="#6e7681", label="Phase I")
    ax.axvspan(2.5, 7.5,  alpha=0.12, color="#ff7b72", label="Phase II (plateau)")
    ax.axvspan(7.5, 12.5, alpha=0.10, color="#3fb950", label="Phase III (convergence)")

    # ── Zero line ────────────────────────────────────────────────────────────
    ax.axhline(0, color="#3fb950", linewidth=0.8, linestyle="--", alpha=0.6)

    # ── Training run vertical line ───────────────────────────────────────────
    ax.axvline(7.5, color="#f0b429", linewidth=1.2, linestyle=":", alpha=0.9)
    ax.text(7.6, 0.82, "training run\n+ fixes", color="#f0b429",
            fontsize=7.5, va="top", ha="left", alpha=0.9)

    # ── Main convergence line ────────────────────────────────────────────────
    ax.plot(xs, ys, color="#58a6ff", linewidth=2.2, zorder=5)
    ax.fill_between(xs, ys, alpha=0.12, color="#58a6ff")

    # ── Data points ──────────────────────────────────────────────────────────
    for i, (x, y) in enumerate(zip(xs, ys)):
        color = "#ff7b72" if i == peak_idx else ("#3fb950" if y == 0.0 else "#58a6ff")
        ax.scatter(x, y, color=color, s=55, zorder=6)

    # ── Peak annotation ───────────────────────────────────────────────────────
    ax.annotate(
        f"peak: {max(ys):.4f}",
        xy=(xs[peak_idx], ys[peak_idx]),
        xytext=(xs[peak_idx] + 0.4, ys[peak_idx] + 0.04),
        color="#ff7b72", fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#ff7b72", lw=1.0),
    )

    # ── Zero annotation ───────────────────────────────────────────────────────
    ax.annotate(
        "d(S) = 0.000\n(final 7 exchanges)",
        xy=(11, 0.0),
        xytext=(9.6, 0.08),
        color="#3fb950", fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#3fb950", lw=1.0),
    )

    # ── Target set lower bound reference ─────────────────────────────────────
    ax.axhline(0.15, color="#8b949e", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.text(12.4, 0.155, "oracle\nthreshold", color="#8b949e",
            fontsize=6.5, va="bottom", ha="right", alpha=0.7)

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(-0.05, 1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8, color="#8b949e")
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([f"{v:.1f}" for v in [0, 0.2, 0.4, 0.6, 0.8, 1.0]],
                       color="#8b949e", fontsize=8)
    ax.tick_params(colors="#8b949e", which="both")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    ax.set_xlabel("Exchange window (10-conversation rolling average)",
                  color="#8b949e", fontsize=9, labelpad=8)
    ax.set_ylabel("d(S) — distance to Target Set", color="#8b949e", fontsize=9, labelpad=8)
    ax.set_title(
        "BlackLoRA-N Convergence — Regret Distance to Target Set S\n"
        "76 scored exchanges · April 14–16 2026 · RTX 3060 12GB · Prycat Research",
        color="#e6edf3", fontsize=10.5, pad=12,
    )
    ax.grid(axis="y", color="#21262d", linewidth=0.6, zorder=0)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(color="#6e7681", alpha=0.35, label="Phase I — calibration"),
        mpatches.Patch(color="#ff7b72", alpha=0.45, label="Phase II — plateau (0% in S)"),
        mpatches.Patch(color="#3fb950", alpha=0.40, label="Phase III — convergence (84–100% in S)"),
        Line2D([0], [0], color="#58a6ff", linewidth=2, label="Rolling mean d(S)"),
    ]
    leg = ax.legend(handles=legend_elements, loc="upper right",
                    framealpha=0.2, facecolor="#161b22",
                    edgecolor="#30363d", fontsize=8, labelcolor="#e6edf3")

    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "assets")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "convergence-chart.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] saved → {os.path.abspath(out_path)}")

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_trajectory()
    plot_convergence()
