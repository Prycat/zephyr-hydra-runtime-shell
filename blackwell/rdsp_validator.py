# -*- coding: utf-8 -*-
"""
blackwell/rdsp_validator.py
Benchmark-gated commit/rollback decision for RDSP.

Before committing a prune, the orchestrator:
  1. Records pre-prune benchmark score
  2. Applies the head mask
  3. Re-exports to GGUF and reloads Ollama
  4. Runs the same benchmark again
  5. Calls score_is_acceptable() — if True, commit; if False, rollback

A ValidationResult is returned with all metadata for logging to blackwell.db.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


@dataclass
class ValidationResult:
    """
    Result of a prune validation cycle.

    Attributes
    ----------
    benchmark        : benchmark name used for validation
    score_before     : benchmark score before pruning
    score_after      : benchmark score after pruning (None if run failed)
    acceptable       : True if the prune should be committed
    n_candidates     : number of heads pruned
    total_heads      : total heads in the model
    compression_ratio: heads_pruned / total_heads (computed property)
    score_delta      : score_after - score_before (computed property, None if no after)
    """
    benchmark:    str
    score_before: float
    score_after:  float | None
    acceptable:   bool
    n_candidates: int
    total_heads:  int

    @property
    def compression_ratio(self) -> float:
        """Fraction of heads pruned."""
        return self.n_candidates / self.total_heads if self.total_heads else 0.0

    @property
    def score_delta(self) -> float | None:
        """Score change from pruning. None if score_after is unavailable."""
        if self.score_after is None:
            return None
        return round(self.score_after - self.score_before, 4)


def score_is_acceptable(
    before: float,
    after: float | None,
    tolerance: float = 0.05,
) -> bool:
    """
    Return True if the post-prune score is acceptable.

    Parameters
    ----------
    before    : benchmark score before pruning
    after     : benchmark score after pruning (None if unavailable)
    tolerance : maximum allowed score drop (default 0.05 = 5 percentage points)

    Returns
    -------
    True if drop <= tolerance (or improvement). False if after is None or drop > tolerance.
    """
    if after is None:
        return False
    return (before - after) <= tolerance


def validate(
    benchmark: str,
    score_before: float,
    score_after: float | None,
    n_candidates: int,
    total_heads: int,
    tolerance: float = 0.05,
) -> ValidationResult:
    """
    Build a ValidationResult and log the commit/rollback decision.

    Parameters
    ----------
    benchmark    : benchmark name used for validation
    score_before : score recorded before pruning
    score_after  : score recorded after pruning (None if run failed)
    n_candidates : number of heads pruned
    total_heads  : total heads in the model (32 * 32 = 1024 for Llama-3.1-8B)
    tolerance    : max acceptable score drop

    Returns
    -------
    ValidationResult with all metadata populated.
    """
    acceptable = score_is_acceptable(before=score_before, after=score_after,
                                     tolerance=tolerance)
    vr = ValidationResult(
        benchmark    = benchmark,
        score_before = score_before,
        score_after  = score_after,
        acceptable   = acceptable,
        n_candidates = n_candidates,
        total_heads  = total_heads,
    )

    verdict = "COMMIT" if acceptable else "ROLLBACK"
    delta   = vr.score_delta
    delta_s = f"{delta:+.3f}" if delta is not None else "N/A"
    print(
        f"[rdsp_validator] {verdict}: {benchmark} "
        f"{score_before:.3f} → {score_after if score_after is not None else 'N/A'} "
        f"(Δ{delta_s}, tolerance={tolerance:.2f})",
        flush=True,
    )
    return vr
