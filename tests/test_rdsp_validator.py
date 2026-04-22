"""Tests for rdsp_validator.py — prune commit/rollback gate."""
import pytest


from blackwell.rdsp_validator import score_is_acceptable, ValidationResult


def test_acceptable_small_drop():
    assert score_is_acceptable(before=0.50, after=0.48, tolerance=0.05) is True


def test_acceptable_improvement():
    assert score_is_acceptable(before=0.45, after=0.50, tolerance=0.02) is True


def test_unacceptable_large_drop():
    assert score_is_acceptable(before=0.50, after=0.40, tolerance=0.05) is False


def test_exact_tolerance_boundary():
    """Exactly at the tolerance boundary is acceptable (drop == tolerance → keep)."""
    assert score_is_acceptable(before=0.50, after=0.45, tolerance=0.05) is True


def test_none_after_score_rejected():
    """If post-prune benchmark returns None, reject."""
    assert score_is_acceptable(before=0.50, after=None, tolerance=0.05) is False


def test_validation_result_compression_ratio():
    vr = ValidationResult(
        benchmark="cruxeval",
        score_before=0.50,
        score_after=0.48,
        acceptable=True,
        n_candidates=48,
        total_heads=1024,
    )
    assert vr.compression_ratio == pytest.approx(48 / 1024, rel=1e-4)


def test_validation_result_score_delta():
    vr = ValidationResult(
        benchmark="cruxeval",
        score_before=0.50,
        score_after=0.48,
        acceptable=True,
        n_candidates=48,
        total_heads=1024,
    )
    assert vr.score_delta == pytest.approx(-0.02, abs=1e-6)


def test_validation_result_delta_none_when_no_after():
    vr = ValidationResult(
        benchmark="cruxeval",
        score_before=0.50,
        score_after=None,
        acceptable=False,
        n_candidates=10,
        total_heads=1024,
    )
    assert vr.score_delta is None
