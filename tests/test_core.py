"""Tests for xicor.core."""

import numpy as np
import pytest

from xicor import xi_coef, xi_coef_matrix, xi_coef_matrix_parallel


# ---------------------------------------------------------------------------
# xi_coef
# ---------------------------------------------------------------------------


def test_self_correlation_approaches_one():
    """xi_coef(x, x) approaches 1 as n grows; should be > 0.98 for n=200."""
    # The coefficient is asymptotic: xi → 1 as n → ∞ for perfect dependence.
    x = np.arange(200, dtype=float)
    assert xi_coef(x, x) > 0.98


def test_returns_float_in_unit_interval():
    rng = np.random.default_rng(0)
    x, y = rng.random(300), rng.random(300)
    result = xi_coef(x, y)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_simple_false_returns_list():
    x = np.arange(50, dtype=float)
    result = xi_coef(x, x, simple=False)
    assert isinstance(result, list) and len(result) == 3


def test_seed_reproducibility():
    rng = np.random.default_rng(1)
    x, y = rng.random(200), rng.random(200)
    assert xi_coef(x, y, seed=7) == xi_coef(x, y, seed=7)


def test_nonlinear_relationship():
    """XI detects x → sin(x) but not the reverse."""
    x = np.linspace(0, 10 * np.pi, 2000)
    y = np.sin(x)
    assert xi_coef(x, y) > 0.7, "should detect x→sin(x)"
    assert xi_coef(y, x) < 0.3, "y→x should be near-zero (not injective)"


def test_independent_variables_near_zero():
    rng = np.random.default_rng(42)
    x, y = rng.random(5000), rng.random(5000)
    # Independence → xi near 0; allow generous tolerance for randomness
    assert xi_coef(x, y) < 0.1


# ---------------------------------------------------------------------------
# xi_coef_matrix
# ---------------------------------------------------------------------------


def test_matrix_shape():
    rng = np.random.default_rng(0)
    arr = rng.random((200, 5))
    result = xi_coef_matrix(arr)
    assert result.shape == (5, 5)


def test_matrix_diagonal_is_one():
    rng = np.random.default_rng(0)
    arr = rng.random((200, 5))
    result = xi_coef_matrix(arr)
    np.testing.assert_array_equal(np.diag(result), np.ones(5))


def test_matrix_values_bounded():
    # Off-diagonal entries can be slightly negative for small samples
    # (theoretical lower bound ≈ -1/(2(n+1))). Upper bound is 1.
    rng = np.random.default_rng(0)
    arr = rng.random((100, 4))
    result = xi_coef_matrix(arr)
    assert np.all(result <= 1.0)
    assert np.all(result >= -1.0)


def test_matrix_not_symmetric():
    """xi_coef(x, y) != xi_coef(y, x) in general → matrix not symmetric."""
    x = np.linspace(0, 4 * np.pi, 500)
    y = np.sin(x)
    arr = np.column_stack([x, y])
    result = xi_coef_matrix(arr)
    # Off-diagonal elements should differ meaningfully
    assert abs(result[0, 1] - result[1, 0]) > 0.2


# ---------------------------------------------------------------------------
# xi_coef_matrix_parallel
# ---------------------------------------------------------------------------


def test_parallel_matches_sequential():
    rng = np.random.default_rng(99)
    arr = rng.random((300, 5))
    seq = xi_coef_matrix(arr)
    par = xi_coef_matrix_parallel(arr)
    np.testing.assert_allclose(seq, par, rtol=1e-10)


def test_parallel_shape():
    rng = np.random.default_rng(0)
    arr = rng.random((100, 3))
    result = xi_coef_matrix_parallel(arr)
    assert result.shape == (3, 3)
