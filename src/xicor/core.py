"""Core numpy/scipy implementation of the XI correlation coefficient.

Reference: Sourav Chatterjee, "A New Coefficient of Correlation",
Journal of the American Statistical Association, 2021.
https://arxiv.org/abs/1909.10140
"""

from __future__ import annotations

import sys

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata


def xi_coef(
    xvec: np.ndarray,
    yvec: np.ndarray,
    simple: bool = True,
    seed: int = 42,
) -> float | list:
    """Compute the XI correlation coefficient between two vectors.

    Measures the degree to which *y* is a function of *x*.  The coefficient
    is in [0, 1]: values near 1 indicate a strong functional relationship
    x → y; values near 0 indicate independence.  Note that the coefficient
    is directional: ``xi_coef(x, y) != xi_coef(y, x)`` in general.

    Parameters
    ----------
    xvec:
        The predictor vector (ranked with random tie-breaking).
    yvec:
        The response vector.
    simple:
        If *True* (default) return only the scalar coefficient.
        If *False* return ``[xi, fr, CU]`` for inspection.
    seed:
        Random seed used to break ties in the x-ranking.

    Returns
    -------
    float or list
        The XI coefficient, or ``[xi, fr, CU]`` when ``simple=False``.
    """
    np.random.seed(seed)
    n = len(xvec)

    # Equivalent to R's rank(xvec, ties.method="random"):
    # lexsort with a random tiebreak gives the permutation order directly,
    # avoiding building a PI array and argsort-ing it.
    ord_ = np.lexsort((np.random.random(n), xvec))

    fr = rankdata(yvec, method="max") / n
    gr = rankdata(-yvec, method="max") / n

    fr = fr[ord_]
    A1 = np.sum(np.abs(fr[:-1] - fr[1:])) / (2.0 * n)
    CU = np.mean(gr * (1.0 - gr))

    xi = 1.0 - A1 / (CU + sys.float_info.epsilon)
    if simple:
        return xi

    return [xi, fr, CU]


def xi_coef_matrix(np_array: np.ndarray) -> np.ndarray:
    """Compute the XI correlation matrix for all column pairs.

    Parameters
    ----------
    np_array:
        2-D array of shape ``(n_samples, n_features)``.

    Returns
    -------
    np.ndarray
        Square matrix of shape ``(n_features, n_features)`` where entry
        ``[i, j]`` is ``xi_coef(np_array[:, i], np_array[:, j])``.
        Diagonal entries are 1.
    """
    assert np_array.ndim == 2
    n, w = np_array.shape

    # Precompute fr and CU for every column once (as y).
    # Reduces rankdata calls from O(w²) to O(w).
    FR = np.empty((n, w))
    CU = np.empty(w)
    for j in range(w):
        FR[:, j] = rankdata(np_array[:, j], method="max") / n
        gr_j = rankdata(-np_array[:, j], method="max") / n
        CU[j] = np.mean(gr_j * (1.0 - gr_j))

    # Precompute the sort permutation for every column once (as x).
    ORD = np.empty((w, n), dtype=np.intp)
    for i in range(w):
        np.random.seed(42)
        ORD[i] = np.lexsort((np.random.random(n), np_array[:, i]))

    # For each x-column reorder all y-rank columns in one fancy-index step,
    # then compute every A1 value simultaneously with vectorised numpy ops.
    result = np.empty((w, w))
    for i in range(w):
        FR_sorted = FR[ORD[i], :]  # shape (n, w)
        A1 = np.sum(np.abs(FR_sorted[:-1] - FR_sorted[1:]), axis=0) / (2.0 * n)
        result[i] = 1.0 - A1 / (CU + sys.float_info.epsilon)
        result[i, i] = 1.0

    return result


def xi_coef_matrix_parallel(np_array: np.ndarray) -> np.ndarray:
    """Parallel version of :func:`xi_coef_matrix` using all available CPUs.

    Parameters
    ----------
    np_array:
        2-D array of shape ``(n_samples, n_features)``.

    Returns
    -------
    np.ndarray
        Square matrix of shape ``(n_features, n_features)``.
    """
    assert np_array.ndim == 2
    n, w = np_array.shape

    # Precompute fr and CU for every column once (as y).
    FR = np.empty((n, w))
    CU = np.empty(w)
    for j in range(w):
        FR[:, j] = rankdata(np_array[:, j], method="max") / n
        gr_j = rankdata(-np_array[:, j], method="max") / n
        CU[j] = np.mean(gr_j * (1.0 - gr_j))

    # Precompute the sort permutation for every column once (as x).
    ORD = np.empty((w, n), dtype=np.intp)
    for i in range(w):
        np.random.seed(42)
        ORD[i] = np.lexsort((np.random.random(n), np_array[:, i]))

    def _process_row(i: int) -> np.ndarray:
        FR_sorted = FR[ORD[i], :]  # shape (n, w)
        A1 = np.sum(np.abs(FR_sorted[:-1] - FR_sorted[1:]), axis=0) / (2.0 * n)
        row = 1.0 - A1 / (CU + sys.float_info.epsilon)
        row[i] = 1.0
        return row

    results = Parallel(n_jobs=-1)(delayed(_process_row)(i) for i in range(w))
    return np.array(results)
