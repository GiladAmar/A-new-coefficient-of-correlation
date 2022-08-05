import sys
from itertools import product

import numba as nb
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata


def x_ordered_rank(x):
    # source (https://stackoverflow.com/a/47430384/1628971)
    len_x = len(x)
    randomized_indices = np.random.choice(np.arange(len_x), len_x, replace=False)
    randomized = [x[idx] for idx in randomized_indices]
    # same as pandas rank method 'first'
    ranked_data = rankdata(randomized, method="ordinal")
    # Reindexing based on pairs of indices before and after
    return [ranked_data[j] for i, j in sorted(zip(randomized_indices, range(len_x)))]


def XI_coef(xvec, yvec, simple=True):
    n = len(xvec)
    PI = x_ordered_rank(xvec)
    fr = rankdata(yvec, method="max") / n
    gr = rankdata(-yvec, method="max") / n

    ord = np.argsort(PI)
    fr = fr[ord]
    A1 = np.sum(np.abs(fr[1: n - 1] - fr[2:n])) / (2.0 * n)
    CU = np.mean(gr * (1.0 - gr))

    xi = 1.0 - A1 / (CU + sys.float_info.epsilon)

    return xi if simple else [xi, fr, CU]


def XI_coef_matrix(np_array):
    assert len(np_array.shape) == 2
    _, w = np_array.shape

    return np.array(
        [
            XI_coef(np_array[:, i], np_array[:, j])
            for i, j in product(range(w), repeat=2)
        ]
    ).reshape((w, w))


def XI_coef_matrix_parallel(np_array):
    assert len(np_array.shape) == 2
    _, w = np_array.shape

    def process_index(i, j):
        return XI_coef(np_array[:, i], np_array[:, j])

    results = Parallel(n_jobs=-1)(
        delayed(process_index)(i, j) for i, j in product(range(w), repeat=2)
    )
    return np.array(results).reshape((w, w))


@nb.njit
def nb_unique1d(ar, return_index=True):
    """
    Find the unique elements of an array, ignoring shape.
    """
    ar = ar.flatten()

    optional_indices = return_index

    if optional_indices:
        perm = ar.argsort(kind="mergesort")
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    mask = np.empty(aux.shape, dtype=np.bool_)
    mask[:1] = True
    if aux.shape[0] > 0 and aux.dtype.kind in "cfmM" and np.isnan(aux[-1]):
        if aux.dtype.kind == "c":  # for complex all NaNs are considered equivalent
            aux_firstnan = np.searchsorted(np.isnan(aux), True, side="left")
        else:
            aux_firstnan = np.searchsorted(aux, aux[-1], side="left")
        mask[1:aux_firstnan] = aux[1:aux_firstnan] != aux[: aux_firstnan - 1]
        mask[aux_firstnan] = True
        mask[aux_firstnan + 1:] = False
    else:
        mask[1:] = aux[1:] != aux[:-1]

    imask = np.cumsum(mask) - 1
    inv_idx = np.empty(mask.shape, dtype=np.intp)
    inv_idx[perm] = imask
    idx = np.append(np.nonzero(mask)[0], mask.size)

    # idx      #inverse   #counts
    return aux[mask], perm[mask], inv_idx, np.diff(idx)


@nb.njit
def XI_numba(X, Y):
    """xi correlation coefficient"""
    X_copy = X.copy()
    n = X.size
    np.random.shuffle(X_copy)  # to make up for breaks not being properly random
    xi = np.argsort(X_copy, kind="quicksort")
    Y = Y[xi]
    _, _, b, c = nb_unique1d(Y)
    r = np.cumsum(c)[b]
    _, _, b, c = nb_unique1d(-Y)
    l = np.cumsum(c)[b]
    return 1 - n * np.abs(np.diff(r)).sum() / (2 * (l * (n - l)).sum())


def XI_numba_matrix(np_array):
    assert len(np_array.shape) == 2
    _, w = np_array.shape

    def process_index(i, j):
        return XI_numba(np_array[:, i], np_array[:, j])

    results = Parallel(n_jobs=-1)(
        delayed(process_index)(i, j) for i, j in product(range(w), repeat=2)
    )
    return np.array(results).reshape((w, w))
