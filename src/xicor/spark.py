"""XI correlation coefficient for PySpark DataFrames.

``xi_coef`` stays fully distributed: it uses Spark Window functions for all
ranking steps and performs only a single lightweight ``.collect()`` call to
retrieve the two final aggregate scalars (A1 and CU).

``xi_coef_matrix`` collects the selected columns to the driver and delegates
to the numpy implementation.  This is unavoidable: the XI coefficient is a
pairwise statistic across *columns*, not rows, so the entire column set must
be available in one place to compute the (w × w) matrix.  For very large
DataFrames sample first::

    xi_coef_matrix(df.sample(fraction=0.1), columns=["a", "b", "c"])

Install PySpark support::

    pip install xicor[spark]
    # or
    uv add "xicor[spark]"
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from pyspark.sql import DataFrame


def _require_pyspark() -> None:
    try:
        import pyspark  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PySpark is required for xicor.spark. "
            "Install it with:  pip install 'xicor[spark]'"
        ) from exc


def xi_coef(
    df: DataFrame,
    x_col: str,
    y_col: str,
    seed: int = 42,
) -> float:
    """Compute the XI correlation coefficient between two columns of a Spark DataFrame.

    All ranking is performed with Spark Window functions so data never leaves
    the cluster during computation.  Only the two final aggregate scalars
    (A1 and CU) are collected to the driver.

    Parameters
    ----------
    df:
        Input Spark DataFrame.
    x_col:
        Name of the predictor column.
    y_col:
        Name of the response column.
    seed:
        Random seed used to break ties in the x-ranking.

    Returns
    -------
    float
        XI coefficient in approximately [0, 1].
    """
    _require_pyspark()
    import pyspark.sql.functions as F
    from pyspark.sql import Window

    n = df.count()
    if n == 0:
        return 0.0

    df_work = df.select(x_col, y_col)

    # ── PI: random rank of x, ties broken by a stable random column ──────────
    # Materialise _rand before using it in the Window so the values are
    # consistent across the plan (rand() without persist would be re-evaluated).
    df_work = df_work.withColumn("_rand", F.rand(seed=seed))
    pi_window = Window.orderBy(x_col, "_rand")
    df_work = df_work.withColumn("_pi", F.row_number().over(pi_window))

    # ── fr: max rank of y / n ────────────────────────────────────────────────
    # COUNT(*) with RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW over
    # ascending y gives count(y_j <= y_i), which equals
    # scipy.stats.rankdata(y, method="max").
    fr_window = Window.orderBy(y_col).rangeBetween(Window.unboundedPreceding, 0)
    df_work = df_work.withColumn("_fr", F.count(F.lit(1)).over(fr_window) / n)

    # ── gr: max rank of (−y) / n  ────────────────────────────────────────────
    # Negating y and using the same ascending-RANGE logic gives
    # count(−y_j <= −y_i) = count(y_j >= y_i), equivalent to
    # scipy.stats.rankdata(−y, method="max").
    df_work = df_work.withColumn("_neg_y", -F.col(y_col))
    gr_window = Window.orderBy("_neg_y").rangeBetween(Window.unboundedPreceding, 0)
    df_work = df_work.withColumn("_gr", F.count(F.lit(1)).over(gr_window) / n)

    # ── Consecutive fr differences after sorting by PI ───────────────────────
    pi_sort_window = Window.orderBy("_pi")
    df_work = (
        df_work
        .withColumn("_fr_lag", F.lag("_fr", 1).over(pi_sort_window))
        .withColumn("_abs_diff", F.abs(F.col("_fr") - F.col("_fr_lag")))
    )

    # ── Single aggregation to the driver ─────────────────────────────────────
    row = df_work.agg(
        (F.sum("_abs_diff") / (2.0 * n)).alias("A1"),
        F.mean(F.col("_gr") * (1.0 - F.col("_gr"))).alias("CU"),
    ).collect()[0]

    A1 = row["A1"] or 0.0  # null if n == 1 (no lag differences)
    CU = row["CU"] or 0.0

    return 1.0 - A1 / (CU + sys.float_info.epsilon)


def xi_coef_matrix(
    df: DataFrame,
    columns: list[str] | None = None,
    parallel: bool = False,
) -> pd.DataFrame:
    """Compute the XI correlation matrix for the given columns.

    Collects the selected columns to the driver and delegates to the numpy
    implementation (:func:`xicor.core.xi_coef_matrix` or its parallel
    variant).  The returned DataFrame has labelled rows and columns.

    Parameters
    ----------
    df:
        Input Spark DataFrame.
    columns:
        Subset of column names to include.  Defaults to all columns.
    parallel:
        If *True* use the joblib-parallel implementation.

    Returns
    -------
    pandas.DataFrame
        ``w × w`` correlation matrix with labelled rows and columns.
    """
    _require_pyspark()
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for xi_coef_matrix. "
            "Install it with:  pip install 'xicor[spark]'"
        ) from exc

    from xicor.core import xi_coef_matrix as _seq
    from xicor.core import xi_coef_matrix_parallel as _par

    if columns is None:
        columns = df.columns

    cols = list(columns)
    arr = df.select(cols).toPandas().to_numpy(dtype=float)
    matrix = _par(arr) if parallel else _seq(arr)
    return pd.DataFrame(matrix, index=cols, columns=cols)
