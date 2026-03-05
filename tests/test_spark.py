"""Tests for xicor.spark.

Skipped automatically when PySpark is not installed.
"""

import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark", reason="pyspark not installed")

from pyspark.sql import SparkSession  # noqa: E402

from xicor.spark import xi_coef as spark_xi_coef  # noqa: E402
from xicor.spark import xi_coef_matrix as spark_xi_coef_matrix  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[1]")
        .appName("xicor-tests")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


# ---------------------------------------------------------------------------
# xi_coef
# ---------------------------------------------------------------------------


def test_spark_xi_coef_self_correlation(spark):
    data = [(float(i), float(i)) for i in range(200)]
    df = spark.createDataFrame(data, ["x", "y"])
    result = spark_xi_coef(df, "x", "y")
    assert result == pytest.approx(1.0, abs=1e-6)


def test_spark_xi_coef_returns_float_in_unit_interval(spark):
    rng = np.random.default_rng(0)
    data = [(float(x), float(y)) for x, y in zip(rng.random(100), rng.random(100))]
    df = spark.createDataFrame(data, ["x", "y"])
    result = spark_xi_coef(df, "x", "y")
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_spark_xi_coef_nonlinear(spark):
    """Spark implementation should detect x→sin(x) strongly."""
    x_vals = np.linspace(0, 10 * np.pi, 500)
    y_vals = np.sin(x_vals)
    data = [(float(x), float(y)) for x, y in zip(x_vals, y_vals)]
    df = spark.createDataFrame(data, ["x", "y"])
    assert spark_xi_coef(df, "x", "y") > 0.7
    assert spark_xi_coef(df, "y", "x") < 0.3


def test_spark_xi_coef_empty_dataframe(spark):
    df = spark.createDataFrame([], "x DOUBLE, y DOUBLE")
    result = spark_xi_coef(df, "x", "y")
    assert result == 0.0


# ---------------------------------------------------------------------------
# xi_coef_matrix
# ---------------------------------------------------------------------------


def test_spark_xi_coef_matrix_shape(spark):
    rng = np.random.default_rng(1)
    data = [tuple(float(v) for v in row) for row in rng.random((100, 3))]
    df = spark.createDataFrame(data, ["a", "b", "c"])
    result = spark_xi_coef_matrix(df, columns=["a", "b", "c"])
    assert result.shape == (3, 3)


def test_spark_xi_coef_matrix_labels(spark):
    rng = np.random.default_rng(2)
    data = [tuple(float(v) for v in row) for row in rng.random((50, 3))]
    df = spark.createDataFrame(data, ["p", "q", "r"])
    result = spark_xi_coef_matrix(df)
    assert list(result.columns) == ["p", "q", "r"]
    assert list(result.index) == ["p", "q", "r"]


def test_spark_xi_coef_matrix_diagonal(spark):
    rng = np.random.default_rng(3)
    data = [tuple(float(v) for v in row) for row in rng.random((100, 3))]
    df = spark.createDataFrame(data, ["a", "b", "c"])
    result = spark_xi_coef_matrix(df)
    np.testing.assert_array_almost_equal(np.diag(result.values), np.ones(3))


def test_spark_xi_coef_matrix_column_subset(spark):
    rng = np.random.default_rng(4)
    data = [tuple(float(v) for v in row) for row in rng.random((80, 4))]
    df = spark.createDataFrame(data, ["a", "b", "c", "d"])
    result = spark_xi_coef_matrix(df, columns=["a", "c"])
    assert result.shape == (2, 2)
    assert list(result.columns) == ["a", "c"]
