# xicor

A Python implementation of the **XI (ξ) correlation coefficient** introduced by Sourav Chatterjee in ["A New Coefficient of Correlation"](https://arxiv.org/abs/1909.10140) (JASA, 2021).

Unlike Pearson correlation, XI detects **non-linear and non-monotonic relationships**, is **directional** (XI(x→y) ≠ XI(y→x)), and always falls in [0, 1] — where 0 means independence and 1 means a perfect functional relationship.

## Installation

Requires Python ≥ 3.10.

```bash
pip install xicor
```

For PySpark support:

```bash
pip install "xicor[spark]"
```

### From source

```bash
git clone https://github.com/giladamar/a-new-coefficient-of-correlation.git
cd a-new-coefficient-of-correlation
pip install .
```

## Quick start

```python
import numpy as np
from xicor import xi_coef, xi_coef_matrix, xi_coef_matrix_parallel

# Two 1-D arrays
x = np.linspace(0, 10 * np.pi, 2000)
y = np.sin(x)

xi_coef(x, y)   # ≈ 0.93  — x strongly determines y
xi_coef(y, x)   # ≈ 0.12  — y does not determine x (sin is not injective)
```

## API

### `xi_coef(xvec, yvec, simple=True, seed=42)`

Compute the XI coefficient between two 1-D arrays.

| Parameter | Type | Description |
|-----------|------|-------------|
| `xvec` | array-like | Predictor variable |
| `yvec` | array-like | Response variable |
| `simple` | bool | If `True` (default), return the scalar coefficient. If `False`, return `[xi, fr, CU]` for inspection. |
| `seed` | int | Random seed used for tie-breaking in ranks. |

```python
xi_coef(x, y)              # scalar: 0.93
xi_coef(x, y, simple=False)  # [0.93, fr_array, cu_value]
```

### `xi_coef_matrix(array)`

Compute XI for every ordered column pair in a 2-D array.

```python
data = np.random.randn(500, 5)
mat = xi_coef_matrix(data)   # shape (5, 5), diagonal = 1
```

The matrix is **not symmetric** — `mat[i, j]` is XI(column i → column j).

### `xi_coef_matrix_parallel(array)`

Same as `xi_coef_matrix` but uses all available CPU cores via `joblib`.

```python
mat = xi_coef_matrix_parallel(data)
```

Recommended for large matrices (many features or large n).

## PySpark API

```python
from xicor.spark import xi_coef as spark_xi_coef
from xicor.spark import xi_coef_matrix as spark_xi_matrix

# Single pair
result = spark_xi_coef(df, x_col="feature_a", y_col="feature_b")

# Full matrix (collects selected columns to driver, runs numpy code)
mat = spark_xi_matrix(df, columns=["a", "b", "c"])
```

## Interpreting results

| XI value | Interpretation |
|----------|----------------|
| ≈ 0 | x and y are independent |
| 0 < XI < 1 | Some functional relationship |
| ≈ 1 | y is nearly a function of x |

XI is **directional**: a high `xi_coef(x, y)` means x is a good predictor of y, but says nothing about y predicting x.

## Development

```bash
pip install "xicor[dev]"
pytest
```

## Citation

Chatterjee, S. (2021). A new coefficient of correlation. *Journal of the American Statistical Association*, 116(536), 2009–2022. https://doi.org/10.1080/01621459.2020.1758115

## License

GPL-3.0-or-later
