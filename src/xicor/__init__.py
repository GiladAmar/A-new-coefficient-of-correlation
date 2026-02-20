"""xicor — XI correlation coefficient.

Reference: Sourav Chatterjee, "A New Coefficient of Correlation",
Journal of the American Statistical Association, 2021.
https://arxiv.org/abs/1909.10140

Basic usage
-----------
>>> import numpy as np
>>> from xicor import xi_coef, xi_coef_matrix
>>>
>>> x = np.linspace(0, 10, 500)
>>> y = np.sin(x)
>>> xi_coef(x, y)   # strong x → y relationship
0.93...
>>> xi_coef(y, x)   # weak y → x (not injective)
0.12...

PySpark usage
-------------
>>> from xicor.spark import xi_coef as spark_xi_coef
>>> from xicor.spark import xi_coef_matrix as spark_xi_coef_matrix
"""

from xicor.core import xi_coef, xi_coef_matrix, xi_coef_matrix_parallel

# Backwards-compatible uppercase aliases matching the original notebook API.
XI_coef = xi_coef
XI_coef_matrix = xi_coef_matrix
XI_coef_matrix_parallel = xi_coef_matrix_parallel

__version__ = "0.1.0"
__all__ = [
    "xi_coef",
    "xi_coef_matrix",
    "xi_coef_matrix_parallel",
    # legacy aliases
    "XI_coef",
    "XI_coef_matrix",
    "XI_coef_matrix_parallel",
]
