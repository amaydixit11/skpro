import pytest
import pandas as pd
import numpy as np
from skpro.regression.online import SGDProbaRegressor, RLSProbaRegressor

def test_sgd_convergence():
    X = pd.DataFrame(np.random.randn(1000, 2))
    true_w = np.array([1.5, -2.0])
    y = pd.DataFrame(X.values @ true_w + np.random.normal(0, 0.1, 1000))

    model = SGDProbaRegressor(learning_rate=0.01)
    model.fit(X, y)

    # Weights should be close to true_w
    np.testing.assert_allclose(model.w_, true_w, atol=0.5)

def test_rls_batch_equivalence():
    X = pd.DataFrame(np.random.randn(50, 2))
    y = pd.DataFrame(X.values @ np.array([1.0, 2.0]))

    # RLS should converge to similar weights as OLS
    model = RLSProbaRegressor()
    model.fit(X, y)

    # Compare to manual OLS
    w_ols = np.linalg.lstsq(X.values, y.values, rcond=None)[0]
    np.testing.assert_allclose(model.w_, w_ols.flatten(), atol=1e-2)
