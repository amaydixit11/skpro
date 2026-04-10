import pytest
import pandas as pd
import numpy as np
from skpro.regression.online import SGDProbaRegressor

def test_sgd_convergence():
    X = pd.DataFrame(np.random.randn(1000, 2))
    true_w = np.array([1.5, -2.0])
    y = pd.DataFrame(X.values @ true_w + np.random.normal(0, 0.1, 1000))

    model = SGDProbaRegressor(learning_rate=0.01)
    model.fit(X, y)

    # Weights should be close to true_w
    np.testing.assert_allclose(model.w_, true_w, atol=0.5)
