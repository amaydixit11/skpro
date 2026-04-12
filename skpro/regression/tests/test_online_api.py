import pytest
import pandas as pd
import numpy as np
from skpro.regression.base import BaseProbaRegressor, OnlineRegressorMixin

class MockOnlineRegressor(BaseProbaRegressor, OnlineRegressorMixin):
    _tags = {"capability:update": True}
    def _fit(self, X, y, C=None):
        self.val_ = 0
        return self
    def _update(self, X, y, C=None):
        self.val_ += 1
        return self
    def _predict_proba(self, X):
        from skpro.distributions.normal import Normal
        return Normal(mu=np.full(len(X), self.val_), sigma=1.0)

def test_update_predict_prequential():
    model = MockOnlineRegressor()
    X = pd.DataFrame([[1], [2]])
    y = pd.DataFrame([1, 2])
    model.fit(X, y) # val_ = 0

    # first call: predict (0) then update (val becomes 1)
    pred = model.update_predict(X, y)
    np.testing.assert_allclose(pred.mean().iloc[0], 0)
    assert model.val_ == 1
