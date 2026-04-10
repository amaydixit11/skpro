import numpy as np
import pandas as pd
from skpro.regression.base import BaseProbaRegressor
from skpro.regression.base import OnlineRegressorMixin
from skpro.distributions.normal import Normal

class SGDProbaRegressor(BaseProbaRegressor, OnlineRegressorMixin):
    _tags = {"capability:update": True}

    def __init__(self, learning_rate=0.01, alpha=0.1):
        self.learning_rate = learning_rate
        self.alpha = alpha
        super().__init__()

    def _fit(self, X, y, C=None):
        self.w_ = np.zeros(X.shape[1])
        self.var_ = 1.0
        self._update(X, y, C)
        return self

    def _update(self, X, y, C=None):
        X_val = X.values if hasattr(X, 'values') else X
        y_val = y.values if hasattr(y, 'values') else y
        for xi, yi in zip(X_val, y_val):
            pred = xi @ self.w_
            err = yi - pred
            self.w_ += self.learning_rate * err * xi
            self.var_ = (1 - self.alpha) * self.var_ + self.alpha * (err**2)
        return self

    def _predict_proba(self, X):
        X_val = X.values if hasattr(X, 'values') else X
        means = X_val @ self.w_
        return Normal(mu=means, sigma=np.sqrt(self.var_))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
        """
        return [
            {"learning_rate": 0.01},
            {"learning_rate": 0.001, "alpha": 0.05},
        ]
