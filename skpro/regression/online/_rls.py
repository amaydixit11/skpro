import numpy as np
import pandas as pd
from skpro.regression.base import BaseProbaRegressor
from skpro.regression.base import OnlineRegressorMixin
from skpro.distributions.normal import Normal

class RLSProbaRegressor(OnlineRegressorMixin, BaseProbaRegressor):
    _tags = {"capability:update": True}

    def __init__(self, forgetting_factor=1.0, noise_var=1.0):
        self.forgetting_factor = forgetting_factor
        self.noise_var = noise_var
        super().__init__()

    def _fit(self, X, y, C=None):
        n_features = X.shape[1]
        self.w_ = np.zeros(n_features)
        self.P_ = np.eye(n_features) * 10.0 # Initial covariance
        self._update(X, y, C)
        return self

    def _update(self, X, y, C=None):
        # Input already validated/normalized by update() via _check_X_y
        X_val = X.values if hasattr(X, "values") else np.asarray(X)
        y_val = y.values if hasattr(y, "values") else np.asarray(y)
        # Ensure 1D
        if y_val.ndim > 1:
            y_val = y_val.ravel()
        lam = self.forgetting_factor

        for xi, yi in zip(X_val, y_val):
            x = xi.reshape(-1, 1)
            # Gain vector k = Px / (lam + xPx)
            denom = lam + (x.T @ self.P_ @ x)
            k = (self.P_ @ x) / denom
            # Update weights
            self.w_ += (k * (yi - xi @ self.w_)).flatten()
            # Update P matrix: P = (P - k x^T P) / lam
            self.P_ = (self.P_ - k @ (x.T @ self.P_)) / lam
        return self

    def _predict_proba(self, X):
        X_val = X.values if hasattr(X, 'values') else X
        means = X_val @ self.w_
        # Per-row predictive variance: x^T P x + noise_var
        # This captures both parameter uncertainty and observation noise
        var_per_row = np.sum(X_val @ self.P_ * X_val, axis=1) + self.noise_var
        return Normal(mu=means, sigma=np.sqrt(var_per_row))

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
            {"forgetting_factor": 1.0, "noise_var": 1.0},
            {"forgetting_factor": 0.95, "noise_var": 0.1},
        ]
