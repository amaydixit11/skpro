import numpy as np
import pandas as pd
from skpro.regression.base import BaseProbaRegressor
from skpro.regression.base import OnlineRegressorMixin
from skpro.distributions.normal import Normal

class RLSProbaRegressor(BaseProbaRegressor, OnlineRegressorMixin):
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
        X_val = X.values if hasattr(X, 'values') else X
        y_val = y.values if hasattr(y, 'values') else y
        lam = self.forgetting_factor

        for xi, yi in zip(X_val, y_val):
            x = xi.reshape(-1, 1)
            # Gain vector k = Px / (lam + xPx)
            denom = lam + (x.T @ self.P_ @ x)
            k = (self.P_ @ x) / denom
            # Update weights
            self.w_ += (k * (yi - xi @ self.w_)).flatten()
            # Update P matrix: P = (P - k x P) / lam
            self.P_ = (self.P_ - k @ (x.T @ self.P_)) / lam
        return self

    def _predict_proba(self, X):
        X_val = X.values if hasattr(X, 'values') else X
        means = X_val @ self.w_
        # Variance = x P x^T + noise_var
        # For efficiency, we can use a constant or compute per-row
        # Simple version: use average P diag + noise
        var = np.diag(self.P_).mean() + self.noise_var
        return Normal(mu=means, sigma=np.sqrt(var))
