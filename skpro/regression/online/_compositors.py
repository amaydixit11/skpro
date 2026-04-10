import pandas as pd
import numpy as np
from sklearn.base import clone
from skpro.regression.base import _DelegatedProbaRegressor, OnlineRegressorMixin

class SlidingWindowRegressor(_DelegatedProbaRegressor, OnlineRegressorMixin):
    _tags = {"capability:update": True}

    def __init__(self, estimator, window_size=100):
        self.estimator = estimator
        self.window_size = window_size
        super().__init__()
        self.estimator_ = clone(self.estimator)
        self._buffer_X = None
        self._buffer_y = None

    def _fit(self, X, y, C=None):
        self._buffer_X = X.copy()
        self._buffer_y = y.copy()
        self.estimator_.fit(X, y)
        return self

    def _update(self, X, y, C=None):
        # Update buffer
        self._buffer_X = pd.concat([self._buffer_X, X], ignore_index=True).tail(self.window_size)
        self._buffer_y = pd.concat([self._buffer_y, y], ignore_index=True).tail(self.window_size)
        # Refit estimator on the current window
        self.estimator_.fit(self._buffer_X, self._buffer_y)
        return self

class ForgettingFactorRegressor(_DelegatedProbaRegressor, OnlineRegressorMixin):
    _tags = {"capability:update": True}

    def __init__(self, estimator, forgetting_factor=0.99):
        self.estimator = estimator
        self.forgetting_factor = forgetting_factor
        super().__init__()
        self.estimator_ = clone(self.estimator)
        self._weights = None

    def _fit(self, X, y, C=None):
        # Initialize weights to 1
        self._weights = np.ones(len(X))
        # Use getattr to safely check for sample_weight support without triggering ValueError from get_tag
        # Check if 'sample_weight' is in the fit method's arguments
        fit_code = getattr(self.estimator_.fit, "__code__", None)
        if fit_code and "sample_weight" in fit_code.co_varnames:
            self.estimator_.fit(X, y, sample_weight=self._weights)
        else:
            self.estimator_.fit(X, y)
        return self

    def _update(self, X, y, C=None):
        # Decay existing weights
        if self._weights is not None:
            self._weights *= self.forgetting_factor

        # New observations have weight 1
        new_weights = np.ones(len(X))

        # Combine old and new data
        # Note: ForgettingFactorRegressor typically maintains all data but decays weights.
        # To avoid memory leak, we might need to prune very small weights, but for now we follow the plan.
        # However, _DelegatedProbaRegressor's fit/update typically expects to call fit on the underlying estimator.

        # We need to store the data to apply weights during fit.
        if not hasattr(self, "_buffer_X"):
            self._buffer_X = pd.DataFrame()
            self._buffer_y = pd.DataFrame()

        self._buffer_X = pd.concat([self._buffer_X, X], ignore_index=True)
        self._buffer_y = pd.concat([self._buffer_y, y], ignore_index=True)

        # Update weight vector
        if self._weights is not None:
            self._weights = np.concatenate([self._weights, new_weights])
        else:
            self._weights = new_weights

        # Apply forgetting factor to ALL weights except the most recent ones
        # (Actually, the decay should happen at every update step)
        # The weights were already decayed at the start of _update, but that was only for old weights.
        # We need to make sure the new weights are 1.0 and old ones are decayed.
        # The current logic:
        # 1. old_weights *= factor
        # 2. total_weights = [old_weights, ones(len(X))]
        # This is correct.

        if self.estimator_.get_tag("capability:sample_weight") or "sample_weight" in self.estimator_.fit.__code__.co_varnames:
            self.estimator_.fit(self._buffer_X, self._buffer_y, sample_weight=self._weights)
        else:
            # Fallback: just fit without weights if not supported
            self.estimator_.fit(self._buffer_X, self._buffer_y)

        return self

class DriftDetectorRegressor(_DelegatedProbaRegressor, OnlineRegressorMixin):
    _tags = {"capability:update": True}

    def __init__(self, estimator, threshold=0.1, window_size=50):
        self.estimator = estimator
        self.threshold = threshold
        self.window_size = window_size
        super().__init__()
        self.estimator_ = clone(self.estimator)
        self._error_buffer = []
        self._buffer_X = pd.DataFrame()
        self._buffer_y = pd.DataFrame()

    def _fit(self, X, y, C=None):
        self._buffer_X = X.copy()
        self._buffer_y = y.copy()
        self.estimator_.fit(X, y)
        return self

    def _update(self, X, y, C=None):
        # Update data buffer
        self._buffer_X = pd.concat([self._buffer_X, X], ignore_index=True).tail(self.window_size)
        self._buffer_y = pd.concat([self._buffer_y, y], ignore_index=True).tail(self.window_size)

        # Monitor prediction error
        preds = self.predict(X)
        errors = np.abs(y.values.flatten() - preds.values.flatten())
        self._error_buffer.extend(errors.tolist())

        # Keep error buffer at window_size
        if len(self._error_buffer) > self.window_size:
            self._error_buffer = self._error_buffer[-self.window_size:]

        # Trigger reset if average error exceeds threshold
        if len(self._error_buffer) == self.window_size:
            avg_error = np.mean(self._error_buffer)
            if avg_error > self.threshold:
                # Drift detected! Reset and refit on the last window of data
                self.estimator_ = clone(self.estimator)
                self.estimator_.fit(self._buffer_X, self._buffer_y)
                self._error_buffer = [] # Reset error buffer after drift detection

        # Regular update: use update if available, otherwise fit (though fit on small batch is bad)
        if self.estimator_.get_tag("capability:update"):
            self.estimator_.update(X, y)
        else:
            # If it's a batch estimator, we refit on the window
            self.estimator_.fit(self._buffer_X, self._buffer_y)

        return self
