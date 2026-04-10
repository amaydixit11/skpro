import pandas as pd
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
