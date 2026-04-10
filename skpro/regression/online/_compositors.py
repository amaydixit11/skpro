import pandas as pd
import numpy as np
from sklearn.base import clone
from skpro.regression.base import _DelegatedProbaRegressor, OnlineRegressorMixin


class SlidingWindowRegressor(_DelegatedProbaRegressor, OnlineRegressorMixin):
    """Compositor that refits the wrapped estimator on a rolling window of recent data.

    Maintains a buffer of the most recent ``window_size`` observations.
    On each ``update``, the buffer is updated and the wrapped estimator
    is refit on the buffered data.

    Parameters
    ----------
    estimator : BaseProbaRegressor
        The regressor to wrap and refit.
    window_size : int, default=100
        Number of recent observations to retain.
    """

    _tags = {"capability:update": True}

    def __init__(self, estimator, window_size=100):
        self.estimator = estimator
        self.window_size = window_size
        super().__init__()
        self.estimator_ = clone(self.estimator)
        self._buffer_X = None
        self._buffer_y = None
        self._buffer_C = None

    def _fit(self, X, y, C=None):
        # Trim initial fit to window_size — ensures sliding window behavior
        # is consistent from the start
        self._buffer_X = X.copy().tail(self.window_size)
        self._buffer_y = y.copy().tail(self.window_size)
        self._buffer_C = C.copy().tail(self.window_size) if C is not None else None
        if C is not None and self.estimator_.get_tag("capability:survival"):
            self.estimator_.fit(self._buffer_X, self._buffer_y, C=self._buffer_C)
        else:
            self.estimator_.fit(self._buffer_X, self._buffer_y)
        return self

    def _update(self, X, y, C=None):
        self._buffer_X = pd.concat([self._buffer_X, X], ignore_index=True).tail(
            self.window_size
        )
        self._buffer_y = pd.concat([self._buffer_y, y], ignore_index=True).tail(
            self.window_size
        )
        if C is not None:
            self._buffer_C = (
                pd.concat([self._buffer_C, C], ignore_index=True).tail(self.window_size)
                if self._buffer_C is not None
                else C.tail(self.window_size)
            )

        if (
            C is not None
            and self._buffer_C is not None
            and self.estimator_.get_tag("capability:survival")
        ):
            self.estimator_.fit(self._buffer_X, self._buffer_y, C=self._buffer_C)
        else:
            self.estimator_.fit(self._buffer_X, self._buffer_y)
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sklearn.linear_model import Ridge

        from skpro.regression.residual import ResidualDouble

        regressor = ResidualDouble(Ridge())
        return [
            {"estimator": regressor, "window_size": 20},
            {"estimator": regressor, "window_size": 10},
        ]


class ForgettingFactorRegressor(_DelegatedProbaRegressor, OnlineRegressorMixin):
    """Compositor that applies exponential forgetting to observation weights.

    Maintains all data seen so far but assigns exponentially decaying weights
    to older observations. On each ``update``, old weights are multiplied
    by ``forgetting_factor`` and new observations receive weight 1.0.
    The wrapped estimator is refit with ``sample_weight`` if supported.

    Parameters
    ----------
    estimator : BaseProbaRegressor
        The regressor to wrap. Must support ``sample_weight`` in ``fit``
        for the forgetting mechanism to be effective.
    forgetting_factor : float in (0, 1], default=0.99
        Decay factor applied to existing weights on each update.
        Closer to 1 means slower forgetting.
    """

    _tags = {"capability:update": True}

    def __init__(self, estimator, forgetting_factor=0.99):
        self.estimator = estimator
        self.forgetting_factor = forgetting_factor
        super().__init__()
        self.estimator_ = clone(self.estimator)
        self._weights = None

    def _fit(self, X, y, C=None):
        self._weights = np.ones(len(X))
        self._buffer_X = X.copy()
        self._buffer_y = y.copy()
        self._buffer_C = C.copy() if C is not None else None
        if self._supports_sample_weight():
            self.estimator_.fit(X, y, sample_weight=self._weights)
        elif C is not None and self.estimator_.get_tag("capability:survival"):
            self.estimator_.fit(X, y, C=C)
        else:
            self.estimator_.fit(X, y)
        return self

    def _update(self, X, y, C=None):
        # Decay existing weights
        if self._weights is not None:
            self._weights *= self.forgetting_factor

        # Store new data with weight 1
        self._buffer_X = pd.concat([self._buffer_X, X], ignore_index=True)
        self._buffer_y = pd.concat([self._buffer_y, y], ignore_index=True)
        new_weights = np.ones(len(X))
        self._weights = np.concatenate([self._weights, new_weights])

        if C is not None:
            self._buffer_C = (
                pd.concat([self._buffer_C, C], ignore_index=True)
                if self._buffer_C is not None
                else C
            )

        if self._supports_sample_weight():
            if (
                C is not None
                and self._buffer_C is not None
                and self.estimator_.get_tag("capability:survival")
            ):
                self.estimator_.fit(
                    self._buffer_X,
                    self._buffer_y,
                    C=self._buffer_C,
                    sample_weight=self._weights,
                )
            else:
                self.estimator_.fit(
                    self._buffer_X, self._buffer_y, sample_weight=self._weights
                )
        elif C is not None and self.estimator_.get_tag("capability:survival"):
            self.estimator_.fit(self._buffer_X, self._buffer_y, C=self._buffer_C)
        else:
            self.estimator_.fit(self._buffer_X, self._buffer_y)

        return self

    def _supports_sample_weight(self):
        """Check if the wrapped estimator supports sample_weight in fit."""
        fit_code = getattr(self.estimator_.fit, "__code__", None)
        return fit_code is not None and "sample_weight" in fit_code.co_varnames

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sklearn.linear_model import Ridge

        from skpro.regression.residual import ResidualDouble

        regressor = ResidualDouble(Ridge())
        return [
            {"estimator": regressor, "forgetting_factor": 0.99},
            {"estimator": regressor, "forgetting_factor": 0.9},
        ]


class DriftDetectorRegressor(_DelegatedProbaRegressor, OnlineRegressorMixin):
    """Compositor that monitors prediction error and resets on drift.

    Maintains a rolling buffer of recent prediction errors. When the
    average error over the buffer exceeds ``threshold``, the wrapped
    estimator is reset and refit on a recent window of data.

    Parameters
    ----------
    estimator : BaseProbaRegressor
        The regressor to wrap and monitor.
    threshold : float, default=0.1
        Error threshold above which drift is triggered.
        Should be set relative to the scale of the target variable.
    window_size : int, default=50
        Size of the rolling error buffer and data window for refit.
    """

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
        self._buffer_C = None

    def _fit(self, X, y, C=None):
        self._buffer_X = X.copy()
        self._buffer_y = y.copy()
        self._buffer_C = C.copy() if C is not None else None
        if C is not None and self.estimator_.get_tag("capability:survival"):
            self.estimator_.fit(X, y, C=C)
        else:
            self.estimator_.fit(X, y)
        return self

    def _update(self, X, y, C=None):
        # Update data buffer (capped at window_size)
        self._buffer_X = pd.concat([self._buffer_X, X], ignore_index=True).tail(
            self.window_size
        )
        self._buffer_y = pd.concat([self._buffer_y, y], ignore_index=True).tail(
            self.window_size
        )
        if C is not None:
            self._buffer_C = (
                pd.concat([self._buffer_C, C], ignore_index=True).tail(self.window_size)
                if self._buffer_C is not None
                else C.tail(self.window_size)
            )

        # Monitor prediction error
        preds = self.predict(X)
        errors = np.abs(y.values.flatten() - preds.values.flatten())
        self._error_buffer.extend(errors.tolist())

        # Keep error buffer at window_size
        if len(self._error_buffer) > self.window_size:
            self._error_buffer = self._error_buffer[-self.window_size :]

        # Check for drift and act accordingly
        if len(self._error_buffer) == self.window_size:
            avg_error = np.mean(self._error_buffer)
            if avg_error > self.threshold:
                # Drift detected: reset and refit on recent window
                self.estimator_ = clone(self.estimator)
                if (
                    self._buffer_C is not None
                    and self.estimator_.get_tag("capability:survival")
                ):
                    self.estimator_.fit(
                        self._buffer_X, self._buffer_y, C=self._buffer_C
                    )
                else:
                    self.estimator_.fit(self._buffer_X, self._buffer_y)
                self._error_buffer = []
            else:
                # No drift: still update the estimator with new data
                if self.estimator_.get_tag("capability:update"):
                    self.estimator_.update(X, y)
                else:
                    if (
                        self._buffer_C is not None
                        and self.estimator_.get_tag("capability:survival")
                    ):
                        self.estimator_.fit(
                            self._buffer_X, self._buffer_y, C=self._buffer_C
                        )
                    else:
                        self.estimator_.fit(self._buffer_X, self._buffer_y)
        else:
            # Buffer not full yet: normal update
            if self.estimator_.get_tag("capability:update"):
                self.estimator_.update(X, y)
            else:
                if (
                    self._buffer_C is not None
                    and self.estimator_.get_tag("capability:survival")
                ):
                    self.estimator_.fit(
                        self._buffer_X, self._buffer_y, C=self._buffer_C
                    )
                else:
                    self.estimator_.fit(self._buffer_X, self._buffer_y)

        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sklearn.linear_model import Ridge

        from skpro.regression.residual import ResidualDouble

        regressor = ResidualDouble(Ridge())
        return [
            {"estimator": regressor, "threshold": 1.0, "window_size": 20},
            {"estimator": regressor, "threshold": 0.5, "window_size": 10},
        ]
