import pandas as pd
from skbase.utils.dependencies import _check_estimator_deps

class OnlineRegressorMixin:
    """Mixin adding stream / update semantics to BaseProbaRegressor subclasses."""

    def update(self, X, y, C=None):
        """Incorporate new observations without full refit."""
        if not self.get_tag("capability:update"):
            return self

        # Basic validation (simplified, assuming BaseProbaRegressor context)
        # In real impl, use self._check_X_y
        self._update(X, y, C)
        return self

    def update_predict(self, X, y, C=None):
        """Prequential predict-then-update."""
        pred = self.predict_proba(X)
        self.update(X, y, C)
        return pred

    def stream_fit(self, X_stream, y_stream, batch_size=1):
        """Consume an iterable stream in mini-batches."""
        first = True
        for X_batch, y_batch in zip(X_stream, y_stream):
            if first:
                self.fit(X_batch, y_batch)
                first = False
            else:
                self.update(X_batch, y_batch)
        return self

    def _update(self, X, y, C=None):
        """Backend hook to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _update")
