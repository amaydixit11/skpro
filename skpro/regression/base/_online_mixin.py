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
        """Consume an iterable stream in mini-batches.

        Parameters
        ----------
        X_stream : iterable of DataFrame
            Feature batches.
        y_stream : iterable of DataFrame
            Target batches.
        batch_size : int, default=1
            Number of (X, y) pairs to accumulate before calling fit/update.
        """
        first = True
        batch_X = []
        batch_y = []

        for X_item, y_item in zip(X_stream, y_stream):
            batch_X.append(X_item)
            batch_y.append(y_item)

            if len(batch_X) >= batch_size:
                X_batch = pd.concat(batch_X, ignore_index=True)
                y_batch = pd.concat(batch_y, ignore_index=True)
                if first:
                    self.fit(X_batch, y_batch)
                    first = False
                else:
                    self.update(X_batch, y_batch)
                batch_X = []
                batch_y = []

        # Process any remaining items
        if batch_X:
            X_batch = pd.concat(batch_X, ignore_index=True)
            y_batch = pd.concat(batch_y, ignore_index=True)
            if first:
                self.fit(X_batch, y_batch)
            else:
                self.update(X_batch, y_batch)

        return self

    def _update(self, X, y, C=None):
        """Backend hook to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _update")
