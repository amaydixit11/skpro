import pandas as pd
from skbase.utils.dependencies import _check_estimator_deps

class OnlineRegressorMixin:
    """Mixin adding stream / update semantics to BaseProbaRegressor subclasses."""

    def update(self, X, y, C=None):
        """Incorporate new observations without full refit."""
        if not self.get_tag("capability:update"):
            return self

        # Route through the same validation path as fit()
        check_ret = self._check_X_y(X, y, C, return_metadata=True)
        X_inner = check_ret["X_inner"]
        y_inner = check_ret["y_inner"]
        if self.get_tag("capability:survival"):
            C_inner = check_ret["C_inner"]
            return self._update(X_inner, y_inner, C=C_inner)
        return self._update(X_inner, y_inner)

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
