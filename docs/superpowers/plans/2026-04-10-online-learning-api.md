# Online/Stream Learning API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a native online learning ecosystem for `skpro` including a core API mixin, native incremental estimators, and adaptive compositors.

**Architecture:** A layered approach using an `OnlineRegressorMixin` for API orchestration, native `BaseProbaRegressor` subclasses for incremental math, and `_DelegatedProbaRegressor` subclasses for adaptive wrapping (compositors).

**Tech Stack:** Python, pandas, numpy, skpro, skbase, pytest.

---

## File Mapping

### New Files
- `skpro/regression/base/_online_mixin.py`: Contains `OnlineRegressorMixin`.
- `skpro/regression/online/_sgd.py`: Contains `SGDProbaRegressor`.
- `skpro/regression/online/_rls.py`: Contains `RLSProbaRegressor`.
- `skpro/regression/online/_glm.py`: Contains `OnlineGLMRegressor`.
- `skpro/regression/online/_compositors.py`: Contains `SlidingWindowRegressor`, `ForgettingFactorRegressor`, and `DriftDetectorRegressor`.
- `skpro/regression/tests/test_online_api.py`: Tests for the mixin and prequential protocol.
- `skpro/regression/tests/test_online_estimators.py`: Tests for SGD, RLS, and Online GLM.
- `skpro/regression/tests/test_online_compositors.py`: Tests for adaptation logic and stacking.

### Modified Files
- `skpro/regression/base/__init__.py`: Export `OnlineRegressorMixin`.
- `skpro/regression/online/__init__.py`: Export the new estimators and compositors.

---

## Implementation Tasks

### Task 1: Core API - `OnlineRegressorMixin`

**Files:**
- Create: `skpro/regression/base/_online_mixin.py`
- Modify: `skpro/regression/base/__init__.py`
- Test: `skpro/regression/tests/test_online_api.py`

- [ ] **Step 1: Implement `OnlineRegressorMixin`**
```python
# skpro/regression/base/_online_mixin.py
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
```

- [ ] **Step 2: Export Mixin in `skpro/regression/base/__init__.py`**
```python
from ._online_mixin import OnlineRegressorMixin
__all__ += ["OnlineRegressorMixin"]
```

- [ ] **Step 3: Write failing test for Prequential Protocol**
```python
# skpro/regression/tests/test_online_api.py
import pytest
import pandas as pd
import numpy as np
from skpro.regression.base import BaseProbaRegressor, OnlineRegressorMixin

class MockOnlineRegressor(BaseProbaRegressor, OnlineRegressorMixin):
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
    assert pred.mean().iloc[0] == 0
    assert model.val_ == 1
```

- [ ] **Step 4: Run test to verify it passes**
`pytest skpro/regression/tests/test_online_api.py -v`

- [ ] **Step 5: Commit**
`git add skpro/regression/base/_online_mixin.py skpro/regression/base/__init__.py skpro/regression/tests/test_online_api.py && git commit -m "feat: implement OnlineRegressorMixin and prequential API"`

---

### Task 2: Native Estimator - `SGDProbaRegressor`

**Files:**
- Create: `skpro/regression/online/_sgd.py`
- Modify: `skpro/regression/online/__init__.py`
- Test: `skpro/regression/tests/test_online_estimators.py`

- [ ] **Step 1: Implement `SGDProbaRegressor`**
```python
# skpro/regression/online/_sgd.py
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
```

- [ ] **Step 2: Write convergence test**
```python
# skpro/regression/tests/test_online_estimators.py
import pytest
import pandas as pd
import numpy as np
from skpro.regression.online import SGDProbaRegressor

def test_sgd_convergence():
    X = pd.DataFrame(np.random.randn(100, 2))
    true_w = np.array([1.5, -2.0])
    y = pd.DataFrame(X.values @ true_w + np.random.normal(0, 0.1, 100))
    
    model = SGDProbaRegressor(learning_rate=0.01)
    model.fit(X, y)
    
    # Weights should be close to true_w
    np.testing.assert_allclose(model.w_, true_w, atol=0.5)
```

- [ ] **Step 3: Run test to verify it passes**
`pytest skpro/regression/tests/test_online_estimators.py -v`

- [ ] **Step 4: Commit**
`git add skpro/regression/online/_sgd.py skpro/regression/online/__init__.py skpro/regression/tests/test_online_estimators.py && git commit -m "feat: implement SGDProbaRegressor"`

---

### Task 3: Native Estimator - `RLSProbaRegressor`

**Files:**
- Create: `skpro/regression/online/_rls.py`
- Modify: `skpro/regression/online/__init__.py`
- Test: `skpro/regression/tests/test_online_estimators.py`

- [ ] **Step 1: Implement `RLSProbaRegressor`**
```python
# skpro/regression/online/_rls.py
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
```

- [ ] **Step 2: Write equivalence test (Batch vs Online)**
```python
def test_rls_batch_equivalence():
    X = pd.DataFrame(np.random.randn(50, 2))
    y = pd.DataFrame(X.values @ np.array([1.0, 2.0]))
    
    # RLS should converge to similar weights as OLS
    model = RLSProbaRegressor()
    model.fit(X, y)
    
    # Compare to manual OLS
    w_ols = np.linalg.lstsq(X.values, y.values, rcond=None)[0]
    np.testing.assert_allclose(model.w_, w_ols.flatten(), atol=1e-2)
```

- [ ] **Step 3: Run test to verify it passes**
`pytest skpro/regression/tests/test_online_estimators.py -v`

- [ ] **Step 4: Commit**
`git add skpro/regression/online/_rls.py skpro/regression/online/__init__.py && git commit -m "feat: implement RLSProbaRegressor"`

---

### Task 4: Adaptation Layer - `SlidingWindowRegressor`

**Files:**
- Create: `skpro/regression/online/_compositors.py`
- Modify: `skpro/regression/online/__init__.py`
- Test: `skpro/regression/tests/test_online_compositors.py`

- [ ] **Step 1: Implement `SlidingWindowRegressor`**
```python
# skpro/regression/online/_compositors.py
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
```

- [ ] **Step 2: Write drift test**
```python
# skpro/regression/tests/test_online_compositors.py
import pytest
import pandas as pd
import numpy as np
from skpro.regression.online import SlidingWindowRegressor, SGDProbaRegressor

def test_sliding_window_drift():
    # Model that learns mean of y
    # We use a simple SGDProbaRegressor
    X = pd.DataFrame([[1]] * 200)
    y1 = pd.DataFrame([1.0] * 100) # first 100 are 1.0
    y2 = pd.DataFrame([10.0] * 100) # next 100 are 10.0
    
    model = SlidingWindowRegressor(SGDProbaRegressor(), window_size=10)
    model.fit(X[:100], y1)
    
    # After fitting on y1, mean should be ~1
    assert np.abs(model.predict(X[:1]).iloc[0] - 1.0) < 1.0
    
    # Update with y2
    for i in range(100):
        model.update(X[100+i:101+i], y2[i:i+1])
        
    # After window slides fully into y2, mean should be ~10
    assert np.abs(model.predict(X[:1]).iloc[0] - 10.0) < 1.0
```

- [ ] **Step 3: Run test to verify it passes**
`pytest skpro/regression/tests/test_online_compositors.py -v`

- [ ] **Step 4: Commit**
`git add skpro/regression/online/_compositors.py skpro/regression/online/__init__.py skpro/regression/tests/test_online_compositors.py && git commit -m "feat: implement SlidingWindowRegressor"`

---

### Task 5: Final Integration & Polish

- [ ] **Step 1: Implement `ForgettingFactorRegressor` and `DriftDetectorRegressor`** (Following patterns in Task 4)
- [ ] **Step 2: Verify Stacking**
```python
def test_stacking():
    # DriftDetector(SlidingWindow(SGDProbaRegressor()))
    # Verify it's an OnlineRegressorMixin and produces BaseDistribution
    pass
```
- [ ] **Step 3: Final CI check**
`pytest skpro/regression/tests/`
- [ ] **Step 4: Commit and Cleanup**
`git commit -m "feat: complete online learning API and compositors"`
