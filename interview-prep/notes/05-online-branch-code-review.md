---
tags: [skpro, online-learning, code-review, esoc-interview, own-code]
created: 2026-04-10
session: 5
---

# Code Review: Your `online` Branch

## Branch Status

```
Branch: online (local, not pushed as PR)
Commits: 6
- feat: implement SGDProbaRegressor
- feat: implement RLSProbaRegressor
- feat: implement SlidingWindowRegressor
- docs: add implementation plan for online learning API
- docs: add design spec for online learning API
- (base) sync with main
```

**Location:** `skpro/regression/online/` — the same directory as the existing refit strategies.

---

## SGDProbaRegressor — Full Review

### Code

```python
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

### Issues Found

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | 🔴 Critical | No `get_test_params()` — CI will fail | Add classmethod returning test params |
| 2 | 🔴 Critical | No input validation in `_fit` — doesn't call `self._check_X_y()` | Rely on public `fit()` which calls it |
| 3 | 🟡 Medium | `var_` initialized to 1.0 — arbitrary | Use `np.var(y.values)` from data |
| 4 | 🟡 Medium | `alpha` parameter name confusing | Rename to `var_smoothing` or `ema_alpha` |
| 5 | 🟡 Medium | Fixed learning rate — never converges | Add learning rate schedule support |
| 6 | 🟡 Medium | Python for-loop in `_update` — slow | Accept for online setting; document batch inefficiency |
| 7 | 🟢 Low | No docstrings | Add full docstrings with math |
| 8 | 🟢 Low | No `mixin_tags` tag | Consider adding for discoverability |

### fkiraly Will Ask

> **"Your SGD uses a Python for-loop — why not vectorize?"**

**Answer:** The loop is intentional for online semantics — each sample updates the model immediately. For batch updates, vectorization is possible but changes the update order (which affects convergence with fixed learning rate). I can add a vectorized batch mode as an option.

> **"How does this compare to batch OLS on stationary data?"**

**Answer:** With a decaying learning rate (η_t = 1/t), SGD converges to the OLS solution asymptotically. With fixed learning rate, it oscillates around the optimum. I should add learning rate schedule support.

> **"Why `alpha=0.1` for variance smoothing?"**

**Answer:** It's the exponential moving average smoothing factor. Higher alpha = more responsive to recent errors, lower = smoother variance estimate. This should be tunable via cross-validation. I'll rename it to avoid confusion with statistical alpha.

---

## RLSProbaRegressor — Full Review

### Code

```python
class RLSProbaRegressor(BaseProbaRegressor, OnlineRegressorMixin):
    _tags = {"capability:update": True}

    def __init__(self, forgetting_factor=1.0, noise_var=1.0):
        self.forgetting_factor = forgetting_factor
        self.noise_var = noise_var
        super().__init__()

    def _fit(self, X, y, C=None):
        n_features = X.shape[1]
        self.w_ = np.zeros(n_features)
        self.P_ = np.eye(n_features) * 10.0  # Initial covariance
        self._update(X, y, C)
        return self

    def _update(self, X, y, C=None):
        X_val = X.values if hasattr(X, 'values') else X
        y_val = y.values if hasattr(y, 'values') else y
        lam = self.forgetting_factor

        for xi, yi in zip(X_val, y_val):
            x = xi.reshape(-1, 1)
            denom = lam + (x.T @ self.P_ @ x)
            k = (self.P_ @ x) / denom
            self.w_ += (k * (yi - xi @ self.w_)).flatten()
            self.P_ = (self.P_ - k @ (x.T @ self.P_)) / lam
        return self

    def _predict_proba(self, X):
        X_val = X.values if hasattr(X, 'values') else X
        means = X_val @ self.w_
        var = np.diag(self.P_).mean() + self.noise_var
        return Normal(mu=means, sigma=np.sqrt(var))
```

### Issues Found

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | 🔴 Critical | No `get_test_params()` | Add classmethod |
| 2 | 🔴 Critical | **Per-row variance is wrong** — uses `np.diag(self.P_).mean()` instead of `x_i @ P @ x_i^T + noise_var` | Fix: compute per-row predictive variance |
| 3 | 🟡 Medium | `P_ = np.eye(n) * 10.0` is arbitrary | Make `initial_covariance` a parameter (default 1.0) |
| 4 | 🟡 Medium | No numerical stability guard | Add periodic re-regularization: `P += εI` every N steps |
| 5 | 🟡 Medium | `_fit` starts from zeros instead of batch OLS | Optionally initialize with `np.linalg.lstsq(X, y)` |
| 6 | 🟢 Low | Matrix operation order hard to read | Add comment explaining the formula |

### fkiraly Will Ask

> **"RLS gives the same variance for all predictions — is that correct?"**

**Answer:** No, that's a bug. The correct per-row predictive variance is `var_i = x_i @ P @ x_i^T + noise_var`. Currently I'm using the average diagonal of P, which loses the per-sample uncertainty information. I'll fix this before merging.

> **"What happens when P becomes ill-conditioned with forgetting_factor < 1?"**

**Answer:** With λ < 1, older data gets downweighted exponentially. The P matrix can grow large and eventually become indefinite due to floating-point accumulation. I need periodic re-regularization: `if step % 100 == 0: self.P_ += 1e-6 * np.eye(n)`. For very high dimensions, QR-based RLS or Kaczmarz methods would be more stable.

> **"Can you prove RLS with λ=1 gives the same result as batch OLS?"**

**Answer:** Yes, mathematically. For λ=1, RLS is algebraically equivalent to batch OLS. The test should verify: process n observations one at a time via `update()` → same weights as `fit()` on all n. This is a key test I'll add.

---

## SlidingWindowRegressor — Full Review

### Code

```python
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
        self._buffer_X = pd.concat([self._buffer_X, X], ignore_index=True).tail(self.window_size)
        self._buffer_y = pd.concat([self._buffer_y, y], ignore_index=True).tail(self.window_size)
        self.estimator_.fit(self._buffer_X, self._buffer_y)
        return self
```

### Issues Found

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | 🔴 Critical | No `get_test_params()` | Add classmethod |
| 2 | 🔴 Critical | No tag cloning from wrapped estimator | Add `self.clone_tags(estimator, ["capability:missing", "capability:survival"])` |
| 3 | 🟡 Medium | Dual inheritance: `_DelegatedProbaRegressor` + `OnlineRegressorMixin` — MRO confusion | Consider refactoring |
| 4 | 🟡 Medium | `tail()` creates new DataFrame each update — inefficient for large windows | Consider deque or circular buffer for production |
| 5 | 🟡 Medium | Missing C (censoring) handling | Add C buffer if wrapped estimator has `capability:survival` |
| 6 | 🟢 Low | No docstrings | Add full docstrings |

### fkiraly Will Ask

> **"Why does SlidingWindowRegressor inherit from both _DelegatedProbaRegressor and OnlineRegressorMixin?"**

**Answer:** It needs delegation for `_fit`, `_predict_proba`, etc. (from `_DelegatedProbaRegressor`), but also the mixin methods `update_predict` and `stream_fit` (from `OnlineRegressorMixin`). The MRO puts `OnlineRegressorMixin` second, so its methods are available but `_update` is overridden by my class. I may refactor to avoid dual inheritance — perhaps just inherit from `_DelegatedProbaRegressor` and implement `update_predict`/`stream_fit` directly.

---

## OnlineRegressorMixin — Full Review

### Code

```python
class OnlineRegressorMixin:
    def update(self, X, y, C=None):
        if not self.get_tag("capability:update"):
            return self
        self._update(X, y, C)
        return self

    def update_predict(self, X, y, C=None):
        pred = self.predict_proba(X)
        self.update(X, y, C)
        return pred

    def stream_fit(self, X_stream, y_stream, batch_size=1):
        first = True
        for X_batch, y_batch in zip(X_stream, y_stream):
            if first:
                self.fit(X_batch, y_batch)
                first = False
            else:
                self.update(X_batch, y_batch)
        return self
```

### Issues Found

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | 🔴 Critical | No input validation — skips `_check_X_y()` | Add validation call |
| 2 | 🟡 Medium | `update_predict` doesn't check if fitted | Add `check_is_fitted()` guard |
| 3 | 🟡 Medium | `stream_fit` doesn't handle empty stream | Add guard or raise informative error |
| 4 | 🟡 Medium | `batch_size` parameter is unused | Either use it or remove it |
| 5 | 🟢 Low | No docstrings | Add full docstrings |

---

## How to Frame This in the Interview

### Option A: Honest Approach (Recommended)

> "I started prototyping during the application period to validate the architecture. The ESoC period will be about **productionizing**: adding tests (`get_test_params`, check_estimator compliance), docs, tag handling, edge cases, missing compositors (ForgettingFactor, DriftDetector), prequential evaluation, benchmarking, and the River bridge adapter."

### Option B: Proposal-Scope Approach

> "The branch has skeleton implementations to prove the concept. The ESoC work is the **full ecosystem**: prequential evaluation framework, drift detection, benchmarking suite, comprehensive tests, API reference updates, tutorial notebooks, and integration with sktime's forecasting pipelines."

**My recommendation:** Go with Option A. fkiraly will respect that you've already started building and can articulate the gap between prototype and production code.

---

## Related

- [[04-online-learning-gap]]
- [[01-skpro-class-hierarchy]]
- [[09-pr-718-glm-dispersion-fix]]
- [[10-pr-721-kernelmixture]]
