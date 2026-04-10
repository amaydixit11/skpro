---
tags: [skpro, online-learning, gap-analysis, esoc-interview]
created: 2026-04-10
session: 4
---

# The Online Learning Gap in skpro

## What Exists on `main` Branch (Production)

```
skpro/regression/online/
├── __init__.py            → exports: OnlineRefit, OnlineRefitEveryN, OnlineDontRefit
├── _refit.py              → OnlineRefit
├── _refit_every.py        → OnlineRefitEveryN
└── _dont_refit.py         → OnlineDontRefit
```

Plus:
- `skpro/regression/ondil.py` → `OndilOnlineGamlss` (external wrapper, soft dep: ondil)
- `skpro/regression/base/_online_mixin.py` → `OnlineRegressorMixin` (interface only, no implementations)

---

## The Three Meta-Estimators

### `OnlineRefit` — "Batch in Online Clothing"

```python
class OnlineRefit(_DelegatedProbaRegressor):
    _tags = {"capability:update": True}

    def _update(self, X, y, C=None):
        # 1. Accumulate ALL data seen so far
        X_pool = pd.concat([self._X, X], ignore_index=True)
        y_pool = pd.concat([self._y, y], ignore_index=True)
        # 2. Refit from SCRATCH on full accumulated data
        estimator = self.estimator.clone()
        estimator.fit(X=X_pool, y=y_pool, C=C_pool)
        self.estimator_ = estimator
        # 3. Remember data
        self._X, self._y, self._C = X_pool, y_pool, C_pool
```

**Problem:** O(N²) memory and time. Each update requires storing all data and refitting from scratch. This is NOT true online learning.

### `OnlineRefitEveryN` — Buffered Refit

```python
class OnlineRefitEveryN(_DelegatedProbaRegressor):
    _tags = {"capability:update": True}

    def _update(self, X, y, C=None):
        self.n_seen_since_last_update_ += len(X)
        if self.n_seen_since_last_update_ >= self.N:
            # Buffer + refit
            self.estimator_.update(X_pool, y_pool, C=C_pool)
            self.n_seen_since_last_update_ = 0
```

**Problem:** Still just batch refitting, just less frequently.

### `OnlineDontRefit` — No-op

```python
class OnlineDontRefit(_DelegatedProbaRegressor):
    _tags = {"capability:update": False}

    def _update(self, X, y, C=None):
        return self  # does nothing
```

**Purpose:** Baseline for comparison when the wrapped estimator is already online.

### `OndilOnlineGamlss` — The Only True Online Estimator

```python
class OndilOnlineGamlss(BaseProbaRegressor):
    # Soft dependency: ondil library
    def _update(self, X, y, C=None):
        # Calls ondil's OnlineGamlss.update() or .partial_fit()
```

**Problem:** Only supports GAMLSS models. One specific model family.

---

## What's Missing

| What's Missing | Why It Matters |
|---|---|
| **Native incremental estimators** | No SGD, RLS, or warm-starting GLM |
| **Prequential evaluation** | No `update_predict()` that does predict-then-update |
| **Stream fitting** | No `stream_fit()` that consumes iterables |
| **Stream compositors** | No sliding window, forgetting factor, or drift detection wrappers |
| **Benchmarking** | No prequential CRPS/log-score time series |
| **River bridge** | No adapter for river library's online estimators |

---

## Why Existing Tools Don't Fill the Gap

| Tool | Why It Doesn't Work |
|------|-------------------|
| **River** | Not sklearn-compatible, returns point predictions (not BaseDistribution), can't compose with skpro metrics |
| **sklearn `partial_fit`** | Only for point predictions, no probabilistic output, not in skpro hierarchy |
| **statsmodels recursive LS** | Not sklearn-compatible, no skpro distribution output contract |
| **skpro GLMRegressor** | Full-batch only, must refit from scratch (PR #718 fixed variance but not online capability) |

---

## The Proposal's Four Pillars

| Pillar | What It Adds |
|--------|-------------|
| 1. OnlineRegressorMixin | `update()`, `update_predict()`, `stream_fit()` interface |
| 2. Concrete Estimators | SGDProbaRegressor, RLSProbaRegressor, OnlineGLMRegressor |
| 3. Stream Compositors | SlidingWindowRegressor, ForgettingFactorRegressor, DriftDetectorRegressor |
| 4. Prequential Eval | `prequential_evaluate()`, online metrics, benchmark suite |

---

## Current State (on `online` branch)

Already implemented:
- ✅ `OnlineRegressorMixin` (with `update`, `update_predict`, `stream_fit`)
- ✅ `SGDProbaRegressor`
- ✅ `RLSProbaRegressor`
- ✅ `SlidingWindowRegressor`

Still missing:
- ❌ `ForgettingFactorRegressor`
- ❌ `DriftDetectorRegressor`
- ❌ `OnlineGLMRegressor`
- ❌ `prequential_evaluate()`
- ❌ `get_test_params()` on all classes
- ❌ Tag cloning on compositors
- ❌ Input validation in mixin
- ❌ Tests, docs, tutorials

---

## Related

- [[05-online-branch-code-review]]
- [[03-base-proba-regressor-deep-dive]]
- [[01-skpro-class-hierarchy]]
