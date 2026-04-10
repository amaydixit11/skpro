---
tags: [skpro, architecture, class-hierarchy, esoc-interview]
created: 2026-04-10
session: 1
---

# skpro Class Hierarchy & Architecture

## The Three Pillars

Every skpro object follows one inheritance chain:

```
skbase.BaseObject          ← Tags, get_params(), clone(), __repr__
    │
    ├── BaseEstimator      ← Fitted state tracking, _check_X_y()
    │       │
    │       └── BaseProbaRegressor  ← fit, predict_proba, update
    │
    ├── BaseDistribution   ← pdf, cdf, ppf, sample, mean, var
    │
    └── BaseProbaMetric    ← evaluate, _evaluate_by_index
```

**Why this structure?** Each pillar has a different contract:
- **Regressors** *produce* distributions
- **Metrics** *consume* distributions + true values
- They're composable but independent

---

## BaseObject (from skbase)

All skpro objects inherit from `skbase.BaseObject`. This provides:

| Feature | Description |
|---------|-------------|
| **Tag system** | `_tags` dict on class, `get_tags()`, `set_tags()` at runtime |
| **Parameter storage** | Everything in `__init__` is a parameter, stored as `self.param` |
| **`get_params()` / `set_params()`** | sklearn-compatible parameter access |
| **`__repr__`** | Nice string representation |
| **`clone()`** | Deep copy for cross-validation |
| **`is_fitted`** property | Tracks fitted state |

### Tag Conventions

```python
_tags = {
    "object_type": "regressor_proba",     # or "distribution", "metric"
    "capability:update": False,           # can the estimator be updated incrementally?
    "capability:multioutput": False,      # can handle multiple output variables?
    "capability:missing": True,           # can handle missing values in X?
    "capability:survival": False,         # can use censoring information?
    "X_inner_mtype": "pd_DataFrame_Table", # expected input format
    "y_inner_mtype": "pd_DataFrame_Table",
    "python_version": None,               # PEP 440 version specifier
    "python_dependencies": None,          # soft dependencies
}
```

### Parameter Naming Convention

- **Constructor params** → `self.paramname` (stored as-is)
- **Fitted attributes** → `self.paramname_` (trailing underscore)
- **Never modify** constructor params after `__init__`

---

## The Mixin Pattern

skpro uses mixins extensively. The pattern:

```python
class MyEstimator(SomeMixin, BaseProbaRegressor):
    ...
```

MRO (Method Resolution Order) determines which method wins. This is critical for the [[04-online-learning-gap|OnlineRegressorMixin]] — it overrides `update()` from `BaseProbaRegressor`.

---

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `skpro/distributions/base/_base.py` | 2,090 | BaseDistribution — pdf, cdf, ppf, sample, mean, var, energy |
| `skpro/regression/base/_base.py` | 972 | BaseProbaRegressor — fit, predict, predict_proba, update |
| `skpro/regression/base/_delegate.py` | ~200 | _DelegatedProbaRegressor — wraps another estimator |
| `skpro/regression/base/_online_mixin.py` | ~40 | OnlineRegressorMixin — update, update_predict, stream_fit |
| `skpro/metrics/base.py` | 642 | BaseProbaMetric, BaseDistrMetric |

---

## Quiz Answers

1. **If I create `Normal(mu=[1,2,3], sigma=1)` with index of length 3, what shape does `sigma` become internally?**
   → `sigma` broadcasts to shape `(3, 1)` — a column vector matching `mu`'s length

2. **I implement only `_ppf` in a new distribution. What methods get auto-derived?**
   → `_mean` (integrate ppf), `_var` (integrate ppf² - mean²), `_sample` (inverse transform), `_energy` (via ppf integration), `_cdf` (MC)

3. **What does `predict()` return by default?**
   → Calls `predict_proba()` then takes `.mean()` of the returned distribution

4. **If a regressor has `capability:update=False` and I call `update(X, y)`, what happens?**
   → Silently returns `self` without doing anything — no error, no update

5. **What's the difference between `_predict_proba` and `predict_proba`?**
   → `predict_proba` is public (handles validation, conversion); `_predict_proba` is private (receives validated numpy/DataFrame input)

---

## Related

- [[02-base-distribution-deep-dive]]
- [[03-base-proba-regressor-deep-dive]]
- [[04-online-learning-gap]]
