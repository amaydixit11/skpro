---
tags: [skpro, regression, base-class, esoc-interview]
created: 2026-04-10
session: 3
---

# BaseProbaRegressor Deep Dive

**File:** `skpro/regression/base/_base.py` (972 lines)  
**Author:** fkiraly

---

## The Predict Contract

`BaseProbaRegressor` defines **5 prediction modes**:

```python
fit(X, y, C=None)           # Train the model
predict(X)                   # Point prediction → DataFrame (uses mean of predict_proba)
predict_proba(X)             # Full distribution → BaseDistribution
predict_interval(X, coverage) # Interval predictions → DataFrame with MultiIndex
predict_quantiles(X, alpha)   # Quantile predictions → DataFrame with MultiIndex
predict_var(X)               # Variance predictions → DataFrame
```

### Input/Output Types

| Method | Input | Output |
|--------|-------|--------|
| `fit` | X: DataFrame, y: DataFrame, C: DataFrame (optional) | self |
| `predict` | X: DataFrame (same columns as fit) | DataFrame (same index as X, same columns as y) |
| `predict_proba` | X: DataFrame | BaseDistribution (same index as X, same columns as y) |
| `predict_interval` | X: DataFrame, coverage: float/list | DataFrame with 3-level MultiIndex columns: `(variable, coverage, lower/upper)` |
| `predict_quantiles` | X: DataFrame, alpha: float/list | DataFrame with 2-level MultiIndex columns: `(variable, alpha)` |
| `predict_var` | X: DataFrame | DataFrame (same index as X, same columns as y) |

---

## The Defaulting Logic Chain

If you don't implement a predict method, the base class derives it:

```
predict_proba ← predict_var → Normal(mu=predict, sigma=sqrt(var))
predict_interval ← predict_quantiles (relabel columns: alpha → coverage/lower-upper)
predict_quantiles ← predict_interval (compute: alpha = 0.5 ± coverage/2)
predict_quantiles ← predict_proba (use ppf of distribution)
predict_interval ← predict_quantiles
predict ← predict_proba (take mean of distribution)
predict_var ← predict_proba (take var of distribution)
```

### How `predict_proba` defaults to `predict_var`:

```python
def _predict_proba(self, X):
    pred_var = self.predict_var(X=X)
    pred_std = np.sqrt(pred_var)
    pred_mean = self.predict(X=X)
    return Normal(mu=pred_mean, sigma=pred_std)
```

This is a **Normal approximation** — it assumes the predictive distribution is Normal with the given mean and variance. This is often wrong but provides a fallback.

---

## The `update()` Method

```python
def update(self, X, y, C=None):
    capa_online = self.get_tag("capability:update")
    if not capa_online:
        return self  # ← silently does nothing!
    check_ret = self._check_X_y(X, y, C, return_metadata=True)
    X_inner = check_ret["X_inner"]
    y_inner = check_ret["y_inner"]
    return self._update(X_inner, y_inner)

def _update(self, X, y, C=None):
    raise NotImplementedError  # ← must be overridden by subclasses
```

**Critical:** The default behavior is to **silently ignore** the update call. This is the gap the proposal addresses.

---

## Fitted State

```python
def fit(self, X, y, C=None):
    # ... validation ...
    self._is_fitted = True  # ← sets fitted flag
    return self._fit(X_inner, y_inner)
```

- `self._is_fitted` — boolean flag, set to True in `fit()`
- `self.check_is_fitted()` — raises `NotFittedError` if not fitted
- Fitted attributes use trailing underscore: `self.coef_`, `self.w_`, `self.var_`

---

## Data Type Conversion

`BaseProbaRegressor` handles multiple input mtypes:

```python
ALLOWED_MTYPES = [
    "pd_DataFrame_Table",   # pandas DataFrame
    "pd_Series_Table",      # pandas Series
    "numpy1D",              # 1D numpy array
    "numpy2D",              # 2D numpy array
]
# Also supports polars_eager_table if polars+pyarrow installed
```

The `_check_X_y()` method:
1. Validates input types
2. Converts to `X_inner_mtype` and `y_inner_mtype` (specified in tags)
3. Returns converted data + metadata

---

## The `__rmul__` Magic Method

```python
def __rmul__(self, other):
    if hasattr(other, "transform"):
        return other * Pipeline([self])
    else:
        return NotImplemented
```

This allows **pipeline composition**:
```python
from sklearn.preprocessing import StandardScaler
from skpro.regression.glm import GLMRegressor

pipeline = StandardScaler() * GLMRegressor()
# Equivalent to: Pipeline([StandardScaler(), GLMRegressor()])
```

---

## Extension Pattern (How to Add a New Regressor)

```python
class MyRegressor(BaseProbaRegressor):
    _tags = {
        "capability:update": True,  # if implementing _update
        "capability:multioutput": False,
        "capability:missing": True,
    }

    def __init__(self, param1=1.0, param2="auto"):
        self.param1 = param1
        self.param2 = param2
        super().__init__()

    def _fit(self, X, y, C=None):
        # Implement training logic
        self.coef_ = ...  # fitted attributes end with _
        return self

    def _predict_proba(self, X):
        # Return a BaseDistribution
        return Normal(mu=..., sigma=...)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [
            {"param1": 1.0},
            {"param1": 0.1, "param2": "custom"},
        ]
```

**Minimum to implement:**
1. `_fit(self, X, y, C=None)` — required
2. At least one of: `_predict_proba`, `_predict_interval`, `_predict_quantiles`, `_predict_var`
3. `get_test_params()` — required for test framework

---

## Related

- [[01-skpro-class-hierarchy]]
- [[02-base-distribution-deep-dive]]
- [[04-online-learning-gap]]
