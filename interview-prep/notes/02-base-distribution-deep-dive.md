---
tags: [skpro, distributions, base-class, esoc-interview]
created: 2026-04-10
session: 2
---

# BaseDistribution Deep Dive

**File:** `skpro/distributions/base/_base.py` (2,090 lines)  
**Author:** fkiraly

---

## What It Does

`BaseDistribution` is a **pandas-like interface for probability distributions**. Each instance represents a grid of random variables:
- **Rows** (`index`) = data points
- **Columns** (`columns`) = output variables

```python
dist = Normal(mu=[1, 2, 3], sigma=1)
# dist.shape = (3, 1) — 3 rows, 1 column
# dist.pdf(x) returns DataFrame with same shape
```

---

## Key Design Patterns

### 1. Pandas-like Interface

| pandas | BaseDistribution |
|--------|-----------------|
| `df.loc[idx, col]` | `dist.loc[idx, col]` |
| `df.iloc[0:3]` | `dist.iloc[0:3]` |
| `df.at[i, j]` | `dist.at[i, j]` |
| `df.iat[i, j]` | `dist.iat[i, j]` |
| `df.head(n)` | `dist.head(n)` |
| `df.tail(n)` | `dist.tail(n)` |
| `df.shape` | `dist.shape` |
| `df.ndim` | `dist.ndim` |
| `len(df)` | `len(dist)` |

### 2. Parameter Broadcasting

When you create a distribution, parameters get broadcast to match the shape:

```python
Normal(mu=[1, 2, 3], sigma=1)
# sigma=1 broadcasts to shape (3, 1)
```

**How it works:**
- `_get_bc_params_dict()` — the core broadcasting engine
- Takes all parameters, adds index/columns, calls `np.broadcast_arrays()`
- Returns dict of broadcast parameters + shape info

**Two methods:**
- `_get_bc_params(*args)` — tuple-based (older)
- `_get_bc_params_dict(**kwargs)` — dict-based (newer, preferred)

### 3. The `_boilerplate()` Method

Every public distribution method calls `_boilerplate()`:

```python
def pdf(self, x):
    return self._boilerplate("_pdf", x=x)
```

**What `_boilerplate` does:**
1. Coerces input (DataFrame → numpy)
2. Broadcasts to self's shape
3. Calls the private method (e.g., `_pdf`)
4. Wraps result back into DataFrame

This means **subclasses only implement private methods** (`_pdf`, `_cdf`, `_ppf`), and the public methods handle all the scaffolding.

### 4. Method Inter-Derivation (CRITICAL)

If you don't implement a method, the base class derives it from others:

| If you implement... | Base derives... | How |
|---|---|---|
| `_ppf` | `_mean` | Integrate ppf over [0,1]: `E[X] = ∫ F⁻¹(p) dp` |
| `_ppf` | `_var` | `Var[X] = ∫ F⁻¹(p)² dp - E[X]²` |
| `_ppf` | `_sample` | Inverse transform: `sample = ppf(uniform_random)` |
| `_ppf` | `_energy` | Integrate `|ppf(p) - x|` over [0,1] |
| `_cdf` | `_pdf` | 6th-order central difference: `f(x) ≈ dF/dx` |
| `_pdf` | `_log_pdf` | `log(pdf)` |
| `_pdf` | `_pmf` | For discrete distributions |
| `_cdf` | `_surv` | `S(x) = 1 - F(x)` |
| `_pdf`, `_surv` | `_haz` | `h(x) = f(x) / S(x)` |
| Nothing | Everything | Monte Carlo with 1000 samples |

**MC sample sizes** (configurable via tags):
- `approx_mean_spl`: 1000
- `approx_var_spl`: 1000
- `approx_energy_spl`: 1000
- `approx_spl`: 1000
- `bisect_iter`: 1000 (for ppf via bisection)

### 5. Capabilities Tags

```python
# Which methods are numerically exact (not MC approximations)?
"capabilities:exact": ["pdf", "cdf", "mean", "var", "sample"]

# Which methods use MC approximation?
"capabilities:approx": ["energy", "pdfnorm"]
```

### 6. Distribution Parameter Types

| Tag | Meaning |
|-----|---------|
| `distr:paramtype: parametric` | Fixed number of params (e.g., Normal: mu, sigma) |
| `distr:paramtype: nonparametric` | Data-driven (e.g., KernelMixture, Empirical) |
| `distr:paramtype: composite` | Made of sub-distributions (e.g., Mixture) |
| `distr:measuretype: continuous` | Has pdf (e.g., Normal, Gamma) |
| `distr:measuretype: discrete` | Has pmf (e.g., Poisson, Binomial) |
| `distr:measuretype: mixed` | Both (e.g., some survival distributions) |

---

## Core Methods to Implement (for a new distribution)

**Minimum:** `_ppf` OR `_sample`

**Recommended for exact (non-MC) computation:**
```python
def _pdf(self, x): ...    # Probability density
def _cdf(self, x): ...    # Cumulative distribution
def _ppf(self, p): ...    # Quantile / inverse CDF
def _mean(self): ...      # Expected value
def _var(self): ...       # Variance
def _sample(self, n_samples=None): ...  # Random sampling
```

---

## Important Implementation Details

### Subset Parameters (`_subset_params`)

When you do `dist.loc[0]`, the base class calls `_subset_params()` which:
1. Gets all distribution params via `_get_dist_params()`
2. Subsets each param array to the requested rows/cols
3. Creates a new instance with subsetted params

This is why distributions are **immutable** — subsetting creates a new instance.

### The `_coerce_to_self_index_df` and `_coerce_to_self_index_np` Methods

These coerce external input to match the distribution's internal shape. Used by `_boilerplate` to ensure consistent input/output formats.

### `quantile()` vs `ppf()`

Both compute quantiles, but with different broadcasting:
- `ppf(p)` — numpy-style broadcasting, shape-preserving
- `quantile(alpha)` — sktime-style, returns DataFrame with MultiIndex columns `(variable, alpha)`

---

## Related

- [[01-skpro-class-hierarchy]]
- [[03-base-proba-regressor-deep-dive]]
- KernelMixture implementation (PR #721)
