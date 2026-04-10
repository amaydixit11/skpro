# Design Doc: Online/Stream Learning API for skpro
Date: 2026-04-10
Status: Proposed
Topic: Incremental and Adaptive Probabilistic Regression

## 1. Abstract
The goal of this project is to introduce a native online/stream learning ecosystem to `skpro`. Currently, `skpro` supports batch-refit strategies. This design introduces a layered architecture that separates the *mechanism* of incremental updates (the API) from the *strategy* of adaptation (compositors).

The system allows for:
1. **Memory-efficient training** on datasets that exceed RAM limits.
2. **Real-time adaptation** to concept drift in non-stationary data streams.
3. **Prequential evaluation** to ensure rigorous, leakage-free performance monitoring.

## 2. Core API: `OnlineRegressorMixin`

To provide a consistent interface without bloating the base class, we introduce a mixin.

### 2.1 Responsibility
The `OnlineRegressorMixin` orchestrates the streaming process. It does not implement the learning logic itself but defines the contract that all online models must follow.

### 2.2 Interface
- **`_update(X, y, C=None)`**: (Abstract) The backend hook. Subclasses must implement this to update their internal state (e.g., weights, covariance matrices).
- **`update(X, y, C=None)`**: 
    - Checks `capability:update` tag.
    - Performs input validation via `_check_X_y`.
    - Calls `_update(X_inner, y_inner, C_inner)`.
- **`update_predict(X, y, C=None)`**: 
    - Implements the **Prequential Protocol**:
        1. `predictions = self.predict_proba(X)`
        2. `self.update(X, y, C)`
        3. `return predictions`
- **`stream_fit(X_stream, y_stream, batch_size=1)`**:
    - Consumes iterables of data.
    - Calls `fit()` on the first batch to initialize.
    - Calls `update()` on all subsequent batches.

## 3. Concrete Native Online Estimators

These models implement the `_update` logic for high-performance incremental learning.

### 3.1 `SGDProbaRegressor`
- **Math**: Implements Stochastic Gradient Descent for linear probabilistic regression.
- **State**: Maintains weight vector $w$ and running variance $\sigma^2$.
- **Output**: `Normal(mu=Xw, sigma=sqrt(var))`.
- **Capability**: High-velocity data, low memory footprint.

### 3.2 `RLSProbaRegressor`
- **Math**: Implements Recursive Least Squares.
- **State**: Maintains weights $w$ and inverse correlation matrix $P$.
- **Output**: `Normal` distribution with variance reflecting both observation noise and parameter uncertainty.
- **Capability**: Mathematically optimal for linear systems, highly stable.

### 3.3 `OnlineGLMRegressor`
- **Math**: Incremental Generalized Linear Model using warm-started IRLS.
- **Integration**: Leverages existing `skpro` distribution families (Poisson, Gamma, etc.).
- **Capability**: Versatile distribution support in an online setting.

## 4. Adaptation Layer (Compositors)

Following the `skpro` delegation pattern, these meta-estimators inherit from `_DelegatedProbaRegressor`.

### 4.1 `SlidingWindowRegressor`
- **Mechanism**: Maintains a FIFO buffer of the last $N$ observations.
- **`_update`**: 
    1. Append new batch to buffer.
    2. Evict old data to maintain size $N$.
    3. Trigger `self.estimator_.fit(buffer_X, buffer_y)`.

### 4.2 `ForgettingFactorRegressor`
- **Mechanism**: Applies exponential decay to observation weights.
- **`_update`**: 
    1. Delegates `update` to wrapped estimator.
    2. If the wrapped estimator supports `sample_weight`, it passes weights $w_t = \lambda^{(T-t)}$.

### 4.3 `DriftDetectorRegressor`
- **Mechanism**: Monitors prediction error (e.g., using a rolling mean or ADWIN).
- **`_update`**: 
    1. Calls `update_predict` to get current error.
    2. If drift is detected $\rightarrow$ triggers a reset or refit of the wrapped estimator.

## 5. Testing & Verification

### 5.1 Prequential Correctness
Verify that `update_predict(X, y)` returns predictions based on the state *before* seeing `y`.

### 5.2 Stream Equivalence
For `RLSProbaRegressor`, verify that updating $n$ observations one-by-one yields the same weights as a single batch `fit()` on all $n$.

### 5.3 Compositor Stacking
Verify that `DriftDetector(SlidingWindow(SGDProbaRegressor()))` produces valid `BaseDistribution` objects at every step and correctly triggers refits upon injected drift.

### 5.4 Tagging
Ensure all online models have `capability:update: True` and all compositors correctly clone tags from their wrapped estimators.
