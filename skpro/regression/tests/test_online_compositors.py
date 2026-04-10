import pytest
import pandas as pd
import numpy as np
from skpro.regression.online import SlidingWindowRegressor, SGDProbaRegressor, ForgettingFactorRegressor, DriftDetectorRegressor

def test_sliding_window_drift():
    # Model that learns mean of y
    # We use a simple SGDProbaRegressor
    X = pd.DataFrame([[1]] * 200)
    y1 = pd.DataFrame([1.0] * 100) # first 100 are 1.0
    y2 = pd.DataFrame([10.0] * 100) # next 100 are 10.0

    model = SlidingWindowRegressor(SGDProbaRegressor(learning_rate=1.0), window_size=10)
    model.fit(X[:100], y1)

    # After fitting on y1, mean should be ~1
    assert np.abs(model.predict(X[:1]).values[0] - 1.0) < 1.0

    # Update with y2
    for i in range(100):
        model.update(X[100+i:101+i], y2[i:i+1])

    # After window slides fully into y2, mean should be ~10
    assert np.abs(model.predict(X[:1]).values[0] - 10.0) < 2.0

def test_forgetting_factor():
    X = pd.DataFrame([[1]] * 100)
    y = pd.DataFrame([1.0] * 100)

    model = ForgettingFactorRegressor(SGDProbaRegressor(), forgetting_factor=0.9)
    model.fit(X, y)

    # Just verify it doesn't crash and produces predictions
    pred = model.predict(X[:1])
    assert pred is not None

def test_drift_detector():
    X = pd.DataFrame([[1]] * 200)
    y1 = pd.DataFrame([1.0] * 100)
    y2 = pd.DataFrame([10.0] * 100)

    model = DriftDetectorRegressor(SGDProbaRegressor(learning_rate=1.0), threshold=1.0, window_size=10)
    model.fit(X[:100], y1)

    # After fitting on y1, mean should be ~1
    assert np.abs(model.predict(X[:1]).values[0] - 1.0) < 1.0

    # Update with y2 to trigger drift
    for i in range(100):
        model.update(X[100+i:101+i], y2[i:i+1])

    # Should have adapted to ~10
    assert np.abs(model.predict(X[:1]).values[0] - 10.0) < 2.0

def test_stacking():
    # DriftDetector(SlidingWindow(SGDProbaRegressor()))
    # Verify it's an OnlineRegressorMixin and produces BaseDistribution

    inner = SlidingWindowRegressor(SGDProbaRegressor(), window_size=10)
    model = DriftDetectorRegressor(inner, threshold=1.0, window_size=10)

    X = pd.DataFrame([[1]] * 20)
    y = pd.DataFrame([1.0] * 20)

    model.fit(X, y)

    # Test update
    model.update(X[:1], y[:1])

    # Test predict_proba returns a BaseDistribution (Normal)
    dist = model.predict_proba(X[:1])
    assert dist.mean().iloc[0] is not None
    assert hasattr(dist, "pdf")
