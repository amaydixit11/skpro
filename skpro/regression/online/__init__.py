"""Meta-algorithms to build online regression models."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

# New online regressors
from ._sgd import SGDProbaRegressor
from ._rls import RLSProbaRegressor
from ._compositors import (
    SlidingWindowRegressor,
    ForgettingFactorRegressor,
    DriftDetectorRegressor,
)

# Legacy meta-estimators (refit-based strategies)
from skpro.regression.online._dont_refit import OnlineDontRefit
from skpro.regression.online._refit import OnlineRefit
from skpro.regression.online._refit_every import OnlineRefitEveryN

__all__ = [
    # New native online estimators
    "SGDProbaRegressor",
    "RLSProbaRegressor",
    "SlidingWindowRegressor",
    "ForgettingFactorRegressor",
    "DriftDetectorRegressor",
    # Legacy refit-based strategies
    "OnlineDontRefit",
    "OnlineRefit",
    "OnlineRefitEveryN",
]
