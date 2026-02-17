# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Kernel mixture distribution (kernel density estimate)."""
__author__ = ["amaydixit11"]

import warnings

import numpy as np
import pandas as pd
from scipy.special import erf

from skpro.distributions.base import BaseDistribution

_KERNEL_VARIANCE = {
    "gaussian": 1.0,
    "epanechnikov": 1.0 / 5.0,
    "tophat": 1.0 / 3.0,
    "cosine": 1.0 - 8.0 / (np.pi**2),
    "linear": 1.0 / 6.0,
}


class KernelMixture(BaseDistribution):
    r"""Kernel mixture distribution, a.k.a. kernel density estimate.

    This distribution represents a smooth nonparametric density estimate
    as a weighted mixture of kernel functions centered at support points:

    .. math::

        f(x) = \sum_{i=1}^{n} w_i \frac{1}{h} K\!\left(\frac{x - x_i}{h}\right)

    where :math:`K` is the kernel function, :math:`h` is the bandwidth,
    :math:`x_i` are the support points, and :math:`w_i` are the weights
    (summing to 1).

    This is a vectorized special case of ``Mixture`` where all components
    share a common kernel type and bandwidth. Unlike ``Mixture``, it does
    not create per-component distribution objects, making it efficient
    for large numbers of support points (e.g., kernel density estimation).

    Parameters
    ----------
    support : array-like
        Support points (data) on which the kernel density is centered.

        Three input shapes are accepted, selected automatically:

        * **1-D** array-like of shape ``(n_support,)`` — a single set of
          support points **shared across all marginals** (existing behaviour).
        * **2-D** array-like of shape ``(n_instances, n_support)`` — one row
          of support points per marginal / location (all rows same length).
        * **list of 1-D array-likes** (ragged) — one array per marginal,
          allowing different numbers of support points per location.

        The 2-D and ragged modes require ``index`` to be set and its length
        to match the number of rows / list elements.

    bandwidth : float, or str ``"scott"`` or ``"silverman"``, default=1.0
        Bandwidth of the kernel.
        If float, used directly as the bandwidth parameter ``h``.
        If ``"scott"``, bandwidth is computed as
        ``n**(-1/5) * std(support, ddof=1)``.
        If ``"silverman"``, bandwidth is computed as
        ``(4/(3*n))**(1/5) * std(support, ddof=1)``.
        For per-location support modes, the rule is applied independently
        to each row's support points and a per-location bandwidth array is
        stored.

    kernel : str or ``BaseDistribution``, default="gaussian"
        The kernel function to use.
        If str, must be one of the built-in kernels:
        ``"gaussian"``, ``"epanechnikov"``, ``"tophat"``,
        ``"cosine"``, ``"linear"``.
        If a ``BaseDistribution`` instance, it is used as a zero-centered,
        unit-scale kernel. The distribution must be scalar (0D).

    weights : array-like or None, default=None
        Weights for each support point. If None, uniform weights are used.
        Weights are normalized to sum to 1.

        Accepted shapes mirror ``support``:

        * ``None`` or 1-D ``(n_support,)`` — shared (or uniform) weights.
        * 2-D ``(n_instances, n_support)`` — per-marginal weights.
        * List of 1-D array-likes — per-marginal weights (ragged-compatible).

    random_state : int, np.random.Generator, or None, default=None
        Controls randomness for reproducible sampling.
        If int, used as seed for ``np.random.default_rng``.
        If ``np.random.Generator``, used directly.
        If None, a fresh unseeded ``default_rng()`` is created at init.

        .. note::
            When ``kernel`` is a ``BaseDistribution`` instance, the kernel's
            own RNG is used for noise generation and is **not** controlled
            by ``random_state``.  Only the support-point selection is
            reproducible in that case.

    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.kernel_mixture import KernelMixture
    >>> import numpy as np

    Scalar distribution with built-in Gaussian kernel:

    >>> support = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> km = KernelMixture(support=support, bandwidth=0.5, kernel="gaussian")
    >>> float(km.mean())
    2.0
    >>> pdf_val = km.pdf(1.5)

    Using a distribution object as a custom kernel:

    >>> from skpro.distributions.normal import Normal
    >>> km_custom = KernelMixture(
    ...     support=[0.0, 1.0, 2.0],
    ...     bandwidth=0.5,
    ...     kernel=Normal(mu=0, sigma=1),
    ... )

    Using weighted support points:

    >>> km_weighted = KernelMixture(
    ...     support=[0.0, 1.0, 2.0],
    ...     bandwidth=0.5,
    ...     weights=[0.1, 0.3, 0.6],
    ... )
    >>> pdf_val = km_weighted.pdf(1.0)

    Per-location support (2-D array — one row per marginal):

    >>> km_2d = KernelMixture(
    ...     support=[[0.0, 1.0, 2.0], [5.0, 6.0, 7.0]],
    ...     bandwidth=0.5,
    ...     kernel="gaussian",
    ...     index=pd.RangeIndex(2),
    ...     columns=pd.Index(["a"]),
    ... )
    >>> pdf_val = km_2d.pdf(pd.DataFrame({"a": [1.0, 6.0]}, index=pd.RangeIndex(2)))

    Per-location support (ragged — different number of points per marginal):

    >>> km_ragged = KernelMixture(
    ...     support=[np.array([0.0, 1.0]), np.array([5.0, 6.0, 7.0, 8.0])],
    ...     bandwidth=0.5,
    ...     kernel="gaussian",
    ...     index=pd.RangeIndex(2),
    ...     columns=pd.Index(["a"]),
    ... )

    See Also
    --------
    Mixture : Mixture of arbitrary distribution objects.
    Empirical : Empirical distribution (weighted sum of deltas).

    Notes
    -----
    **Support modes** — the ``support`` parameter accepts three formats:

    * *Shared* (1-D): a single support array is broadcast across all marginals.
      This is the original behaviour and remains the default.
    * *Per-location 2-D*: a 2-D array whose rows are aligned with ``index``.
      Fully vectorised; all rows must have the same length.
    * *Per-location ragged*: a list of 1-D arrays, one per location in ``index``.
      Rows may have different lengths; a Python loop is used internally so it
      is somewhat slower than the 2-D path for large batches.

    Evaluation cost is ``O(len(support) * len(x))`` for ``pdf`` and ``cdf``.
    Very large support arrays (e.g. >10 000 points) may be slow.
    """

    _tags = {
        "capabilities:approx": ["energy", "ppf", "pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "nonparametric",
        "broadcast_init": "off",
    }

    _VALID_KERNELS = {"gaussian", "epanechnikov", "tophat", "cosine", "linear"}

    def __init__(
        self,
        support,
        bandwidth=1.0,
        kernel="gaussian",
        weights=None,
        random_state=None,
        index=None,
        columns=None,
    ):
        self.support = support
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.weights = weights
        self.random_state = random_state

        if isinstance(random_state, np.random.Generator):
            self._rng = random_state
        elif random_state is not None:
            self._rng = np.random.default_rng(random_state)
        else:
            self._rng = np.random.default_rng()

        if isinstance(kernel, str):
            if kernel not in self._VALID_KERNELS:
                raise ValueError(
                    f"Unknown kernel '{kernel}'. "
                    f"Must be one of {sorted(self._VALID_KERNELS)}."
                )
            self._kernel_mode = "builtin"
        elif isinstance(kernel, BaseDistribution):
            if kernel.ndim != 0:
                raise ValueError(
                    "kernel distribution must be scalar (0D), "
                    f"got ndim={kernel.ndim}."
                )
            kernel_mean = float(np.ravel(np.asarray(kernel.mean()))[0])
            if abs(kernel_mean) > 1e-6:
                warnings.warn(
                    f"kernel distribution has non-zero mean ({kernel_mean}). "
                    "KernelMixture assumes a zero-centered kernel; "
                    "mean() and var() may be incorrect.",
                    UserWarning,
                    stacklevel=2,
                )
            self._kernel_mode = "distribution"
        else:
            raise TypeError(
                f"kernel must be a string or a BaseDistribution instance, "
                f"got {type(kernel).__name__}."
            )

        self._support_mode, self._support = self._parse_support(support)
        self._bandwidth = self._parse_bandwidth(bandwidth, self._support_mode)
        self._weights = self._parse_weights(weights, self._support_mode)

        if self._support_mode != "shared" and index is not None:
            n_rows = (
                len(self._support)
                if self._support_mode == "per_location_ragged"
                else self._support.shape[0]
            )
            if n_rows != len(index):
                raise ValueError(
                    f"Number of support rows ({n_rows}) must match "
                    f"length of index ({len(index)})."
                )

        super().__init__(index=index, columns=columns)

    @staticmethod
    def _parse_support(support):
        """Return (support_mode, internal_support) from raw input.

        Parameters
        ----------
        support : array-like or list of array-likes

        Returns
        -------
        support_mode : str
            One of ``"shared"``, ``"per_location_2d"``, ``"per_location_ragged"``.
        internal_support : np.ndarray or list[np.ndarray]
        """
        # List input — could be a list of scalars (→ 1-D) or list of arrays (→ ragged)
        if isinstance(support, list):
            # Try to promote to 2-D array first (uniform-length rows)
            try:
                arr = np.asarray(support, dtype=float)
                if arr.ndim == 1:
                    return "shared", arr
                if arr.ndim == 2:
                    return "per_location_2d", arr
                # ndim > 2 after list promotion — fall through to ragged
            except (ValueError, TypeError):
                pass
            # Ragged list: each element is its own support array
            parsed = [np.asarray(s, dtype=float).ravel() for s in support]
            return "per_location_ragged", parsed

        arr = np.asarray(support, dtype=float)
        if arr.ndim == 1:
            return "shared", arr
        if arr.ndim == 2:
            return "per_location_2d", arr
        raise ValueError(
            f"support must be 1-D or 2-D array-like, got shape {arr.shape}."
        )

    def _parse_bandwidth(self, bandwidth, support_mode):
        """Compute internal bandwidth from the raw parameter.

        For per-location modes, ``"scott"`` / ``"silverman"`` rules are applied
        independently to each row's support points, yielding a 1-D array of
        per-location bandwidths.
        """
        if isinstance(bandwidth, str):
            rule = bandwidth
            if rule not in ("scott", "silverman"):
                raise ValueError(
                    f"Unknown bandwidth rule '{rule}'. "
                    "Must be a float, 'scott', or 'silverman'."
                )
            if support_mode == "shared":
                return self._bw_rule(rule, self._support)
            elif support_mode == "per_location_2d":
                return np.array(
                    [self._bw_rule(rule, row) for row in self._support]
                )
            else:  # ragged
                return np.array(
                    [self._bw_rule(rule, row) for row in self._support]
                )
        else:
            bw = float(bandwidth)
            if bw <= 0:
                raise ValueError(f"bandwidth must be positive, got {bw}.")
            return bw

    @staticmethod
    def _bw_rule(rule, support_arr):
        """Apply Scott or Silverman rule to a 1-D support array."""
        n = len(support_arr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            std = np.std(support_arr, ddof=1)
        if np.isnan(std) or std < 1e-15:
            std = 1.0
        if rule == "scott":
            return n ** (-1.0 / 5.0) * std
        return (4.0 / (3.0 * n)) ** (1.0 / 5.0) * std  # silverman

    def _parse_weights(self, weights, support_mode):
        """Validate and normalise weights to match the support mode."""
        def _normalise_1d(w_raw, n, label=""):
            w = np.asarray(w_raw, dtype=float).ravel()
            if len(w) != n:
                raise ValueError(
                    f"weights length ({len(w)}) must match "
                    f"support length ({n}){label}."
                )
            if np.any(w < 0):
                raise ValueError("All weights must be non-negative.")
            s = w.sum()
            if s <= 0:
                raise ValueError("weights must have positive sum.")
            return w / s

        if support_mode == "shared":
            n = len(self._support)
            if weights is None:
                return np.ones(n) / n
            return _normalise_1d(weights, n)

        # Per-location modes
        rows = (
            self._support
            if support_mode == "per_location_ragged"
            else list(self._support)  # iterate over rows of 2-D array
        )
        ns = [len(row) for row in rows]

        if weights is None:
            return [np.ones(n) / n for n in ns]

        # weights supplied — try to parse analogously to support
        if isinstance(weights, list):
            try:
                w_arr = np.asarray(weights, dtype=float)
            except (ValueError, TypeError):
                w_arr = None
        else:
            w_arr = np.asarray(weights, dtype=float)

        if w_arr is not None and w_arr.ndim == 2:
            # 2-D weights — one row per location
            if w_arr.shape[0] != len(rows):
                raise ValueError(
                    f"2-D weights row count ({w_arr.shape[0]}) must match "
                    f"number of support locations ({len(rows)})."
                )
            return [_normalise_1d(w_arr[i], ns[i], f" for location {i}")
                    for i in range(len(rows))]

        if w_arr is not None and w_arr.ndim == 1:
            # Ambiguous: a shared 1-D weight array with per-location support
            raise ValueError(
                "1-D weights are ambiguous when support is per-location "
                "(2-D or ragged). Provide weights with the same shape as "
                "support (2-D array or list of 1-D arrays)."
            )

        if w_arr is not None and w_arr.ndim == 0:
            raise ValueError(
                "Scalar weights are not supported with per-location support. "
                "Provide weights with the same shape as support."
            )

        # Treat as a list of 1-D weight arrays (ragged compatible)
        w_list = (
            weights if isinstance(weights, list)
            else [weights[i] for i in range(len(rows))]
        )
        return [_normalise_1d(w_list[i], ns[i], f" for location {i}")
                for i in range(len(rows))]

    def _bw_for(self, loc_idx=None):
        """Return bandwidth for location ``loc_idx`` (or scalar if shared)."""
        if np.isscalar(self._bandwidth):
            return self._bandwidth
        return self._bandwidth[loc_idx]

    def _support_for(self, loc_idx):
        """Return 1-D support array for location index ``loc_idx``."""
        if self._support_mode == "shared":
            return self._support
        if self._support_mode == "per_location_2d":
            return self._support[loc_idx]
        return self._support[loc_idx]  # ragged list

    def _weights_for(self, loc_idx):
        """Return 1-D weight array for location index ``loc_idx``."""
        if self._support_mode == "shared":
            return self._weights
        return self._weights[loc_idx]

    def _kernel_pdf(self, u):
        """Evaluate kernel pdf K(u), vectorized."""
        if self._kernel_mode == "distribution":
            u_arr = np.asarray(u, dtype=float)
            return np.asarray(
                self.kernel._pdf(u_arr), dtype=float
            ).reshape(u_arr.shape)
        if self.kernel == "gaussian":
            return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
        elif self.kernel == "epanechnikov":
            return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)
        elif self.kernel == "tophat":
            return np.where(np.abs(u) <= 1, 0.5, 0.0)
        elif self.kernel == "cosine":
            return np.where(
                np.abs(u) <= 1,
                (np.pi / 4) * np.cos(np.pi * u / 2),
                0.0,
            )
        elif self.kernel == "linear":
            return np.where(np.abs(u) <= 1, 1 - np.abs(u), 0.0)
        else:
            raise ValueError(f"Unsupported kernel '{self.kernel}'.")

    def _kernel_cdf(self, u):
        """Evaluate kernel cdf, vectorized."""
        if self._kernel_mode == "distribution":
            u_arr = np.asarray(u, dtype=float)
            return np.asarray(
                self.kernel._cdf(u_arr), dtype=float
            ).reshape(u_arr.shape)
        if self.kernel == "gaussian":
            return 0.5 * (1 + erf(u / np.sqrt(2)))
        elif self.kernel == "epanechnikov":
            cdf_inner = 0.5 + 0.75 * u - 0.25 * u**3
            return np.where(u < -1, 0.0, np.where(u > 1, 1.0, cdf_inner))
        elif self.kernel == "tophat":
            cdf_inner = 0.5 * (1 + u)
            return np.where(u < -1, 0.0, np.where(u > 1, 1.0, cdf_inner))
        elif self.kernel == "cosine":
            cdf_inner = 0.5 + 0.5 * np.sin(np.pi * u / 2)
            return np.where(u < -1, 0.0, np.where(u > 1, 1.0, cdf_inner))
        elif self.kernel == "linear":
            cdf_low = 0.5 * (1 + u) ** 2
            cdf_high = 1 - 0.5 * (1 - u) ** 2
            cdf_inner = np.where(u <= 0, cdf_low, cdf_high)
            return np.where(u < -1, 0.0, np.where(u > 1, 1.0, cdf_inner))
        else:
            raise ValueError(f"Unsupported kernel '{self.kernel}'.")

    def _kernel_sample(self, size, rng):
        """Sample from the kernel distribution."""
        if self._kernel_mode == "distribution":
            if self.random_state is not None:
                warnings.warn(
                    "random_state does not control reproducibility when "
                    "kernel is a BaseDistribution instance. The kernel's "
                    "own RNG is used for noise generation.",
                    UserWarning,
                    stacklevel=3,
                )
            n_total = int(np.prod(size)) if isinstance(size, tuple) else size
            samples_df = self.kernel.sample(n_total)
            samples = samples_df.values.ravel()
            if isinstance(size, tuple):
                return samples.reshape(size)
            return samples
        if self.kernel == "gaussian":
            return rng.standard_normal(size)
        elif self.kernel == "tophat":
            return rng.uniform(-1, 1, size)
        elif self.kernel == "epanechnikov":
            return self._rejection_sample(size, lambda u: 1 - u**2, rng)
        elif self.kernel == "cosine":
            return self._rejection_sample(
                size, lambda u: np.cos(np.pi * u / 2), rng
            )
        elif self.kernel == "linear":
            return rng.triangular(-1, 0, 1, size)
        else:
            raise ValueError(f"Unsupported kernel '{self.kernel}'.")

    @staticmethod
    def _rejection_sample(size, accept_fn, rng):
        """Rejection-sample from uniform(-1,1) with given acceptance function."""
        n_total = int(np.prod(size)) if isinstance(size, tuple) else size
        samples = np.empty(n_total)
        count = 0
        while count < n_total:
            proposal = rng.uniform(-1, 1, n_total - count)
            accept_prob = accept_fn(proposal)
            accepted = proposal[rng.random(n_total - count) < accept_prob]
            n_accept = min(len(accepted), n_total - count)
            samples[count : count + n_accept] = accepted[:n_accept]
            count += n_accept
        return samples.reshape(size) if isinstance(size, tuple) else samples

    def _kernel_variance(self):
        """Return the variance of the kernel function."""
        if self._kernel_mode == "distribution":
            var_val = self.kernel.var()
            if hasattr(var_val, "values"):
                return float(var_val.values.ravel()[0])
            return float(var_val)
        return _KERNEL_VARIANCE[self.kernel]

    def _mean(self):
        r"""Return expected value of the distribution.

        For a kernel mixture:
        :math:`\mathbb{E}[X] = \sum_i w_i x_i`
        """
        if self._support_mode == "shared":
            mean_val = np.sum(self._weights * self._support)
            if self.ndim > 0:
                return np.full(self.shape, mean_val)
            return mean_val

        # Per-location: compute a mean for each location
        n_locs = (
            len(self._support)
            if self._support_mode == "per_location_ragged"
            else self._support.shape[0]
        )
        means = np.array(
            [
                np.sum(self._weights_for(i) * self._support_for(i))
                for i in range(n_locs)
            ]
        )
        # Broadcast to (n_rows, n_cols) — all columns share the same row mean
        return np.broadcast_to(means[:, None], self.shape).copy()

    def _var(self):
        r"""Return element/entry-wise variance of the distribution.

        For a kernel mixture, by the law of total variance:
        :math:`\mathrm{Var}[X] = h^2 \mathrm{Var}[K] + \sum_i w_i (x_i - \mu)^2`
        """
        kernel_var = self._kernel_variance()

        if self._support_mode == "shared":
            h = self._bandwidth
            mean_val = np.sum(self._weights * self._support)
            weighted_var = np.sum(
                self._weights * (self._support - mean_val) ** 2
            )
            var_val = h**2 * kernel_var + weighted_var
            if self.ndim > 0:
                return np.full(self.shape, var_val)
            return var_val

        n_locs = (
            len(self._support)
            if self._support_mode == "per_location_ragged"
            else self._support.shape[0]
        )
        vars_ = np.empty(n_locs)
        for i in range(n_locs):
            h = self._bw_for(i)
            sup = self._support_for(i)
            w = self._weights_for(i)
            mu = np.sum(w * sup)
            vars_[i] = h**2 * kernel_var + np.sum(w * (sup - mu) ** 2)
        return np.broadcast_to(vars_[:, None], self.shape).copy()

    def _pdf_shared(self, x):
        """PDF evaluation for shared-support mode (original logic)."""
        h = self._bandwidth
        support = self._support
        weights = self._weights
        if self.ndim == 0:
            x_val = float(x)
            u = (x_val - support) / h
            return np.sum(weights * self._kernel_pdf(u)) / h
        x_flat = x.ravel()
        u = (x_flat[:, None] - support[None, :]) / h
        K = self._kernel_pdf(u)
        pdf_flat = np.sum(weights[None, :] * K, axis=1) / h
        return pdf_flat.reshape(x.shape)

    def _pdf_per_location(self, x):
        """PDF evaluation for per-location modes (2-D or ragged support)."""
        # x has shape (n_rows, n_cols); output should match
        out = np.empty_like(x, dtype=float)
        n_rows = x.shape[0]
        for i in range(n_rows):
            h = self._bw_for(i)
            sup = self._support_for(i)
            w = self._weights_for(i)
            row = x[i]  # shape (n_cols,)
            u = (row[:, None] - sup[None, :]) / h  # (n_cols, n_support)
            K = self._kernel_pdf(u)
            out[i] = np.sum(w[None, :] * K, axis=1) / h
        return out

    def _pdf(self, x):
        """Probability density function."""
        if self._support_mode == "shared":
            return self._pdf_shared(x)
        return self._pdf_per_location(x)

    def _log_pdf(self, x):
        """Logarithmic probability density function."""
        pdf_val = self._pdf(x)
        if np.isscalar(pdf_val):
            pdf_val = max(pdf_val, 1e-300)
        else:
            pdf_val = np.clip(pdf_val, 1e-300, None)
        return np.log(pdf_val)

    def _cdf_shared(self, x):
        """CDF evaluation for shared-support mode."""
        h = self._bandwidth
        support = self._support
        weights = self._weights
        if self.ndim == 0:
            x_val = float(x)
            u = (x_val - support) / h
            return np.sum(weights * self._kernel_cdf(u))
        x_flat = x.ravel()
        u = (x_flat[:, None] - support[None, :]) / h
        K_cdf = self._kernel_cdf(u)
        cdf_flat = np.sum(weights[None, :] * K_cdf, axis=1)
        return cdf_flat.reshape(x.shape)

    def _cdf_per_location(self, x):
        """CDF evaluation for per-location modes."""
        out = np.empty_like(x, dtype=float)
        n_rows = x.shape[0]
        for i in range(n_rows):
            h = self._bw_for(i)
            sup = self._support_for(i)
            w = self._weights_for(i)
            row = x[i]
            u = (row[:, None] - sup[None, :]) / h
            K_cdf = self._kernel_cdf(u)
            out[i] = np.sum(w[None, :] * K_cdf, axis=1)
        return out

    def _cdf(self, x):
        """Cumulative distribution function."""
        if self._support_mode == "shared":
            return self._cdf_shared(x)
        return self._cdf_per_location(x)

    def _sample_shared(self, n_samples):
        """Sampling for shared-support mode (original logic)."""
        rng = self._rng
        h = self._bandwidth
        support = self._support
        weights = self._weights
        n_draw = 1 if n_samples is None else n_samples

        if self.ndim == 0:
            idx = rng.choice(len(support), size=n_draw, p=weights)
            centers = support[idx]
            noise = self._kernel_sample(n_draw, rng)
            samples = centers + h * noise
            if n_samples is None:
                return float(samples[0])
            return pd.DataFrame(samples, columns=self.columns)

        n_rows, n_cols = self.shape
        total = n_draw * n_rows * n_cols
        idx = rng.choice(len(support), size=total, p=weights)
        centers = support[idx]
        noise = self._kernel_sample(total, rng)
        samples_flat = centers + h * noise
        samples = samples_flat.reshape(n_draw, n_rows, n_cols)
        if n_samples is None:
            return pd.DataFrame(
                samples[0], index=self.index, columns=self.columns
            )
        spl_index = pd.MultiIndex.from_product([range(n_draw), self.index])
        return pd.DataFrame(
            samples.reshape(n_draw * n_rows, n_cols),
            index=spl_index,
            columns=self.columns,
        )

    def _sample_per_location(self, n_samples):
        """Sampling for per-location modes."""
        rng = self._rng
        n_draw = 1 if n_samples is None else n_samples
        n_rows, n_cols = self.shape

        # samples array shape: (n_draw, n_rows, n_cols)
        samples = np.empty((n_draw, n_rows, n_cols))
        for i in range(n_rows):
            h = self._bw_for(i)
            sup = self._support_for(i)
            w = self._weights_for(i)
            n_total = n_draw * n_cols
            idx = rng.choice(len(sup), size=n_total, p=w)
            centers = sup[idx]
            noise = self._kernel_sample(n_total, rng)
            row_samples = (centers + h * noise).reshape(n_draw, n_cols)
            samples[:, i, :] = row_samples

        if n_samples is None:
            return pd.DataFrame(
                samples[0], index=self.index, columns=self.columns
            )
        spl_index = pd.MultiIndex.from_product([range(n_draw), self.index])
        return pd.DataFrame(
            samples.reshape(n_draw * n_rows, n_cols),
            index=spl_index,
            columns=self.columns,
        )

    def _sample(self, n_samples=None):
        """Sample from the distribution."""
        raise NotImplementedError("TODO")

    def _iloc(self, rowidx=None, colidx=None):
        """Subset distribution by integer row/column indices."""
        raise NotImplementedError("TODO")

    def _iat(self, rowidx=None, colidx=None):
        """Subset distribution to a single scalar element."""
        raise NotImplementedError("TODO")
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {
            "support": [0.0, 1.0, 2.0, 3.0, 4.0],
            "bandwidth": 0.5,
            "kernel": "gaussian",
        }
        params2 = {
            "support": [0.0, 1.0, 2.0, 3.0, 4.0],
            "bandwidth": 1.0,
            "kernel": "gaussian",
            "index": pd.RangeIndex(3),
            "columns": pd.Index(["a", "b"]),
        }
        params3 = {
            "support": [-1.0, 0.0, 1.0, 2.0],
            "bandwidth": 0.8,
            "kernel": "epanechnikov",
            "weights": [0.1, 0.4, 0.4, 0.1],
        }
        params4 = {
            "support": np.linspace(-2, 2, 20),
            "bandwidth": "scott",
            "kernel": "tophat",
        }
        params5 = {
            "support": [0.0, 1.0, 2.0],
            "bandwidth": 0.5,
            "kernel": "cosine",
            "index": pd.RangeIndex(2),
            "columns": pd.Index(["x"]),
        }
        params6 = {
            "support": [[0.0, 1.0, 2.0], [5.0, 6.0, 7.0]],
            "bandwidth": 0.5,
            "kernel": "gaussian",
            "index": pd.RangeIndex(2),
            "columns": pd.Index(["a"]),
        }
        params7 = {
            "support": [
                np.array([0.0, 1.0]),
                np.array([5.0, 6.0, 7.0]),
            ],
            "bandwidth": 0.5,
            "kernel": "gaussian",
            "index": pd.RangeIndex(2),
            "columns": pd.Index(["a"]),
        }
        return [params1, params2, params3, params4, params5, params6, params7]
