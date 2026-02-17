"""Tests for KernelMixture distribution."""

# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["amaydixit11"]

import numpy as np
import pandas as pd
import pytest

try:
    _trapezoid = np.trapezoid
except AttributeError:
    _trapezoid = np.trapz

from skbase.utils.dependencies import _check_soft_dependencies

from skpro.distributions.kernel_mixture import KernelMixture
from skpro.tests.test_switch import run_test_module_changed


def _km_2d(bw=0.5, kernel="gaussian"):
    """A 2-row, 1-column KernelMixture with 2-D support."""
    support = np.array([[0.0, 1.0, 2.0], [5.0, 6.0, 7.0]])
    return KernelMixture(
        support=support,
        bandwidth=bw,
        kernel=kernel,
        index=pd.RangeIndex(2),
        columns=pd.Index(["a"]),
    )


def _km_ragged(bw=0.5, kernel="gaussian"):
    """A 2-row, 1-column KernelMixture with ragged support."""
    support = [np.array([0.0, 1.0, 2.0]), np.array([5.0, 6.0, 7.0, 8.0])]
    return KernelMixture(
        support=support,
        bandwidth=bw,
        kernel=kernel,
        index=pd.RangeIndex(2),
        columns=pd.Index(["a"]),
    )



@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
class TestKernelMixture:
    """Tests for KernelMixture distribution."""

    @pytest.fixture
    def simple_km(self):
        """Simple Gaussian kernel mixture for testing."""
        support = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        return KernelMixture(support=support, bandwidth=0.5, kernel="gaussian")

    @pytest.fixture
    def weighted_km(self):
        """Weighted kernel mixture for testing."""
        support = np.array([-1.0, 0.0, 1.0])
        weights = np.array([0.25, 0.5, 0.25])
        return KernelMixture(
            support=support,
            bandwidth=0.5,
            kernel="gaussian",
            weights=weights,
        )

    @pytest.mark.parametrize(
        "kernel", ["gaussian", "epanechnikov", "tophat", "cosine", "linear"]
    )
    def test_pdf_integrates_to_one(self, kernel):
        """Test pdf integrates to 1."""
        support = np.array([0.0, 1.0, 2.0, 3.0])
        km = KernelMixture(support=support, bandwidth=0.5, kernel=kernel)
        xs = np.linspace(-5, 8, 10000)
        pdfs = np.array([km.pdf(x) for x in xs])
        integral = _trapezoid(pdfs, xs)
        assert abs(integral - 1.0) < 0.01

    def test_mean_correctness(self, simple_km):
        """Test mean equals weighted average of support."""
        assert abs(simple_km.mean() - 2.0) < 1e-10

    def test_weighted_mean(self, weighted_km):
        """Test weighted mixture mean."""
        assert abs(weighted_km.mean()) < 1e-10

    def test_var_correctness(self, simple_km):
        """Test variance formula."""
        h = 0.5
        support = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        expected_var = h**2 * 1.0 + np.var(support)
        assert abs(simple_km.var() - expected_var) < 1e-10

    def test_cdf_monotonicity(self, simple_km):
        """Test CDF is non-decreasing."""
        xs = np.linspace(-3, 7, 200)
        cdfs = np.array([simple_km.cdf(x) for x in xs])
        assert np.all(np.diff(cdfs) >= -1e-10)

    def test_cdf_limits(self, simple_km):
        """Test CDF limits."""
        assert simple_km.cdf(-20.0) < 1e-6
        assert simple_km.cdf(25.0) > 1 - 1e-6

    def test_sample_shape_scalar(self, simple_km):
        """Test scalar sample shape."""
        s = simple_km.sample()
        assert np.isscalar(s)
        s_multi = simple_km.sample(10)
        assert isinstance(s_multi, pd.DataFrame)
        assert s_multi.shape[0] == 10

    def test_sample_shape_2d(self):
        """Test 2D sample shape."""
        km = KernelMixture(
            support=[0.0, 1.0, 2.0],
            bandwidth=0.5,
            kernel="gaussian",
            index=pd.RangeIndex(3),
            columns=pd.Index(["a", "b"]),
        )
        s = km.sample()
        assert s.shape == (3, 2)
        s_multi = km.sample(5)
        assert s_multi.shape == (15, 2)

    def test_invalid_kernel_raises(self):
        """Test invalid kernel raises."""
        with pytest.raises(ValueError, match="Unknown kernel"):
            KernelMixture(support=[0, 1], bandwidth=1.0, kernel="invalid")

    def test_weight_length_mismatch_raises(self):
        """Test mismatched weights raises."""
        with pytest.raises(ValueError, match="weights length"):
            KernelMixture(support=[0, 1, 2], bandwidth=1.0, weights=[0.5, 0.5])

    def test_non_positive_bandwidth_raises(self):
        """Test non-positive bandwidth raises."""
        with pytest.raises(ValueError, match="bandwidth must be positive"):
            KernelMixture(support=[0, 1, 2], bandwidth=0.0)
        with pytest.raises(ValueError, match="bandwidth must be positive"):
            KernelMixture(support=[0, 1, 2], bandwidth=-1.0)

    def test_negative_weights_raises(self):
        """Test negative weights raise."""
        with pytest.raises(ValueError, match="non-negative"):
            KernelMixture(support=[0, 1, 2], bandwidth=1.0, weights=[1.0, -0.5, 0.5])

    def test_log_pdf_consistency(self, simple_km):
        """Test log_pdf consistency with pdf."""
        for x in [0.0, 1.0, 2.0, 3.0]:
            assert abs(simple_km.log_pdf(x) - np.log(simple_km.pdf(x))) < 1e-10

    @pytest.mark.parametrize(
        "kernel", ["gaussian", "epanechnikov", "tophat", "cosine", "linear"]
    )
    def test_all_kernels_basic(self, kernel):
        """Test basic functionality for all kernels."""
        km = KernelMixture(support=[0.0, 1.0, 2.0], bandwidth=0.5, kernel=kernel)
        assert np.isfinite(km.mean())
        assert km.var() > 0
        assert km.pdf(1.0) > 0
        assert 0 <= km.cdf(1.0) <= 1
        assert np.isfinite(km.sample())

    def test_cdf_pdf_consistency(self, simple_km):
        """Test CDF derivative matches PDF."""
        xs = np.linspace(-2, 6, 50)
        eps = 1e-5
        for x in xs:
            pdf_val = simple_km.pdf(x)
            cdf_deriv = (simple_km.cdf(x + eps) - simple_km.cdf(x - eps)) / (2 * eps)
            assert abs(pdf_val - cdf_deriv) < 1e-3

    def test_sample_mean_convergence(self, simple_km):
        """Test sample mean convergence."""
        samples = simple_km.sample(10000)
        sample_mean = samples.values.mean()
        assert abs(sample_mean - simple_km.mean()) < 0.1

    def test_random_state_reproducibility(self):
        """Test random_state reproducibility."""
        support = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        km1 = KernelMixture(
            support=support, bandwidth=0.5, kernel="gaussian", random_state=42
        )
        km2 = KernelMixture(
            support=support, bandwidth=0.5, kernel="gaussian", random_state=42
        )
        s1 = km1.sample(100)
        s2 = km2.sample(100)
        np.testing.assert_array_equal(s1.values, s2.values)

    def test_auto_bandwidth_scott(self):
        """Test Scott bandwidth rule."""
        support = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        km = KernelMixture(support=support, bandwidth="scott", kernel="gaussian")
        expected = len(support) ** (-1.0 / 5.0) * np.std(support, ddof=1)
        assert abs(km._bandwidth - expected) < 1e-10

    def test_auto_bandwidth_silverman(self):
        """Test Silverman bandwidth rule."""
        support = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        km = KernelMixture(support=support, bandwidth="silverman", kernel="gaussian")
        expected = (4.0 / (3.0 * 5)) ** (1.0 / 5.0) * np.std(support, ddof=1)
        assert abs(km._bandwidth - expected) < 1e-10

    def test_subsetting_2d(self):
        """Test iloc subsetting."""
        km = KernelMixture(
            support=[0.0, 1.0, 2.0],
            bandwidth=0.5,
            kernel="gaussian",
            index=pd.RangeIndex(3),
            columns=pd.Index(["a", "b"]),
        )
        sub = km.iloc[[0, 1], [0]]
        assert sub.shape == (2, 1)
        sub_scalar = km.iloc[0, 0]
        assert sub_scalar.shape == ()

    @pytest.mark.parametrize("rule", ["scott", "silverman"])
    def test_auto_bandwidth_single_element(self, rule):
        """Test bandwidth with single support point."""
        km = KernelMixture(support=[5.0], bandwidth=rule, kernel="gaussian")
        assert np.isfinite(km._bandwidth)
        assert km._bandwidth > 0

    def test_invalid_kernel_type_raises(self):
        """Test non-string/distribution kernel raises."""
        with pytest.raises(TypeError, match="kernel must be a string"):
            KernelMixture(support=[0, 1], bandwidth=1.0, kernel=42)

    def test_non_scalar_kernel_raises(self):
        """Test non-scalar distribution kernel raises."""
        from skpro.distributions.normal import Normal

        kernel_2d = Normal(
            mu=[[0, 0]],
            sigma=[[1, 1]],
            index=pd.RangeIndex(1),
            columns=pd.Index(["a", "b"]),
        )
        with pytest.raises(ValueError, match="scalar"):
            KernelMixture(support=[0, 1], bandwidth=1.0, kernel=kernel_2d)

    def test_nonzero_mean_kernel_warns(self):
        """Test non-zero mean kernel warns."""
        from skpro.distributions.normal import Normal

        with pytest.warns(UserWarning, match="non-zero mean"):
            KernelMixture(
                support=[0, 1, 2], bandwidth=0.5, kernel=Normal(mu=5, sigma=1)
            )

    def test_distribution_kernel_rng_warns(self):
        """Test distribution kernel RNG warning."""
        from skpro.distributions.normal import Normal

        km = KernelMixture(
            support=[0, 1, 2],
            bandwidth=0.5,
            kernel=Normal(mu=0, sigma=1),
            random_state=42,
        )
        with pytest.warns(UserWarning, match="random_state"):
            km.sample(10)

    def test_distribution_kernel_pdf(self):
        """Test Normal kernel matches gaussian."""
        from skpro.distributions.normal import Normal

        support = np.array([0.0, 1.0, 2.0])
        bw = 0.5
        km_str = KernelMixture(support=support, bandwidth=bw, kernel="gaussian")
        km_dist = KernelMixture(
            support=support, bandwidth=bw, kernel=Normal(mu=0, sigma=1)
        )
        for x in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
            assert abs(km_str.pdf(x) - km_dist.pdf(x)) < 1e-6

    def test_distribution_kernel_cdf(self):
        """Test Normal kernel CDF matches gaussian."""
        from skpro.distributions.normal import Normal

        support = np.array([0.0, 1.0, 2.0])
        bw = 0.5
        km_str = KernelMixture(support=support, bandwidth=bw, kernel="gaussian")
        km_dist = KernelMixture(
            support=support, bandwidth=bw, kernel=Normal(mu=0, sigma=1)
        )
        for x in [-1.0, 0.0, 1.0, 2.0, 3.0]:
            assert abs(km_str.cdf(x) - km_dist.cdf(x)) < 1e-6

    def test_distribution_kernel_mean_var(self):
        """Test Normal kernel mean/var matches gaussian."""
        from skpro.distributions.normal import Normal

        support = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        bw = 0.5
        km_str = KernelMixture(support=support, bandwidth=bw, kernel="gaussian")
        km_dist = KernelMixture(
            support=support, bandwidth=bw, kernel=Normal(mu=0, sigma=1)
        )
        assert abs(km_str.mean() - km_dist.mean()) < 1e-10
        assert abs(km_str.var() - km_dist.var()) < 1e-10

    def test_distribution_kernel_sample(self):
        """Test distribution kernel sampling."""
        from skpro.distributions.normal import Normal

        km = KernelMixture(
            support=[0.0, 1.0, 2.0],
            bandwidth=0.5,
            kernel=Normal(mu=0, sigma=1),
        )
        assert np.isfinite(km.sample())
        s_multi = km.sample(10)
        assert isinstance(s_multi, pd.DataFrame)
        assert s_multi.shape[0] == 10

    @pytest.mark.skipif(
        not _check_soft_dependencies("sklearn", severity="none"),
        reason="sklearn not available",
    )
    def test_sklearn_parity(self):
        """Test parity with sklearn KernelDensity."""
        from sklearn.neighbors import KernelDensity

        support = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        bw = 0.5
        km = KernelMixture(support=support, bandwidth=bw, kernel="gaussian")
        kde = KernelDensity(bandwidth=bw, kernel="gaussian")
        kde.fit(support.reshape(-1, 1))
        xs = np.linspace(-2, 6, 50)
        for x in xs:
            skpro_pdf = km.pdf(x)
            sklearn_pdf = np.exp(kde.score_samples(np.array([[x]]))[0])
            assert abs(skpro_pdf - sklearn_pdf) < 1e-6

    # ---------------------------------------------------------------------- #
    # 2D / Ragged Support Tests
    # ---------------------------------------------------------------------- #

    def test_shared_mode_detected(self):
        """1-D support → shared mode (backward-compat)."""
        km = KernelMixture(support=[0.0, 1.0, 2.0], bandwidth=0.5)
        assert km._support_mode == "shared"
        assert km._support.ndim == 1

    def test_2d_mode_detected_from_array(self):
        """2-D numpy array → per_location_2d mode."""
        km = _km_2d()
        assert km._support_mode == "per_location_2d"
        assert km._support.shape == (2, 3)

    def test_2d_mode_detected_from_nested_list(self):
        """Nested list of equal-length rows → per_location_2d mode."""
        km = KernelMixture(
            support=[[0.0, 1.0], [5.0, 6.0]],
            bandwidth=0.5,
            index=pd.RangeIndex(2),
            columns=pd.Index(["a"]),
        )
        assert km._support_mode == "per_location_2d"

    def test_ragged_mode_detected(self):
        """List of unequal-length arrays → per_location_ragged mode."""
        km = _km_ragged()
        assert km._support_mode == "per_location_ragged"
        assert isinstance(km._support, list)
        assert len(km._support[0]) == 3
        assert len(km._support[1]) == 4

    @pytest.mark.parametrize("mode", ["2d"])
    def test_mean_per_location(self, mode):
        """Mean for each row equals the weighted average of that row's support."""
        km = _km_2d() if mode == "2d" else _km_ragged()
        means = km.mean()
        # Row 0 support: [0, 1, 2], uniform → mean = 1.0
        assert abs(means.iloc[0, 0] - 1.0) < 1e-10
        # Row 1 support: [5, 6, 7] or [5, 6, 7, 8]
        expected_row1 = 6.0 if mode == "2d" else 6.5
        assert abs(means.iloc[1, 0] - expected_row1) < 1e-10

    @pytest.mark.parametrize("mode", ["2d"])
    def test_var_per_location_positive(self, mode):
        """Variance is positive and finite for each location."""
        km = _km_2d() if mode == "2d" else _km_ragged()
        variances = km.var()
        assert np.all(variances.values > 0)
        assert np.all(np.isfinite(variances.values))

    def test_var_law_of_total_variance_2d(self):
        """Variance formula: h^2 * kernel_var + weighted_variance_of_support."""
        km = _km_2d(bw=0.5, kernel="gaussian")
        variances = km.var()
        h = 0.5
        kernel_var_gaussian = 1.0

        # Row 0: support [0, 1, 2], uniform weights
        sup0 = np.array([0.0, 1.0, 2.0])
        mu0 = sup0.mean()
        expected_var0 = h**2 * kernel_var_gaussian + np.mean((sup0 - mu0) ** 2)
        assert abs(variances.iloc[0, 0] - expected_var0) < 1e-10

    @pytest.mark.parametrize("mode", ["2d"])
    def test_pdf_positive_at_support_center(self, mode):
        """PDF should be positive at the center of each location's support."""
        km = _km_2d() if mode == "2d" else _km_ragged()
        # Center of row 0 is ~1.0, center of row 1 is ~6.0 (or 6.5 ragged)
        x = pd.DataFrame({"a": [1.0, 6.0]}, index=pd.RangeIndex(2))
        pdf_vals = km.pdf(x)
        assert np.all(pdf_vals.values > 0)

    @pytest.mark.parametrize("mode", ["2d"])
    def test_pdf_near_zero_far_from_support(self, mode):
        """PDF should be near zero far from either location's support."""
        km = _km_2d() if mode == "2d" else _km_ragged()
        # x = 50 is far from both [0,1,2] and [5,6,7/8]
        x = pd.DataFrame({"a": [50.0, 50.0]}, index=pd.RangeIndex(2))
        pdf_vals = km.pdf(x)
        assert np.all(pdf_vals.values < 1e-6)

    @pytest.mark.parametrize("mode", ["2d"])
    def test_pdf_integrates_per_row(self, mode):
        """Each row's marginal PDF integrates to ~1."""
        km = _km_2d() if mode == "2d" else _km_ragged()

        for row_i, (lo, hi) in enumerate([(-4, 7), (1, 12)]):
            xs = np.linspace(lo, hi, 5000)
            # Build a single-row km for that location
            sup_i = km._support_for(row_i)
            w_i = km._weights_for(row_i)
            bw_i = km._bw_for(row_i)
            km_i = KernelMixture(
                support=sup_i,
                bandwidth=bw_i,
                kernel=km.kernel,
                weights=w_i,
            )
            pdfs = np.array([km_i.pdf(x) for x in xs])
            integral = _trapezoid(pdfs, xs)
            assert abs(integral - 1.0) < 0.02, (
                f"Row {row_i} PDF integral = {integral:.4f}, expected ~1.0"
            )

    @pytest.mark.parametrize("mode", ["2d"])
    def test_cdf_monotone_per_row(self, mode):
        """Each row's CDF is non-decreasing."""
        km = _km_2d() if mode == "2d" else _km_ragged()
        for row_i, (lo, hi) in enumerate([(-3, 6), (2, 12)]):
            xs = np.linspace(lo, hi, 200)
            sup_i = km._support_for(row_i)
            w_i = km._weights_for(row_i)
            bw_i = km._bw_for(row_i)
            km_i = KernelMixture(
                support=sup_i, bandwidth=bw_i,
                kernel=km.kernel, weights=w_i,
            )
            cdfs = np.array([km_i.cdf(x) for x in xs])
            assert np.all(np.diff(cdfs) >= -1e-10), (
                f"CDF non-monotone for row {row_i}"
            )

    @pytest.mark.parametrize("mode", ["2d"])
    def test_sample_shape(self, mode):
        """Single draw has shape (n_rows, n_cols)."""
        km = _km_2d() if mode == "2d" else _km_ragged()
        s = km.sample()
        assert s.shape == (2, 1)

    @pytest.mark.parametrize("mode", ["2d"])
    def test_sample_multi_shape(self, mode):
        """n_samples draws has shape (n_samples * n_rows, n_cols)."""
        km = _km_2d() if mode == "2d" else _km_ragged()
        s = km.sample(5)
        assert s.shape == (10, 1)

    @pytest.mark.parametrize("mode", ["2d"])
    def test_sample_row0_near_support0(self, mode):
        """Samples for row 0 should cluster near [0, 1, 2]."""
        km = _km_2d() if mode == "2d" else _km_ragged()
        s = km.sample(2000)
        # Row 0 indices in multi-index: every other starting from 0
        row0_vals = s.xs(0, level=1).values.ravel()
        assert row0_vals.mean() < 4.0  # nowhere near [5-8]

    @pytest.mark.parametrize("mode", ["2d"])
    def test_sample_row1_near_support1(self, mode):
        """Samples for row 1 should cluster near [5, 6, 7] or [5-8]."""
        km = _km_2d() if mode == "2d" else _km_ragged()
        s = km.sample(2000)
        row1_vals = s.xs(1, level=1).values.ravel()
        assert row1_vals.mean() > 3.0  # nowhere near [0, 1, 2]

    def test_per_location_weights_2d(self):
        """2-D weights array is correctly parsed and normalised."""
        support = np.array([[0.0, 1.0, 2.0], [5.0, 6.0, 7.0]])
        weights = np.array([[0.1, 0.3, 0.6], [0.5, 0.3, 0.2]])
        km = KernelMixture(
            support=support,
            bandwidth=0.5,
            weights=weights,
            index=pd.RangeIndex(2),
            columns=pd.Index(["a"]),
        )
        # Row 0 mean = 0*0.1 + 1*0.3 + 2*0.6 = 1.5
        assert abs(km.mean().iloc[0, 0] - 1.5) < 1e-10
        # Row 1 mean = 5*0.5 + 6*0.3 + 7*0.2 = 5.7
        assert abs(km.mean().iloc[1, 0] - 5.7) < 1e-10

    def test_per_location_weights_ragged(self):
        """List of per-location weight arrays is parsed and normalised."""
        support = [np.array([0.0, 1.0, 2.0]), np.array([5.0, 6.0, 7.0, 8.0])]
        weights = [np.array([0.1, 0.3, 0.6]), np.array([0.25, 0.25, 0.25, 0.25])]
        km = KernelMixture(
            support=support,
            bandwidth=0.5,
            weights=weights,
            index=pd.RangeIndex(2),
            columns=pd.Index(["a"]),
        )
        # Row 0 mean = 0*0.1 + 1*0.3 + 2*0.6 = 1.5
        assert abs(km.mean().iloc[0, 0] - 1.5) < 1e-10
        # Row 1 mean = uniform over [5,6,7,8] = 6.5
        assert abs(km.mean().iloc[1, 0] - 6.5) < 1e-10

    def test_1d_weights_with_2d_support_raises(self):
        """1-D weights with 2-D support is ambiguous and must raise."""
        with pytest.raises(ValueError, match="ambiguous"):
            KernelMixture(
                support=[[0.0, 1.0], [5.0, 6.0]],
                bandwidth=0.5,
                weights=[0.5, 0.5],
                index=pd.RangeIndex(2),
                columns=pd.Index(["a"]),
            )

    def test_weight_row_mismatch_raises(self):
        """2-D weights with wrong row count raises."""
        with pytest.raises(ValueError, match="row count"):
            KernelMixture(
                support=np.array([[0.0, 1.0], [5.0, 6.0]]),
                bandwidth=0.5,
                weights=np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]),
                index=pd.RangeIndex(2),
                columns=pd.Index(["a"]),
            )

    def test_scott_per_location(self):
        """Scott rule is applied independently per row."""
        support = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                            [10.0, 20.0, 30.0, 40.0, 50.0]])
        km = KernelMixture(
            support=support,
            bandwidth="scott",
            index=pd.RangeIndex(2),
            columns=pd.Index(["a"]),
        )
        assert isinstance(km._bandwidth, np.ndarray)
        assert len(km._bandwidth) == 2
        # Row 1 has 10× larger std than row 0 → bandwidth should be ~10× larger
        assert km._bandwidth[1] > km._bandwidth[0] * 5

    def test_iloc_subset_2d(self):
        """iloc on a 2-D KernelMixture yields a correctly-scoped sub-distribution."""
        km = _km_2d()
        sub = km.iloc[[0], :]
        assert sub.shape == (1, 1)
        # sub should have only row-0 support
        assert sub._support_mode == "per_location_2d"
        assert sub._support.shape[0] == 1
        # Mean of sub should match row 0 mean of original
        assert abs(sub.mean().iloc[0, 0] - km.mean().iloc[0, 0]) < 1e-10

    def test_iloc_subset_ragged(self):
        """iloc on a ragged KernelMixture yields correct sub-distribution."""
        km = _km_ragged()
        sub = km.iloc[[1], :]
        assert sub.shape == (1, 1)
        assert sub._support_mode == "per_location_2d"
        assert len(sub._support[0]) == 4  # row 1 had 4 support points
        assert abs(sub.mean().iloc[0, 0] - km.mean().iloc[1, 0]) < 1e-10

    def test_iat_returns_scalar_km_2d(self):
        """_iat on a 2-D KernelMixture returns a scalar KernelMixture."""
        pass # TODO

    def test_shared_support_unchanged(self):
        """Existing 1-D shared-support API is completely unchanged."""
        support = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        km = KernelMixture(support=support, bandwidth=0.5, kernel="gaussian")
        assert km._support_mode == "shared"
        assert abs(km.mean() - 2.0) < 1e-10
        assert km.var() > 0
        assert km.pdf(2.0) > 0
        assert 0 < km.cdf(2.0) < 1
        assert np.isfinite(km.sample())

    def test_shared_2d_dist_shape(self):
        """Shared support with 2-D index/columns still works (existing test)."""
        km = KernelMixture(
            support=[0.0, 1.0, 2.0],
            bandwidth=0.5,
            kernel="gaussian",
            index=pd.RangeIndex(3),
            columns=pd.Index(["a", "b"]),
        )
        s = km.sample()
        assert s.shape == (3, 2)
