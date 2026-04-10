---
tags: [esoc-interview, faq, expected-questions, skpro]
created: 2026-04-10
---

# Interview FAQ & Expected Questions

## Format

- **Duration:** 30-60 minutes
- **Structure:**
  1. Your 5-min code presentation (screen-share)
  2. Technical Q&A
  3. Non-technical / motivation Q&A
  4. Your questions to them
- **Interviewer:** Dr Franz Király (fkiraly) — UCL Statistics PhD, director at GC.OS, creator of sktime/skpro

---

## Technical Questions — Categorized

### A. On Your Proposal

| Question | Key Points to Cover |
|----------|-------------------|
| "Why a mixin instead of putting update() in BaseProbaRegressor?" | `update()` already exists (PR #462, by fkiraly). Mixin adds stream semantics without modifying base. Purely additive. |
| "What currently exists for online learning?" | OnlineRefit, OnlineRefitEveryN, OnlineDontRefit (all batch-refit). OndilOnlineGamlss (one external wrapper). No native incrementals. |
| "Is the scope realistic for 12 weeks?" | Pillars 1-2 are core (done as prototypes). 3-4 are stretch. Can deprioritize River bridge if needed. |
| "Your SGD uses a Python for-loop — why not vectorize?" | Intentional for online semantics — each sample updates immediately. Can add vectorized batch mode as option. |
| "OnlineGLMRegressor with warm-starting is essentially OnlineRefit" | Statsmodels GLM has no true online update. IRLS warm-start is an approximation. May deprioritize this estimator. |
| "DriftDetectorRegressor depends on external libraries" | ADWIN/Page-Hinkley from river or scikit-multiflow. These are optional deps. Can implement simple rolling-mean detector as default. |
| "Why not contribute to the existing Ondil adapter instead?" | Ondil is one specific model (GAMLSS). My proposal is general-purpose — works with any skpro regressor. |

### B. On Your PRs

#### PR #718 (GLM Dispersion Fix)

| Question | Answer |
|----------|--------|
| "Walk us through the GLM theory — what is the dispersion parameter ϕ?" | In GLM: Var(Y\|X) = ϕ·V(μ). ϕ controls observation noise, V(μ) is the variance function. `mean_se` estimates uncertainty in the mean coefficient, not observation noise. |
| "How did you verify that mean_se shrinks as O(1/√N)?" | Simulation: generated data with known variance, fit GLM with increasing N, plotted mean_se vs N. Showed 1/√N decay. |
| "Why doesn't this affect the Poisson family?" | Poisson has fixed variance function V(μ) = μ. The dispersion ϕ = 1 by definition. No separate scale parameter to fix. |

#### PR #721 (KernelMixture)

| Question | Answer |
|----------|--------|
| "Why did you share support points across all marginals in 2D?" | Simplicity and vectorization. Per-location supports would require O(N×M) storage. Clean type-dispatch path exists for future per-location supports. |
| "How does vectorization work for CDF computation?" | For each kernel center, compute CDF contribution to all evaluation points simultaneously via broadcasting. Sum across kernels. |
| "What's the computational complexity vs sklearn's KernelDensity?" | Same O(N·M) for N support points, M evaluation points. But sklearn uses ball-tree/Barnes-Hut for large N, while mine is brute-force vectorized. |
| "Why did fkiraly ask you to rename bandwidth to h?" | Statistical convention — bandwidth is denoted h in the literature. Consistency with notation. |

#### PR #735 (HistogramCDERegressor)

| Question | Answer |
|----------|--------|
| "Why KNN neighbourhoods instead of fixed bins?" | KNN adapts to local data density. Fixed bins would be sparse in some regions, dense in others. KNN ensures each neighbourhood has enough data for a meaningful histogram. |
| "What's the status?" | Open, awaiting review. CI lint check failing — need to run pre-commit. Baseline implementation complete. |

### C. General Python / Data Science

| Topic | Likely Questions |
|-------|-----------------|
| **OOP patterns** | Mixin vs inheritance, delegation pattern, factory pattern |
| **Numpy** | Broadcasting rules, ufuncs, vectorization vs loops |
| **Pandas** | DataFrame manipulation, MultiIndex, groupby |
| **sklearn API** | fit/predict convention, tags, `get_test_params`, `clone()` |
| **Probability theory** | PDF vs CDF vs PPF, KL divergence, proper scoring rules |
| **Statistics** | Bias-variance tradeoff, consistency, MLE, dispersion |

### D. Non-Technical

| Question | Preparation |
|----------|------------|
| "Why skpro? Why ESoC?" | Passion for probabilistic ML, open-source experience, skpro's architecture impressed you |
| "Describe your experience with large codebases" | MOSIP, OSDAG, skpro — all large, multi-contributor repos |
| "How do you handle code review feedback?" | Iterative, responsive, ask clarifying questions, don't take it personally |
| "What do you do when stuck on a bug?" | Reproduce, isolate, search docs/issues, ask on Discord, write minimal test case |
| "Tell us about OpenLake coordination" | Leadership, mentoring, organizing workshops — shows you can work in a team |
| "How will you manage 12 weeks full-time?" | 3rd year, summer break, flexible schedule, already contributing to skpro |

---

## Questions to Ask Them

1. "What would success look like for this project at the mid-point review?"
2. "Are there any existing skpro issues or design discussions around online learning that I should read before starting?"
3. "How does the skpro team envision the relationship between this project and the broader sktime forecasting ecosystem?"
4. "What's the mentoring structure — will I have a dedicated weekly 1-on-1?"

---

## Presentation — 5-Min Walkthrough

**Recommended:** PR #721 (KernelMixture)

**Script outline:**
1. **Problem (30 sec):** skpro needed a nonparametric KDE distribution. Existing distributions are all parametric.
2. **Solution (2 min):** KernelMixture — 5 kernel types, Scott/Silverman bandwidth, exact pdf/cdf/mean/var/sample, sklearn parity tests.
3. **How (2 min):** Vectorized computations, custom BaseDistribution kernels, parameter broadcasting, 852 lines, 39 CI checks.
4. **Impact (30 sec):** Added directly to API reference, enables nonparametric probabilistic prediction.

**Files to show:**
- `skpro/distributions/kernel_mixture.py` — main implementation
- `skpro/distributions/tests/test_kernel_mixture.py` — test coverage

---

## Related

- [[MOC]]
- [[05-online-branch-code-review]]
- [[09-pr-718-glm-dispersion-fix]]
- [[10-pr-721-kernelmixture]]
