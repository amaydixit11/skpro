---
tags: [MOC, esoc, interview, skpro]
created: 2026-04-10
updated: 2026-04-10
status: active
---

# ESoC Interview 2026 — Map of Content

**Interview Date:** April 14, 2026 at 14:20 UTC  
**Interviewer:** Dr Franz Király (fkiraly)  
**Organisation:** skpro / sktime / GC.OS  
**Proposal:** Online/Stream Learning API for skpro

---

## 📋 Interview Requirements

- [ ] 5-minute code presentation (screen-share, no slides)
- [ ] Explain problem solved + how code solves it
- [ ] Structured Q&A (technical + non-technical)
- [ ] 10 min early arrival, audio/video tested
- [ ] Prepare questions to ask them

## 📚 Session Notes

| Session | Topic | Status |
|---------|-------|--------|
| 1 | [[01-skpro-class-hierarchy\|skpro Class Hierarchy]] | ✅ Done |
| 2 | [[02-base-distribution-deep-dive\|BaseDistribution Deep Dive]] | ✅ Done |
| 3 | [[03-base-proba-regressor-deep-dive\|BaseProbaRegressor Deep Dive]] | ✅ Done |
| 4 | [[04-online-learning-gap\|The Online Learning Gap in skpro]] | ✅ Done |
| 5 | [[05-online-branch-code-review\|Code Review: Your `online` Branch]] | ✅ Done |
| 6 | Probabilistic Regression Fundamentals | 🔄 Pending |
| 7 | Online Learning Algorithms (SGD, RLS, GLM) | 🔄 Pending |
| 8 | Metrics & Scoring Rules (CRPS, Log-Loss, Pinball) | 🔄 Pending |
| 9 | PR #718 Deep Review (GLM Dispersion Fix) | 🔄 Pending |
| 10 | PR #721 Deep Review (KernelMixture) | 🔄 Pending |
| 11 | PR #735 Deep Review (HistogramCDERegressor) | 🔄 Pending |
| 12 | Proposal Defense | 🔄 Pending |
| 13 | Mock Interview Q&A | 🔄 Pending |

## 🔑 Key Facts

- **Stipend:** €4,800 (pro-rated, 12 weeks FTE + 5 days off)
- **Hub:** German Center for Open Source AI (GC.OS)
- **Mentoring:** Weekly 1-on-1, rapid feedback (≤2 days), peer review required
- **Your PRs:** 2 merged (#718, #721), 1 open (#735)
- **Your Branch:** `online` — contains SGD, RLS, SlidingWindow, OnlineRegressorMixin

## ⚠️ Critical Weak Spots to Address

1. SGD/RLS already implemented on branch — how to frame remaining work
2. RLS `predict_proba` gives same variance for all predictions (bug)
3. No `get_test_params()` on any online branch class
4. No input validation in `OnlineRegressorMixin.update()`
5. SGD uses Python for-loop — will be questioned on performance
6. OnlineGLMRegressor essentially = OnlineRefit with warm start
7. Missing ForgettingFactorRegressor and DriftDetectorRegressor entirely
8. No prequential evaluation framework exists

## 🎯 Presentation Choice

Best option: **PR #721 (KernelMixture)** — merged, 852 lines, 39 CI checks, added to API reference
