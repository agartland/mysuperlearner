# R SuperLearner vs Python mysuperlearner Algorithm Comparison

## Executive Summary

This document provides a detailed comparison of the CV.SuperLearner algorithm between the R `SuperLearner` package and the Python `mysuperlearner` package. After thorough analysis and implementation updates, **both implementations now have consistent algorithms** with the same key features.

## Algorithm Overview

### CV.SuperLearner Double Cross-Validation

Both R and Python implement the same nested (double) cross-validation structure:

```
For each outer fold i = 1...V:
  1. Split data: training fold (outer train) and validation fold (outer test)
  2. Fit SuperLearner on outer training fold:
     a. Inner CV: Split outer training fold into inner CV folds
     b. Build Z matrix from inner CV predictions
     c. Train meta-learner on Z to get coefficients
     d. Refit base learners on full outer training fold
  3. Predict on outer test fold using refitted base learners
  4. Combine predictions with meta-learner weights
  5. Select discrete SL (best base learner by inner CV risk)
  6. Record: predictions, coefficients, CV risks, discrete SL selection
```

## Detailed Component Comparison

### 1. Inner CV for Meta-Learner Training ✅ CONSISTENT

**R Implementation** (`SuperLearner.R` lines 166-172):
```r
crossValFUN_out <- lapply(validRows, FUN = .crossValFUN,
                          Y = Y, dataX = X, id = id,
                          obsWeights = obsWeights,
                          library = library, kScreen = kScreen,
                          k = k, p = p, libraryNames = libraryNames,
                          saveCVFitLibrary = control$saveCVFitLibrary)
Z[unlist(validRows, use.names = FALSE), ] <- do.call('rbind', lapply(crossValFUN_out, "[[", "out"))
```

**Python Implementation** (`super_learner.py` lines 244-257):
```python
Z, cv_preds, fold_indices, cv_risks = self._build_level1(X_arr, y_arr, base_learners,
                                                   cv=self.cv, random_state=self.random_state,
                                                   sample_weight=sample_weight, groups=groups)
self.Z_ = Z
self.cv_risks_ = cv_risks
```

**Verdict**: Both implementations create the Z matrix via inner cross-validation. The meta-learner is trained on this Z matrix.

---

### 2. Base Learner Refit on Full Outer Training Data ✅ CONSISTENT

**R Implementation** (`SuperLearner.R` lines 217-242):
After computing coefficients, R refits base learners on full training data and uses these to make predictions on `newX`.

**Python Implementation** (`super_learner.py` lines 297-313):
```python
# Refit base learners on full data with comprehensive error handling
self.base_learners_full_ = []
for name, estimator in base_learners:
    mdl = clone(estimator)
    mdl.fit(X_arr, y_arr, sample_weight=sample_weight)
    self.base_learners_full_.append((name, mdl))
```

**Verdict**: Both implementations refit base learners on the full outer training fold before making predictions on the outer test fold.

---

### 3. CV Risk Computation ✅ NOW CONSISTENT

**R Implementation** (`SuperLearner.R`):
R computes `cvRisk` as the mean squared error between CV predictions (Z matrix) and true outcomes.

**Python Implementation** (`super_learner.py` lines 191-199):
```python
# Compute CV risk for each learner (mean squared error on CV predictions)
cv_risks = np.zeros(K)
for j in range(K):
    valid_mask = ~np.isnan(Z[:, j])
    if valid_mask.sum() > 0:
        cv_risks[j] = mean_squared_error(y_arr[valid_mask], Z[valid_mask, j])
    else:
        cv_risks[j] = np.inf
```

**Verdict**: ✅ **IMPLEMENTED** - Python now computes and stores CV risk matching R's calculation.

---

### 4. Discrete SuperLearner Selection ✅ NOW CONSISTENT

**R Implementation** (`CV.SuperLearner.R` line 89):
```r
cvdiscreteSL.predict = fit.SL$library.predict[, which.min(fit.SL$cvRisk)]
cvwhichDiscreteSL = names(which.min(fit.SL$cvRisk))
```

Selects the base learner with minimum CV risk from the inner CV.

**Python Implementation** (`cv_super_learner.py` lines 126-137):
```python
# Determine discrete SuperLearner (best base learner by CV risk)
discrete_sl_name = None
discrete_sl_p = None
if fold_cv_risks is not None and len(fold_cv_risks) > 0:
    # Find learner with minimum CV risk
    best_idx = np.argmin(fold_cv_risks)
    discrete_sl_name = sl.base_learner_names_[best_idx]
    # Get predictions from that learner
    discrete_sl_p = _get_proba_fallback(sl.base_learners_full_[best_idx][1], X_te)
```

**Verdict**: ✅ **IMPLEMENTED** - Python now implements discrete SuperLearner selection using the same criterion (minimum CV risk).

---

### 5. Meta-Learner Coefficients ✅ NOW CONSISTENT

**R Implementation** (`CV.SuperLearner.R` line 112):
```r
coef <- do.call('rbind', lapply(cvList, '[[', 'cvcoef'))
```

Returns a matrix with coefficients per fold.

**Python Implementation** (`cv_super_learner.py` lines 250-264):
```python
# Build coefficient dataframe
coef_df = None
if all_coefs and all_coefs[0] is not None:
    coef_data = []
    learner_names_coef = [n for n, _ in base_learners]
    for fold_idx, coef in enumerate(all_coefs):
        if coef is not None:
            for learner_name, weight in zip(learner_names_coef, coef):
                coef_data.append({
                    'fold': fold_idx + 1,
                    'learner': learner_name,
                    'coefficient': weight
                })
    if coef_data:
        coef_df = pd.DataFrame(coef_data)
```

**Verdict**: ✅ **IMPLEMENTED** - Python now returns coefficients in a structured DataFrame.

---

## Output Structure Comparison

### R CV.SuperLearner Returns:

| Field | Description |
|-------|-------------|
| `SL.predict` | SuperLearner predictions (concatenated across folds) |
| `discreteSL.predict` | Discrete SL predictions |
| `whichDiscreteSL` | Selected learner per fold |
| `library.predict` | All base learner predictions |
| `coef` | Meta-learner weights per fold |
| `folds` | Fold assignments |
| `V` | Number of folds |
| `libraryNames` | Learner names |
| `method` | Meta-learning method |
| `AllSL` | Full SuperLearner objects (optional) |

### Python CVSuperLearner Returns (via `SuperLearnerCVResults`):

| Field | Description |
|-------|-------------|
| `predictions['SuperLearner']` | SuperLearner predictions ✅ |
| `predictions['DiscreteSL']` | Discrete SL predictions ✅ NEW |
| `which_discrete_sl` | Selected learner per fold ✅ NEW |
| `predictions[learner_name]` | All base learner predictions ✅ |
| `coef` | Meta-learner weights per fold ✅ NEW |
| `cv_risk` | Inner CV risk per learner/fold ✅ NEW |
| `metrics` | Per-fold performance metrics ✅ |
| `config` | Configuration dict ✅ |

## Implementation Changes Made

### 1. SuperLearner Class (`super_learner.py`)

**Added CV Risk Computation:**
- Modified `_build_level1()` to compute and return `cv_risks`
- CV risk calculated as MSE between Z matrix predictions and true outcomes
- Stored in `self.cv_risks_` attribute

```python
cv_risks = np.zeros(K)
for j in range(K):
    valid_mask = ~np.isnan(Z[:, j])
    if valid_mask.sum() > 0:
        cv_risks[j] = mean_squared_error(y_arr[valid_mask], Z[valid_mask, j])
```

### 2. CVSuperLearner (`cv_super_learner.py`)

**Added Discrete SuperLearner:**
- Extract `cv_risks_` from fitted SuperLearner
- Select learner with minimum CV risk
- Generate predictions using selected learner
- Add to metrics with `learner_type='discrete'`

**Added Coefficient and CV Risk Collection:**
- Collect coefficients from each fold's fitted SuperLearner
- Collect CV risks from each fold
- Build DataFrames for coefficients and CV risks
- Pass to `SuperLearnerCVResults`

### 3. SuperLearnerCVResults (`results.py`)

**Added New Attributes:**
- `coef`: DataFrame with fold-wise coefficients
- `cv_risk`: DataFrame with fold-wise CV risks
- `which_discrete_sl`: List of discrete SL selections per fold

## Validation Tests

### Unit Tests (`test_r_python_consistency.py`)

Comprehensive test suite validating:
1. ✅ Discrete SL selection matches minimum CV risk
2. ✅ Coefficients sum to 1 (when normalized)
3. ✅ Coefficients are non-negative (for NNLS/NNLogLik)
4. ✅ CV risk computation correctness
5. ✅ Discrete SL performance bounds
6. ✅ SuperLearner outperforms baseline
7. ✅ Prediction structure and shape
8. ✅ Reproducibility with random seeds
9. ✅ Output structure matches R's fields

All 11 tests pass successfully.

### Cross-Platform Comparison Scripts

**R Script** (`test_r_python_comparison.R`):
- Generates test data
- Runs R's CV.SuperLearner
- Saves results for comparison

**Python Script** (`test_r_python_comparison.py`):
- Loads same data
- Runs Python's CVSuperLearner
- Compares outputs side-by-side
- Reports matches/differences

## Known Differences (Expected and Acceptable)

### 1. Random Number Generation
- R and Python use different RNGs
- Results in different fold assignments and randomForest behavior
- **Impact**: Exact numeric values differ, but algorithm is identical

### 2. RandomForest Implementations
- R uses `randomForest` package
- Python uses `sklearn.ensemble.RandomForestClassifier`
- Different tree-building algorithms
- **Impact**: RF predictions differ, but this is expected

### 3. Numerical Precision
- Minor differences in floating-point arithmetic
- **Impact**: Coefficients may differ at 6th+ decimal place

### 4. Screening Algorithms
- R has extensive screening algorithm support
- Python currently focuses on learner ensembling
- **Impact**: Python doesn't have screening, but this is a feature difference, not algorithmic inconsistency

## Conclusion

### Summary of Findings

| Component | Status | Notes |
|-----------|--------|-------|
| Inner CV for meta-learner | ✅ Consistent | Both use inner CV to build Z matrix |
| Base learner refit | ✅ Consistent | Both refit on full outer training fold |
| Meta-learner coefficients | ✅ Consistent | Now returned in Python |
| CV risk computation | ✅ Consistent | Now computed in Python |
| Discrete SL selection | ✅ Consistent | Now implemented in Python |
| Prediction structure | ✅ Consistent | Both return same prediction types |
| Output format | ✅ Consistent | Python returns equivalent information |

### Recommendations

1. ✅ **Discrete SuperLearner** - Now implemented and tested
2. ✅ **Coefficient reporting** - Now available via `results.coef`
3. ✅ **CV risk reporting** - Now available via `results.cv_risk`
4. ✅ **Validation tests** - Comprehensive test suite created

### Future Enhancements

1. **Screening algorithms** - Add variable screening to match R's functionality
2. **Additional meta-learners** - Implement more method.* functions from R
3. **Parallel processing** - Full parallel support for CVSuperLearner
4. **Custom CV splitters** - Enhanced support for grouped/stratified CV

## References

- R SuperLearner package: https://github.com/ecpolley/SuperLearner
- van der Laan, M. J., Polley, E. C. and Hubbard, A. E. (2007) "Super Learner." Statistical Applications in Genetics and Molecular Biology, 6(1).
- Python implementation: mysuperlearner package v0.2.0+

---

**Document Version**: 1.0
**Date**: 2025-11-30
**Author**: Claude Code (Anthropic)
**Status**: Implementation Complete, Tests Passing
