# mysuperlearner Refactoring Plan

**Version:** 0.2.0
**Date:** 2025-01-30
**Status:** Approved - Ready for Implementation

## Executive Summary

This plan documents the comprehensive refactoring of mysuperlearner to align with R SuperLearner and sl3 best practices while maintaining sklearn compatibility.

### User-Confirmed Priorities

- ✅ **Backward compatibility NOT important** - breaking changes acceptable
- ✅ **Fix NNLogLik optimization** (logit scale) - HIGHEST PRIORITY
- ✅ **Add screeners** (correlation and lasso) - HIGH PRIORITY
- ✅ **Add user-defined variable sets** - HIGH PRIORITY
- ✅ **Rename to match R SuperLearner** - SuperLearner, CVSuperLearner
- ✅ **sklearn-style class-based API** - Standard fit(X, y) signature

## Implementation Phases

### Phase 1: Core Algorithmic Fixes + Naming (HIGHEST PRIORITY)

#### 1.1 Rename Main Classes (BREAKING CHANGE)
- `ExtendedSuperLearner` → `SuperLearner`
- `evaluate_super_learner_cv` → `CVSuperLearner` (class-based)
- Update all imports and documentation

**Files:**
- `mysuperlearner/extended_super_learner.py` → `mysuperlearner/super_learner.py`
- `mysuperlearner/evaluation.py` → `mysuperlearner/cv_super_learner.py`
- `mysuperlearner/__init__.py`

#### 1.2 Fix NNLogLik Optimization (CRITICAL STATISTICAL FIX)
- Implement logit-scale optimization matching R SuperLearner
- Change default trim from 0.025 to 0.001
- Transform Z to logit scale, optimize, transform back

**Impact:** Results will differ slightly (more accurate, ~0.1-0.5% AUC improvement)

**File:** `mysuperlearner/meta_learners.py`

#### 1.3 sklearn-Compatible API (BREAKING CHANGE)
- Remove `fit_explicit()` method
- Move learners from fit to constructor: `SuperLearner(learners=[...])`
- Implement standard `fit(X, y)` signature
- Enable use in GridSearchCV, sklearn Pipelines

**Before:**
```python
sl = ExtendedSuperLearner(method='nnloglik', folds=5)
sl.fit_explicit(X, y, learners=[('RF', rf), ('GLM', glm)])
```

**After:**
```python
sl = SuperLearner(learners=[('RF', rf), ('GLM', glm)], method='nnloglik', cv=5)
sl.fit(X, y)
```

#### 1.4 Flexible CV Strategy
- Accept sklearn CV splitter objects (not just integer)
- Support GroupKFold, TimeSeriesSplit, custom splitters

**File:** `mysuperlearner/super_learner.py`

### Phase 2: Screening and Composition (HIGH PRIORITY)

#### 2.1 Variable Sets and Screening Framework
Implement unified interface where user-defined variable sets and statistical screeners work identically.

**New file:** `mysuperlearner/screening.py`

Classes:
- `VariableSet`: Base class for manual and statistical screening
- `CorrelationScreener`: Screen by correlation with outcome
- `LassoScreener`: Screen by Lasso coefficients

**Usage:**
```python
variable_sets = {
    'baseline': ['age', 'sex', 'bmi'],
    'biomarkers': ['crp', 'ldl', 'hdl']
}

screeners = [
    VariableSet('baseline', variable_sets['baseline']),
    VariableSet('biomarkers', variable_sets['biomarkers']),
    CorrelationScreener('corr_selected', threshold=0.2),
    LassoScreener('lasso_selected', alpha=0.01)
]
```

#### 2.2 Composition Patterns
Implement Stack and Pipeline for compositional learning.

**New file:** `mysuperlearner/composition.py`

Classes:
- `Learner`: Base class for unified interface
- `Stack`: Parallel learner ensemble
- `Pipeline`: Sequential screening → learning

**Usage:**
```python
from mysuperlearner import Pipeline, Stack

screened_glm = Pipeline(CorrelationScreener(), LogisticRegression())
stack = Stack(LogisticRegression(), RandomForestClassifier(), screened_glm)

sl = SuperLearner(learners=stack, method='nnloglik')
sl.fit(X, y)
```

### Phase 3: Documentation and QMD Rendering (CRITICAL FINAL STEP)

#### 3.1 Update All Documentation
Update all code examples to use new API:

**Python Files:**
- `README.md` - Quick Start, API examples
- `examples/*.py` - All example scripts
- `mysuperlearner/example_usage.py`
- All module docstrings

**Quarto (.qmd) Files:**
- `docs/examples/example_analysis.qmd`
- `docs/examples/TESTING_SUMMARY.qmd`
- `docs/examples/variable_importance_guide.qmd`
- `docs/analysis/r_superlearner.qmd`
- `docs/analysis/python_superlearner.qmd`
- `docs/validation/python_cv_superlearner.qmd`
- `docs/validation/r_cv_superlearner.qmd`

#### 3.2 Re-render QMD Files
```bash
# Set R path if needed
export QUARTO_R="/app/software/R/4.4.2-gfbf-2024a/bin/R"

# Render all docs
cd /fh/fast/gilbert_p/agartlan/gitrepo/mysuperlearner
quarto render docs/

# Or render specific files
quarto render docs/example_analysis.qmd
```

#### 3.3 Version Bump
- Update `__version__` in `mysuperlearner/__init__.py` to '0.2.0'
- Create or update `CHANGELOG.md`

## Key Design Decisions

### CVSuperLearner API (sklearn-style)
```python
class CVSuperLearner(BaseEstimator, ClassifierMixin):
    """Cross-validated Super Learner for unbiased performance evaluation."""

    def __init__(self, learners, method='nnloglik', cv=5, inner_cv=5, **kwargs):
        self.learners = learners
        self.method = method
        self.cv = cv
        self.inner_cv = inner_cv

    def fit(self, X, y, groups=None, sample_weight=None):
        """Fit CV Super Learner and return results."""
        # Perform outer CV loop
        self.results_ = SuperLearnerCVResults(...)
        return self

    def get_results(self):
        """Return SuperLearnerCVResults object."""
        return self.results_

# Usage
cv_sl = CVSuperLearner(learners=[...], method='nnloglik', cv=10)
cv_sl.fit(X, y)
results = cv_sl.get_results()
results.summary()
```

### NNLogLik Logit-Scale Optimization
```python
class NNLogLikEstimator:
    def __init__(self, trim=0.001):  # Changed from 0.025
        self.trim = trim

    def _logit_trim(self, Z):
        """Apply trimLogit transformation like R SuperLearner."""
        Z_trimmed = np.clip(Z, self.trim, 1 - self.trim)
        return np.log(Z_trimmed / (1 - Z_trimmed))

    def fit(self, Z, y, sample_weight=None):
        """Fit on logit scale, transform back to probability."""
        Z_logit = self._logit_trim(Z)
        # Optimize on logit scale
        # Transform back for predictions
```

## Migration Guide

### Breaking Changes

1. **Class Renaming:**
```python
# OLD (v0.1.0)
from mysuperlearner import ExtendedSuperLearner
sl = ExtendedSuperLearner(method='nnloglik', folds=5)
sl.fit_explicit(X, y, learners=[...])

# NEW (v0.2.0)
from mysuperlearner import SuperLearner
sl = SuperLearner(learners=[...], method='nnloglik', cv=5)
sl.fit(X, y)
```

2. **NNLogLik Results Will Change:**
- Logit-scale optimization (more accurate)
- Default trim: 0.025 → 0.001 (less bias)
- Typical change: ~0.1-0.5% AUC difference (usually improvement)

3. **CV Function → Class:**
```python
# OLD
from mysuperlearner import evaluate_super_learner_cv
results = evaluate_super_learner_cv(X, y, base_learners=[...], ...)

# NEW
from mysuperlearner import CVSuperLearner
cv_sl = CVSuperLearner(learners=[...], method='nnloglik', cv=5)
results = cv_sl.fit(X, y)
```

### New Features

1. **Variable Sets:**
```python
variable_sets = {'baseline': ['age', 'sex', 'bmi']}
screeners = [VariableSet('baseline', variable_sets['baseline'])]
learners = [('GLM_baseline', Pipeline(screeners[0], LogisticRegression()))]
sl = SuperLearner(learners=learners, method='nnloglik')
```

2. **Flexible CV:**
```python
from sklearn.model_selection import GroupKFold
sl = SuperLearner(learners=[...], cv=GroupKFold(n_splits=5))
sl.fit(X, y, groups=patient_ids)
```

## Implementation Checklist

### Phase 1: Core Fixes
- [ ] Rename ExtendedSuperLearner → SuperLearner
- [ ] Rename evaluate_super_learner_cv → CVSuperLearner
- [ ] Fix NNLogLik logit-scale optimization
- [ ] Move learners to constructor (sklearn API)
- [ ] Accept sklearn CV splitters

### Phase 2: Screening & Composition
- [ ] Implement VariableSet base class
- [ ] Implement CorrelationScreener
- [ ] Implement LassoScreener
- [ ] Implement Stack class
- [ ] Implement Pipeline class

### Phase 3: Documentation
- [ ] Update README.md
- [ ] Update all example scripts
- [ ] Update all .qmd files (7 files)
- [ ] Re-render all .qmd files with quarto
- [ ] Update module docstrings
- [ ] Update __version__ to 0.2.0
- [ ] Create/update CHANGELOG.md

## Estimated Effort

- Phase 1: 4-7 days
- Phase 2: 6-10 days
- Phase 3: 2-3 days

**Total: 12-20 days**

## References

- R SuperLearner: `./tmp/SuperLearner/`
- sl3: `./tmp/sl3/`
- Plan: `/home/agartlan/.claude/plans/sequential-honking-anchor.md`
