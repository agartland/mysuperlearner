# Error Handling User Guide

## Overview

The `mysuperlearner` package provides comprehensive error handling to make ensemble methods robust to real-world data challenges. This guide explains how to configure and use error handling features effectively.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Common Error Scenarios](#common-error-scenarios)
3. [Configuration Options](#configuration-options)
4. [Error Handling Policies](#error-handling-policies)
5. [Diagnostics and Debugging](#diagnostics-and-debugging)
6. [Best Practices](#best-practices)
7. [Examples](#examples)

---

## Quick Start

### Basic Usage (Default Behavior)

By default, `ExtendedSuperLearner` uses a **permissive** error handling policy:
- Tracks all errors and warnings
- Continues if at least 1 learner succeeds
- Issues warnings for failed learners
- Works with partially failed learners

```python
from mysuperlearner import ExtendedSuperLearner
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Default: permissive error handling with tracking
sl = ExtendedSuperLearner(
    method='nnloglik',
    folds=5,
    track_errors=True,  # default
    verbose=False
)

learners = [
    ('rf', RandomForestClassifier(n_estimators=50)),
    ('logistic', LogisticRegression(max_iter=1000))
]

sl.fit_explicit(X_train, y_train, learners)
predictions = sl.predict_proba(X_test)

# Check for errors
diagnostics = sl.get_diagnostics()
if diagnostics['n_errors'] > 0:
    print(f"Warning: {diagnostics['n_errors']} errors occurred")
    for error in diagnostics['errors']:
        print(f"  - {error.learner_name}: {error.message}")
```

### Enhanced Error Handling (New!)

For more control, use the enhanced configuration system:

```python
from mysuperlearner import ExtendedSuperLearner
from mysuperlearner.error_handling_enhanced import (
    SuperLearnerConfig,
    ErrorHandlingPolicy
)

# Configure strict error handling
config = SuperLearnerConfig(
    error_policy=ErrorHandlingPolicy.STRICT,  # Fail on any error
    min_viable_learners=2,
    verbose_errors=True
)

sl = ExtendedSuperLearner(method='nnloglik', config=config)
sl.fit_explicit(X_train, y_train, learners)
```

---

## Common Error Scenarios

### 1. Missing Data

**Problem**: Real-world data often has missing values that cause learner failures.

**Solution Options**:

#### Option A: Pre-process Data (Recommended)
```python
from sklearn.impute import SimpleImputer

# Impute before fitting
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

sl.fit_explicit(X_train_imputed, y_train, learners)
```

#### Option B: Automatic Imputation (Enhanced)
```python
config = SuperLearnerConfig(
    impute_missing=True,
    imputation_strategy='mean'  # or 'median', 'most_frequent'
)

sl = ExtendedSuperLearner(method='nnloglik', config=config)
# Missing values automatically handled
sl.fit_explicit(X_train, y_train, learners)
```

#### Option C: Use Robust Learners
```python
from sklearn.ensemble import HistGradientBoostingClassifier

# Tree-based learners that handle missing values
learners = [
    ('hist_gb', HistGradientBoostingClassifier()),  # Native missing support
    ('rf', RandomForestClassifier())  # Will fail with NaN
]
```

---

### 2. Convergence Failures

**Problem**: Learners may not converge due to:
- Low iteration limits
- Difficult optimization landscape
- Perfect separation
- Low variability

**Solution**:

```python
# Increase iteration limits
learners = [
    ('logistic', LogisticRegression(max_iter=2000)),  # Increased from 100
    ('svm', SVC(probability=True, max_iter=2000)),
]

# Or use permissive policy to continue despite convergence warnings
config = SuperLearnerConfig(
    error_policy=ErrorHandlingPolicy.PERMISSIVE,
    raise_on_meta_convergence_failure=False  # Don't fail on meta convergence
)

sl = ExtendedSuperLearner(method='nnloglik', config=config)
```

**Check convergence**:

```python
sl.fit_explicit(X_train, y_train, learners)

# Check meta-learner convergence
if hasattr(sl.meta_learner_, 'convergence_info_'):
    info = sl.meta_learner_.convergence_info_
    if not info.get('success', True):
        print(f"Warning: Meta-learner did not converge: {info.get('message')}")
```

---

### 3. Imbalanced Classes

**Problem**: Highly imbalanced classes can cause:
- Some learners to predict only majority class
- Convergence issues
- AUC calculation failures in some folds

**Solution**:

```python
from sklearn.utils.class_weight import compute_sample_weight

# Use sample weights
sample_weight = compute_sample_weight('balanced', y_train)

sl.fit_explicit(X_train, y_train, learners, sample_weight=sample_weight)

# Or use learners with built-in class weighting
from sklearn.ensemble import RandomForestClassifier

learners = [
    ('rf_balanced', RandomForestClassifier(class_weight='balanced')),
    ('logistic_balanced', LogisticRegression(class_weight='balanced'))
]

# Include intercept-only baseline to compare
from mysuperlearner.meta_learners import InterceptOnlyEstimator
learners.append(('intercept', InterceptOnlyEstimator()))
```

---

### 4. Perfect Separation

**Problem**: When classes are perfectly separable, logistic regression can fail to converge.

**Solution**:

```python
# Use regularized learners
learners = [
    ('logistic_l2', LogisticRegression(penalty='l2', C=1.0)),  # L2 regularization
    ('ridge', RidgeClassifier(alpha=1.0)),
]

# Or use permissive policy
config = SuperLearnerConfig(
    error_policy=ErrorHandlingPolicy.PERMISSIVE,
    min_viable_learners=1
)
```

---

### 5. Collinearity

**Problem**: Highly collinear features can cause:
- Numerical instability
- Convergence issues
- Poor weight estimation

**Solution**:

```python
from sklearn.linear_model import RidgeClassifier, LogisticRegression

# Use regularization
learners = [
    ('ridge', RidgeClassifier(alpha=10.0)),  # Strong regularization
    ('logistic_l2', LogisticRegression(penalty='l2', C=0.1)),
    ('rf', RandomForestClassifier())  # Naturally robust to collinearity
]

# Or pre-process with PCA
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

learners = [
    ('pca_logistic', Pipeline([
        ('pca', PCA(n_components=0.95)),
        ('logistic', LogisticRegression())
    ]))
]
```

---

### 6. Learner Fails on Specific Folds

**Problem**: A learner may fail on specific CV folds due to:
- Small fold size
- Unusual class distribution in fold
- Specific data characteristics

**Solution**:

```python
config = SuperLearnerConfig(
    error_policy=ErrorHandlingPolicy.PERMISSIVE,
    min_viable_folds=3,  # Require success in at least 3/5 folds
    max_error_rate=0.4   # Allow up to 40% fold failures
)

sl = ExtendedSuperLearner(method='nnloglik', folds=5, config=config)
sl.fit_explicit(X_train, y_train, learners)

# Check which folds failed per learner
diagnostics = sl.get_diagnostics()
if diagnostics['n_errors'] > 0:
    for error in diagnostics['errors']:
        if error.fold is not None:
            print(f"{error.learner_name} failed on fold {error.fold}")
```

---

### 7. Mixed Variable Types

**Problem**: Some learners don't handle categorical or binary variables well.

**Solution**:

```python
# Use tree-based learners that handle mixed types
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

learners = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('gbm', GradientBoostingClassifier(n_estimators=100)),
]

# Or pre-process categorical variables
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Separate continuous and categorical
continuous_features = [0, 1, 2]
categorical_features = [3, 4, 5]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), continuous_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
```

---

## Configuration Options

### SuperLearnerConfig Parameters

```python
from mysuperlearner.error_handling_enhanced import SuperLearnerConfig

config = SuperLearnerConfig(
    # Overall error handling strategy
    error_policy=ErrorHandlingPolicy.PERMISSIVE,

    # Viability thresholds
    min_viable_learners=1,        # Minimum learners needed
    min_viable_folds=None,        # Minimum successful folds (None = folds // 2)
    max_error_rate=0.5,           # Maximum error rate for ADAPTIVE policy

    # Prediction handling
    neutral_probability=0.5,      # Probability when prediction fails
    prediction_error_handling='neutral',  # 'neutral', 'skip', or 'fail'

    # Missing data
    impute_missing=False,         # Auto-impute missing values
    imputation_strategy='mean',   # 'mean', 'median', or 'most_frequent'

    # Meta-learner
    raise_on_meta_convergence_failure=False,
    track_convergence_info=True,

    # Diagnostics
    verbose_errors=False,         # Print detailed error info
    collect_error_context=True,   # Collect sample sizes, distributions, etc.
)
```

---

## Error Handling Policies

### STRICT Policy

**Use when**: You need guaranteed quality and can't tolerate any failures.

```python
config = SuperLearnerConfig(error_policy=ErrorHandlingPolicy.STRICT)
```

**Behavior**:
- ❌ Fails immediately if any learner fails
- ❌ Fails if any fold fails
- ✅ Ensures all learners worked successfully
- ✅ Best for critical applications

**Example**:
```python
try:
    sl = ExtendedSuperLearner(method='nnloglik', config=config)
    sl.fit_explicit(X_train, y_train, learners)
except RuntimeError as e:
    print(f"Fitting failed: {e}")
    # Handle by fixing data or adjusting learners
```

---

### PERMISSIVE Policy (Default)

**Use when**: You want robustness with awareness of issues.

```python
config = SuperLearnerConfig(error_policy=ErrorHandlingPolicy.PERMISSIVE)
```

**Behavior**:
- ✅ Continues if minimum viable learners succeed
- ⚠️ Issues warnings for failures
- ✅ Tracks all errors for later review
- ✅ Best for exploratory analysis and production

**Example**:
```python
import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    sl = ExtendedSuperLearner(method='nnloglik', config=config)
    sl.fit_explicit(X_train, y_train, learners)

    # Check warnings
    for warning in w:
        print(f"Warning: {warning.message}")
```

---

### SILENT Policy

**Use when**: You trust your setup and want no warnings (not recommended for production).

```python
config = SuperLearnerConfig(error_policy=ErrorHandlingPolicy.SILENT)
```

**Behavior**:
- ✅ Continues silently if minimum viable learners succeed
- 🔇 No warnings issued
- ✅ Tracks errors but doesn't alert user
- ⚠️ Use with caution

---

### ADAPTIVE Policy

**Use when**: You want dynamic error tolerance based on error rate.

```python
config = SuperLearnerConfig(
    error_policy=ErrorHandlingPolicy.ADAPTIVE,
    max_error_rate=0.3  # Fail if >30% of learners fail
)
```

**Behavior**:
- ✅ Continues if error rate below threshold
- ❌ Fails if error rate exceeds max_error_rate
- ✅ Adapts to ensemble size
- ✅ Best for large ensembles

---

## Diagnostics and Debugging

### Basic Error Checking

```python
sl.fit_explicit(X_train, y_train, learners)

# Get diagnostics
diag = sl.get_diagnostics()

print(f"Method: {diag['method']}")
print(f"Number of folds: {diag['n_folds']}")
print(f"Number of errors: {diag['n_errors']}")

if diag['n_errors'] > 0:
    print("\nErrors:")
    for error in diag['errors']:
        print(f"  Learner: {error.learner_name}")
        print(f"  Type: {error.error_type.value}")
        print(f"  Phase: {error.phase}")
        print(f"  Message: {error.message}")
        if error.fold is not None:
            print(f"  Fold: {error.fold}")
        print()
```

### Enhanced Error Summary (with Enhanced Error Handling)

```python
from mysuperlearner.error_handling_enhanced import EnhancedErrorTracker

config = SuperLearnerConfig(
    error_policy=ErrorHandlingPolicy.PERMISSIVE,
    collect_error_context=True,
    verbose_errors=True
)

sl = ExtendedSuperLearner(method='nnloglik', config=config)
sl.fit_explicit(X_train, y_train, learners)

# Get error summary as DataFrame
if hasattr(sl.error_tracker, 'get_error_summary_df'):
    summary = sl.error_tracker.get_error_summary_df()
    print("\nError Summary:")
    print(summary)
```

### CV Performance by Learner

```python
diag = sl.get_diagnostics()

if 'cv_scores' in diag:
    print("\nCV AUC Scores:")
    for learner, score in diag['cv_scores'].items():
        print(f"  {learner}: {score:.4f}")

# Check meta-weights
if diag['meta_weights'] is not None:
    print("\nMeta-Learner Weights:")
    for learner, weight in zip(diag['base_learner_names'], diag['meta_weights']):
        print(f"  {learner}: {weight:.4f}")
```

### External CV Evaluation

```python
from mysuperlearner.evaluation import evaluate_super_learner_cv

# Evaluate with external CV
results = evaluate_super_learner_cv(
    X, y, learners, sl,
    outer_folds=5,
    return_predictions=True  # Get predictions for further analysis
)

if isinstance(results, tuple):
    results_df, predictions = results
else:
    results_df = results

# Analyze results
print("\nExternal CV Results:")
print(results_df.groupby('learner')[['auc', 'logloss', 'accuracy']].mean())

# Check for problematic folds
for learner in results_df['learner'].unique():
    learner_results = results_df[results_df['learner'] == learner]
    if learner_results['auc'].std() > 0.1:
        print(f"\nWarning: High variance for {learner}")
        print(learner_results[['fold', 'auc']])
```

---

## Best Practices

### 1. Start Permissive, Then Tune

```python
# Phase 1: Explore with permissive policy
config_explore = SuperLearnerConfig(
    error_policy=ErrorHandlingPolicy.PERMISSIVE,
    verbose_errors=True,
    collect_error_context=True
)

sl = ExtendedSuperLearner(method='nnloglik', config=config_explore)
sl.fit_explicit(X_train, y_train, learners)

# Review errors and adjust learners
diag = sl.get_diagnostics()
# ... analyze errors ...

# Phase 2: Production with stricter policy
config_prod = SuperLearnerConfig(
    error_policy=ErrorHandlingPolicy.ADAPTIVE,
    max_error_rate=0.2,
    min_viable_learners=2
)

sl_prod = ExtendedSuperLearner(method='nnloglik', config=config_prod)
sl_prod.fit_explicit(X_train, y_train, working_learners)
```

### 2. Always Include Baseline Learners

```python
from mysuperlearner.meta_learners import InterceptOnlyEstimator

learners = [
    ('intercept', InterceptOnlyEstimator()),  # Baseline - should always work
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('logistic', LogisticRegression(max_iter=1000)),
    # ... other learners
]
```

### 3. Use External CV for Unbiased Evaluation

```python
from mysuperlearner.evaluation import evaluate_super_learner_cv

# External CV gives unbiased performance estimate
results = evaluate_super_learner_cv(
    X, y, learners, sl,
    outer_folds=5,
    random_state=42
)

# Compare SuperLearner to individual learners
sl_auc = results[results['learner'] == 'SuperLearner']['auc'].mean()
base_aucs = results[results['learner_type'] == 'base']['auc'].mean()

print(f"SuperLearner AUC: {sl_auc:.4f}")
print(f"Average Base Learner AUC: {base_aucs:.4f}")
```

### 4. Handle Data Quality Upfront

```python
# Check for issues before fitting
print(f"Missing values: {np.isnan(X_train).sum()}")
print(f"Class balance: {np.bincount(y_train)}")
print(f"Feature correlation: {np.corrcoef(X_train.T).max():.3f}")

# Address issues
if np.isnan(X_train).any():
    # Impute missing
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

if np.bincount(y_train)[1] / len(y_train) < 0.1:
    # Use sample weights for imbalance
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weight = compute_sample_weight('balanced', y_train)
    sl.fit_explicit(X_train, y_train, learners, sample_weight=sample_weight)
```

### 5. Monitor Production Performance

```python
# In production, track errors over time
error_log = []

for batch_X, batch_y in data_stream:
    try:
        predictions = sl.predict_proba(batch_X)

        # Log any prediction errors
        if hasattr(sl, 'error_tracker') and sl.error_tracker:
            recent_errors = sl.error_tracker.error_records[-10:]  # Last 10 errors
            for error in recent_errors:
                if error.phase == 'prediction':
                    error_log.append({
                        'timestamp': datetime.now(),
                        'learner': error.learner_name,
                        'error': error.message
                    })

    except Exception as e:
        print(f"Critical error: {e}")
        # Retrain or fallback to simpler model

# Periodic error analysis
if len(error_log) > 100:
    print("Warning: Many prediction errors, consider retraining")
```

---

## Examples

### Example 1: Handling Real-World Messy Data

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from mysuperlearner import ExtendedSuperLearner
from mysuperlearner.error_handling_enhanced import SuperLearnerConfig, ErrorHandlingPolicy

# Simulate messy real-world data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Add missing values (5% of data)
rng = np.random.RandomState(42)
missing_mask = rng.random(X.shape) < 0.05
X = X.astype(float)
X[missing_mask] = np.nan

# Create imbalanced classes
y = y.copy()
y[rng.random(len(y)) < 0.85] = 0  # 85% class 0

print(f"Missing values: {np.isnan(X).sum()}")
print(f"Class distribution: {np.bincount(y)}")

# Configure for messy data
config = SuperLearnerConfig(
    error_policy=ErrorHandlingPolicy.PERMISSIVE,
    impute_missing=True,
    imputation_strategy='median',
    min_viable_learners=2,
    verbose_errors=False
)

# Use robust learners
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from mysuperlearner.meta_learners import InterceptOnlyEstimator

learners = [
    ('intercept', InterceptOnlyEstimator()),
    ('rf', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
    ('gbm', GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)),
    ('logistic_l2', LogisticRegression(penalty='l2', C=0.1, max_iter=1000,
                                        class_weight='balanced', random_state=42)),
]

# Fit with sample weights
from sklearn.utils.class_weight import compute_sample_weight
sample_weight = compute_sample_weight('balanced', y)

sl = ExtendedSuperLearner(method='nnloglik', folds=5, random_state=42, config=config)
sl.fit_explicit(X, y, learners, sample_weight=sample_weight)

# Evaluate
from mysuperlearner.evaluation import evaluate_super_learner_cv
results = evaluate_super_learner_cv(X, y, learners, sl, outer_folds=5,
                                   sample_weight=sample_weight, random_state=42)

print("\nResults:")
print(results.groupby('learner')['auc'].agg(['mean', 'std']))

# Check diagnostics
diag = sl.get_diagnostics()
print(f"\nErrors: {diag['n_errors']}")
print("Meta-weights:")
for name, weight in zip(diag['base_learner_names'], diag['meta_weights']):
    print(f"  {name}: {weight:.3f}")
```

### Example 2: Strict Quality Control

```python
# For critical applications where quality is paramount
config = SuperLearnerConfig(
    error_policy=ErrorHandlingPolicy.STRICT,
    raise_on_meta_convergence_failure=True,
    collect_error_context=True,
    verbose_errors=True
)

# Use only well-tested, robust learners
learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gbm', GradientBoostingClassifier(n_estimators=100, random_state=42)),
]

try:
    sl = ExtendedSuperLearner(method='nnloglik', folds=10, random_state=42, config=config)
    sl.fit_explicit(X_clean, y_clean, learners)
    print("✓ All learners succeeded")

except RuntimeError as e:
    print(f"✗ Fitting failed: {e}")
    # Investigate and fix issues before proceeding
```

### Example 3: Large Ensemble with Adaptive Policy

```python
# Large ensemble with many learners
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

learners = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('gbm', GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ('ada', AdaBoostClassifier(n_estimators=50, random_state=42)),
    ('et', ExtraTreesClassifier(n_estimators=50, random_state=42)),
    ('logistic', LogisticRegression(max_iter=1000, random_state=42)),
    ('ridge', RidgeClassifier(random_state=42)),
    ('svm_rbf', SVC(probability=True, random_state=42)),
    ('naive_bayes', GaussianNB()),
]

# Adaptive policy: tolerate some failures but not too many
config = SuperLearnerConfig(
    error_policy=ErrorHandlingPolicy.ADAPTIVE,
    max_error_rate=0.25,  # Up to 25% can fail (2/8 learners)
    min_viable_learners=5,
    neutral_probability=0.5
)

sl = ExtendedSuperLearner(method='nnloglik', folds=5, random_state=42, config=config)
sl.fit_explicit(X, y, learners)

# Review which learners contributed
diag = sl.get_diagnostics()
working_learners = [name for name, weight in zip(diag['base_learner_names'],
                                                   diag['meta_weights'])
                   if weight > 0.01]
print(f"Working learners ({len(working_learners)}/{len(learners)}): {working_learners}")
```

---

## Summary

**Key Takeaways**:

1. **Default behavior is robust**: The package works well out-of-the-box with permissive error handling
2. **Configure for your needs**: Use `SuperLearnerConfig` for fine-grained control
3. **Check diagnostics**: Always review `get_diagnostics()` output
4. **Handle data quality upfront**: Address missing values and class imbalance before fitting
5. **Use external CV**: Get unbiased performance estimates with `evaluate_super_learner_cv`
6. **Include baselines**: Always add `InterceptOnlyEstimator` as a sanity check
7. **Monitor in production**: Track errors and retrain when needed

For more information, see:
- [API Documentation](API_REFERENCE.md)
- [Error Handling Analysis](ERROR_HANDLING_ANALYSIS.md)
- [Examples](../examples/)
