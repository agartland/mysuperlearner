"""
Example: Parallel Variable Importance Computation with MySuperLearner

This script demonstrates how to use parallel processing to speed up variable
importance computation in MySuperLearner. It includes timing comparisons and
shows how to use different parallelization levels.

Key Features:
- Permutation importance with parallel execution
- Drop-column importance in parallel
- Timing comparisons between sequential and parallel
- Recommendations for choosing n_jobs parameter
"""

import numpy as np
import pandas as pd
import time
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from mysuperlearner import SuperLearner
from mysuperlearner.variable_importance import compute_variable_importance
from mysuperlearner.meta_learners import NNLogLikEstimator


# ============================================================================
# Create Example Dataset
# ============================================================================

print("=" * 80)
print("Parallel Variable Importance Example")
print("=" * 80)

# Create a moderately-sized dataset (30 features)
print("\nGenerating dataset with 30 features...")
np.random.seed(42)
X, y = make_classification(
    n_samples=500,
    n_features=30,
    n_informative=20,
    n_redundant=5,
    n_clusters_per_class=3,
    flip_y=0.05,
    random_state=42
)

X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

print(f"Dataset shape: {X_df.shape}")
print(f"  - Samples: {len(y)}")
print(f"  - Features: {X_df.shape[1]}")
print(f"  - Positive class proportion: {y.mean():.2%}")


# ============================================================================
# Fit SuperLearner
# ============================================================================

print("\nFitting SuperLearner with multiple base learners...")

# Define diverse set of base learners
learners = [
    ("logistic", LogisticRegression(max_iter=1000, random_state=42)),
    ("rf_shallow", RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)),
    ("rf_deep", RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)),
    ("gbm", GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)),
    ("tree", DecisionTreeClassifier(max_depth=5, random_state=42))
]

# Fit SuperLearner
sl = SuperLearner(
    method=NNLogLikEstimator(),
    folds=5,
    shuffle=True,
    random_state=42,
    store_X=True  # Required for variable importance
)

start_fit = time.time()
sl.fit_explicit(X_df, y, learners)
fit_time = time.time() - start_fit

print(f"SuperLearner fitted in {fit_time:.2f} seconds")
print(f"  - Meta-learner: {sl.method}")
print(f"  - CV folds: {sl.folds}")
print(f"  - Base learners: {len(learners)}")


# ============================================================================
# Example 1: Permutation Importance - Sequential vs Parallel
# ============================================================================

print("\n" + "=" * 80)
print("Example 1: Permutation Importance (Sequential vs Parallel)")
print("=" * 80)

# Sequential execution
print("\nComputing permutation importance sequentially (n_jobs=1)...")
start = time.time()
results_seq = compute_variable_importance(
    sl,
    method='permutation',
    n_repeats=5,
    random_state=42,
    n_jobs=1,
    verbose=False
)
time_seq = time.time() - start

print(f"Sequential execution time: {time_seq:.2f} seconds")

# Parallel execution with 2 CPUs
print("\nComputing permutation importance in parallel (n_jobs=2)...")
start = time.time()
results_par2 = compute_variable_importance(
    sl,
    method='permutation',
    n_repeats=5,
    random_state=42,
    n_jobs=2,
    verbose=False
)
time_par2 = time.time() - start

print(f"Parallel execution time (n_jobs=2): {time_par2:.2f} seconds")
speedup_2 = time_seq / time_par2
print(f"Speedup: {speedup_2:.2f}x")

# Parallel execution with all available CPUs
print("\nComputing permutation importance in parallel (n_jobs=-1, all CPUs)...")
start = time.time()
results_par_all = compute_variable_importance(
    sl,
    method='permutation',
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
    verbose=False
)
time_par_all = time.time() - start

print(f"Parallel execution time (n_jobs=-1): {time_par_all:.2f} seconds")
speedup_all = time_seq / time_par_all
print(f"Speedup: {speedup_all:.2f}x")

# Show top 10 most important features
print("\nTop 10 Most Important Features:")
print(results_par_all.summary(top_n=10)[['feature', 'importance', 'rank']])


# ============================================================================
# Example 2: Drop-Column Importance in Parallel
# ============================================================================

print("\n" + "=" * 80)
print("Example 2: Drop-Column Importance (Parallel)")
print("=" * 80)

print("\nComputing drop-column importance in parallel (n_jobs=4)...")
start = time.time()
results_drop = compute_variable_importance(
    sl,
    method='drop_column',
    random_state=42,
    n_jobs=4,
    verbose=True  # Show progress
)
time_drop = time.time() - start

print(f"\nDrop-column importance computed in {time_drop:.2f} seconds")

# Show top 10
print("\nTop 10 Most Important Features (Drop-Column):")
print(results_drop.summary(top_n=10)[['feature', 'importance', 'rank']])


# ============================================================================
# Example 3: Comparing Multiple Methods in Parallel
# ============================================================================

print("\n" + "=" * 80)
print("Example 3: Multiple Methods in Parallel")
print("=" * 80)

print("\nComputing multiple importance methods in parallel...")
start = time.time()
results_multi = compute_variable_importance(
    sl,
    method=['permutation', 'drop_column'],
    n_repeats=3,
    random_state=42,
    n_jobs=4,
    verbose=False
)
time_multi = time.time() - start

print(f"Multiple methods computed in {time_multi:.2f} seconds")

# Compare rankings between methods
comparison = results_multi.compare_methods()
print("\nFeature Rank Comparison (Permutation vs Drop-Column):")
print(comparison.head(10))


# ============================================================================
# Example 4: Grouped Importance in Parallel
# ============================================================================

print("\n" + "=" * 80)
print("Example 4: Grouped Permutation Importance (Parallel)")
print("=" * 80)

print("\nComputing grouped importance with correlation threshold=0.7...")
start = time.time()
results_grouped = compute_variable_importance(
    sl,
    method='grouped',
    grouped_threshold=0.7,
    n_repeats=3,
    random_state=42,
    n_jobs=4,
    verbose=True
)
time_grouped = time.time() - start

print(f"\nGrouped importance computed in {time_grouped:.2f} seconds")

# Show group information
if results_grouped.cluster_info is not None:
    n_groups = len(results_grouped.importance_df)
    print(f"\nFeatures were clustered into {n_groups} groups")
    print("\nTop 5 Most Important Groups:")
    print(results_grouped.summary(top_n=5)[['feature', 'importance', 'rank']])


# ============================================================================
# Performance Summary and Recommendations
# ============================================================================

print("\n" + "=" * 80)
print("Performance Summary")
print("=" * 80)

print(f"\nPermutation Importance Timing:")
print(f"  Sequential (n_jobs=1):     {time_seq:.2f}s")
print(f"  Parallel (n_jobs=2):       {time_par2:.2f}s  ({speedup_2:.2f}x speedup)")
print(f"  Parallel (n_jobs=-1):      {time_par_all:.2f}s  ({speedup_all:.2f}x speedup)")

print(f"\nDrop-Column Importance:")
print(f"  Parallel (n_jobs=4):       {time_drop:.2f}s")

print(f"\nGrouped Importance:")
print(f"  Parallel (n_jobs=4):       {time_grouped:.2f}s")

print("\n" + "=" * 80)
print("Recommendations for Choosing n_jobs")
print("=" * 80)

print("""
1. **For small datasets (<10 features):**
   - Use n_jobs=1 or n_jobs=2
   - Parallel overhead may dominate for very small problems

2. **For medium datasets (10-50 features):**
   - Use n_jobs=4 to n_jobs=8
   - Good balance between speedup and resource usage

3. **For large datasets (50+ features):**
   - Use n_jobs=-1 (all available CPUs)
   - Maximum speedup for large-scale importance computation

4. **Memory considerations:**
   - Each parallel worker creates a copy of the data
   - If memory is limited, reduce n_jobs
   - For most datasets < 1GB, n_jobs=-1 is fine

5. **Reproducibility:**
   - Always set random_state for reproducible results
   - Results are identical regardless of n_jobs value

6. **When to use parallelization:**
   - Permutation importance: HIGHLY BENEFICIAL (n repeats * n features)
   - Drop-column importance: BENEFICIAL (n features refit operations)
   - Grouped importance: BENEFICIAL (n groups * n repeats)
   - SHAP importance: Not parallelized via n_jobs (use SHAP's built-in)
""")

print("\nExample complete!")
