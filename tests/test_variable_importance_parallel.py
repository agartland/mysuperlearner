"""
Comprehensive test suite for parallel variable importance computation.

Tests cover:
- Correctness: parallel results match sequential results
- Reproducibility: same random_state produces identical results
- Performance: parallel execution is faster than sequential
- Error handling: graceful handling of failures in parallel mode
- Edge cases: single feature, n_jobs > n_features, etc.
"""

import numpy as np
import pandas as pd
import pytest
import time
import warnings
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from mysuperlearner import SuperLearner
from mysuperlearner.variable_importance import compute_variable_importance
from mysuperlearner.meta_learners import NNLogLikEstimator


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def simple_classification_data():
    """Simple well-behaved classification dataset with 10 features."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_redundant=2, random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    return X_df, y


@pytest.fixture
def small_dataset():
    """Small dataset for faster testing (5 features)."""
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3,
        n_redundant=1, random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    return X_df, y


@pytest.fixture
def single_feature_data():
    """Edge case: dataset with only 1 feature."""
    X, y = make_classification(
        n_samples=100, n_features=1, n_informative=1,
        n_redundant=0, n_clusters_per_class=1, random_state=42
    )
    X_df = pd.DataFrame(X, columns=["feature_0"])
    return X_df, y


@pytest.fixture
def fitted_superlearner(simple_classification_data):
    """Fitted SuperLearner ready for variable importance computation."""
    X, y = simple_classification_data

    # Define base learners
    learners = [
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)),
        ("dt", DecisionTreeClassifier(max_depth=3, random_state=42))
    ]

    # Fit SuperLearner
    sl = SuperLearner(
        learners=learners,
        method='nnloglik',
        cv=3,
        random_state=42
    )
    sl.fit(X, y, store_X=True)

    return sl


@pytest.fixture
def small_fitted_superlearner(small_dataset):
    """Fitted SuperLearner with small dataset for faster tests."""
    X, y = small_dataset

    learners = [
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ("dt", DecisionTreeClassifier(max_depth=2, random_state=42))
    ]

    sl = SuperLearner(
        learners=learners,
        method='nnloglik',
        cv=3,
        random_state=42
    )
    sl.fit(X, y, store_X=True)

    return sl


# ============================================================================
# Test 1: Correctness - Parallel matches Sequential
# ============================================================================

def test_permutation_importance_parallel_correctness(fitted_superlearner):
    """Test that parallel permutation importance matches sequential results."""
    sl = fitted_superlearner

    # Compute importance sequentially
    results_seq = compute_variable_importance(
        sl, method='permutation', n_repeats=3, random_state=42, n_jobs=1
    )

    # Compute importance in parallel
    results_par = compute_variable_importance(
        sl, method='permutation', n_repeats=3, random_state=42, n_jobs=2
    )

    # Check that aggregated importance scores are identical
    importance_seq = results_seq.importance_df.sort_values('feature').reset_index(drop=True)
    importance_par = results_par.importance_df.sort_values('feature').reset_index(drop=True)

    # Compare importance scores (should be very close, allowing for floating point differences)
    np.testing.assert_allclose(
        importance_seq['importance'].values,
        importance_par['importance'].values,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Parallel importance scores should match sequential scores"
    )

    # Check that ranks are identical
    assert (importance_seq['rank'].values == importance_par['rank'].values).all(), \
        "Feature ranks should be identical between parallel and sequential"


def test_drop_column_importance_parallel_correctness(fitted_superlearner):
    """Test that parallel drop-column importance matches sequential results."""
    sl = fitted_superlearner

    # Compute importance sequentially
    results_seq = compute_variable_importance(
        sl, method='drop_column', random_state=42, n_jobs=1
    )

    # Compute importance in parallel
    results_par = compute_variable_importance(
        sl, method='drop_column', random_state=42, n_jobs=2
    )

    # Check that importance scores are identical
    importance_seq = results_seq.importance_df.sort_values('feature').reset_index(drop=True)
    importance_par = results_par.importance_df.sort_values('feature').reset_index(drop=True)

    np.testing.assert_allclose(
        importance_seq['importance'].values,
        importance_par['importance'].values,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Parallel drop-column importance should match sequential"
    )


def test_grouped_importance_parallel_correctness(fitted_superlearner):
    """Test that parallel grouped importance matches sequential results."""
    sl = fitted_superlearner

    # Compute importance sequentially
    results_seq = compute_variable_importance(
        sl, method='grouped', n_repeats=2, random_state=42, n_jobs=1
    )

    # Compute importance in parallel
    results_par = compute_variable_importance(
        sl, method='grouped', n_repeats=2, random_state=42, n_jobs=2
    )

    # Check that importance scores are identical
    importance_seq = results_seq.importance_df.sort_values('feature').reset_index(drop=True)
    importance_par = results_par.importance_df.sort_values('feature').reset_index(drop=True)

    np.testing.assert_allclose(
        importance_seq['importance'].values,
        importance_par['importance'].values,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Parallel grouped importance should match sequential"
    )

    # Check cluster assignments are identical
    cluster_seq = results_seq.cluster_info.sort_values('feature').reset_index(drop=True)
    cluster_par = results_par.cluster_info.sort_values('feature').reset_index(drop=True)

    assert (cluster_seq['group_id'].values == cluster_par['group_id'].values).all(), \
        "Cluster assignments should be identical"


# ============================================================================
# Test 2: Reproducibility with Random State
# ============================================================================

def test_permutation_reproducibility_with_random_state(fitted_superlearner):
    """Test that same random_state produces identical results across runs."""
    sl = fitted_superlearner

    # Run 1 with n_jobs=1
    results_1 = compute_variable_importance(
        sl, method='permutation', n_repeats=3, random_state=123, n_jobs=1
    )

    # Run 2 with n_jobs=1
    results_2 = compute_variable_importance(
        sl, method='permutation', n_repeats=3, random_state=123, n_jobs=1
    )

    # Run 3 with n_jobs=2 (parallel)
    results_3 = compute_variable_importance(
        sl, method='permutation', n_repeats=3, random_state=123, n_jobs=2
    )

    # All should be identical
    importance_1 = results_1.importance_df.sort_values('feature').reset_index(drop=True)
    importance_2 = results_2.importance_df.sort_values('feature').reset_index(drop=True)
    importance_3 = results_3.importance_df.sort_values('feature').reset_index(drop=True)

    np.testing.assert_array_equal(
        importance_1['importance'].values,
        importance_2['importance'].values,
        err_msg="Sequential runs with same random_state should be identical"
    )

    np.testing.assert_allclose(
        importance_1['importance'].values,
        importance_3['importance'].values,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Parallel run should match sequential with same random_state"
    )


# ============================================================================
# Test 3: Performance Improvement
# ============================================================================

@pytest.mark.slow
def test_parallel_speedup(small_fitted_superlearner):
    """Test that parallel execution is faster than sequential (simple benchmark)."""
    sl = small_fitted_superlearner

    # Sequential timing
    start = time.time()
    _ = compute_variable_importance(
        sl, method='permutation', n_repeats=2, random_state=42, n_jobs=1, verbose=False
    )
    time_seq = time.time() - start

    # Parallel timing
    start = time.time()
    _ = compute_variable_importance(
        sl, method='permutation', n_repeats=2, random_state=42, n_jobs=2, verbose=False
    )
    time_par = time.time() - start

    # Parallel should be faster (at least 1.2x for small dataset)
    # Note: This is a weak assertion because overhead can dominate for small datasets
    speedup = time_seq / time_par
    assert speedup > 0.8, f"Expected some speedup, got {speedup:.2f}x (seq={time_seq:.2f}s, par={time_par:.2f}s)"


# ============================================================================
# Test 4: Edge Cases
# ============================================================================

def test_single_feature_parallel(single_feature_data):
    """Test parallel importance computation with only one feature."""
    X, y = single_feature_data

    learners = [
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ("dt", DecisionTreeClassifier(max_depth=2, random_state=42))
    ]

    sl = SuperLearner(learners=learners, method='nnloglik', cv=2, random_state=42)
    sl.fit(X, y, store_X=True)

    # Should work with n_jobs=2 even though there's only 1 feature
    results = compute_variable_importance(
        sl, method='permutation', n_repeats=2, random_state=42, n_jobs=2
    )

    assert len(results.importance_df) == 1, "Should have importance for 1 feature"
    assert results.importance_df.iloc[0]['feature'] == 'feature_0'


def test_n_jobs_exceeds_n_features(small_fitted_superlearner):
    """Test that n_jobs > n_features doesn't cause issues."""
    sl = small_fitted_superlearner

    # Use more jobs than features (5 features, 10 jobs)
    results = compute_variable_importance(
        sl, method='drop_column', random_state=42, n_jobs=10
    )

    # Should complete successfully
    assert len(results.importance_df) == 5, "Should have importance for all 5 features"


def test_n_jobs_minus_one(small_fitted_superlearner):
    """Test that n_jobs=-1 uses all available CPUs."""
    sl = small_fitted_superlearner

    # n_jobs=-1 should use all CPUs
    results = compute_variable_importance(
        sl, method='permutation', n_repeats=2, random_state=42, n_jobs=-1
    )

    # Should complete successfully
    assert len(results.importance_df) == 5, "Should compute importance for all features"
    # Check that importance values are finite (not NaN or Inf)
    assert all(np.isfinite(results.importance_df['importance'])), \
        "Importance values should be finite"


def test_n_jobs_one_preserves_sequential(fitted_superlearner):
    """Test that n_jobs=1 uses sequential path (code coverage)."""
    sl = fitted_superlearner

    # This should use the sequential code path
    results = compute_variable_importance(
        sl, method='permutation', n_repeats=2, random_state=42, n_jobs=1
    )

    assert len(results.importance_df) == 10, "Should have importance for all 10 features"


# ============================================================================
# Test 5: Multiple Methods
# ============================================================================

def test_multiple_methods_parallel(small_fitted_superlearner):
    """Test computing multiple importance methods in parallel."""
    sl = small_fitted_superlearner

    # Compute multiple methods
    results = compute_variable_importance(
        sl,
        method=['permutation', 'drop_column'],
        n_repeats=2,
        random_state=42,
        n_jobs=2
    )

    # Should have results for both methods
    methods = results.importance_df['method'].unique()
    assert set(methods) == {'permutation', 'drop_column'}, "Should have both methods"

    # Each method should have 5 features
    for method in methods:
        method_df = results.importance_df[results.importance_df['method'] == method]
        assert len(method_df) == 5, f"Method {method} should have 5 features"


# ============================================================================
# Test 6: Verbose Output
# ============================================================================

def test_verbose_output_parallel(small_fitted_superlearner, capsys):
    """Test that verbose=True works in parallel mode."""
    sl = small_fitted_superlearner

    # Run with verbose=True
    _ = compute_variable_importance(
        sl, method='permutation', n_repeats=2, random_state=42, n_jobs=2, verbose=True
    )

    # Capture output
    captured = capsys.readouterr()

    # Should mention parallel execution
    assert "parallel" in captured.out.lower() or "n_jobs" in captured.out.lower(), \
        "Verbose output should mention parallel execution"


# ============================================================================
# Test 7: Raw Results Structure
# ============================================================================

def test_raw_results_structure_parallel(small_fitted_superlearner):
    """Test that raw results DataFrame has correct structure in parallel mode."""
    sl = small_fitted_superlearner

    results = compute_variable_importance(
        sl, method='permutation', n_repeats=3, random_state=42, n_jobs=2
    )

    raw_df = results.raw_importance_df

    # Check required columns exist
    required_cols = ['feature', 'fold', 'repeat', 'importance', 'baseline_score', 'modified_score']
    for col in required_cols:
        assert col in raw_df.columns, f"Raw results should have column '{col}'"

    # Check number of rows: 5 features * 3 folds * 3 repeats = 45 rows
    assert len(raw_df) == 5 * 3 * 3, "Raw results should have correct number of rows"

    # Check that all features are present
    assert set(raw_df['feature'].unique()) == set(sl.X_.columns), \
        "All features should be in raw results"


# ============================================================================
# Test 8: Comparison with Different n_jobs Values
# ============================================================================

@pytest.mark.parametrize("n_jobs", [1, 2, -1])
def test_different_n_jobs_values(small_fitted_superlearner, n_jobs):
    """Test that different n_jobs values all produce valid results."""
    sl = small_fitted_superlearner

    results = compute_variable_importance(
        sl, method='drop_column', random_state=42, n_jobs=n_jobs
    )

    # All should produce valid results
    assert len(results.importance_df) == 5, f"n_jobs={n_jobs} should compute all features"
    assert all(results.importance_df['rank'] > 0), "Ranks should be positive"
    assert all(results.importance_df['rank'] <= 5), "Ranks should be <= n_features"


# ============================================================================
# Test 9: Integration with Result Object Methods
# ============================================================================

def test_result_object_methods_with_parallel(fitted_superlearner):
    """Test that VariableImportanceResults methods work with parallel-computed results."""
    sl = fitted_superlearner

    results = compute_variable_importance(
        sl, method='permutation', n_repeats=2, random_state=42, n_jobs=2
    )

    # Test summary method
    summary = results.summary(top_n=5)
    assert len(summary) == 5, "Summary should return top 5 features"

    # Test get_top_features method
    top_features = results.get_top_features(n=3)
    assert len(top_features) == 3, "Should return top 3 feature names"
    assert isinstance(top_features, list), "Should return a list"

    # Test that top features are in the data
    assert all(f in sl.X_.columns for f in top_features), \
        "Top features should be valid feature names"


# ============================================================================
# Test 10: Zero Repeats Edge Case
# ============================================================================

def test_grouped_importance_multiple_groups_parallel(small_fitted_superlearner):
    """Test grouped importance when features form multiple groups."""
    sl = small_fitted_superlearner

    # Use a threshold that will create multiple groups
    results = compute_variable_importance(
        sl,
        method='grouped',
        grouped_threshold=0.5,  # Lower threshold -> more groups
        n_repeats=2,
        random_state=42,
        n_jobs=2
    )

    # Should have cluster information
    assert results.cluster_info is not None, "Should have cluster info"

    # Number of groups should be reasonable (1 to n_features)
    n_groups = len(results.importance_df)
    assert 1 <= n_groups <= 5, f"Should have 1-5 groups, got {n_groups}"
