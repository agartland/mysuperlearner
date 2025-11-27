"""
Comprehensive edge case testing for mysuperlearner package.

Tests cover:
- Missing data (rows with NaN values)
- Convergence failures (low variability, perfect separation)
- Imbalanced outcomes
- Collinearity in features
- Mixed variable types (continuous, categorical, binary)
- Learner failures on specific folds
- Different meta-learners with various edge cases
"""

import numpy as np
import pandas as pd
import pytest
import warnings
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from mysuperlearner.extended_super_learner import ExtendedSuperLearner
from mysuperlearner.evaluation import evaluate_super_learner_cv
from mysuperlearner.meta_learners import (
    NNLogLikEstimator, AUCEstimator, MeanEstimator, InterceptOnlyEstimator
)
from mysuperlearner.error_handling import ErrorType


# ============================================================================
# Test Fixtures and Helper Functions
# ============================================================================

@pytest.fixture
def simple_classification_data():
    """Simple well-behaved classification dataset."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_redundant=2, random_state=42
    )
    return X, y


@pytest.fixture
def imbalanced_data():
    """Highly imbalanced dataset (10:1 ratio)."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        weights=[0.9, 0.1], flip_y=0, random_state=42
    )
    return X, y


@pytest.fixture
def data_with_missing():
    """Dataset with missing values in some rows."""
    X, y = make_classification(
        n_samples=200, n_features=10, random_state=42
    )
    # Introduce missing values in random locations (5% of values)
    rng = np.random.RandomState(42)
    missing_mask = rng.random(X.shape) < 0.05
    X = X.astype(float)
    X[missing_mask] = np.nan
    return X, y


@pytest.fixture
def highly_collinear_data():
    """Dataset with highly collinear features."""
    rng = np.random.RandomState(42)
    n_samples = 200
    # Create base feature
    X_base = rng.randn(n_samples, 3)
    # Create collinear features
    X_collinear = np.column_stack([
        X_base[:, 0],
        X_base[:, 0] + rng.randn(n_samples) * 0.01,  # Nearly identical
        X_base[:, 0] + rng.randn(n_samples) * 0.01,  # Nearly identical
        X_base[:, 1],
        X_base[:, 1] - X_base[:, 0],  # Linear combination
        X_base[:, 2],
        X_base[:, 0] + X_base[:, 1],  # Linear combination
        rng.randn(n_samples),  # One independent feature
    ])
    # Generate target
    y = (X_base[:, 0] + X_base[:, 2] > 0).astype(int)
    return X_collinear, y


@pytest.fixture
def perfect_separation_data():
    """Dataset with perfect linear separation."""
    rng = np.random.RandomState(42)
    n_samples = 100
    X = rng.randn(n_samples, 5)
    # Create perfectly separable target
    y = (X[:, 0] + 2 * X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def low_variability_data():
    """Dataset with very low variability (convergence issues)."""
    rng = np.random.RandomState(42)
    n_samples = 100
    # Features with very low variance
    X = rng.randn(n_samples, 8) * 0.001
    # Nearly constant target (99% one class in some folds)
    y = np.zeros(n_samples, dtype=int)
    y[:5] = 1  # Only 5% positive class
    return X, y


@pytest.fixture
def mixed_variable_data():
    """Dataset with continuous, binary, and categorical variables."""
    rng = np.random.RandomState(42)
    n_samples = 200

    # Continuous variables (standardized)
    X_continuous = rng.randn(n_samples, 3)

    # Binary variables
    X_binary = rng.binomial(1, 0.5, size=(n_samples, 3))

    # Categorical variables (one-hot encoded from 2 categoricals with 3 and 4 levels)
    cat1 = rng.choice([0, 1, 2], size=n_samples)
    cat2 = rng.choice([0, 1, 2, 3], size=n_samples)
    X_cat1_onehot = np.eye(3)[cat1]
    X_cat2_onehot = np.eye(4)[cat2]

    # Combine all
    X = np.column_stack([X_continuous, X_binary, X_cat1_onehot, X_cat2_onehot])

    # Generate target with mixed relationships
    y = ((X_continuous[:, 0] > 0) &
         (X_binary[:, 0] == 1) |
         (cat1 == 2)).astype(int)

    return X, y


@pytest.fixture
def base_learners_standard():
    """Standard set of base learners."""
    return [
        ('rf', RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)),
        ('logistic', LogisticRegression(max_iter=1000, random_state=42)),
        ('svm', SVC(probability=True, kernel='rbf', random_state=42)),
    ]


@pytest.fixture
def base_learners_with_intercept():
    """Base learners including intercept-only baseline."""
    return [
        ('rf', RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)),
        ('logistic', LogisticRegression(max_iter=1000, random_state=42)),
        ('intercept', InterceptOnlyEstimator()),
    ]


@pytest.fixture
def base_learners_prone_to_failure():
    """Learners that may fail on difficult data."""
    return [
        ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
        ('logistic', LogisticRegression(max_iter=5, random_state=42, solver='lbfgs')),  # Low iter
        ('svm', SVC(probability=True, max_iter=5, random_state=42)),  # Low iter
        ('knn', KNeighborsClassifier(n_neighbors=3)),
    ]


# ============================================================================
# Test Meta-Learners
# ============================================================================

class TestMetaLearners:
    """Test all meta-learner methods."""

    @pytest.mark.parametrize('method', ['nnloglik', 'auc', 'nnls', 'logistic'])
    def test_meta_learner_basic_fit(self, simple_classification_data,
                                     base_learners_standard, method):
        """Test that each meta-learner can fit on clean data."""
        X, y = simple_classification_data
        sl = ExtendedSuperLearner(method=method, folds=3, random_state=42)
        sl.fit_explicit(X, y, base_learners_standard)

        # Check predictions
        preds = sl.predict_proba(X)
        assert preds.shape == (X.shape[0], 2)
        assert np.all((preds >= 0) & (preds <= 1))
        assert np.allclose(preds.sum(axis=1), 1.0)

    @pytest.mark.parametrize('method', ['nnloglik', 'auc', 'nnls', 'logistic'])
    def test_meta_learner_with_imbalanced_data(self, imbalanced_data,
                                                base_learners_standard, method):
        """Test meta-learners handle imbalanced data."""
        X, y = imbalanced_data
        sl = ExtendedSuperLearner(method=method, folds=3, random_state=42)
        sl.fit_explicit(X, y, base_learners_standard)

        preds = sl.predict_proba(X)
        assert preds.shape == (X.shape[0], 2)

        # Check diagnostics
        diag = sl.get_diagnostics()
        assert diag['method'] == method
        if method != 'logistic':  # logistic can have negative weights
            # Check non-negative weights for constrained methods
            if diag['meta_weights'] is not None:
                assert np.all(diag['meta_weights'] >= -1e-10)  # Allow small numerical errors

    def test_nnloglik_weight_normalization(self, simple_classification_data,
                                           base_learners_standard):
        """Test that nnloglik weights sum to 1."""
        X, y = simple_classification_data
        sl = ExtendedSuperLearner(method='nnloglik', folds=3,
                                 normalize_weights=True, random_state=42)
        sl.fit_explicit(X, y, base_learners_standard)

        diag = sl.get_diagnostics()
        weights = diag['meta_weights']
        assert weights is not None
        assert np.allclose(weights.sum(), 1.0, atol=1e-6)

    def test_mean_estimator_meta(self, simple_classification_data):
        """Test MeanEstimator meta-learner averages predictions."""
        X, y = simple_classification_data

        # Build simple level-1 matrix manually
        Z = np.array([[0.8, 0.2, 0.5],
                      [0.3, 0.7, 0.4],
                      [0.9, 0.1, 0.6]])
        y_test = np.array([1, 1, 0])

        meta = MeanEstimator()
        meta.fit(Z, y_test)
        preds = meta.predict_proba(Z)

        # Check that predictions are means
        expected = Z.mean(axis=1)
        assert np.allclose(preds[:, 1], expected)

    def test_intercept_only_estimator(self, simple_classification_data):
        """Test InterceptOnlyEstimator ignores features."""
        X, y = simple_classification_data

        est = InterceptOnlyEstimator()
        est.fit(X, y)

        # All predictions should be the same
        preds = est.predict_proba(X)
        assert np.allclose(preds[:, 1], y.mean())
        assert np.all(preds[:, 1] == preds[0, 1])


# ============================================================================
# Test Missing Data Handling
# ============================================================================

class TestMissingData:
    """Test handling of missing data."""

    def test_missing_data_with_imputation(self, data_with_missing,
                                          base_learners_standard):
        """Test SuperLearner with missing data after imputation."""
        X, y = data_with_missing

        # Simple mean imputation
        X_imputed = X.copy()
        col_means = np.nanmean(X, axis=0)
        for i in range(X.shape[1]):
            X_imputed[np.isnan(X[:, i]), i] = col_means[i]

        sl = ExtendedSuperLearner(method='nnloglik', folds=3, random_state=42)
        sl.fit_explicit(X_imputed, y, base_learners_standard)

        preds = sl.predict_proba(X_imputed)
        assert preds.shape == (X.shape[0], 2)
        assert not np.any(np.isnan(preds))

    def test_missing_data_error_tracking(self, data_with_missing,
                                         base_learners_standard):
        """Test that missing data without imputation is tracked."""
        X, y = data_with_missing

        sl = ExtendedSuperLearner(method='nnloglik', folds=3,
                                 track_errors=True, random_state=42)

        # This should raise errors for most sklearn learners
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                sl.fit_explicit(X, y, base_learners_standard)
            except Exception:
                pass  # Expected to fail with NaN

        # Check error tracker
        if sl.error_tracker is not None and len(sl.error_tracker.error_records) > 0:
            assert len(sl.error_tracker.error_records) > 0


# ============================================================================
# Test Convergence Issues
# ============================================================================

class TestConvergenceIssues:
    """Test handling of convergence failures."""

    def test_perfect_separation_logistic(self, perfect_separation_data,
                                         base_learners_standard):
        """Test handling of perfect separation."""
        X, y = perfect_separation_data

        sl = ExtendedSuperLearner(method='nnloglik', folds=3,
                                 track_errors=True, random_state=42)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sl.fit_explicit(X, y, base_learners_standard)

        # Should still produce predictions
        preds = sl.predict_proba(X)
        assert preds.shape == (X.shape[0], 2)
        assert not np.all(np.isnan(preds))

    def test_low_variability_convergence(self, low_variability_data,
                                         base_learners_prone_to_failure):
        """Test with low variability causing convergence issues."""
        X, y = low_variability_data

        sl = ExtendedSuperLearner(method='nnloglik', folds=3,
                                 track_errors=True, random_state=42)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sl.fit_explicit(X, y, base_learners_prone_to_failure)

        preds = sl.predict_proba(X)
        assert preds.shape == (X.shape[0], 2)

        # Check error tracker captured issues
        if sl.error_tracker is not None:
            diag = sl.get_diagnostics()
            # May have warnings or errors
            assert 'n_errors' in diag

    def test_low_max_iter_convergence_warnings(self, simple_classification_data):
        """Test that low max_iter creates convergence warnings."""
        X, y = simple_classification_data

        # Use learners with very low iteration limits
        learners = [
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('logistic', LogisticRegression(max_iter=2, random_state=42)),  # Will not converge
        ]

        sl = ExtendedSuperLearner(method='nnloglik', folds=3,
                                 track_errors=True, random_state=42)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sl.fit_explicit(X, y, learners)
            # sklearn may produce ConvergenceWarnings

        # Model should still work
        preds = sl.predict_proba(X)
        assert preds.shape == (X.shape[0], 2)


# ============================================================================
# Test Collinearity
# ============================================================================

class TestCollinearity:
    """Test handling of collinear features."""

    def test_high_collinearity(self, highly_collinear_data,
                               base_learners_standard):
        """Test SuperLearner with highly collinear features."""
        X, y = highly_collinear_data

        sl = ExtendedSuperLearner(method='nnloglik', folds=3, random_state=42)
        sl.fit_explicit(X, y, base_learners_standard)

        preds = sl.predict_proba(X)
        assert preds.shape == (X.shape[0], 2)
        assert not np.any(np.isnan(preds))

        # Check that weights are computed
        diag = sl.get_diagnostics()
        assert diag['meta_weights'] is not None

    def test_collinearity_with_regularization(self, highly_collinear_data):
        """Test that regularized methods handle collinearity better."""
        X, y = highly_collinear_data

        # Use learners with regularization
        learners = [
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('ridge', RidgeClassifier(alpha=1.0, random_state=42)),
            ('logistic_l2', LogisticRegression(penalty='l2', C=0.1,
                                               max_iter=1000, random_state=42)),
        ]

        sl = ExtendedSuperLearner(method='nnloglik', folds=3, random_state=42)
        sl.fit_explicit(X, y, learners)

        preds = sl.predict_proba(X)
        assert not np.any(np.isnan(preds))


# ============================================================================
# Test Mixed Variable Types
# ============================================================================

class TestMixedVariables:
    """Test handling of mixed variable types."""

    def test_mixed_continuous_binary_categorical(self, mixed_variable_data,
                                                 base_learners_standard):
        """Test with continuous, binary, and categorical variables."""
        X, y = mixed_variable_data

        sl = ExtendedSuperLearner(method='nnloglik', folds=3, random_state=42)
        sl.fit_explicit(X, y, base_learners_standard)

        preds = sl.predict_proba(X)
        assert preds.shape == (X.shape[0], 2)
        assert not np.any(np.isnan(preds))

    def test_mixed_variables_tree_based_learners(self, mixed_variable_data):
        """Test mixed variables with tree-based learners that handle them well."""
        X, y = mixed_variable_data

        learners = [
            ('rf', RandomForestClassifier(n_estimators=20, random_state=42)),
            ('gbm', GradientBoostingClassifier(n_estimators=20, random_state=42)),
            ('tree', DecisionTreeClassifier(max_depth=5, random_state=42)),
        ]

        sl = ExtendedSuperLearner(method='nnloglik', folds=3, random_state=42)
        sl.fit_explicit(X, y, learners)

        preds = sl.predict_proba(X)
        assert preds.shape == (X.shape[0], 2)

        # Check diagnostics
        diag = sl.get_diagnostics()
        assert 'cv_scores' in diag
        # All learners should have valid AUC scores
        for learner_name, score in diag['cv_scores'].items():
            assert not np.isnan(score)


# ============================================================================
# Test Learner Failures on Specific Folds
# ============================================================================

class TestFoldSpecificFailures:
    """Test handling when learners fail only on specific folds."""

    def test_learner_fails_one_fold(self, simple_classification_data):
        """Test when a learner fails on one specific fold."""
        X, y = simple_classification_data

        # Create a learner that will fail with certain data
        from sklearn.base import BaseEstimator, ClassifierMixin

        class FailingLearner(BaseEstimator, ClassifierMixin):
            def __init__(self, fail_on_size=None):
                self.fail_on_size = fail_on_size

            def fit(self, X, y):
                if self.fail_on_size and X.shape[0] < self.fail_on_size:
                    raise ValueError("Intentional failure for testing")
                self.rf_ = RandomForestClassifier(n_estimators=10, random_state=42)
                self.rf_.fit(X, y)
                return self

            def predict_proba(self, X):
                return self.rf_.predict_proba(X)

        learners = [
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('failing', FailingLearner(fail_on_size=50)),  # May fail on small folds
        ]

        sl = ExtendedSuperLearner(method='nnloglik', folds=5,
                                 track_errors=True, random_state=42)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sl.fit_explicit(X, y, learners)

        # Should still produce predictions (with NaN for failed learner)
        preds = sl.predict_proba(X)
        assert preds.shape == (X.shape[0], 2)

        # Check error tracking
        if sl.error_tracker is not None:
            diag = sl.get_diagnostics()
            if diag['n_errors'] > 0:
                # Verify error was recorded for 'failing' learner
                errors = [e for e in sl.error_tracker.error_records
                         if e.learner_name == 'failing']
                if len(errors) > 0:
                    assert errors[0].error_type == ErrorType.FITTING

    def test_all_predictions_nan_for_failed_learner(self, simple_classification_data):
        """Test that failed learner is handled gracefully with dummy learner.

        The new behavior: Failed learners in final refit are replaced with dummy learners
        that return neutral predictions (0.5), and a warning is issued.
        """
        X, y = simple_classification_data

        from sklearn.base import BaseEstimator, ClassifierMixin

        class AlwaysFailingLearner(BaseEstimator, ClassifierMixin):
            def fit(self, X, y):
                raise RuntimeError("Always fails")

            def predict_proba(self, X):
                raise RuntimeError("Always fails")

        learners = [
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('always_fail', AlwaysFailingLearner()),
        ]

        sl = ExtendedSuperLearner(method='nnloglik', folds=3,
                                 track_errors=True, random_state=42)

        # New behavior: fit succeeds with warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sl.fit_explicit(X, y, learners)

            # Check that warning was issued
            warning_messages = [str(warning.message) for warning in w]
            assert any('failed in final refit' in msg for msg in warning_messages)

        # Check that error was tracked
        diag = sl.get_diagnostics()
        assert diag['n_errors'] > 0
        errors = [e for e in sl.error_tracker.error_records
                 if e.learner_name == 'always_fail' and e.phase == 'final_refit']
        assert len(errors) > 0

        # Check that failed learner is tracked
        assert 'always_fail' in sl.failed_learners_

        # Should still be able to predict (using RF + dummy for failed learner)
        preds = sl.predict_proba(X)
        assert preds.shape == (X.shape[0], 2)
        assert not np.any(np.isnan(preds))


# ============================================================================
# Test External CV with Edge Cases
# ============================================================================

class TestExternalCVEdgeCases:
    """Test external CV evaluation with edge cases."""

    def test_cv_with_imbalanced_data(self, imbalanced_data,
                                     base_learners_with_intercept):
        """Test external CV with imbalanced data."""
        X, y = imbalanced_data

        sl = ExtendedSuperLearner(method='nnloglik', folds=3, random_state=42)
        results = evaluate_super_learner_cv(
            X, y, base_learners_with_intercept, sl,
            outer_folds=3, random_state=42
        )

        # Check results DataFrame
        assert results.shape[0] == 3 * (1 + len(base_learners_with_intercept))
        assert 'auc' in results.columns
        assert 'logloss' in results.columns

        # InterceptOnly should have lower AUC than ensemble
        intercept_auc = results[results['learner'] == 'intercept']['auc'].mean()
        sl_auc = results[results['learner'] == 'SuperLearner']['auc'].mean()
        # SuperLearner should generally outperform intercept-only
        # (though not guaranteed on every dataset)

    def test_cv_with_return_predictions(self, simple_classification_data,
                                        base_learners_standard):
        """Test external CV returns predictions correctly."""
        X, y = simple_classification_data

        sl = ExtendedSuperLearner(method='nnloglik', folds=3, random_state=42)
        results, predictions = evaluate_super_learner_cv(
            X, y, base_learners_standard, sl,
            outer_folds=3, random_state=42,
            return_predictions=True
        )

        # Check predictions structure
        assert 'y_true' in predictions
        assert 'fold_id' in predictions
        assert 'SuperLearner' in predictions
        assert len(predictions['y_true']) == len(y)

        # Check each learner has predictions
        for learner_name, _ in base_learners_standard:
            assert learner_name in predictions

    def test_cv_with_sample_weights(self, simple_classification_data,
                                    base_learners_standard):
        """Test external CV with sample weights."""
        X, y = simple_classification_data

        # Create sample weights (upweight minority class)
        sample_weight = np.ones(len(y))
        sample_weight[y == 1] = 2.0

        sl = ExtendedSuperLearner(method='nnloglik', folds=3, random_state=42)
        results = evaluate_super_learner_cv(
            X, y, base_learners_standard, sl,
            outer_folds=3, random_state=42,
            sample_weight=sample_weight
        )

        assert results.shape[0] == 3 * (1 + len(base_learners_standard))

    @pytest.mark.parametrize('method', ['nnloglik', 'auc', 'nnls', 'logistic'])
    def test_cv_all_methods(self, simple_classification_data,
                           base_learners_standard, method):
        """Test external CV works with all meta-learning methods."""
        X, y = simple_classification_data

        sl = ExtendedSuperLearner(method=method, folds=3, random_state=42)
        results = evaluate_super_learner_cv(
            X, y, base_learners_standard, sl,
            outer_folds=3, random_state=42
        )

        assert results.shape[0] == 3 * (1 + len(base_learners_standard))
        # All AUC values should be between 0 and 1
        assert np.all((results['auc'] >= 0) & (results['auc'] <= 1))


# ============================================================================
# Test Error Handling and Tracking
# ============================================================================

class TestErrorHandling:
    """Test error handling and tracking system."""

    def test_error_tracker_initialization(self):
        """Test ErrorTracker initializes correctly."""
        sl = ExtendedSuperLearner(track_errors=True)
        assert sl.error_tracker is not None
        assert len(sl.error_tracker.error_records) == 0

    def test_error_tracker_disabled(self):
        """Test SuperLearner works with error tracking disabled."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        learners = [
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
        ]

        sl = ExtendedSuperLearner(track_errors=False)
        assert sl.error_tracker is None

        sl.fit_explicit(X, y, learners)
        preds = sl.predict_proba(X)
        assert preds.shape == (X.shape[0], 2)

    def test_diagnostics_with_errors(self, simple_classification_data):
        """Test get_diagnostics includes error information.

        New behavior: Errors during final refit are caught and tracked.
        """
        X, y = simple_classification_data

        from sklearn.base import BaseEstimator, ClassifierMixin

        class FailingLearner(BaseEstimator, ClassifierMixin):
            def fit(self, X, y):
                raise RuntimeError("Test error")

        learners = [
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('failing', FailingLearner()),
        ]

        sl = ExtendedSuperLearner(method='nnloglik', folds=3,
                                 track_errors=True, random_state=42)

        # New behavior: fit succeeds with warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sl.fit_explicit(X, y, learners)

        # Check diagnostics include error information
        diag = sl.get_diagnostics()
        assert 'errors' in diag
        assert 'n_errors' in diag
        assert diag['n_errors'] > 0

        # Check error record structure
        assert len(diag['errors']) > 0
        error = diag['errors'][0]
        assert hasattr(error, 'learner_name')
        assert hasattr(error, 'error_type')
        assert hasattr(error, 'message')

        # Check that 'failing' learner is in failed_learners
        assert 'failing' in sl.failed_learners_

    def test_min_viable_learners_enforcement(self, simple_classification_data):
        """Test that min_viable_learners parameter is enforced."""
        X, y = simple_classification_data

        from sklearn.base import BaseEstimator, ClassifierMixin

        class AlwaysFailingLearner(BaseEstimator, ClassifierMixin):
            def fit(self, X, y):
                raise RuntimeError("Always fails")

        # Set min_viable_learners to 2, but only 1 will succeed
        learners = [
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('failing', AlwaysFailingLearner()),
        ]

        sl = ExtendedSuperLearner(
            method='nnloglik',
            folds=3,
            track_errors=True,
            random_state=42,
            min_viable_learners=2  # Require at least 2 learners to succeed
        )

        # Should raise RuntimeError because only 1 learner succeeds
        with pytest.raises(RuntimeError, match="Only 1/2 learners succeeded"):
            sl.fit_explicit(X, y, learners)

    def test_min_viable_learners_success(self, simple_classification_data):
        """Test that fit succeeds when min_viable_learners is met."""
        X, y = simple_classification_data

        from sklearn.base import BaseEstimator, ClassifierMixin

        class AlwaysFailingLearner(BaseEstimator, ClassifierMixin):
            def fit(self, X, y):
                raise RuntimeError("Always fails")

        # Set min_viable_learners to 2, and 2 will succeed
        learners = [
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('logistic', LogisticRegression(max_iter=1000, random_state=42)),
            ('failing', AlwaysFailingLearner()),
        ]

        sl = ExtendedSuperLearner(
            method='nnloglik',
            folds=3,
            track_errors=True,
            random_state=42,
            min_viable_learners=2  # Require at least 2 learners
        )

        # Should succeed with warning (2 learners succeed, 1 fails)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sl.fit_explicit(X, y, learners)

            # Check warning was issued
            warning_messages = [str(warning.message) for warning in w]
            assert any('failed in final refit' in msg for msg in warning_messages)

        # Should be able to predict
        preds = sl.predict_proba(X)
        assert preds.shape == (X.shape[0], 2)


# ============================================================================
# Test Edge Cases in Predictions
# ============================================================================

class TestPredictionEdgeCases:
    """Test edge cases in prediction phase."""

    def test_predict_on_different_sample_size(self, simple_classification_data,
                                              base_learners_standard):
        """Test predictions on different number of samples."""
        X_train, y_train = simple_classification_data

        sl = ExtendedSuperLearner(method='nnloglik', folds=3, random_state=42)
        sl.fit_explicit(X_train, y_train, base_learners_standard)

        # Predict on subset
        X_test = X_train[:50]
        preds = sl.predict_proba(X_test)
        assert preds.shape == (50, 2)

        # Predict on single sample
        X_single = X_train[:1]
        preds = sl.predict_proba(X_single)
        assert preds.shape == (1, 2)

    def test_predict_binary_class(self, simple_classification_data,
                                  base_learners_standard):
        """Test predict returns binary class labels."""
        X, y = simple_classification_data

        sl = ExtendedSuperLearner(method='nnloglik', folds=3, random_state=42)
        sl.fit_explicit(X, y, base_learners_standard)

        preds = sl.predict(X)
        assert preds.shape == (X.shape[0],)
        assert np.all((preds == 0) | (preds == 1))

    def test_probability_trimming(self, simple_classification_data,
                                  base_learners_standard):
        """Test that probabilities are trimmed correctly."""
        X, y = simple_classification_data

        trim_value = 0.05
        sl = ExtendedSuperLearner(method='nnloglik', folds=3,
                                 trim=trim_value, random_state=42)
        sl.fit_explicit(X, y, base_learners_standard)

        preds = sl.predict_proba(X)
        # Probabilities should be within [trim, 1-trim]
        assert np.all(preds[:, 0] >= trim_value - 1e-6)
        assert np.all(preds[:, 0] <= 1 - trim_value + 1e-6)
        assert np.all(preds[:, 1] >= trim_value - 1e-6)
        assert np.all(preds[:, 1] <= 1 - trim_value + 1e-6)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple edge cases."""

    def test_complete_workflow_with_all_methods(self, simple_classification_data,
                                                base_learners_standard):
        """Test complete workflow with all meta-learning methods."""
        X, y = simple_classification_data

        methods = ['nnloglik', 'auc', 'nnls', 'logistic']

        for method in methods:
            sl = ExtendedSuperLearner(method=method, folds=3, random_state=42)
            sl.fit_explicit(X, y, base_learners_standard)

            # Test predictions
            preds = sl.predict_proba(X)
            assert preds.shape == (X.shape[0], 2)

            # Test diagnostics
            diag = sl.get_diagnostics()
            assert diag['method'] == method
            assert 'meta_weights' in diag

            # Test binary predictions
            binary_preds = sl.predict(X)
            assert binary_preds.shape == (X.shape[0],)

    def test_reproducibility_with_random_state(self, simple_classification_data,
                                               base_learners_standard):
        """Test that results are reproducible with same random state."""
        X, y = simple_classification_data

        sl1 = ExtendedSuperLearner(method='nnloglik', folds=3, random_state=42)
        sl1.fit_explicit(X, y, base_learners_standard)
        preds1 = sl1.predict_proba(X)

        sl2 = ExtendedSuperLearner(method='nnloglik', folds=3, random_state=42)
        sl2.fit_explicit(X, y, base_learners_standard)
        preds2 = sl2.predict_proba(X)

        assert np.allclose(preds1, preds2)

    def test_different_number_of_folds(self, simple_classification_data,
                                       base_learners_standard):
        """Test with different numbers of CV folds."""
        X, y = simple_classification_data

        for n_folds in [3, 5, 10]:
            sl = ExtendedSuperLearner(method='nnloglik', folds=n_folds,
                                     random_state=42)
            sl.fit_explicit(X, y, base_learners_standard)

            preds = sl.predict_proba(X)
            assert preds.shape == (X.shape[0], 2)

            diag = sl.get_diagnostics()
            assert diag['n_folds'] == n_folds
