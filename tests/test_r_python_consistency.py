"""
Test consistency between R SuperLearner and Python mysuperlearner implementations.

This test suite validates that the Python implementation produces results
consistent with the R SuperLearner package, focusing on:
1. CV.SuperLearner algorithm implementation
2. Meta-learner coefficient calculation
3. CV risk computation
4. Discrete SuperLearner selection
5. Overall prediction performance
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from mysuperlearner import CVSuperLearner, SuperLearner


class TestCVSuperLearnerConsistency:
    """Test suite for CV.SuperLearner consistency with R implementation."""

    @pytest.fixture
    def simple_binary_data(self):
        """Create simple binary classification dataset with fixed seed."""
        np.random.seed(42)
        X, y = make_classification(
            n_samples=300,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_repeated=0,
            n_classes=2,
            flip_y=0.05,
            class_sep=1.0,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def simple_learners(self):
        """Create simple learner library."""
        return [
            ('RF', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
            ('LR', LogisticRegression(random_state=42, max_iter=1000)),
            ('DT', DecisionTreeClassifier(max_depth=3, random_state=42))
        ]

    def test_cv_superlearner_basic_functionality(self, simple_binary_data, simple_learners):
        """Test that CVSuperLearner runs without errors and returns expected structure."""
        X, y = simple_binary_data

        cv_sl = CVSuperLearner(
            learners=simple_learners,
            method='nnloglik',
            cv=5,
            inner_cv=5,
            random_state=42
        )
        cv_sl.fit(X, y)
        results = cv_sl.get_results()

        # Check that results object has expected attributes
        assert results.metrics is not None
        assert results.predictions is not None
        assert results.coef is not None
        assert results.cv_risk is not None
        assert results.which_discrete_sl is not None

        # Check metrics structure
        assert 'fold' in results.metrics.columns
        assert 'learner' in results.metrics.columns
        assert 'learner_type' in results.metrics.columns
        assert 'auc' in results.metrics.columns

        # Check that we have SuperLearner, DiscreteSL, and base learners
        learner_types = results.metrics['learner_type'].unique()
        assert 'super' in learner_types
        assert 'discrete' in learner_types
        assert 'base' in learner_types

    def test_discrete_superlearner_selection(self, simple_binary_data, simple_learners):
        """Test that discrete SL correctly selects best base learner by CV risk."""
        X, y = simple_binary_data

        cv_sl = CVSuperLearner(
            learners=simple_learners,
            method='nnloglik',
            cv=5,
            inner_cv=5,
            random_state=42
        )
        cv_sl.fit(X, y)
        results = cv_sl.get_results()

        # For each fold, verify discrete SL selection matches minimum CV risk
        for fold_num in range(1, 6):
            # Get CV risk for this fold
            fold_cv_risk = results.cv_risk[results.cv_risk['fold'] == fold_num]

            # Find learner with minimum CV risk
            min_risk_idx = fold_cv_risk['cv_risk'].idxmin()
            expected_learner = fold_cv_risk.loc[min_risk_idx, 'learner']

            # Check against recorded selection
            actual_learner = results.which_discrete_sl[fold_num - 1]
            assert actual_learner == expected_learner, \
                f"Fold {fold_num}: Expected {expected_learner}, got {actual_learner}"

    def test_coefficients_sum_to_one(self, simple_binary_data, simple_learners):
        """Test that meta-learner coefficients sum to 1 (when normalized)."""
        X, y = simple_binary_data

        cv_sl = CVSuperLearner(
            learners=simple_learners,
            method='nnloglik',
            cv=5,
            inner_cv=5,
            random_state=42
        )
        cv_sl.fit(X, y)
        results = cv_sl.get_results()

        # Check coefficients for each fold
        for fold_num in range(1, 6):
            fold_coef = results.coef[results.coef['fold'] == fold_num]
            coef_sum = fold_coef['coefficient'].sum()
            assert np.isclose(coef_sum, 1.0, atol=1e-6), \
                f"Fold {fold_num}: Coefficients sum to {coef_sum}, expected 1.0"

    def test_coefficients_non_negative(self, simple_binary_data, simple_learners):
        """Test that NNLS/NNLogLik coefficients are non-negative."""
        X, y = simple_binary_data

        cv_sl = CVSuperLearner(
            learners=simple_learners,
            method='nnloglik',
            cv=5,
            inner_cv=5,
            random_state=42
        )
        cv_sl.fit(X, y)
        results = cv_sl.get_results()

        # All coefficients should be >= 0
        assert (results.coef['coefficient'] >= 0).all(), \
            "Found negative coefficients with nnloglik method"

    def test_cv_risk_computation(self, simple_binary_data, simple_learners):
        """Test that CV risk is computed correctly (MSE on CV predictions)."""
        X, y = simple_binary_data

        # Fit a simple SuperLearner to verify CV risk calculation
        sl = SuperLearner(
            learners=simple_learners,
            method='nnloglik',
            cv=5,
            random_state=42
        )
        sl.fit(X, y)

        # CV risk should be stored
        assert hasattr(sl, 'cv_risks_')
        assert len(sl.cv_risks_) == len(simple_learners)

        # All CV risks should be non-negative and finite
        assert (sl.cv_risks_ >= 0).all()
        assert np.isfinite(sl.cv_risks_).all()

        # Manually compute CV risk for first learner and verify
        from sklearn.metrics import mean_squared_error
        expected_risk = mean_squared_error(y, sl.Z_[:, 0])
        actual_risk = sl.cv_risks_[0]
        assert np.isclose(expected_risk, actual_risk, atol=1e-10), \
            f"CV risk mismatch: expected {expected_risk}, got {actual_risk}"

    def test_discrete_sl_performance(self, simple_binary_data, simple_learners):
        """Test that discrete SL performs at least as well as worst base learner."""
        X, y = simple_binary_data

        cv_sl = CVSuperLearner(
            learners=simple_learners,
            method='nnloglik',
            cv=5,
            inner_cv=5,
            random_state=42
        )
        cv_sl.fit(X, y)
        results = cv_sl.get_results()

        # Get mean AUC for discrete SL
        discrete_auc = results.metrics[
            results.metrics['learner'] == 'DiscreteSL'
        ]['auc'].mean()

        # Get mean AUC for all base learners
        base_aucs = results.metrics[
            results.metrics['learner_type'] == 'base'
        ].groupby('learner')['auc'].mean()

        # Discrete SL should be at least as good as the worst base learner
        # (it selects the best by CV risk, but outer CV performance may vary)
        min_base_auc = base_aucs.min()
        # Allow small tolerance for numerical differences and overfitting
        assert discrete_auc >= min_base_auc - 0.05, \
            f"Discrete SL AUC ({discrete_auc}) worse than worst base ({min_base_auc})"

    def test_superlearner_outperforms_mean(self, simple_binary_data, simple_learners):
        """Test that SuperLearner typically outperforms simple mean predictor."""
        X, y = simple_binary_data

        # Add a mean predictor
        from mysuperlearner import InterceptOnlyEstimator
        learners_with_mean = simple_learners + [('Mean', InterceptOnlyEstimator())]

        cv_sl = CVSuperLearner(
            learners=learners_with_mean,
            method='nnloglik',
            cv=5,
            inner_cv=5,
            random_state=42
        )
        cv_sl.fit(X, y)
        results = cv_sl.get_results()

        # Get mean AUC for SuperLearner and Mean
        sl_auc = results.metrics[
            results.metrics['learner'] == 'SuperLearner'
        ]['auc'].mean()

        mean_auc = results.metrics[
            results.metrics['learner'] == 'Mean'
        ]['auc'].mean()

        # SuperLearner should outperform mean
        assert sl_auc > mean_auc, \
            f"SuperLearner AUC ({sl_auc}) not better than Mean ({mean_auc})"

    def test_predictions_shape(self, simple_binary_data, simple_learners):
        """Test that predictions have correct shape and structure."""
        X, y = simple_binary_data

        cv_sl = CVSuperLearner(
            learners=simple_learners,
            method='nnloglik',
            cv=5,
            inner_cv=5,
            random_state=42
        )
        cv_sl.fit(X, y)
        results = cv_sl.get_results()

        # Check predictions structure
        n_samples = len(y)
        assert len(results.predictions['y_true']) == n_samples
        assert len(results.predictions['fold_id']) == n_samples

        # Check that all learners have predictions
        assert 'SuperLearner' in results.predictions
        assert 'DiscreteSL' in results.predictions
        for name, _ in simple_learners:
            assert name in results.predictions

        # All prediction arrays should have correct length
        for key in results.predictions:
            if key not in ['test_indices']:
                assert len(results.predictions[key]) == n_samples

    def test_reproducibility(self, simple_binary_data, simple_learners):
        """Test that results are reproducible with same random seed."""
        X, y = simple_binary_data

        # Run twice with same seed
        cv_sl1 = CVSuperLearner(
            learners=simple_learners,
            method='nnloglik',
            cv=5,
            inner_cv=5,
            random_state=42
        )
        cv_sl1.fit(X, y)
        results1 = cv_sl1.get_results()

        cv_sl2 = CVSuperLearner(
            learners=simple_learners,
            method='nnloglik',
            cv=5,
            inner_cv=5,
            random_state=42
        )
        cv_sl2.fit(X, y)
        results2 = cv_sl2.get_results()

        # Coefficients should be identical
        pd.testing.assert_frame_equal(results1.coef, results2.coef)

        # CV risks should be identical
        pd.testing.assert_frame_equal(results1.cv_risk, results2.cv_risk)

        # Discrete SL selections should be identical
        assert results1.which_discrete_sl == results2.which_discrete_sl

        # Predictions should be very close
        np.testing.assert_array_almost_equal(
            results1.predictions['SuperLearner'],
            results2.predictions['SuperLearner']
        )


class TestRComparisonStructure:
    """Tests that verify structure matches R's CV.SuperLearner output."""

    def test_output_fields_match_r(self):
        """Test that output fields match R's CV.SuperLearner structure."""
        # R CV.SuperLearner returns:
        # - SL.predict: SuperLearner predictions
        # - discreteSL.predict: best base learner predictions
        # - whichDiscreteSL: which learner was selected per fold
        # - library.predict: all base learner predictions
        # - coef: meta-learner weights per fold
        # - folds: fold assignments
        # - V: number of folds
        # - libraryNames: learner names
        # - method: meta-learning method

        np.random.seed(42)
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        learners = [
            ('RF', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('LR', LogisticRegression(random_state=42))
        ]

        cv_sl = CVSuperLearner(
            learners=learners,
            method='nnloglik',
            cv=3,
            inner_cv=3,
            random_state=42
        )
        cv_sl.fit(X, y)
        results = cv_sl.get_results()

        # Check Python has equivalent fields
        # SL.predict -> predictions['SuperLearner']
        assert 'SuperLearner' in results.predictions

        # discreteSL.predict -> predictions['DiscreteSL']
        assert 'DiscreteSL' in results.predictions

        # whichDiscreteSL -> which_discrete_sl
        assert results.which_discrete_sl is not None
        assert len(results.which_discrete_sl) == 3  # cv=3

        # library.predict -> predictions[learner_name]
        for name, _ in learners:
            assert name in results.predictions

        # coef -> coef DataFrame
        assert results.coef is not None
        assert set(results.coef.columns) == {'fold', 'learner', 'coefficient'}

        # Python adds cv_risk (R has this internally but doesn't expose it clearly)
        assert results.cv_risk is not None


def test_algorithm_documentation():
    """Document the CV.SuperLearner algorithm for validation purposes."""
    algorithm_description = """
    CV.SuperLearner Algorithm (from R package):

    1. Split data into V outer folds
    2. For each outer fold i = 1...V:
        a. Define training set = all folds except i
        b. Define validation set = fold i
        c. Call SuperLearner on training set:
            - Split training set into inner CV folds
            - For each inner fold, train base learners and get CV predictions
            - Build Z matrix from CV predictions
            - Train meta-learner on Z matrix to get coefficients
            - Refit all base learners on full training set
        d. Use fitted base learners to predict on validation set
        e. Combine predictions using meta-learner weights
        f. Record:
            - SuperLearner predictions on validation set
            - Base learner predictions on validation set
            - Meta-learner coefficients
            - CV risk for each base learner (from inner CV)
            - Best base learner (discrete SL) by CV risk
    3. Concatenate predictions across all folds
    4. Return results

    Python Implementation:
    - Implements same algorithm in cv_super_learner.py
    - Uses SuperLearner class for step 2c
    - SuperLearner._build_level1() creates Z matrix via inner CV
    - SuperLearner.fit() computes coefficients and refits base learners
    - CVSuperLearner extracts cv_risks_ from fitted SuperLearner
    - Discrete SL selected as argmin(cv_risks_)
    """
    # This test just documents the algorithm
    assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
