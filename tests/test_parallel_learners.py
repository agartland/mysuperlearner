"""
Tests for learner-level parallelization (n_jobs_learners parameter).
"""
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification

from mysuperlearner import SuperLearner, CVSuperLearner


class TestLearnerParallelization:
    """Test suite for n_jobs_learners parallelization."""

    @pytest.fixture
    def simple_data(self):
        """Create simple classification dataset."""
        X, y = make_classification(
            n_samples=200, n_features=10, n_informative=5,
            n_redundant=2, random_state=42
        )
        return X, y

    @pytest.fixture
    def base_learners(self):
        """Create list of base learners."""
        return [
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('dt', DecisionTreeClassifier(max_depth=3, random_state=42)),
            ('svm', SVC(probability=True, random_state=42))
        ]

    def test_sequential_vs_parallel_identical_results(self, simple_data, base_learners):
        """Test that n_jobs_learners=1 and n_jobs_learners=2 give identical results."""
        X, y = simple_data

        # Sequential
        sl_seq = SuperLearner(
            learners=base_learners,
            method='nnloglik',
            cv=3,
            random_state=42,
            n_jobs_learners=1
        )
        sl_seq.fit(X, y)
        preds_seq = sl_seq.predict_proba(X)
        coef_seq = sl_seq.meta_weights_
        cv_risks_seq = sl_seq.cv_risks_

        # Parallel
        sl_par = SuperLearner(
            learners=base_learners,
            method='nnloglik',
            cv=3,
            random_state=42,
            n_jobs_learners=2
        )
        sl_par.fit(X, y)
        preds_par = sl_par.predict_proba(X)
        coef_par = sl_par.meta_weights_
        cv_risks_par = sl_par.cv_risks_

        # Assertions - predictions should be very close
        np.testing.assert_allclose(preds_seq, preds_par, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(coef_seq, coef_par, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(cv_risks_seq, cv_risks_par, rtol=1e-10, atol=1e-10)

    def test_reproducibility_with_random_state(self, simple_data, base_learners):
        """Test that same random_state gives same results across runs."""
        X, y = simple_data

        # Run 1
        sl1 = SuperLearner(
            learners=base_learners,
            method='nnloglik',
            cv=3,
            random_state=123,
            n_jobs_learners=2
        )
        sl1.fit(X, y)
        preds1 = sl1.predict_proba(X)
        coef1 = sl1.meta_weights_

        # Run 2
        sl2 = SuperLearner(
            learners=base_learners,
            method='nnloglik',
            cv=3,
            random_state=123,
            n_jobs_learners=2
        )
        sl2.fit(X, y)
        preds2 = sl2.predict_proba(X)
        coef2 = sl2.meta_weights_

        # Should be identical
        np.testing.assert_allclose(preds1, preds2, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(coef1, coef2, rtol=1e-10, atol=1e-10)

    def test_cv_superlearner_learner_parallelization(self, simple_data, base_learners):
        """Test CVSuperLearner with n_jobs_learners."""
        X, y = simple_data

        # Sequential
        cv_sl_seq = CVSuperLearner(
            learners=base_learners,
            method='nnloglik',
            cv=3,
            inner_cv=3,
            random_state=42,
            n_jobs=1,
            n_jobs_learners=1
        )
        cv_sl_seq.fit(X, y)
        results_seq = cv_sl_seq.get_results()

        # Parallel learners
        cv_sl_par = CVSuperLearner(
            learners=base_learners,
            method='nnloglik',
            cv=3,
            inner_cv=3,
            random_state=42,
            n_jobs=1,
            n_jobs_learners=2
        )
        cv_sl_par.fit(X, y)
        results_par = cv_sl_par.get_results()

        # Results should be very close (allowing small numerical differences)
        np.testing.assert_allclose(
            results_seq.cv_risk['cv_risk'].values,
            results_par.cv_risk['cv_risk'].values,
            rtol=1e-8
        )

    def test_nested_parallelism_warning(self, simple_data, base_learners):
        """Test that using both n_jobs and n_jobs_learners raises warning."""
        X, y = simple_data

        with pytest.warns(UserWarning, match="Both n_jobs.*and n_jobs_learners"):
            cv_sl = CVSuperLearner(
                learners=base_learners,
                cv=3,
                n_jobs=2,
                n_jobs_learners=2
            )
            cv_sl.fit(X, y)

    def test_error_handling_preserved(self, simple_data):
        """Test that error handling works correctly with parallelization."""
        X, y = simple_data

        # Learner that will fail
        class FailingLearner:
            def fit(self, X, y):
                raise ValueError("Intentional failure")
            def predict_proba(self, X):
                return np.zeros((len(X), 2))

        learners = [
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('fail', FailingLearner()),
            ('dt', DecisionTreeClassifier(random_state=42))
        ]

        sl = SuperLearner(
            learners=learners,
            method='nnloglik',
            cv=3,
            n_jobs_learners=2,
            track_errors=True
        )

        sl.fit(X, y)

        # Check that failing learner was tracked
        assert 'fail' in sl.failed_learners_
        assert len(sl.base_learners_full_) == 3  # All learners present (with dummy for failed)

    def test_learner_timings_recorded(self, simple_data, base_learners):
        """Test that learner timings are recorded correctly with parallelization."""
        X, y = simple_data

        sl = SuperLearner(
            learners=base_learners,
            method='nnloglik',
            cv=3,
            n_jobs_learners=2
        )
        sl.fit(X, y)

        assert hasattr(sl, 'learner_timings_')
        # Should have timing for each learner × each fold
        assert len(sl.learner_timings_) == 3 * len(base_learners)  # 3 folds × K learners

    def test_different_n_jobs_learners_values(self, simple_data, base_learners):
        """Test that different n_jobs_learners values all work correctly."""
        X, y = simple_data

        for n_jobs_learners in [1, 2, -1]:
            sl = SuperLearner(
                learners=base_learners,
                method='nnloglik',
                cv=3,
                random_state=42,
                n_jobs_learners=n_jobs_learners
            )
            sl.fit(X, y)
            preds = sl.predict_proba(X)

            # Should produce valid predictions
            assert preds.shape == (len(X), 2)
            assert np.all(preds >= 0) and np.all(preds <= 1)
            assert np.allclose(preds.sum(axis=1), 1.0)

    def test_single_learner_edge_case(self, simple_data):
        """Test parallelization with just one learner."""
        X, y = simple_data

        learners = [('lr', LogisticRegression(max_iter=1000, random_state=42))]

        sl = SuperLearner(
            learners=learners,
            method='nnloglik',
            cv=3,
            n_jobs_learners=2
        )
        sl.fit(X, y)
        preds = sl.predict_proba(X)

        # Should still work correctly
        assert preds.shape == (len(X), 2)

    def test_parallel_with_sample_weights(self, simple_data, base_learners):
        """Test that parallel execution works with sample weights."""
        X, y = simple_data
        sample_weight = np.random.rand(len(y))

        # Sequential
        sl_seq = SuperLearner(
            learners=base_learners,
            method='nnloglik',
            cv=3,
            random_state=42,
            n_jobs_learners=1
        )
        sl_seq.fit(X, y, sample_weight=sample_weight)
        preds_seq = sl_seq.predict_proba(X)

        # Parallel
        sl_par = SuperLearner(
            learners=base_learners,
            method='nnloglik',
            cv=3,
            random_state=42,
            n_jobs_learners=2
        )
        sl_par.fit(X, y, sample_weight=sample_weight)
        preds_par = sl_par.predict_proba(X)

        # Should produce similar results
        np.testing.assert_allclose(preds_seq, preds_par, rtol=1e-9, atol=1e-9)

    def test_verbose_mode_with_parallelization(self, simple_data, base_learners, capsys):
        """Test that verbose mode works with parallelization."""
        X, y = simple_data

        sl = SuperLearner(
            learners=base_learners,
            method='nnloglik',
            cv=3,
            n_jobs_learners=2,
            verbose=True
        )
        sl.fit(X, y)

        # Verbose should have printed something (timing messages)
        captured = capsys.readouterr()
        assert len(captured.out) > 0  # Some output was produced

    def test_z_matrix_assembly_parallel(self, simple_data, base_learners):
        """Test that Z matrix is correctly assembled in parallel mode."""
        X, y = simple_data

        # Sequential
        sl_seq = SuperLearner(
            learners=base_learners,
            method='nnloglik',
            cv=3,
            random_state=42,
            n_jobs_learners=1
        )
        sl_seq.fit(X, y)
        Z_seq = sl_seq.Z_

        # Parallel
        sl_par = SuperLearner(
            learners=base_learners,
            method='nnloglik',
            cv=3,
            random_state=42,
            n_jobs_learners=2
        )
        sl_par.fit(X, y)
        Z_par = sl_par.Z_

        # Z matrices should be identical
        np.testing.assert_allclose(Z_seq, Z_par, rtol=1e-10, atol=1e-10)

    def test_cv_predictions_parallel(self, simple_data, base_learners):
        """Test that CV predictions are correct in parallel mode."""
        X, y = simple_data

        # Sequential
        sl_seq = SuperLearner(
            learners=base_learners,
            method='nnloglik',
            cv=3,
            random_state=42,
            n_jobs_learners=1
        )
        sl_seq.fit(X, y)
        cv_preds_seq = sl_seq.cv_predictions_

        # Parallel
        sl_par = SuperLearner(
            learners=base_learners,
            method='nnloglik',
            cv=3,
            random_state=42,
            n_jobs_learners=2
        )
        sl_par.fit(X, y)
        cv_preds_par = sl_par.cv_predictions_

        # CV predictions should match
        for seq_pred, par_pred in zip(cv_preds_seq, cv_preds_par):
            np.testing.assert_allclose(seq_pred, par_pred, rtol=1e-10, atol=1e-10)
