import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mysuperlearner.cv_super_learner import evaluate_super_learner_cv
from mysuperlearner import SuperLearner
from mysuperlearner.results import SuperLearnerCVResults


def test_evaluate_super_learner_cv_basic():
    X, y = make_classification(n_samples=300, n_features=8, random_state=1)
    base_learners = [
        ("rf", RandomForestClassifier(n_estimators=10, random_state=1)),
        ("log", LogisticRegression(max_iter=200))
    ]
    sl = SuperLearner(learners=base_learners, method='nnloglik', cv=3, random_state=1)
    df = evaluate_super_learner_cv(X, y, base_learners, sl, outer_folds=3, random_state=1)
    # expect rows = outer_folds * (1 SuperLearner + 1 DiscreteSL + K base learners)
    assert df.shape[0] == 3 * (1 + 1 + len(base_learners))
    assert {'auc', 'logloss', 'accuracy'}.issubset(set(df.columns))


def test_compare_to_best_with_loss_metrics():
    """Test that compare_to_best correctly identifies best learner for loss metrics (lower is better)."""
    # Create synthetic metrics where one learner has lowest logloss
    metrics_data = []
    for fold in range(1, 4):
        # SuperLearner has moderate logloss
        metrics_data.append({
            'fold': fold, 'learner': 'SuperLearner', 'learner_type': 'super',
            'logloss': 0.5, 'auc': 0.85
        })
        # Learner1 has BEST (lowest) logloss
        metrics_data.append({
            'fold': fold, 'learner': 'Learner1', 'learner_type': 'base',
            'logloss': 0.4, 'auc': 0.80
        })
        # Learner2 has WORST (highest) logloss
        metrics_data.append({
            'fold': fold, 'learner': 'Learner2', 'learner_type': 'base',
            'logloss': 0.7, 'auc': 0.75
        })

    metrics_df = pd.DataFrame(metrics_data)
    results = SuperLearnerCVResults(metrics=metrics_df)
    comparison = results.compare_to_best()

    # Find logloss row
    logloss_row = comparison[comparison['metric'] == 'logloss'].iloc[0]

    # Best base should be Learner1 (lowest logloss)
    assert logloss_row['Best_Base_Name'] == 'Learner1', \
        f"Expected Learner1 as best, got {logloss_row['Best_Base_Name']}"
    assert abs(logloss_row['Best_Base'] - 0.4) < 0.001, \
        f"Expected 0.4 as best logloss, got {logloss_row['Best_Base']}"
    # SuperLearner logloss is 0.5, best base is 0.4
    # Improvement should be negative (0.4 - 0.5 = -0.1) since SL is worse
    assert logloss_row['Improvement'] < 0, \
        f"Expected negative improvement (SL worse than best), got {logloss_row['Improvement']}"


def test_compare_to_best_with_score_metrics():
    """Test that compare_to_best correctly identifies best learner for score metrics (higher is better)."""
    # Create synthetic metrics where one learner has highest AUC
    metrics_data = []
    for fold in range(1, 4):
        # SuperLearner has moderate AUC
        metrics_data.append({
            'fold': fold, 'learner': 'SuperLearner', 'learner_type': 'super',
            'auc': 0.85, 'logloss': 0.5
        })
        # Learner1 has BEST (highest) AUC
        metrics_data.append({
            'fold': fold, 'learner': 'Learner1', 'learner_type': 'base',
            'auc': 0.90, 'logloss': 0.4
        })
        # Learner2 has WORST (lowest) AUC
        metrics_data.append({
            'fold': fold, 'learner': 'Learner2', 'learner_type': 'base',
            'auc': 0.75, 'logloss': 0.6
        })

    metrics_df = pd.DataFrame(metrics_data)
    results = SuperLearnerCVResults(metrics=metrics_df)
    comparison = results.compare_to_best()

    # Find AUC row
    auc_row = comparison[comparison['metric'] == 'auc'].iloc[0]

    # Best base should be Learner1 (highest AUC)
    assert auc_row['Best_Base_Name'] == 'Learner1', \
        f"Expected Learner1 as best, got {auc_row['Best_Base_Name']}"
    assert abs(auc_row['Best_Base'] - 0.90) < 0.001, \
        f"Expected 0.90 as best AUC, got {auc_row['Best_Base']}"
    # SuperLearner AUC is 0.85, best base is 0.90
    # Improvement should be negative (0.85 - 0.90 = -0.05) since SL is worse
    assert auc_row['Improvement'] < 0, \
        f"Expected negative improvement (SL worse than best), got {auc_row['Improvement']}"


def test_compare_to_best_sl_better_than_all():
    """Test that improvement is positive when SuperLearner outperforms all base learners."""
    metrics_data = []
    for fold in range(1, 4):
        # SuperLearner has BEST logloss (lowest)
        metrics_data.append({
            'fold': fold, 'learner': 'SuperLearner', 'learner_type': 'super',
            'logloss': 0.3, 'auc': 0.95
        })
        # Base learners are worse
        metrics_data.append({
            'fold': fold, 'learner': 'Learner1', 'learner_type': 'base',
            'logloss': 0.5, 'auc': 0.85
        })
        metrics_data.append({
            'fold': fold, 'learner': 'Learner2', 'learner_type': 'base',
            'logloss': 0.6, 'auc': 0.80
        })

    metrics_df = pd.DataFrame(metrics_data)
    results = SuperLearnerCVResults(metrics=metrics_df)
    comparison = results.compare_to_best()

    # For logloss (lower is better)
    logloss_row = comparison[comparison['metric'] == 'logloss'].iloc[0]
    # Best base logloss is 0.5, SL is 0.3
    # Improvement = 0.5 - 0.3 = 0.2 (positive, SL is better)
    assert logloss_row['Improvement'] > 0, \
        f"Expected positive improvement for logloss, got {logloss_row['Improvement']}"

    # For AUC (higher is better)
    auc_row = comparison[comparison['metric'] == 'auc'].iloc[0]
    # Best base AUC is 0.85, SL is 0.95
    # Improvement = 0.95 - 0.85 = 0.10 (positive, SL is better)
    assert auc_row['Improvement'] > 0, \
        f"Expected positive improvement for AUC, got {auc_row['Improvement']}"


def test_compare_to_best_improvement_percentage():
    """Test that improvement percentage is calculated correctly for both metric types."""
    metrics_data = []
    for fold in range(1, 4):
        metrics_data.append({
            'fold': fold, 'learner': 'SuperLearner', 'learner_type': 'super',
            'logloss': 0.4, 'auc': 0.80
        })
        metrics_data.append({
            'fold': fold, 'learner': 'Learner1', 'learner_type': 'base',
            'logloss': 0.5, 'auc': 0.80
        })

    metrics_df = pd.DataFrame(metrics_data)
    results = SuperLearnerCVResults(metrics=metrics_df)
    comparison = results.compare_to_best()

    # For logloss: improvement = 0.5 - 0.4 = 0.1, percentage = 0.1 / 0.5 * 100 = 20%
    logloss_row = comparison[comparison['metric'] == 'logloss'].iloc[0]
    expected_pct = (0.5 - 0.4) / 0.5 * 100
    assert abs(logloss_row['Improvement_Pct'] - expected_pct) < 0.01, \
        f"Expected improvement % ~{expected_pct}, got {logloss_row['Improvement_Pct']}"


def test_compare_to_best_with_various_loss_metric_names():
    """Test that various loss metric naming conventions are recognized."""
    # Test different loss metric names
    for metric_name in ['logloss', 'log_loss', 'mse', 'mae', 'rmse', 'cv_risk', 'brier']:
        metrics_data = []
        for fold in range(1, 3):
            metrics_data.append({
                'fold': fold, 'learner': 'SuperLearner', 'learner_type': 'super',
                metric_name: 0.5
            })
            metrics_data.append({
                'fold': fold, 'learner': 'Learner1', 'learner_type': 'base',
                metric_name: 0.3  # Better (lower)
            })

        metrics_df = pd.DataFrame(metrics_data)
        results = SuperLearnerCVResults(metrics=metrics_df)
        comparison = results.compare_to_best()

        row = comparison[comparison['metric'] == metric_name].iloc[0]
        # For loss metrics, best base should have LOWEST value
        assert abs(row['Best_Base'] - 0.3) < 0.001, \
            f"For {metric_name}, expected 0.3 as best (lowest), got {row['Best_Base']}"
