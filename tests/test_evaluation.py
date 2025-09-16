import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mysuperlearner.evaluation import evaluate_super_learner_cv
from mysuperlearner.extended_super_learner import ExtendedSuperLearner


def test_evaluate_super_learner_cv_basic():
    X, y = make_classification(n_samples=300, n_features=8, random_state=1)
    base_learners = [
        ("rf", RandomForestClassifier(n_estimators=10, random_state=1)),
        ("log", LogisticRegression(max_iter=200))
    ]
    sl = ExtendedSuperLearner(method='nnloglik', folds=3, random_state=1)
    df = evaluate_super_learner_cv(X, y, base_learners, sl, outer_folds=3, random_state=1)
    # expect rows = outer_folds * (1 super + K base)
    assert df.shape[0] == 3 * (1 + len(base_learners))
    assert {'auc', 'logloss', 'accuracy'}.issubset(set(df.columns))
