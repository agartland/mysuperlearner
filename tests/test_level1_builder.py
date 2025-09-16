import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from mysuperlearner.meta_learners import NNLogLikEstimator


def build_level1_preds(learners, X, y, folds=5, random_state=42):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    n, k = X.shape[0], len(learners)
    Z = np.zeros((n, k))
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]
        for j, (_, clf) in enumerate(learners):
            model = clf.__class__(**{k: v for k, v in clf.get_params().items() if k in clf.get_params()})
            model.set_params(**clf.get_params())
            model.fit(X_tr, y_tr)
            if hasattr(model, 'predict_proba'):
                Z[test_idx, j] = model.predict_proba(X_te)[:, 1]
            elif hasattr(model, 'decision_function'):
                # naive sigmoid on decision function
                from scipy.special import expit
                Z[test_idx, j] = expit(model.decision_function(X_te))
            else:
                Z[test_idx, j] = model.predict(X_te)
    return Z


def test_level1_and_nnloglik():
    X, y = make_classification(n_samples=200, n_features=10, random_state=0)
    learners = [
        ('rf', RandomForestClassifier(n_estimators=10, random_state=0)),
        ('log', LogisticRegression(max_iter=200))
    ]
    Z = build_level1_preds(learners, X, y, folds=5, random_state=0)
    assert Z.shape == (X.shape[0], len(learners))

    # Fit NNLogLik meta-learner on Z
    meta = NNLogLikEstimator(trim=0.01, maxiter=200)
    meta.fit(Z, y)
    preds = meta.predict_proba(Z)[:, 1]
    assert preds.shape[0] == X.shape[0]
