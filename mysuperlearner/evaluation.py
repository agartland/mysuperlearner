"""
Outer CV evaluator mimicking R's CV.SuperLearner for binary classification.
Provides `evaluate_super_learner_cv` which runs an outer Stratified K-fold,
fits the provided ExtendedSuperLearner (using explicit builder), refits base
learners on inner training folds, and computes metrics for each fold for both
base learners and the ensemble.
"""
from copy import deepcopy
from typing import List, Tuple, Callable, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from scipy.special import expit

from .extended_super_learner import ExtendedSuperLearner


def _get_proba_fallback(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return expit(model.decision_function(X))
    return np.asarray(model.predict(X), dtype=float)


def evaluate_super_learner_cv(
    X,
    y,
    base_learners: List[Tuple[str, Any]],
    super_learner: ExtendedSuperLearner,
    outer_folds: int = 5,
    random_state: Optional[int] = None,
    sample_weight: Optional[Any] = None,
    metrics: Optional[Dict[str, Callable]] = None,
    n_jobs: Optional[int] = 1,
):
    """Run outer CV and return a DataFrame of per-fold metrics.

    metrics: dict mapping metric name -> function(y_true, y_score) -> float
    """
    if metrics is None:
        metrics = {
            "auc": lambda y, p: float(roc_auc_score(y, p)),
            "logloss": lambda y, p: float(log_loss(y, p, labels=[0, 1])),
            "accuracy": lambda y, p: float(accuracy_score(y, (p >= 0.5).astype(int))),
        }

    X_arr, y_arr = check_X_y(X, y)
    skf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)

    def _run_fold(fold_idx, train_idx, test_idx):
        """Work for a single outer fold: fit SL on train, evaluate on test, return rows."""
        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]
        sw_tr = None
        if sample_weight is not None:
            sw_tr = np.asarray(sample_weight)[train_idx]

        sl = clone(super_learner)
        # Use explicit builder to ensure parity with R
        sl.fit_explicit(X_tr, y_tr, base_learners, sample_weight=sw_tr)

        try:
            sl_p = sl.predict_proba(X_te)[:, 1]
        except Exception:
            sl_p = np.zeros_like(y_te, dtype=float)

        local_rows = []
        # record SL metrics
        row = {"fold": int(fold_idx), "learner": "SuperLearner", "learner_type": "super"}
        for mname, mfun in metrics.items():
            try:
                row[mname] = mfun(y_te, sl_p)
            except Exception:
                row[mname] = np.nan
        local_rows.append(row)

        # Evaluate each base learner (they were refit on full training data by fit_explicit)
        for name, mdl in sl.base_learners_full_:
            try:
                p = _get_proba_fallback(mdl, X_te)
            except Exception:
                p = np.zeros_like(y_te, dtype=float)
            row = {"fold": int(fold_idx), "learner": name, "learner_type": "base"}
            for mname, mfun in metrics.items():
                try:
                    row[mname] = mfun(y_te, p)
                except Exception:
                    row[mname] = np.nan
            local_rows.append(row)

        return local_rows

    # Prepare fold inputs
    fold_inputs = []
    fold_num = 0
    for train_idx, test_idx in skf.split(X_arr, y_arr):
        fold_num += 1
        fold_inputs.append((fold_num, train_idx, test_idx))

    # Run folds either in parallel or sequentially
    if n_jobs is None or n_jobs == 1:
        results = [ _run_fold(fi[0], fi[1], fi[2]) for fi in fold_inputs ]
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(_run_fold)(fi[0], fi[1], fi[2]) for fi in fold_inputs)

    # flatten results
    rows = [r for fold_res in results for r in fold_res]
    return pd.DataFrame(rows)
