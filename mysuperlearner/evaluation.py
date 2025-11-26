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
    return_predictions: bool = False,
    return_object: bool = False,
):
    """Run outer CV and return a DataFrame of per-fold metrics.

    Parameters
    ----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    base_learners : list of (name, estimator) tuples
        Base learners to include in the ensemble
    super_learner : ExtendedSuperLearner
        SuperLearner instance (will be cloned for each fold)
    outer_folds : int, default=5
        Number of outer cross-validation folds
    random_state : int, optional
        Random seed for reproducibility
    sample_weight : array-like, optional
        Sample weights
    metrics : dict, optional
        Dict mapping metric name -> function(y_true, y_score) -> float
        Default: {'auc', 'logloss', 'accuracy'}
    n_jobs : int, default=1
        Number of parallel jobs
    return_predictions : bool, default=False
        If True, return predictions dict (or include in result object)
    return_object : bool, default=False
        If True, return SuperLearnerCVResults object instead of DataFrame

    Returns
    -------
    results : pd.DataFrame, tuple, or SuperLearnerCVResults
        If return_object=True: SuperLearnerCVResults object
        Elif return_predictions=True: (DataFrame, predictions_dict)
        Else: DataFrame with per-fold metrics

        predictions_dict (when applicable) contains:
            - 'y_true': array of true labels (concatenated across folds)
            - 'fold_id': array indicating fold membership
            - 'test_indices': list of test indices per fold
            - '<learner_name>': predicted probabilities for each learner
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
        local_predictions = {} if return_predictions else None

        # Store predictions if requested
        if return_predictions:
            local_predictions['y_true'] = y_te
            local_predictions['test_idx'] = test_idx
            local_predictions['SuperLearner'] = sl_p

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

            if return_predictions:
                local_predictions[name] = p

            row = {"fold": int(fold_idx), "learner": name, "learner_type": "base"}
            for mname, mfun in metrics.items():
                try:
                    row[mname] = mfun(y_te, p)
                except Exception:
                    row[mname] = np.nan
            local_rows.append(row)

        if return_predictions:
            return local_rows, local_predictions
        else:
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

    # Process results
    if return_predictions or return_object:
        # Separate metrics and predictions
        rows = []
        all_predictions_list = []
        for result in results:
            if return_predictions or return_object:
                rows.extend(result[0])
                all_predictions_list.append(result[1])
            else:
                rows.extend(result)

        # Aggregate predictions across folds
        all_predictions = {'y_true': [], 'fold_id': [], 'test_indices': []}

        # Get learner names from first fold
        learner_names = [k for k in all_predictions_list[0].keys()
                        if k not in ['y_true', 'test_idx']]
        for learner_name in learner_names:
            all_predictions[learner_name] = []

        # Concatenate across folds
        for fold_idx, preds in enumerate(all_predictions_list):
            n_samples = len(preds['y_true'])
            all_predictions['y_true'].extend(preds['y_true'])
            all_predictions['fold_id'].extend([fold_idx] * n_samples)
            all_predictions['test_indices'].append(preds['test_idx'])

            for learner_name in learner_names:
                all_predictions[learner_name].extend(preds[learner_name])

        # Convert to arrays
        for key in all_predictions:
            if key != 'test_indices':
                all_predictions[key] = np.array(all_predictions[key])

        results_df = pd.DataFrame(rows)

        # Return appropriate format
        if return_object:
            from .results import SuperLearnerCVResults
            config = {
                'outer_folds': outer_folds,
                'random_state': random_state,
                'method': super_learner.method,
                'inner_folds': super_learner.folds
            }
            return SuperLearnerCVResults(
                metrics=results_df,
                predictions=all_predictions if return_predictions else None,
                config=config
            )
        else:
            return results_df, all_predictions
    else:
        # flatten results
        rows = [r for fold_res in results for r in fold_res]
        return pd.DataFrame(rows)
