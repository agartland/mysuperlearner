"""
Outer CV evaluator mimicking R's CV.SuperLearner for binary classification.
Provides `CVSuperLearner` class and `evaluate_super_learner_cv` function (deprecated).
Runs an outer Stratified K-fold, fits the provided SuperLearner, refits base
learners on inner training folds, and computes metrics for each fold for both
base learners and the ensemble.
"""
from copy import deepcopy
from typing import List, Tuple, Callable, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from scipy.special import expit

from .super_learner import SuperLearner


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
    super_learner: SuperLearner,
    outer_folds: int = 5,
    random_state: Optional[int] = None,
    sample_weight: Optional[Any] = None,
    metrics: Optional[Dict[str, Callable]] = None,
    n_jobs: Optional[int] = 1,
    n_jobs_learners: Optional[int] = 1,
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
    super_learner : SuperLearner
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

    # Get verbose setting from super_learner
    verbose = getattr(super_learner, 'verbose', False)

    # Handle both int and CV splitter objects for outer CV
    if isinstance(outer_folds, int):
        cv_splitter = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
    else:
        # Assume it's a cross-validation splitter
        cv_splitter = outer_folds

    def _run_fold(fold_idx, train_idx, test_idx):
        """Work for a single outer fold: fit SL on train, evaluate on test, return rows."""
        import time
        fold_start_time = time.time()  # New: track fold timing

        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]
        sw_tr = None
        if sample_weight is not None:
            sw_tr = np.asarray(sample_weight)[train_idx]

        sl = clone(super_learner)
        # Set learners for this fold and fit
        sl.learners = base_learners
        sl.n_jobs_learners = n_jobs_learners  # Pass through learner parallelization setting
        sl.fit(X_tr, y_tr, sample_weight=sw_tr)

        try:
            sl_p = sl.predict_proba(X_te)[:, 1]
        except Exception:
            sl_p = np.zeros_like(y_te, dtype=float)

        local_rows = []
        local_predictions = {} if return_predictions else None

        # Extract coefficients and cv_risks from fitted SuperLearner
        fold_coef = getattr(sl, 'meta_weights_', None)
        fold_cv_risks = getattr(sl, 'cv_risks_', None)

        # Determine discrete SuperLearner (best base learner by CV risk)
        discrete_sl_name = None
        discrete_sl_p = None
        if fold_cv_risks is not None and len(fold_cv_risks) > 0:
            # Find learner with minimum CV risk
            best_idx = np.argmin(fold_cv_risks)
            discrete_sl_name = sl.base_learner_names_[best_idx]
            # Get predictions from that learner
            try:
                discrete_sl_p = _get_proba_fallback(sl.base_learners_full_[best_idx][1], X_te)
            except Exception:
                discrete_sl_p = np.zeros_like(y_te, dtype=float)

        # Store predictions if requested
        if return_predictions:
            local_predictions['y_true'] = y_te
            local_predictions['test_idx'] = test_idx
            local_predictions['SuperLearner'] = sl_p
            if discrete_sl_p is not None:
                local_predictions['DiscreteSL'] = discrete_sl_p

        # record SL metrics
        row = {"fold": int(fold_idx), "learner": "SuperLearner", "learner_type": "super"}
        for mname, mfun in metrics.items():
            try:
                row[mname] = mfun(y_te, sl_p)
            except Exception:
                row[mname] = np.nan
        local_rows.append(row)

        # Add discrete SL metrics
        if discrete_sl_p is not None:
            row = {"fold": int(fold_idx), "learner": "DiscreteSL", "learner_type": "discrete"}
            for mname, mfun in metrics.items():
                try:
                    row[mname] = mfun(y_te, discrete_sl_p)
                except Exception:
                    row[mname] = np.nan
            local_rows.append(row)

        # Evaluate each base learner (they were refit on full training data by fit)
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

        # New: Create fold timing summary
        learner_timings = getattr(sl, 'learner_timings_', [])
        fold_total_time = time.time() - fold_start_time
        fold_timing_summary = {
            'outer_fold': fold_idx,
            'learner_timings': learner_timings,
            'total_time': fold_total_time,
            'start_time': fold_start_time,
            'end_time': time.time()
        }

        if return_predictions:
            return local_rows, local_predictions, fold_coef, fold_cv_risks, discrete_sl_name, fold_timing_summary
        else:
            return local_rows, fold_coef, fold_cv_risks, discrete_sl_name, fold_timing_summary

    # Prepare fold inputs
    fold_inputs = []
    fold_num = 0
    for train_idx, test_idx in cv_splitter.split(X_arr, y_arr):
        fold_num += 1
        fold_inputs.append((fold_num, train_idx, test_idx))

    # New: Initialize progress tracking
    if verbose:
        import time
        from .progress import print_progress_header, print_learner_summary, print_overall_progress, print_final_summary
        n_learners = len(base_learners)
        n_outer_folds = len(fold_inputs)
        print_progress_header(n_outer_folds, n_learners)
        cv_start_time = time.time()

    # Run folds either in parallel or sequentially
    if n_jobs is None or n_jobs == 1:
        # Sequential mode - display progress in real-time
        results = []
        fold_summaries = []
        for fi in fold_inputs:
            if verbose:
                print(f"\nFitting fold {fi[0]}/{len(fold_inputs)}...")
            result = _run_fold(fi[0], fi[1], fi[2])
            results.append(result)

            if verbose:
                # Extract timing summary (last element of result tuple)
                fold_timing = result[-1]
                fold_summaries.append(fold_timing)

                # Display per-fold summary
                print_learner_summary(fold_timing['learner_timings'], fold_timing['outer_fold'] - 1)
                print_overall_progress(
                    len(fold_summaries), len(fold_inputs),
                    cv_start_time, fold_summaries
                )
    else:
        # Parallel mode - collect all results, then display
        if verbose:
            print(f"\nRunning {len(fold_inputs)} folds in parallel with {n_jobs} workers...")
        results = Parallel(n_jobs=n_jobs)(delayed(_run_fold)(fi[0], fi[1], fi[2]) for fi in fold_inputs)

        if verbose:
            # Extract and display timing after parallel execution completes
            fold_summaries = [result[-1] for result in results]
            total_time = time.time() - cv_start_time
            print_final_summary(fold_summaries, total_time)

    # Process results
    if return_predictions or return_object:
        # Separate metrics, predictions, coefficients, cv_risks, discrete SL info, and timing
        rows = []
        all_predictions_list = []
        all_coefs = []
        all_cv_risks = []
        discrete_sl_selections = []
        timing_summaries = []  # New

        for result in results:
            if return_predictions or return_object:
                rows.extend(result[0])
                all_predictions_list.append(result[1])
                all_coefs.append(result[2])
                all_cv_risks.append(result[3])
                discrete_sl_selections.append(result[4])
                timing_summaries.append(result[5])  # New
            else:
                rows.extend(result[0])
                all_coefs.append(result[1])
                all_cv_risks.append(result[2])
                discrete_sl_selections.append(result[3])
                timing_summaries.append(result[4])  # New

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

        # Build coefficient dataframe
        coef_df = None
        if all_coefs and all_coefs[0] is not None:
            coef_data = []
            learner_names_coef = [n for n, _ in base_learners]
            for fold_idx, coef in enumerate(all_coefs):
                if coef is not None:
                    for learner_name, weight in zip(learner_names_coef, coef):
                        coef_data.append({
                            'fold': fold_idx + 1,
                            'learner': learner_name,
                            'coefficient': weight
                        })
            if coef_data:
                coef_df = pd.DataFrame(coef_data)

        # Build CV risk dataframe
        cv_risk_df = None
        if all_cv_risks and all_cv_risks[0] is not None:
            cv_risk_data = []
            learner_names_risk = [n for n, _ in base_learners]
            for fold_idx, risks in enumerate(all_cv_risks):
                if risks is not None:
                    for learner_name, risk in zip(learner_names_risk, risks):
                        cv_risk_data.append({
                            'fold': fold_idx + 1,
                            'learner': learner_name,
                            'cv_risk': risk
                        })
            if cv_risk_data:
                cv_risk_df = pd.DataFrame(cv_risk_data)

        # New: Build timing dataframe
        timing_df = None
        if timing_summaries:
            timing_data = []
            for fold_summary in timing_summaries:
                for record in fold_summary['learner_timings']:
                    timing_data.append({
                        'outer_fold': fold_summary['outer_fold'],
                        'learner': record['learner_name'],
                        'inner_fold': record['inner_fold_idx'],
                        'fit_time': record['fit_time'],
                        'error': record.get('error', None)
                    })
            if timing_data:
                timing_df = pd.DataFrame(timing_data)

        # Return appropriate format
        if return_object:
            from .results import SuperLearnerCVResults
            config = {
                'outer_folds': outer_folds,
                'random_state': random_state,
                'method': super_learner.method,
                'inner_folds': super_learner.cv
            }
            return SuperLearnerCVResults(
                metrics=results_df,
                predictions=all_predictions if return_predictions else None,
                config=config,
                coef=coef_df,
                cv_risk=cv_risk_df,
                which_discrete_sl=discrete_sl_selections,
                timing=timing_df  # New
            )
        else:
            return results_df, all_predictions
    else:
        # flatten results
        rows = []
        all_coefs = []
        all_cv_risks = []
        discrete_sl_selections = []
        for result in results:
            rows.extend(result[0])
            all_coefs.append(result[1])
            all_cv_risks.append(result[2])
            discrete_sl_selections.append(result[3])
        return pd.DataFrame(rows)


class CVSuperLearner(BaseEstimator, ClassifierMixin):
    """
    Cross-validated Super Learner for unbiased performance evaluation.

    Equivalent to R's CV.SuperLearner function but with sklearn-style API.

    Parameters
    ----------
    learners : list of (name, estimator) tuples
        Base learners to include in the ensemble
    method : str, default='nnloglik'
        Meta-learning strategy ('nnloglik', 'nnls', 'auc', 'logistic')
    cv : int, default=5
        Number of outer cross-validation folds
    inner_cv : int, default=5
        Number of inner CV folds for meta-learner training
    random_state : int, optional
        Random seed for reproducibility
    n_jobs : int, default=1
        Number of parallel jobs for outer CV
    verbose : bool, default=False
        Whether to print progress messages

    Attributes
    ----------
    results_ : SuperLearnerCVResults
        Results object containing metrics and predictions

    Examples
    --------
    >>> from mysuperlearner import CVSuperLearner
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>>
    >>> learners = [
    ...     ('RF', RandomForestClassifier(random_state=42)),
    ...     ('LR', LogisticRegression(random_state=42))
    ... ]
    >>> cv_sl = CVSuperLearner(learners=learners, method='nnloglik', cv=5)
    >>> cv_sl.fit(X_train, y_train)
    >>> results = cv_sl.get_results()
    >>> print(results.summary())
    """

    def __init__(self, learners, method='nnloglik', cv=5, inner_cv=5,
                 random_state=None, n_jobs=1, n_jobs_learners=1, verbose=False, **kwargs):
        self.learners = learners
        self.method = method
        self.cv = cv
        self.inner_cv = inner_cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_jobs_learners = n_jobs_learners
        self.verbose = verbose
        self.kwargs = kwargs
        self.results_ = None

    def fit(self, X, y, sample_weight=None, groups=None):
        """
        Fit CV Super Learner and return self.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        sample_weight : array-like, shape (n_samples,), optional
            Sample weights
        groups : array-like, shape (n_samples,), optional
            Group labels (not currently used)

        Returns
        -------
        self : CVSuperLearner
            Fitted estimator with results_ attribute
        """
        # Create a SuperLearner instance for evaluation
        from .super_learner import SuperLearner
        sl = SuperLearner(learners=self.learners, method=self.method, cv=self.inner_cv,
                          random_state=self.random_state,
                          verbose=self.verbose, **self.kwargs)

        # Handle nested parallelism - warn and adjust if both are set
        effective_n_jobs_learners = self.n_jobs_learners
        if self.n_jobs > 1 and self.n_jobs_learners > 1:
            import warnings
            warnings.warn(
                f"Both n_jobs={self.n_jobs} and n_jobs_learners={self.n_jobs_learners} are > 1. "
                "Using n_jobs for fold parallelization and forcing n_jobs_learners=1 within "
                "each fold to avoid nested parallelism overhead. For best performance, use "
                "either n_jobs OR n_jobs_learners, not both.",
                UserWarning
            )
            effective_n_jobs_learners = 1

        # Run CV evaluation using existing function
        self.results_ = evaluate_super_learner_cv(
            X=X,
            y=y,
            base_learners=self.learners,
            super_learner=sl,
            outer_folds=self.cv,
            random_state=self.random_state,
            sample_weight=sample_weight,
            n_jobs=self.n_jobs,
            n_jobs_learners=effective_n_jobs_learners,
            return_predictions=True,
            return_object=True
        )

        return self

    def get_results(self):
        """
        Return SuperLearnerCVResults object.

        Returns
        -------
        results : SuperLearnerCVResults
            Results object containing metrics and predictions
        """
        if self.results_ is None:
            raise ValueError('CVSuperLearner not fitted yet. Call fit() first.')
        return self.results_

    def predict(self, X):
        """
        Predict class labels (not implemented - use get_results() instead).

        Note: CVSuperLearner is for evaluation, not prediction.
        Use SuperLearner for prediction on new data.
        """
        raise NotImplementedError(
            "CVSuperLearner is for cross-validation evaluation only. "
            "Use SuperLearner for prediction on new data."
        )

    def predict_proba(self, X):
        """
        Predict class probabilities (not implemented - use get_results() instead).

        Note: CVSuperLearner is for evaluation, not prediction.
        Use SuperLearner for prediction on new data.
        """
        raise NotImplementedError(
            "CVSuperLearner is for cross-validation evaluation only. "
            "Use SuperLearner for prediction on new data."
        )
