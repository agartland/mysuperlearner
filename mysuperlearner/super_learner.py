"""
Extended SuperLearner class with R SuperLearner-like functionality.
Includes custom meta-learners, robust error handling, and CV evaluation.
"""

import numpy as np
import pandas as pd
import warnings
from typing import List, Dict, Any, Optional, Tuple, Union
from copy import deepcopy
from sklearn.base import clone
from sklearn.utils.validation import check_X_y, check_array
from scipy.optimize import nnls
from scipy.special import expit

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

from .meta_learners import NNLogLikEstimator, AUCEstimator, MeanEstimator
from .error_handling import ErrorTracker, safe_fit, safe_predict, ErrorType

class SuperLearner(BaseEstimator, ClassifierMixin):
    """
    Super Learner ensemble method for binary classification.

    Replicates R SuperLearner functionality with sklearn-compatible API.

    Features:
    - Custom binary classification meta-learners (NNLogLik, AUC, NNLS)
    - Robust error handling with detailed tracking
    - Cross-validation for meta-learner training
    - Compatible with sklearn pipelines and model selection

    Parameters
    ----------
    learners : list of (name, estimator) tuples
        Base learning algorithms to include in the ensemble.
        Each tuple should be (str, estimator) where estimator implements
        sklearn's fit/predict interface.
    method : str, default='nnloglik'
        Meta-learning strategy. Options:
        - 'nnloglik': Non-negative binomial log-likelihood (R's method.NNloglik)
        - 'nnls': Non-negative least squares
        - 'auc': AUC optimization via Nelder-Mead
        - 'logistic': Logistic regression meta-learner
    cv : int or cross-validation generator, default=5
        Cross-validation strategy for meta-learner training.
        - If int, use StratifiedKFold with this many folds
        - If cross-validation generator (e.g., GroupKFold, TimeSeriesSplit),
          use the provided splitter directly
    random_state : int, optional
        Random seed for reproducibility
    verbose : bool, default=False
        Whether to print detailed information during fitting
    track_errors : bool, default=True
        Whether to track errors and warnings for diagnostics
    trim : float, default=0.001
        Probability trimming bounds to avoid numerical issues.
        Changed from 0.025 to 0.001 to match R SuperLearner default.
    normalize_weights : bool, default=True
        Whether to normalize meta-learner weights to sum to 1
    n_jobs : int, default=1
        Number of parallel jobs (not yet fully implemented)
    min_viable_learners : int, default=1
        Minimum number of learners that must succeed in final refit.
        If fewer learners succeed, an exception is raised.

    Attributes
    ----------
    base_learners_full_ : list of (name, estimator) tuples
        Fitted base learners on full training data
    meta_learner_ : estimator or None
        Fitted meta-learner (if applicable)
    meta_weights_ : ndarray or None
        Meta-learner weights for combining base predictions
    base_learner_names_ : list of str
        Names of base learners
    Z_ : ndarray, shape (n_samples, n_learners)
        Cross-validated predictions from base learners
    cv_predictions_ : list of ndarray
        Per-learner CV predictions
    feature_names_ : list of str
        Feature names (extracted from DataFrame or generated)

    Examples
    --------
    >>> from mysuperlearner import SuperLearner
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>>
    >>> learners = [
    ...     ('RF', RandomForestClassifier(random_state=42)),
    ...     ('LR', LogisticRegression(random_state=42))
    ... ]
    >>> sl = SuperLearner(learners=learners, method='nnloglik', cv=5)
    >>> sl.fit(X_train, y_train)
    >>> predictions = sl.predict_proba(X_test)
    """

    def __init__(self, learners=None, method='nnloglik', cv=5, random_state=None,
                 verbose=False, track_errors=True, trim=0.001,
                 normalize_weights=True, n_jobs=1, min_viable_learners=1):
        self.learners = learners
        self.method = method
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose
        self.track_errors = track_errors
        self.trim = trim
        self.normalize_weights = normalize_weights
        self.n_jobs = n_jobs
        self.min_viable_learners = min_viable_learners

        # Initialize error tracking
        if self.track_errors:
            self.error_tracker = ErrorTracker(verbose=verbose)
        else:
            self.error_tracker = None

    # ... (rest of the ExtendedSuperLearner class)

    def _get_proba(self, model, X):
        """Return probability for class 1 from model, with fallbacks."""
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)
            return probs[:, 1]
        if hasattr(model, 'decision_function'):
            return expit(model.decision_function(X))
        # fallback to predict (0/1)
        return np.asarray(model.predict(X), dtype=float)

    def _build_level1(self, X, y, learners: List[tuple], cv=None,
                      random_state: Optional[int] = None, sample_weight=None, groups=None):
        """Construct level-1 matrix Z (n x K) of CV predictions.

        learners: list of (name, estimator) tuples.
        cv: int or CV splitter object
        groups: array-like, optional. Group labels for GroupKFold, etc.
        Returns Z, list_of_cv_preds (list of arrays per learner), fold_indices, cv_risks, learner_timings
        """
        import time

        if cv is None:
            cv = self.cv

        X_arr, y_arr = check_X_y(X, y)

        # Handle both int and CV splitter objects
        if isinstance(cv, int):
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        else:
            # Assume it's a cross-validation splitter
            cv_splitter = cv
        n_samples = X_arr.shape[0]
        K = len(learners)
        Z = np.zeros((n_samples, K), dtype=float)
        cv_preds = [np.zeros(n_samples, dtype=float) for _ in range(K)]
        fold_indices = []
        learner_timings = []  # New: collect timing data

        for inner_fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X_arr, y_arr, groups)):
            fold_indices.append((train_idx, test_idx))
            X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
            y_tr = y_arr[train_idx]
            sw_tr = None
            if sample_weight is not None:
                sw_tr = np.asarray(sample_weight)[train_idx]

            for j, (name, estimator) in enumerate(learners):
                start_time = time.time()  # New: start timing
                try:
                    mdl = clone(estimator)
                    # fit with sample_weight when supported
                    try:
                        if sw_tr is not None:
                            mdl.fit(X_tr, y_tr, sample_weight=sw_tr)
                        else:
                            mdl.fit(X_tr, y_tr)
                    except TypeError:
                        mdl.fit(X_tr, y_tr)
                    preds = self._get_proba(mdl, X_te)
                    fit_time = time.time() - start_time  # New: capture time

                    # New: record timing
                    learner_timings.append({
                        'learner_name': name,
                        'inner_fold_idx': inner_fold_idx,
                        'fit_time': fit_time,
                        'timestamp': time.time()
                    })

                    # New: display per-learner update if verbose
                    if self.verbose:
                        print(f"  [{name}] completed inner fold {inner_fold_idx + 1} in {fit_time:.2f}s")

                except Exception as e:
                    fit_time = time.time() - start_time  # New: capture time even on error
                    # New: record timing with error flag
                    learner_timings.append({
                        'learner_name': name,
                        'inner_fold_idx': inner_fold_idx,
                        'fit_time': fit_time,
                        'timestamp': time.time(),
                        'error': str(e)
                    })
                    # register error and fill with nan
                    if self.error_tracker is not None:
                        self.error_tracker.add_error(name, ErrorType.FITTING, str(e), fold=None, phase='cv')
                    preds = np.full(len(test_idx), np.nan)
                Z[test_idx, j] = preds
                cv_preds[j][test_idx] = preds

        # Clip/trimming
        Z = np.clip(Z, self.trim, 1.0 - self.trim)

        # Compute CV risk for each learner (mean squared error on CV predictions)
        # This matches R's cvRisk calculation
        cv_risks = np.zeros(K)
        for j in range(K):
            valid_mask = ~np.isnan(Z[:, j])
            if valid_mask.sum() > 0:
                cv_risks[j] = mean_squared_error(y_arr[valid_mask], Z[valid_mask, j])
            else:
                cv_risks[j] = np.inf

        return Z, cv_preds, fold_indices, cv_risks, learner_timings

    def fit(self, X, y, sample_weight=None, store_X=False, groups=None):
        """Fit Super Learner ensemble.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Target values (binary: 0 or 1).
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights. If provided, learners that support sample_weight
            will use them during training.
        store_X : bool, default=False
            Whether to store training data for variable importance calculations.
            If True, enables variable importance methods but increases memory usage.
            Feature names are always stored (minimal overhead).
        groups : array-like of shape (n_samples,), optional
            Group labels for samples. Used by GroupKFold and similar CV splitters.
            Ignored if cv is an integer.

        Returns
        -------
        self : SuperLearner
            Fitted estimator.
        """
        if self.learners is None:
            raise ValueError(
                "No learners provided. Either pass learners to the constructor "
                "(SuperLearner(learners=[...])) or use fit_explicit(X, y, base_learners) "
                "(deprecated)."
            )
        base_learners = self.learners
        # Store feature names before validation converts to array
        if hasattr(X, 'columns'):
            # DataFrame input - extract feature names
            self.feature_names_ = list(X.columns)
            if store_X:
                import pandas as pd
                self.X_ = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_)
        else:
            # Array input - create generic feature names
            n_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])
            self.feature_names_ = [f"feature_{i}" for i in range(n_features)]
            if store_X:
                import pandas as pd
                self.X_ = pd.DataFrame(X, columns=self.feature_names_)

        # Validate inputs
        X_arr, y_arr = check_X_y(X, y)
        self.y_ = y_arr  # Store for diagnostics

        # Build level-1 matrix
        Z, cv_preds, fold_indices, cv_risks, learner_timings = self._build_level1(X_arr, y_arr, base_learners,
                                                       cv=self.cv, random_state=self.random_state,
                                                       sample_weight=sample_weight, groups=groups)
        self.Z_ = Z
        self.cv_predictions_ = cv_preds
        self.fold_indices_ = fold_indices
        self.cv_risks_ = cv_risks
        self.learner_timings_ = learner_timings  # New: store timing data
        self.base_learner_names_ = [n for n, _ in base_learners]

        # Select meta learner
        method = self.method.lower()
        if method == 'nnloglik':
            meta = NNLogLikEstimator(trim=self.trim)
            meta.fit(Z, y_arr)
            # try to extract weights
            weights = getattr(meta, 'coef_', None)
        elif method == 'nnls':
            # use NNLS to solve for weights
            w, _ = nnls(Z, y_arr)
            weights = w
            meta = None
        elif method == 'logistic':
            meta = LogisticRegression(max_iter=200)
            meta.fit(Z, y_arr)
            weights = getattr(meta, 'coef_', None)
            if weights is not None:
                weights = weights.ravel()
        elif method == 'auc':
            meta = AUCEstimator()
            meta.fit(Z, y_arr)
            weights = getattr(meta, 'coef_', None)
        else:
            raise ValueError(f"Unknown meta method: {self.method}")

        # Normalize weights if requested and available
        if weights is None:
            # attempt to compute weights from logistic meta coef or fallback to mean
            if meta is not None and hasattr(meta, 'predict_proba'):
                # no direct weights, keep meta as predictor
                self.meta_learner_ = meta
                self.meta_weights_ = None
            else:
                # fallback: equal weights
                K = Z.shape[1]
                weights = np.ones(K) / float(K)
                self.meta_weights_ = weights
                self.meta_learner_ = None
        else:
            w = np.asarray(weights, dtype=float)
            if self.normalize_weights and w.sum() > 0:
                w = w / w.sum()
            self.meta_weights_ = w
            self.meta_learner_ = meta

        # Refit base learners on full data with comprehensive error handling
        self.base_learners_full_ = []
        self.failed_learners_ = set()

        for name, estimator in base_learners:
            mdl = clone(estimator)
            try:
                # Try to fit with sample_weight
                if sample_weight is not None:
                    try:
                        mdl.fit(X_arr, y_arr, sample_weight=sample_weight)
                    except TypeError:
                        # Estimator doesn't support sample_weight
                        mdl.fit(X_arr, y_arr)
                else:
                    mdl.fit(X_arr, y_arr)
                self.base_learners_full_.append((name, mdl))

            except Exception as e:
                # Track error during final refit
                if self.error_tracker is not None:
                    import traceback
                    self.error_tracker.add_error(
                        learner_name=name,
                        error_type=ErrorType.FITTING,
                        message=f"Final refit failed: {str(e)}",
                        fold=None,
                        phase='final_refit',
                        severity='error',
                        traceback=traceback.format_exc() if self.verbose else None
                    )
                self.failed_learners_.add(name)

                # Add dummy learner that returns neutral predictions
                from sklearn.base import ClassifierMixin
                class _DummyFailedLearner(BaseEstimator, ClassifierMixin):
                    def __init__(self, name):
                        self.name = name
                    def fit(self, X, y, sample_weight=None):
                        return self
                    def predict_proba(self, X):
                        n = X.shape[0] if hasattr(X, 'shape') else len(X)
                        return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])
                    def predict(self, X):
                        n = X.shape[0] if hasattr(X, 'shape') else len(X)
                        return np.zeros(n, dtype=int)

                dummy = _DummyFailedLearner(name=name)
                self.base_learners_full_.append((name, dummy))

        # Check if we have enough working learners
        working_learners = len(self.base_learners_full_) - len(self.failed_learners_)
        min_viable = getattr(self, 'min_viable_learners', 1)

        if working_learners < min_viable:
            raise RuntimeError(
                f"Only {working_learners}/{len(base_learners)} learners succeeded in final refit. "
                f"Minimum required: {min_viable}. "
                f"Failed learners: {self.failed_learners_}"
            )
        elif len(self.failed_learners_) > 0:
            warnings.warn(
                f"Warning: {len(self.failed_learners_)} learner(s) failed in final refit: "
                f"{self.failed_learners_}. Ensemble will use {working_learners} learners.",
                category=UserWarning
            )

        return self

    def fit_explicit(self, X, y, base_learners, sample_weight=None, store_X=False):
        """
        Deprecated: Use fit() instead.

        This method is provided for backward compatibility and will be removed in v0.3.0.
        Use SuperLearner(learners=base_learners).fit(X, y) instead.
        """
        warnings.warn(
            "fit_explicit() is deprecated and will be removed in v0.3.0. "
            "Use SuperLearner(learners=base_learners).fit(X, y) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Temporarily override self.learners
        self.learners = base_learners
        return self.fit(X, y, sample_weight=sample_weight, store_X=store_X)

    def predict_proba(self, X):
        """Predict probabilities for X using fitted base learners and meta weights/learner."""
        X_arr = check_array(X)
        if not hasattr(self, 'base_learners_full_'):
            raise RuntimeError('Model not fitted. Call fit() first.')

        # build matrix of base predictions on X
        K = len(self.base_learners_full_)
        Z_new = np.zeros((X_arr.shape[0], K), dtype=float)
        prediction_failures = []

        for j, (name, mdl) in enumerate(self.base_learners_full_):
            try:
                Z_new[:, j] = self._get_proba(mdl, X_arr)
            except Exception as e:
                # Track prediction error
                if self.error_tracker is not None:
                    self.error_tracker.add_error(
                        learner_name=name,
                        error_type=ErrorType.PREDICTION,
                        message=f"Prediction failed: {str(e)}",
                        fold=None,
                        phase='prediction',
                        severity='warning'
                    )
                prediction_failures.append(name)
                # Use neutral probability (0.5) instead of 0.0 for failed predictions
                Z_new[:, j] = 0.5

        # Warn about prediction failures if verbose
        if len(prediction_failures) > 0 and self.verbose:
            warnings.warn(
                f"Prediction failed for {len(prediction_failures)} learner(s): "
                f"{prediction_failures}. Using neutral probability (0.5).",
                category=UserWarning
            )

        Z_new = np.clip(Z_new, self.trim, 1.0 - self.trim)

        # If meta_learner_ provides predict_proba, use it
        if hasattr(self, 'meta_learner_') and self.meta_learner_ is not None and hasattr(self.meta_learner_, 'predict_proba'):
            return self.meta_learner_.predict_proba(Z_new)

        # else combine via meta_weights_
        if getattr(self, 'meta_weights_', None) is None:
            # equal weights fallback
            w = np.ones(Z_new.shape[1]) / Z_new.shape[1]
        else:
            w = self.meta_weights_

        p = Z_new.dot(w)
        p = np.clip(p, 0.0, 1.0)
        # return shape (n,2)
        proba = np.vstack([1 - p, p]).T
        return proba

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        return (p >= 0.5).astype(int)

    def get_diagnostics(self):
        """
        Get diagnostic information about the fitted SuperLearner.

        Returns
        -------
        diagnostics : dict
            Dictionary containing:
            - 'method': Meta-learning method used
            - 'n_folds': Number of CV folds
            - 'base_learner_names': Names of base learners
            - 'meta_weights': Array of meta-learner weights (if available)
            - 'cv_scores': Dict of per-learner CV AUC scores (if available)
            - 'errors': Error tracker records (if tracking enabled)
            - 'cv_predictions_shape': Shape of cross-validated predictions matrix

        Examples
        --------
        >>> sl = SuperLearner(learners=learners, method='nnloglik', cv=5)
        >>> sl.fit(X_train, y_train)
        >>> diagnostics = sl.get_diagnostics()
        >>> print(f"Meta weights: {diagnostics['meta_weights']}")
        """
        diagnostics = {
            'method': self.method,
            'n_folds': self.cv,
            'base_learner_names': getattr(self, 'base_learner_names_', []),
            'meta_weights': getattr(self, 'meta_weights_', None),
        }

        # Add CV predictions shape if available
        if hasattr(self, 'Z_'):
            diagnostics['cv_predictions_shape'] = self.Z_.shape

        # Add CV scores if available
        if hasattr(self, 'cv_predictions_') and hasattr(self, 'Z_'):
            cv_scores = {}
            # Check if we have stored y_ (would need to be added in fit_explicit)
            if hasattr(self, 'y_'):
                for i, name in enumerate(self.base_learner_names_):
                    try:
                        cv_scores[name] = roc_auc_score(self.y_, self.Z_[:, i])
                    except Exception:
                        cv_scores[name] = np.nan
                diagnostics['cv_scores'] = cv_scores

        # Add error information
        if self.error_tracker is not None:
            diagnostics['errors'] = self.error_tracker.error_records
            diagnostics['n_errors'] = len(self.error_tracker.error_records)
        else:
            diagnostics['errors'] = None
            diagnostics['n_errors'] = 0

        # Add meta-learner info
        if hasattr(self, 'meta_learner_') and self.meta_learner_ is not None:
            diagnostics['meta_learner_type'] = type(self.meta_learner_).__name__
        else:
            diagnostics['meta_learner_type'] = 'WeightedCombination'

        return diagnostics
