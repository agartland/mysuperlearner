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

class ExtendedSuperLearner(BaseEstimator, ClassifierMixin):
    """
    Extended SuperLearner that replicates R SuperLearner functionality.
    
    Features:
    - Custom binary classification meta-learners (NNLogLik, AUC)
    - Robust error handling with detailed tracking
    - Easy external CV evaluation
    - R-like simple learners (SL.mean equivalent)
    """
    
    def __init__(self, method='nnloglik', folds=5, random_state=None, 
                 verbose=False, track_errors=True, **kwargs):
        """
        Parameters:
        -----------
        method : str, default='nnloglik'
            Meta learning method. Options: 'nnloglik', 'auc', 'nnls', 'logistic'
        folds : int, default=5
            Number of cross-validation folds for internal CV
        random_state : int, optional
            Random state for reproducibility
        verbose : bool, default=False
            Whether to print detailed information
        track_errors : bool, default=True
            Whether to track errors and warnings
        **kwargs : additional arguments passed to SuperLearner
        """
        self.method = method
        self.folds = folds
        self.random_state = random_state
        self.verbose = verbose
        self.track_errors = track_errors
        self.kwargs = kwargs
        
        # Initialize error tracking
        if self.track_errors:
            self.error_tracker = ErrorTracker(verbose=verbose)
        else:
            self.error_tracker = None
        
    # Note: we implement explicit level-1 builder and do not rely on mlens
        
        # Track if we've added the meta learner
        self._meta_added = False
        self._base_learners_added = False
        
        # Store information about learners and performance
        self.learner_names_ = []
        self.cv_results_ = None
        self.individual_predictions_ = None
        # explicit builder settings
        self.trim = self.kwargs.pop('trim', 0.025)
        self.normalize_weights = self.kwargs.pop('normalize_weights', True)
        self.n_jobs = self.kwargs.pop('n_jobs', 1)

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

    def _build_level1(self, X, y, learners: List[tuple], folds: int = None,
                      random_state: Optional[int] = None, sample_weight=None):
        """Construct level-1 matrix Z (n x K) of CV predictions.

        learners: list of (name, estimator) tuples.
        Returns Z, list_of_cv_preds (list of arrays per learner), fold_indices
        """
        if folds is None:
            folds = self.folds
        X_arr, y_arr = check_X_y(X, y)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        n_samples = X_arr.shape[0]
        K = len(learners)
        Z = np.zeros((n_samples, K), dtype=float)
        cv_preds = [np.zeros(n_samples, dtype=float) for _ in range(K)]
        fold_indices = []

        for train_idx, test_idx in skf.split(X_arr, y_arr):
            fold_indices.append((train_idx, test_idx))
            X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
            y_tr = y_arr[train_idx]
            sw_tr = None
            if sample_weight is not None:
                sw_tr = np.asarray(sample_weight)[train_idx]

            for j, (name, estimator) in enumerate(learners):
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
                except Exception as e:
                    # register error and fill with nan
                    if self.error_tracker is not None:
                        self.error_tracker.add_error(name, ErrorType.FITTING, str(e), fold=None, phase='cv')
                    preds = np.full(len(test_idx), np.nan)
                Z[test_idx, j] = preds
                cv_preds[j][test_idx] = preds

        # Clip/trimming
        Z = np.clip(Z, self.trim, 1.0 - self.trim)
        return Z, cv_preds, fold_indices

    def fit_explicit(self, X, y, base_learners: List[tuple], sample_weight=None):
        """Fit using explicit level-1 builder and refit base learners on full data.

        base_learners: list of (name, estimator) tuples
        """
        # Validate inputs
        X_arr, y_arr = check_X_y(X, y)

        # Build level-1 matrix
        Z, cv_preds, fold_indices = self._build_level1(X_arr, y_arr, base_learners,
                                                       folds=self.folds, random_state=self.random_state,
                                                       sample_weight=sample_weight)
        self.Z_ = Z
        self.cv_predictions_ = cv_preds
        self.fold_indices_ = fold_indices
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

        # Refit base learners on full data
        self.base_learners_full_ = []
        for name, estimator in base_learners:
            mdl = clone(estimator)
            try:
                if sample_weight is not None:
                    mdl.fit(X_arr, y_arr, sample_weight=sample_weight)
                else:
                    mdl.fit(X_arr, y_arr)
            except TypeError:
                mdl.fit(X_arr, y_arr)
            self.base_learners_full_.append((name, mdl))

        return self

    def predict_proba(self, X):
        """Predict probabilities for X using fitted base learners and meta weights/learner."""
        X_arr = check_array(X)
        if not hasattr(self, 'base_learners_full_'):
            raise RuntimeError('Model not fitted. Call fit_explicit first with base_learners.')

        # build matrix of base predictions on X
        K = len(self.base_learners_full_)
        Z_new = np.zeros((X_arr.shape[0], K), dtype=float)
        for j, (name, mdl) in enumerate(self.base_learners_full_):
            try:
                Z_new[:, j] = self._get_proba(mdl, X_arr)
            except Exception:
                Z_new[:, j] = 0.0

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
