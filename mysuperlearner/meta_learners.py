"""
Custom meta-learners for binary classification that replicate R SuperLearner methods.
Implements method.NNloglik and method.AUC from R SuperLearner package.
"""

import numpy as np
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from scipy.optimize import minimize
from scipy.optimize import nnls
from sklearn.metrics import roc_auc_score

class NNLogLikEstimator(BaseEstimator, ClassifierMixin):
    """
    Non-negative binomial log-likelihood meta learner.
    Implements method.NNloglik from R SuperLearner.
    
    Uses L-BFGS-B optimization with non-negative constraints to find optimal
    weights that minimize negative log-likelihood on logit scale.
    """
    
    def __init__(self, trim=0.025, maxiter=1000, verbose=False):
        self.trim = trim
        self.maxiter = maxiter
        self.verbose = verbose
        self.coef_ = None
        self.intercept_ = 0.0
        self.convergence_info_ = {}
        self.errors_ = []
        self.warnings_ = []
        
    def _neg_loglik(self, w, X, y, trim):
        # w are non-negative weights
        w = np.asarray(w)
        w = np.clip(w, 0, None)
        if w.sum() == 0:
            # uniform fallback
            w = np.ones_like(w) / float(len(w))
        else:
            w = w / w.sum()
        p = X.dot(w)
        p = np.clip(p, trim, 1.0 - trim)
        # negative log-likelihood
        ll = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        return ll

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n, K = X.shape

        # initial guess: NNLS solution
        try:
            init_w, _ = nnls(X, y)
            if init_w.sum() == 0:
                x0 = np.ones(K) / float(K)
            else:
                x0 = init_w / init_w.sum()
        except Exception:
            x0 = np.ones(K) / float(K)

        bounds = [(0, None)] * K
        res = minimize(self._neg_loglik, x0, args=(X, y, self.trim), method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': self.maxiter, 'disp': self.verbose})
        w = res.x
        w = np.clip(w, 0, None)
        if w.sum() > 0:
            w = w / w.sum()
        else:
            w = np.ones_like(w) / float(len(w))
        self.coef_ = w
        self.convergence_info_ = {'success': res.success, 'message': res.message}
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        if self.coef_ is None:
            raise ValueError('Estimator not fitted yet')
        w = np.asarray(self.coef_)
        p = X.dot(w)
        p = np.clip(p, self.trim, 1.0 - self.trim)
        return np.vstack([1 - p, p]).T


class AUCEstimator(BaseEstimator, ClassifierMixin):
    """Simple AUC-optimizing meta-learner using Nelder-Mead."""
    def __init__(self, maxiter=200, verbose=False):
        self.maxiter = maxiter
        self.verbose = verbose
        self.coef_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        K = X.shape[1]

        def neg_auc(w):
            w = np.asarray(w)
            w = np.clip(w, 0, None)
            if w.sum() == 0:
                w = np.ones_like(w) / float(len(w))
            else:
                w = w / w.sum()
            s = X.dot(w)
            return -roc_auc_score(y, s)

        x0 = np.ones(K) / float(K)
        res = minimize(neg_auc, x0, method='Nelder-Mead', options={'maxiter': self.maxiter, 'disp': self.verbose})
        w = res.x
        w = np.clip(w, 0, None)
        if w.sum() > 0:
            w = w / w.sum()
        self.coef_ = w
        return self


class MeanEstimator(BaseEstimator, ClassifierMixin):
    """Simple mean predictor meta-learner (SL.mean equivalent)."""
    def fit(self, X, y):
        # no fitting required
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        p = X.mean(axis=1)
        return np.vstack([1 - p, p]).T


class InterceptOnlyEstimator(BaseEstimator, ClassifierMixin):
    """
    Intercept-only baseline predictor (equivalent to R's SL.mean base learner).

    Predicts the empirical mean of the training data for all samples.
    This serves as a simple baseline that ignores all features.
    Useful for establishing a performance floor.
    """
    def __init__(self):
        self.mean_ = None

    def fit(self, X, y, sample_weight=None):
        """Fit by computing mean of training labels."""
        y = np.asarray(y)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            self.mean_ = np.average(y, weights=sample_weight)
        else:
            self.mean_ = np.mean(y)
        return self

    def predict(self, X):
        """Predict class based on threshold at 0.5."""
        if self.mean_ is None:
            raise ValueError('Estimator not fitted yet')
        return (self.mean_ >= 0.5).astype(int) * np.ones(len(X), dtype=int)

    def predict_proba(self, X):
        """Return constant probability for all samples."""
        if self.mean_ is None:
            raise ValueError('Estimator not fitted yet')
        n = len(X)
        p = self.mean_
        return np.vstack([np.full(n, 1 - p), np.full(n, p)]).T
