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
    weights that minimize negative log-likelihood. Optimizes on logit scale
    for numerical stability, matching R SuperLearner implementation.

    Parameters
    ----------
    trim : float, default=0.001
        Probability bounds for trimLogit transformation. Changed from 0.025
        to 0.001 to match R SuperLearner default.
    maxiter : int, default=1000
        Maximum iterations for L-BFGS-B optimizer.
    verbose : bool, default=False
        Whether to print optimization messages.

    Notes
    -----
    This implementation matches R's method.NNloglik by:
    1. Applying trimLogit transformation to clip probabilities to [trim, 1-trim]
    2. Transforming to logit scale (unbounded) for optimization
    3. Optimizing on logit scale for numerical stability
    4. Transforming back to probability scale for predictions
    """

    def __init__(self, trim=0.001, maxiter=1000, verbose=False):
        self.trim = trim
        self.maxiter = maxiter
        self.verbose = verbose
        self.coef_ = None
        self.intercept_ = 0.0
        self.convergence_info_ = {}
        self.errors_ = []
        self.warnings_ = []

    def _logit_trim(self, Z):
        """
        Apply trimLogit transformation like R SuperLearner.

        This clips probabilities to [trim, 1-trim] then applies logit transformation.

        Parameters
        ----------
        Z : array-like, shape (n_samples, n_learners)
            Probability predictions from base learners

        Returns
        -------
        Z_logit : ndarray, shape (n_samples, n_learners)
            Logit-transformed predictions
        """
        Z_trimmed = np.clip(Z, self.trim, 1 - self.trim)
        return np.log(Z_trimmed / (1 - Z_trimmed))

    def _inv_logit(self, Z_logit):
        """
        Inverse logit (expit) transformation.

        Parameters
        ----------
        Z_logit : array-like
            Logit-scale values

        Returns
        -------
        probs : ndarray
            Probabilities in [0, 1]
        """
        from scipy.special import expit
        return expit(Z_logit)

    def _neg_loglik_logit(self, w, Z_logit, y, trim):
        """
        Negative log-likelihood on logit scale.

        Parameters
        ----------
        w : array-like
            Non-negative weights
        Z_logit : ndarray, shape (n_samples, n_learners)
            Base learner predictions on logit scale
        y : array-like
            True binary outcomes
        trim : float
            Probability bounds for safety clipping

        Returns
        -------
        neg_ll : float
            Negative log-likelihood
        """
        # Ensure non-negative, normalized weights
        w = np.asarray(w)
        w = np.clip(w, 0, None)
        if w.sum() == 0:
            w = np.ones_like(w) / float(len(w))
        else:
            w = w / w.sum()

        # Combine on logit scale, transform back to probability
        pred_logit = Z_logit.dot(w)
        p = self._inv_logit(pred_logit)

        # Safety clip (should rarely trigger after logit transform)
        p = np.clip(p, trim, 1.0 - trim)

        # Negative log-likelihood
        ll = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        return ll

    def fit(self, X, y, sample_weight=None):
        """
        Fit meta-learner on CV predictions using logit-scale optimization.

        This matches R SuperLearner's method.NNloglik implementation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_learners)
            CV predictions from base learners (probabilities)
        y : array-like, shape (n_samples,)
            True binary outcomes
        sample_weight : array-like, shape (n_samples,), optional
            Sample weights (currently not used in optimization)

        Returns
        -------
        self : NNLogLikEstimator
            Fitted estimator
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n, K = X.shape

        # Transform to logit scale
        Z_logit = self._logit_trim(X)

        # Initial guess: NNLS solution on probability scale
        try:
            Z_prob = np.clip(X, self.trim, 1 - self.trim)
            init_w, _ = nnls(Z_prob, y)
            if init_w.sum() == 0:
                x0 = np.ones(K) / float(K)
            else:
                x0 = init_w / init_w.sum()
        except Exception:
            x0 = np.ones(K) / float(K)

        # Optimize on logit scale
        bounds = [(0, None)] * K
        res = minimize(self._neg_loglik_logit, x0, args=(Z_logit, y, self.trim),
                       method='L-BFGS-B', bounds=bounds,
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
        """
        Predict class probabilities using fitted weights.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_learners)
            Predictions from base learners (probabilities)

        Returns
        -------
        probs : ndarray, shape (n_samples, 2)
            Class probabilities [P(y=0), P(y=1)]
        """
        X = np.asarray(X)
        if self.coef_ is None:
            raise ValueError('Estimator not fitted yet')
        w = np.asarray(self.coef_)

        # Combine predictions (linear combination on probability scale)
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
