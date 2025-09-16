"""
Custom meta-learners for binary classification that replicate R SuperLearner methods.
Implements method.NNloglik and method.AUC from R SuperLearner package.
"""

import numpy as np
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from scipy.optimize import minimize
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
        
    def _trim_logit(self, p, trim=None):
        """Trim probabilities to avoid log(0) issues and apply logit transform"""
        if trim is None:
            trim = self.trim
        p_trimmed = np.clip(p, trim, 1 - trim)
        return np.log(p_trimmed / (1 - p_trimmed))
    
    def _inv_logit(self, x):
        """Inverse logit (sigmoid) function with numerical stability"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def _negative_log_likelihood(self, weights, X_logit, y):
        """Negative log-likelihood loss function"""
        try:
            # Weighted predictions on logit scale
            logits = X_logit @ weights
            probs = self._inv_logit(logits)
            
            # Avoid numerical issues
            probs = np.clip(probs, 1e-15, 1 - 1e-15)
            
            # Negative log likelihood
            nll = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))
            
            if np.isnan(nll) or np.isinf(nll):
                return 1e10  # Return large value for invalid results
                
            return nll
        except Exception as e:
            self.errors_.append(f"Error in NLL computation: {str(e)}")
            return 1e10
    
    def _gradient(self, weights, X_logit, y):
        """Gradient of negative log-likelihood"""
        try:
            logits = X_logit @ weights
            probs = self._inv_logit(logits)
            residuals = probs - y
            grad = X_logit.T @ residuals / len(y)
            return grad
        except Exception as e:
            self.errors_.append(f"Error in gradient computation: {str(e)}")
            return np.zeros_like(weights)
    
    def fit(self, X, y):
        """Fit the NNLogLik meta learner"""
        X, y = check_X_y(X, y)
        self.errors_ = []
        self.warnings_ = []
        
        # Handle edge cases
        if len(np.unique(y)) < 2:
            self.warnings_.append("Only one class present in y")
            self.coef_ = np.ones(X.shape[1]) / X.shape[1]
            return self
        
        # Transform base learner predictions to logit scale if they're probabilities
        X_logit = np.zeros_like(X)
        for i in range(X.shape[1]):
            col = X[:, i]
            if np.all((col >= 0) & (col <= 1)):
                # Check for extreme values
                if np.any((col <= self.trim) | (col >= 1 - self.trim)):
                    self.warnings_.append(f"Extreme probability values in column {i}, applying trimming")
                X_logit[:, i] = self._trim_logit(col)
            else:
                X_logit[:, i] = col
        
        n_features = X_logit.shape[1]
        
        # Initialize with equal weights
        initial_weights = np.ones(n_features) / n_features
        
        # Bounds: weights must be non-negative
        bounds = [(0, None)] * n_features
        
        # Constraint: weights sum to 1
        constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Optimize using L-BFGS-B (matches R implementation)
        try:
            result = minimize(
                fun=self._negative_log_likelihood,
                x0=initial_weights,
                args=(X_logit, y),
                method='SLSQP',  # Use SLSQP for constraints
                bounds=bounds,
                constraints=constraint,
                jac=self._gradient,
                options={'maxiter': self.maxiter, 'ftol': 1e-9}
            )
            
            self.convergence_info_ = {
                'success': result.success,
                'message': result.message,
                'nit': result.nit,
                'fun': result.fun
            }
            
            if result.success:
                self.coef_ = result.x
                # Ensure weights sum to 1 (numerical stability)
                if np.sum(self.coef_) > 0:
                    self.coef_ = self.coef_ / np.sum(self.coef_)
            else:
                self.warnings_.append(f"Optimization failed: {result.message}")
                # Fallback to uniform weights
                self.coef_ = np.ones(n_features) / n_features
                
        except Exception as e:
            self.errors_.append(f"Optimization error: {str(e)}")
            # Fallback to uniform weights
            self.coef_ = np.ones(n_features) / n_features
        
        if self.verbose and (self.errors_ or self.warnings_):
            print(f"NNLogLik Meta-learner - Errors: {len(self.errors_)}, Warnings: {len(self.warnings_)}")
            
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        X = check_array(X)
        
        # Transform to logit scale like in training
        X_logit = np.zeros_like(X)
        for i in range(X.shape[1]):
            col = X[:, i]
            if np.all((col >= 0) & (col <= 1)):
                X_logit[:, i] = self._trim_logit(col)
            else:
                X_logit[:, i] = col
        
        # Weighted combination on logit scale
        logits = X_logit @ self.coef_
        probs = self._inv_logit(logits)
        
        # Return probabilities for both classes
        return np.column_stack([1 - probs, probs])
    
    def predict(self, X):
        """Predict class labels"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


class AUCEstimator(BaseEstimator, ClassifierMixin):
    """
    AUC-maximizing meta learner.
    Implements method.AUC from R SuperLearner using Nelder-Mead optimization.
    """
    
    def __init__(self, maxiter=1000, verbose=False):
        self.maxiter = maxiter
        self.verbose = verbose
        self.coef_ = None
        self.convergence_info_ = {}
        self.errors_ = []
        self.warnings_ = []
        
    def _auc_loss(self, weights, X, y):
        """Negative AUC loss (minimize negative AUC to maximize AUC)"""
        try:
            # Ensure weights are non-negative and sum to 1
            weights = np.abs(weights)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                return 1.0  # Worst possible AUC
            
            # Weighted combination
            y_pred = X @ weights
            
            # Handle edge cases
            if len(np.unique(y)) < 2:
                return 1.0
            
            if len(np.unique(y_pred)) < 2:
                return 1.0
            
            auc = roc_auc_score(y, y_pred)
            return 1 - auc  # Return negative AUC for minimization
            
        except Exception as e:
            self.errors_.append(f"Error in AUC computation: {str(e)}")
            return 1.0
    
    def fit(self, X, y):
        """Fit the AUC-maximizing meta learner"""
        X, y = check_X_y(X, y)
        self.errors_ = []
        self.warnings_ = []
        
        # Handle edge cases
        if len(np.unique(y)) < 2:
            self.warnings_.append("Only one class present in y")
            self.coef_ = np.ones(X.shape[1]) / X.shape[1]
            return self
        
        n_features = X.shape[1]
        
        # Initialize with equal weights
        initial_weights = np.ones(n_features) / n_features
        
        # Use Nelder-Mead optimization (matches R implementation)
        try:
            result = minimize(
                fun=self._auc_loss,
                x0=initial_weights,
                args=(X, y),
                method='Nelder-Mead',
                options={'maxiter': self.maxiter, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            
            self.convergence_info_ = {
                'success': result.success,
                'message': result.message,
                'nit': result.nit,
                'fun': result.fun,
                'final_auc': 1 - result.fun
            }
            
            if result.success:
                weights = np.abs(result.x)
                self.coef_ = weights / np.sum(weights) if np.sum(weights) > 0 else weights
            else:
                self.warnings_.append(f"Optimization failed: {result.message}")
                # Fallback to uniform weights
                self.coef_ = np.ones(n_features) / n_features
                
        except Exception as e:
            self.errors_.append(f"Optimization error: {str(e)}")
            # Fallback to uniform weights
            self.coef_ = np.ones(n_features) / n_features
        
        if self.verbose and (self.errors_ or self.warnings_):
            print(f"AUC Meta-learner - Errors: {len(self.errors_)}, Warnings: {len(self.warnings_)}")
            
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        X = check_array(X)
        
        # Weighted combination
        probs = X @ self.coef_
        
        # Ensure probabilities are in valid range
        probs = np.clip(probs, 0, 1)
        
        # Return probabilities for both classes
        return np.column_stack([1 - probs, probs])
    
    def predict(self, X):
        """Predict class labels"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


class MeanEstimator(BaseEstimator, ClassifierMixin):
    """
    Simple mean estimator equivalent to SL.mean in R SuperLearner.
    Always predicts the mean of the training outcomes.
    """
    
    def __init__(self):
        self.mean_ = None
        self.errors_ = []
        self.warnings_ = []
    
    def fit(self, X, y):
        """Fit the mean estimator"""
        X, y = check_X_y(X, y)
        self.errors_ = []
        self.warnings_ = []
        
        self.mean_ = np.mean(y)
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities (constant)"""
        X = check_array(X)
        n_samples = X.shape[0]
        
        prob_class_1 = np.full(n_samples, self.mean_)
        prob_class_0 = 1 - prob_class_1
        
        return np.column_stack([prob_class_0, prob_class_1])
    
    def predict(self, X):
        """Predict class labels"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)