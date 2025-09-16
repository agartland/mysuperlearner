"""
Extended SuperLearner class with R SuperLearner-like functionality.
Includes custom meta-learners, robust error handling, and CV evaluation.
"""

import numpy as np
import pandas as pd
import warnings
from typing import List, Dict, Any, Optional, Tuple, Union
from copy import deepcopy

# Removed mlens dependency entirely: we provide explicit CV and evaluation
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

from .custom_meta_learners import NNLogLikEstimator, AUCEstimator, MeanEstimator
from .error_handling import ErrorTracker, safe_fit, safe_predict, ErrorType


# Note: this top-level module previously relied on mlens.SuperLearner and
# mlens.Evaluator. The package now implements explicit CV builders and an
# external CV evaluator. Keep a thin compatibility shim that proxies to the
# internal package implementation to avoid external dependency on mlens.

from mysuperlearner import ExtendedSuperLearner as PackageExtendedSuperLearner


class ExtendedSuperLearner(PackageExtendedSuperLearner):
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
        
    # Initialize the underlying package implementation
    super().__init__(method=method, folds=folds, random_state=random_state,
             verbose=verbose, track_errors=track_errors, **kwargs)
        
        # Track if we've added the meta learner
        self._meta_added = False
        self._base_learners_added = False
        
        # Store information about learners and performance
        self.learner_names_ = []
        self.cv_results_ = None
        self.individual_predictions_ = None
        
    def add_learners(self, learners: List[Tuple[str, BaseEstimator]], **kwargs):
        """Add base learners to the ensemble"""
        self.learner_names_ = [name for name, _ in learners]
        
        # For binary classification, we typically want probabilities
        kwargs.setdefault('proba', True)
        
        # Add learners to the underlying SuperLearner
        super().add(learners, **kwargs)
        self._base_learners_added = True
        
        return self
    
    def add_simple_mean(self):
        """Add SL.mean equivalent - simple mean predictor"""
        mean_learner = [('SL.mean', MeanEstimator())]
        if not hasattr(self, 'learner_names_'):
            self.learner_names_ = []
        
        self.learner_names_.extend(['SL.mean'])
        super().add(mean_learner, proba=True)
        return self
    
    def fit(self, X, y):
        """Fit the SuperLearner with the specified meta learning method"""
        # Add meta learner if not already added
        if not self._meta_added:
            self._add_meta_learner()
        
        if self.track_errors:
            # Reset error tracker for this fit
            self.error_tracker = ErrorTracker(verbose=self.verbose)
        
        try:
            # Fit the underlying SuperLearner
            result = super().fit(X, y)
            
            # Extract CV results if available
            self._extract_cv_results()
            
            return result
            
        except Exception as e:
            if self.error_tracker:
                self.error_tracker.add_error('SuperLearner', ErrorType.FITTING,
                                          f"SuperLearner fit failed: {str(e)}", 
                                          phase='final')
            raise
    
    def predict(self, X):
        """Predict class labels"""
        try:
            return super().predict(X)
        except Exception as e:
            if self.error_tracker:
                self.error_tracker.add_error('SuperLearner', ErrorType.PREDICTION,
                                          f"SuperLearner predict failed: {str(e)}", 
                                          phase='final')
            raise
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        try:
            return super().predict_proba(X)
        except Exception as e:
            if self.error_tracker:
                self.error_tracker.add_error('SuperLearner', ErrorType.PREDICTION,
                                          f"SuperLearner predict_proba failed: {str(e)}", 
                                          phase='final')
            raise
    
    def _add_meta_learner(self):
        """Add the appropriate meta learner based on the method"""
        if self.method == 'nnloglik':
            meta_learner = NNLogLikEstimator(verbose=self.verbose)
        elif self.method == 'auc':
            meta_learner = AUCEstimator(verbose=self.verbose)
        elif self.method == 'nnls':
            meta_learner = LinearRegression(positive=True, fit_intercept=False)
        elif self.method == 'logistic':
            meta_learner = LogisticRegression(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        super().add_meta(meta_learner)
        self._meta_added = True
    
    def _extract_cv_results(self):
        """Extract cross-validation results from the fitted SuperLearner"""
        try:
            # Access the data attribute if available
            if hasattr(self, 'data') and self.data is not None:
                self.cv_results_ = self.data.copy()
            elif hasattr(self, 'scores_') and self.scores_ is not None:
                self.cv_results_ = pd.DataFrame(self.scores_)
        except Exception as e:
            if self.error_tracker:
                self.error_tracker.add_warning('SuperLearner', ErrorType.OTHER,
                                             f"Could not extract CV results: {str(e)}")
    
    def get_meta_learner_info(self) -> Dict[str, Any]:
        """Get information about the meta learner"""
        if not self._meta_added:
            return {}
        
        try:
            # Access the meta learner from the fitted SuperLearner
            meta_layer = self.layers_[-1]  # Last layer should be meta
            meta_learner = meta_layer.learners[0].estimator
            
            info = {
                'method': self.method,
                'meta_learner_type': type(meta_learner).__name__
            }
            
            # Add specific information based on meta learner type
            if hasattr(meta_learner, 'coef_'):
                info['coefficients'] = meta_learner.coef_
                info['n_features'] = len(meta_learner.coef_)
            
            if hasattr(meta_learner, 'convergence_info_'):
                info['convergence_info'] = meta_learner.convergence_info_
            
            if hasattr(meta_learner, 'errors_'):
                info['meta_errors'] = len(meta_learner.errors_)
                info['meta_warnings'] = len(meta_learner.warnings_)
            
            return info
            
        except Exception as e:
            if self.error_tracker:
                self.error_tracker.add_warning('SuperLearner', ErrorType.OTHER,
                                             f"Could not get meta learner info: {str(e)}")
            return {}
    
    def print_summary(self):
        """Print a summary of the SuperLearner results"""
        print("\n" + "="*70)
        print("EXTENDED SUPERLEARNER SUMMARY")
        print("="*70)
        
        print(f"Method: {self.method}")
        print(f"Number of base learners: {len(self.learner_names_)}")
        print(f"CV folds: {self.folds}")
        
        # Meta learner information
        meta_info = self.get_meta_learner_info()
        if meta_info:
            print(f"\nMeta Learner: {meta_info.get('meta_learner_type', 'Unknown')}")
            if 'coefficients' in meta_info:
                print("Base learner weights:")
                for name, coef in zip(self.learner_names_, meta_info['coefficients']):
                    print(f"  {name:<20}: {coef:.6f}")
        
        # CV results
        if self.cv_results_ is not None:
            print(f"\nCross-Validation Results:")
            print(self.cv_results_)
        
        # Error summary
        if self.error_tracker and self.error_tracker.error_records:
            print(f"\nError Summary:")
            self.error_tracker.print_summary()
        else:
            print(f"\nNo errors or warnings recorded.")
        
        print("="*70)
    
    def get_error_summary(self) -> Optional[pd.DataFrame]:
        """Get error summary DataFrame"""
        if self.error_tracker:
            return self.error_tracker.get_error_summary()
        return None
    
    def get_detailed_errors(self) -> Optional[pd.DataFrame]:
        """Get detailed error DataFrame"""
        if self.error_tracker:
            return self.error_tracker.get_detailed_errors()
        return None


class SuperLearnerCV:
    """
    Cross-validation wrapper for ExtendedSuperLearner.
    Equivalent to CV.SuperLearner in R - provides external CV evaluation.
    """
    
    def __init__(self, method='nnloglik', inner_cv=5, outer_cv=10, 
                 random_state=None, verbose=False, n_jobs=-1, **kwargs):
        """
        Parameters:
        -----------
        method : str, default='nnloglik'
            Meta learning method for SuperLearner
        inner_cv : int, default=5
            Number of CV folds for internal SuperLearner
        outer_cv : int, default=10
            Number of CV folds for external evaluation
        random_state : int, optional
            Random state for reproducibility
        verbose : bool, default=False
            Whether to print detailed information
        n_jobs : int, default=-1
            Number of parallel jobs for evaluation
        **kwargs : additional arguments passed to SuperLearner
        """
        self.method = method
        self.inner_cv = inner_cv
        self.outer_cv = outer_cv
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        
        # Results storage
        self.cv_results_ = None
        self.individual_results_ = None
        self.learner_names_ = []
        self.superlearner_template = None
        self.error_tracker = ErrorTracker(verbose=verbose)
    
    def fit(self, X, y, learners: List[Tuple[str, BaseEstimator]], 
            scoring='accuracy', include_individual=True):
        """
        Fit SuperLearner with external cross-validation.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        learners : List[Tuple[str, BaseEstimator]]
            List of (name, estimator) tuples
        scoring : str or callable, default='accuracy'
            Scoring function for evaluation
        include_individual : bool, default=True
            Whether to evaluate individual learners
        """
        self.learner_names_ = [name for name, _ in learners]
        
        # Create SuperLearner template
        self.superlearner_template = ExtendedSuperLearner(
            method=self.method,
            folds=self.inner_cv,
            random_state=self.random_state,
            verbose=False,  # Reduce verbosity for CV
            track_errors=True,
            **self.kwargs
        )
        
        # Add learners to template
        self.superlearner_template.add_learners(learners)
        
        # Set up scoring
        if scoring == 'accuracy':
            scorer_func = accuracy_score
        elif scoring == 'auc':
            scorer_func = roc_auc_score
        elif scoring == 'mse':
            scorer_func = lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
        elif callable(scoring):
            scorer_func = scoring
        else:
            raise ValueError(f"Unknown scoring: {scoring}")
        
        # Prepare estimators for Evaluator
        estimators_for_eval = []
        
        # Add SuperLearner
        sl_copy = deepcopy(self.superlearner_template)
        estimators_for_eval.append(('SuperLearner', sl_copy))
        
        # Add individual learners if requested
        if include_individual:
            for name, learner in learners:
                estimators_for_eval.append((name, deepcopy(learner)))
        
        # Set up cross-validation
        if hasattr(y, 'dtype') and y.dtype == 'object' or len(np.unique(y)) < 20:
            # Likely classification
            cv_splitter = StratifiedKFold(n_splits=self.outer_cv, 
                                        shuffle=True, 
                                        random_state=self.random_state)
        else:
            # Likely regression
            cv_splitter = KFold(n_splits=self.outer_cv, 
                              shuffle=True, 
                              random_state=self.random_state)
        
        # Use mlens Evaluator for external CV
        evaluator = Evaluator(
            scorer=scorer_func,
            cv=cv_splitter,
            random_state=self.random_state,
            verbose=self.verbose,
            n_jobs=self.n_jobs
        )
        
        try:
            # Run evaluation (no parameter search, just CV scoring)
            evaluator.fit(X, y, estimators_for_eval, param_dicts={})
            
            # Extract results
            self.cv_results_ = pd.DataFrame(evaluator.results)
            
            # Separate SuperLearner and individual results
            sl_results = self.cv_results_[self.cv_results_['estimator'] == 'SuperLearner']
            if include_individual:
                individual_results = self.cv_results_[self.cv_results_['estimator'] != 'SuperLearner']
                self.individual_results_ = individual_results
            
            if self.verbose:
                print(f"External CV completed successfully")
                self._print_cv_summary()
                
        except Exception as e:
            self.error_tracker.add_error('SuperLearnerCV', ErrorType.OTHER,
                                       f"External CV failed: {str(e)}", phase='external_cv')
            raise
    
    def _print_cv_summary(self):
        """Print summary of CV results"""
        if self.cv_results_ is None:
            print("No CV results available")
            return
        
        print("\n" + "="*70)
        print("EXTERNAL CROSS-VALIDATION RESULTS")
        print("="*70)
        
        # Summary statistics
        summary_stats = self.cv_results_.groupby('estimator').agg({
            'test_score': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        summary_stats.columns = ['Mean', 'Std', 'Min', 'Max']
        
        print("Performance Summary:")
        print(summary_stats)
        
        # Highlight best performer
        best_estimator = self.cv_results_.groupby('estimator')['test_score'].mean().idxmax()
        best_score = self.cv_results_.groupby('estimator')['test_score'].mean().max()
        
        print(f"\nBest Performer: {best_estimator} (Score: {best_score:.4f})")
        
        # SuperLearner specific info
        sl_scores = self.cv_results_[self.cv_results_['estimator'] == 'SuperLearner']['test_score']
        if len(sl_scores) > 0:
            print(f"SuperLearner: {sl_scores.mean():.4f} ± {sl_scores.std():.4f}")
        
        print("="*70)
    
    def get_cv_summary(self) -> Optional[pd.DataFrame]:
        """Get summary of CV results as DataFrame"""
        if self.cv_results_ is None:
            return None
        
        summary = self.cv_results_.groupby('estimator').agg({
            'test_score': ['mean', 'std', 'min', 'max', 'count']
        }).round(4)
        
        summary.columns = ['Mean', 'Std', 'Min', 'Max', 'N_Folds']
        summary = summary.reset_index()
        
        # Add rank
        summary['Rank'] = summary['Mean'].rank(ascending=False).astype(int)
        summary = summary.sort_values('Rank')
        
        return summary
    
    def plot_cv_results(self, figsize=(10, 6)):
        """Plot CV results"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if self.cv_results_ is None:
                print("No CV results to plot")
                return
            
            plt.figure(figsize=figsize)
            
            # Box plot of CV scores
            sns.boxplot(data=self.cv_results_, x='estimator', y='test_score')
            plt.xticks(rotation=45, ha='right')
            plt.title('Cross-Validation Performance Comparison')
            plt.ylabel('CV Score')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib/Seaborn not available for plotting")
        except Exception as e:
            print(f"Plotting failed: {str(e)}")


# Convenience functions
def create_superlearner(learners: List[Tuple[str, BaseEstimator]], 
                       method='nnloglik', folds=5, random_state=None, 
                       add_mean=True, **kwargs) -> ExtendedSuperLearner:
    """
    Convenience function to create a SuperLearner.
    
    Parameters:
    -----------
    learners : List[Tuple[str, BaseEstimator]]
        List of (name, estimator) tuples
    method : str, default='nnloglik'
        Meta learning method
    folds : int, default=5
        Number of CV folds
    random_state : int, optional
        Random state for reproducibility
    add_mean : bool, default=True
        Whether to add SL.mean equivalent
    **kwargs : additional arguments passed to ExtendedSuperLearner
    
    Returns:
    --------
    ExtendedSuperLearner instance
    """
    sl = ExtendedSuperLearner(
        method=method,
        folds=folds,
        random_state=random_state,
        **kwargs
    )
    
    sl.add_learners(learners)
    
    if add_mean:
        sl.add_simple_mean()
    
    return sl


def evaluate_superlearner(X, y, learners: List[Tuple[str, BaseEstimator]], 
                         method='nnloglik', inner_cv=5, outer_cv=10,
                         scoring='accuracy', random_state=None, 
                         include_individual=True, **kwargs) -> SuperLearnerCV:
    """
    Convenience function for external CV evaluation of SuperLearner.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    learners : List[Tuple[str, BaseEstimator]]
        List of (name, estimator) tuples
    method : str, default='nnloglik'
        Meta learning method
    inner_cv : int, default=5
        Number of CV folds for internal SuperLearner
    outer_cv : int, default=10
        Number of CV folds for external evaluation
    scoring : str or callable, default='accuracy'
        Scoring function
    random_state : int, optional
        Random state for reproducibility
    include_individual : bool, default=True
        Whether to evaluate individual learners
    **kwargs : additional arguments
    
    Returns:
    --------
    SuperLearnerCV instance with fitted results
    """
    sl_cv = SuperLearnerCV(
        method=method,
        inner_cv=inner_cv,
        outer_cv=outer_cv,
        random_state=random_state,
        **kwargs
    )
    
    sl_cv.fit(X, y, learners, scoring=scoring, include_individual=include_individual)
    
    return sl_cv