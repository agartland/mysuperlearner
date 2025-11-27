"""
Enhanced error handling system for SuperLearner implementations.

This module extends the base error handling with:
- Configurable error policies
- Better error recovery strategies
- Enhanced diagnostics
- Missing data handling
- Adaptive error management
"""

import numpy as np
import pandas as pd
import warnings
import traceback
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer

from .error_handling import ErrorType, ErrorRecord, ErrorTracker


class ErrorHandlingPolicy(Enum):
    """Policy for handling errors in SuperLearner."""
    STRICT = "strict"          # Fail on any error
    PERMISSIVE = "permissive"  # Warn and continue with viable learners
    SILENT = "silent"          # Continue silently (not recommended)
    ADAPTIVE = "adaptive"      # Adapt based on error rate


class SuperLearnerWarning(UserWarning):
    """Custom warning class for SuperLearner issues."""
    pass


class SuperLearnerConvergenceWarning(SuperLearnerWarning):
    """Warning for convergence issues in optimization."""
    pass


class SuperLearnerDataWarning(SuperLearnerWarning):
    """Warning for data quality issues."""
    pass


class SuperLearnerFitWarning(SuperLearnerWarning):
    """Warning for fitting issues."""
    pass


@dataclass
class SuperLearnerConfig:
    """Configuration for error handling in SuperLearner.

    This class provides comprehensive control over how SuperLearner handles
    errors, warnings, and edge cases during fitting and prediction.

    Parameters
    ----------
    error_policy : ErrorHandlingPolicy, default=PERMISSIVE
        Overall error handling policy:
        - STRICT: Fail on any error
        - PERMISSIVE: Warn and continue with viable learners (default)
        - SILENT: Continue silently (not recommended for production)
        - ADAPTIVE: Adapt based on error rate

    min_viable_learners : int, default=1
        Minimum number of learners required for ensemble to work.
        If fewer learners succeed, an exception is raised.

    min_viable_folds : int, optional
        Minimum number of successful folds per learner (None = folds // 2).
        Learners failing more folds are excluded from ensemble.

    max_error_rate : float, default=0.5
        Maximum proportion of learners/folds that can fail (0.0 to 1.0).
        Used with ADAPTIVE policy.

    neutral_probability : float, default=0.5
        Probability to use when predictions fail. Should be in [0, 1].
        0.5 is neutral for binary classification.

    prediction_error_handling : str, default='neutral'
        How to handle prediction errors:
        - 'neutral': Use neutral_probability
        - 'skip': Exclude failed learner from ensemble
        - 'fail': Raise exception

    impute_missing : bool, default=False
        Whether to automatically impute missing values in features.
        If False and missing values present, learners may fail.

    imputation_strategy : str, default='mean'
        Strategy for imputing missing values:
        - 'mean': Mean imputation
        - 'median': Median imputation
        - 'most_frequent': Mode imputation

    raise_on_meta_convergence_failure : bool, default=False
        Whether to raise exception if meta-learner optimization doesn't converge.
        If False, issues a warning instead.

    track_convergence_info : bool, default=True
        Whether to track convergence information from meta-learners.

    verbose_errors : bool, default=False
        Whether to print detailed error information including tracebacks.

    collect_error_context : bool, default=True
        Whether to collect contextual information (sample sizes, class distribution)
        for errors. Useful for debugging but adds overhead.

    Examples
    --------
    >>> from mysuperlearner import ExtendedSuperLearner, SuperLearnerConfig
    >>> from mysuperlearner.error_handling_enhanced import ErrorHandlingPolicy
    >>>
    >>> # Strict mode - fail on any error
    >>> config = SuperLearnerConfig(error_policy=ErrorHandlingPolicy.STRICT)
    >>> sl = ExtendedSuperLearner(method='nnloglik', config=config)
    >>>
    >>> # Permissive mode with automatic missing data imputation
    >>> config = SuperLearnerConfig(
    ...     error_policy=ErrorHandlingPolicy.PERMISSIVE,
    ...     impute_missing=True,
    ...     min_viable_learners=2
    ... )
    >>> sl = ExtendedSuperLearner(method='nnloglik', config=config)
    >>>
    >>> # Adaptive mode with custom thresholds
    >>> config = SuperLearnerConfig(
    ...     error_policy=ErrorHandlingPolicy.ADAPTIVE,
    ...     max_error_rate=0.3,
    ...     neutral_probability=0.5,
    ...     verbose_errors=True
    ... )
    >>> sl = ExtendedSuperLearner(method='nnloglik', config=config)
    """
    error_policy: ErrorHandlingPolicy = ErrorHandlingPolicy.PERMISSIVE
    min_viable_learners: int = 1
    min_viable_folds: Optional[int] = None
    max_error_rate: float = 0.5
    neutral_probability: float = 0.5
    prediction_error_handling: str = 'neutral'
    impute_missing: bool = False
    imputation_strategy: str = 'mean'
    raise_on_meta_convergence_failure: bool = False
    track_convergence_info: bool = True
    verbose_errors: bool = False
    collect_error_context: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 <= self.neutral_probability <= 1:
            raise ValueError("neutral_probability must be in [0, 1]")

        if not 0 <= self.max_error_rate <= 1:
            raise ValueError("max_error_rate must be in [0, 1]")

        if self.min_viable_learners < 1:
            raise ValueError("min_viable_learners must be at least 1")

        if self.prediction_error_handling not in ['neutral', 'skip', 'fail']:
            raise ValueError(
                "prediction_error_handling must be 'neutral', 'skip', or 'fail'"
            )

        if self.imputation_strategy not in ['mean', 'median', 'most_frequent']:
            raise ValueError(
                "imputation_strategy must be 'mean', 'median', or 'most_frequent'"
            )


@dataclass
class EnhancedErrorRecord(ErrorRecord):
    """Enhanced error record with additional context.

    Extends ErrorRecord with contextual information useful for debugging.
    """
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    class_distribution: Optional[Dict[int, int]] = None
    fold_number: Optional[int] = None
    total_folds: Optional[int] = None
    convergence_info: Optional[Dict] = None
    timestamp: Optional[str] = None


class EnhancedErrorTracker(ErrorTracker):
    """Enhanced error tracking with better reporting and policy enforcement.

    Extends ErrorTracker with:
    - Policy-based error handling
    - Enhanced diagnostics
    - Error rate monitoring
    - Learner viability assessment
    """

    def __init__(self, config: Optional[SuperLearnerConfig] = None,
                 verbose: bool = False):
        """
        Parameters
        ----------
        config : SuperLearnerConfig, optional
            Configuration for error handling
        verbose : bool, default=False
            Whether to print errors as they occur
        """
        super().__init__(verbose=verbose)
        self.config = config or SuperLearnerConfig()
        self.failed_learners_ = set()
        self.warning_learners_ = set()

    def add_error(self, learner_name: str, error_type: ErrorType, message: str,
                  fold: Optional[int] = None, phase: str = 'unknown',
                  severity: str = 'error', traceback_str: Optional[str] = None,
                  **context):
        """Add an error record with enhanced context.

        Parameters
        ----------
        learner_name : str
            Name of learner where error occurred
        error_type : ErrorType
            Categorized error type
        message : str
            Human-readable error description
        fold : int, optional
            CV fold number
        phase : str
            Stage where error occurred ('cv', 'final_refit', 'meta', 'prediction')
        severity : str
            'error' or 'warning'
        traceback_str : str, optional
            Full traceback
        **context : dict
            Additional contextual information (n_samples, class_distribution, etc.)
        """
        # Create enhanced error record
        if self.config.collect_error_context:
            record = EnhancedErrorRecord(
                learner_name=learner_name,
                fold=fold,
                error_type=error_type,
                message=message,
                phase=phase,
                severity=severity,
                traceback=traceback_str,
                **context
            )
        else:
            record = ErrorRecord(
                learner_name=learner_name,
                fold=fold,
                error_type=error_type,
                message=message,
                phase=phase,
                severity=severity,
                traceback=traceback_str
            )

        self.error_records.append(record)

        # Update learner status
        if learner_name not in self.learner_status:
            self.learner_status[learner_name] = {
                'total_errors': 0,
                'total_warnings': 0,
                'failed_folds': set(),
                'error_types': set(),
                'is_functional': True
            }

        status = self.learner_status[learner_name]
        if severity == 'error':
            status['total_errors'] += 1
            if fold is not None:
                status['failed_folds'].add(fold)
        else:
            status['total_warnings'] += 1

        status['error_types'].add(error_type.value)

        # Track failed/warning learners
        if severity == 'error':
            if phase == 'final_refit':
                self.failed_learners_.add(learner_name)
        else:
            self.warning_learners_.add(learner_name)

        # Print if verbose
        if self.verbose or self.config.verbose_errors:
            print(f"[{severity.upper()}] {learner_name} (fold {fold}, phase {phase}): {message}")
            if self.config.verbose_errors and traceback_str:
                print(f"Traceback:\n{traceback_str}")

    def check_viability(self, learner_name: str, n_folds: int) -> bool:
        """Check if a learner is viable based on error rate.

        Parameters
        ----------
        learner_name : str
            Name of learner to check
        n_folds : int
            Total number of folds

        Returns
        -------
        bool
            True if learner is viable, False otherwise
        """
        if learner_name not in self.learner_status:
            return True

        status = self.learner_status[learner_name]
        min_viable_folds = self.config.min_viable_folds
        if min_viable_folds is None:
            min_viable_folds = n_folds // 2

        n_failed_folds = len(status['failed_folds'])
        n_successful_folds = n_folds - n_failed_folds

        return n_successful_folds >= min_viable_folds

    def enforce_policy(self, n_learners: int, n_successful: int,
                       phase: str = 'unknown'):
        """Enforce error handling policy.

        Parameters
        ----------
        n_learners : int
            Total number of learners
        n_successful : int
            Number of successful learners
        phase : str
            Current phase

        Raises
        ------
        RuntimeError
            If policy is violated (STRICT mode or below min viable learners)
        """
        n_failed = n_learners - n_successful
        error_rate = n_failed / n_learners if n_learners > 0 else 0

        # Check minimum viable learners
        if n_successful < self.config.min_viable_learners:
            raise RuntimeError(
                f"Only {n_successful}/{n_learners} learners succeeded in {phase}. "
                f"Minimum required: {self.config.min_viable_learners}. "
                f"Failed learners: {self.failed_learners_}"
            )

        # Check error policy
        if self.config.error_policy == ErrorHandlingPolicy.STRICT:
            if n_failed > 0:
                raise RuntimeError(
                    f"{n_failed}/{n_learners} learner(s) failed in {phase}. "
                    f"STRICT policy does not allow errors. "
                    f"Failed learners: {self.failed_learners_}"
                )

        elif self.config.error_policy == ErrorHandlingPolicy.ADAPTIVE:
            if error_rate > self.config.max_error_rate:
                raise RuntimeError(
                    f"Error rate {error_rate:.1%} exceeds maximum "
                    f"{self.config.max_error_rate:.1%} in {phase}. "
                    f"{n_failed}/{n_learners} learners failed."
                )

        # Issue warnings for PERMISSIVE mode
        if self.config.error_policy == ErrorHandlingPolicy.PERMISSIVE:
            if n_failed > 0:
                warnings.warn(
                    f"Warning: {n_failed}/{n_learners} learner(s) failed in {phase}. "
                    f"Ensemble will use {n_successful} learners. "
                    f"Failed learners: {self.failed_learners_}",
                    category=SuperLearnerFitWarning
                )

    def get_error_summary_df(self) -> pd.DataFrame:
        """Get summary of errors by learner.

        Returns
        -------
        pd.DataFrame
            Summary with columns:
            - learner: learner name
            - n_errors: number of errors
            - n_warnings: number of warnings
            - error_types: list of error types
            - failed_folds: list of failed folds
            - is_functional: whether learner can be used
        """
        if len(self.learner_status) == 0:
            return pd.DataFrame()

        summary_rows = []
        for learner, status in self.learner_status.items():
            summary_rows.append({
                'learner': learner,
                'n_errors': status['total_errors'],
                'n_warnings': status['total_warnings'],
                'error_types': ', '.join(status['error_types']),
                'failed_folds': ', '.join(map(str, sorted(status['failed_folds']))),
                'is_functional': status['is_functional'],
            })

        return pd.DataFrame(summary_rows)


class _DummyFailedLearner(BaseEstimator, ClassifierMixin):
    """Placeholder for learners that failed to fit.

    Returns neutral predictions based on configuration.
    """

    def __init__(self, name: str, neutral_probability: float = 0.5):
        self.name = name
        self.neutral_probability = neutral_probability

    def fit(self, X, y, sample_weight=None):
        return self

    def predict_proba(self, X):
        n = X.shape[0] if hasattr(X, 'shape') else len(X)
        p = self.neutral_probability
        return np.column_stack([np.full(n, 1 - p), np.full(n, p)])

    def predict(self, X):
        n = X.shape[0] if hasattr(X, 'shape') else len(X)
        # Predict class 1 if neutral_prob > 0.5, else 0
        return np.full(n, int(self.neutral_probability > 0.5), dtype=int)


def categorize_error(exception: Exception) -> ErrorType:
    """Categorize exception into ErrorType.

    Parameters
    ----------
    exception : Exception
        The exception to categorize

    Returns
    -------
    ErrorType
        Categorized error type
    """
    error_msg = str(exception).lower()

    # Check for convergence issues
    if any(keyword in error_msg for keyword in
           ['converge', 'convergence', 'iteration', 'max_iter']):
        return ErrorType.CONVERGENCE

    # Check for NaN/Inf issues
    if any(keyword in error_msg for keyword in
           ['nan', 'inf', 'infinite', 'invalid value']):
        return ErrorType.NAN_INF

    # Check for data issues
    if any(keyword in error_msg for keyword in
           ['dimension', 'shape', 'empty', 'missing']):
        return ErrorType.DATA

    # Check for optimization issues
    if any(keyword in error_msg for keyword in
           ['optimization', 'minimize', 'gradient']):
        return ErrorType.OPTIMIZATION

    # Check exception type
    if isinstance(exception, (ValueError, TypeError)):
        return ErrorType.DATA

    if isinstance(exception, RuntimeError):
        return ErrorType.FITTING

    return ErrorType.OTHER


def handle_missing_data(X: np.ndarray, strategy: str = 'mean',
                       verbose: bool = False) -> Tuple[np.ndarray, Optional[SimpleImputer]]:
    """Handle missing data in features.

    Parameters
    ----------
    X : array-like
        Feature matrix potentially with NaN values
    strategy : str, default='mean'
        Imputation strategy ('mean', 'median', 'most_frequent')
    verbose : bool, default=False
        Whether to print information about imputation

    Returns
    -------
    X_imputed : np.ndarray
        Feature matrix with imputed values
    imputer : SimpleImputer or None
        Fitted imputer (for transform on test data), or None if no missing data

    Examples
    --------
    >>> X_train, imputer = handle_missing_data(X_train, strategy='mean')
    >>> X_test = imputer.transform(X_test) if imputer is not None else X_test
    """
    X_arr = np.asarray(X)

    if not np.any(np.isnan(X_arr)):
        return X_arr, None

    imputer = SimpleImputer(strategy=strategy)
    X_imputed = imputer.fit_transform(X_arr)

    if verbose:
        n_missing = np.sum(np.isnan(X_arr))
        pct_missing = 100 * n_missing / X_arr.size
        warnings.warn(
            f"Found {n_missing} missing values ({pct_missing:.2f}% of data). "
            f"Imputed using '{strategy}' strategy.",
            category=SuperLearnerDataWarning
        )

    return X_imputed, imputer


def safe_fit_with_policy(estimator, X, y, learner_name: str,
                        error_tracker: Optional[EnhancedErrorTracker] = None,
                        config: Optional[SuperLearnerConfig] = None,
                        sample_weight=None, fold: Optional[int] = None,
                        phase: str = 'cv') -> Tuple[Any, bool]:
    """Fit estimator with comprehensive error handling.

    Parameters
    ----------
    estimator : sklearn estimator
        Estimator to fit
    X : array-like
        Features
    y : array-like
        Target
    learner_name : str
        Name of learner (for error tracking)
    error_tracker : EnhancedErrorTracker, optional
        Error tracker instance
    config : SuperLearnerConfig, optional
        Configuration
    sample_weight : array-like, optional
        Sample weights
    fold : int, optional
        Fold number
    phase : str
        Phase of fitting ('cv', 'final_refit', etc.)

    Returns
    -------
    fitted_estimator : estimator or None
        Fitted estimator, or None if fitting failed
    success : bool
        Whether fitting succeeded
    """
    config = config or SuperLearnerConfig()

    try:
        # Try to fit with sample_weight
        if sample_weight is not None:
            try:
                estimator.fit(X, y, sample_weight=sample_weight)
            except TypeError:
                # Estimator doesn't support sample_weight
                estimator.fit(X, y)
        else:
            estimator.fit(X, y)

        return estimator, True

    except Exception as e:
        # Collect context if enabled
        context = {}
        if config.collect_error_context:
            context['n_samples'] = X.shape[0]
            context['n_features'] = X.shape[1]
            if hasattr(y, '__len__'):
                unique, counts = np.unique(y, return_counts=True)
                context['class_distribution'] = dict(zip(unique.tolist(), counts.tolist()))

        # Track error
        if error_tracker is not None:
            error_tracker.add_error(
                learner_name=learner_name,
                error_type=categorize_error(e),
                message=str(e),
                fold=fold,
                phase=phase,
                severity='error',
                traceback_str=traceback.format_exc() if config.verbose_errors else None,
                **context
            )

        return None, False


def safe_predict_with_policy(estimator, X, learner_name: str,
                             error_tracker: Optional[EnhancedErrorTracker] = None,
                             config: Optional[SuperLearnerConfig] = None,
                             get_proba_func=None) -> Tuple[np.ndarray, bool]:
    """Predict with comprehensive error handling.

    Parameters
    ----------
    estimator : sklearn estimator
        Fitted estimator
    X : array-like
        Features
    learner_name : str
        Name of learner (for error tracking)
    error_tracker : EnhancedErrorTracker, optional
        Error tracker instance
    config : SuperLearnerConfig, optional
        Configuration
    get_proba_func : callable, optional
        Function to get probabilities from estimator

    Returns
    -------
    predictions : np.ndarray
        Predictions (or neutral if failed)
    success : bool
        Whether prediction succeeded
    """
    config = config or SuperLearnerConfig()

    try:
        if get_proba_func is not None:
            preds = get_proba_func(estimator, X)
        elif hasattr(estimator, 'predict_proba'):
            preds = estimator.predict_proba(X)[:, 1]
        elif hasattr(estimator, 'decision_function'):
            from scipy.special import expit
            preds = expit(estimator.decision_function(X))
        else:
            preds = estimator.predict(X).astype(float)

        return preds, True

    except Exception as e:
        # Track error
        if error_tracker is not None:
            error_tracker.add_error(
                learner_name=learner_name,
                error_type=ErrorType.PREDICTION,
                message=f"Prediction failed: {str(e)}",
                fold=None,
                phase='prediction',
                severity='warning',
                n_samples=X.shape[0] if hasattr(X, 'shape') else len(X)
            )

        # Return neutral predictions
        n = X.shape[0] if hasattr(X, 'shape') else len(X)
        neutral = config.neutral_probability
        return np.full(n, neutral), False
