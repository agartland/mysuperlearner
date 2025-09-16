"""
Error handling system for SuperLearner implementations.
Tracks convergence errors, NaN/Inf issues, and other failures.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ErrorType(Enum):
    """Types of errors that can occur in SuperLearner"""
    CONVERGENCE = "convergence"
    NAN_INF = "nan_inf"
    PREDICTION = "prediction"
    FITTING = "fitting"
    OPTIMIZATION = "optimization"
    DATA = "data"
    OTHER = "other"


@dataclass
class ErrorRecord:
    """Individual error record"""
    learner_name: str
    fold: Optional[int]
    error_type: ErrorType
    message: str
    phase: str  # 'cv', 'final', 'meta'
    severity: str  # 'error', 'warning'
    traceback: Optional[str] = None


class ErrorTracker:
    """
    Comprehensive error tracking system for SuperLearner.
    Tracks errors per learner, per fold, with detailed categorization.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.error_records: List[ErrorRecord] = []
        self.learner_status: Dict[str, Dict] = {}
        
    def add_error(self, learner_name: str, error_type: ErrorType, message: str,
                  fold: Optional[int] = None, phase: str = 'unknown', 
                  severity: str = 'error', traceback: Optional[str] = None):
        """Add an error record"""
    self.error_records.append(ErrorRecord(
            learner_name=learner_name,
            fold=fold,
            error_type=error_type,
            message=message,
            phase=phase,
            severity=severity,
            traceback=traceback
    ))
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
        
        if self.verbose:
            print(f"[{severity.upper()}] {learner_name} (fold {fold}): {message}")
    
    def add_warning(self, learner_name: str, error_type: ErrorType, message: str,
                   fold: Optional[int] = None, phase: str = 'unknown'):
        """Add a warning record"""
        self.add_error(learner_name, error_type, message, fold, phase, 'warning')
    
    def check_predictions(self, predictions: np.ndarray, learner_name: str, 
                         fold: Optional[int] = None, phase: str = 'cv') -> bool:
        """
        Check predictions for NaN/Inf values and other issues.
        Returns True if predictions are valid, False otherwise.
        """
        if predictions is None:
            self.add_error(learner_name, ErrorType.PREDICTION, 
                          "Predictions are None", fold, phase)
            return False
        
        # Check for NaN values
        if np.any(np.isnan(predictions)):
            nan_count = np.sum(np.isnan(predictions))
            self.add_error(learner_name, ErrorType.NAN_INF, 
                          f"Found {nan_count} NaN values in predictions", fold, phase)
            return False
        
        # Check for Inf values
        if np.any(np.isinf(predictions)):
            inf_count = np.sum(np.isinf(predictions))
            self.add_error(learner_name, ErrorType.NAN_INF, 
                          f"Found {inf_count} Inf values in predictions", fold, phase)
            return False
        
        # Check for extremely large values that might cause issues
        if np.any(np.abs(predictions) > 1e10):
            large_count = np.sum(np.abs(predictions) > 1e10)
            self.add_warning(learner_name, ErrorType.PREDICTION, 
                           f"Found {large_count} extremely large values (>1e10)", fold, phase)
        
        # For binary classification, check if probabilities are in valid range
        if np.all((predictions >= 0) & (predictions <= 1)):
            # Looks like probabilities
            if np.any((predictions <= 0.001) | (predictions >= 0.999)):
                extreme_count = np.sum((predictions <= 0.001) | (predictions >= 0.999))
                self.add_warning(learner_name, ErrorType.PREDICTION, 
                               f"Found {extreme_count} extreme probability values", fold, phase)
        
        return True
    
    def mark_learner_failed(self, learner_name: str, reason: str = "Multiple failures"):
        """Mark a learner as completely failed"""
        if learner_name in self.learner_status:
            self.learner_status[learner_name]['is_functional'] = False
        else:
            self.learner_status[learner_name] = {
                'total_errors': 1,
                'total_warnings': 0,
                'failed_folds': set(),
                'error_types': {'complete_failure'},
                'is_functional': False
            }
        
        self.add_error(learner_name, ErrorType.OTHER, reason, None, 'final')
    
    def get_functional_learners(self) -> List[str]:
        """Get list of learners that are still functional"""
        return [name for name, status in self.learner_status.items() 
                if status['is_functional']]
    
    def get_failed_learners(self) -> List[str]:
        """Get list of completely failed learners"""
        return [name for name, status in self.learner_status.items() 
                if not status['is_functional']]
    
    def get_error_summary(self) -> pd.DataFrame:
        """Generate summary DataFrame of errors and warnings"""
        if not self.error_records:
            return pd.DataFrame(columns=['learner', 'total_errors', 'total_warnings', 
                                       'failed_folds', 'error_types', 'is_functional'])
        
        summary_data = []
        for learner_name, status in self.learner_status.items():
            summary_data.append({
                'learner': learner_name,
                'total_errors': status['total_errors'],
                'total_warnings': status['total_warnings'],
                'failed_folds': len(status['failed_folds']),
                'error_types': ', '.join(sorted(status['error_types'])),
                'is_functional': status['is_functional']
            })
        
        return pd.DataFrame(summary_data)
    
    def get_detailed_errors(self) -> pd.DataFrame:
        """Get detailed DataFrame of all error records"""
        if not self.error_records:
            return pd.DataFrame(columns=['learner', 'fold', 'error_type', 'phase', 
                                       'severity', 'message'])
        
        records_data = []
        for record in self.error_records:
            records_data.append({
                'learner': record.learner_name,
                'fold': record.fold,
                'error_type': record.error_type.value,
                'phase': record.phase,
                'severity': record.severity,
                'message': record.message
            })
        
        return pd.DataFrame(records_data)
    
    def print_summary(self):
        """Print a formatted summary of errors and warnings"""
        print("\n" + "="*70)
        print("SUPERLEARNER ERROR SUMMARY")
        print("="*70)
        
        if not self.error_records:
            print("No errors or warnings recorded.")
            return
        
        summary_df = self.get_error_summary()
        
        # Overall statistics
        total_errors = summary_df['total_errors'].sum()
        total_warnings = summary_df['total_warnings'].sum()
        failed_learners = summary_df[~summary_df['is_functional']]['learner'].tolist()
        
        print(f"Total Errors: {total_errors}")
        print(f"Total Warnings: {total_warnings}")
        print(f"Failed Learners: {len(failed_learners)}")
        if failed_learners:
            print(f"  - {', '.join(failed_learners)}")
        
        print("\nPer-Learner Summary:")
        print("-" * 70)
        
        # Format the summary table
        for _, row in summary_df.iterrows():
            status = "FAILED" if not row['is_functional'] else "OK"
            print(f"{row['learner']:<20} | Errors: {row['total_errors']:>3} | "
                  f"Warnings: {row['total_warnings']:>3} | Status: {status}")
        
        # Show most common error types
        error_types = [record.error_type.value for record in self.error_records 
                      if record.severity == 'error']
        if error_types:
            from collections import Counter
            common_errors = Counter(error_types).most_common(3)
            print(f"\nMost Common Error Types:")
            for error_type, count in common_errors:
                print(f"  - {error_type}: {count} occurrences")
        
        print("="*70)


def safe_predict(estimator, X, learner_name: str, error_tracker: ErrorTracker, 
                fold: Optional[int] = None, phase: str = 'cv', 
                predict_proba: bool = True) -> Optional[np.ndarray]:
    """
    Safely make predictions with error handling.
    
    Parameters:
    -----------
    estimator : sklearn estimator
        The fitted estimator
    X : array-like
        Input features
    learner_name : str
        Name of the learner for error tracking
    error_tracker : ErrorTracker
        Error tracking instance
    fold : int, optional
        CV fold number
    phase : str
        Phase of learning ('cv', 'final', 'meta')
    predict_proba : bool
        Whether to use predict_proba (for classification)
    
    Returns:
    --------
    predictions : np.ndarray or None
        Predictions if successful, None if failed
    """
    try:
        if predict_proba and hasattr(estimator, 'predict_proba'):
            predictions = estimator.predict_proba(X)
            # For binary classification, return probabilities for class 1
            if predictions.shape[1] == 2:
                predictions = predictions[:, 1]
        elif hasattr(estimator, 'predict'):
            predictions = estimator.predict(X)
        else:
            error_tracker.add_error(learner_name, ErrorType.PREDICTION,
                                  "Estimator has no predict method", fold, phase)
            return None
        
        # Check predictions for issues
        if not error_tracker.check_predictions(predictions, learner_name, fold, phase):
            return None
            
        return predictions
        
    except Exception as e:
        error_tracker.add_error(learner_name, ErrorType.PREDICTION,
                              f"Prediction failed: {str(e)}", fold, phase)
        return None


def safe_fit(estimator, X, y, learner_name: str, error_tracker: ErrorTracker,
            fold: Optional[int] = None, phase: str = 'cv') -> bool:
    """
    Safely fit an estimator with error handling.
    
    Returns:
    --------
    success : bool
        True if fitting succeeded, False otherwise
    """
    try:
        estimator.fit(X, y)
        
        # Check for convergence warnings in some estimators
        if hasattr(estimator, 'n_iter_'):
            if hasattr(estimator, 'max_iter') and estimator.n_iter_ >= estimator.max_iter:
                error_tracker.add_warning(learner_name, ErrorType.CONVERGENCE,
                                        f"May not have converged (n_iter={estimator.n_iter_})",
                                        fold, phase)
        
        return True
        
    except Exception as e:
        error_message = str(e)
        
        # Categorize the error
        if any(keyword in error_message.lower() for keyword in 
               ['convergence', 'converge', 'iteration', 'tolerance']):
            error_type = ErrorType.CONVERGENCE
        elif any(keyword in error_message.lower() for keyword in 
                ['nan', 'inf', 'infinity', 'overflow']):
            error_type = ErrorType.NAN_INF
        elif any(keyword in error_message.lower() for keyword in 
                ['singular', 'rank', 'matrix']):
            error_type = ErrorType.DATA
        else:
            error_type = ErrorType.FITTING
        
        error_tracker.add_error(learner_name, error_type, 
                              f"Fitting failed: {error_message}", fold, phase)
        return False