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


def safe_fit(estimator, X, y, **kwargs):
    """Try to fit estimator, return fitted estimator or raise after recording error."""
    try:
        return estimator.fit(X, y, **kwargs)
    except Exception as e:
        raise


def safe_predict(estimator, X):
    """Try to predict probabilities or decision function, with fallbacks."""
    if hasattr(estimator, 'predict_proba'):
        return estimator.predict_proba(X)
    if hasattr(estimator, 'decision_function'):
        from scipy.special import expit
        return expit(estimator.decision_function(X))
    return estimator.predict(X)

    # ... (rest of the ErrorTracker class)
