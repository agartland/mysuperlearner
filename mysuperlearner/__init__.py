"""mysuperlearner package public interface"""

from .super_learner import SuperLearner
from .meta_learners import NNLogLikEstimator, AUCEstimator, MeanEstimator, InterceptOnlyEstimator
from .error_handling import ErrorTracker, ErrorType
from .cv_super_learner import CVSuperLearner
from .results import SuperLearnerCVResults
from .variable_importance import compute_variable_importance, VariableImportanceResults
from .screening import VariableSet, CorrelationScreener, LassoScreener
from . import visualization

# Deprecated aliases for backward compatibility (will be removed in v0.3.0)
from .super_learner import SuperLearner as ExtendedSuperLearner
from .cv_super_learner import CVSuperLearner as evaluate_super_learner_cv

__version__ = '0.2.0'

__all__ = [
	'SuperLearner',
	'CVSuperLearner',
	'NNLogLikEstimator', 'AUCEstimator', 'MeanEstimator', 'InterceptOnlyEstimator',
	'ErrorTracker', 'ErrorType',
	'SuperLearnerCVResults',
	'compute_variable_importance',
	'VariableImportanceResults',
	'VariableSet', 'CorrelationScreener', 'LassoScreener',
	'visualization',
	# Deprecated (will be removed in v0.3.0)
	'ExtendedSuperLearner',
	'evaluate_super_learner_cv',
]
