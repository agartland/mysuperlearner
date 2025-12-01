# mysuperlearner

A comprehensive Python implementation of the SuperLearner ensemble method with R SuperLearner-like functionality, featuring custom meta-learners, robust error handling, and external cross-validation.

NOTE: This package is also an experiment in GenAI code development and was created using Claude Code with prompts focused on mimicking the functionality of R's SuperLearner package, and generating extensive tests, examples and documentation to validate its functionality.

## Key Features

### R SuperLearner Compatibility
- **method.NNloglik**: Non-negative log-likelihood optimization for binary classification
- **method.AUC**: AUC-maximizing meta-learner using Nelder-Mead optimization
- **External CV evaluation**: External cross-validation for unbiased performance evaluation (similar to CV.SuperLearner)
- **SL.mean equivalent**: Simple mean predictor baseline

### Robust Error Handling
- Comprehensive error tracking per learner and CV fold
- Handles convergence failures, NaN/Inf values, and data issues
- Detailed error categorization and reporting
- Graceful degradation when individual learners fail

### Enhanced Evaluation
- External cross-validation with individual learner comparison
- Performance benchmarking and timing analysis
- Detailed diagnostic summaries and visualizations
- Built-in evaluation framework (mlens not required)

## Quick Start

### Installation

```bash
pip install .
```

### Requirements
- Python ≥ 3.8
- scikit-learn
- numpy
- pandas
- scipy
- matplotlib (optional, for plotting)
- seaborn (optional, for plotting)

### Basic Usage

```python
from mysuperlearner import SuperLearner
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base learners
learners = [
    ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('LogisticRegression', LogisticRegression(random_state=42)),
    ('SVM', SVC(probability=True, random_state=42))
]

# Create SuperLearner with NNLogLik meta-learner (like R's method.NNloglik)
sl = SuperLearner(learners=learners, method='nnloglik', cv=5, random_state=42, verbose=True)

# Fit using standard sklearn API
sl.fit(X_train, y_train)

# Make predictions
y_pred_proba = sl.predict_proba(X_test)
y_pred = sl.predict(X_test)

# Evaluate
from sklearn.metrics import roc_auc_score, accuracy_score
print(f"AUC: {roc_auc_score(y_test, y_pred_proba[:, 1]):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

## Meta-Learning Methods

| Method | Description | Use Case | R Equivalent |
|--------|-------------|----------|--------------|
| `'nnloglik'` | Non-negative log-likelihood | Binary classification (recommended) | method.NNloglik |
| `'auc'` | AUC maximization | Binary classification | method.AUC |
| `'nnls'` | Non-negative least squares | General purpose | method.NNLS |
| `'logistic'` | Logistic regression | Binary classification | Custom wrapper |

## Comparison with R SuperLearner

| Feature | R SuperLearner | mysuperlearner | Notes |
|---------|----------------|----------------|-------|
| **Core Algorithm** | ✅ | ✅ | Identical cross-validation approach |
| **method.NNLS** | ✅ | ✅ | Via sklearn LinearRegression(positive=True) |
| **method.NNloglik** | ✅ | ✅ | Logit-scale optimization with L-BFGS-B |
| **method.AUC** | ✅ | ✅ | Nelder-Mead optimization |
| **CV.SuperLearner** | ✅ | ✅ | CVSuperLearner class |
| **Error Handling** | ✅ | ✅ | Enhanced with detailed tracking |
| **SL.mean** (base) | ✅ | ✅ | InterceptOnlyEstimator class |
| **SL.mean** (meta) | ✅ | ✅ | MeanEstimator class |
| **Screening** | ✅ | ✅ | VariableSet, CorrelationScreener, LassoScreener |
| **Flexible CV** | ✅ | ✅ | Supports sklearn CV splitters (GroupKFold, TimeSeriesSplit) |
| **Parallel Processing** | ✅ | ✅ | Via joblib / sklearn backends or user configuration |

## Examples and Tutorials

### Example 1: Binary Classification with Different Meta-Learners

```python
import numpy as np
from mysuperlearner import SuperLearner
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define base learners
learners = [
    ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('LogisticRegression', LogisticRegression(random_state=42)),
    ('SVM', SVC(probability=True, random_state=42))
]

# Test different meta-learning methods
methods = ['nnloglik', 'auc', 'nnls', 'logistic']
results = {}

for method in methods:
    sl = SuperLearner(learners=learners, method=method, cv=5, random_state=42)
    sl.fit(X_train, y_train)

    y_pred_proba = sl.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)

    results[method] = auc_score
    print(f"{method}: AUC = {auc_score:.4f}")
```

### Example 2: External Cross-Validation Study

```python
from mysuperlearner import CVSuperLearner
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Load real dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Define base learners
learners = [
    ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('LogisticRegression', LogisticRegression(random_state=42)),
    ('SVM', SVC(probability=True, random_state=42))
]

# Run external cross-validation
cv_sl = CVSuperLearner(learners=learners, method='nnloglik', cv=10, inner_cv=5, random_state=42)
cv_sl.fit(X, y)
cv_results = cv_sl.get_results()

# Analyze results
print("\nCross-Validation Results:")
print(cv_results.summary()[['auc', 'accuracy']])

# Compare SuperLearner to best individual
comparison = cv_results.compare_to_best()
print("\nSuperLearner vs Best Base Learner:")
print(comparison[['metric', 'SuperLearner', 'Best_Base', 'Best_Base_Name', 'Improvement']])
```

## Testing and Development

### Running Tests
```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/ -v
```

## API Reference

### Main Classes

#### `SuperLearner`
SuperLearner ensemble with R-like functionality and sklearn-compatible API.

**Parameters:**
- `learners`: List of (name, estimator) tuples - base learning algorithms
- `method` (str): Meta-learning method ('nnloglik', 'auc', 'nnls', 'logistic') (default: 'nnloglik')
- `cv` (int or CV splitter): Cross-validation strategy (default: 5)
  - Integer: number of StratifiedKFold splits
  - CV splitter: sklearn CV object (e.g., GroupKFold, TimeSeriesSplit)
- `random_state` (int): Random seed for reproducibility
- `verbose` (bool): Enable detailed output (default: False)
- `track_errors` (bool): Enable error tracking (default: True)
- `trim` (float): Probability trimming for NNLogLik (default: 0.001)

**Key Methods:**
- `fit(X, y, sample_weight=None, store_X=False, groups=None)`: Fit the SuperLearner
  - `groups`: Group labels for GroupKFold CV
- `predict(X)`: Make binary predictions (0 or 1)
- `predict_proba(X)`: Get prediction probabilities (returns array of shape (n_samples, 2))
- `get_diagnostics()`: Get model diagnostics and error information

**Attributes:**
- `meta_weights_`: Learned meta-learner weights (if applicable)
- `base_learners_full_`: List of (name, fitted_model) tuples trained on full data
- `Z_`: Level-1 cross-validated predictions matrix
- `cv_predictions_`: List of CV predictions per learner

#### `CVSuperLearner`
External cross-validation for unbiased performance evaluation (similar to R's CV.SuperLearner).

**Parameters:**
- `learners`: List of (name, estimator) tuples - base learning algorithms
- `method` (str): Meta-learning method (default: 'nnloglik')
- `cv` (int or CV splitter): Outer CV strategy (default: 5)
- `inner_cv` (int or CV splitter): Inner CV for meta-learner (default: 5)
- `random_state` (int): Random seed
- `verbose` (bool): Enable detailed output (default: False)
- `n_jobs` (int): Number of parallel jobs (default: 1)

**Key Methods:**
- `fit(X, y, sample_weight=None, groups=None)`: Perform CV evaluation
- `get_results()`: Get SuperLearnerCVResults object with metrics and predictions

**SuperLearnerCVResults Methods:**
- `summary()`: Get summary statistics across CV folds
- `compare_to_best()`: Compare SuperLearner to best base learner
- `plot_forest()`, `plot_boxplot()`, `plot_roc_curves()`, `plot_calibration()`: Visualization methods

### Base Learners

#### `InterceptOnlyEstimator`
Intercept-only baseline predictor (equivalent to R's `SL.mean` base learner).

Predicts the empirical mean of the training data for all samples, completely ignoring input features. This serves as a performance baseline - if more complex learners can't beat this simple predictor, they're not adding value.

**Example:**
```python
from mysuperlearner import InterceptOnlyEstimator

learners = [
    ('Baseline', InterceptOnlyEstimator()),
    ('RF', RandomForestClassifier()),
    # ... other learners
]
```

### Custom Meta-Learners

#### `NNLogLikEstimator`
Non-negative log-likelihood meta-learner (method.NNloglik equivalent).

#### `AUCEstimator`
AUC-maximizing meta-learner (method.AUC equivalent).

#### `MeanEstimator`
Simple mean predictor meta-learner - averages predictions from base learners.

### Screening Classes

#### `VariableSet`
Manual feature selection by variable names or indices.

**Example:**
```python
from mysuperlearner import VariableSet
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Select specific features
var_set = VariableSet(variables=['age', 'sex', 'bmi'], name='baseline')
baseline_lr = Pipeline([('screen', var_set), ('lr', LogisticRegression())])

learners = [
    ('Full_LR', LogisticRegression()),
    ('Baseline_LR', baseline_lr)
]
```

#### `CorrelationScreener`
Screen features by correlation with outcome.

**Example:**
```python
from mysuperlearner import CorrelationScreener

# Select features with |correlation| >= 0.15
screener = CorrelationScreener(threshold=0.15, name='high_corr')
screened_lr = Pipeline([('screen', screener), ('lr', LogisticRegression())])
```

#### `LassoScreener`
Screen features using Lasso regularization.

**Example:**
```python
from mysuperlearner import LassoScreener

# Automatic alpha selection via CV
screener = LassoScreener(classification=True, name='lasso_selected')
screened_lr = Pipeline([('screen', screener), ('lr', LogisticRegression())])
```

### Error Handling

The package includes comprehensive error tracking via the `ErrorTracker` class:

**Key Methods:**
- `add_error(learner_name, error_type, message, fold=None, phase='unknown', severity='error')`: Record error
- Error types: CONVERGENCE, NAN_INF, PREDICTION, FITTING, OPTIMIZATION, DATA, OTHER

**Usage:**
Error tracking is enabled by default when creating a `SuperLearner` with `track_errors=True`. Errors are stored in `sl.error_tracker.error_records`.

## Contributing

Contributions are welcome! Areas for contribution include:
- Additional meta-learning methods
- Performance optimizations
- Documentation improvements
- Bug fixes and testing

## License

This project is licensed under the MIT License.

## Acknowledgments

- **R SuperLearner Team**: Original SuperLearner algorithm and implementation
- **scikit-learn**: Foundation for machine learning in Python

## Related Projects

- [R SuperLearner](https://github.com/ecpolley/SuperLearner): Original R implementation
- [scikit-learn](https://github.com/scikit-learn/scikit-learn): Machine learning library

