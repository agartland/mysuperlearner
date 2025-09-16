# Extended SuperLearner

A comprehensive Python implementation of the SuperLearner workflow with R SuperLearner-like functionality, featuring custom meta-learners, robust error handling, and external cross-validation.

## 🌟 Key Features

### 🎯 R SuperLearner Compatibility
- **method.NNloglik**: Non-negative log-likelihood optimization for binary classification
- **method.AUC**: AUC-maximizing meta-learner using Nelder-Mead optimization  
- **CV.SuperLearner equivalent**: External cross-validation for unbiased performance evaluation
- **SL.mean equivalent**: Simple mean predictor baseline

### 🛡️ Robust Error Handling
- Comprehensive error tracking per learner and CV fold
- Handles convergence failures, NaN/Inf values, and data issues
- Detailed error categorization and reporting
- Graceful degradation when individual learners fail

### 📊 Enhanced Evaluation
- External cross-validation with individual learner comparison
- Performance benchmarking and timing analysis
- Detailed diagnostic summaries and visualizations
    - Integration with a built-in evaluation framework (mlens not required)

## 🚀 Quick Start

### Basic Usage

```python
from extended_superlearner import create_superlearner
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Define base learners
learners = [
    ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('LogisticRegression', LogisticRegression(random_state=42)),
    ('SVM', SVC(probability=True, random_state=42))
]

# Create SuperLearner with NNLogLik meta-learner (like R's method.NNloglik)
sl = create_superlearner(
    learners=learners,
    method='nnloglik',  # or 'auc', 'nnls', 'logistic'
    # mysuperlearner

    A modern Python implementation of the Super Learner ensemble method.

    ## Installation

    ```bash
    pip install .
    ```

    ## Usage

    See `mysuperlearner/example_usage.py` for a usage example.

    ## Project Structure

    - `mysuperlearner/` - Main package code
    - `mysuperlearner/example_usage.py` - Example usage script
    - `README.md` - This file
    - `pyproject.toml` - Build configuration

    ## License

    MIT
- Python ≥ 3.7
- (no external mlens requirement)
- scikit-learn ≥ 0.24.0
- numpy ≥ 1.19.0
- pandas ≥ 1.1.0
- scipy ≥ 1.5.0

### Optional Dependencies
```bash
# For plotting
pip install matplotlib seaborn

# For development
pip install pytest pytest-cov black

# For documentation
pip install sphinx sphinx-rtd-theme
```

## 🎯 Meta-Learning Methods

| Method | Description | Use Case | R Equivalent |
|--------|-------------|----------|--------------|
| `'nnloglik'` | Non-negative log-likelihood | Binary classification (recommended) | method.NNloglik |
| `'auc'` | AUC maximization | Binary classification | method.AUC |
| `'nnls'` | Non-negative least squares | General purpose | method.NNLS |
| `'logistic'` | Logistic regression | Binary classification | Custom wrapper |

## 🔍 Comparison with R SuperLearner

| Feature | R SuperLearner | Extended SuperLearner | Notes |
|---------|----------------|----------------------|-------|
| **Core Algorithm** | ✅ | ✅ | Identical cross-validation approach |
| **method.NNLS** | ✅ | ✅ | Via sklearn LinearRegression(positive=True) |
| **method.NNloglik** | ✅ | ✅ | Custom implementation with L-BFGS-B |
| **method.AUC** | ✅ | ✅ | Nelder-Mead optimization |
| **CV.SuperLearner** | ✅ | ✅ | SuperLearnerCV class |
| **Error Handling** | ✅ | ✅ | Enhanced with detailed tracking |
| **SL.mean** | ✅ | ✅ | MeanEstimator class |
| **Screening** | ✅ | ✅ | Via built-in or user-provided preprocessing |
| **Parallel Processing** | ✅ | ✅ | Via joblib / sklearn backends or user configuration |

## 📈 Examples and Tutorials

### Example 1: Binary Classification with Different Meta-Learners

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Test different meta-learning methods
methods = ['nnloglik', 'auc', 'nnls', 'logistic']
results = {}

for method in methods:
    sl = create_superlearner(learners, method=method, random_state=42)
    sl.fit(X_train, y_train)
    
    y_pred_proba = sl.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    results[method] = auc_score
    print(f"{method}: AUC = {auc_score:.4f}")
```

### Example 2: Comprehensive Error Analysis

```python
# Create problematic dataset
X_prob = np.random.randn(500, 15)
X_prob[:, 0] = 1  # Constant feature
X_prob[0:50, 1] = np.inf  # Some infinite values
y_prob = np.random.binomial(1, 0.3, 500)

# SuperLearner with error tracking
sl = ExtendedSuperLearner(method='nnloglik', track_errors=True, verbose=True)
sl.add_learners(learners)

try:
    sl.fit(X_prob, y_prob)
    print("Training completed despite data issues!")
except Exception as e:
    print(f"Training failed: {e}")

# Analyze errors
error_df = sl.get_error_summary()
print("\nError Summary:")
print(error_df)
```

### Example 3: External Cross-Validation Study

```python
from sklearn.datasets import load_breast_cancer

# Load real dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Comprehensive evaluation
sl_cv = evaluate_superlearner(
    X=X, y=y,
    learners=learners,
    method='nnloglik',
    inner_cv=5,
    outer_cv=10,
    scoring='auc',
    random_state=42,
    include_individual=True
)

# Results analysis
cv_summary = sl_cv.get_cv_summary()
print("Cross-Validation Results:")
print(cv_summary)

# Statistical significance testing
from scipy import stats
sl_scores = sl_cv.cv_results_[sl_cv.cv_results_['estimator'] == 'SuperLearner']['test_score']
best_individual = sl_cv.cv_results_.groupby('estimator')['test_score'].mean().drop('SuperLearner').max()

print(f"\nSuperLearner mean: {sl_scores.mean():.4f} ± {sl_scores.std():.4f}")
print(f"Best individual: {best_individual:.4f}")
```

## 🧪 Testing and Development

### Running Tests
```bash
# Install development dependencies
pip install -e .[dev]

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=extended_superlearner --cov-report=html
```

### Code Formatting
```bash
# Format code
black extended_superlearner/

# Check formatting
black --check extended_superlearner/
```

## 📚 API Reference

### Main Classes

#### `ExtendedSuperLearner`
Extended SuperLearner with R-like functionality and error handling.

**Parameters:**
- `method` (str): Meta-learning method ('nnloglik', 'auc', 'nnls', 'logistic')
- `folds` (int): Number of CV folds for internal cross-validation
- `random_state` (int): Random seed for reproducibility
- `verbose` (bool): Enable detailed output
- `track_errors` (bool): Enable error tracking

**Key Methods:**
- `add_learners(learners)`: Add base learners
- `fit(X, y)`: Fit the SuperLearner
- `predict(X)`: Make predictions
- `predict_proba(X)`: Get prediction probabilities
- `print_summary()`: Print comprehensive summary
- `get_error_summary()`: Get error tracking DataFrame

#### `SuperLearnerCV`
External cross-validation for SuperLearner (CV.SuperLearner equivalent).

**Parameters:**
- `method` (str): Meta-learning method
- `inner_cv` (int): CV folds for SuperLearner construction
- `outer_cv` (int): CV folds for performance evaluation
- `random_state` (int): Random seed
- `verbose` (bool): Enable detailed output
- `n_jobs` (int): Number of parallel jobs

**Key Methods:**
- `fit(X, y, learners, scoring)`: Run external cross-validation
- `get_cv_summary()`: Get CV results summary
- `plot_cv_results()`: Plot performance comparison

### Custom Meta-Learners

#### `NNLogLikEstimator`
Non-negative log-likelihood meta-learner (method.NNloglik equivalent).

#### `AUCEstimator` 
AUC-maximizing meta-learner (method.AUC equivalent).

#### `MeanEstimator`
Simple mean predictor (SL.mean equivalent).

### Error Handling

#### `ErrorTracker`
Comprehensive error tracking system.

**Key Methods:**
- `add_error(learner_name, error_type, message)`: Record error
- `check_predictions(predictions, learner_name)`: Validate predictions
- `get_error_summary()`: Get summary DataFrame
- `print_summary()`: Print formatted error report

### Convenience Functions

#### `create_superlearner(learners, method='nnloglik', **kwargs)`
Create configured SuperLearner instance.

#### `evaluate_superlearner(X, y, learners, **kwargs)`
Run external cross-validation evaluation.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone repository
git clone https://github.com/your-username/extended-superlearner
cd extended-superlearner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/
```

### Contribution Areas
- Additional meta-learning methods
- More screening algorithms
- Performance optimizations
- Documentation improvements
- Bug fixes and testing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **R SuperLearner Team**: Original SuperLearner algorithm and implementation
- **mlens** (optional): Excellent ensemble learning framework for Python
- **scikit-learn**: Foundation for machine learning in Python

## 📞 Support

- **Documentation**: [Read the Docs](https://extended-superlearner.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/your-username/extended-superlearner/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/extended-superlearner/discussions)

## 🔗 Related Projects

- [R SuperLearner](https://github.com/ecpolley/SuperLearner): Original R implementation
- [mlens](https://github.com/flennerhag/mlens): High-performance ensemble learning (optional)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn): Machine learning library
- [Targeted Learning](https://tlverse.org/): Causal inference with SuperLearner

