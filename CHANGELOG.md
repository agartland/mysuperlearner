# Changelog

All notable changes to the mysuperlearner package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-29

### Added

#### Core Features
- **ExtendedSuperLearner**: Main ensemble class with R SuperLearner compatibility
  - Support for multiple meta-learning methods: 'nnloglik', 'auc', 'nnls', 'logistic'
  - Internal cross-validation with configurable folds
  - Explicit learner builder API (`fit_explicit()`)
  - Support for sample weights in fitting and cross-validation

#### Meta-Learners
- **NNLogLikEstimator**: Non-negative log-likelihood optimization (R's method.NNloglik)
- **AUCEstimator**: AUC-maximizing meta-learner (R's method.AUC)
- **MeanEstimator**: Simple mean predictor meta-learner
- **InterceptOnlyEstimator**: Baseline predictor (R's SL.mean as base learner)

#### Error Handling & Robustness
- **ErrorTracker**: Comprehensive error tracking system
  - Per-learner error categorization (CONVERGENCE, NAN_INF, PREDICTION, FITTING, etc.)
  - Phase tracking (cv, final_refit, prediction)
  - Detailed error context with optional tracebacks
- **Graceful degradation**: Failed learners replaced with dummy predictors
- **min_viable_learners**: Configurable threshold for minimum working learners
- **Automatic error recovery**: Ensemble continues working when individual learners fail
- **Prediction error tracking**: Failed predictions use neutral probability (0.5) with warnings

#### Evaluation & Cross-Validation
- **evaluate_super_learner_cv()**: External cross-validation (R's CV.SuperLearner equivalent)
  - Per-fold metrics for all learners
  - Support for custom metrics
  - Optional prediction storage for ROC/calibration curves
  - Parallel execution support with joblib
- **SuperLearnerCVResults**: Result container with built-in analysis methods
  - `.summary()`: Statistical summaries with confidence intervals
  - `.plot_forest()`: Forest plots with CIs
  - `.plot_roc_curves()`: Cross-validated ROC curves
  - `.plot_calibration()`: Calibration plots
  - `.compare_to_best()`: SuperLearner vs best individual learner

#### Visualization
- **plot_cv_forest()**: Forest plots of cross-validation results
- **plot_cv_roc_curves()**: ROC curves from CV predictions
- **plot_calibration_curves()**: Calibration plots
- **plot_learner_comparison()**: Multi-metric comparison plots
- All plots support customization (figsize, title, colors, etc.)

#### Variable Importance
- **compute_variable_importance()**: Multiple importance methods
  - Permutation Feature Importance (PFI) with re-training
  - Drop-column importance
  - Grouped PFI with hierarchical clustering
  - SHAP support (optional, requires shap package)
- **VariableImportanceResults**: Container with plotting methods
  - `.summary()`: Top N important features
  - `.plot_importance_bar()`: Bar chart of feature importance
  - `.plot_importance_heatmap()`: Heatmap by method
  - `.plot_feature_clusters()`: Hierarchical clustering dendrogram

#### Testing & Documentation
- **42 comprehensive tests** covering:
  - All meta-learning methods
  - Edge cases (imbalanced data, missing values, convergence failures)
  - Error handling and recovery
  - External cross-validation
  - Variable importance methods
- **Extensive documentation**:
  - Error handling guide (45 pages)
  - Example Quarto documents with executable code
  - Comprehensive API documentation
  - Comparison with R SuperLearner

### Technical Details

#### API Design
- Clean, consistent API following scikit-learn conventions
- All core classes are sklearn-compatible (BaseEstimator, ClassifierMixin)
- Support for pipelines and scikit-learn utilities
- Type hints throughout codebase

#### Performance
- Efficient numpy-based implementations
- Optional parallel execution for cross-validation
- Minimal computational overhead for error handling (<0.1%)

#### Compatibility
- Python ≥ 3.8
- Core dependencies: scikit-learn, numpy, pandas, scipy
- Optional dependencies: matplotlib, seaborn (for visualization), shap (for SHAP importance)

### Package Structure
```
mysuperlearner/
├── __init__.py                    # Public API
├── extended_super_learner.py      # Main SuperLearner class
├── meta_learners.py               # Meta-learning methods
├── error_handling.py              # Error tracking system
├── evaluation.py                  # External CV evaluation
├── results.py                     # Result container classes
├── visualization.py               # Plotting functions
└── variable_importance.py         # Feature importance methods

docs/
├── ERROR_HANDLING_GUIDE.md        # User guide for error handling
├── ERROR_HANDLING_ANALYSIS.md     # Technical analysis
├── examples/                      # Example analyses and tutorials
│   ├── example_analysis.qmd
│   ├── variable_importance_guide.qmd
│   ├── TESTING_SUMMARY.qmd
│   └── variable_importance_example.py
├── analysis/                      # Comparison notebooks
│   ├── python_superlearner.qmd
│   └── r_superlearner.qmd
└── archive/                       # Development documentation

tests/
├── test_edge_cases.py             # Comprehensive edge case tests
├── test_evaluation.py             # CV evaluation tests
└── test_level1_builder.py         # Level-1 builder tests
```

### Acknowledgments
- R SuperLearner team for the original algorithm and design
- scikit-learn for the excellent ML framework

---

## [Unreleased]

### Planned Features
- Additional meta-learning methods
- More sophisticated ensemble strategies
- Enhanced parallel processing
- Additional visualization options
