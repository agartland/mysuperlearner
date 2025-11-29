# Suggested Package Enhancements

This document outlines suggested enhancements to the `mysuperlearner` package to enable richer visualizations and analysis capabilities, particularly for creating comprehensive reports like those in the example Quarto document.

## 1. Store Fold-Level Predictions in CV Evaluation

### Current Limitation
The `evaluate_super_learner_cv()` function returns only aggregated metrics (AUC, accuracy, log loss) per fold. It does **not** store the actual predictions, which are needed for:
- Generating ROC curves from cross-validation
- Calibration plots
- Prediction distribution analysis
- Detailed diagnostic plots

### Proposed Enhancement

**File**: `mysuperlearner/evaluation.py`

Add an optional parameter `return_predictions=False` to `evaluate_super_learner_cv()`:

```python
def evaluate_super_learner_cv(
    X,
    y,
    base_learners: List[Tuple[str, Any]],
    super_learner: ExtendedSuperLearner,
    outer_folds: int = 5,
    random_state: Optional[int] = None,
    sample_weight: Optional[Any] = None,
    metrics: Optional[Dict[str, Callable]] = None,
    n_jobs: Optional[int] = 1,
    return_predictions: bool = False,  # NEW
):
    """
    ...

    Parameters
    ----------
    ...
    return_predictions : bool, default=False
        If True, return both metrics DataFrame and predictions dict

    Returns
    -------
    results : pd.DataFrame
        Per-fold metrics for each learner
    predictions : dict (only if return_predictions=True)
        Dictionary with keys:
        - 'y_true': concatenated true labels across all folds
        - 'y_pred': dict mapping learner name to predicted probabilities
        - 'fold_id': array indicating which fold each sample belongs to
        - 'test_indices': list of test indices per fold
    """
```

**Implementation**:
```python
# In _run_fold function, also return predictions
def _run_fold(fold_idx, train_idx, test_idx):
    # ... existing code ...

    local_predictions = {
        'SuperLearner': sl_p,
        'y_true': y_te,
        'test_idx': test_idx
    }

    for name, mdl in sl.base_learners_full_:
        # ... existing code ...
        local_predictions[name] = p

    return local_rows, local_predictions  # Return both

# After parallel execution, aggregate predictions
if return_predictions:
    all_predictions = {'y_true': [], 'fold_id': []}
    for learner_name in ['SuperLearner'] + [n for n, _ in base_learners]:
        all_predictions[learner_name] = []

    for fold_idx, (rows, preds) in enumerate(results):
        n_samples = len(preds['y_true'])
        all_predictions['y_true'].extend(preds['y_true'])
        all_predictions['fold_id'].extend([fold_idx] * n_samples)

        for learner_name in all_predictions.keys():
            if learner_name not in ['y_true', 'fold_id']:
                all_predictions[learner_name].extend(preds[learner_name])

    # Convert to arrays
    for key in all_predictions:
        all_predictions[key] = np.array(all_predictions[key])

    return results_df, all_predictions
else:
    return results_df
```

### Benefits
- Generate cross-validated ROC curves
- Create calibration plots
- Analyze prediction distributions
- Compare predicted probabilities across learners

---

## 2. Add Visualization Module

### Proposed Addition

**New File**: `mysuperlearner/visualization.py`

Create a dedicated visualization module with common plotting functions:

```python
"""
Visualization utilities for SuperLearner results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, calibration_curve
from typing import Dict, List, Optional, Tuple


def plot_cv_forest(cv_results: pd.DataFrame,
                   metric: str = 'auc',
                   figsize: Tuple[int, int] = (10, 6),
                   title: Optional[str] = None):
    """
    Create forest plot of cross-validation results.

    Parameters
    ----------
    cv_results : pd.DataFrame
        Results from evaluate_super_learner_cv()
    metric : str
        Metric to plot (e.g., 'auc', 'accuracy')
    figsize : tuple
        Figure size
    title : str, optional
        Plot title

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    # Calculate mean and 95% CI
    forest_data = []
    learners = cv_results['learner'].unique()

    for learner in learners:
        learner_data = cv_results[cv_results['learner'] == learner][metric]
        mean_val = learner_data.mean()
        std_val = learner_data.std()
        n = len(learner_data)
        ci = 1.96 * std_val / np.sqrt(n)

        forest_data.append({
            'learner': learner,
            'mean': mean_val,
            'ci_lower': mean_val - ci,
            'ci_upper': mean_val + ci,
            'is_super': learner == 'SuperLearner'
        })

    forest_df = pd.DataFrame(forest_data)
    forest_df = forest_df.sort_values('mean')

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    colors = ['#d62728' if x else '#1f77b4' for x in forest_df['is_super']]
    y_pos = np.arange(len(forest_df))

    for i, row in forest_df.iterrows():
        ax.plot([row['ci_lower'], row['ci_upper']], [i, i],
               color=colors[i], linewidth=2, alpha=0.8)
        ax.scatter(row['mean'], i, s=200, color=colors[i],
                  zorder=3, edgecolors='black', linewidth=1.5,
                  marker='D' if row['is_super'] else 'o')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(forest_df['learner'])
    ax.set_xlabel(f'{metric.upper()} (95% CI)', fontsize=12)
    ax.set_title(title or f'Cross-Validated {metric.upper()}: Forest Plot',
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    return fig, ax


def plot_cv_roc_curves(predictions: Dict[str, np.ndarray],
                       figsize: Tuple[int, int] = (10, 8),
                       title: Optional[str] = None):
    """
    Plot ROC curves from cross-validation predictions.

    Parameters
    ----------
    predictions : dict
        Dictionary from evaluate_super_learner_cv() with return_predictions=True
        Must contain 'y_true' and learner prediction arrays
    figsize : tuple
        Figure size
    title : str, optional
        Plot title

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    y_true = predictions['y_true']

    # Plot SuperLearner first
    if 'SuperLearner' in predictions:
        y_pred = predictions['SuperLearner']
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=3,
               label=f'SuperLearner (AUC={roc_auc:.4f})',
               color='#d62728', linestyle='-')

    # Plot base learners
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    color_idx = 0

    for learner_name in predictions.keys():
        if learner_name in ['y_true', 'fold_id', 'SuperLearner']:
            continue

        y_pred = predictions[learner_name]
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2,
               label=f'{learner_name} (AUC={roc_auc:.4f})',
               color=colors[color_idx], linestyle='--', alpha=0.7)
        color_idx += 1

    # Diagonal reference
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5,
           label='Random (AUC=0.5000)')

    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(title or 'Cross-Validated ROC Curves',
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)

    return fig, ax


def plot_calibration_curves(predictions: Dict[str, np.ndarray],
                           n_bins: int = 10,
                           figsize: Tuple[int, int] = (10, 8),
                           title: Optional[str] = None):
    """
    Plot calibration curves for learners.

    Parameters
    ----------
    predictions : dict
        Dictionary from evaluate_super_learner_cv() with return_predictions=True
    n_bins : int
        Number of bins for calibration
    figsize : tuple
        Figure size
    title : str, optional
        Plot title

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    y_true = predictions['y_true']

    # Plot perfect calibration
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)

    # Plot each learner
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    color_idx = 0

    for learner_name in predictions.keys():
        if learner_name in ['y_true', 'fold_id']:
            continue

        y_pred = predictions[learner_name]

        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred, n_bins=n_bins, strategy='uniform'
            )

            marker = 'D' if learner_name == 'SuperLearner' else 'o'
            linewidth = 3 if learner_name == 'SuperLearner' else 2

            ax.plot(mean_predicted_value, fraction_of_positives,
                   marker=marker, linewidth=linewidth,
                   label=learner_name,
                   color='#d62728' if learner_name == 'SuperLearner' else colors[color_idx])

            if learner_name != 'SuperLearner':
                color_idx += 1
        except:
            continue

    ax.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
    ax.set_title(title or 'Calibration Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)

    return fig, ax


def plot_learner_comparison(cv_results: pd.DataFrame,
                           metrics: List[str] = ['auc', 'accuracy'],
                           figsize: Tuple[int, int] = (12, 5)):
    """
    Create comparison plot across multiple metrics.

    Parameters
    ----------
    cv_results : pd.DataFrame
        Results from evaluate_super_learner_cv()
    metrics : list of str
        Metrics to compare
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        summary = cv_results.groupby('learner')[metric].agg(['mean', 'std'])
        summary = summary.sort_values('mean')

        colors = ['#d62728' if x == 'SuperLearner' else '#1f77b4'
                 for x in summary.index]

        axes[i].barh(summary.index, summary['mean'], xerr=summary['std'],
                    color=colors, alpha=0.7, ecolor='black', capsize=5)
        axes[i].set_xlabel(metric.upper(), fontsize=12)
        axes[i].set_ylabel('Learner', fontsize=12)
        axes[i].set_title(f'{metric.upper()} Comparison',
                         fontsize=13, fontweight='bold')
        axes[i].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig, axes
```

### Benefits
- Standardized, reusable plotting functions
- Consistent styling across visualizations
- Easy to maintain and extend
- Reduces code duplication in analysis scripts

---

## 3. Enhanced Result Object

### Proposed Enhancement

Create a dedicated result class that bundles all information together:

**New File**: `mysuperlearner/results.py`

```python
"""
Result classes for SuperLearner evaluation.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class SuperLearnerCVResults:
    """
    Container for cross-validation results with built-in plotting methods.

    Attributes
    ----------
    metrics : pd.DataFrame
        Per-fold metrics for each learner
    predictions : dict, optional
        Dictionary of predictions if return_predictions=True
    config : dict
        Configuration used for CV (folds, method, etc.)
    """
    metrics: pd.DataFrame
    predictions: Optional[Dict[str, np.ndarray]] = None
    config: Optional[Dict] = None

    def summary(self) -> pd.DataFrame:
        """Return summary statistics (mean, std, CI) for each learner."""
        summary = self.metrics.groupby('learner').agg(['mean', 'std', 'count'])
        # Add 95% CI
        for col in summary.columns.levels[0]:
            summary[(col, 'ci_lower')] = (
                summary[(col, 'mean')] - 1.96 * summary[(col, 'std')] / np.sqrt(summary[(col, 'count')])
            )
            summary[(col, 'ci_upper')] = (
                summary[(col, 'mean')] + 1.96 * summary[(col, 'std')] / np.sqrt(summary[(col, 'count')])
            )
        return summary

    def plot_forest(self, metric='auc', **kwargs):
        """Create forest plot for specified metric."""
        from .visualization import plot_cv_forest
        return plot_cv_forest(self.metrics, metric=metric, **kwargs)

    def plot_roc_curves(self, **kwargs):
        """Create ROC curve plot (requires predictions)."""
        if self.predictions is None:
            raise ValueError("Predictions not available. Set return_predictions=True in evaluate_super_learner_cv()")
        from .visualization import plot_cv_roc_curves
        return plot_cv_roc_curves(self.predictions, **kwargs)

    def plot_calibration(self, **kwargs):
        """Create calibration plot (requires predictions)."""
        if self.predictions is None:
            raise ValueError("Predictions not available. Set return_predictions=True in evaluate_super_learner_cv()")
        from .visualization import plot_calibration_curves
        return plot_calibration_curves(self.predictions, **kwargs)

    def compare_to_best(self) -> pd.DataFrame:
        """Compare SuperLearner to best individual learner."""
        sl_metrics = self.metrics[self.metrics['learner'] == 'SuperLearner'].mean(numeric_only=True)
        base_metrics = self.metrics[self.metrics['learner_type'] == 'base']

        comparison = []
        for metric in sl_metrics.index:
            if metric in base_metrics.columns:
                best_base = base_metrics.groupby('learner')[metric].mean().max()
                best_name = base_metrics.groupby('learner')[metric].mean().idxmax()

                comparison.append({
                    'metric': metric,
                    'SuperLearner': sl_metrics[metric],
                    'Best_Base': best_base,
                    'Best_Base_Name': best_name,
                    'Improvement': sl_metrics[metric] - best_base,
                    'Improvement_Pct': ((sl_metrics[metric] - best_base) / best_base * 100)
                })

        return pd.DataFrame(comparison)

    def __repr__(self):
        n_folds = self.metrics['fold'].nunique()
        n_learners = self.metrics['learner'].nunique()
        has_preds = self.predictions is not None
        return (f"SuperLearnerCVResults(n_folds={n_folds}, n_learners={n_learners}, "
                f"has_predictions={has_preds})")
```

**Update** `evaluation.py` to return this object:

```python
def evaluate_super_learner_cv(..., return_object=True):
    """
    ...

    Returns
    -------
    results : SuperLearnerCVResults or pd.DataFrame
        If return_object=True, returns SuperLearnerCVResults object
        Otherwise returns pd.DataFrame (backward compatible)
    """
    # ... existing code ...

    if return_object:
        from .results import SuperLearnerCVResults
        config = {
            'outer_folds': outer_folds,
            'random_state': random_state,
            'method': super_learner.method
        }
        return SuperLearnerCVResults(
            metrics=results_df,
            predictions=all_predictions if return_predictions else None,
            config=config
        )
    else:
        if return_predictions:
            return results_df, all_predictions
        else:
            return results_df
```

### Benefits
- Cleaner API with built-in plotting methods
- Easy access to summary statistics
- Encapsulated result handling
- Backward compatible with existing code

---

## 4. Add Meta-Learner Diagnostics

### Proposed Enhancement

Add a method to `ExtendedSuperLearner` to extract diagnostic information:

```python
def get_diagnostics(self) -> Dict:
    """
    Get diagnostic information about the fitted SuperLearner.

    Returns
    -------
    diagnostics : dict
        Dictionary containing:
        - 'meta_weights': Array of meta-learner weights (if available)
        - 'cv_scores': Per-fold CV scores for each base learner
        - 'base_learner_names': Names of base learners
        - 'method': Meta-learning method used
        - 'n_folds': Number of CV folds
        - 'errors': Error tracker records (if available)
    """
    diagnostics = {
        'method': self.method,
        'n_folds': self.folds,
        'base_learner_names': getattr(self, 'base_learner_names_', []),
        'meta_weights': getattr(self, 'meta_weights_', None),
    }

    # Add CV scores if available
    if hasattr(self, 'cv_predictions_') and hasattr(self, 'Z_'):
        # Calculate per-learner CV AUC
        cv_scores = {}
        if hasattr(self, 'y_'):  # Store y in fit_explicit
            for i, name in enumerate(self.base_learner_names_):
                try:
                    from sklearn.metrics import roc_auc_score
                    cv_scores[name] = roc_auc_score(self.y_, self.Z_[:, i])
                except:
                    cv_scores[name] = np.nan
        diagnostics['cv_scores'] = cv_scores

    # Add error information
    if self.error_tracker is not None:
        diagnostics['errors'] = self.error_tracker.error_records

    return diagnostics
```

Also store `y` in `fit_explicit`:
```python
def fit_explicit(self, X, y, base_learners, sample_weight=None):
    X_arr, y_arr = check_X_y(X, y)
    self.y_ = y_arr  # Store for diagnostics
    # ... rest of method
```

---

## 5. Summary of Enhancements

| Enhancement | Priority | Impact | Effort |
|------------|----------|--------|--------|
| Store fold-level predictions | HIGH | Enables CV-based ROC curves, calibration plots | Medium |
| Visualization module | HIGH | Standardized, reusable plots | Medium |
| Enhanced result object | MEDIUM | Cleaner API, easier to use | Low |
| Meta-learner diagnostics | MEDIUM | Better introspection and debugging | Low |

## Implementation Priority

1. **Phase 1** (High Priority):
   - Add `return_predictions` to `evaluate_super_learner_cv()`
   - Create `visualization.py` module with core plotting functions

2. **Phase 2** (Medium Priority):
   - Create `SuperLearnerCVResults` class
   - Add `get_diagnostics()` method to `ExtendedSuperLearner`

3. **Phase 3** (Enhancement):
   - Add more visualization options (calibration curves, prediction distributions)
   - Add statistical tests for comparing learners
   - Add feature importance aggregation across base learners

## Backward Compatibility

All enhancements use optional parameters with defaults that maintain backward compatibility:
- `return_predictions=False` by default
- `return_object=True` can be added with fallback to DataFrame
- New visualization module doesn't affect existing code

## Testing Considerations

Each enhancement should include:
- Unit tests for new functions
- Integration tests with existing code
- Example notebooks demonstrating usage
- Documentation updates
