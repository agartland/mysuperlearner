"""
Visualization utilities for SuperLearner results.

This module provides standalone plotting functions that accept SuperLearner
result data as parameters. These functions can be used independently or
called as convenience methods from result objects.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from typing import Dict, List, Optional, Tuple, Union

# Try to import calibration_curve, provide fallback if not available
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    # Fallback for older sklearn versions
    try:
        from sklearn.metrics import calibration_curve
    except ImportError:
        calibration_curve = None


def plot_cv_forest(
    cv_results: pd.DataFrame,
    metric: str = 'auc',
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create forest plot of cross-validation results with confidence intervals.

    Parameters
    ----------
    cv_results : pd.DataFrame
        Results from evaluate_super_learner_cv() containing columns:
        'learner', 'learner_type', and the specified metric
    metric : str, default='auc'
        Metric to plot (e.g., 'auc', 'accuracy', 'logloss')
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches
    title : str, optional
        Plot title. If None, generates default title
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates new figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object

    Examples
    --------
    >>> from mysuperlearner.visualization import plot_cv_forest
    >>> fig, ax = plot_cv_forest(cv_results, metric='auc')
    >>> plt.show()
    """
    # Calculate mean and 95% CI for each learner
    forest_data = []
    learners = cv_results['learner'].unique()

    for learner in learners:
        learner_data = cv_results[cv_results['learner'] == learner][metric]
        mean_val = learner_data.mean()
        std_val = learner_data.std()
        n = len(learner_data)
        ci = 1.96 * std_val / np.sqrt(n)  # 95% CI

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
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    colors = ['#d62728' if x else '#1f77b4' for x in forest_df['is_super']]
    y_pos = np.arange(len(forest_df))

    # Plot error bars and points
    for i, (idx, row) in enumerate(forest_df.iterrows()):
        ax.plot([row['ci_lower'], row['ci_upper']], [i, i],
               color=colors[i], linewidth=2, alpha=0.8)
        ax.scatter(row['mean'], i, s=200, color=colors[i],
                  zorder=3, edgecolors='black', linewidth=1.5,
                  marker='D' if row['is_super'] else 'o')

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(forest_df['learner'])
    ax.set_xlabel(f'{metric.upper()} (95% CI)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learner', fontsize=12, fontweight='bold')
    ax.set_title(title or f'Cross-Validated {metric.upper()}: Forest Plot',
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (idx, row) in enumerate(forest_df.iterrows()):
        ax.text(row['mean'], i + 0.25, f"{row['mean']:.4f}",
               ha='center', va='bottom', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', edgecolor='black', label='SuperLearner'),
        Patch(facecolor='#1f77b4', edgecolor='black', label='Base Learner')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    return fig, ax


def plot_cv_roc_curves(
    predictions: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot ROC curves from cross-validation predictions.

    Parameters
    ----------
    predictions : dict
        Dictionary from evaluate_super_learner_cv() with return_predictions=True.
        Must contain 'y_true' and arrays for each learner's predictions
    figsize : tuple, default=(10, 8)
        Figure size (width, height) in inches
    title : str, optional
        Plot title. If None, generates default title
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates new figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object

    Examples
    --------
    >>> from mysuperlearner.visualization import plot_cv_roc_curves
    >>> fig, ax = plot_cv_roc_curves(predictions)
    >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    y_true = predictions['y_true']

    # Plot SuperLearner first
    if 'SuperLearner' in predictions:
        y_pred = predictions['SuperLearner']
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=3,
               label=f'SuperLearner (AUC={roc_auc:.4f})',
               color='#d62728', linestyle='-', marker='o',
               markersize=4, markevery=max(1, len(fpr)//20))

    # Plot base learners
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    color_idx = 0

    for learner_name in sorted(predictions.keys()):
        if learner_name in ['y_true', 'fold_id', 'test_indices', 'SuperLearner']:
            continue

        y_pred = predictions[learner_name]
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2,
               label=f'{learner_name} (AUC={roc_auc:.4f})',
               color=colors[color_idx], linestyle='--', alpha=0.7,
               marker='s', markersize=3, markevery=max(1, len(fpr)//20))
        color_idx += 1

    # Diagonal reference
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5,
           label='Random (AUC=0.5000)')

    # Formatting
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(title or 'Cross-Validated ROC Curves',
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    return fig, ax


def plot_calibration_curves(
    predictions: Dict[str, np.ndarray],
    n_bins: int = 10,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot calibration curves for learners.

    Parameters
    ----------
    predictions : dict
        Dictionary from evaluate_super_learner_cv() with return_predictions=True.
        Must contain 'y_true' and arrays for each learner's predictions
    n_bins : int, default=10
        Number of bins for calibration
    figsize : tuple, default=(10, 8)
        Figure size (width, height) in inches
    title : str, optional
        Plot title. If None, generates default title
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates new figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object

    Examples
    --------
    >>> from mysuperlearner.visualization import plot_calibration_curves
    >>> fig, ax = plot_calibration_curves(predictions, n_bins=10)
    >>> plt.show()
    """
    if calibration_curve is None:
        raise ImportError(
            "calibration_curve not available. Please install scikit-learn >= 0.22 "
            "or use: pip install scikit-learn --upgrade"
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    y_true = predictions['y_true']

    # Plot perfect calibration
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration',
           linewidth=2, alpha=0.7)

    # Plot each learner
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    color_idx = 0

    for learner_name in sorted(predictions.keys()):
        if learner_name in ['y_true', 'fold_id', 'test_indices']:
            continue

        y_pred = predictions[learner_name]

        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred, n_bins=n_bins, strategy='uniform'
            )

            marker = 'D' if learner_name == 'SuperLearner' else 'o'
            linewidth = 3 if learner_name == 'SuperLearner' else 2
            color = '#d62728' if learner_name == 'SuperLearner' else colors[color_idx]

            ax.plot(mean_predicted_value, fraction_of_positives,
                   marker=marker, linewidth=linewidth, markersize=8,
                   label=learner_name, color=color, alpha=0.8)

            if learner_name != 'SuperLearner':
                color_idx += 1
        except Exception as e:
            # Skip learners that cause errors (e.g., constant predictions)
            continue

    # Formatting
    ax.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
    ax.set_title(title or 'Calibration Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    return fig, ax


def plot_learner_comparison(
    cv_results: pd.DataFrame,
    metrics: List[str] = ['auc', 'accuracy'],
    figsize: Optional[Tuple[int, int]] = None
) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
    """
    Create comparison plot across multiple metrics.

    Parameters
    ----------
    cv_results : pd.DataFrame
        Results from evaluate_super_learner_cv() containing columns:
        'learner' and the specified metrics
    metrics : list of str, default=['auc', 'accuracy']
        Metrics to compare
    figsize : tuple, optional
        Figure size (width, height) in inches.
        If None, auto-calculated based on number of metrics

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : matplotlib.axes.Axes or np.ndarray
        The axes object(s)

    Examples
    --------
    >>> from mysuperlearner.visualization import plot_learner_comparison
    >>> fig, axes = plot_learner_comparison(cv_results, metrics=['auc', 'accuracy'])
    >>> plt.show()
    """
    n_metrics = len(metrics)

    if figsize is None:
        figsize = (6 * n_metrics, 5)

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
        axes[i].set_xlabel(metric.upper(), fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Learner', fontsize=12, fontweight='bold')
        axes[i].set_title(f'{metric.upper()} Comparison',
                         fontsize=13, fontweight='bold')
        axes[i].grid(axis='x', alpha=0.3)

        # Add value labels
        for j, (learner, row) in enumerate(summary.iterrows()):
            axes[i].text(row['mean'], j, f" {row['mean']:.4f}",
                       va='center', ha='left', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='white', alpha=0.7))

    plt.tight_layout()
    return fig, axes if n_metrics > 1 else axes[0]


def plot_cv_boxplot(
    cv_results: pd.DataFrame,
    metric: str = 'auc',
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create box plot showing distribution of metric across CV folds.

    Parameters
    ----------
    cv_results : pd.DataFrame
        Results from evaluate_super_learner_cv() containing columns:
        'learner', 'learner_type', and the specified metric
    metric : str, default='auc'
        Metric to plot (e.g., 'auc', 'accuracy', 'logloss')
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches
    title : str, optional
        Plot title. If None, generates default title
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates new figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object

    Examples
    --------
    >>> from mysuperlearner.visualization import plot_cv_boxplot
    >>> fig, ax = plot_cv_boxplot(cv_results, metric='auc')
    >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Prepare data for boxplot
    learners = cv_results['learner'].unique()
    learner_order = (['SuperLearner'] if 'SuperLearner' in learners else []) + \
                    [l for l in learners if l != 'SuperLearner']

    box_data = []
    box_labels = []
    box_colors = []

    for learner in learner_order:
        learner_data = cv_results[cv_results['learner'] == learner][metric]
        box_data.append(learner_data)
        box_labels.append(learner)
        box_colors.append('#d62728' if learner == 'SuperLearner' else '#1f77b4')

    # Create boxplot
    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                   notch=True, showmeans=True,
                   meanprops=dict(marker='D', markerfacecolor='yellow',
                                markeredgecolor='black', markersize=8))

    # Color the boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Formatting
    ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
    ax.set_xlabel('Learner', fontsize=12, fontweight='bold')
    ax.set_title(title or f'{metric.upper()} Distribution Across CV Folds',
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Rotate labels if needed
    if len(learner_order) > 5:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    return fig, ax
