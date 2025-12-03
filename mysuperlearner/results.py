"""
Result classes for SuperLearner evaluation.

This module provides container classes for SuperLearner cross-validation results
with convenient methods for analysis and visualization.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class SuperLearnerCVResults:
    """
    Container for cross-validation results with built-in analysis and plotting methods.

    This class wraps the metrics DataFrame and optional predictions dictionary
    returned by evaluate_super_learner_cv(), providing convenient methods for
    analysis and visualization.

    Attributes
    ----------
    metrics : pd.DataFrame
        Per-fold metrics for each learner. Contains columns:
        - 'fold': fold number
        - 'learner': learner name
        - 'learner_type': 'super' for SuperLearner, 'base' for base learners, 'discrete' for discrete SL
        - metric columns (e.g., 'auc', 'accuracy', 'logloss')
    predictions : dict, optional
        Dictionary of predictions (only if return_predictions=True).
        Contains:
        - 'y_true': array of true labels (concatenated across folds)
        - 'fold_id': array indicating fold membership
        - 'test_indices': list of test indices per fold
        - '<learner_name>': predicted probabilities for each learner
    config : dict, optional
        Configuration used for cross-validation (outer_folds, method, etc.)
    coef : pd.DataFrame, optional
        Meta-learner coefficients per fold. Contains columns:
        - 'fold': fold number
        - 'learner': learner name
        - 'coefficient': meta-learner weight
    cv_risk : pd.DataFrame, optional
        Inner CV risk for each base learner per fold. Contains columns:
        - 'fold': fold number
        - 'learner': learner name
        - 'cv_risk': cross-validation risk (MSE)
    which_discrete_sl : list, optional
        List of selected discrete SuperLearner (best base learner) per fold
    timing : pd.DataFrame, optional
        Timing information for learner fitting (only if verbose=True). Contains columns:
        - 'outer_fold': outer fold number
        - 'learner': learner name
        - 'inner_fold': inner fold index
        - 'fit_time': time to fit in seconds
        - 'error': error message if fit failed (None otherwise)

    Examples
    --------
    >>> from mysuperlearner import ExtendedSuperLearner
    >>> from mysuperlearner.evaluation import evaluate_super_learner_cv
    >>> sl = ExtendedSuperLearner(method='nnloglik', folds=5)
    >>> results = evaluate_super_learner_cv(X, y, learners, sl,
    ...                                      return_predictions=True,
    ...                                      return_object=True)
    >>> print(results.summary())
    >>> fig, ax = results.plot_forest(metric='auc')
    >>> fig, ax = results.plot_roc_curves()
    """
    metrics: pd.DataFrame
    predictions: Optional[Dict[str, np.ndarray]] = None
    config: Optional[Dict] = None
    coef: Optional[pd.DataFrame] = None
    cv_risk: Optional[pd.DataFrame] = None
    which_discrete_sl: Optional[list] = None
    timing: Optional[pd.DataFrame] = None

    def summary(self, metrics: Optional[list] = None) -> pd.DataFrame:
        """
        Return summary statistics for each learner.

        Parameters
        ----------
        metrics : list of str, optional
            Metrics to summarize. If None, uses all numeric columns

        Returns
        -------
        summary : pd.DataFrame
            Multi-index DataFrame with (metric, statistic) columns.
            Statistics include: mean, std, count, ci_lower, ci_upper
        """
        if metrics is None:
            metrics = [col for col in self.metrics.columns
                      if col not in ['fold', 'learner', 'learner_type']]

        summary = self.metrics.groupby('learner')[metrics].agg(['mean', 'std', 'count'])

        # Add 95% confidence intervals
        for metric in metrics:
            if (metric, 'mean') in summary.columns:
                summary[(metric, 'ci_lower')] = (
                    summary[(metric, 'mean')] -
                    1.96 * summary[(metric, 'std')] / np.sqrt(summary[(metric, 'count')])
                )
                summary[(metric, 'ci_upper')] = (
                    summary[(metric, 'mean')] +
                    1.96 * summary[(metric, 'std')] / np.sqrt(summary[(metric, 'count')])
                )

        return summary

    def compare_to_best(self) -> pd.DataFrame:
        """
        Compare SuperLearner to best individual base learner.

        For each metric, compares SuperLearner performance to the best-performing base learner.
        'Best' is defined appropriately for each metric type:
        - For loss metrics (logloss, MSE, etc.): lowest value is best
        - For score metrics (AUC, accuracy, etc.): highest value is best

        Improvement is calculated so that positive values always indicate SuperLearner
        is better than the best base learner, regardless of metric type.

        Returns
        -------
        comparison : pd.DataFrame
            Comparison table with columns:
            - metric: metric name
            - SuperLearner: SuperLearner mean performance
            - Best_Base: best base learner mean performance
            - Best_Base_Name: name of best base learner
            - Improvement: absolute improvement (positive = SL better)
            - Improvement_Pct: percentage improvement (positive = SL better)
        """
        # Define metrics where lower is better
        LOWER_IS_BETTER = {'logloss', 'log_loss', 'mse', 'mae', 'rmse', 'cv_risk', 'brier'}

        sl_metrics = self.metrics[self.metrics['learner'] == 'SuperLearner']
        base_metrics = self.metrics[self.metrics['learner_type'] == 'base']

        comparison = []
        for metric in sl_metrics.columns:
            if metric in ['fold', 'learner', 'learner_type']:
                continue

            sl_mean = sl_metrics[metric].mean()

            # Normalize metric name for comparison (lowercase, no underscores/spaces)
            metric_normalized = metric.lower().replace('_', '').replace(' ', '')

            # Determine if this is a loss metric (lower is better)
            is_loss_metric = any(loss_name in metric_normalized for loss_name in LOWER_IS_BETTER)

            if is_loss_metric:
                # Lower is better - find minimum
                best_base_mean = base_metrics.groupby('learner')[metric].mean().min()
                best_name = base_metrics.groupby('learner')[metric].mean().idxmin()
                # For loss metrics: improvement = best_base - sl (positive when SL is lower/better)
                improvement = best_base_mean - sl_mean
            else:
                # Higher is better - find maximum
                best_base_mean = base_metrics.groupby('learner')[metric].mean().max()
                best_name = base_metrics.groupby('learner')[metric].mean().idxmax()
                # For score metrics: improvement = sl - best_base (positive when SL is higher/better)
                improvement = sl_mean - best_base_mean

            comparison.append({
                'metric': metric,
                'SuperLearner': sl_mean,
                'Best_Base': best_base_mean,
                'Best_Base_Name': best_name,
                'Improvement': improvement,
                'Improvement_Pct': (improvement / abs(best_base_mean) * 100)
                                  if best_base_mean != 0 else np.nan
            })

        return pd.DataFrame(comparison)

    def plot_forest(self, metric: str = 'auc', **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create forest plot for specified metric.

        This is a convenience wrapper around visualization.plot_cv_forest().

        Parameters
        ----------
        metric : str, default='auc'
            Metric to plot
        **kwargs
            Additional arguments passed to plot_cv_forest()

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib.axes.Axes
            The axes object
        """
        from .visualization import plot_cv_forest
        return plot_cv_forest(self.metrics, metric=metric, **kwargs)

    def plot_roc_curves(self, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create ROC curve plot.

        This is a convenience wrapper around visualization.plot_cv_roc_curves().
        Requires predictions to be available.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to plot_cv_roc_curves()

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib.axes.Axes
            The axes object

        Raises
        ------
        ValueError
            If predictions were not stored (return_predictions=False)
        """
        if self.predictions is None:
            raise ValueError(
                "Predictions not available. Set return_predictions=True "
                "when calling evaluate_super_learner_cv()"
            )
        from .visualization import plot_cv_roc_curves
        return plot_cv_roc_curves(self.predictions, **kwargs)

    def plot_calibration(self, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create calibration plot.

        This is a convenience wrapper around visualization.plot_calibration_curves().
        Requires predictions to be available.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to plot_calibration_curves()

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib.axes.Axes
            The axes object

        Raises
        ------
        ValueError
            If predictions were not stored (return_predictions=False)
        """
        if self.predictions is None:
            raise ValueError(
                "Predictions not available. Set return_predictions=True "
                "when calling evaluate_super_learner_cv()"
            )
        from .visualization import plot_calibration_curves
        return plot_calibration_curves(self.predictions, **kwargs)

    def plot_boxplot(self, metric: str = 'auc', **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create box plot showing distribution of metric across CV folds.

        This is a convenience wrapper around visualization.plot_cv_boxplot().

        Parameters
        ----------
        metric : str, default='auc'
            Metric to plot
        **kwargs
            Additional arguments passed to plot_cv_boxplot()

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib.axes.Axes
            The axes object
        """
        from .visualization import plot_cv_boxplot
        return plot_cv_boxplot(self.metrics, metric=metric, **kwargs)

    def plot_comparison(self, metrics: list = ['auc', 'accuracy'], **kwargs):
        """
        Create comparison plot across multiple metrics.

        This is a convenience wrapper around visualization.plot_learner_comparison().

        Parameters
        ----------
        metrics : list of str, default=['auc', 'accuracy']
            Metrics to compare
        **kwargs
            Additional arguments passed to plot_learner_comparison()

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        axes : matplotlib.axes.Axes or np.ndarray
            The axes object(s)
        """
        from .visualization import plot_learner_comparison
        return plot_learner_comparison(self.metrics, metrics=metrics, **kwargs)

    def get_timing_summary(self) -> pd.DataFrame:
        """
        Return summary of learner timings.

        Returns
        -------
        summary : pd.DataFrame
            Aggregated timing statistics per learner. Contains columns:
            - 'n_fits': number of fits
            - 'total_time': total time across all fits (seconds)
            - 'mean_time': mean time per fit (seconds)
            - 'std_time': standard deviation of fit times
            - 'min_time': minimum fit time
            - 'max_time': maximum fit time

        Raises
        ------
        ValueError
            If timing data not available (verbose=False during fit)

        Examples
        --------
        >>> results = evaluate_super_learner_cv(X, y, learners, sl,
        ...                                      verbose=True, return_object=True)
        >>> timing_summary = results.get_timing_summary()
        >>> print(timing_summary.sort_values('total_time', ascending=False))
        """
        if self.timing is None:
            raise ValueError("Timing data not available. Set verbose=True during fit.")

        summary = self.timing.groupby('learner')['fit_time'].agg([
            'count', 'sum', 'mean', 'std', 'min', 'max'
        ])
        summary.columns = ['n_fits', 'total_time', 'mean_time', 'std_time', 'min_time', 'max_time']
        summary = summary.sort_values('total_time', ascending=False)
        return summary

    def plot_timing(self, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create visualization of learner timings.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to matplotlib plotting functions

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib.axes.Axes
            The axes object

        Raises
        ------
        ValueError
            If timing data not available (verbose=False during fit)

        Examples
        --------
        >>> results = evaluate_super_learner_cv(X, y, learners, sl,
        ...                                      verbose=True, return_object=True)
        >>> fig, ax = results.plot_timing()
        >>> plt.show()
        """
        if self.timing is None:
            raise ValueError("Timing data not available. Set verbose=True during fit.")

        summary = self.get_timing_summary()

        fig, ax = plt.subplots(figsize=kwargs.pop('figsize', (10, 6)))
        summary['total_time'].plot(kind='barh', ax=ax, **kwargs)
        ax.set_xlabel('Total Time (seconds)')
        ax.set_ylabel('Learner')
        ax.set_title('Learner Timing Summary')
        plt.tight_layout()

        return fig, ax

    def __repr__(self):
        n_folds = self.metrics['fold'].nunique() if 'fold' in self.metrics.columns else 0
        n_learners = self.metrics['learner'].nunique() if 'learner' in self.metrics.columns else 0
        has_preds = self.predictions is not None
        return (f"SuperLearnerCVResults(n_folds={n_folds}, n_learners={n_learners}, "
                f"has_predictions={has_preds})")

    def __str__(self):
        summary = self.summary()
        return f"{self.__repr__()}\n\nSummary Statistics:\n{summary}"
