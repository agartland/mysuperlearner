"""
Variable importance methods for SuperLearner ensembles.

This module provides multiple methods for evaluating feature importance in fitted
SuperLearner models, including:
- Permutation Feature Importance (PFI) with re-training
- Drop-column importance
- Grouped PFI with hierarchical clustering
- SHAP (Shapley Additive Explanations)
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Union, Callable
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from joblib import Parallel, delayed

# Conditional SHAP import
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from .error_handling import ErrorType


def _has_shap() -> bool:
    """Check if shap package is available."""
    return HAS_SHAP


@dataclass
class VariableImportanceResults:
    """
    Container for variable importance results with analysis and plotting methods.

    Attributes
    ----------
    importance_df : pd.DataFrame
        Aggregated importance results with columns:
        - feature: str, feature name
        - method: str, importance method used
        - importance: float, mean importance score
        - importance_se: float, standard error
        - importance_ci_lower: float, 95% CI lower bound
        - importance_ci_upper: float, 95% CI upper bound
        - importance_normalized: float, normalized to [0, 1]
        - rank: int, rank by importance (1 = most important)

    raw_importance_df : pd.DataFrame
        Raw fold-level importance with columns:
        - feature: str
        - method: str
        - fold: int
        - repeat: int (for permutation-based methods, -1 for non-repeated)
        - importance: float, raw importance for this fold/repeat
        - baseline_score: float
        - modified_score: float

    cluster_info : pd.DataFrame, optional
        For grouped PFI, contains:
        - feature: str
        - group_id: int
        - group_size: int
        - features_in_group: str (comma-separated)

    shap_values : np.ndarray, optional
        For SHAP method: array of shape (n_samples, n_features)

    config : dict
        Configuration used: method, metric, n_repeats, etc.

    feature_names : list
        List of feature names
    """
    importance_df: pd.DataFrame
    raw_importance_df: pd.DataFrame
    cluster_info: Optional[pd.DataFrame] = None
    shap_values: Optional[np.ndarray] = None
    config: Optional[Dict] = None
    feature_names: Optional[List[str]] = None

    def summary(self, top_n: int = 10, method: Optional[str] = None) -> pd.DataFrame:
        """
        Return top N most important features.

        Parameters
        ----------
        top_n : int, default=10
            Number of top features to return
        method : str, optional
            Filter by specific importance method. If None, includes all methods.

        Returns
        -------
        summary_df : pd.DataFrame
            Top features with importance scores and confidence intervals
        """
        df = self.importance_df.copy()
        if method is not None:
            df = df[df['method'] == method]

        return df.nlargest(top_n, 'importance')

    def get_top_features(self, n: int = 10, method: Optional[str] = None) -> List[str]:
        """
        Return list of top N feature names.

        Parameters
        ----------
        n : int, default=10
            Number of top features
        method : str, optional
            Filter by method

        Returns
        -------
        features : list of str
            Top feature names
        """
        summary = self.summary(top_n=n, method=method)
        return summary['feature'].tolist()

    def compare_methods(self) -> pd.DataFrame:
        """
        Compare feature rankings across different importance methods.

        Returns
        -------
        comparison_df : pd.DataFrame
            Rows: features, Columns: method names with ranks
        """
        pivot = self.importance_df.pivot_table(
            index='feature',
            columns='method',
            values='rank'
        )
        # Sort by first method's rank
        if len(pivot.columns) > 0:
            pivot = pivot.sort_values(by=pivot.columns[0])
        return pivot

    # Visualization wrappers
    def plot_importance_bar(self, top_n: int = 20, method: Optional[str] = None,
                           **kwargs) -> Tuple:
        """
        Bar plot of feature importances with confidence intervals.

        Parameters
        ----------
        top_n : int, default=20
            Number of top features to display
        method : str, optional
            Filter by method
        **kwargs
            Additional arguments passed to plot function

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        from .visualization import plot_variable_importance_bar
        return plot_variable_importance_bar(self.importance_df, top_n=top_n,
                                           method=method, **kwargs)

    def plot_importance_heatmap(self, **kwargs) -> Tuple:
        """
        Heatmap comparing importance across methods.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to plot function

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        from .visualization import plot_variable_importance_heatmap
        return plot_variable_importance_heatmap(self.importance_df, **kwargs)

    def plot_grouped_clusters(self, **kwargs) -> Tuple:
        """
        Dendrogram showing feature groupings (for grouped PFI).

        Parameters
        ----------
        **kwargs
            Additional arguments passed to plot function

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes

        Raises
        ------
        ValueError
            If cluster_info is not available
        """
        if self.cluster_info is None:
            raise ValueError("No cluster information available. "
                           "Run with method='grouped' to generate clusters.")
        from .visualization import plot_feature_clusters
        return plot_feature_clusters(self.cluster_info, **kwargs)

    def plot_shap_summary(self, X: Optional[pd.DataFrame] = None, **kwargs) -> Tuple:
        """
        SHAP summary plot (beeswarm style).

        Parameters
        ----------
        X : pd.DataFrame, optional
            Feature matrix for detailed SHAP plots
        **kwargs
            Additional arguments passed to plot function

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes

        Raises
        ------
        ValueError
            If SHAP values are not available
        """
        if self.shap_values is None:
            raise ValueError("No SHAP values available. "
                           "Run with method='shap' to generate SHAP values.")
        from .visualization import plot_shap_summary
        return plot_shap_summary(self.shap_values, self.feature_names, X=X, **kwargs)

    def __repr__(self):
        n_features = len(self.feature_names) if self.feature_names else 0
        methods = self.importance_df['method'].unique() if 'method' in self.importance_df.columns else []
        return (f"VariableImportanceResults(n_features={n_features}, "
                f"methods={list(methods)})")


# ============================================================================
# Main API Function
# ============================================================================

def compute_variable_importance(
    sl,  # ExtendedSuperLearner
    method: Union[str, List[str]] = 'permutation',
    X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    y: Optional[np.ndarray] = None,
    metric: Union[str, Callable] = 'auc',
    n_repeats: int = 5,
    grouped_threshold: float = 0.7,
    random_state: Optional[int] = None,
    verbose: bool = False,
    n_jobs: int = 1
) -> VariableImportanceResults:
    """
    Compute variable importance for a fitted SuperLearner.

    Parameters
    ----------
    sl : ExtendedSuperLearner
        Fitted SuperLearner object
    method : str or list of str, default='permutation'
        Importance method(s): 'permutation', 'drop_column', 'grouped', 'shap'
    X : array-like or DataFrame, optional
        Feature matrix. If None, uses sl.X_ (must have store_X=True during fit)
    y : array-like, optional
        Target vector. If None, uses sl.y_
    metric : str or callable, default='auc'
        Performance metric. Options: 'auc', 'accuracy', 'logloss', or custom function
    n_repeats : int, default=5
        Number of permutation repeats for variance estimation
    grouped_threshold : float, default=0.7
        Correlation threshold for grouped PFI (hierarchical clustering)
    random_state : int, optional
        Random seed for reproducibility
    verbose : bool, default=False
        Print progress information
    n_jobs : int, default=1
        Number of parallel jobs for computation. If -1, uses all available CPUs.
        Parallelizes computation across features (permutation/drop_column methods)
        or groups (grouped method). Uses joblib with 'loky' backend.
        Note: SHAP method does not support parallelization via this parameter.

    Returns
    -------
    results : VariableImportanceResults
        Container with importance scores, statistics, and visualization methods

    Raises
    ------
    ValueError
        If SuperLearner not fitted or required data not available
    ImportError
        If 'shap' method requested but shap package not installed

    Examples
    --------
    >>> from mysuperlearner import ExtendedSuperLearner
    >>> from mysuperlearner.variable_importance import compute_variable_importance
    >>>
    >>> # Fit SuperLearner
    >>> sl = ExtendedSuperLearner(method='nnloglik', folds=5)
    >>> sl.fit_explicit(X_train, y_train, learners, store_X=True)
    >>>
    >>> # Compute permutation importance
    >>> results = compute_variable_importance(sl, method='permutation', n_repeats=10)
    >>> print(results.summary(top_n=10))
    >>>
    >>> # Visualize
    >>> fig, ax = results.plot_importance_bar(top_n=20)
    """
    # Validate SuperLearner is fitted
    if not hasattr(sl, 'base_learners_full_'):
        raise ValueError("SuperLearner not fitted. Call fit_explicit() first.")

    if not hasattr(sl, 'fold_indices_'):
        raise ValueError("SuperLearner missing fold_indices_. Cannot compute importance.")

    # Get X and y
    if X is None:
        if not hasattr(sl, 'X_'):
            raise ValueError(
                "X not provided and not stored in SuperLearner. "
                "Either provide X explicitly or fit with store_X=True."
            )
        X = sl.X_

    if y is None:
        if not hasattr(sl, 'y_'):
            raise ValueError("y not provided and not stored in SuperLearner.")
        y = sl.y_

    # Convert X to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        if hasattr(sl, 'feature_names_'):
            feature_names = sl.feature_names_
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)

    # Get feature names
    feature_names = X.columns.tolist()

    # Convert method to list
    if isinstance(method, str):
        methods = [method]
    else:
        methods = list(method)

    # Validate methods
    valid_methods = ['permutation', 'drop_column', 'grouped', 'shap']
    for m in methods:
        if m not in valid_methods:
            raise ValueError(f"Unknown method: {m}. Choose from {valid_methods}")

    # Check for SHAP
    if 'shap' in methods and not HAS_SHAP:
        raise ImportError(
            "SHAP method requested but shap package not installed. "
            "Install with: pip install shap"
        )

    # Get metric function
    metric_func = _get_metric_function(metric)

    # Store configuration
    config = {
        'methods': methods,
        'metric': metric if isinstance(metric, str) else 'custom',
        'n_repeats': n_repeats,
        'grouped_threshold': grouped_threshold,
        'random_state': random_state,
        'n_jobs': n_jobs
    }

    # Compute importance for each method
    all_raw_results = []
    all_aggregated_results = []
    cluster_info = None
    shap_values = None

    for method_name in methods:
        if verbose:
            print(f"\nComputing {method_name} importance...")

        if method_name == 'permutation':
            raw_df = _permutation_importance(
                sl, X, y, metric_func, n_repeats, random_state, verbose, n_jobs
            )
            agg_df = _aggregate_importance_results(raw_df, method_name)

        elif method_name == 'drop_column':
            raw_df = _drop_column_importance(
                sl, X, y, metric_func, verbose, n_jobs
            )
            agg_df = _aggregate_importance_results(raw_df, method_name)

        elif method_name == 'grouped':
            raw_df, cluster_df = _grouped_importance(
                sl, X, y, metric_func, grouped_threshold, n_repeats, random_state, verbose, n_jobs
            )
            agg_df = _aggregate_importance_results(raw_df, method_name)
            cluster_info = cluster_df

        elif method_name == 'shap':
            raw_df, shap_vals = _shap_importance(sl, X, verbose=verbose)
            agg_df = _aggregate_importance_results(raw_df, method_name)
            shap_values = shap_vals

        all_raw_results.append(raw_df)
        all_aggregated_results.append(agg_df)

    # Combine results from all methods
    importance_df = pd.concat(all_aggregated_results, ignore_index=True)
    raw_importance_df = pd.concat(all_raw_results, ignore_index=True)

    # Create result object
    results = VariableImportanceResults(
        importance_df=importance_df,
        raw_importance_df=raw_importance_df,
        cluster_info=cluster_info,
        shap_values=shap_values,
        config=config,
        feature_names=feature_names
    )

    if verbose:
        print("\nVariable importance computation complete!")
        print(f"Top 5 features by {methods[0]}:")
        print(results.summary(top_n=5, method=methods[0]))

    return results


# ============================================================================
# Helper Functions
# ============================================================================

def _get_metric_function(metric: Union[str, Callable]) -> Callable:
    """
    Convert metric string to callable function.

    Parameters
    ----------
    metric : str or callable
        Metric name or custom function

    Returns
    -------
    metric_func : callable
        Function that takes (y_true, y_pred) and returns score

    Raises
    ------
    ValueError
        If metric string not recognized
    """
    if callable(metric):
        return metric

    metric_map = {
        'auc': lambda y_true, y_pred: roc_auc_score(y_true, y_pred),
        'accuracy': lambda y_true, y_pred: accuracy_score(y_true, (y_pred > 0.5).astype(int)),
        'logloss': lambda y_true, y_pred: log_loss(y_true, y_pred)
    }

    if metric not in metric_map:
        raise ValueError(f"Unknown metric: {metric}. Choose from {list(metric_map.keys())} or provide callable.")

    return metric_map[metric]


def _aggregate_importance_results(
    raw_df: pd.DataFrame,
    method_name: str
) -> pd.DataFrame:
    """
    Aggregate importance across folds and repeats.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw importance with columns: feature, fold, repeat, importance
    method_name : str
        Name of the importance method

    Returns
    -------
    agg_df : pd.DataFrame
        Aggregated results with columns:
        - feature
        - method
        - importance (mean)
        - importance_se (standard error)
        - importance_ci_lower (95% CI)
        - importance_ci_upper (95% CI)
        - importance_normalized (scaled to [0, 1])
        - rank (1 = most important)
    """
    # Group by feature and aggregate
    agg = raw_df.groupby('feature')['importance'].agg(['mean', 'std', 'count'])

    # Calculate standard error
    agg['se'] = agg['std'] / np.sqrt(agg['count'])

    # Calculate 95% CI using t-distribution
    # For single observation (count=1), use large confidence interval
    t_values = np.where(
        agg['count'] > 1,
        stats.t.ppf(0.975, agg['count'] - 1),
        12.706  # t-value for df=1, alpha=0.05 (very wide CI)
    )
    agg['ci_lower'] = agg['mean'] - t_values * agg['se']
    agg['ci_upper'] = agg['mean'] + t_values * agg['se']

    # Normalize importance to [0, 1] range
    # Handle case where all importances are the same
    importance_range = agg['mean'].max() - agg['mean'].min()
    if importance_range > 0:
        agg['normalized'] = (agg['mean'] - agg['mean'].min()) / importance_range
    else:
        agg['normalized'] = 0.5  # All equal importance

    # Rank features (1 = most important)
    agg['rank'] = agg['mean'].rank(ascending=False, method='dense').astype(int)

    # Format output DataFrame
    result = agg.reset_index()
    result = result.rename(columns={
        'mean': 'importance',
        'se': 'importance_se',
        'ci_lower': 'importance_ci_lower',
        'ci_upper': 'importance_ci_upper',
        'normalized': 'importance_normalized'
    })
    result['method'] = method_name

    # Select and reorder columns
    output_cols = [
        'feature', 'method', 'importance', 'importance_se',
        'importance_ci_lower', 'importance_ci_upper',
        'importance_normalized', 'rank'
    ]

    return result[output_cols]


def _refit_with_modified_data(
    sl,  # ExtendedSuperLearner
    X_modified: pd.DataFrame,
    y: np.ndarray,
    fold_indices: List[Tuple[np.ndarray, np.ndarray]],
    metric_func: Callable,
    verbose: bool = False
) -> Tuple[float, List[float]]:
    """
    Re-train SuperLearner with modified data using existing folds.

    Parameters
    ----------
    sl : ExtendedSuperLearner
        Original fitted SuperLearner
    X_modified : pd.DataFrame
        Modified feature matrix
    y : np.ndarray
        Target vector
    fold_indices : list of tuples
        CV fold indices from original fit
    metric_func : callable
        Metric function for evaluation
    verbose : bool, default=False
        Print progress

    Returns
    -------
    mean_score : float
        Average metric across folds
    fold_scores : list of float
        Per-fold scores
    """
    from .super_learner import SuperLearner

    X_arr = X_modified.values if isinstance(X_modified, pd.DataFrame) else X_modified
    fold_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(fold_indices):
        if verbose and fold_idx % max(1, len(fold_indices) // 5) == 0:
            print(f"  Processing fold {fold_idx + 1}/{len(fold_indices)}")

        # Extract fold data
        X_train_fold = X_arr[train_idx]
        y_train_fold = y[train_idx]
        X_test_fold = X_arr[test_idx]
        y_test_fold = y[test_idx]

        # Train base learners on this fold
        predictions = []
        for name, base_estimator in sl.base_learners_full_:
            try:
                # Clone and fit the base learner
                mdl = clone(base_estimator)
                mdl.fit(X_train_fold, y_train_fold)

                # Get predictions
                if hasattr(mdl, 'predict_proba'):
                    pred_proba = mdl.predict_proba(X_test_fold)
                    # Handle different output formats
                    if pred_proba.ndim == 2 and pred_proba.shape[1] == 2:
                        pred = pred_proba[:, 1]  # Probability of class 1
                    elif pred_proba.ndim == 2 and pred_proba.shape[1] == 1:
                        pred = pred_proba.ravel()
                    else:
                        pred = pred_proba  # Assume already correct format
                else:
                    # Fallback for classifiers without predict_proba
                    pred = mdl.predict(X_test_fold).astype(float)

                predictions.append(pred)

            except Exception as e:
                # On error, use neutral prediction (0.5)
                if verbose:
                    warnings.warn(f"Learner {name} failed in fold {fold_idx}: {str(e)}")
                predictions.append(np.full(len(test_idx), 0.5))

        # Stack predictions and compute ensemble prediction
        if len(predictions) == 0:
            # No successful predictions - use neutral
            ensemble_pred = np.full(len(test_idx), 0.5)
        else:
            Z_fold = np.column_stack(predictions)

            # Use meta-learner weights if available
            if sl.meta_weights_ is not None:
                # Weighted average using meta weights
                ensemble_pred = np.average(Z_fold, weights=sl.meta_weights_, axis=1)
            elif sl.meta_learner_ is not None and hasattr(sl.meta_learner_, 'predict_proba'):
                # Use meta-learner to make predictions
                try:
                    meta_pred = sl.meta_learner_.predict_proba(Z_fold)
                    ensemble_pred = meta_pred[:, 1] if meta_pred.ndim == 2 else meta_pred
                except:
                    # Fallback to simple average
                    ensemble_pred = Z_fold.mean(axis=1)
            else:
                # Simple average
                ensemble_pred = Z_fold.mean(axis=1)

        # Clip predictions for numerical stability
        ensemble_pred = np.clip(ensemble_pred, 0.001, 0.999)

        # Evaluate performance
        try:
            score = metric_func(y_test_fold, ensemble_pred)
            fold_scores.append(score)
        except Exception as e:
            if verbose:
                warnings.warn(f"Metric computation failed in fold {fold_idx}: {str(e)}")
            # Use worst possible score for this fold
            fold_scores.append(0.0)

    mean_score = np.mean(fold_scores) if len(fold_scores) > 0 else 0.0
    return mean_score, fold_scores


# ============================================================================
# Importance Method Implementations
# ============================================================================

def _compute_single_feature_permutation(
    feat_idx: int,
    feature_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    sl,  # ExtendedSuperLearner
    n_repeats: int,
    baseline_fold_scores: List[float],
    metric_func: Callable,
    random_state: Optional[int]
) -> List[Dict]:
    """
    Compute permutation importance for a single feature.

    This helper function is designed for parallel execution via joblib.
    Each call processes one feature independently across all repeats and folds.

    Parameters
    ----------
    feat_idx : int
        Index of the feature to permute
    feature_name : str
        Name of the feature
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Target vector
    sl : ExtendedSuperLearner
        Fitted SuperLearner object
    n_repeats : int
        Number of permutation repeats
    baseline_fold_scores : list of float
        Baseline scores for each fold
    metric_func : callable
        Metric function for evaluation
    random_state : int, optional
        Random seed (offset by feat_idx for reproducibility)

    Returns
    -------
    results_list : list of dict
        Raw importance results for this feature across all folds and repeats
    """
    # Initialize random number generator with feature-specific offset
    rng = np.random.RandomState(random_state + feat_idx if random_state is not None else None)
    results_list = []

    for repeat_idx in range(n_repeats):
        # Create copy of X and permute this feature
        X_permuted = X.copy()

        # Permute the feature column
        if isinstance(X_permuted, pd.DataFrame):
            X_permuted.iloc[:, feat_idx] = rng.permutation(X_permuted.iloc[:, feat_idx].values)
        else:
            X_permuted[:, feat_idx] = rng.permutation(X_permuted[:, feat_idx])

        # Re-train with permuted data
        _, permuted_fold_scores = _refit_with_modified_data(
            sl, X_permuted, y, sl.fold_indices_, metric_func, verbose=False
        )

        # Compute importance for each fold
        for fold_idx, (baseline_fold, permuted_fold) in enumerate(
            zip(baseline_fold_scores, permuted_fold_scores)
        ):
            importance = baseline_fold - permuted_fold

            results_list.append({
                'feature': feature_name,
                'fold': fold_idx,
                'repeat': repeat_idx,
                'importance': importance,
                'baseline_score': baseline_fold,
                'modified_score': permuted_fold
            })

    return results_list


def _permutation_importance(
    sl,  # ExtendedSuperLearner
    X: pd.DataFrame,
    y: np.ndarray,
    metric_func: Callable,
    n_repeats: int,
    random_state: Optional[int],
    verbose: bool,
    n_jobs: int = 1
) -> pd.DataFrame:
    """
    Permutation Feature Importance with re-training.

    Algorithm:
    1. Compute baseline using sl.fold_indices_ for CV consistency
    2. For each feature (parallelized if n_jobs > 1):
       a. For each fold in sl.fold_indices_:
          - For each repeat:
            - Permute feature values within training set
            - Re-train all base learners on permuted data
            - Re-fit meta-learner on CV predictions
            - Predict on test fold
            - Compute metric
            - Importance = baseline - permuted_score
    3. Return raw DataFrame: [feature, fold, repeat, importance, baseline, permuted]

    Parameters
    ----------
    sl : ExtendedSuperLearner
        Fitted SuperLearner object
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Target vector
    metric_func : callable
        Metric function for evaluation
    n_repeats : int
        Number of permutation repeats
    random_state : int, optional
        Random seed for reproducibility
    verbose : bool
        Print progress information
    n_jobs : int, default=1
        Number of parallel jobs. If -1, uses all CPUs.
        Parallelizes computation across features.

    Returns
    -------
    results_df : pd.DataFrame
        Raw fold-level importance results
    """
    if verbose:
        print("Computing permutation importance...")

    # Compute baseline performance
    if verbose:
        print("  Computing baseline performance...")
    baseline_score, baseline_fold_scores = _refit_with_modified_data(
        sl, X, y, sl.fold_indices_, metric_func, verbose=False
    )

    # Get feature names
    feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]

    # Parallel or sequential computation
    if n_jobs == 1:
        # Sequential execution (original behavior)
        all_results = []
        for feat_idx, feature_name in enumerate(feature_names):
            if verbose:
                print(f"  Feature {feat_idx + 1}/{len(feature_names)}: {feature_name}")
            results = _compute_single_feature_permutation(
                feat_idx, feature_name, X, y, sl, n_repeats,
                baseline_fold_scores, metric_func, random_state
            )
            all_results.extend(results)
    else:
        # Parallel execution
        if verbose:
            print(f"  Computing importance for {len(feature_names)} features in parallel (n_jobs={n_jobs})...")

        all_results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(_compute_single_feature_permutation)(
                feat_idx, feature_name, X, y, sl, n_repeats,
                baseline_fold_scores, metric_func, random_state
            )
            for feat_idx, feature_name in enumerate(feature_names)
        )
        # Flatten results (list of lists -> single list)
        all_results = [item for sublist in all_results for item in sublist]

    results_df = pd.DataFrame(all_results)
    return results_df


def _compute_single_feature_drop_column(
    feat_idx: int,
    feature_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    sl,  # ExtendedSuperLearner
    baseline_fold_scores: List[float],
    metric_func: Callable
) -> List[Dict]:
    """
    Compute drop-column importance for a single feature.

    This helper function is designed for parallel execution via joblib.
    Each call processes one feature independently across all folds.

    Parameters
    ----------
    feat_idx : int
        Index of the feature to drop
    feature_name : str
        Name of the feature
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Target vector
    sl : ExtendedSuperLearner
        Fitted SuperLearner object
    baseline_fold_scores : list of float
        Baseline scores for each fold
    metric_func : callable
        Metric function for evaluation

    Returns
    -------
    results_list : list of dict
        Raw importance results for this feature across all folds
    """
    # Create X with feature removed
    if isinstance(X, pd.DataFrame):
        X_reduced = X.drop(columns=[feature_name])
    else:
        # For numpy arrays, remove the column
        cols_to_keep = [i for i in range(X.shape[1]) if i != feat_idx]
        X_reduced = X[:, cols_to_keep]

    # Re-train with reduced data
    _, reduced_fold_scores = _refit_with_modified_data(
        sl, X_reduced, y, sl.fold_indices_, metric_func, verbose=False
    )

    # Compute importance for each fold
    results_list = []
    for fold_idx, (baseline_fold, reduced_fold) in enumerate(
        zip(baseline_fold_scores, reduced_fold_scores)
    ):
        importance = baseline_fold - reduced_fold

        results_list.append({
            'feature': feature_name,
            'fold': fold_idx,
            'repeat': -1,  # No repeats for drop-column
            'importance': importance,
            'baseline_score': baseline_fold,
            'modified_score': reduced_fold
        })

    return results_list


def _drop_column_importance(
    sl,  # ExtendedSuperLearner
    X: pd.DataFrame,
    y: np.ndarray,
    metric_func: Callable,
    verbose: bool,
    n_jobs: int = 1
) -> pd.DataFrame:
    """
    Drop-column importance (remove feature entirely and re-train).

    Parameters
    ----------
    sl : ExtendedSuperLearner
        Fitted SuperLearner object
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Target vector
    metric_func : callable
        Metric function for evaluation
    verbose : bool
        Print progress information
    n_jobs : int, default=1
        Number of parallel jobs. If -1, uses all CPUs.
        Parallelizes computation across features.

    Returns
    -------
    results_df : pd.DataFrame
        Raw fold-level importance results
    """
    if verbose:
        print("Computing drop-column importance...")

    # Compute baseline performance
    if verbose:
        print("  Computing baseline performance...")
    baseline_score, baseline_fold_scores = _refit_with_modified_data(
        sl, X, y, sl.fold_indices_, metric_func, verbose=False
    )

    # Get feature names
    feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]

    # Parallel or sequential computation
    if n_jobs == 1:
        # Sequential execution (original behavior)
        all_results = []
        for feat_idx, feature_name in enumerate(feature_names):
            if verbose:
                print(f"  Feature {feat_idx + 1}/{len(feature_names)}: {feature_name}")
            results = _compute_single_feature_drop_column(
                feat_idx, feature_name, X, y, sl, baseline_fold_scores, metric_func
            )
            all_results.extend(results)
    else:
        # Parallel execution
        if verbose:
            print(f"  Computing importance for {len(feature_names)} features in parallel (n_jobs={n_jobs})...")

        all_results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(_compute_single_feature_drop_column)(
                feat_idx, feature_name, X, y, sl, baseline_fold_scores, metric_func
            )
            for feat_idx, feature_name in enumerate(feature_names)
        )
        # Flatten results (list of lists -> single list)
        all_results = [item for sublist in all_results for item in sublist]

    results_df = pd.DataFrame(all_results)
    return results_df


def _compute_single_group_permutation(
    group_id: int,
    group_features: List[str],
    X: pd.DataFrame,
    y: np.ndarray,
    sl,  # ExtendedSuperLearner
    n_repeats: int,
    baseline_fold_scores: List[float],
    metric_func: Callable,
    random_state: Optional[int]
) -> List[Dict]:
    """
    Compute permutation importance for a single feature group.

    This helper function is designed for parallel execution via joblib.
    Each call processes one feature group independently across all repeats and folds.

    Parameters
    ----------
    group_id : int
        ID of the feature group
    group_features : list of str
        List of feature names in this group
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Target vector
    sl : ExtendedSuperLearner
        Fitted SuperLearner object
    n_repeats : int
        Number of permutation repeats
    baseline_fold_scores : list of float
        Baseline scores for each fold
    metric_func : callable
        Metric function for evaluation
    random_state : int, optional
        Random seed (offset by group_id for reproducibility)

    Returns
    -------
    results_list : list of dict
        Raw importance results for this group across all folds and repeats
    """
    # Initialize random number generator with group-specific offset
    rng = np.random.RandomState(random_state + group_id if random_state is not None else None)
    results_list = []
    group_name = f"group_{group_id}"

    for repeat_idx in range(n_repeats):
        # Create copy of X and permute all features in this group
        X_permuted = X.copy()

        for feature in group_features:
            X_permuted[feature] = rng.permutation(X_permuted[feature].values)

        # Re-train with permuted data
        _, permuted_fold_scores = _refit_with_modified_data(
            sl, X_permuted, y, sl.fold_indices_, metric_func, verbose=False
        )

        # Compute importance for each fold
        for fold_idx, (baseline_fold, permuted_fold) in enumerate(
            zip(baseline_fold_scores, permuted_fold_scores)
        ):
            importance = baseline_fold - permuted_fold

            results_list.append({
                'feature': group_name,
                'group_id': group_id,
                'features_in_group': ','.join(group_features),
                'fold': fold_idx,
                'repeat': repeat_idx,
                'importance': importance,
                'baseline_score': baseline_fold,
                'modified_score': permuted_fold
            })

    return results_list


def _grouped_importance(
    sl,  # ExtendedSuperLearner
    X: pd.DataFrame,
    y: np.ndarray,
    metric_func: Callable,
    correlation_threshold: float,
    n_repeats: int,
    random_state: Optional[int],
    verbose: bool,
    n_jobs: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Grouped Permutation Feature Importance using hierarchical clustering.

    Parameters
    ----------
    sl : ExtendedSuperLearner
        Fitted SuperLearner object
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Target vector
    metric_func : callable
        Metric function for evaluation
    correlation_threshold : float
        Correlation threshold for grouping features
    n_repeats : int
        Number of permutation repeats
    random_state : int, optional
        Random seed for reproducibility
    verbose : bool
        Print progress information
    n_jobs : int, default=1
        Number of parallel jobs. If -1, uses all CPUs.
        Parallelizes computation across feature groups.

    Returns
    -------
    importance_df : pd.DataFrame
        Importance for each group
    cluster_df : pd.DataFrame
        Cluster assignments for features
    """
    if verbose:
        print("Computing grouped permutation importance...")
        print(f"  Correlation threshold: {correlation_threshold}")

    # Compute correlation matrix
    corr_matrix = X.corr().abs()

    # Convert correlation to distance: distance = 1 - |correlation|
    distance_matrix = 1 - corr_matrix

    # Perform hierarchical clustering
    # Convert to condensed distance matrix (required by linkage)
    condensed_dist = squareform(distance_matrix.values, checks=False)

    # Perform hierarchical clustering using average linkage
    linkage_matrix = hierarchy.linkage(condensed_dist, method='average')

    # Cut the dendrogram at the specified threshold
    distance_threshold = 1 - correlation_threshold
    cluster_labels = hierarchy.fcluster(linkage_matrix, distance_threshold, criterion='distance')

    # Create cluster info DataFrame
    feature_names = X.columns.tolist()
    cluster_df_list = []

    for feat_idx, (feature_name, cluster_id) in enumerate(zip(feature_names, cluster_labels)):
        # Get all features in this cluster
        features_in_cluster = [f for f, c in zip(feature_names, cluster_labels) if c == cluster_id]

        cluster_df_list.append({
            'feature': feature_name,
            'group_id': int(cluster_id),
            'group_size': len(features_in_cluster),
            'features_in_group': ','.join(features_in_cluster)
        })

    cluster_df = pd.DataFrame(cluster_df_list)

    if verbose:
        n_groups = len(set(cluster_labels))
        print(f"  Found {n_groups} feature groups")

    # Compute baseline performance
    if verbose:
        print("  Computing baseline performance...")
    baseline_score, baseline_fold_scores = _refit_with_modified_data(
        sl, X, y, sl.fold_indices_, metric_func, verbose=False
    )

    # Prepare groups for parallel/sequential computation
    unique_groups = sorted(set(cluster_labels))
    group_features_list = [
        [f for f, c in zip(feature_names, cluster_labels) if c == group_id]
        for group_id in unique_groups
    ]

    # Parallel or sequential computation
    if n_jobs == 1:
        # Sequential execution (original behavior)
        all_results = []
        for group_id, group_features in zip(unique_groups, group_features_list):
            if verbose:
                print(f"  Group {group_id} ({len(group_features)} features): {', '.join(group_features[:3])}{'...' if len(group_features) > 3 else ''}")
            results = _compute_single_group_permutation(
                group_id, group_features, X, y, sl, n_repeats,
                baseline_fold_scores, metric_func, random_state
            )
            all_results.extend(results)
    else:
        # Parallel execution
        if verbose:
            print(f"  Computing importance for {len(unique_groups)} groups in parallel (n_jobs={n_jobs})...")

        all_results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(_compute_single_group_permutation)(
                group_id, group_features, X, y, sl, n_repeats,
                baseline_fold_scores, metric_func, random_state
            )
            for group_id, group_features in zip(unique_groups, group_features_list)
        )
        # Flatten results (list of lists -> single list)
        all_results = [item for sublist in all_results for item in sublist]

    results_df = pd.DataFrame(all_results)
    return results_df, cluster_df


def _shap_importance(
    sl,  # ExtendedSuperLearner
    X: pd.DataFrame,
    background_samples: int = 100,
    verbose: bool = False
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    SHAP (Shapley Additive Explanations) importance using shap package.

    Returns
    -------
    importance_df : pd.DataFrame
        Aggregated SHAP importance (formatted to match other methods)
    shap_values : np.ndarray
        Raw SHAP values (n_samples, n_features)
    """
    if not HAS_SHAP:
        raise ImportError("shap package required for SHAP importance. Install with: pip install shap")

    if verbose:
        print("Computing SHAP importance...")

    # Create prediction function for SHAP
    def predict_fn(X_input):
        """Wrapper function that returns class 1 probabilities."""
        proba = sl.predict_proba(X_input)
        return proba[:, 1] if proba.ndim == 2 else proba

    # Sample background data if X is large
    if len(X) > background_samples:
        if verbose:
            print(f"  Subsampling {background_samples} background samples from {len(X)} samples")
        background_idx = np.random.choice(len(X), size=background_samples, replace=False)
        background = X.iloc[background_idx]
    else:
        background = X

    # Create SHAP explainer
    if verbose:
        print("  Initializing SHAP explainer...")

    try:
        # Try to use Explainer (auto-selects best method)
        explainer = shap.Explainer(predict_fn, background)
    except:
        # Fallback to KernelExplainer
        if verbose:
            print("  Using KernelExplainer (this may be slow)...")
        explainer = shap.KernelExplainer(predict_fn, background)

    # Compute SHAP values
    if verbose:
        print(f"  Computing SHAP values for {len(X)} samples...")

    shap_values = explainer.shap_values(X)

    # If shap_values is a list (multi-class), take values for class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Compute mean absolute SHAP value per feature
    feature_names = X.columns.tolist()
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Create DataFrame in format compatible with aggregation function
    # We'll create fake fold/repeat structure since SHAP doesn't use CV
    results_list = []
    for feat_idx, feature_name in enumerate(feature_names):
        results_list.append({
            'feature': feature_name,
            'fold': 0,  # Dummy fold
            'repeat': 0,  # Dummy repeat
            'importance': mean_abs_shap[feat_idx],
            'baseline_score': 0.0,  # Not applicable for SHAP
            'modified_score': 0.0   # Not applicable for SHAP
        })

    results_df = pd.DataFrame(results_list)

    if verbose:
        print("  SHAP computation complete!")

    return results_df, shap_values
