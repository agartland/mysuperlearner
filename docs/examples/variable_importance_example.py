"""
Example script demonstrating variable importance functionality in mysuperlearner.

This example shows how to:
1. Fit a SuperLearner model
2. Compute variable importance using multiple methods
3. Visualize and analyze the results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from mysuperlearner import ExtendedSuperLearner, compute_variable_importance

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# 1. Create synthetic data with known feature importance
# ============================================================================

print("Creating synthetic dataset...")
n_samples = 200
n_features = 10

# Create features with varying importance
X_data = {}
for i in range(n_features):
    X_data[f'feature_{i}'] = np.random.randn(n_samples)

X = pd.DataFrame(X_data)

# Create target with known feature importance:
# feature_0 and feature_1 are highly important
# feature_2 and feature_3 are moderately important
# remaining features are noise
y = (
    2.0 * X['feature_0'] +
    1.5 * X['feature_1'] +
    0.5 * X['feature_2'] +
    0.3 * X['feature_3'] +
    np.random.randn(n_samples) > 0
).astype(int)

print(f"Dataset shape: {X.shape}")
print(f"Target distribution: {np.bincount(y)}")

# ============================================================================
# 2. Create base learners
# ============================================================================

learners = [
    ('logistic', LogisticRegression(max_iter=500)),
    ('rf', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
    ('tree', DecisionTreeClassifier(max_depth=3, random_state=42))
]

print(f"\nBase learners: {[name for name, _ in learners]}")

# ============================================================================
# 3. Fit SuperLearner
# ============================================================================

print("\nFitting SuperLearner...")
sl = ExtendedSuperLearner(
    method='nnloglik',
    folds=5,
    random_state=42
)

# IMPORTANT: Set store_X=True to enable variable importance
sl.fit_explicit(X, y, learners, store_X=True)

print("SuperLearner fitted successfully!")
print(f"Meta-learner weights: {sl.meta_weights_}")

# ============================================================================
# 4. Compute variable importance using permutation method
# ============================================================================

print("\n" + "="*70)
print("Computing Permutation Feature Importance...")
print("="*70)

results_perm = compute_variable_importance(
    sl,
    method='permutation',
    n_repeats=5,  # Repeat permutations for variance estimation
    metric='auc',
    random_state=42,
    verbose=True
)

print("\nTop 5 most important features (Permutation):")
print(results_perm.summary(top_n=5))

# ============================================================================
# 5. Compute drop-column importance
# ============================================================================

print("\n" + "="*70)
print("Computing Drop-Column Feature Importance...")
print("="*70)

results_drop = compute_variable_importance(
    sl,
    method='drop_column',
    metric='auc',
    verbose=True
)

print("\nTop 5 most important features (Drop-Column):")
print(results_drop.summary(top_n=5))

# ============================================================================
# 6. Compute grouped importance with hierarchical clustering
# ============================================================================

print("\n" + "="*70)
print("Computing Grouped Feature Importance...")
print("="*70)

results_grouped = compute_variable_importance(
    sl,
    method='grouped',
    grouped_threshold=0.7,  # Cluster features with |correlation| > 0.7
    n_repeats=5,
    metric='auc',
    random_state=42,
    verbose=True
)

print("\nFeature clusters:")
print(results_grouped.cluster_info.groupby('group_id')['feature'].apply(list))

# ============================================================================
# 7. Compute all methods at once and compare
# ============================================================================

print("\n" + "="*70)
print("Computing All Methods for Comparison...")
print("="*70)

results_all = compute_variable_importance(
    sl,
    method=['permutation', 'drop_column'],
    n_repeats=3,
    metric='auc',
    random_state=42,
    verbose=False
)

print("\nComparison of feature rankings across methods:")
comparison = results_all.compare_methods()
print(comparison.head(5))

# ============================================================================
# 8. Visualize results
# ============================================================================

print("\n" + "="*70)
print("Creating visualizations...")
print("="*70)

# Bar plot of permutation importance
fig1, ax1 = results_perm.plot_importance_bar(top_n=10)
plt.savefig('variable_importance_bar.png', dpi=150, bbox_inches='tight')
print("Saved: variable_importance_bar.png")

# Heatmap comparing methods
fig2, ax2 = results_all.plot_importance_heatmap(top_n=10)
plt.savefig('variable_importance_heatmap.png', dpi=150, bbox_inches='tight')
print("Saved: variable_importance_heatmap.png")

# Feature clusters
fig3, ax3 = results_grouped.plot_grouped_clusters()
plt.savefig('feature_clusters.png', dpi=150, bbox_inches='tight')
print("Saved: feature_clusters.png")

# ============================================================================
# 9. Access raw fold-level importance
# ============================================================================

print("\n" + "="*70)
print("Raw fold-level importance data:")
print("="*70)
print(results_perm.raw_importance_df.head(10))

# ============================================================================
# 10. Get top features programmatically
# ============================================================================

print("\n" + "="*70)
print("Programmatic access to results:")
print("="*70)

top_features = results_perm.get_top_features(n=5)
print(f"Top 5 features: {top_features}")

# Access config
print(f"\nAnalysis configuration:")
for key, value in results_perm.config.items():
    print(f"  {key}: {value}")

print("\n" + "="*70)
print("Example complete!")
print("="*70)

# Show plots if running interactively
# plt.show()
