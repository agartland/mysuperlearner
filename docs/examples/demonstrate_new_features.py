#!/usr/bin/env python
"""
Demonstrate New Features in mysuperlearner

This script demonstrates the newly implemented features that match R's CV.SuperLearner:
1. Discrete SuperLearner selection (best base learner by CV risk)
2. Meta-learner coefficients per fold
3. CV risk reporting for each base learner
4. Enhanced results object with comprehensive diagnostics

These features enable:
- Understanding which learner performs best in each fold
- Analyzing meta-learner weight stability across folds
- Comparing learner performance via CV risk
- Full algorithmic parity with R's SuperLearner package
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from mysuperlearner import CVSuperLearner, InterceptOnlyEstimator


def create_example_data():
    """Create example classification dataset."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        flip_y=0.1,
        class_sep=0.8,
        random_state=42
    )
    return X, y


def main():
    print("="*80)
    print("MYSUPERLEARNER NEW FEATURES DEMONSTRATION")
    print("="*80)

    # Create data
    print("\n1. Creating example dataset...")
    X, y = create_example_data()
    print(f"   Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Class balance: {np.bincount(y)}")

    # Define learner library
    print("\n2. Defining learner library...")
    learners = [
        ('RF', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
        ('GBM', GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
        ('LR', LogisticRegression(random_state=42, max_iter=1000)),
        ('SVM', SVC(probability=True, random_state=42)),
        ('Mean', InterceptOnlyEstimator())
    ]
    print("   Learners:", [name for name, _ in learners])

    # Run CVSuperLearner
    print("\n3. Running CVSuperLearner with 5-fold CV...")
    cv_sl = CVSuperLearner(
        learners=learners,
        method='nnloglik',
        cv=5,
        inner_cv=5,
        random_state=42,
        verbose=False
    )
    cv_sl.fit(X, y)
    results = cv_sl.get_results()
    print("   ✓ Complete")

    # Display new features
    print("\n" + "="*80)
    print("FEATURE 1: DISCRETE SUPERLEARNER SELECTION")
    print("="*80)
    print("\nDiscrete SL selects the best single learner per fold based on CV risk.")
    print("This is equivalent to R's discreteSL.predict and whichDiscreteSL.\n")

    print(f"{'Fold':<10} {'Selected Learner':<20} {'Reason':<40}")
    print("-"*80)
    for fold_idx, selected in enumerate(results.which_discrete_sl, 1):
        fold_cv_risk = results.cv_risk[results.cv_risk['fold'] == fold_idx]
        min_risk = fold_cv_risk['cv_risk'].min()
        print(f"{fold_idx:<10} {selected:<20} Minimum CV risk: {min_risk:.6f}")

    # Discrete SL performance
    print("\n" + "-"*80)
    discrete_metrics = results.metrics[results.metrics['learner'] == 'DiscreteSL']
    print(f"Discrete SL Mean AUC: {discrete_metrics['auc'].mean():.4f}")
    print(f"Discrete SL Mean Accuracy: {discrete_metrics['accuracy'].mean():.4f}")

    # Display new features - Coefficients
    print("\n" + "="*80)
    print("FEATURE 2: META-LEARNER COEFFICIENTS PER FOLD")
    print("="*80)
    print("\nCoefficients show how the meta-learner weights each base learner.")
    print("This is equivalent to R's coef output.\n")

    # Pivot coefficients for better display
    coef_pivot = results.coef.pivot(index='fold', columns='learner', values='coefficient')
    print(coef_pivot.to_string())

    print("\n" + "-"*80)
    print("Coefficient Statistics Across Folds:")
    print("-"*80)
    coef_stats = results.coef.groupby('learner')['coefficient'].agg(['mean', 'std', 'min', 'max'])
    print(coef_stats.to_string())

    print("\nInterpretation:")
    mean_coefs = coef_stats['mean'].sort_values(ascending=False)
    print(f"  - Most weighted learner on average: {mean_coefs.index[0]} ({mean_coefs.iloc[0]:.3f})")
    print(f"  - Least weighted learner on average: {mean_coefs.index[-1]} ({mean_coefs.iloc[-1]:.3f})")

    # Most stable learner (lowest std)
    most_stable = coef_stats['std'].idxmin()
    print(f"  - Most stable weights: {most_stable} (std={coef_stats.loc[most_stable, 'std']:.4f})")

    # Display new features - CV Risk
    print("\n" + "="*80)
    print("FEATURE 3: CV RISK (INNER CROSS-VALIDATION RISK)")
    print("="*80)
    print("\nCV risk measures how well each learner performs on the inner CV folds.")
    print("Lower CV risk indicates better performance. This is equivalent to R's cvRisk.\n")

    # Pivot CV risk for better display
    cv_risk_pivot = results.cv_risk.pivot(index='fold', columns='learner', values='cv_risk')
    print(cv_risk_pivot.to_string())

    print("\n" + "-"*80)
    print("CV Risk Statistics Across Folds:")
    print("-"*80)
    cv_risk_stats = results.cv_risk.groupby('learner')['cv_risk'].agg(['mean', 'std', 'min', 'max'])
    print(cv_risk_stats.to_string())

    print("\nInterpretation:")
    ranked_risk = cv_risk_stats['mean'].sort_values()
    print(f"  - Best learner by CV risk: {ranked_risk.index[0]} ({ranked_risk.iloc[0]:.6f})")
    print(f"  - Worst learner by CV risk: {ranked_risk.index[-1]} ({ranked_risk.iloc[-1]:.6f})")

    # Relationship between CV risk and discrete SL selection
    print("\nDiscrete SL Selection Frequency:")
    selection_counts = pd.Series(results.which_discrete_sl).value_counts()
    for learner, count in selection_counts.items():
        print(f"  - {learner}: {count} folds ({count/5*100:.0f}%)")

    # Display comprehensive metrics
    print("\n" + "="*80)
    print("COMPREHENSIVE PERFORMANCE COMPARISON")
    print("="*80)
    print("\nComparison of SuperLearner, Discrete SL, and base learners:\n")

    summary = results.summary()
    # Select key metrics for display
    metrics_to_show = ['auc', 'accuracy', 'logloss']
    print(summary[metrics_to_show].to_string())

    # Compare SuperLearner vs Discrete SL
    print("\n" + "-"*80)
    print("SuperLearner vs Discrete SuperLearner:")
    print("-"*80)

    sl_metrics = results.metrics[results.metrics['learner'] == 'SuperLearner']
    discrete_metrics = results.metrics[results.metrics['learner'] == 'DiscreteSL']

    print(f"{'Metric':<15} {'SuperLearner':<15} {'Discrete SL':<15} {'Difference':<15}")
    print("-"*80)
    for metric in ['auc', 'accuracy', 'logloss']:
        sl_val = sl_metrics[metric].mean()
        discrete_val = discrete_metrics[metric].mean()
        diff = sl_val - discrete_val
        print(f"{metric:<15} {sl_val:<15.4f} {discrete_val:<15.4f} {diff:<+15.4f}")

    # Insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    # 1. Meta-learner benefit
    sl_auc = sl_metrics['auc'].mean()
    best_base_auc = results.metrics[results.metrics['learner_type'] == 'base'].groupby('learner')['auc'].mean().max()
    improvement = sl_auc - best_base_auc

    print(f"\n1. SuperLearner Ensemble Benefit:")
    print(f"   SuperLearner AUC: {sl_auc:.4f}")
    print(f"   Best base learner AUC: {best_base_auc:.4f}")
    print(f"   Improvement: {improvement:+.4f} ({improvement/best_base_auc*100:+.2f}%)")

    # 2. Coefficient stability
    coef_cv = results.coef.groupby('learner')['coefficient'].std().mean()
    print(f"\n2. Meta-Learner Weight Stability:")
    print(f"   Average coefficient std across learners: {coef_cv:.4f}")
    if coef_cv < 0.1:
        print("   → Weights are very stable across folds (good!)")
    elif coef_cv < 0.2:
        print("   → Weights are moderately stable across folds")
    else:
        print("   → Weights vary substantially across folds (high variability)")

    # 3. Discrete SL diversity
    n_unique_selections = len(set(results.which_discrete_sl))
    print(f"\n3. Discrete SuperLearner Selection Diversity:")
    print(f"   Number of different learners selected: {n_unique_selections} out of {len(learners)}")
    if n_unique_selections == 1:
        print(f"   → Same learner ({results.which_discrete_sl[0]}) selected in all folds")
    else:
        print(f"   → Different learners selected across folds (diverse)")

    # 4. Mean baseline comparison
    mean_auc = results.metrics[results.metrics['learner'] == 'Mean']['auc'].mean()
    mean_improvement = sl_auc - mean_auc
    print(f"\n4. Improvement Over Baseline (Mean Predictor):")
    print(f"   Mean predictor AUC: {mean_auc:.4f}")
    print(f"   SuperLearner improvement: {mean_improvement:+.4f} ({mean_improvement/mean_auc*100:+.2f}%)")

    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nThese new features provide:")
    print("  ✓ Full algorithmic parity with R's CV.SuperLearner")
    print("  ✓ Discrete SL for best single learner selection")
    print("  ✓ Meta-learner coefficient tracking and analysis")
    print("  ✓ CV risk for comparing learner performance")
    print("  ✓ Enhanced diagnostics for ensemble understanding")
    print("\nFor more details, see docs/R_PYTHON_ALGORITHM_COMPARISON.md")
    print("="*80)


if __name__ == '__main__':
    main()
