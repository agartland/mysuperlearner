"""
Example usage of mysuperlearner package

This script demonstrates:
1. Basic SuperLearner usage with different meta-learners
2. Using diagnostics to inspect fitted models
3. External cross-validation for unbiased performance evaluation
4. Using visualization functions and result objects
"""

from mysuperlearner import SuperLearner, CVSuperLearner, visualization
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("MySuperLearner Example Usage")
print("=" * 70)

# Generate synthetic data
print("\n1. Generating synthetic classification dataset...")
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# Define base learners
learners = [
    ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=42)),
    ('SVM', SVC(probability=True, random_state=42))
]

# Example 1: Basic SuperLearner with NNLogLik meta-learner
print("\n2. Training SuperLearner with NNLogLik meta-learner...")
sl = SuperLearner(learners=learners, method='nnloglik', cv=5, random_state=42, verbose=False)
sl.fit(X_train, y_train)

# Make predictions
y_pred_proba = sl.predict_proba(X_test)[:, 1]
y_pred = sl.predict(X_test)

# Evaluate
auc = roc_auc_score(y_test, y_pred_proba)
acc = accuracy_score(y_test, y_pred)
print(f"   AUC: {auc:.4f}")
print(f"   Accuracy: {acc:.4f}")

# Show meta-learner weights
if sl.meta_weights_ is not None:
    print("\n   Meta-learner weights:")
    for (name, _), weight in zip(learners, sl.meta_weights_):
        print(f"   - {name}: {weight:.4f}")

# Get diagnostics
print("\n   Model diagnostics:")
diagnostics = sl.get_diagnostics()
print(f"   - Method: {diagnostics['method']}")
print(f"   - Number of folds: {diagnostics['n_folds']}")
print(f"   - Number of errors: {diagnostics['n_errors']}")
if 'cv_scores' in diagnostics and diagnostics['cv_scores']:
    print("   - CV AUC scores:")
    for name, score in diagnostics['cv_scores'].items():
        print(f"     - {name}: {score:.4f}")

# Example 2: Compare different meta-learning methods
print("\n3. Comparing different meta-learning methods...")
methods = ['nnloglik', 'auc', 'nnls', 'logistic']
results = {}

for method in methods:
    sl = SuperLearner(learners=learners, method=method, cv=5, random_state=42, verbose=False)
    sl.fit(X_train, y_train)

    y_pred_proba = sl.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)

    results[method] = auc_score
    print(f"   {method:12s}: AUC = {auc_score:.4f}")

# Example 3: External cross-validation with result object
print("\n4. Running external cross-validation (like R's CV.SuperLearner)...")
cv_sl = CVSuperLearner(learners=learners, method='nnloglik', cv=5, inner_cv=5, random_state=42, verbose=False, n_jobs=1)
cv_sl.fit(X, y)
cv_results = cv_sl.get_results()

print(f"\nCV Results object: {cv_results}")
print(f"   - {cv_results.metrics.shape[0]} total evaluations")
print(f"   - Predictions stored: {cv_results.predictions is not None}")

# Use built-in summary method
print("\n   Cross-validation summary statistics:")
summary = cv_results.summary()
print(summary[['auc', 'accuracy']])

# Compare SuperLearner to best individual using built-in method
print("\n   Comparison to best base learner:")
comparison = cv_results.compare_to_best()
print(comparison[['metric', 'SuperLearner', 'Best_Base', 'Best_Base_Name', 'Improvement']])

# Example 4: Using standalone visualization functions
print("\n5. Creating visualizations...")

# Option 1: Use standalone functions from visualization module
fig, ax = visualization.plot_cv_forest(cv_results.metrics, metric='auc')
plt.savefig('forest_plot.png', dpi=150, bbox_inches='tight')
print("   - Forest plot saved to forest_plot.png")
plt.close()

# Option 2: Use convenience methods on result object
fig, ax = cv_results.plot_boxplot(metric='auc')
plt.savefig('boxplot.png', dpi=150, bbox_inches='tight')
print("   - Box plot saved to boxplot.png")
plt.close()

# ROC curves (requires predictions)
fig, ax = cv_results.plot_roc_curves()
plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
print("   - ROC curves saved to roc_curves.png")
plt.close()

# Calibration curves (requires predictions)
fig, ax = cv_results.plot_calibration(n_bins=10)
plt.savefig('calibration.png', dpi=150, bbox_inches='tight')
print("   - Calibration curves saved to calibration.png")
plt.close()

print("\n" + "=" * 70)
print("Example completed successfully!")
print("=" * 70)
