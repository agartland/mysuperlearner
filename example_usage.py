"""
Example usage of the Extended SuperLearner implementation.
Demonstrates the R SuperLearner-like functionality with error handling.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Import our extended SuperLearner components
from .extended_superlearner import (
    ExtendedSuperLearner, 
    SuperLearnerCV, 
    create_superlearner, 
    evaluate_superlearner
)
from .custom_meta_learners import NNLogLikEstimator, AUCEstimator, MeanEstimator
from .error_handling import ErrorTracker, ErrorType


def demonstrate_basic_usage():
    """Demonstrate basic SuperLearner usage"""
    print("="*70)
    print("BASIC EXTENDED SUPERLEARNER DEMONSTRATION")
    print("="*70)
    
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=15, 
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Define base learners
    base_learners = [
        ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('GradientBoosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('LogisticRegression', LogisticRegression(random_state=42, max_iter=1000)),
        ('SVM', SVC(probability=True, random_state=42)),
        ('KNN', KNeighborsClassifier(n_neighbors=5)),
        ('NaiveBayes', GaussianNB())
    ]
    
    print(f"Dataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]}")
    print(f"Base learners: {len(base_learners)}")
    
    # Test different meta-learning methods
    methods = ['nnloglik', 'auc', 'nnls', 'logistic']
    results = {}
    
    for method in methods:
        print(f"\n--- Testing method: {method.upper()} ---")
        
        # Create SuperLearner
        sl = create_superlearner(
            learners=base_learners,
            method=method,
            folds=5,
            random_state=42,
            verbose=True,
            add_mean=True  # Add SL.mean equivalent
        )
        
        # Fit and predict
        sl.fit(X_train, y_train)
        y_pred_proba = sl.predict_proba(X_test)[:, 1]
        y_pred = sl.predict(X_test)
        
        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results[method] = {'accuracy': acc, 'auc': auc, 'superlearner': sl}
        
        print(f"Accuracy: {acc:.4f}")
        print(f"AUC: {auc:.4f}")
        
        # Print detailed summary
        sl.print_summary()
    
    # Compare results
    print(f"\n{'='*70}")
    print("COMPARISON OF META-LEARNING METHODS")
    print("="*70)
    print(f"{'Method':<12} {'Accuracy':<10} {'AUC':<10}")
    print("-"*32)
    
    best_acc = max(results.values(), key=lambda x: x['accuracy'])
    best_auc = max(results.values(), key=lambda x: x['auc'])
    
    for method, metrics in results.items():
        acc_marker = " *" if metrics == best_acc else ""
        auc_marker = " *" if metrics == best_auc else ""
        
        print(f"{method:<12} {metrics['accuracy']:<10.4f}{acc_marker:<2} {metrics['auc']:<10.4f}{auc_marker}")
    
    print("\n* indicates best performance")
    
    return results


def demonstrate_external_cv():
    """Demonstrate external cross-validation (CV.SuperLearner equivalent)"""
    print("\n" + "="*70)
    print("EXTERNAL CROSS-VALIDATION DEMONSTRATION")
    print("="*70)
    
    # Use real dataset for more realistic evaluation
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    print(f"Dataset: Breast Cancer (Wisconsin)")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Define base learners
    base_learners = [
        ('RandomForest', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('LogisticRegression', LogisticRegression(random_state=42, max_iter=1000)),
        ('SVM', SVC(probability=True, random_state=42)),
        ('KNN', KNeighborsClassifier(n_neighbors=7))
    ]
    
    # External cross-validation evaluation
    sl_cv = evaluate_superlearner(
        X=X, 
        y=y,
        learners=base_learners,
        method='nnloglik',
        inner_cv=5,
        outer_cv=10,
        scoring='auc',
        random_state=42,
        include_individual=True,
        verbose=True
    )
    
    # Get and display results
    cv_summary = sl_cv.get_cv_summary()
    print(f"\nExternal CV Summary:")
    print(cv_summary)
    
    # Plot results if possible
    try:
        sl_cv.plot_cv_results()
    except:
        print("Plotting not available")
    
    return sl_cv


def demonstrate_error_handling():
    """Demonstrate robust error handling"""
    print("\n" + "="*70)
    print("ERROR HANDLING DEMONSTRATION")
    print("="*70)
    
    # Create problematic dataset
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.binomial(1, 0.5, 100)
    
    # Add some problematic features
    X[:, 0] = 1  # Constant feature
    X[:, 1] = X[:, 0] + 1e-10 * np.random.randn(100)  # Nearly constant
    X[0:10, 2] = np.inf  # Some infinite values
    X[10:20, 3] = np.nan  # Some NaN values
    
    print("Created problematic dataset with:")
    print("- Constant features")
    print("- Nearly constant features")
    print("- Infinite values")
    print("- NaN values")
    
    # Define learners (some will likely fail)
    problematic_learners = [
        ('LogisticRegression', LogisticRegression(random_state=42, max_iter=10)),  # Low max_iter
        ('SVM', SVC(probability=True, random_state=42)),
        ('RandomForest', RandomForestClassifier(n_estimators=10, random_state=42)),
        ('KNN', KNeighborsClassifier(n_neighbors=5))
    ]
    
    # Create SuperLearner with detailed error tracking
    sl = ExtendedSuperLearner(
        method='nnloglik',
        folds=3,
        random_state=42,
        verbose=True,
        track_errors=True
    )
    
    sl.add_learners(problematic_learners)
    
    try:
        # This will likely encounter errors
        sl.fit(X, y)
        print("SuperLearner fit completed (with possible warnings)")
        
        # Try prediction
        y_pred = sl.predict(X)
        print("Predictions completed")
        
    except Exception as e:
        print(f"SuperLearner failed completely: {str(e)}")
    
    # Show error summary
    print(f"\nError Tracking Results:")
    error_summary = sl.get_error_summary()
    if error_summary is not None:
        print(error_summary)
    
    detailed_errors = sl.get_detailed_errors()
    if detailed_errors is not None and len(detailed_errors) > 0:
        print(f"\nDetailed Errors:")
        print(detailed_errors)
    
    # Print comprehensive summary
    sl.print_summary()
    
    return sl


def demonstrate_custom_meta_learners():
    """Demonstrate custom meta-learner functionality in isolation"""
    print("\n" + "="*70)
    print("CUSTOM META-LEARNERS DEMONSTRATION")
    print("="*70)
    
    # Generate sample meta-learner training data (like Z matrix in SuperLearner)
    np.random.seed(42)
    n_samples = 500
    n_base_learners = 4
    
    # Simulate base learner predictions (probabilities for binary classification)
    Z = np.random.beta(2, 2, size=(n_samples, n_base_learners))  # Beta distribution for probabilities
    y = np.random.binomial(1, 0.4, n_samples)  # Binary outcomes
    
    print(f"Meta-learner training data: {Z.shape[0]} samples, {Z.shape[1]} base learners")
    
    # Test custom meta-learners
    meta_learners = {
        'NNLogLik': NNLogLikEstimator(verbose=True),
        'AUC': AUCEstimator(verbose=True),
        'Mean': MeanEstimator()
    }
    
    results = {}
    
    for name, meta_learner in meta_learners.items():
        print(f"\n--- Testing {name} Meta-Learner ---")
        
        try:
            # Fit meta-learner
            meta_learner.fit(Z, y)
            
            # Make predictions
            y_pred_proba = meta_learner.predict_proba(Z)[:, 1]
            y_pred = meta_learner.predict(Z)
            
            # Evaluate
            acc = accuracy_score(y, y_pred)
            try:
                auc = roc_auc_score(y, y_pred_proba)
            except:
                auc = np.nan
            
            results[name] = {'accuracy': acc, 'auc': auc}
            
            print(f"Accuracy: {acc:.4f}")
            print(f"AUC: {auc:.4f}")
            
            # Show coefficients if available
            if hasattr(meta_learner, 'coef_'):
                print(f"Coefficients: {meta_learner.coef_}")
            
            # Show convergence info if available
            if hasattr(meta_learner, 'convergence_info_'):
                print(f"Convergence: {meta_learner.convergence_info_}")
                
        except Exception as e:
            print(f"Meta-learner {name} failed: {str(e)}")
            results[name] = {'accuracy': np.nan, 'auc': np.nan}
    
    # Summary
    print(f"\nMeta-Learner Comparison:")
    for name, metrics in results.items():
        print(f"{name:<10}: Acc={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")
    
    return results


def run_comprehensive_example():
    """Run all demonstrations"""
    print("COMPREHENSIVE EXTENDED SUPERLEARNER DEMONSTRATION")
    print("="*70)
    
    try:
        # 1. Basic usage demonstration
        basic_results = demonstrate_basic_usage()
        
        # 2. External CV demonstration  
        cv_results = demonstrate_external_cv()
        
        # 3. Error handling demonstration
        error_results = demonstrate_error_handling()
        
        # 4. Custom meta-learners demonstration
        meta_results = demonstrate_custom_meta_learners()
        
        print(f"\n" + "="*70)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("="*70)
        
        return {
            'basic': basic_results,
            'cv': cv_results, 
            'errors': error_results,
            'meta': meta_results
        }
        
    except Exception as e:
        print(f"Demonstration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def compare_with_sklearn_stacking():
    """Compare Extended SuperLearner with sklearn's StackingClassifier"""
    print("\n" + "="*70)
    print("COMPARISON WITH SKLEARN STACKING")
    print("="*70)
    
    try:
        from sklearn.ensemble import StackingClassifier
    except ImportError:
        print("sklearn.ensemble.StackingClassifier not available")
        return None
    
    # Generate dataset
    X, y = make_classification(
        n_samples=1000, 
        n_features=15, 
        n_informative=10,
        n_redundant=5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Define base learners
    base_learners = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('lr', LogisticRegression(random_state=42, max_iter=1000)),
        ('svm', SVC(probability=True, random_state=42))
    ]
    
    results = {}
    
    # 1. Extended SuperLearner (our implementation)
    print("Training Extended SuperLearner...")
    sl_extended = create_superlearner(
        learners=base_learners,
        method='nnloglik',
        folds=5,
        random_state=42,
        add_mean=False  # For fair comparison
    )
    
    sl_extended.fit(X_train, y_train)
    y_pred_sl = sl_extended.predict_proba(X_test)[:, 1]
    
    results['Extended_SuperLearner'] = {
        'accuracy': accuracy_score(y_test, (y_pred_sl > 0.5).astype(int)),
        'auc': roc_auc_score(y_test, y_pred_sl)
    }
    
    # 2. sklearn StackingClassifier  
    print("Training sklearn StackingClassifier...")
    sklearn_stacking = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(random_state=42),
        cv=5,
        n_jobs=1
    )
    
    sklearn_stacking.fit(X_train, y_train)
    y_pred_sklearn = sklearn_stacking.predict_proba(X_test)[:, 1]
    
    results['sklearn_Stacking'] = {
        'accuracy': accuracy_score(y_test, (y_pred_sklearn > 0.5).astype(int)),
        'auc': roc_auc_score(y_test, y_pred_sklearn)
    }
    
    # 3. Individual base learners for reference
    for name, learner in base_learners:
        learner_copy = type(learner)(**learner.get_params())
        learner_copy.fit(X_train, y_train)
        
        if hasattr(learner_copy, 'predict_proba'):
            y_pred_base = learner_copy.predict_proba(X_test)[:, 1]
        else:
            y_pred_base = learner_copy.decision_function(X_test)
        
        results[f'Individual_{name}'] = {
            'accuracy': accuracy_score(y_test, learner_copy.predict(X_test)),
            'auc': roc_auc_score(y_test, y_pred_base)
        }
    
    # Display comparison
    print(f"\nPerformance Comparison:")
    print(f"{'Method':<25} {'Accuracy':<10} {'AUC':<10}")
    print("-"*45)
    
    for method, metrics in results.items():
        print(f"{method:<25} {metrics['accuracy']:<10.4f} {metrics['auc']:<10.4f}")
    
    # Highlight best performers
    best_acc = max(results.values(), key=lambda x: x['accuracy'])['accuracy']
    best_auc = max(results.values(), key=lambda x: x['auc'])['auc']
    
    print(f"\nBest Accuracy: {best_acc:.4f}")
    print(f"Best AUC: {best_auc:.4f}")
    
    return results


def benchmark_performance():
    """Benchmark computational performance"""
    print("\n" + "="*70)
    print("PERFORMANCE BENCHMARKING")
    print("="*70)
    
    import time
    
    # Generate larger dataset for meaningful timing
    X, y = make_classification(
        n_samples=2000, 
        n_features=50, 
        n_informative=30,
        n_redundant=20,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    base_learners = [
        ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('GradientBoosting', GradientBoostingClassifier(n_estimators=50, random_state=42)),
        ('LogisticRegression', LogisticRegression(random_state=42, max_iter=1000)),
        ('SVM', SVC(probability=True, random_state=42))
    ]
    
    print(f"Dataset: {X_train.shape[0]} train, {X_test.shape[0]} test")
    print(f"Features: {X_train.shape[1]}")
    print(f"Base learners: {len(base_learners)}")
    
    # Benchmark different configurations
    configs = [
        ('nnloglik_5fold', {'method': 'nnloglik', 'folds': 5}),
        ('auc_5fold', {'method': 'auc', 'folds': 5}),
        ('nnloglik_10fold', {'method': 'nnloglik', 'folds': 10}),
        ('logistic_5fold', {'method': 'logistic', 'folds': 5})
    ]
    
    timing_results = {}
    
    for config_name, config_params in configs:
        print(f"\nBenchmarking {config_name}...")
        
        # Training time
        start_time = time.time()
        
        sl = create_superlearner(
            learners=base_learners,
            random_state=42,
            verbose=False,
            **config_params
        )
        
        sl.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Prediction time
        start_time = time.time()
        y_pred = sl.predict_proba(X_test)
        predict_time = time.time() - start_time
        
        # Performance
        y_pred_labels = sl.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_labels)
        auc = roc_auc_score(y_test, y_pred[:, 1])
        
        timing_results[config_name] = {
            'train_time': train_time,
            'predict_time': predict_time,
            'accuracy': accuracy,
            'auc': auc
        }
        
        print(f"Train time: {train_time:.2f}s")
        print(f"Predict time: {predict_time:.4f}s")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
    
    # Summary table
    print(f"\nBenchmarking Summary:")
    print(f"{'Config':<20} {'Train(s)':<10} {'Pred(s)':<10} {'Accuracy':<10} {'AUC':<10}")
    print("-"*60)
    
    for config, metrics in timing_results.items():
        print(f"{config:<20} {metrics['train_time']:<10.2f} "
              f"{metrics['predict_time']:<10.4f} {metrics['accuracy']:<10.4f} "
              f"{metrics['auc']:<10.4f}")
    
    return timing_results


if __name__ == "__main__":
    # Run comprehensive demonstration
    print("Starting Extended SuperLearner Demonstrations...")
    
    # Main demonstrations
    results = run_comprehensive_example()
    
    # Additional comparisons and benchmarks
    if results is not None:
        print("\nRunning additional comparisons...")
        
        # Compare with sklearn
        sklearn_comparison = compare_with_sklearn_stacking()
        
        # Performance benchmarking
        timing_results = benchmark_performance()
        
        print(f"\n" + "="*70)
        print("ALL DEMONSTRATIONS AND BENCHMARKS COMPLETED")
        print("="*70)
        print("\nKey Features Demonstrated:")
        print("✓ Multiple meta-learning methods (NNLogLik, AUC, NNLS, Logistic)")
        print("✓ External cross-validation (CV.SuperLearner equivalent)")
        print("✓ Robust error handling and tracking")
        print("✓ R SuperLearner-like functionality")
        print("✓ Performance comparison with sklearn")
        print("✓ Comprehensive benchmarking")
        print("\nImplementation ready for package development!")
        
    else:
        print("Demonstrations failed. Check error messages above.")