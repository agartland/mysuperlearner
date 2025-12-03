"""
Progress monitoring utilities for MySuperLearner.
Provides text-based progress updates and timing estimates.
"""

import time
import numpy as np
from typing import List, Dict, Optional


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable string.

    Parameters
    ----------
    seconds : float
        Time in seconds

    Returns
    -------
    str
        Formatted time string (e.g., '2m 30s', '1h 15m', '45.2s')
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def estimate_time_remaining(completed_folds: int, total_folds: int,
                           elapsed_time: float) -> Optional[float]:
    """
    Estimate remaining time based on average fold time.

    Parameters
    ----------
    completed_folds : int
        Number of folds completed
    total_folds : int
        Total number of folds
    elapsed_time : float
        Time elapsed so far (seconds)

    Returns
    -------
    float or None
        Estimated time remaining in seconds, or None if can't estimate
    """
    if completed_folds == 0:
        return None
    avg_time_per_fold = elapsed_time / completed_folds
    remaining_folds = total_folds - completed_folds
    return avg_time_per_fold * remaining_folds


def print_learner_summary(learner_timings: List[Dict], outer_fold: int):
    """
    Print summary of learner timings for a completed outer fold.

    Parameters
    ----------
    learner_timings : list of dict
        List of timing records from _build_level1
    outer_fold : int
        Outer fold index (0-based)

    Example Output
    --------------
    Fold 1 summary (45.2s total):
      RandomForest: 25.3s (avg 5.1s/inner-fold, 5 folds)
      LogisticRegression: 12.4s (avg 2.5s/inner-fold, 5 folds)
      XGBoost: 7.5s (avg 1.5s/inner-fold, 5 folds)
    """
    # Aggregate by learner
    learner_stats = {}
    for record in learner_timings:
        name = record['learner_name']
        if name not in learner_stats:
            learner_stats[name] = []
        learner_stats[name].append(record['fit_time'])

    # Calculate totals
    total_time = sum(record['fit_time'] for record in learner_timings)

    print(f"\n  Fold {outer_fold + 1} summary ({format_time(total_time)} total):")

    # Sort by total time descending
    sorted_learners = sorted(
        learner_stats.items(),
        key=lambda x: sum(x[1]),
        reverse=True
    )

    for learner_name, times in sorted_learners:
        total = sum(times)
        avg = np.mean(times)
        n_folds = len(times)
        print(f"    {learner_name}: {format_time(total)} "
              f"(avg {format_time(avg)}/inner-fold, {n_folds} folds)")


def print_progress_header(total_folds: int, n_learners: int):
    """
    Print initial header for CV progress.

    Parameters
    ----------
    total_folds : int
        Total number of outer folds
    n_learners : int
        Number of learners in the ensemble
    """
    print(f"\nStarting CV evaluation: {total_folds} outer folds, "
          f"{n_learners} learners")
    print("=" * 70)


def print_overall_progress(completed_folds: int, total_folds: int,
                          start_time: float, fold_summaries: List[Dict]):
    """
    Print overall progress update after each fold completes.

    Parameters
    ----------
    completed_folds : int
        Number of folds completed
    total_folds : int
        Total number of folds
    start_time : float
        CV start time (time.time())
    fold_summaries : list of dict
        List of fold timing summaries

    Example Output
    --------------
    Progress: 2/5 folds completed (40%)
    Elapsed: 1m 30s | Estimated remaining: 2m 15s
    ----------------------------------------------------------------------
    """
    elapsed = time.time() - start_time
    eta = estimate_time_remaining(completed_folds, total_folds, elapsed)

    pct = (completed_folds / total_folds) * 100
    print(f"\nProgress: {completed_folds}/{total_folds} folds completed ({pct:.0f}%)")

    if eta is not None:
        print(f"Elapsed: {format_time(elapsed)} | "
              f"Estimated remaining: {format_time(eta)}")
    else:
        print(f"Elapsed: {format_time(elapsed)}")

    print("-" * 70)


def print_final_summary(fold_summaries: List[Dict], total_time: float):
    """
    Print final summary when all folds complete.

    Parameters
    ----------
    fold_summaries : list of dict
        List of fold timing summaries
    total_time : float
        Total elapsed time (seconds)

    Example Output
    --------------
    ======================================================================
    CV Evaluation Complete!
    Total time: 5m 30s
    Average per fold: 1m 6s

    Learner timing summary (across all folds):
      RandomForest: 3m 15s (avg 39s/fold)
      LogisticRegression: 1m 45s (avg 21s/fold)
      XGBoost: 30s (avg 6s/fold)
    ======================================================================
    """
    print("\n" + "=" * 70)
    print("CV Evaluation Complete!")
    print(f"Total time: {format_time(total_time)}")

    avg_fold_time = total_time / len(fold_summaries)
    print(f"Average per fold: {format_time(avg_fold_time)}")

    # Aggregate across all folds
    all_learner_times = {}
    for fold_summary in fold_summaries:
        for record in fold_summary['learner_timings']:
            name = record['learner_name']
            if name not in all_learner_times:
                all_learner_times[name] = []
            all_learner_times[name].append(record['fit_time'])

    print("\nLearner timing summary (across all folds):")
    sorted_learners = sorted(
        all_learner_times.items(),
        key=lambda x: sum(x[1]),
        reverse=True
    )

    n_folds = len(fold_summaries)
    for learner_name, times in sorted_learners:
        total = sum(times)
        avg_per_fold = total / n_folds
        print(f"  {learner_name}: {format_time(total)} "
              f"(avg {format_time(avg_per_fold)}/fold)")

    print("=" * 70)
