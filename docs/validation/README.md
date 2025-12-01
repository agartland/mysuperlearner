# MySuperLearner Validation

This directory contains validation materials comparing Python's `mysuperlearner` package with R's `SuperLearner` package.

## Files

### Python Implementation
- **[python_cv_superlearner.qmd](python_cv_superlearner.qmd)**: Cross-validated SuperLearner analysis using Python's `mysuperlearner` package

### R Implementation
- **[r_cv_superlearner.qmd](r_cv_superlearner.qmd)**: Cross-validated SuperLearner analysis using R's `SuperLearner` package (CV.SuperLearner function)

## Purpose

These QMD files validate `mysuperlearner` by comparing it against the established R `SuperLearner` package. Both implementations:

1. Use the same synthetic dataset structure (500 samples, 20 features, 3 informative)
2. Use comparable base learners:
   - SL.mean (intercept-only baseline)
   - SL.glm (logistic regression)
   - SL.randomForest
   - SL.gbm (gradient boosting)
   - SL.svm (support vector machine)
3. Use 5-fold cross-validation
4. Use the NNLogLik meta-learner method
5. Generate comparable visualizations:
   - Forest plots showing performance with confidence intervals
   - ROC curves for all learners
   - Box plots of cross-validated performance

## Rendering the Documents

### Python Version

Render the Python QMD file with the package in the PYTHONPATH:

```bash
PYTHONPATH=/fh/fast/gilbert_p/agartlan/gitrepo/mysuperlearner:$PYTHONPATH quarto render docs/validation/python_cv_superlearner.qmd
```

Output: `docs/validation/python_cv_superlearner.html`

### R Version

**Important**: Set the R environment first using `ml fhR/4.4.2`. This will affect the Python environment, so render the R and Python QMDs separately.

```bash
ml fhR/4.4.2
quarto render docs/validation/r_cv_superlearner.qmd
```

Output: `docs/validation/r_cv_superlearner.html`

## Interpreting Results

### Expected Similarities

Since both implementations use similar algorithms and the same meta-learning approach, you should see:

1. **Similar ranking of learners**: The relative performance order should generally match
2. **Comparable forest plots**: Confidence intervals should overlap for most learners
3. **Similar ROC curves**: Overall curve shapes should be comparable
4. **Consistent SuperLearner advantage**: The ensemble should typically match or exceed the best base learner

### Expected Differences

Exact numerical values will differ due to:

1. **Different random number generators**: Python and R use different RNG algorithms
2. **Different CV fold assignments**: Fold splits won't be identical
3. **Implementation differences**: sklearn vs R package implementations have algorithm variations
4. **Different default parameters**: Some learners may have different defaults across languages

The goal is **visual alignment** and **consistent patterns**, not exact numerical matching.

## Dependencies

### Python
- mysuperlearner
- scikit-learn
- numpy
- matplotlib

### R
- SuperLearner
- randomForest
- gbm
- e1071
- cvAUC
- ggplot2
- dplyr
- tidyr
- pROC

## Notes

- The R CV.SuperLearner can take several minutes to run with 5 learners and 5-fold CV
- Both QMDs create self-contained HTML files that can be opened in any browser
- The synthetic data generation attempts to create similar (but not identical) datasets in both languages
