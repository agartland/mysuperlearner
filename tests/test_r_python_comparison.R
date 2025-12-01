#!/usr/bin/env Rscript
# Direct comparison of R SuperLearner and Python mysuperlearner
# This script runs CV.SuperLearner in R and saves results for comparison with Python

library(SuperLearner)
library(jsonlite)

# Set seed for reproducibility
set.seed(42)

# Generate same data as Python (need to export from Python)
# For now, we'll create similar data
n <- 300
p <- 10

# Simple data generation (will be replaced with exported Python data)
X <- matrix(rnorm(n * p), nrow = n, ncol = p)
# Create binary outcome with some signal
probs <- plogis(0.5 * X[,1] + 0.3 * X[,2] - 0.4 * X[,3])
y <- rbinom(n, 1, probs)

# Save data for Python to use
write.csv(data.frame(y = y, X),
          file = "tests/test_data_r_python_comparison.csv",
          row.names = FALSE)

cat("Data shape: ", dim(X), "\n")
cat("Outcome distribution: ", table(y), "\n")

# Define learner library (matching Python)
SL.library <- c("SL.glm", "SL.mean", "SL.randomForest")

# Run CV.SuperLearner
cat("\nRunning CV.SuperLearner...\n")
cv_fit <- CV.SuperLearner(
  Y = y,
  X = data.frame(X),
  V = 5,  # 5-fold outer CV
  SL.library = SL.library,
  method = "method.NNloglik",
  cvControl = list(V = 5),  # 5-fold inner CV
  verbose = FALSE
)

cat("CV.SuperLearner completed\n")

# Extract key results
results <- list(
  # Predictions
  SL_predict = cv_fit$SL.predict,
  discreteSL_predict = cv_fit$discreteSL.predict,
  library_predict = cv_fit$library.predict,

  # Metadata
  whichDiscreteSL = cv_fit$whichDiscreteSL,
  coef = cv_fit$coef,
  libraryNames = cv_fit$libraryNames,

  # Fold info
  folds = cv_fit$folds,
  V = cv_fit$V,

  # True outcomes
  Y = y
)

# Compute CV risk from each fold's SuperLearner object
cv_risks_list <- list()
for (i in 1:cv_fit$V) {
  if (!is.null(cv_fit$AllSL[[i]])) {
    cv_risks_list[[i]] <- cv_fit$AllSL[[i]]$cvRisk
  }
}

# Add CV risks to results
results$cv_risks <- cv_risks_list

# Save results
saveRDS(results, file = "tests/r_cv_superlearner_results.rds")
write_json(
  list(
    whichDiscreteSL = unlist(results$whichDiscreteSL),
    libraryNames = results$libraryNames,
    coef = as.data.frame(results$coef),
    V = results$V
  ),
  path = "tests/r_cv_superlearner_results.json",
  pretty = TRUE,
  auto_unbox = TRUE
)

cat("\nResults saved to tests/r_cv_superlearner_results.rds and .json\n")

# Print summary
cat("\n=== R CV.SuperLearner Summary ===\n")
cat("Number of folds:", cv_fit$V, "\n")
cat("Learners:", paste(cv_fit$libraryNames, collapse = ", "), "\n")
cat("\nDiscrete SL selections per fold:\n")
for (i in 1:cv_fit$V) {
  cat("  Fold", i, ":", cv_fit$whichDiscreteSL[[i]], "\n")
}

cat("\nMeta-learner coefficients:\n")
print(cv_fit$coef)

# Compute performance metrics
library(pROC)
sl_auc <- auc(y, cv_fit$SL.predict)
discrete_auc <- auc(y, cv_fit$discreteSL.predict)

cat("\nPerformance (AUC):\n")
cat("  SuperLearner:", sl_auc, "\n")
cat("  Discrete SL:", discrete_auc, "\n")

# Base learner performance
for (i in 1:ncol(cv_fit$library.predict)) {
  lib_auc <- auc(y, cv_fit$library.predict[, i])
  cat(" ", cv_fit$libraryNames[i], ":", lib_auc, "\n")
}

cat("\n=== Validation Complete ===\n")
cat("Run the Python comparison script to verify consistency.\n")
