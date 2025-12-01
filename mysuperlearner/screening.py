"""
Screening and variable selection for Super Learner.

Provides:
- VariableSet: Manual feature selection by name or index
- CorrelationScreener: Select features correlated with outcome
- LassoScreener: Select features with non-zero Lasso coefficients
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from typing import List, Union, Optional


class VariableSet(BaseEstimator, TransformerMixin):
    """
    Manual feature selection by variable names or indices.

    Equivalent to user-defined variable sets in sl3.
    Useful for domain knowledge-based feature selection.

    Parameters
    ----------
    variables : list of str or int
        Variable names (if X is DataFrame) or indices (if X is array)
    name : str, optional
        Name for this variable set (for tracking/reporting)

    Examples
    --------
    >>> # With DataFrame input
    >>> var_set = VariableSet(variables=['age', 'sex', 'bmi'], name='baseline')
    >>> X_selected = var_set.fit_transform(X_df, y)
    >>>
    >>> # With array input (use indices)
    >>> var_set = VariableSet(variables=[0, 1, 2], name='first_three')
    >>> X_selected = var_set.fit_transform(X_array, y)
    """

    def __init__(self, variables: Union[List[str], List[int]], name: Optional[str] = None):
        self.variables = variables
        self.name = name
        self.feature_indices_ = None
        self.feature_names_ = None

    def fit(self, X, y=None):
        """
        Fit screener by identifying feature indices.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,), optional
            Target values (not used, present for API consistency)

        Returns
        -------
        self : VariableSet
            Fitted screener
        """
        if isinstance(X, pd.DataFrame):
            # DataFrame input - select by column names
            self.feature_names_ = list(X.columns)
            if all(isinstance(v, str) for v in self.variables):
                # Variable names provided
                self.feature_indices_ = [self.feature_names_.index(v) for v in self.variables]
            elif all(isinstance(v, int) for v in self.variables):
                # Indices provided
                self.feature_indices_ = list(self.variables)
            else:
                raise ValueError("variables must be all strings (names) or all ints (indices)")
        else:
            # Array input - select by indices
            X = check_array(X)
            if all(isinstance(v, int) for v in self.variables):
                self.feature_indices_ = list(self.variables)
                self.feature_names_ = [f"X{i}" for i in range(X.shape[1])]
            else:
                raise ValueError(
                    "For array input, variables must be integers (indices). "
                    "Use DataFrame input for selecting by column names."
                )

        return self

    def transform(self, X):
        """
        Select features.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Data to transform

        Returns
        -------
        X_selected : array or DataFrame, shape (n_samples, n_selected_features)
            Transformed data with selected features
        """
        if self.feature_indices_ is None:
            raise ValueError("VariableSet not fitted. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.feature_indices_]
        else:
            X = check_array(X)
            return X[:, self.feature_indices_]

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if self.feature_names_ is None:
            raise ValueError("VariableSet not fitted. Call fit() first.")
        return [self.feature_names_[i] for i in self.feature_indices_]


class CorrelationScreener(BaseEstimator, TransformerMixin):
    """
    Screen features by correlation with outcome.

    Selects features with absolute correlation >= threshold.
    Equivalent to correlation-based screening in sl3.

    Parameters
    ----------
    threshold : float, default=0.1
        Minimum absolute correlation to retain feature
    n_features : int, optional
        Alternative: select top n_features by correlation (ignores threshold)
    name : str, optional
        Name for this screener (for tracking/reporting)

    Examples
    --------
    >>> # Select features with |correlation| >= 0.2
    >>> screener = CorrelationScreener(threshold=0.2)
    >>> X_screened = screener.fit_transform(X, y)
    >>>
    >>> # Select top 10 features by correlation
    >>> screener = CorrelationScreener(n_features=10)
    >>> X_screened = screener.fit_transform(X, y)
    """

    def __init__(self, threshold: float = 0.1, n_features: Optional[int] = None,
                 name: Optional[str] = None):
        self.threshold = threshold
        self.n_features = n_features
        self.name = name
        self.feature_indices_ = None
        self.correlations_ = None
        self.feature_names_ = None

    def fit(self, X, y):
        """
        Fit screener by computing correlations.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values

        Returns
        -------
        self : CorrelationScreener
            Fitted screener
        """
        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.values
        else:
            X_arr = check_array(X)
            self.feature_names_ = [f"X{i}" for i in range(X_arr.shape[1])]

        y = np.asarray(y)

        # Compute correlations
        n_features = X_arr.shape[1]
        self.correlations_ = np.zeros(n_features)

        for i in range(n_features):
            # Use Pearson correlation
            corr = np.corrcoef(X_arr[:, i], y)[0, 1]
            self.correlations_[i] = corr if not np.isnan(corr) else 0.0

        # Select features
        abs_corr = np.abs(self.correlations_)

        if self.n_features is not None:
            # Select top n_features
            indices = np.argsort(abs_corr)[::-1][:self.n_features]
            self.feature_indices_ = sorted(indices.tolist())
        else:
            # Select by threshold
            self.feature_indices_ = np.where(abs_corr >= self.threshold)[0].tolist()

        if len(self.feature_indices_) == 0:
            raise ValueError(
                f"No features passed screening (threshold={self.threshold}). "
                f"Max correlation: {abs_corr.max():.4f}"
            )

        return self

    def transform(self, X):
        """
        Select screened features.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Data to transform

        Returns
        -------
        X_screened : array or DataFrame, shape (n_samples, n_selected_features)
            Transformed data with screened features
        """
        if self.feature_indices_ is None:
            raise ValueError("CorrelationScreener not fitted. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.feature_indices_]
        else:
            X = check_array(X)
            return X[:, self.feature_indices_]

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if self.feature_names_ is None:
            raise ValueError("CorrelationScreener not fitted. Call fit() first.")
        return [self.feature_names_[i] for i in self.feature_indices_]


class LassoScreener(BaseEstimator, TransformerMixin):
    """
    Screen features using Lasso regularization.

    Selects features with non-zero Lasso coefficients.
    Automatically determines regularization strength via cross-validation.

    Parameters
    ----------
    alpha : float, optional
        Regularization strength. If None, use LassoCV for automatic selection.
    min_features : int, default=1
        Minimum number of features to retain. If Lasso selects fewer,
        use top features by coefficient magnitude.
    n_alphas : int, default=100
        Number of alphas to try in LassoCV (if alpha=None)
    cv : int, default=5
        Number of CV folds for LassoCV (if alpha=None)
    name : str, optional
        Name for this screener (for tracking/reporting)
    classification : bool, default=True
        Whether this is a classification problem (use LogisticRegressionCV)
        or regression (use LassoCV)

    Examples
    --------
    >>> # Automatic alpha selection
    >>> screener = LassoScreener()
    >>> X_screened = screener.fit_transform(X, y)
    >>>
    >>> # Fixed alpha
    >>> screener = LassoScreener(alpha=0.01)
    >>> X_screened = screener.fit_transform(X, y)
    """

    def __init__(self, alpha: Optional[float] = None, min_features: int = 1,
                 n_alphas: int = 100, cv: int = 5, name: Optional[str] = None,
                 classification: bool = True):
        self.alpha = alpha
        self.min_features = min_features
        self.n_alphas = n_alphas
        self.cv = cv
        self.name = name
        self.classification = classification
        self.feature_indices_ = None
        self.coefficients_ = None
        self.feature_names_ = None
        self.model_ = None

    def fit(self, X, y):
        """
        Fit screener using Lasso.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values

        Returns
        -------
        self : LassoScreener
            Fitted screener
        """
        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.values
        else:
            X_arr = check_array(X)
            self.feature_names_ = [f"X{i}" for i in range(X_arr.shape[1])]

        y = np.asarray(y)

        # Fit Lasso
        if self.classification:
            if self.alpha is None:
                # Use LogisticRegressionCV for automatic alpha selection
                self.model_ = LogisticRegressionCV(
                    penalty='l1',
                    solver='saga',
                    cv=self.cv,
                    max_iter=1000,
                    random_state=42
                )
            else:
                from sklearn.linear_model import LogisticRegression
                self.model_ = LogisticRegression(
                    penalty='l1',
                    C=1.0/self.alpha,
                    solver='saga',
                    max_iter=1000,
                    random_state=42
                )
        else:
            if self.alpha is None:
                self.model_ = LassoCV(
                    n_alphas=self.n_alphas,
                    cv=self.cv,
                    random_state=42
                )
            else:
                from sklearn.linear_model import Lasso
                self.model_ = Lasso(alpha=self.alpha, random_state=42)

        self.model_.fit(X_arr, y)

        # Extract coefficients
        if hasattr(self.model_, 'coef_'):
            coef = self.model_.coef_
            if coef.ndim > 1:
                # Multi-class: use first class coefficients
                coef = coef[0]
            self.coefficients_ = coef
        else:
            raise ValueError("Model does not have coef_ attribute")

        # Select non-zero features
        nonzero = np.where(self.coefficients_ != 0)[0]

        if len(nonzero) >= self.min_features:
            self.feature_indices_ = sorted(nonzero.tolist())
        else:
            # Not enough non-zero coefficients, select top by magnitude
            abs_coef = np.abs(self.coefficients_)
            indices = np.argsort(abs_coef)[::-1][:self.min_features]
            self.feature_indices_ = sorted(indices.tolist())

        return self

    def transform(self, X):
        """
        Select screened features.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Data to transform

        Returns
        -------
        X_screened : array or DataFrame, shape (n_samples, n_selected_features)
            Transformed data with screened features
        """
        if self.feature_indices_ is None:
            raise ValueError("LassoScreener not fitted. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.feature_indices_]
        else:
            X = check_array(X)
            return X[:, self.feature_indices_]

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if self.feature_names_ is None:
            raise ValueError("LassoScreener not fitted. Call fit() first.")
        return [self.feature_names_[i] for i in self.feature_indices_]
