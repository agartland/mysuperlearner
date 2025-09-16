# Example usage of mysuperlearner package

from mysuperlearner.extended_super_learner import ExtendedSuperLearner
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base learners
learners = [
    ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('LogisticRegression', LogisticRegression(random_state=42)),
    ('SVM', SVC(probability=True, random_state=42))
]

# Create and fit SuperLearner
sl = ExtendedSuperLearner(method='nnloglik', folds=5, random_state=42, verbose=True)
sl.add(learners)
sl.add_meta()
sl.fit(X_train, y_train)

# Predict
y_pred = sl.predict(X_test)
print('Predictions:', y_pred)
