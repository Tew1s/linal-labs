import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_diabetes

# 'Please, compare and analyze results. Add conclusions as comments here or to a readme file.'
# Ok

def preprocess(X: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    """
    Preprocesses the input data by scaling features and splitting into training and test sets.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        list[np.ndarray]: List containing training and test sets for features and target.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

    return [X_train, X_test, y_train, y_test]


def get_regression_data() -> list[np.ndarray]:
    """
    Loads and preprocesses the diabetes dataset for regression tasks.

    Returns:
        list[np.ndarray]: List containing training and test sets for features and target.
    """
    data = load_diabetes()
    X, y = data.data, data.target
    return preprocess(X, y)


def get_classification_data() -> list[np.ndarray]:
    """
    Loads and preprocesses the breast cancer dataset for classification tasks.

    Returns:
        list[np.ndarray]: List containing training and test sets for features and target.
    """
    data = load_breast_cancer()
    X, y = data.data, data.target
    return preprocess(X, y)


def linear_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a linear regression model on the given data.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Trained linear regression model.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model


def ridge_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a ridge regression model with hyperparameter tuning using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best ridge regression model found by GridSearchCV.
    """
    params = {"alpha": [0.1, 1.0, 10.0, 100.0]}
    model = GridSearchCV(Ridge(), param_grid=params, cv=5, scoring="neg_mean_squared_error")
    model.fit(X, y)
    return model.best_estimator_


def lasso_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a lasso regression model with hyperparameter tuning using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best lasso regression model found by GridSearchCV.
    """
    params = {"alpha": [0.1, 1.0, 10.0, 100.0]}
    model = GridSearchCV(Lasso(max_iter=10000), param_grid=params, cv=5, scoring="neg_mean_squared_error")
    model.fit(X, y)
    return model.best_estimator_


def logistic_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model without regularization on the given data.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Trained logistic regression model.
    """
    model = LogisticRegression(penalty='none', solver='lbfgs', max_iter=10000)
    model.fit(X, y)
    return model


def logistic_l2_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model with L2 regularization using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best logistic regression model with L2 regularization found by GridSearchCV.
    """
    params = {"C": [0.1, 1.0, 10.0, 100.0]}
    model = GridSearchCV(LogisticRegression(penalty='l2', solver='lbfgs', max_iter=10000), param_grid=params, cv=5)
    model.fit(X, y)
    return model.best_estimator_


def logistic_l1_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model with L1 regularization using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best logistic regression model with L1 regularization found by GridSearchCV.
    """
    params = {"C": [0.1, 1.0, 10.0, 100.0]}
    model = GridSearchCV(LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000), param_grid=params, cv=5)
    model.fit(X, y)
    return model.best_estimator_

#
#=== Regression ===
#Linear Regression: MSE = 2894.212, R2 = 0.496
#Ridge Regression: MSE = 2877.719, R2 = 0.499
#Lasso Regression: MSE = 2863.670, R2 = 0.501
#All models perform similarly, with Lasso slightly outperforming the others. Regularization has a marginal effect on this dataset.

#=== Classification ===
#Logistic Regression: Accuracy = 0.947, F1 Score = 0.958
#Logistic Regression (L2): Accuracy = 0.956, F1 Score = 0.966
#Logistic Regression (L1): Accuracy = 0.965, F1 Score = 0.973
#All models perform similarly.
