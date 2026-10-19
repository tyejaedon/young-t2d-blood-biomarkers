"""
Supervised Classifiers Configuration & Pipeline Factory
Defines standardized, reproducible classifiers:
- Random Forest Classifier (RF)
- Support Vector Machine with Linear Kernel (SVM-Linear)
- Support Vector Machine with Radial Basis Function Kernel (SVM-RBF)
- Logistic Regression with L2 Regularization (LR)
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin


class CalibratedSVM(BaseEstimator, ClassifierMixin):
    """
    Wrapper around SVC with Platt scaling calibration to provide robust,
    unbiased posterior probability estimates without deprecation warnings.
    """
    def __init__(self, kernel: str = "linear", C: float = 1.0, random_state: int = 42):
        self.kernel = kernel
        self.C = C
        self.random_state = random_state
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        base_svc = SVC(kernel=self.kernel, C=self.C, random_state=self.random_state)
        # Use cv=2 or cv=3 depending on min class count
        min_class_count = min(np.sum(y == 0), np.sum(y == 1))
        cv_splits = min(3, max(2, min_class_count))
        self.model = CalibratedClassifierCV(estimator=base_svc, cv=cv_splits)
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


def get_baseline_classifiers(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns the standard suite of supervised machine learning classifiers
    configured with fixed random seeds and regularized hyperparameters
    suited for low sample size, high-dimensional biological data.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            penalty="l2",
            C=0.5,
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=random_state
        ),
        "SVM (Linear)": CalibratedSVM(
            kernel="linear",
            C=0.5,
            random_state=random_state
        ),
        "SVM (RBF)": CalibratedSVM(
            kernel="rbf",
            C=1.0,
            random_state=random_state
        )
    }
    return models


def fit_and_predict_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fits an estimator on training data and predicts binary classes and
    continuous probabilities on test data.
    """
    m = model.__class__(**model.get_params())
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    probs = m.predict_proba(X_test)[:, 1]
    return preds, probs
