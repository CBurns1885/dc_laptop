# ordinal.py - STUB for DC-ONLY system
"""
Minimal stub for ordinal regression - not used in DC-only system
BTTS and O/U are binary markets, not ordinal
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class CORALOrdinal(BaseEstimator, ClassifierMixin):
    """Stub CORAL ordinal regression classifier - not used in DC-only"""
    def __init__(self, C=1.0, max_iter=2000):
        self.C = C
        self.max_iter = max_iter
        self.classes_ = None

    def fit(self, X, y):
        """Stub fit - does minimal initialization"""
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        return self

    def predict_proba(self, X):
        """Stub predict_proba - returns uniform distribution"""
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        n_classes = self.n_classes_ if self.n_classes_ else 2
        # Return uniform probabilities
        return np.ones((n_samples, n_classes)) / n_classes

    def predict(self, X):
        """Stub predict - returns first class"""
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        return np.repeat(self.classes_[0], n_samples)
