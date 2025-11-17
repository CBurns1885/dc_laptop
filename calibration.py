# calibration.py - STUB for DC-ONLY system
"""
Minimal stub for calibration classes - not used in DC-only system
Dixon-Coles probabilities are inherently calibrated from the statistical model
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DirichletCalibrator(BaseEstimator, TransformerMixin):
    """Stub Dirichlet calibrator - not used in DC-only"""
    def __init__(self, C=1.0, max_iter=2000):
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y):
        """Stub fit - does nothing"""
        return self

    def transform(self, X):
        """Stub transform - returns input unchanged"""
        return X

class TemperatureScaler(BaseEstimator, TransformerMixin):
    """Stub temperature scaler - not used in DC-only"""
    def __init__(self):
        self.temperature = 1.0

    def fit(self, X, y):
        """Stub fit - does nothing"""
        return self

    def transform(self, X):
        """Stub transform - returns softmax of input"""
        # Simple softmax
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)
