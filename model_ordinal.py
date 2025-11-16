# model_ordinal_LITE.py
# LAPTOP-FRIENDLY: Ultra-fast ordinal predictions
# Goal Range, CS - Minimal trees

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.ensemble import RandomForestClassifier

class OrdinalMarketModelLite:
    """
    Ultra-lightweight ordinal model for basic laptops.
    
    Key optimizations:
    - Very few trees (50 vs 200)
    - Very shallow (4 vs 10)
    - Single model (RF only)
    - Fast smoothing
    """
    
    def __init__(self, target_name: str, ordered_classes: List[str], random_state: int = 42):
        self.target_name = target_name
        self.ordered_classes = ordered_classes
        self.n_classes = len(ordered_classes)
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        
    def _build_model(self):
        """Build single lightweight model."""
        return RandomForestClassifier(
            n_estimators=50,   # Very few trees
            max_depth=4,       # Very shallow
            min_samples_leaf=20,
            max_features='sqrt',
            n_jobs=1,
            random_state=self.random_state
        )
    
    def _fast_smoothing(self, probs: np.ndarray) -> np.ndarray:
        """Fast adjacent smoothing."""
        n_samples, n_classes = probs.shape
        smoothed = probs.copy()
        
        # Simple: add 5% from neighbors
        for i in range(n_samples):
            for j in range(1, n_classes-1):
                smoothed[i, j] += 0.05 * (probs[i, j-1] + probs[i, j+1])
            
            # Edge cases
            smoothed[i, 0] += 0.05 * probs[i, 1]
            smoothed[i, -1] += 0.05 * probs[i, -2]
            
            # Normalize
            smoothed[i] /= smoothed[i].sum()
        
        return smoothed
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Ultra-fast training."""
        self.model = self._build_model()
        
        try:
            self.model.fit(X, y)
            print(f"  ✓ RF (ordinal-lite, {self.n_classes} classes)")
        except Exception as e:
            print(f"  ✗ RF: {e}")
            raise
        
        self.is_fitted = True
    
    def predict_proba(self, X: np.ndarray, smoothing: bool = True) -> np.ndarray:
        """Ultra-fast prediction."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        pred = self.model.predict_proba(X)
        
        if smoothing:
            pred = self._fast_smoothing(pred)
        
        return pred


def is_ordinal_market(target_col: str) -> bool:
    """Check if ordinal market."""
    return target_col in ['y_GOAL_RANGE', 'y_CS', 'y_HomeCardsY_BAND', 
                          'y_AwayCardsY_BAND', 'y_HomeCorners_BAND', 'y_AwayCorners_BAND']
