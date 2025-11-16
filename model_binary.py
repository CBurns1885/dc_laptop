# model_binary_LITE.py
# LAPTOP-FRIENDLY: Optimized for speed on basic hardware
# Binary markets: O/U, BTTS - Fast & lightweight

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class BinaryMarketModelLite:
    """
    Lightweight binary model for basic laptops.
    
    Key optimizations:
    - Fewer trees (100 vs 500)
    - Shallower depth (6 vs 12)
    - Only 2 models (RF + LR)
    - No calibration (faster)
    - Single-threaded to avoid CPU overload
    """
    
    def __init__(self, target_name: str, random_state: int = 42):
        self.target_name = target_name
        self.random_state = random_state
        self.models = {}
        self.is_fitted = False
        
    def _build_models(self) -> Dict:
        """Build minimal model set."""
        models = {}
        
        # Lightweight Random Forest
        models['rf'] = RandomForestClassifier(
            n_estimators=100,  # Much fewer trees
            max_depth=6,       # Much shallower
            min_samples_leaf=10,
            max_features='sqrt',
            n_jobs=1,          # Single thread
            random_state=self.random_state
        )
        
        # Logistic Regression (very fast)
        models['lr'] = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='lbfgs',
            max_iter=500,
            n_jobs=1,
            random_state=self.random_state
        )
        
        return models
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train models without calibration."""
        if len(np.unique(y)) != 2:
            raise ValueError(f"Binary model requires 2 classes")
        
        base_models = self._build_models()
        
        for name, model in base_models.items():
            try:
                model.fit(X, y)
                self.models[name] = model
                print(f"  ✓ {name.upper()} (binary-lite)")
            except Exception as e:
                print(f"  ✗ {name.upper()}: {e}")
        
        if not self.models:
            raise RuntimeError(f"No models trained")
        
        self.is_fitted = True
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Fast prediction."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        # Simple average (equal weight)
        preds = [m.predict_proba(X) for m in self.models.values()]
        return np.mean(preds, axis=0)


def is_binary_market(target_col: str) -> bool:
    """Check if binary market."""
    return any(target_col.startswith(p) for p in ['y_OU_', 'y_BTTS', 'y_AH_'])
