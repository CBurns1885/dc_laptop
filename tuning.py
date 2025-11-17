# tuning.py - STUB for DC-ONLY system
"""
Minimal stub for tuning functions - not used in DC-only system
These are only needed if using ML ensemble models
"""

import numpy as np
from typing import List, Callable
from dataclasses import dataclass

@dataclass
class CVData:
    """Stub class for cross-validation data"""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    classes: List[str]

class TimeSeriesSplit:
    """Simple time series split stub"""
    def __init__(self, n_samples: int, n_folds: int = 5):
        self.n_samples = n_samples
        self.n_folds = n_folds
        fold_size = n_samples // (n_folds + 1)
        self.test_fold = np.repeat(np.arange(n_folds), fold_size)[:n_samples]

def make_time_split(n_samples: int, n_folds: int = 5) -> TimeSeriesSplit:
    """Create time-series aware cross-validation split"""
    return TimeSeriesSplit(n_samples, n_folds)

def objective_factory(algorithm: str, cvd: CVData) -> Callable:
    """Stub factory for Optuna objectives - not used in DC-only"""
    def dummy_objective(trial):
        return 0.5  # Dummy accuracy
    return dummy_objective
