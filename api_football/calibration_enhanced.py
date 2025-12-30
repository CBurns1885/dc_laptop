# calibration_enhanced.py
"""
Enhanced Calibration Methods for Maximum Accuracy

Advanced calibration techniques beyond basic Platt/Isotonic:
1. Beta Calibration - better for skewed distributions
2. Ensemble Calibration - combines multiple methods
3. Adaptive Bin Calibration - specialized per probability range
4. Temperature Scaling with optimization
5. Comprehensive evaluation metrics

Usage:
    from calibration_enhanced import EnhancedCalibrationSystem

    system = EnhancedCalibrationSystem()
    system.fit(y_true_train, y_prob_train)
    calibrated_probs = system.transform(y_prob_test)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy.optimize import minimize, minimize_scalar
from scipy.special import betaln, gammaln
from scipy.stats import beta as beta_dist

from calibration import (
    PlattScaler, IsotonicCalibrator, BinningCalibrator,
    calculate_brier_score, calculate_log_loss,
    calculate_ece_mce
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# BETA CALIBRATION
# =============================================================================

class BetaCalibrator:
    """
    Beta Calibration - generalizes Platt scaling

    Maps probabilities via Beta distribution for better handling of
    skewed probability distributions (common in football markets).

    Reference: Kull et al. (2017) "Beyond temperature scaling"
    """

    def __init__(self):
        self.a = 1.0  # Shape parameter alpha
        self.b = 1.0  # Shape parameter beta
        self.m = 0.0  # Location shift
        self.fitted = False

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray, max_iter: int = 100):
        """
        Fit Beta calibration parameters via maximum likelihood

        Args:
            y_true: Binary outcomes (0/1)
            y_prob: Predicted probabilities
            max_iter: Maximum iterations for optimization
        """
        def neg_log_likelihood(params):
            a, b, m = params

            # Constrain parameters
            if a <= 0.1 or b <= 0.1:
                return 1e10

            # Transform probabilities
            p_transformed = np.clip(y_prob + m, 0.01, 0.99)

            # Beta distribution log-likelihood
            ll = (
                (a - 1) * np.log(p_transformed + 1e-10) +
                (b - 1) * np.log(1 - p_transformed + 1e-10) -
                betaln(a, b)
            )

            # Weight by actual outcomes
            weighted_ll = y_true * ll + (1 - y_true) * ll

            return -np.sum(weighted_ll)

        # Optimize
        result = minimize(
            neg_log_likelihood,
            x0=[1.0, 1.0, 0.0],
            method='L-BFGS-B',
            bounds=[(0.1, 20), (0.1, 20), (-0.3, 0.3)],
            options={'maxiter': max_iter}
        )

        if result.success:
            self.a, self.b, self.m = result.x
            self.fitted = True
            logger.info(f"Beta calibration fitted: a={self.a:.3f}, b={self.b:.3f}, m={self.m:.3f}")
        else:
            logger.warning(f"Beta calibration failed to converge: {result.message}")
            self.a, self.b, self.m = 1.0, 1.0, 0.0
            self.fitted = False

        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply Beta calibration"""
        if not self.fitted:
            return y_prob

        p_transformed = np.clip(y_prob + self.m, 0.01, 0.99)
        return beta_dist.cdf(p_transformed, self.a, self.b)


# =============================================================================
# TEMPERATURE SCALING (ENHANCED)
# =============================================================================

class TemperatureScaler:
    """
    Temperature Scaling with learned temperature parameter

    Scales logits by optimal temperature T to minimize NLL.
    Simple but effective calibration method.

    Reference: Guo et al. (2017) "On Calibration of Modern Neural Networks"
    """

    def __init__(self):
        self.temperature = 1.0
        self.fitted = False

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray):
        """
        Fit optimal temperature parameter

        Args:
            y_true: Binary outcomes
            y_prob: Predicted probabilities (will be converted to logits)
        """
        # Convert probabilities to logits
        y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
        y_logits = np.log(y_prob_clipped / (1 - y_prob_clipped))

        def nll(temp):
            """Negative log-likelihood as function of temperature"""
            if temp <= 0.1:
                return 1e10

            # Apply temperature scaling
            scaled_logits = y_logits / temp
            probs = 1 / (1 + np.exp(-scaled_logits))
            probs = np.clip(probs, 1e-10, 1 - 1e-10)

            # NLL
            return -np.mean(
                y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs)
            )

        # Optimize temperature
        result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')

        if result.success:
            self.temperature = result.x
            self.fitted = True
            logger.info(f"Temperature scaling fitted: T={self.temperature:.3f}")
        else:
            self.temperature = 1.0
            self.fitted = False

        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply temperature scaling"""
        if not self.fitted or self.temperature == 1.0:
            return y_prob

        # Convert to logits
        y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
        y_logits = np.log(y_prob_clipped / (1 - y_prob_clipped))

        # Scale and convert back
        scaled_logits = y_logits / self.temperature
        return 1 / (1 + np.exp(-scaled_logits))


# =============================================================================
# ENSEMBLE CALIBRATION
# =============================================================================

class EnsembleCalibrator:
    """
    Ensemble calibration combining multiple methods

    Learns optimal weights for combining:
    - Isotonic Regression (non-parametric, monotonic)
    - Platt Scaling (parametric, logistic)
    - Beta Calibration (parametric, flexible)

    Uses validation set performance to determine weights.
    """

    def __init__(self, methods: List[str] = None):
        """
        Args:
            methods: List of methods to ensemble. Options:
                     'isotonic', 'platt', 'beta', 'temperature'
        """
        self.methods = methods or ['isotonic', 'platt', 'beta']
        self.calibrators = {}
        self.weights = {}
        self.fitted = False

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray,
            validation_split: float = 0.3):
        """
        Fit ensemble of calibrators with learned weights

        Args:
            y_true: Binary outcomes
            y_prob: Predicted probabilities
            validation_split: Fraction of data for weight learning
        """
        # Split into train/validation
        n = len(y_true)
        n_val = int(n * validation_split)
        indices = np.random.permutation(n)

        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        # Initialize calibrators
        calibrator_classes = {
            'isotonic': IsotonicCalibrator,
            'platt': PlattScaler,
            'beta': BetaCalibrator,
            'temperature': TemperatureScaler
        }

        # Fit each calibrator on training set
        for method in self.methods:
            if method not in calibrator_classes:
                logger.warning(f"Unknown calibration method: {method}")
                continue

            calibrator = calibrator_classes[method]()
            try:
                calibrator.fit(y_true[train_idx], y_prob[train_idx])
                self.calibrators[method] = calibrator
            except Exception as e:
                logger.warning(f"Failed to fit {method}: {e}")

        if not self.calibrators:
            logger.error("No calibrators fitted successfully")
            return self

        # Evaluate on validation set and learn weights
        val_true = y_true[val_idx]
        val_prob = y_prob[val_idx]

        scores = {}
        for method, calibrator in self.calibrators.items():
            try:
                cal_prob = calibrator.transform(val_prob)
                # Use Brier score (lower is better)
                scores[method] = calculate_brier_score(val_true, cal_prob)
            except Exception as e:
                logger.warning(f"Failed to evaluate {method}: {e}")
                scores[method] = 1.0  # Worst possible Brier score

        # Convert scores to weights via softmax of inverse scores
        inv_scores = {k: 1 / (v + 1e-6) for k, v in scores.items()}
        total = sum(inv_scores.values())
        self.weights = {k: v / total for k, v in inv_scores.items()}

        logger.info("Ensemble calibration fitted:")
        for method in self.methods:
            if method in self.weights:
                logger.info(f"  {method}: weight={self.weights[method]:.3f}, "
                          f"brier={scores[method]:.4f}")

        self.fitted = True
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply ensemble calibration"""
        if not self.fitted or not self.calibrators:
            return y_prob

        # Weighted average of calibrated predictions
        calibrated = np.zeros_like(y_prob, dtype=float)

        for method, calibrator in self.calibrators.items():
            try:
                weight = self.weights.get(method, 0.0)
                if weight > 0:
                    calibrated += weight * calibrator.transform(y_prob)
            except Exception as e:
                logger.warning(f"Failed to transform with {method}: {e}")

        return np.clip(calibrated, 0.0, 1.0)


# =============================================================================
# ADAPTIVE BIN CALIBRATION
# =============================================================================

class AdaptiveBinCalibrator:
    """
    Adaptive calibration using different methods per probability bin

    Rationale: Calibration quality varies by probability range.
    E.g., model may be well-calibrated at 50% but poorly at 90%.

    Uses best-performing calibrator for each probability bin.
    """

    def __init__(self, bin_edges: List[float] = None):
        """
        Args:
            bin_edges: Bin boundaries (default: [0, 0.3, 0.7, 1.0])
        """
        self.bin_edges = bin_edges or [0.0, 0.3, 0.7, 1.0]
        self.calibrators = []
        self.calibrator_names = []
        self.fitted = False

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray,
            min_samples: int = 20):
        """
        Fit optimal calibrator for each probability bin

        Args:
            y_true: Binary outcomes
            y_prob: Predicted probabilities
            min_samples: Minimum samples required per bin
        """
        n_bins = len(self.bin_edges) - 1
        self.calibrators = []
        self.calibrator_names = []

        # Candidate calibrators
        candidates = {
            'isotonic': IsotonicCalibrator,
            'platt': PlattScaler,
            'beta': BetaCalibrator,
            'binning': BinningCalibrator
        }

        for i in range(n_bins):
            bin_start = self.bin_edges[i]
            bin_end = self.bin_edges[i + 1]

            # Get data in this bin
            if i == n_bins - 1:  # Last bin includes upper edge
                mask = (y_prob >= bin_start) & (y_prob <= bin_end)
            else:
                mask = (y_prob >= bin_start) & (y_prob < bin_end)

            bin_count = np.sum(mask)

            if bin_count < min_samples:
                # Not enough data, use identity (no calibration)
                self.calibrators.append(None)
                self.calibrator_names.append('none')
                logger.info(f"Bin [{bin_start:.1f}, {bin_end:.1f}): "
                          f"insufficient data ({bin_count} samples)")
                continue

            bin_true = y_true[mask]
            bin_prob = y_prob[mask]

            # Try each calibrator, keep best
            best_calibrator = None
            best_name = 'none'
            best_score = float('inf')

            for name, cal_class in candidates.items():
                try:
                    # Split bin data for evaluation
                    n_train = int(len(bin_true) * 0.7)

                    calibrator = cal_class()
                    calibrator.fit(bin_true[:n_train], bin_prob[:n_train])

                    # Evaluate on remaining data
                    if len(bin_true) - n_train >= 5:
                        cal_prob = calibrator.transform(bin_prob[n_train:])
                        score = calculate_brier_score(bin_true[n_train:], cal_prob)
                    else:
                        # Use training score if not enough validation data
                        cal_prob = calibrator.transform(bin_prob[:n_train])
                        score = calculate_brier_score(bin_true[:n_train], cal_prob)

                    if score < best_score:
                        best_score = score
                        best_calibrator = calibrator
                        best_name = name
                except Exception as e:
                    logger.debug(f"Failed {name} for bin [{bin_start:.1f}, {bin_end:.1f}): {e}")
                    continue

            self.calibrators.append(best_calibrator)
            self.calibrator_names.append(best_name)

            logger.info(f"Bin [{bin_start:.1f}, {bin_end:.1f}): "
                      f"using {best_name} (n={bin_count}, brier={best_score:.4f})")

        self.fitted = True
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply adaptive calibration"""
        if not self.fitted:
            return y_prob

        calibrated = y_prob.copy()
        n_bins = len(self.bin_edges) - 1

        for i in range(n_bins):
            bin_start = self.bin_edges[i]
            bin_end = self.bin_edges[i + 1]

            if i == n_bins - 1:
                mask = (y_prob >= bin_start) & (y_prob <= bin_end)
            else:
                mask = (y_prob >= bin_start) & (y_prob < bin_end)

            if self.calibrators[i] is not None and np.sum(mask) > 0:
                try:
                    calibrated[mask] = self.calibrators[i].transform(y_prob[mask])
                except Exception as e:
                    logger.warning(f"Failed to calibrate bin {i}: {e}")

        return np.clip(calibrated, 0.0, 1.0)


# =============================================================================
# ENHANCED CALIBRATION SYSTEM
# =============================================================================

class EnhancedCalibrationSystem:
    """
    Complete calibration system with automatic method selection

    Automatically selects best calibration approach:
    1. If dataset large enough (>500): Use Ensemble
    2. If skewed distribution: Prefer Beta
    3. If small dataset: Use Temperature
    4. Apply Adaptive Bin for extreme probabilities

    Usage:
        system = EnhancedCalibrationSystem()
        system.fit(y_true_train, y_prob_train)
        calibrated = system.transform(y_prob_test)
    """

    def __init__(self, auto_select: bool = True):
        """
        Args:
            auto_select: Automatically select best method
        """
        self.auto_select = auto_select
        self.primary_calibrator = None
        self.adaptive_calibrator = None
        self.use_adaptive = False
        self.method_name = None
        self.fitted = False

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray,
            force_method: str = None):
        """
        Fit calibration system

        Args:
            y_true: Binary outcomes
            y_prob: Predicted probabilities
            force_method: Force specific method ('ensemble', 'beta', 'isotonic', etc.)
        """
        n = len(y_true)

        # Analyze distribution
        base_rate = np.mean(y_true)
        prob_std = np.std(y_prob)
        is_skewed = abs(base_rate - 0.5) > 0.15

        logger.info(f"Calibration dataset: n={n}, base_rate={base_rate:.3f}, "
                  f"prob_std={prob_std:.3f}, skewed={is_skewed}")

        # Select method
        if force_method:
            method = force_method
        elif n >= 500:
            method = 'ensemble'
        elif is_skewed:
            method = 'beta'
        elif n >= 100:
            method = 'isotonic'
        else:
            method = 'temperature'

        self.method_name = method
        logger.info(f"Selected calibration method: {method}")

        # Fit primary calibrator
        if method == 'ensemble':
            self.primary_calibrator = EnsembleCalibrator()
        elif method == 'beta':
            self.primary_calibrator = BetaCalibrator()
        elif method == 'isotonic':
            self.primary_calibrator = IsotonicCalibrator()
        elif method == 'platt':
            self.primary_calibrator = PlattScaler()
        elif method == 'temperature':
            self.primary_calibrator = TemperatureScaler()
        else:
            raise ValueError(f"Unknown method: {method}")

        self.primary_calibrator.fit(y_true, y_prob)

        # Fit adaptive calibrator for extreme probabilities if enough data
        if n >= 200:
            self.use_adaptive = True
            self.adaptive_calibrator = AdaptiveBinCalibrator(
                bin_edges=[0.0, 0.25, 0.75, 1.0]
            )
            self.adaptive_calibrator.fit(y_true, y_prob)
        else:
            self.use_adaptive = False

        self.fitted = True
        logger.info("Calibration system fitted successfully")

        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply calibration"""
        if not self.fitted:
            logger.warning("Calibration system not fitted, returning original probabilities")
            return y_prob

        # Primary calibration
        calibrated = self.primary_calibrator.transform(y_prob)

        # Apply adaptive calibration for extremes
        if self.use_adaptive and self.adaptive_calibrator:
            # Only apply adaptive to probabilities < 0.25 or > 0.75
            extreme_mask = (calibrated < 0.25) | (calibrated > 0.75)
            if np.sum(extreme_mask) > 0:
                calibrated[extreme_mask] = self.adaptive_calibrator.transform(
                    calibrated[extreme_mask]
                )

        return np.clip(calibrated, 0.0, 1.0)

    def evaluate(self, y_true: np.ndarray, y_prob_original: np.ndarray) -> Dict:
        """
        Evaluate calibration quality

        Returns metrics before and after calibration
        """
        # Original metrics
        brier_orig = calculate_brier_score(y_true, y_prob_original)
        logloss_orig = calculate_log_loss(y_true, y_prob_original)
        ece_orig, mce_orig = calculate_ece_mce(y_true, y_prob_original)

        # Calibrated metrics
        y_prob_calibrated = self.transform(y_prob_original)
        brier_cal = calculate_brier_score(y_true, y_prob_calibrated)
        logloss_cal = calculate_log_loss(y_true, y_prob_calibrated)
        ece_cal, mce_cal = calculate_ece_mce(y_true, y_prob_calibrated)

        return {
            'method': self.method_name,
            'n_samples': len(y_true),
            'original': {
                'brier': brier_orig,
                'log_loss': logloss_orig,
                'ece': ece_orig,
                'mce': mce_orig
            },
            'calibrated': {
                'brier': brier_cal,
                'log_loss': logloss_cal,
                'ece': ece_cal,
                'mce': mce_cal
            },
            'improvement': {
                'brier': brier_orig - brier_cal,
                'log_loss': logloss_orig - logloss_cal,
                'ece': ece_orig - ece_cal,
                'mce': mce_orig - mce_cal
            }
        }


# =============================================================================
# TESTING & DEMO
# =============================================================================

if __name__ == "__main__":
    # Generate synthetic test data
    np.random.seed(42)
    n = 1000

    # Simulate uncalibrated predictions (overconfident)
    y_true = np.random.binomial(1, 0.5, n)
    y_prob_raw = y_true * 0.7 + (1 - y_true) * 0.3 + np.random.normal(0, 0.15, n)
    y_prob = np.clip(y_prob_raw, 0.05, 0.95)

    # Split train/test
    train_size = 700
    y_true_train = y_true[:train_size]
    y_prob_train = y_prob[:train_size]
    y_true_test = y_true[train_size:]
    y_prob_test = y_prob[train_size:]

    print("="*70)
    print("ENHANCED CALIBRATION SYSTEM DEMO")
    print("="*70)

    # Test Enhanced System
    print("\n[1] Testing Enhanced Calibration System (Auto-Select)")
    print("-"*70)

    system = EnhancedCalibrationSystem(auto_select=True)
    system.fit(y_true_train, y_prob_train)

    results = system.evaluate(y_true_test, y_prob_test)

    print(f"\nMethod: {results['method']}")
    print(f"Samples: {results['n_samples']}")
    print("\nMetrics:")
    print(f"  {'Metric':<12} {'Original':>12} {'Calibrated':>12} {'Improvement':>12}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for metric in ['brier', 'log_loss', 'ece', 'mce']:
        orig = results['original'][metric]
        cal = results['calibrated'][metric]
        imp = results['improvement'][metric]
        print(f"  {metric:<12} {orig:>12.4f} {cal:>12.4f} {imp:>+12.4f}")

    # Test Individual Methods
    print("\n[2] Comparing Individual Methods")
    print("-"*70)

    methods = {
        'Beta': BetaCalibrator(),
        'Isotonic': IsotonicCalibrator(),
        'Temperature': TemperatureScaler(),
        'Ensemble': EnsembleCalibrator(),
        'Adaptive': AdaptiveBinCalibrator()
    }

    comparison = []
    for name, calibrator in methods.items():
        try:
            calibrator.fit(y_true_train, y_prob_train)
            y_prob_cal = calibrator.transform(y_prob_test)

            ece, mce = calculate_ece_mce(y_true_test, y_prob_cal)
            brier = calculate_brier_score(y_true_test, y_prob_cal)

            comparison.append({
                'Method': name,
                'ECE': ece,
                'MCE': mce,
                'Brier': brier
            })
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    if comparison:
        df_comparison = pd.DataFrame(comparison).sort_values('ECE')
        print("\n" + df_comparison.to_string(index=False))

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
