# calibration.py
"""
Calibration Module for Dixon-Coles Model
Analyzes and improves probability calibration

Features:
- Probability calibration curves (reliability diagrams)
- Platt scaling and isotonic regression
- Brier score decomposition
- Optimal threshold finding
- Per-league calibration analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CALIBRATION METRICS
# =============================================================================

@dataclass
class CalibrationMetrics:
    """Container for calibration analysis results"""
    market: str
    n_samples: int
    
    # Core metrics
    accuracy: float
    brier_score: float
    log_loss: float
    
    # Brier decomposition
    reliability: float      # Calibration component (lower = better calibrated)
    resolution: float       # Discrimination component (higher = better)
    uncertainty: float      # Base rate uncertainty
    
    # Calibration curve data
    bin_edges: List[float] = field(default_factory=list)
    bin_pred_means: List[float] = field(default_factory=list)
    bin_actual_means: List[float] = field(default_factory=list)
    bin_counts: List[int] = field(default_factory=list)
    
    # Optimal thresholds
    optimal_threshold: float = 0.5
    threshold_accuracy: float = 0.0
    
    # Expected Calibration Error
    ece: float = 0.0  # Expected Calibration Error
    mce: float = 0.0  # Maximum Calibration Error


def calculate_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate Brier score (mean squared error of probabilities)"""
    return np.mean((y_prob - y_true) ** 2)


def calculate_log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """Calculate log loss (cross-entropy)"""
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))


def brier_decomposition(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, float, float]:
    """
    Decompose Brier score into reliability, resolution, and uncertainty
    
    Brier = Reliability - Resolution + Uncertainty
    
    - Reliability: How well calibrated (0 = perfect calibration)
    - Resolution: How well it discriminates (higher = better)
    - Uncertainty: Base rate variance (constant for given data)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])
    
    base_rate = np.mean(y_true)
    uncertainty = base_rate * (1 - base_rate)
    
    reliability = 0.0
    resolution = 0.0
    
    for i in range(n_bins):
        mask = bin_indices == i
        n_k = np.sum(mask)
        
        if n_k > 0:
            avg_pred = np.mean(y_prob[mask])
            avg_actual = np.mean(y_true[mask])
            
            reliability += n_k * (avg_pred - avg_actual) ** 2
            resolution += n_k * (avg_actual - base_rate) ** 2
    
    n = len(y_true)
    reliability /= n
    resolution /= n
    
    return reliability, resolution, uncertainty


def calculate_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                                n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate calibration curve data
    
    Returns:
        bin_edges, bin_pred_means, bin_actual_means, bin_counts
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])
    
    bin_pred_means = []
    bin_actual_means = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        n_k = np.sum(mask)
        bin_counts.append(n_k)
        
        if n_k > 0:
            bin_pred_means.append(np.mean(y_prob[mask]))
            bin_actual_means.append(np.mean(y_true[mask]))
        else:
            bin_pred_means.append((bin_edges[i] + bin_edges[i+1]) / 2)
            bin_actual_means.append(np.nan)
    
    return bin_edges, np.array(bin_pred_means), np.array(bin_actual_means), np.array(bin_counts)


def calculate_ece_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
    """
    Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    
    ECE = weighted average of |predicted - actual| per bin
    MCE = maximum |predicted - actual| across bins
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])
    
    n = len(y_true)
    ece = 0.0
    mce = 0.0
    
    for i in range(n_bins):
        mask = bin_indices == i
        n_k = np.sum(mask)
        
        if n_k > 0:
            avg_pred = np.mean(y_prob[mask])
            avg_actual = np.mean(y_true[mask])
            gap = abs(avg_pred - avg_actual)
            
            ece += (n_k / n) * gap
            mce = max(mce, gap)
    
    return ece, mce


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, 
                           metric: str = 'accuracy') -> Tuple[float, float]:
    """
    Find optimal probability threshold for classification
    
    Args:
        y_true: Binary outcomes
        y_prob: Predicted probabilities
        metric: 'accuracy', 'f1', or 'youden' (Youden's J)
    
    Returns:
        optimal_threshold, metric_value
    """
    thresholds = np.arange(0.35, 0.70, 0.01)
    best_threshold = 0.5
    best_metric = 0.0
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        
        if metric == 'accuracy':
            score = np.mean(y_pred == y_true)
        elif metric == 'f1':
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        elif metric == 'youden':
            # Youden's J = Sensitivity + Specificity - 1
            tp = np.sum((y_pred == 1) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sensitivity + specificity - 1
        else:
            score = np.mean(y_pred == y_true)
        
        if score > best_metric:
            best_metric = score
            best_threshold = thresh
    
    return best_threshold, best_metric


def analyze_calibration(y_true: np.ndarray, y_prob: np.ndarray, 
                        market: str, n_bins: int = 10) -> CalibrationMetrics:
    """
    Full calibration analysis for a market
    
    Args:
        y_true: Binary outcomes (0/1)
        y_prob: Predicted probabilities
        market: Market name for labeling
        n_bins: Number of bins for calibration curve
    
    Returns:
        CalibrationMetrics with all analysis results
    """
    # Basic metrics
    y_pred = (y_prob >= 0.5).astype(int)
    accuracy = np.mean(y_pred == y_true)
    brier = calculate_brier_score(y_true, y_prob)
    logloss = calculate_log_loss(y_true, y_prob)
    
    # Brier decomposition
    reliability, resolution, uncertainty = brier_decomposition(y_true, y_prob, n_bins)
    
    # Calibration curve
    bin_edges, bin_pred, bin_actual, bin_counts = calculate_calibration_curve(y_true, y_prob, n_bins)
    
    # ECE/MCE
    ece, mce = calculate_ece_mce(y_true, y_prob, n_bins)
    
    # Optimal threshold
    opt_thresh, opt_acc = find_optimal_threshold(y_true, y_prob, metric='accuracy')
    
    return CalibrationMetrics(
        market=market,
        n_samples=len(y_true),
        accuracy=accuracy,
        brier_score=brier,
        log_loss=logloss,
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        bin_edges=bin_edges.tolist(),
        bin_pred_means=bin_pred.tolist(),
        bin_actual_means=[x if not np.isnan(x) else None for x in bin_actual],
        bin_counts=bin_counts.tolist(),
        optimal_threshold=opt_thresh,
        threshold_accuracy=opt_acc,
        ece=ece,
        mce=mce
    )


# =============================================================================
# CALIBRATION ADJUSTMENTS
# =============================================================================

class PlattScaler:
    """
    Platt scaling for probability calibration
    Fits a logistic regression to map raw probabilities to calibrated ones
    """
    
    def __init__(self):
        self.a = 0.0  # Slope
        self.b = 0.0  # Intercept
        self.fitted = False
    
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray, 
            lr: float = 0.1, n_iter: int = 1000):
        """
        Fit Platt scaling parameters using gradient descent
        
        Calibrated prob = 1 / (1 + exp(a * prob + b))
        """
        # Initialize
        self.a = 1.0
        self.b = 0.0
        
        for _ in range(n_iter):
            # Forward pass
            z = self.a * y_prob + self.b
            p = 1 / (1 + np.exp(-z))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            
            # Gradients
            error = p - y_true
            grad_a = np.mean(error * y_prob)
            grad_b = np.mean(error)
            
            # Update
            self.a -= lr * grad_a
            self.b -= lr * grad_b
        
        self.fitted = True
        return self
    
    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply Platt scaling to probabilities"""
        if not self.fitted:
            return y_prob
        z = self.a * y_prob + self.b
        return 1 / (1 + np.exp(-z))
    
    def fit_transform(self, y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(y_true, y_prob)
        return self.transform(y_prob)


class IsotonicCalibrator:
    """
    Isotonic regression for probability calibration
    Non-parametric, monotonic calibration
    """
    
    def __init__(self):
        self.x_vals = None
        self.y_vals = None
        self.fitted = False
    
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray):
        """Fit isotonic regression using pool adjacent violators"""
        # Sort by predicted probability
        order = np.argsort(y_prob)
        y_prob_sorted = y_prob[order]
        y_true_sorted = y_true[order]
        
        # Pool Adjacent Violators Algorithm (PAVA)
        n = len(y_true)
        y_calibrated = y_true_sorted.astype(float).copy()
        
        i = 0
        while i < n - 1:
            if y_calibrated[i] > y_calibrated[i + 1]:
                # Pool
                j = i + 1
                while j < n and y_calibrated[i] > y_calibrated[j]:
                    j += 1
                
                # Average the pool
                pool_mean = np.mean(y_calibrated[i:j])
                y_calibrated[i:j] = pool_mean
                
                # Step back to check previous
                if i > 0:
                    i -= 1
                else:
                    i = j
            else:
                i += 1
        
        # Store unique values for interpolation
        # Group by predicted prob and take mean calibrated value
        unique_probs, indices = np.unique(y_prob_sorted, return_inverse=True)
        unique_calibrated = np.array([y_calibrated[indices == i].mean() 
                                      for i in range(len(unique_probs))])
        
        self.x_vals = unique_probs
        self.y_vals = unique_calibrated
        self.fitted = True
        
        return self
    
    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration via interpolation"""
        if not self.fitted:
            return y_prob
        
        return np.interp(y_prob, self.x_vals, self.y_vals)
    
    def fit_transform(self, y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
        """Fit and transform"""
        self.fit(y_true, y_prob)
        return self.transform(y_prob)


class BinningCalibrator:
    """
    Simple binning calibration
    Maps probability bins to actual frequencies
    """
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.bin_edges = None
        self.bin_values = None
        self.fitted = False
    
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray):
        """Fit binning calibration"""
        self.bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(y_prob, self.bin_edges[1:-1])
        
        self.bin_values = []
        for i in range(self.n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                self.bin_values.append(np.mean(y_true[mask]))
            else:
                # Use bin midpoint as fallback
                self.bin_values.append((self.bin_edges[i] + self.bin_edges[i+1]) / 2)
        
        self.bin_values = np.array(self.bin_values)
        self.fitted = True
        return self
    
    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply binning calibration"""
        if not self.fitted:
            return y_prob
        
        bin_indices = np.digitize(y_prob, self.bin_edges[1:-1])
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        return self.bin_values[bin_indices]


# =============================================================================
# CALIBRATION ANALYSIS
# =============================================================================

class CalibrationAnalyzer:
    """
    Comprehensive calibration analysis for prediction model
    """
    
    def __init__(self):
        self.results: Dict[str, CalibrationMetrics] = {}
        self.league_results: Dict[str, Dict[str, CalibrationMetrics]] = {}
        self.calibrators: Dict[str, object] = {}
    
    def analyze(self, df: pd.DataFrame, markets: List[str] = None) -> Dict[str, CalibrationMetrics]:
        """
        Analyze calibration for all markets
        
        Args:
            df: DataFrame with prob_{market} and actual_{market} columns
            markets: List of markets to analyze (default: BTTS, OU_2_5)
        
        Returns:
            Dict mapping market name to CalibrationMetrics
        """
        markets = markets or ['BTTS', 'OU_2_5', 'OU_1_5', 'OU_3_5']
        
        for market in markets:
            prob_col = f'prob_{market}'
            actual_col = f'actual_{market}'
            
            if prob_col not in df.columns or actual_col not in df.columns:
                logger.warning(f"Missing columns for {market}")
                continue
            
            # Convert actuals to binary
            if market == 'BTTS':
                y_true = (df[actual_col] == 'Y').astype(int).values
            else:
                y_true = (df[actual_col] == 'O').astype(int).values
            
            y_prob = df[prob_col].values
            
            # Remove NaN
            mask = ~(np.isnan(y_prob) | np.isnan(y_true))
            y_true = y_true[mask]
            y_prob = y_prob[mask]
            
            if len(y_true) < 50:
                logger.warning(f"Insufficient data for {market}: {len(y_true)}")
                continue
            
            self.results[market] = analyze_calibration(y_true, y_prob, market)
        
        return self.results
    
    def analyze_by_league(self, df: pd.DataFrame, markets: List[str] = None) -> Dict[str, Dict[str, CalibrationMetrics]]:
        """
        Analyze calibration per league
        """
        markets = markets or ['BTTS', 'OU_2_5']
        
        for league in df['League'].unique():
            league_df = df[df['League'] == league]
            
            if len(league_df) < 50:
                continue
            
            self.league_results[league] = {}
            
            for market in markets:
                prob_col = f'prob_{market}'
                actual_col = f'actual_{market}'
                
                if prob_col not in league_df.columns:
                    continue
                
                if market == 'BTTS':
                    y_true = (league_df[actual_col] == 'Y').astype(int).values
                else:
                    y_true = (league_df[actual_col] == 'O').astype(int).values
                
                y_prob = league_df[prob_col].values
                
                mask = ~(np.isnan(y_prob) | np.isnan(y_true))
                y_true = y_true[mask]
                y_prob = y_prob[mask]
                
                if len(y_true) < 30:
                    continue
                
                self.league_results[league][market] = analyze_calibration(y_true, y_prob, f"{league}_{market}")
        
        return self.league_results
    
    def fit_calibrators(self, df: pd.DataFrame, method: str = 'isotonic',
                        markets: List[str] = None) -> Dict[str, object]:
        """
        Fit calibration models for each market
        
        Args:
            df: Training data with predictions
            method: 'platt', 'isotonic', or 'binning'
            markets: Markets to calibrate
        
        Returns:
            Dict of fitted calibrators
        """
        markets = markets or ['BTTS', 'OU_2_5']
        
        for market in markets:
            prob_col = f'prob_{market}'
            actual_col = f'actual_{market}'
            
            if prob_col not in df.columns:
                continue
            
            if market == 'BTTS':
                y_true = (df[actual_col] == 'Y').astype(int).values
            else:
                y_true = (df[actual_col] == 'O').astype(int).values
            
            y_prob = df[prob_col].values
            
            mask = ~(np.isnan(y_prob) | np.isnan(y_true))
            y_true = y_true[mask]
            y_prob = y_prob[mask]
            
            if len(y_true) < 50:
                continue
            
            if method == 'platt':
                calibrator = PlattScaler()
            elif method == 'isotonic':
                calibrator = IsotonicCalibrator()
            else:
                calibrator = BinningCalibrator()
            
            calibrator.fit(y_true, y_prob)
            self.calibrators[market] = calibrator
            
            logger.info(f"Fitted {method} calibrator for {market}")
        
        return self.calibrators
    
    def apply_calibration(self, df: pd.DataFrame, markets: List[str] = None) -> pd.DataFrame:
        """
        Apply fitted calibrators to predictions
        
        Adds {prob_col}_calibrated columns
        """
        df = df.copy()
        markets = markets or list(self.calibrators.keys())
        
        for market in markets:
            if market not in self.calibrators:
                continue
            
            prob_col = f'prob_{market}'
            if prob_col not in df.columns:
                continue
            
            calibrator = self.calibrators[market]
            df[f'{prob_col}_calibrated'] = calibrator.transform(df[prob_col].values)
        
        return df
    
    def get_optimal_thresholds(self) -> Dict[str, float]:
        """Get optimal thresholds for all markets"""
        return {market: metrics.optimal_threshold 
                for market, metrics in self.results.items()}
    
    def get_calibration_adjustments(self) -> Dict[str, Dict[str, float]]:
        """
        Get recommended probability adjustments per probability bin
        
        Returns adjustment factors: actual_rate / predicted_rate
        """
        adjustments = {}
        
        for market, metrics in self.results.items():
            market_adj = {}
            for i, (pred, actual) in enumerate(zip(metrics.bin_pred_means, metrics.bin_actual_means)):
                if actual is not None and pred > 0:
                    bin_label = f"{metrics.bin_edges[i]:.1f}-{metrics.bin_edges[i+1]:.1f}"
                    market_adj[bin_label] = actual / pred
            adjustments[market] = market_adj
        
        return adjustments
    
    def print_report(self):
        """Print calibration analysis report"""
        print("\n" + "="*70)
        print("CALIBRATION ANALYSIS REPORT")
        print("="*70)
        
        for market, metrics in self.results.items():
            print(f"\n{market}")
            print("-"*50)
            print(f"  Samples:          {metrics.n_samples:,}")
            print(f"  Accuracy:         {metrics.accuracy:.1%}")
            print(f"  Brier Score:      {metrics.brier_score:.4f}")
            print(f"  Log Loss:         {metrics.log_loss:.4f}")
            print(f"  ECE:              {metrics.ece:.4f}")
            print(f"  MCE:              {metrics.mce:.4f}")
            print(f"\n  Brier Decomposition:")
            print(f"    Reliability:    {metrics.reliability:.4f} (lower = better calibrated)")
            print(f"    Resolution:     {metrics.resolution:.4f} (higher = better discrimination)")
            print(f"    Uncertainty:    {metrics.uncertainty:.4f}")
            print(f"\n  Optimal Threshold: {metrics.optimal_threshold:.2f} (accuracy: {metrics.threshold_accuracy:.1%})")
            
            print(f"\n  Calibration Curve:")
            print(f"  {'Bin':<12} {'Predicted':>10} {'Actual':>10} {'Count':>8} {'Gap':>8}")
            for i in range(len(metrics.bin_pred_means)):
                pred = metrics.bin_pred_means[i]
                actual = metrics.bin_actual_means[i]
                count = metrics.bin_counts[i]
                if actual is not None:
                    gap = actual - pred
                    gap_str = f"{gap:+.3f}"
                else:
                    gap_str = "N/A"
                    actual = float('nan')
                bin_label = f"{metrics.bin_edges[i]:.1f}-{metrics.bin_edges[i+1]:.1f}"
                print(f"  {bin_label:<12} {pred:>10.3f} {actual:>10.3f} {count:>8} {gap_str:>8}")
    
    def export_results(self, path: Path):
        """Export calibration results to JSON"""
        export_data = {
            'overall': {market: {
                'accuracy': m.accuracy,
                'brier_score': m.brier_score,
                'log_loss': m.log_loss,
                'ece': m.ece,
                'mce': m.mce,
                'reliability': m.reliability,
                'resolution': m.resolution,
                'optimal_threshold': m.optimal_threshold,
                'calibration_curve': {
                    'bin_edges': m.bin_edges,
                    'predicted': m.bin_pred_means,
                    'actual': m.bin_actual_means,
                    'counts': m.bin_counts
                }
            } for market, m in self.results.items()},
            'by_league': {league: {market: {
                'accuracy': m.accuracy,
                'brier_score': m.brier_score,
                'ece': m.ece,
                'optimal_threshold': m.optimal_threshold
            } for market, m in markets.items()} 
            for league, markets in self.league_results.items()}
        }
        
        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported calibration results to {path}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_calibration_check(df: pd.DataFrame, market: str = 'BTTS') -> Dict:
    """
    Quick calibration check for a market
    
    Returns summary dict
    """
    prob_col = f'prob_{market}'
    actual_col = f'actual_{market}'
    
    if prob_col not in df.columns:
        return {'error': f'Missing {prob_col}'}
    
    if market == 'BTTS':
        y_true = (df[actual_col] == 'Y').astype(int).values
    else:
        y_true = (df[actual_col] == 'O').astype(int).values
    
    y_prob = df[prob_col].values
    
    mask = ~(np.isnan(y_prob) | np.isnan(y_true))
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    
    if len(y_true) < 50:
        return {'error': 'Insufficient data'}
    
    metrics = analyze_calibration(y_true, y_prob, market)
    
    return {
        'market': market,
        'n_samples': metrics.n_samples,
        'accuracy': metrics.accuracy,
        'brier_score': metrics.brier_score,
        'ece': metrics.ece,
        'optimal_threshold': metrics.optimal_threshold,
        'calibration_gap': metrics.reliability
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)
    
    n = 1000
    y_true = np.random.binomial(1, 0.5, n)
    
    # Simulated uncalibrated predictions (slightly overconfident)
    y_prob = np.clip(y_true * 0.6 + (1 - y_true) * 0.4 + np.random.normal(0, 0.15, n), 0.1, 0.9)
    
    print("Calibration Analysis Demo")
    print("="*60)
    
    metrics = analyze_calibration(y_true, y_prob, 'Demo')
    
    print(f"\nMetrics:")
    print(f"  Brier Score: {metrics.brier_score:.4f}")
    print(f"  ECE: {metrics.ece:.4f}")
    print(f"  Optimal Threshold: {metrics.optimal_threshold:.2f}")
    
    print(f"\nCalibration Curve:")
    for i in range(len(metrics.bin_pred_means)):
        pred = metrics.bin_pred_means[i]
        actual = metrics.bin_actual_means[i]
        if actual is not None:
            print(f"  Pred {pred:.2f} -> Actual {actual:.2f}")
    
    # Test calibrators
    print(f"\nTesting Calibrators:")
    
    platt = PlattScaler()
    platt.fit(y_true[:800], y_prob[:800])
    y_platt = platt.transform(y_prob[800:])
    print(f"  Platt Brier: {calculate_brier_score(y_true[800:], y_platt):.4f}")
    
    iso = IsotonicCalibrator()
    iso.fit(y_true[:800], y_prob[:800])
    y_iso = iso.transform(y_prob[800:])
    print(f"  Isotonic Brier: {calculate_brier_score(y_true[800:], y_iso):.4f}")
