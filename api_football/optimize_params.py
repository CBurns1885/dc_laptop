# optimize_params.py
"""
Parameter Optimization for Dixon-Coles Model
Grid search and optimization for:
- Rho per league
- Time decay
- xG weights
- Feature weights
- Calibration parameters

Uses walk-forward validation to avoid overfitting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import json
import itertools
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from models_dc_xg import fit_league_xg, price_match_xg, DCParamsXG, LEAGUE_CONFIG
from calibration import (
    calculate_brier_score, calculate_log_loss, analyze_calibration,
    CalibrationAnalyzer, PlattScaler, IsotonicCalibrator
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class OptimizationConfig:
    """Configuration for parameter optimization"""
    
    # Objective function
    objective: str = 'brier'  # 'brier', 'log_loss', 'accuracy', 'roi'
    
    # Markets to optimize
    markets: List[str] = field(default_factory=lambda: ['BTTS', 'OU_2_5'])
    
    # Validation settings
    n_folds: int = 5
    min_train_samples: int = 100
    
    # Parameter search spaces
    rho_range: Tuple[float, float, float] = (-0.05, 0.15, 0.02)  # min, max, step
    decay_range: Tuple[int, int, int] = (200, 500, 50)  # days
    xg_weight_range: Tuple[float, float, float] = (0.3, 0.9, 0.1)
    
    # Feature weight ranges
    rest_weight_range: Tuple[float, float, float] = (0.0, 0.3, 0.05)
    injury_weight_range: Tuple[float, float, float] = (0.0, 0.3, 0.05)
    formation_weight_range: Tuple[float, float, float] = (0.0, 0.2, 0.05)
    h2h_weight_range: Tuple[float, float, float] = (0.0, 0.3, 0.05)


@dataclass
class OptimizationResult:
    """Results from parameter optimization"""
    league: str
    market: str
    best_params: Dict
    best_score: float
    all_results: List[Dict]
    validation_scores: List[float]
    improvement: float  # vs baseline


# =============================================================================
# OBJECTIVE FUNCTIONS
# =============================================================================

def brier_objective(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score (lower is better) - negated for maximization"""
    return -calculate_brier_score(y_true, y_prob)


def log_loss_objective(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Log loss (lower is better) - negated for maximization"""
    return -calculate_log_loss(y_true, y_prob)


def accuracy_objective(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """Accuracy at given threshold"""
    y_pred = (y_prob >= threshold).astype(int)
    return np.mean(y_pred == y_true)


def roi_objective(y_true: np.ndarray, y_prob: np.ndarray, 
                  odds: float = 1.90, min_edge: float = 0.05) -> float:
    """
    Simulated ROI with edge-based betting
    Only bet when predicted edge > min_edge
    """
    implied_prob = 1 / odds
    edge = y_prob - implied_prob
    
    # Bet where edge > min_edge
    bet_mask = edge > min_edge
    if np.sum(bet_mask) == 0:
        return 0.0
    
    wins = np.sum(y_true[bet_mask])
    bets = np.sum(bet_mask)
    
    returns = wins * odds
    roi = (returns - bets) / bets
    
    return roi


OBJECTIVES = {
    'brier': brier_objective,
    'log_loss': log_loss_objective,
    'accuracy': accuracy_objective,
    'roi': roi_objective
}


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

def walk_forward_split(df: pd.DataFrame, n_folds: int = 5, 
                       min_train_pct: float = 0.5) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create walk-forward validation splits
    
    Each fold trains on all previous data and tests on next chunk
    """
    df = df.sort_values('Date')
    n = len(df)
    
    # Minimum training set
    min_train = int(n * min_train_pct)
    
    # Size of each test fold
    remaining = n - min_train
    fold_size = remaining // n_folds
    
    splits = []
    for i in range(n_folds):
        train_end = min_train + i * fold_size
        test_end = train_end + fold_size
        
        if test_end > n:
            test_end = n
        
        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]
        
        if len(test_df) > 0:
            splits.append((train_df, test_df))
    
    return splits


def time_series_cv(df: pd.DataFrame, n_splits: int = 5,
                   test_size: int = None) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Time series cross-validation with expanding window
    """
    df = df.sort_values('Date')
    n = len(df)
    
    if test_size is None:
        test_size = n // (n_splits + 1)
    
    min_train = n - (n_splits * test_size)
    
    splits = []
    for i in range(n_splits):
        train_end = min_train + i * test_size
        test_start = train_end
        test_end = test_start + test_size
        
        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end]
        
        splits.append((train_df, test_df))
    
    return splits


# =============================================================================
# SINGLE PARAMETER EVALUATION
# =============================================================================

def evaluate_params(train_df: pd.DataFrame, test_df: pd.DataFrame,
                    params: Dict, market: str, objective_fn: Callable) -> float:
    """
    Evaluate a parameter configuration
    
    Args:
        train_df: Training data
        test_df: Test data
        params: Parameter dict with rho, decay_days, xg_weight, etc.
        market: Market to evaluate
        objective_fn: Objective function
    
    Returns:
        Objective score (higher is better)
    """
    # Override league config temporarily
    league = train_df['League'].iloc[0]
    
    # Fit model with specified parameters
    try:
        # Create custom config
        custom_config = LEAGUE_CONFIG.get(league, {}).copy()
        custom_config['rho_init'] = params.get('rho', custom_config.get('rho_init', 0.05))
        custom_config['decay_days'] = params.get('decay_days', custom_config.get('decay_days', 365))
        custom_config['xg_weight'] = params.get('xg_weight', custom_config.get('xg_weight', 0.5))
        
        # Temporarily update config
        original_config = LEAGUE_CONFIG.get(league, {}).copy()
        LEAGUE_CONFIG[league] = custom_config
        
        # Fit model
        dc_params = fit_league_xg(train_df, use_xg=True)
        
        # Generate predictions on test set
        predictions = []
        actuals = []
        
        for _, row in test_df.iterrows():
            home = row['HomeTeam']
            away = row['AwayTeam']
            
            if home not in dc_params.attack or away not in dc_params.attack:
                continue
            
            # Build features
            features = {}
            
            # Apply feature weights from params
            if params.get('rest_weight', 0) > 0 and 'home_rest_days' in row:
                features['home_rest_days'] = row.get('home_rest_days')
                features['away_rest_days'] = row.get('away_rest_days')
            
            if params.get('injury_weight', 0) > 0 and 'home_injuries_count' in row:
                features['home_injuries_count'] = row.get('home_injuries_count')
                features['away_injuries_count'] = row.get('away_injuries_count')
            
            if params.get('formation_weight', 0) > 0 and 'home_formation_attack' in row:
                features['home_formation_attack'] = row.get('home_formation_attack')
                features['away_formation_attack'] = row.get('away_formation_attack')
            
            if params.get('h2h_weight', 0) > 0:
                features['h2h_total_goals_avg'] = row.get('h2h_total_goals_avg')
                features['h2h_btts_rate'] = row.get('h2h_btts_rate')
            
            # Get prediction
            probs = price_match_xg(dc_params, home, away, **features)
            
            if not probs:
                continue
            
            # Extract relevant probability
            if market == 'BTTS':
                prob = probs.get('DC_BTTS_Y', 0.5)
                actual = 1 if (row['FTHG'] > 0 and row['FTAG'] > 0) else 0
            else:
                line = market.replace('OU_', '').replace('_', '.')
                prob = probs.get(f'DC_OU_{market.split("_")[1]}_{market.split("_")[2]}_O', 0.5)
                total = row['FTHG'] + row['FTAG']
                actual = 1 if total > float(line) else 0
            
            predictions.append(prob)
            actuals.append(actual)
        
        # Restore original config
        LEAGUE_CONFIG[league] = original_config
        
        if len(predictions) < 20:
            return float('-inf')
        
        y_prob = np.array(predictions)
        y_true = np.array(actuals)
        
        return objective_fn(y_true, y_prob)
        
    except Exception as e:
        logger.debug(f"Evaluation failed: {e}")
        return float('-inf')


# =============================================================================
# GRID SEARCH
# =============================================================================

class ParameterOptimizer:
    """
    Grid search parameter optimization with walk-forward validation
    """
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.results: Dict[str, Dict[str, OptimizationResult]] = {}
        self.best_params: Dict[str, Dict[str, Dict]] = {}
    
    def _generate_param_grid(self, param_type: str = 'core') -> List[Dict]:
        """Generate parameter combinations to search"""
        cfg = self.config
        
        if param_type == 'core':
            # Core DC parameters
            rho_values = np.arange(*cfg.rho_range)
            decay_values = range(*cfg.decay_range)
            xg_values = np.arange(*cfg.xg_weight_range)
            
            grid = []
            for rho in rho_values:
                for decay in decay_values:
                    for xg in xg_values:
                        grid.append({
                            'rho': round(rho, 3),
                            'decay_days': decay,
                            'xg_weight': round(xg, 2)
                        })
            return grid
        
        elif param_type == 'features':
            # Feature weights
            rest_values = np.arange(*cfg.rest_weight_range)
            injury_values = np.arange(*cfg.injury_weight_range)
            formation_values = np.arange(*cfg.formation_weight_range)
            h2h_values = np.arange(*cfg.h2h_weight_range)
            
            grid = []
            for rest in rest_values:
                for injury in injury_values:
                    for formation in formation_values:
                        for h2h in h2h_values:
                            grid.append({
                                'rest_weight': round(rest, 2),
                                'injury_weight': round(injury, 2),
                                'formation_weight': round(formation, 2),
                                'h2h_weight': round(h2h, 2)
                            })
            return grid
        
        return []
    
    def optimize_league(self, df: pd.DataFrame, league: str, 
                        market: str, param_type: str = 'core') -> OptimizationResult:
        """
        Optimize parameters for a specific league and market
        
        Args:
            df: Full dataset
            league: League code
            market: Market to optimize
            param_type: 'core' or 'features'
        
        Returns:
            OptimizationResult
        """
        league_df = df[df['League'] == league].copy()
        
        if len(league_df) < self.config.min_train_samples:
            raise ValueError(f"Insufficient data for {league}: {len(league_df)}")
        
        objective_fn = OBJECTIVES[self.config.objective]
        param_grid = self._generate_param_grid(param_type)
        
        logger.info(f"Optimizing {league} {market}: {len(param_grid)} combinations")
        
        # Walk-forward splits
        splits = walk_forward_split(league_df, n_folds=self.config.n_folds)
        
        # Evaluate baseline
        baseline_scores = []
        for train_df, test_df in splits:
            baseline_score = evaluate_params(train_df, test_df, {}, market, objective_fn)
            if baseline_score > float('-inf'):
                baseline_scores.append(baseline_score)
        
        baseline_mean = np.mean(baseline_scores) if baseline_scores else 0
        
        # Grid search
        all_results = []
        best_score = float('-inf')
        best_params = {}
        
        for i, params in enumerate(param_grid):
            fold_scores = []
            
            for train_df, test_df in splits:
                score = evaluate_params(train_df, test_df, params, market, objective_fn)
                if score > float('-inf'):
                    fold_scores.append(score)
            
            if fold_scores:
                mean_score = np.mean(fold_scores)
                std_score = np.std(fold_scores)
                
                all_results.append({
                    'params': params,
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'n_folds': len(fold_scores)
                })
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
            
            if (i + 1) % 50 == 0:
                logger.info(f"  Progress: {i+1}/{len(param_grid)}")
        
        # Sort results
        all_results.sort(key=lambda x: x['mean_score'], reverse=True)
        
        improvement = best_score - baseline_mean
        
        result = OptimizationResult(
            league=league,
            market=market,
            best_params=best_params,
            best_score=best_score,
            all_results=all_results[:20],  # Top 20
            validation_scores=[r['mean_score'] for r in all_results[:5]],
            improvement=improvement
        )
        
        logger.info(f"  Best params: {best_params}")
        logger.info(f"  Best score: {best_score:.4f} (baseline: {baseline_mean:.4f}, improvement: {improvement:+.4f})")
        
        return result
    
    def optimize_all(self, df: pd.DataFrame, leagues: List[str] = None,
                     param_type: str = 'core') -> Dict[str, Dict[str, OptimizationResult]]:
        """
        Optimize parameters for all leagues and markets
        """
        leagues = leagues or df['League'].unique().tolist()
        
        for league in leagues:
            self.results[league] = {}
            
            for market in self.config.markets:
                try:
                    result = self.optimize_league(df, league, market, param_type)
                    self.results[league][market] = result
                    
                    # Store best params
                    if league not in self.best_params:
                        self.best_params[league] = {}
                    self.best_params[league][market] = result.best_params
                    
                except Exception as e:
                    logger.warning(f"Optimization failed for {league} {market}: {e}")
        
        return self.results
    
    def export_optimal_config(self, path: Path):
        """
        Export optimized parameters as Python config
        """
        config_lines = [
            "# Optimized League Configuration",
            f"# Generated: {datetime.now().isoformat()}",
            "",
            "OPTIMIZED_LEAGUE_CONFIG = {"
        ]
        
        for league, markets in self.best_params.items():
            # Merge market-specific params (use BTTS as primary)
            params = markets.get('BTTS', markets.get('OU_2_5', {}))
            
            config_lines.append(f"    '{league}': {{")
            config_lines.append(f"        'rho_init': {params.get('rho', 0.05)},")
            config_lines.append(f"        'rho_bounds': ({params.get('rho', 0.05) - 0.05}, {params.get('rho', 0.05) + 0.05}),")
            config_lines.append(f"        'decay_days': {params.get('decay_days', 365)},")
            config_lines.append(f"        'xg_weight': {params.get('xg_weight', 0.5)},")
            config_lines.append(f"    }},")
        
        config_lines.append("}")
        
        with open(path, 'w') as f:
            f.write('\n'.join(config_lines))
        
        logger.info(f"Exported optimized config to {path}")
    
    def export_results_json(self, path: Path):
        """Export full results as JSON"""
        export_data = {}
        
        for league, markets in self.results.items():
            export_data[league] = {}
            for market, result in markets.items():
                export_data[league][market] = {
                    'best_params': result.best_params,
                    'best_score': result.best_score,
                    'improvement': result.improvement,
                    'top_5': result.all_results[:5]
                }
        
        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported results to {path}")
    
    def print_summary(self):
        """Print optimization summary"""
        print("\n" + "="*70)
        print("PARAMETER OPTIMIZATION SUMMARY")
        print("="*70)
        
        for league, markets in self.results.items():
            print(f"\n{league}")
            print("-"*50)
            
            for market, result in markets.items():
                print(f"\n  {market}:")
                print(f"    Best Score: {result.best_score:.4f}")
                print(f"    Improvement: {result.improvement:+.4f}")
                print(f"    Best Params:")
                for k, v in result.best_params.items():
                    print(f"      {k}: {v}")


# =============================================================================
# BAYESIAN OPTIMIZATION (Optional, requires scipy)
# =============================================================================

def bayesian_optimize(df: pd.DataFrame, league: str, market: str,
                      n_calls: int = 50) -> Dict:
    """
    Bayesian optimization for parameters (more efficient than grid search)
    
    Requires: pip install scikit-optimize
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Real, Integer
    except ImportError:
        logger.warning("scikit-optimize not installed. Using grid search instead.")
        return {}
    
    league_df = df[df['League'] == league].copy()
    splits = walk_forward_split(league_df, n_folds=3)
    objective_fn = brier_objective
    
    def objective(params):
        rho, decay, xg_weight = params
        param_dict = {
            'rho': rho,
            'decay_days': int(decay),
            'xg_weight': xg_weight
        }
        
        scores = []
        for train_df, test_df in splits:
            score = evaluate_params(train_df, test_df, param_dict, market, objective_fn)
            if score > float('-inf'):
                scores.append(score)
        
        if scores:
            return -np.mean(scores)  # Minimize negative score
        return 1.0
    
    space = [
        Real(-0.05, 0.15, name='rho'),
        Integer(200, 500, name='decay'),
        Real(0.3, 0.9, name='xg_weight')
    ]
    
    result = gp_minimize(objective, space, n_calls=n_calls, random_state=42)
    
    return {
        'rho': result.x[0],
        'decay_days': result.x[1],
        'xg_weight': result.x[2],
        'best_score': -result.fun
    }


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def calculate_feature_importance(df: pd.DataFrame, league: str, 
                                  market: str) -> Dict[str, float]:
    """
    Calculate feature importance via ablation
    
    Returns importance score for each feature (higher = more important)
    """
    league_df = df[df['League'] == league].copy()
    splits = walk_forward_split(league_df, n_folds=3)
    objective_fn = brier_objective
    
    features = ['rest_weight', 'injury_weight', 'formation_weight', 'h2h_weight']
    
    # Baseline: all features on
    all_on_params = {f: 0.15 for f in features}
    baseline_scores = []
    for train_df, test_df in splits:
        score = evaluate_params(train_df, test_df, all_on_params, market, objective_fn)
        if score > float('-inf'):
            baseline_scores.append(score)
    baseline = np.mean(baseline_scores) if baseline_scores else 0
    
    # Test each feature off
    importance = {}
    for feature in features:
        test_params = all_on_params.copy()
        test_params[feature] = 0.0
        
        scores = []
        for train_df, test_df in splits:
            score = evaluate_params(train_df, test_df, test_params, market, objective_fn)
            if score > float('-inf'):
                scores.append(score)
        
        if scores:
            without_feature = np.mean(scores)
            # Importance = how much worse without this feature
            importance[feature] = baseline - without_feature
        else:
            importance[feature] = 0.0
    
    return importance


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parameter Optimization")
    parser.add_argument("--features", type=Path, required=True, help="Features parquet")
    parser.add_argument("--leagues", nargs="+", help="Leagues to optimize")
    parser.add_argument("--markets", nargs="+", default=['BTTS', 'OU_2_5'])
    parser.add_argument("--objective", choices=['brier', 'log_loss', 'accuracy'], default='brier')
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick optimization (fewer combinations)")
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_parquet(args.features)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Configure
    if args.quick:
        config = OptimizationConfig(
            objective=args.objective,
            markets=args.markets,
            n_folds=3,
            rho_range=(-0.03, 0.12, 0.05),
            decay_range=(250, 450, 100),
            xg_weight_range=(0.4, 0.8, 0.2)
        )
    else:
        config = OptimizationConfig(
            objective=args.objective,
            markets=args.markets
        )
    
    # Optimize
    optimizer = ParameterOptimizer(config)
    results = optimizer.optimize_all(df, leagues=args.leagues)
    
    # Output
    optimizer.print_summary()
    
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        optimizer.export_optimal_config(args.output / "optimized_config.py")
        optimizer.export_results_json(args.output / "optimization_results.json")
