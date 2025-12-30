# backtest_api.py
"""
Backtesting Module for API-Football Enhanced Model
Tests xG, injuries, formations, cup competitions and all new features

Run backtests to find:
- Which features improve accuracy
- Optimal feature weights
- Cup vs league performance differences
- xG vs raw goals comparison
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from collections import defaultdict

from models_dc_xg import fit_league_xg, price_match_xg, DCParamsXG, LEAGUE_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"
RESULTS_DIR = Path("backtest_results")


# =============================================================================
# BACKTEST CONFIGURATION
# =============================================================================

@dataclass
class BacktestConfig:
    """Configuration for backtest run"""
    name: str = "default"
    
    # Data settings
    train_seasons: int = 2          # Seasons for initial training
    test_start_date: str = None     # Start of test period
    test_end_date: str = None       # End of test period
    
    # Feature flags to test
    use_xg: bool = True
    use_injuries: bool = True
    use_formations: bool = True
    use_h2h: bool = True
    use_rest_days: bool = True
    use_advanced_stats: bool = True
    
    # Markets to evaluate
    markets: List[str] = None
    
    # Leagues to include
    leagues: List[str] = None
    include_cups: bool = True
    
    # Rolling window
    retrain_frequency: int = 50     # Retrain every N matches
    min_train_matches: int = 100    # Minimum matches for training
    
    def __post_init__(self):
        if self.markets is None:
            self.markets = ['BTTS', 'OU_2_5', 'OU_1_5', 'OU_3_5']
        if self.leagues is None:
            self.leagues = ['E0', 'D1', 'SP1', 'I1', 'F1', 'E1', 'FA_CUP', 'UCL']


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(predictions: pd.DataFrame, market: str) -> Dict:
    """
    Calculate accuracy metrics for a market
    
    Args:
        predictions: DataFrame with pred_{market} and actual_{market} columns
        market: Market name (e.g., 'BTTS', 'OU_2_5')
    
    Returns:
        Dict of metrics
    """
    pred_col = f'pred_{market}'
    actual_col = f'actual_{market}'
    prob_col = f'prob_{market}'
    
    if pred_col not in predictions.columns or actual_col not in predictions.columns:
        return {}
    
    df = predictions.dropna(subset=[pred_col, actual_col])
    
    if len(df) == 0:
        return {}
    
    # Basic accuracy
    correct = (df[pred_col] == df[actual_col]).sum()
    total = len(df)
    accuracy = correct / total if total > 0 else 0
    
    # Brier score (if probabilities available)
    brier = None
    if prob_col in df.columns:
        actual_binary = (df[actual_col] == 'Y').astype(int) if market == 'BTTS' else (df[actual_col] == 'O').astype(int)
        brier = ((df[prob_col] - actual_binary) ** 2).mean()
    
    # ROI simulation (flat stakes)
    # Assume 1.90 odds for each side
    roi = None
    if total > 0:
        wins = correct
        stake = total
        returns = wins * 1.90
        roi = (returns - stake) / stake
    
    # Confidence-weighted accuracy (high confidence bets)
    high_conf_acc = None
    if prob_col in df.columns:
        high_conf = df[df[prob_col] > 0.55]
        if len(high_conf) > 0:
            high_correct = (high_conf[pred_col] == high_conf[actual_col]).sum()
            high_conf_acc = high_correct / len(high_conf)
    
    # Calibration (predicted prob vs actual rate in buckets)
    calibration = {}
    if prob_col in df.columns:
        df['prob_bucket'] = pd.cut(df[prob_col], bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0])
        for bucket, group in df.groupby('prob_bucket', observed=True):
            if len(group) > 10:
                actual_rate = (group[actual_col] == 'Y').mean() if market == 'BTTS' else (group[actual_col] == 'O').mean()
                calibration[str(bucket)] = {
                    'predicted': group[prob_col].mean(),
                    'actual': actual_rate,
                    'count': len(group)
                }
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'brier_score': brier,
        'roi_flat': roi,
        'high_conf_accuracy': high_conf_acc,
        'calibration': calibration
    }


def calculate_league_metrics(predictions: pd.DataFrame, market: str) -> Dict[str, Dict]:
    """Calculate metrics broken down by league"""
    results = {}
    
    for league in predictions['League'].unique():
        league_preds = predictions[predictions['League'] == league]
        results[league] = calculate_metrics(league_preds, market)
    
    return results


def calculate_cup_vs_league_metrics(predictions: pd.DataFrame, market: str) -> Dict:
    """Compare cup vs league accuracy"""
    if 'LeagueType' not in predictions.columns:
        return {}
    
    league_preds = predictions[predictions['LeagueType'] == 'league']
    cup_preds = predictions[predictions['LeagueType'] == 'cup']
    
    return {
        'league': calculate_metrics(league_preds, market),
        'cup': calculate_metrics(cup_preds, market)
    }


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class Backtester:
    """
    Rolling window backtester for DC model
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.results = []
        self.params_cache = {}
    
    def load_data(self, path: Path = FEATURES_PARQUET) -> pd.DataFrame:
        """Load features data"""
        df = pd.read_parquet(path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Filter leagues
        if self.config.leagues:
            if not self.config.include_cups:
                # Exclude cup competitions
                df = df[df['LeagueType'] != 'cup']
            df = df[df['League'].isin(self.config.leagues)]
        
        logger.info(f"Loaded {len(df)} matches from {df['Date'].min()} to {df['Date'].max()}")
        logger.info(f"Leagues: {df['League'].nunique()}, Cups: {(df['LeagueType'] == 'cup').sum()}")
        
        return df
    
    def _add_actuals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add actual outcome columns"""
        df = df.copy()
        
        # BTTS
        df['actual_BTTS'] = np.where(
            (df['FTHG'] > 0) & (df['FTAG'] > 0), 'Y', 'N'
        )
        
        # Over/Under
        total = df['FTHG'] + df['FTAG']
        for line in [1.5, 2.5, 3.5]:
            col = f"actual_OU_{str(line).replace('.', '_')}"
            df[col] = np.where(total > line, 'O', 'U')
        
        return df
    
    def _prepare_features(self, row: pd.Series) -> Dict:
        """Extract features from a row for prediction"""
        features = {}
        
        # xG features
        if self.config.use_xg:
            for col in ['home_xG_for_ma5', 'away_xG_for_ma5', 
                       'home_xG_overperformance', 'away_xG_overperformance']:
                if col in row.index and pd.notna(row[col]):
                    features[col] = row[col]
        
        # Rest days
        if self.config.use_rest_days:
            if 'home_rest_days' in row.index and pd.notna(row['home_rest_days']):
                features['home_rest_days'] = int(row['home_rest_days'])
            if 'away_rest_days' in row.index and pd.notna(row['away_rest_days']):
                features['away_rest_days'] = int(row['away_rest_days'])
            if 'home_had_cup_midweek' in row.index:
                features['home_had_cup_midweek'] = int(row['home_had_cup_midweek'])
            if 'away_had_cup_midweek' in row.index:
                features['away_had_cup_midweek'] = int(row['away_had_cup_midweek'])
        
        # Injuries
        if self.config.use_injuries:
            for col in ['home_injuries_count', 'away_injuries_count',
                       'home_key_injuries', 'away_key_injuries']:
                if col in row.index and pd.notna(row[col]):
                    features[col] = int(row[col])
        
        # Formations
        if self.config.use_formations:
            for col in ['home_formation_attack', 'away_formation_attack',
                       'formation_matchup_goals_mult']:
                if col in row.index and pd.notna(row[col]):
                    features[col] = row[col]
        
        # H2H
        if self.config.use_h2h:
            for col in ['h2h_total_goals_avg', 'h2h_btts_rate']:
                if col in row.index and pd.notna(row[col]):
                    features[col] = row[col]
        
        # Advanced stats
        if self.config.use_advanced_stats:
            for col in ['home_attack_quality', 'away_attack_quality',
                       'home_shot_accuracy_ma5', 'away_shot_accuracy_ma5']:
                if col in row.index and pd.notna(row[col]):
                    features[col] = row[col]
        
        # Cup indicator
        if 'is_cup_match' in row.index:
            features['is_cup_match'] = int(row['is_cup_match'])
        if 'is_knockout' in row.index:
            features['is_knockout'] = int(row['is_knockout'])
        
        return features
    
    def run(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Run rolling window backtest
        
        Returns:
            DataFrame with predictions and actuals
        """
        if df is None:
            df = self.load_data()
        
        df = self._add_actuals(df)
        
        # Determine test period
        if self.config.test_start_date:
            test_start = pd.to_datetime(self.config.test_start_date)
        else:
            # Use last 30% of data for testing
            test_start = df['Date'].quantile(0.7)
        
        if self.config.test_end_date:
            test_end = pd.to_datetime(self.config.test_end_date)
        else:
            test_end = df['Date'].max()
        
        test_df = df[(df['Date'] >= test_start) & (df['Date'] <= test_end)]
        
        logger.info(f"Test period: {test_start.date()} to {test_end.date()}")
        logger.info(f"Test matches: {len(test_df)}")
        
        predictions = []
        matches_since_retrain = defaultdict(int)
        
        for idx, row in test_df.iterrows():
            league = row['League']
            match_date = row['Date']
            
            # Check if we need to retrain
            if (league not in self.params_cache or 
                matches_since_retrain[league] >= self.config.retrain_frequency):
                
                # Get training data (all matches before this date)
                train_df = df[(df['Date'] < match_date) & (df['League'] == league)]
                
                if len(train_df) >= self.config.min_train_matches:
                    try:
                        self.params_cache[league] = fit_league_xg(
                            train_df, 
                            use_xg=self.config.use_xg
                        )
                        matches_since_retrain[league] = 0
                        logger.debug(f"Retrained {league} with {len(train_df)} matches")
                    except Exception as e:
                        logger.warning(f"Could not fit {league}: {e}")
                        continue
                else:
                    continue
            
            # Get parameters
            params = self.params_cache.get(league)
            if not params:
                continue
            
            # Prepare features
            features = self._prepare_features(row)
            
            # Get predictions
            try:
                probs = price_match_xg(
                    params,
                    row['HomeTeam'],
                    row['AwayTeam'],
                    use_xg_form=self.config.use_xg,
                    **features
                )
            except Exception as e:
                logger.debug(f"Could not price match: {e}")
                continue
            
            if not probs:
                continue
            
            matches_since_retrain[league] += 1
            
            # Store prediction
            pred_row = {
                'Date': match_date,
                'League': league,
                'LeagueType': row.get('LeagueType', 'league'),
                'HomeTeam': row['HomeTeam'],
                'AwayTeam': row['AwayTeam'],
                'FTHG': row['FTHG'],
                'FTAG': row['FTAG'],
                # Actuals
                'actual_BTTS': row['actual_BTTS'],
                'actual_OU_1_5': row['actual_OU_1_5'],
                'actual_OU_2_5': row['actual_OU_2_5'],
                'actual_OU_3_5': row['actual_OU_3_5'],
                # Probabilities
                'prob_BTTS': probs.get('DC_BTTS_Y', 0.5),
                'prob_OU_1_5': probs.get('DC_OU_1_5_O', 0.5),
                'prob_OU_2_5': probs.get('DC_OU_2_5_O', 0.5),
                'prob_OU_3_5': probs.get('DC_OU_3_5_O', 0.5),
                # Predictions (threshold at 0.5)
                'pred_BTTS': 'Y' if probs.get('DC_BTTS_Y', 0.5) > 0.5 else 'N',
                'pred_OU_1_5': 'O' if probs.get('DC_OU_1_5_O', 0.5) > 0.5 else 'U',
                'pred_OU_2_5': 'O' if probs.get('DC_OU_2_5_O', 0.5) > 0.5 else 'U',
                'pred_OU_3_5': 'O' if probs.get('DC_OU_3_5_O', 0.5) > 0.5 else 'U',
                # Expected goals
                'expected_home': probs.get('expected_home_goals', 0),
                'expected_away': probs.get('expected_away_goals', 0),
                'expected_total': probs.get('expected_total_goals', 0),
            }
            
            predictions.append(pred_row)
        
        results_df = pd.DataFrame(predictions)
        logger.info(f"Generated {len(results_df)} predictions")
        
        return results_df
    
    def evaluate(self, predictions: pd.DataFrame) -> Dict:
        """
        Evaluate backtest results
        
        Returns:
            Dict with all metrics
        """
        results = {
            'config': {
                'name': self.config.name,
                'use_xg': self.config.use_xg,
                'use_injuries': self.config.use_injuries,
                'use_formations': self.config.use_formations,
                'use_h2h': self.config.use_h2h,
                'include_cups': self.config.include_cups,
            },
            'summary': {},
            'by_league': {},
            'by_market': {},
            'cup_vs_league': {}
        }
        
        # Overall metrics by market
        for market in self.config.markets:
            metrics = calculate_metrics(predictions, market)
            results['by_market'][market] = metrics
            
            if metrics:
                results['summary'][f'{market}_accuracy'] = metrics['accuracy']
                if metrics.get('brier_score'):
                    results['summary'][f'{market}_brier'] = metrics['brier_score']
        
        # By league
        for market in self.config.markets:
            results['by_league'][market] = calculate_league_metrics(predictions, market)
        
        # Cup vs League comparison
        for market in self.config.markets:
            results['cup_vs_league'][market] = calculate_cup_vs_league_metrics(predictions, market)
        
        return results
    
    def run_feature_ablation(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Run ablation study to measure feature importance
        
        Tests each feature on/off to measure impact
        """
        if df is None:
            df = self.load_data()
        
        feature_flags = [
            ('use_xg', 'xG Features'),
            ('use_injuries', 'Injuries'),
            ('use_formations', 'Formations'),
            ('use_h2h', 'H2H'),
            ('use_rest_days', 'Rest Days'),
            ('use_advanced_stats', 'Advanced Stats'),
        ]
        
        ablation_results = []
        
        # Baseline (all features on)
        logger.info("Running baseline (all features)...")
        baseline_config = BacktestConfig(
            name='baseline_all',
            use_xg=True,
            use_injuries=True,
            use_formations=True,
            use_h2h=True,
            use_rest_days=True,
            use_advanced_stats=True,
        )
        baseline_bt = Backtester(baseline_config)
        baseline_preds = baseline_bt.run(df.copy())
        baseline_eval = baseline_bt.evaluate(baseline_preds)
        
        ablation_results.append({
            'feature': 'ALL_ON',
            **{f'{m}_acc': baseline_eval['by_market'].get(m, {}).get('accuracy', 0) 
               for m in self.config.markets}
        })
        
        # Test each feature off
        for flag, name in feature_flags:
            logger.info(f"Testing without {name}...")
            
            test_config = BacktestConfig(
                name=f'without_{flag}',
                use_xg=True,
                use_injuries=True,
                use_formations=True,
                use_h2h=True,
                use_rest_days=True,
                use_advanced_stats=True,
            )
            setattr(test_config, flag, False)
            
            test_bt = Backtester(test_config)
            test_preds = test_bt.run(df.copy())
            test_eval = test_bt.evaluate(test_preds)
            
            ablation_results.append({
                'feature': f'WITHOUT_{name.upper()}',
                **{f'{m}_acc': test_eval['by_market'].get(m, {}).get('accuracy', 0) 
                   for m in self.config.markets}
            })
        
        # Test all features off (pure DC)
        logger.info("Running pure DC (no enhancements)...")
        pure_config = BacktestConfig(
            name='pure_dc',
            use_xg=False,
            use_injuries=False,
            use_formations=False,
            use_h2h=False,
            use_rest_days=False,
            use_advanced_stats=False,
        )
        pure_bt = Backtester(pure_config)
        pure_preds = pure_bt.run(df.copy())
        pure_eval = pure_bt.evaluate(pure_preds)
        
        ablation_results.append({
            'feature': 'PURE_DC',
            **{f'{m}_acc': pure_eval['by_market'].get(m, {}).get('accuracy', 0) 
               for m in self.config.markets}
        })
        
        results_df = pd.DataFrame(ablation_results)
        return results_df


def print_results(results: Dict):
    """Print backtest results"""
    print("\n" + "="*70)
    print(f"BACKTEST RESULTS: {results['config']['name']}")
    print("="*70)
    
    print("\nConfiguration:")
    for k, v in results['config'].items():
        print(f"  {k}: {v}")
    
    print("\n" + "-"*70)
    print("OVERALL ACCURACY BY MARKET")
    print("-"*70)
    
    for market, metrics in results['by_market'].items():
        if metrics:
            acc = metrics.get('accuracy', 0)
            total = metrics.get('total', 0)
            brier = metrics.get('brier_score')
            high_conf = metrics.get('high_conf_accuracy')
            
            print(f"\n{market}:")
            print(f"  Accuracy: {acc:.1%} ({metrics.get('correct', 0)}/{total})")
            if brier:
                print(f"  Brier Score: {brier:.4f}")
            if high_conf:
                print(f"  High Confidence (>55%): {high_conf:.1%}")
    
    print("\n" + "-"*70)
    print("CUP VS LEAGUE COMPARISON")
    print("-"*70)
    
    for market, comparison in results['cup_vs_league'].items():
        if comparison:
            league_acc = comparison.get('league', {}).get('accuracy', 0)
            cup_acc = comparison.get('cup', {}).get('accuracy', 0)
            league_n = comparison.get('league', {}).get('total', 0)
            cup_n = comparison.get('cup', {}).get('total', 0)
            
            print(f"\n{market}:")
            print(f"  League: {league_acc:.1%} (n={league_n})")
            print(f"  Cup:    {cup_acc:.1%} (n={cup_n})")
            if cup_n > 50:
                diff = cup_acc - league_acc
                print(f"  Diff:   {diff:+.1%}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run backtests")
    parser.add_argument("--features", type=Path, default=FEATURES_PARQUET,
                        help="Path to features parquet")
    parser.add_argument("--ablation", action="store_true",
                        help="Run feature ablation study")
    parser.add_argument("--leagues", nargs="+",
                        help="Leagues to include")
    parser.add_argument("--no-cups", action="store_true",
                        help="Exclude cup competitions")
    parser.add_argument("--output", type=Path,
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    config = BacktestConfig(
        leagues=args.leagues,
        include_cups=not args.no_cups
    )
    
    backtester = Backtester(config)
    
    if args.ablation:
        print("\nRunning Feature Ablation Study...")
        print("="*70)
        
        df = backtester.load_data(args.features)
        ablation_df = backtester.run_feature_ablation(df)
        
        print("\nAblation Results:")
        print(ablation_df.to_string())
        
        # Save
        output_path = RESULTS_DIR / f"ablation_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        ablation_df.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path}")
        
    else:
        # Standard backtest
        predictions = backtester.run()
        results = backtester.evaluate(predictions)
        
        print_results(results)
        
        # Save predictions
        output_path = RESULTS_DIR / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        predictions.to_csv(output_path, index=False)
        print(f"\nPredictions saved to {output_path}")
