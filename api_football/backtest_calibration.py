# backtest_calibration.py
"""
Calibration-Focused Backtest
Combines backtesting with calibration analysis and parameter optimization

Key outputs:
1. Calibration curves (predicted vs actual by probability bin)
2. Optimal parameters per league
3. Feature importance rankings
4. Recommended betting thresholds
5. Exportable tuned config file

Usage:
    python backtest_calibration.py --features data/processed/features.parquet
    python backtest_calibration.py --features data/processed/features.parquet --optimize
    python backtest_calibration.py --features data/processed/features.parquet --league E0 --detailed
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import logging

from models_dc_xg import fit_league_xg, price_match_xg, LEAGUE_CONFIG
from calibration import (
    CalibrationAnalyzer, CalibrationMetrics, 
    PlattScaler, IsotonicCalibrator, BinningCalibrator,
    analyze_calibration, calculate_brier_score, calculate_log_loss,
    find_optimal_threshold, brier_decomposition
)
from optimize_params import (
    ParameterOptimizer, OptimizationConfig,
    walk_forward_split, calculate_feature_importance
)
from backtest_api import Backtester, BacktestConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


RESULTS_DIR = Path("calibration_results")


@dataclass
class CalibrationBacktestResult:
    """Complete calibration backtest results"""
    leagues: List[str]
    markets: List[str]
    n_matches: int
    date_range: Tuple[str, str]
    overall_metrics: Dict[str, Dict]
    league_metrics: Dict[str, Dict[str, Dict]]
    calibration_curves: Dict[str, Dict]
    optimal_thresholds: Dict[str, Dict[str, float]]
    optimized_params: Dict[str, Dict] = field(default_factory=dict)
    feature_importance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    fitted_calibrators: Dict[str, str] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class CalibrationBacktest:
    """
    Comprehensive calibration-focused backtester
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.predictions_df: Optional[pd.DataFrame] = None
        self.calibration_analyzer = CalibrationAnalyzer()
        self.results: Optional[CalibrationBacktestResult] = None
    
    def run(self, df: pd.DataFrame, 
            optimize_params: bool = False,
            fit_calibrators: bool = True,
            analyze_features: bool = True) -> CalibrationBacktestResult:
        """
        Run full calibration backtest
        """
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        leagues = self.config.leagues or df['League'].unique().tolist()
        markets = self.config.markets
        
        logger.info("="*60)
        logger.info("CALIBRATION BACKTEST")
        logger.info("="*60)
        logger.info(f"Leagues: {leagues}")
        logger.info(f"Markets: {markets}")
        
        # Step 1: Generate predictions
        logger.info("\n[1/5] Generating predictions...")
        backtester = Backtester(self.config)
        self.predictions_df = backtester.run(df)
        
        if len(self.predictions_df) == 0:
            raise ValueError("No predictions generated")
        
        logger.info(f"  Generated {len(self.predictions_df)} predictions")
        
        # Step 2: Calibration analysis
        logger.info("\n[2/5] Analyzing calibration...")
        overall_metrics = self.calibration_analyzer.analyze(self.predictions_df, markets)
        league_metrics = self.calibration_analyzer.analyze_by_league(self.predictions_df, markets)
        
        # Step 3: Fit calibrators
        calibrator_info = {}
        if fit_calibrators:
            logger.info("\n[3/5] Fitting calibrators...")
            
            train_size = int(len(self.predictions_df) * 0.7)
            train_preds = self.predictions_df.iloc[:train_size]
            test_preds = self.predictions_df.iloc[train_size:]
            
            for method in ['isotonic', 'platt', 'binning']:
                self.calibration_analyzer.fit_calibrators(train_preds, method=method, markets=markets)
                test_calibrated = self.calibration_analyzer.apply_calibration(test_preds, markets)
                
                for market in markets:
                    orig_col = f'prob_{market}'
                    cal_col = f'{orig_col}_calibrated'
                    actual_col = f'actual_{market}'
                    
                    if cal_col not in test_calibrated.columns:
                        continue
                    
                    if market == 'BTTS':
                        y_true = (test_calibrated[actual_col] == 'Y').astype(int).values
                    else:
                        y_true = (test_calibrated[actual_col] == 'O').astype(int).values
                    
                    orig_brier = calculate_brier_score(y_true, test_calibrated[orig_col].values)
                    cal_brier = calculate_brier_score(y_true, test_calibrated[cal_col].values)
                    
                    if market not in calibrator_info:
                        calibrator_info[market] = {}
                    
                    calibrator_info[market][method] = {
                        'original_brier': orig_brier,
                        'calibrated_brier': cal_brier,
                        'improvement': orig_brier - cal_brier
                    }
            
            for market in markets:
                if market in calibrator_info:
                    best_method = max(calibrator_info[market].keys(),
                                     key=lambda m: calibrator_info[market][m]['improvement'])
                    calibrator_info[market]['best_method'] = best_method
                    logger.info(f"  {market}: Best = {best_method} "
                               f"(+{calibrator_info[market][best_method]['improvement']:.4f})")
        else:
            logger.info("\n[3/5] Skipping calibrator fitting")
        
        # Step 4: Parameter optimization
        optimized_params = {}
        if optimize_params:
            logger.info("\n[4/5] Optimizing parameters...")
            
            opt_config = OptimizationConfig(
                objective='brier',
                markets=markets,
                n_folds=3,
                rho_range=(-0.03, 0.12, 0.03),
                decay_range=(250, 450, 100),
                xg_weight_range=(0.4, 0.8, 0.2)
            )
            
            optimizer = ParameterOptimizer(opt_config)
            opt_results = optimizer.optimize_all(df, leagues=leagues[:5])
            
            for league, market_results in opt_results.items():
                optimized_params[league] = {}
                for market, result in market_results.items():
                    optimized_params[league][market] = {
                        'params': result.best_params,
                        'score': result.best_score,
                        'improvement': result.improvement
                    }
        else:
            logger.info("\n[4/5] Skipping parameter optimization")
        
        # Step 5: Feature importance
        feature_importance = {}
        if analyze_features:
            logger.info("\n[5/5] Calculating feature importance...")
            
            for league in leagues[:3]:
                league_df = df[df['League'] == league]
                if len(league_df) < 100:
                    continue
                
                for market in markets:
                    try:
                        importance = calculate_feature_importance(df, league, market)
                        
                        if league not in feature_importance:
                            feature_importance[league] = {}
                        feature_importance[league][market] = importance
                        
                        top_features = sorted(importance.items(), key=lambda x: -x[1])[:3]
                        logger.info(f"  {league} {market}: " + 
                                   ", ".join(f"{k}={v:.4f}" for k, v in top_features))
                    except Exception as e:
                        logger.warning(f"  Failed for {league} {market}: {e}")
        else:
            logger.info("\n[5/5] Skipping feature importance")
        
        # Build optimal thresholds
        optimal_thresholds = {}
        for league, markets_data in self.calibration_analyzer.league_results.items():
            optimal_thresholds[league] = {}
            for market, metrics in markets_data.items():
                optimal_thresholds[league][market] = metrics.optimal_threshold
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_metrics, league_metrics, calibrator_info, optimized_params
        )
        
        # Build calibration curves
        calibration_curves = {}
        for market, metrics in overall_metrics.items():
            calibration_curves[market] = {
                'bin_edges': metrics.bin_edges,
                'predicted': metrics.bin_pred_means,
                'actual': metrics.bin_actual_means,
                'counts': metrics.bin_counts
            }
        
        # Create result
        self.results = CalibrationBacktestResult(
            leagues=leagues,
            markets=markets,
            n_matches=len(self.predictions_df),
            date_range=(str(df['Date'].min().date()), str(df['Date'].max().date())),
            overall_metrics={m: {
                'accuracy': metrics.accuracy,
                'brier_score': metrics.brier_score,
                'log_loss': metrics.log_loss,
                'ece': metrics.ece,
                'mce': metrics.mce,
                'reliability': metrics.reliability,
                'resolution': metrics.resolution,
                'optimal_threshold': metrics.optimal_threshold
            } for m, metrics in overall_metrics.items()},
            league_metrics={league: {market: {
                'accuracy': m.accuracy,
                'brier_score': m.brier_score,
                'ece': m.ece,
                'optimal_threshold': m.optimal_threshold
            } for market, m in markets_data.items()} 
            for league, markets_data in league_metrics.items()},
            calibration_curves=calibration_curves,
            optimal_thresholds=optimal_thresholds,
            optimized_params=optimized_params,
            feature_importance=feature_importance,
            fitted_calibrators={m: info.get('best_method', 'none') 
                               for m, info in calibrator_info.items()},
            recommendations=recommendations
        )
        
        return self.results
    
    def _generate_recommendations(self, overall_metrics, league_metrics, 
                                  calibrator_info, optimized_params) -> List[str]:
        """Generate actionable recommendations"""
        recs = []
        
        for market, metrics in overall_metrics.items():
            if metrics.ece > 0.05:
                recs.append(f"⚠️ {market}: High ECE ({metrics.ece:.3f}). Apply calibration.")
            
            if metrics.optimal_threshold != 0.5:
                diff = abs(metrics.optimal_threshold - 0.5)
                if diff > 0.03:
                    recs.append(f"🎯 {market}: Use threshold {metrics.optimal_threshold:.2f} instead of 0.50")
            
            # Check for over/under confidence
            if metrics.reliability > 0.008:
                # Determine direction from calibration curve
                mid_pred = metrics.bin_pred_means[4] if len(metrics.bin_pred_means) > 4 else 0.5
                mid_actual = metrics.bin_actual_means[4] if len(metrics.bin_actual_means) > 4 else None
                if mid_actual is not None:
                    if mid_pred > mid_actual:
                        recs.append(f"📊 {market}: Model is overconfident. Consider conservative adjustment.")
                    else:
                        recs.append(f"📊 {market}: Model is underconfident. Predictions may have more value.")
        
        for market, info in calibrator_info.items():
            if 'best_method' in info:
                improvement = info[info['best_method']]['improvement']
                if improvement > 0.003:
                    recs.append(f"✅ {market}: Apply {info['best_method']} calibration (Brier +{improvement:.4f})")
        
        return recs
    
    def print_report(self):
        """Print comprehensive calibration report"""
        if not self.results:
            print("No results. Run backtest first.")
            return
        
        r = self.results
        
        print("\n" + "="*70)
        print("CALIBRATION BACKTEST REPORT")
        print("="*70)
        
        print(f"\nDataset: {r.n_matches:,} matches ({r.date_range[0]} to {r.date_range[1]})")
        print(f"Leagues: {len(r.leagues)}")
        print(f"Markets: {r.markets}")
        
        # Overall metrics
        print("\n" + "-"*70)
        print("OVERALL CALIBRATION METRICS")
        print("-"*70)
        
        for market, metrics in r.overall_metrics.items():
            print(f"\n{market}:")
            print(f"  Accuracy:          {metrics['accuracy']:.1%}")
            print(f"  Brier Score:       {metrics['brier_score']:.4f}")
            print(f"  Log Loss:          {metrics['log_loss']:.4f}")
            print(f"  ECE:               {metrics['ece']:.4f}")
            print(f"  Reliability:       {metrics['reliability']:.4f}")
            print(f"  Resolution:        {metrics['resolution']:.4f}")
            print(f"  Optimal Threshold: {metrics['optimal_threshold']:.2f}")
        
        # Calibration curves
        print("\n" + "-"*70)
        print("CALIBRATION CURVES (Predicted vs Actual)")
        print("-"*70)
        
        for market, curve in r.calibration_curves.items():
            print(f"\n{market}:")
            print(f"  {'Bin':<12} {'Predicted':>10} {'Actual':>10} {'Gap':>10} {'Count':>8}")
            
            for i in range(len(curve['predicted'])):
                pred = curve['predicted'][i]
                actual = curve['actual'][i]
                count = curve['counts'][i]
                
                if actual is not None:
                    gap = actual - pred
                    actual_str = f"{actual:.3f}"
                    gap_str = f"{gap:+.3f}"
                else:
                    actual_str = "N/A"
                    gap_str = "N/A"
                
                bin_label = f"{curve['bin_edges'][i]:.1f}-{curve['bin_edges'][i+1]:.1f}"
                print(f"  {bin_label:<12} {pred:>10.3f} {actual_str:>10} {gap_str:>10} {count:>8}")
        
        # League breakdown
        if r.league_metrics:
            print("\n" + "-"*70)
            print("PER-LEAGUE METRICS")
            print("-"*70)
            
            print(f"\n  {'League':<10} {'Market':<10} {'Accuracy':>10} {'Brier':>10} {'ECE':>10} {'Threshold':>10}")
            
            for league, markets in sorted(r.league_metrics.items()):
                for market, m in markets.items():
                    print(f"  {league:<10} {market:<10} {m['accuracy']:>10.1%} {m['brier_score']:>10.4f} "
                          f"{m['ece']:>10.4f} {m['optimal_threshold']:>10.2f}")
        
        # Feature importance
        if r.feature_importance:
            print("\n" + "-"*70)
            print("FEATURE IMPORTANCE (by Brier improvement)")
            print("-"*70)
            
            for league, markets in r.feature_importance.items():
                for market, importance in markets.items():
                    sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
                    print(f"\n  {league} {market}:")
                    for feat, imp in sorted_imp:
                        bar = "█" * int(abs(imp) * 200) if imp > 0 else ""
                        print(f"    {feat:<20} {imp:+.4f} {bar}")
        
        # Recommendations
        if r.recommendations:
            print("\n" + "-"*70)
            print("RECOMMENDATIONS")
            print("-"*70)
            for rec in r.recommendations:
                print(f"  {rec}")
        
        # Optimal thresholds summary
        print("\n" + "-"*70)
        print("OPTIMAL THRESHOLDS BY LEAGUE")
        print("-"*70)
        
        for market in r.markets:
            thresholds_for_market = []
            for league, thresholds in r.optimal_thresholds.items():
                if market in thresholds:
                    thresh = thresholds[market]
                    if abs(thresh - 0.5) > 0.02:
                        thresholds_for_market.append((league, thresh))
            
            if thresholds_for_market:
                print(f"\n  {market}:")
                for league, thresh in sorted(thresholds_for_market, key=lambda x: x[1]):
                    diff = thresh - 0.5
                    print(f"    {league}: {thresh:.2f} ({diff:+.2f})")
    
    def export_config(self, path: Path):
        """Export optimized configuration as JSON"""
        if not self.results:
            logger.error("No results to export")
            return
        
        config = {
            'generated': datetime.now().isoformat(),
            'optimal_thresholds': self.results.optimal_thresholds,
            'calibrators': self.results.fitted_calibrators,
            'optimized_params': self.results.optimized_params,
            'recommendations': self.results.recommendations
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Exported config to {path}")
    
    def export_python_config(self, path: Path):
        """Export as Python config file for direct use"""
        if not self.results:
            logger.error("No results to export")
            return
        
        lines = [
            "# Auto-generated Calibration Config",
            f"# Generated: {datetime.now().isoformat()}",
            "",
            "# Optimal thresholds per league/market",
            "OPTIMAL_THRESHOLDS = {"
        ]
        
        for league, markets in self.results.optimal_thresholds.items():
            lines.append(f"    '{league}': {{")
            for market, thresh in markets.items():
                lines.append(f"        '{market}': {thresh:.3f},")
            lines.append("    },")
        lines.append("}")
        
        lines.extend([
            "",
            "# Best calibration method per market",
            f"CALIBRATION_METHODS = {self.results.fitted_calibrators}",
            "",
            "# Overall metrics",
            "OVERALL_METRICS = {"
        ])
        
        for market, metrics in self.results.overall_metrics.items():
            lines.append(f"    '{market}': {{")
            for key, val in metrics.items():
                if isinstance(val, float):
                    lines.append(f"        '{key}': {val:.4f},")
                else:
                    lines.append(f"        '{key}': {val},")
            lines.append("    },")
        lines.append("}")
        
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Exported Python config to {path}")
    
    def export_full_results(self, path: Path):
        """Export all results as JSON"""
        if not self.results:
            logger.error("No results to export")
            return
        
        r = self.results
        
        export = {
            'metadata': {
                'generated': datetime.now().isoformat(),
                'n_matches': r.n_matches,
                'date_range': r.date_range,
                'leagues': r.leagues,
                'markets': r.markets
            },
            'overall_metrics': r.overall_metrics,
            'league_metrics': r.league_metrics,
            'calibration_curves': r.calibration_curves,
            'optimal_thresholds': r.optimal_thresholds,
            'optimized_params': r.optimized_params,
            'feature_importance': r.feature_importance,
            'fitted_calibrators': r.fitted_calibrators,
            'recommendations': r.recommendations
        }
        
        with open(path, 'w') as f:
            json.dump(export, f, indent=2)
        
        logger.info(f"Exported full results to {path}")


# =============================================================================
# QUICK CALIBRATION CHECK
# =============================================================================

def quick_calibration_report(df: pd.DataFrame, markets: List[str] = None) -> Dict:
    """
    Quick calibration check without full backtest
    
    Args:
        df: DataFrame with prob_{market} and actual_{market} columns
        markets: Markets to check
    
    Returns:
        Summary dict
    """
    markets = markets or ['BTTS', 'OU_2_5']
    
    results = {}
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
        
        metrics = analyze_calibration(y_true, y_prob, market)
        
        results[market] = {
            'n_samples': len(y_true),
            'accuracy': metrics.accuracy,
            'brier_score': metrics.brier_score,
            'ece': metrics.ece,
            'optimal_threshold': metrics.optimal_threshold,
            'calibration_gap': metrics.reliability
        }
    
    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calibration Backtest")
    parser.add_argument("--features", type=Path, required=True, help="Features parquet file")
    parser.add_argument("--leagues", nargs="+", help="Leagues to analyze")
    parser.add_argument("--markets", nargs="+", default=['BTTS', 'OU_2_5', 'OU_1_5', 'OU_3_5'])
    parser.add_argument("--optimize", action="store_true", help="Run parameter optimization")
    parser.add_argument("--no-calibrators", action="store_true", help="Skip calibrator fitting")
    parser.add_argument("--no-features", action="store_true", help="Skip feature importance")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--detailed", action="store_true", help="Show detailed output")
    
    args = parser.parse_args()
    
    # Create output directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = args.output or RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading features from {args.features}")
    df = pd.read_parquet(args.features)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Configure
    config = BacktestConfig(
        leagues=args.leagues,
        markets=args.markets
    )
    
    # Run calibration backtest
    backtest = CalibrationBacktest(config)
    results = backtest.run(
        df,
        optimize_params=args.optimize,
        fit_calibrators=not args.no_calibrators,
        analyze_features=not args.no_features
    )
    
    # Print report
    backtest.print_report()
    
    # Export results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    backtest.export_full_results(output_dir / f"calibration_results_{timestamp}.json")
    backtest.export_config(output_dir / f"calibration_config_{timestamp}.json")
    backtest.export_python_config(output_dir / f"calibration_config_{timestamp}.py")
    
    # Save predictions
    if backtest.predictions_df is not None:
        backtest.predictions_df.to_csv(output_dir / f"calibration_predictions_{timestamp}.csv", index=False)
    
    print(f"\nResults saved to {output_dir}")
