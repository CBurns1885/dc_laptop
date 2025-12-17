# backtest_adaptive_dc.py
"""
Adaptive Dixon-Coles Backtest Engine

Compares two DC fitting strategies:
1. STATIC: Fit once on all historical data (current production method)
2. ADAPTIVE: Re-fit DC parameters each test period (walk-forward)

This answers: "Does recalibrating DC weights improve accuracy?"
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from config import DATA_DIR, OUTPUT_DIR, FEATURES_PARQUET
from models_dc import fit_all, price_match, DCParams


class AdaptiveDCBacktest:
    """
    Walk-forward backtest comparing static vs adaptive DC fitting
    """

    def __init__(self,
                 start_date: str,
                 end_date: str,
                 test_window_days: int = 7,
                 min_training_days: int = 365,
                 refit_frequency: str = 'weekly'):
        """
        Args:
            start_date: Backtest start (YYYY-MM-DD)
            end_date: Backtest end (YYYY-MM-DD)
            test_window_days: Days per test window
            min_training_days: Minimum training data required
            refit_frequency: 'weekly', 'biweekly', or 'monthly'
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.test_window_days = test_window_days
        self.min_training_days = min_training_days
        self.refit_frequency = refit_frequency

        # Results storage
        self.static_results = []
        self.adaptive_results = []

        # Cached static params (fit once at start)
        self.static_params = None

    def load_data(self) -> pd.DataFrame:
        """Load features data"""
        if not FEATURES_PARQUET.exists():
            raise FileNotFoundError(
                f"Features not found: {FEATURES_PARQUET}\n"
                "Run build_features(force=True) first"
            )

        df = pd.read_parquet(FEATURES_PARQUET)
        df['Date'] = pd.to_datetime(df['Date'])

        # Keep only columns needed for DC fitting
        required_cols = ['Date', 'League', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']

        # Add target columns if they exist
        target_cols = ['y_BTTS', 'y_OU_0_5', 'y_OU_1_5', 'y_OU_2_5',
                      'y_OU_3_5', 'y_OU_4_5', 'y_OU_5_5']
        for col in target_cols:
            if col in df.columns:
                required_cols.append(col)

        df = df[required_cols].copy()
        df = df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])
        df = df.sort_values('Date').reset_index(drop=True)

        print(f" Loaded {len(df)} matches from {df['Date'].min().date()} to {df['Date'].max().date()}")

        return df

    def get_test_periods(self) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Generate test periods"""
        periods = []
        current = self.start_date

        while current <= self.end_date:
            period_end = current + timedelta(days=self.test_window_days)
            periods.append((current, period_end))
            current = period_end

        return periods

    def split_train_test(self,
                        df: pd.DataFrame,
                        test_start: pd.Timestamp,
                        test_end: pd.Timestamp,
                        use_all_history: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/test

        Args:
            df: Full dataset
            test_start: Test period start
            test_end: Test period end
            use_all_history: If True, use ALL data before test (STATIC)
                           If False, use only recent history (ADAPTIVE with time decay)

        Returns:
            train_df, test_df
        """
        # Test: matches in test window
        test_df = df[(df['Date'] >= test_start) & (df['Date'] < test_end)].copy()

        # Train: all data before test period
        train_cutoff = test_start - timedelta(days=1)
        train_df = df[df['Date'] <= train_cutoff].copy()

        # For adaptive, optionally limit training window (optional enhancement)
        # DC model already uses time-weighted decay, so we can use all history
        # but the weighting will favor recent matches

        return train_df, test_df

    def fit_static_params(self, df: pd.DataFrame) -> Dict[str, DCParams]:
        """
        Fit DC parameters ONCE using all available data
        This simulates current production behavior
        """
        print("\n Fitting STATIC DC parameters (once, all data)...")

        # Use all data up to backtest start
        train_df = df[df['Date'] < self.start_date].copy()

        if len(train_df) < 100:
            raise ValueError(f"Insufficient data for static fit: {len(train_df)} matches")

        print(f"   Training on {len(train_df)} matches")
        print(f"   Date range: {train_df['Date'].min().date()} to {train_df['Date'].max().date()}")

        params = fit_all(train_df, use_form=True)

        print(f"   Fitted {len(params)} leagues")

        return params

    def fit_adaptive_params(self, train_df: pd.DataFrame) -> Dict[str, DCParams]:
        """
        Fit DC parameters using only data available at this point in time
        This is walk-forward: params update each period
        """
        if len(train_df) < 100:
            return {}

        params = fit_all(train_df, use_form=True)
        return params

    def predict_period(self,
                       params: Dict[str, DCParams],
                       test_df: pd.DataFrame,
                       label: str = '') -> pd.DataFrame:
        """
        Generate predictions for test period using given params

        Returns:
            test_df with DC probability columns added
        """
        predictions = []

        for idx, row in test_df.iterrows():
            league = row['League']
            home = row['HomeTeam']
            away = row['AwayTeam']

            # Get DC predictions if league params exist
            if league in params:
                pred = price_match(params[league], home, away, use_form=True)
            else:
                pred = {}

            predictions.append(pred)

        # Merge predictions with test data
        pred_df = pd.DataFrame(predictions)
        result = pd.concat([test_df.reset_index(drop=True), pred_df], axis=1)

        return result

    def evaluate_predictions(self,
                            df: pd.DataFrame,
                            method: str) -> Dict:
        """
        Evaluate DC predictions against actual outcomes

        Args:
            df: DataFrame with predictions and actuals
            method: 'static' or 'adaptive' (for labeling)

        Returns:
            Dictionary with evaluation metrics
        """
        results = {
            'method': method,
            'total_matches': len(df),
            'markets': {}
        }

        # Markets to evaluate
        markets = {
            'BTTS': {
                'actual': 'y_BTTS',
                'pred_cols': ['DC_BTTS_Y', 'DC_BTTS_N'],
                'outcomes': ['Y', 'N']
            },
            'OU_0_5': {
                'actual': 'y_OU_0_5',
                'pred_cols': ['DC_OU_0_5_O', 'DC_OU_0_5_U'],
                'outcomes': ['O', 'U']
            },
            'OU_1_5': {
                'actual': 'y_OU_1_5',
                'pred_cols': ['DC_OU_1_5_O', 'DC_OU_1_5_U'],
                'outcomes': ['O', 'U']
            },
            'OU_2_5': {
                'actual': 'y_OU_2_5',
                'pred_cols': ['DC_OU_2_5_O', 'DC_OU_2_5_U'],
                'outcomes': ['O', 'U']
            },
            'OU_3_5': {
                'actual': 'y_OU_3_5',
                'pred_cols': ['DC_OU_3_5_O', 'DC_OU_3_5_U'],
                'outcomes': ['O', 'U']
            },
            'OU_4_5': {
                'actual': 'y_OU_4_5',
                'pred_cols': ['DC_OU_4_5_O', 'DC_OU_4_5_U'],
                'outcomes': ['O', 'U']
            },
            'OU_5_5': {
                'actual': 'y_OU_5_5',
                'pred_cols': ['DC_OU_5_5_O', 'DC_OU_5_5_U'],
                'outcomes': ['O', 'U']
            },
        }

        for market_name, market_info in markets.items():
            actual_col = market_info['actual']
            pred_cols = market_info['pred_cols']
            outcomes = market_info['outcomes']

            # Skip if columns don't exist
            if actual_col not in df.columns:
                continue

            available_pred_cols = [c for c in pred_cols if c in df.columns]
            if not available_pred_cols:
                continue

            # Get valid predictions
            valid = df[actual_col].notna()
            actual = df.loc[valid, actual_col]

            if len(actual) == 0:
                continue

            # Get predicted outcome (highest probability)
            predictions = df.loc[valid, available_pred_cols]
            pred_outcome_idx = predictions.idxmax(axis=1, skipna=True)

            # Filter out rows where prediction is NaN (no valid predictions)
            valid_predictions = pred_outcome_idx.notna()
            pred_outcome_idx = pred_outcome_idx[valid_predictions]
            predictions = predictions.loc[valid_predictions]
            actual = actual.loc[valid_predictions]

            if len(actual) == 0:
                # No valid predictions for this market
                continue

            # Map to outcomes
            outcome_map = {}
            for col, outcome in zip(pred_cols, outcomes):
                outcome_map[col] = outcome

            predicted = pred_outcome_idx.map(outcome_map)

            # Calculate accuracy
            correct = (predicted == actual).sum()
            total = len(actual)
            accuracy = correct / total if total > 0 else 0

            # Brier score (calibration metric)
            brier_scores = []
            for idx in actual.index:
                true_outcome = actual[idx]
                for col, outcome in zip(available_pred_cols, outcomes):
                    pred_prob = df.loc[idx, col]
                    if pd.notna(pred_prob):
                        true_prob = 1.0 if outcome == true_outcome else 0.0
                        brier_scores.append((pred_prob - true_prob) ** 2)

            brier = np.mean(brier_scores) if brier_scores else 0

            results['markets'][market_name] = {
                'total': int(total),
                'correct': int(correct),
                'accuracy': float(accuracy),
                'brier_score': float(brier)
            }

        return results

    def run_backtest(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run complete adaptive backtest comparing static vs adaptive DC

        Returns:
            static_summary, adaptive_summary
        """
        print("\n" + "="*70)
        print(" ADAPTIVE DC BACKTEST ENGINE")
        print("="*70)
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Test window: {self.test_window_days} days")
        print(f"Refit frequency: {self.refit_frequency}")
        print("="*70)

        # Load data
        full_df = self.load_data()

        # Fit static params (once at start)
        self.static_params = self.fit_static_params(full_df)

        # Get test periods
        periods = self.get_test_periods()
        print(f"\n Testing {len(periods)} periods")

        # Track when to refit adaptive params
        refit_counter = 0
        refit_interval = {
            'weekly': 1,
            'biweekly': 2,
            'monthly': 4
        }.get(self.refit_frequency, 1)

        adaptive_params = None

        for i, (test_start, test_end) in enumerate(periods, 1):
            print(f"\n{'─'*70}")
            print(f"Period {i}/{len(periods)}: {test_start.date()} to {test_end.date()}")

            # Split data
            train_df, test_df = self.split_train_test(full_df, test_start, test_end)

            if len(test_df) == 0:
                print("   ⚠ No test matches - skipping")
                continue

            print(f"   Train: {len(train_df)} matches | Test: {len(test_df)} matches")

            # === STATIC APPROACH ===
            # Use pre-fitted params (never update)
            print(f"   [STATIC] Using pre-fitted params...")
            static_preds = self.predict_period(self.static_params, test_df, 'static')
            static_eval = self.evaluate_predictions(static_preds, 'static')
            static_eval['period_start'] = test_start
            static_eval['period_end'] = test_end
            self.static_results.append(static_eval)

            # === ADAPTIVE APPROACH ===
            # Refit params periodically
            if adaptive_params is None or refit_counter % refit_interval == 0:
                print(f"   [ADAPTIVE] Re-fitting DC parameters...")
                adaptive_params = self.fit_adaptive_params(train_df)
                print(f"   [ADAPTIVE] Fitted {len(adaptive_params)} leagues")
            else:
                print(f"   [ADAPTIVE] Using cached params (refit in {refit_interval - (refit_counter % refit_interval)} periods)")

            adaptive_preds = self.predict_period(adaptive_params, test_df, 'adaptive')
            adaptive_eval = self.evaluate_predictions(adaptive_preds, 'adaptive')
            adaptive_eval['period_start'] = test_start
            adaptive_eval['period_end'] = test_end
            self.adaptive_results.append(adaptive_eval)

            refit_counter += 1

            # Quick comparison
            print(f"   Results:")
            for market in static_eval.get('markets', {}).keys():
                static_acc = static_eval['markets'][market]['accuracy']
                adaptive_acc = adaptive_eval['markets'][market]['accuracy']

                diff = (adaptive_acc - static_acc) * 100
                symbol = "✓" if diff > 0 else "✗" if diff < 0 else "="

                print(f"      {symbol} {market}: "
                      f"Static {static_acc:.1%} vs Adaptive {adaptive_acc:.1%} "
                      f"({diff:+.1f}%)")

        # Generate summaries
        static_summary = self.generate_summary(self.static_results, 'STATIC')
        adaptive_summary = self.generate_summary(self.adaptive_results, 'ADAPTIVE')

        # Compare
        self.compare_methods(static_summary, adaptive_summary)

        return static_summary, adaptive_summary

    def generate_summary(self, results: List[Dict], method: str) -> pd.DataFrame:
        """Generate summary statistics for a method"""
        if not results:
            return pd.DataFrame()

        market_summary = {}
        all_markets = set()

        for result in results:
            all_markets.update(result.get('markets', {}).keys())

        for market in sorted(all_markets):
            total = 0
            correct = 0
            brier_scores = []

            for result in results:
                if market in result.get('markets', {}):
                    stats = result['markets'][market]
                    total += stats['total']
                    correct += stats['correct']
                    brier_scores.append(stats['brier_score'])

            if total > 0:
                accuracy = correct / total

                market_summary[market] = {
                    'Total_Matches': total,
                    'Correct': correct,
                    'Accuracy_%': round(accuracy * 100, 2),
                    'Brier_Score': round(np.mean(brier_scores), 4)
                }

        summary_df = pd.DataFrame.from_dict(market_summary, orient='index')

        if not summary_df.empty:
            summary_df = summary_df.sort_values('Accuracy_%', ascending=False)

        print(f"\n{'='*70}")
        print(f" {method} DC SUMMARY")
        print(f"{'='*70}")
        print(summary_df.to_string())

        # Save
        output_path = OUTPUT_DIR / f"backtest_dc_{method.lower()}_summary.csv"
        summary_df.to_csv(output_path)
        print(f"\n Saved: {output_path}")

        return summary_df

    def compare_methods(self, static_df: pd.DataFrame, adaptive_df: pd.DataFrame):
        """Compare static vs adaptive performance"""
        print(f"\n{'='*70}")
        print(" STATIC vs ADAPTIVE COMPARISON")
        print(f"{'='*70}")

        if static_df.empty or adaptive_df.empty:
            print(" ⚠ Insufficient data for comparison")
            return

        comparison = []

        # Get common markets
        common_markets = set(static_df.index) & set(adaptive_df.index)

        for market in sorted(common_markets):
            static_acc = static_df.loc[market, 'Accuracy_%']
            adaptive_acc = adaptive_df.loc[market, 'Accuracy_%']

            static_brier = static_df.loc[market, 'Brier_Score']
            adaptive_brier = adaptive_df.loc[market, 'Brier_Score']

            acc_diff = adaptive_acc - static_acc
            brier_diff = adaptive_brier - static_brier  # Lower is better

            # Determine winner
            if acc_diff > 0.5:
                winner = "✓ ADAPTIVE"
            elif acc_diff < -0.5:
                winner = "✓ STATIC"
            else:
                winner = "≈ TIE"

            comparison.append({
                'Market': market,
                'Static_Acc_%': round(static_acc, 2),
                'Adaptive_Acc_%': round(adaptive_acc, 2),
                'Acc_Diff_%': round(acc_diff, 2),
                'Static_Brier': round(static_brier, 4),
                'Adaptive_Brier': round(adaptive_brier, 4),
                'Brier_Diff': round(brier_diff, 4),
                'Winner': winner
            })

        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('Acc_Diff_%', ascending=False)

        print(comparison_df.to_string(index=False))

        # Overall verdict
        print(f"\n{'='*70}")
        print(" VERDICT")
        print(f"{'='*70}")

        adaptive_wins = (comparison_df['Acc_Diff_%'] > 0.5).sum()
        static_wins = (comparison_df['Acc_Diff_%'] < -0.5).sum()
        ties = len(comparison_df) - adaptive_wins - static_wins

        print(f"Adaptive wins: {adaptive_wins}")
        print(f"Static wins: {static_wins}")
        print(f"Ties: {ties}")

        avg_acc_improvement = comparison_df['Acc_Diff_%'].mean()
        avg_brier_improvement = -comparison_df['Brier_Diff'].mean()  # Negative = better

        print(f"\nAverage accuracy improvement: {avg_acc_improvement:+.2f}%")
        print(f"Average Brier improvement: {avg_brier_improvement:+.4f}")

        if adaptive_wins > static_wins and avg_acc_improvement > 0:
            print(f"\n✓ RECOMMENDATION: Use ADAPTIVE DC fitting")
            print(f"   Refit frequency: {self.refit_frequency}")
        elif static_wins > adaptive_wins:
            print(f"\n✓ RECOMMENDATION: Keep STATIC DC fitting (current)")
            print(f"   Adaptive refitting does not improve accuracy")
        else:
            print(f"\n≈ NEUTRAL: No clear winner")
            print(f"   Both methods perform similarly")

        # Save comparison
        comparison_path = OUTPUT_DIR / "backtest_dc_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\n Saved comparison: {comparison_path}")


# ============================================================================
# CLI
# ============================================================================

def run_adaptive_backtest_cli():
    """Command-line interface"""
    print("\n" + "="*70)
    print(" ADAPTIVE DC BACKTEST")
    print("="*70)
    print("\nThis compares:")
    print("  1. STATIC: Fit DC once on all data (current production)")
    print("  2. ADAPTIVE: Re-fit DC parameters each test period")
    print()

    # Date selection
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    print(f"Default period: Last 6 months")
    print(f"   From: {start_date.date()}")
    print(f"   To: {end_date.date()}")

    choice = input("\nUse default? (y/n, default=y): ").strip().lower()

    if choice == 'n':
        start_input = input("Start date (YYYY-MM-DD): ").strip()
        end_input = input("End date (YYYY-MM-DD): ").strip()
        start_date = datetime.strptime(start_input, '%Y-%m-%d')
        end_date = datetime.strptime(end_input, '%Y-%m-%d')

    # Refit frequency
    print("\nRefit frequency options:")
    print("  1. Weekly (refit every week)")
    print("  2. Biweekly (refit every 2 weeks)")
    print("  3. Monthly (refit every 4 weeks)")

    freq_choice = input("\nSelect (1-3, default=1): ").strip() or "1"
    freq_map = {
        '1': 'weekly',
        '2': 'biweekly',
        '3': 'monthly'
    }
    refit_freq = freq_map.get(freq_choice, 'weekly')

    # Run backtest
    engine = AdaptiveDCBacktest(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        test_window_days=7,
        refit_frequency=refit_freq
    )

    static_summary, adaptive_summary = engine.run_backtest()

    print("\n" + "="*70)
    print(" BACKTEST COMPLETE")
    print("="*70)
    print("\nCheck outputs folder for:")
    print("  • backtest_dc_static_summary.csv")
    print("  • backtest_dc_adaptive_summary.csv")
    print("  • backtest_dc_comparison.csv")


if __name__ == "__main__":
    run_adaptive_backtest_cli()
