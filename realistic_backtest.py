# backtest_realistic.py
"""
Realistic Backtesting - YOUR Betting Style
Tests doubles only using your actual markets and odds
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from itertools import combinations
import shutil

from config import DATA_DIR, OUTPUT_DIR, FEATURES_PARQUET, MODEL_ARTIFACTS_DIR

class RealisticBacktest:
    """Backtest with YOUR actual betting strategy"""
    
    # YOUR MARKETS WITH ODDS AND STAKES (weighted by risk)
    MARKETS = {
        'OU_0_5': {
            'bet': 'Over 0.5',
            'actual_col': 'y_OU_0_5',
            'pred_col': 'BLEND_OU_0_5_O',
            'winning_outcome': 'O',
            'odds': 1.08,
            'stake': 20.0  # Safest - higher stake
        },
        'OU_1_5': {
            'bet': 'Over 1.5',
            'actual_col': 'y_OU_1_5',
            'pred_col': 'BLEND_OU_1_5_O',
            'winning_outcome': 'O',
            'odds': 1.25,
            'stake': 15.0  # Safe - medium-high stake
        },
        'OU_2_5': {
            'bet': 'Over 2.5',
            'actual_col': 'y_OU_2_5',
            'pred_col': 'BLEND_OU_2_5_O',
            'winning_outcome': 'O',
            'odds': 1.4,
            'stake': 10.0  # Medium risk - medium stake
        },
        'OU_3_5': {
            'bet': 'Over 3.5',
            'actual_col': 'y_OU_3_5',
            'pred_col': 'BLEND_OU_3_5_O',
            'winning_outcome': 'O',
            'odds': 1.85,
            'stake': 5.0   # Riskier - lower stake
        },
        'BTTS': {
            'bet': 'BTTS Yes',
            'actual_col': 'y_BTTS',
            'pred_col': 'BLEND_BTTS_Y',
            'winning_outcome': 'Y',
            'odds': 1.5,
            'stake': 8.0   # Medium-high risk - lower stake
        }
    }
    
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 min_confidence: float = 0.65):
        """
        Args:
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD
            min_confidence: Minimum prediction confidence (0.65 = 65%)
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.min_confidence = min_confidence
        
        self.weekly_results = []
        self.all_doubles = []
        
        # Create backtest-specific output directory
        self.backtest_dir = OUTPUT_DIR / "backtest_outputs"
        self.backtest_dir.mkdir(exist_ok=True)
        
        # Backup existing weekly_bets_lite.csv
        self.backup_existing_outputs()
        
    def load_features_data(self) -> pd.DataFrame:
        """Load historical data"""
        try:
            if not FEATURES_PARQUET.exists():
                raise FileNotFoundError("Run build_features(force=True) first")
            
            print(f" Loading features...")
            df = pd.read_parquet(FEATURES_PARQUET)
            df['Date'] = pd.to_datetime(df['Date'])
            print(f" Loaded {len(df)} matches")
            return df
        except Exception as e:
            print(f" Error loading data: {e}")
            raise
    
    def backup_existing_outputs(self):
        """Backup existing weekly_bets_lite.csv before backtesting"""
        try:
            weekly_bets_lite = OUTPUT_DIR / "weekly_bets_lite.csv"
            if weekly_bets_lite.exists():
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = OUTPUT_DIR / f"weekly_bets_lite_backup_{timestamp}.csv"
                shutil.copy(weekly_bets_lite, backup_path)
                print(f" Backed up existing weekly_bets_lite.csv to {backup_path.name}")
        except Exception as e:
            print(f" Could not backup weekly_bets_lite.csv: {e}")
    
    def get_weekly_periods(self) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Generate weekly test periods"""
        periods = []
        current = self.start_date
        
        while current <= self.end_date:
            week_end = current + timedelta(days=7)
            periods.append((current, week_end))
            current = week_end
        
        return periods
    
    def train_models(self, train_df: pd.DataFrame) -> bool:
        """Train models on historical data"""
        try:
            print("    Preparing training data...")
            
            # Fill missing values
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if train_df[col].isna().sum() > 0:
                    train_df[col] = train_df[col].fillna(train_df[col].median())
            train_df = train_df.fillna(0)
            
            # Backup and replace features file
            temp_features = DATA_DIR / "temp_backtest_features.parquet"
            backup_features = DATA_DIR / "original_features_backup.parquet"
            
            train_df.to_parquet(temp_features)
            
            if FEATURES_PARQUET.exists():
                shutil.copy(FEATURES_PARQUET, backup_features)
            
            shutil.copy(temp_features, FEATURES_PARQUET)
            
            try:
                from models import train_all_targets
                models = train_all_targets(MODEL_ARTIFACTS_DIR)
                success = len(models) > 0
                
                # Restore
                if backup_features.exists():
                    shutil.copy(backup_features, FEATURES_PARQUET)
                
                temp_features.unlink(missing_ok=True)
                backup_features.unlink(missing_ok=True)
                
                return success
            except Exception as e:
                print(f"    Training failed: {e}")
                if backup_features.exists():
                    shutil.copy(backup_features, FEATURES_PARQUET)
                return False
                
        except Exception as e:
            print(f"    Error in training: {e}")
            return False
    
    def generate_predictions(self, test_df: pd.DataFrame, week_num: int) -> pd.DataFrame:
        """Generate predictions for test week"""
        try:
            fixtures = test_df[['Date', 'League', 'HomeTeam', 'AwayTeam']].copy()
            temp_fixtures = self.backtest_dir / f"temp_fixtures_week_{week_num}.csv"
            fixtures.to_csv(temp_fixtures, index=False)
            
            try:
                from predict import predict_week
                predict_week(temp_fixtures)
                
                # Read from temporary location
                predictions_file = OUTPUT_DIR / "weekly_bets_lite.csv"
                
                if predictions_file.exists():
                    predictions = pd.read_csv(predictions_file)
                    
                    # Save this week's predictions to backtest folder
                    week_preds_path = self.backtest_dir / f"predictions_week_{week_num}.csv"
                    predictions.to_csv(week_preds_path, index=False)
                    
                    test_with_preds = test_df.merge(
                        predictions,
                        on=['Date', 'League', 'HomeTeam', 'AwayTeam'],
                        how='left'
                    )
                    temp_fixtures.unlink(missing_ok=True)
                    return test_with_preds
                else:
                    print("    No predictions generated")
                    return test_df
                    
            except Exception as e:
                print(f"    Prediction error: {e}")
                temp_fixtures.unlink(missing_ok=True)
                return test_df
                
        except Exception as e:
            print(f"    Error generating predictions: {e}")
            return test_df
    
    def find_confident_bets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to high-confidence bets in YOUR markets"""
        confident_bets = []
        
        for idx, row in df.iterrows():
            for market_key, market_info in self.MARKETS.items():
                pred_col = market_info['pred_col']
                actual_col = market_info['actual_col']
                
                # Skip if columns missing
                if pred_col not in df.columns or actual_col not in df.columns:
                    continue
                
                confidence = row.get(pred_col)
                actual_outcome = row.get(actual_col)
                
                # Skip if no prediction or actual
                if pd.isna(confidence) or pd.isna(actual_outcome):
                    continue
                
                # Convert confidence if string
                if isinstance(confidence, str):
                    confidence = float(confidence.strip('%')) / 100
                
                # Only include if meets confidence threshold
                if confidence >= self.min_confidence:
                    won = (actual_outcome == market_info['winning_outcome'])
                    
                    confident_bets.append({
                        'Date': row['Date'],
                        'League': row.get('League', 'Unknown'),
                        'Match': f"{row.get('HomeTeam', 'Home')} vs {row.get('AwayTeam', 'Away')}",
                        'Market': market_key,
                        'Bet': market_info['bet'],
                        'Confidence': confidence,
                        'Odds': market_info['odds'],
                        'Stake': market_info['stake'],  # Individual stake per market
                        'Won': won,
                        'Actual': actual_outcome
                    })
        
        return pd.DataFrame(confident_bets)
    
    def build_doubles(self, bets_df: pd.DataFrame) -> List[Dict]:
        """Build all possible doubles from confident bets"""
        doubles = []
        
        # Group by date to ensure same-day doubles
        for date, day_bets in bets_df.groupby('Date'):
            if len(day_bets) < 2:
                continue
            
            # Generate all doubles for this day
            for (idx1, bet1), (idx2, bet2) in combinations(day_bets.iterrows(), 2):
                # Skip if same market (diversify)
                if bet1['Market'] == bet2['Market']:
                    continue
                
                combined_odds = bet1['Odds'] * bet2['Odds']
                both_won = bet1['Won'] and bet2['Won']
                
                # Calculate stake - average of both leg stakes
                double_stake = (bet1['Stake'] + bet2['Stake']) / 2
                
                profit = (double_stake * combined_odds - double_stake) if both_won else -double_stake
                
                doubles.append({
                    'Date': date,
                    'Leg1': f"{bet1['Match']} - {bet1['Bet']}",
                    'Leg1_Conf': bet1['Confidence'],
                    'Leg1_Stake': bet1['Stake'],
                    'Leg1_Won': bet1['Won'],
                    'Leg2': f"{bet2['Match']} - {bet2['Bet']}",
                    'Leg2_Conf': bet2['Confidence'],
                    'Leg2_Stake': bet2['Stake'],
                    'Leg2_Won': bet2['Won'],
                    'Combined_Odds': round(combined_odds, 2),
                    'Avg_Confidence': round((bet1['Confidence'] + bet2['Confidence']) / 2, 3),
                    'Won': both_won,
                    'Stake': round(double_stake, 2),
                    'Profit': round(profit, 2)
                })
        
        return doubles
    
    def run_backtest(self):
        """Run realistic backtest"""
        try:
            print("\n REALISTIC BACKTEST - YOUR BETTING STYLE")
            print("="*60)
            print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
            print(f"Strategy: DOUBLES ONLY")
            print(f"Markets & Stakes:")
            for market, info in self.MARKETS.items():
                print(f"   • {info['bet']}: £{info['stake']} stake @ {info['odds']} odds")
            print(f"Min confidence: {self.min_confidence*100:.0f}%")
            print("="*60)
            
            full_df = self.load_features_data()
            full_df = full_df[
                (full_df['Date'] >= self.start_date) & 
                (full_df['Date'] <= self.end_date)
            ]
            
            periods = self.get_weekly_periods()
            print(f"\n Testing {len(periods)} weeks\n")
            
            total_doubles = 0
            total_won = 0
            total_staked = 0
            total_returns = 0
            
            for i, (week_start, week_end) in enumerate(periods, 1):
                try:
                    print(f"Week {i}/{len(periods)}: {week_start.date()} to {week_end.date()}")
                    
                    # Split train/test
                    train_df = full_df[full_df['Date'] < week_start].copy()
                    test_df = full_df[(full_df['Date'] >= week_start) & (full_df['Date'] < week_end)].copy()
                    
                    if len(train_df) < 100 or len(test_df) == 0:
                        print(f"    Insufficient data")
                        continue
                    
                    print(f"    {len(test_df)} matches this week")
                    
                    # Train
                    print(f"    Training...")
                    if not self.train_models(train_df):
                        print(f"    Training failed")
                        continue
                    
                    # Predict
                    print(f"    Predicting...")
                    test_with_preds = self.generate_predictions(test_df, i)
                    
                    # Find confident bets
                    confident_bets = self.find_confident_bets(test_with_preds)
                    
                    if len(confident_bets) < 2:
                        print(f"   ℹ Not enough confident bets ({len(confident_bets)})")
                        continue
                    
                    # Build doubles
                    week_doubles = self.build_doubles(confident_bets)
                    
                    if not week_doubles:
                        print(f"   ℹ No valid doubles")
                        continue
                    
                    # Stats for this week
                    week_won = sum(1 for d in week_doubles if d['Won'])
                    week_profit = sum(d['Profit'] for d in week_doubles)
                    week_staked = sum(d['Stake'] for d in week_doubles)
                    
                    total_doubles += len(week_doubles)
                    total_won += week_won
                    total_staked += week_staked
                    total_returns += sum(d['Stake'] * d['Combined_Odds'] for d in week_doubles if d['Won'])
                    
                    print(f"    {len(week_doubles)} doubles | {week_won} won | £{week_profit:+.2f}")
                    
                    self.all_doubles.extend(week_doubles)
                    
                    self.weekly_results.append({
                        'week_start': week_start,
                        'week_end': week_end,
                        'num_doubles': len(week_doubles),
                        'won': week_won,
                        'profit': week_profit
                    })
                    
                except Exception as e:
                    print(f"    Error in week {i}: {e}")
                    continue
            
            # Generate summary
            self.print_summary(total_doubles, total_won, total_staked, total_returns)
            self.save_results()
            self.restore_outputs()
            
        except Exception as e:
            print(f" Backtest error: {e}")
            self.restore_outputs()
    
    def print_summary(self, total_doubles, total_won, total_staked, total_returns):
        """Print final results"""
        print("\n" + "="*60)
        print(" BACKTEST RESULTS")
        print("="*60)
        
        if total_doubles == 0:
            print(" No doubles placed")
            return
        
        hit_rate = (total_won / total_doubles) * 100
        total_profit = total_returns - total_staked
        roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0
        
        print(f"\n FINANCIAL SUMMARY:")
        print(f"   Total Doubles: {total_doubles}")
        print(f"   Won: {total_won} ({hit_rate:.1f}%)")
        print(f"   Total Staked: £{total_staked:.2f}")
        print(f"   Total Returns: £{total_returns:.2f}")
        print(f"   Net Profit: £{total_profit:+.2f}")
        print(f"   ROI: {roi:+.1f}%")
        
        # Profitability assessment
        print(f"\n VERDICT:")
        if total_profit > 0:
            print(f"    PROFITABLE - Would have made £{total_profit:.2f}")
            print(f"    Average profit per week: £{total_profit/len(self.weekly_results):.2f}")
        else:
            print(f"    LOSING - Would have lost £{abs(total_profit):.2f}")
            print(f"    Strategy needs adjustment")
        
        # Best doubles
        if self.all_doubles:
            doubles_df = pd.DataFrame(self.all_doubles)
            won_doubles = doubles_df[doubles_df['Won']]
            
            if len(won_doubles) > 0:
                print(f"\n BEST WINNING DOUBLES:")
                best = won_doubles.nlargest(5, 'Profit')
                for idx, row in best.iterrows():
                    print(f"   • {row['Date'].date()} - {row['Combined_Odds']} odds - £{row['Profit']:.2f} profit")
                    print(f"     {row['Leg1']}")
                    print(f"     {row['Leg2']}")
        
        print("\n" + "="*60)
    
    def save_results(self):
        """Save detailed results to backtest folder"""
        try:
            if self.all_doubles:
                doubles_df = pd.DataFrame(self.all_doubles)
                output_path = self.backtest_dir / "realistic_doubles.csv"
                doubles_df.to_csv(output_path, index=False)
                print(f" Saved: {output_path}")
            
            if self.weekly_results:
                weekly_df = pd.DataFrame(self.weekly_results)
                output_path = self.backtest_dir / "realistic_weekly.csv"
                weekly_df.to_csv(output_path, index=False)
                print(f" Saved: {output_path}")
                
        except Exception as e:
            print(f" Error saving results: {e}")
    
    def restore_outputs(self):
        """Clean up backtest temporary files"""
        try:
            print(f"\n Cleaning up...")
            
            # Remove weekly_bets_lite.csv created during backtest
            weekly_bets_lite = OUTPUT_DIR / "weekly_bets_lite.csv"
            if weekly_bets_lite.exists():
                # Check if it's a backtest file by seeing if backtest_outputs exists
                latest_backup = None
                for backup in OUTPUT_DIR.glob("weekly_bets_lite_backup_*.csv"):
                    if latest_backup is None or backup.stat().st_mtime > latest_backup.stat().st_mtime:
                        latest_backup = backup
                
                if latest_backup:
                    shutil.copy(latest_backup, weekly_bets_lite)
                    print(f" Restored original weekly_bets_lite.csv from backup")
                else:
                    weekly_bets_lite.unlink()
                    print(f" Removed backtest weekly_bets_lite.csv")
            
            print(f" All backtest files saved to: {self.backtest_dir}")
            
        except Exception as e:
            print(f" Error during cleanup: {e}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    from datetime import datetime
    
    try:
        print("\n REALISTIC BACKTEST - YOUR BETTING STYLE")
        print("="*60)
        print("Default Stakes (adjust below if needed):")
        print("  • Over 0.5: £20 (safest)")
        print("  • Over 1.5: £15")
        print("  • Over 2.5: £10")
        print("  • BTTS: £8")
        print("  • Over 3.5: £5 (riskiest)")
        print("="*60)
        
        # Get period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        print(f"\nDefault: Last 6 months ({start_date.date()} to {end_date.date()})")
        choice = input("Use default? (y/n, default=y): ").strip().lower()
        
        if choice == 'n':
            start_input = input("Start date (YYYY-MM-DD): ").strip()
            end_input = input("End date (YYYY-MM-DD): ").strip()
            start_date = datetime.strptime(start_input, '%Y-%m-%d')
            end_date = datetime.strptime(end_input, '%Y-%m-%d')
        
        # Get min confidence
        conf_input = input("\nMin confidence % (default=65): ").strip()
        min_conf = float(conf_input) / 100 if conf_input else 0.65
        
        # Run backtest
        backtest = RealisticBacktest(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            min_confidence=min_conf
        )
        
        backtest.run_backtest()
        
        print("\n COMPLETE!")
        print(" All backtest files saved to: outputs/backtest_outputs/")
        print("   • realistic_doubles.csv - All doubles placed")
        print("   • realistic_weekly.csv - Week-by-week profit")
        print("   • predictions_week_X.csv - Predictions for each week")
        print("\n Your original weekly_bets_lite.csv was preserved!")
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        input("\nPress Enter to exit...")
