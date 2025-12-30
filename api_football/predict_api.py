# predict_api.py
"""
Prediction Module for API-Football Data
Uses xG-integrated DC model with all enhanced features

Usage:
    python predict_api.py                    # Predict upcoming fixtures
    python predict_api.py --date 2024-01-15  # Predict specific date
    python predict_api.py --fixture 123456   # Predict specific fixture
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3
import json
import logging

from api_football_client import APIFootballClient, LEAGUES, get_league_code
from models_dc_xg import fit_all_xg, price_match_xg, DCParamsXG
from injury_tracker import InjuryTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "football_api.db"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"
OUTPUT_DIR = Path("outputs")


class MatchPredictor:
    """
    Predict match outcomes using xG-integrated DC model
    """
    
    def __init__(self, api_key: str = None, features_path: Path = None):
        """
        Initialize predictor
        
        Args:
            api_key: API-Football key (for fetching upcoming fixtures)
            features_path: Path to features parquet (for model fitting)
        """
        self.api_key = api_key
        self.client = APIFootballClient(api_key) if api_key else None
        self.features_path = features_path or FEATURES_PARQUET
        
        self.params_by_league: Dict[str, DCParamsXG] = {}
        self.injury_tracker = InjuryTracker()
        
        self._load_and_fit()
    
    def _load_and_fit(self):
        """Load features and fit DC models"""
        if not self.features_path.exists():
            logger.warning(f"Features file not found: {self.features_path}")
            return
        
        logger.info(f"Loading features from {self.features_path}")
        self.df = pd.read_parquet(self.features_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        logger.info(f"Fitting DC models for {self.df['League'].nunique()} leagues...")
        self.params_by_league = fit_all_xg(self.df, use_xg=True)
        
        logger.info(f"Fitted models for: {list(self.params_by_league.keys())}")
    
    def _get_team_features(self, team: str, league: str, as_of_date: datetime) -> Dict:
        """
        Get rolling features for a team as of a specific date
        """
        # Filter to matches before this date
        team_home = self.df[
            (self.df['HomeTeam'] == team) & 
            (self.df['Date'] < as_of_date) &
            (self.df['League'] == league)
        ].tail(10)
        
        team_away = self.df[
            (self.df['AwayTeam'] == team) & 
            (self.df['Date'] < as_of_date) &
            (self.df['League'] == league)
        ].tail(10)
        
        features = {}
        
        # xG features
        xg_cols = ['home_xG_for_ma5', 'home_xG_against_ma5', 'home_xG_overperformance']
        for col in xg_cols:
            if col in team_home.columns and len(team_home) > 0:
                val = team_home[col].iloc[-1]
                if pd.notna(val):
                    features[col.replace('home_', '')] = val
        
        # Form features
        form_cols = ['home_attack_quality', 'home_scoring_streak']
        for col in form_cols:
            if col in team_home.columns and len(team_home) > 0:
                val = team_home[col].iloc[-1]
                if pd.notna(val):
                    features[col.replace('home_', '')] = val
        
        return features
    
    def _get_h2h_features(self, home: str, away: str, league: str) -> Dict:
        """Get H2H features for a matchup"""
        # Find past meetings
        h2h_home = self.df[
            (self.df['HomeTeam'] == home) & 
            (self.df['AwayTeam'] == away)
        ]
        h2h_away = self.df[
            (self.df['HomeTeam'] == away) & 
            (self.df['AwayTeam'] == home)
        ]
        
        h2h = pd.concat([h2h_home, h2h_away])
        
        if len(h2h) < 3:
            return {}
        
        total_goals = (h2h['FTHG'] + h2h['FTAG']).mean()
        btts_rate = ((h2h['FTHG'] > 0) & (h2h['FTAG'] > 0)).mean()
        
        return {
            'h2h_total_goals_avg': total_goals,
            'h2h_btts_rate': btts_rate,
            'h2h_meetings': len(h2h)
        }
    
    def _get_rest_days(self, team: str, match_date: datetime) -> Optional[int]:
        """Calculate rest days for a team"""
        # Find last match (home or away)
        last_home = self.df[
            (self.df['HomeTeam'] == team) & 
            (self.df['Date'] < match_date)
        ].tail(1)
        
        last_away = self.df[
            (self.df['AwayTeam'] == team) & 
            (self.df['Date'] < match_date)
        ].tail(1)
        
        last_dates = []
        if len(last_home) > 0:
            last_dates.append(last_home['Date'].iloc[0])
        if len(last_away) > 0:
            last_dates.append(last_away['Date'].iloc[0])
        
        if not last_dates:
            return 14  # Default
        
        last_match = max(last_dates)
        rest_days = (match_date - last_match).days
        
        return max(1, min(14, rest_days))
    
    def predict_fixture(self, home: str, away: str, league: str,
                        match_date: datetime = None,
                        home_team_id: int = None,
                        away_team_id: int = None,
                        is_cup: bool = None) -> Dict:
        """
        Predict a single fixture
        
        Args:
            home: Home team name
            away: Away team name
            league: League code (e.g., 'E0')
            match_date: Date of match (default: today)
            home_team_id: Team ID for injury lookup
            away_team_id: Team ID for injury lookup
            is_cup: Whether it's a cup match
        
        Returns:
            Dict with predictions and probabilities
        """
        match_date = match_date or datetime.now()
        
        # Check if we have model for this league
        if league not in self.params_by_league:
            # Try to find closest league or use default
            logger.warning(f"No model for {league}, using E0 as fallback")
            params = self.params_by_league.get('E0')
            if not params:
                return {'error': f'No model available for {league}'}
        else:
            params = self.params_by_league[league]
        
        # Determine if cup
        if is_cup is None:
            is_cup = LEAGUES.get(league, {}).get('type') == 'cup'
        
        # Get features
        home_features = self._get_team_features(home, league, match_date)
        away_features = self._get_team_features(away, league, match_date)
        h2h_features = self._get_h2h_features(home, away, league)
        
        # Rest days
        home_rest = self._get_rest_days(home, match_date)
        away_rest = self._get_rest_days(away, match_date)
        
        # Injury features
        injury_features = {}
        if home_team_id and away_team_id:
            injury_features = self.injury_tracker.get_fixture_injury_features(
                home_team_id, away_team_id, match_date
            )

        # Build feature dict for price_match - only include supported parameters
        features = {
            'home_rest_days': home_rest,
            'away_rest_days': away_rest,
            'home_xG_for_ma5': home_features.get('xG_for_ma5'),
            'away_xG_for_ma5': away_features.get('xG_for_ma5'),
            'home_xG_overperformance': home_features.get('xG_overperformance'),
            'away_xG_overperformance': away_features.get('xG_overperformance'),
            'home_attack_quality': home_features.get('attack_quality'),
            'away_attack_quality': away_features.get('away_attack_quality'),
            'h2h_total_goals_avg': h2h_features.get('h2h_total_goals_avg'),
            'h2h_btts_rate': h2h_features.get('h2h_btts_rate'),
            'is_cup_match': 1 if is_cup else 0,
            # Only include injury counts, not multipliers
            'home_injuries_count': injury_features.get('home_injuries_count', 0),
            'away_injuries_count': injury_features.get('away_injuries_count', 0),
            'home_key_injuries': injury_features.get('home_key_injuries', 0),
            'away_key_injuries': injury_features.get('away_key_injuries', 0),
        }
        
        # Get probabilities
        probs = price_match_xg(params, home, away, **features)
        
        if not probs:
            return {'error': f'Could not price {home} vs {away}'}
        
        # Build prediction result
        result = {
            'home': home,
            'away': away,
            'league': league,
            'date': match_date.isoformat() if match_date else None,
            'is_cup': is_cup,
            # Probabilities
            'btts_yes': probs.get('DC_BTTS_Y', 0),
            'btts_no': probs.get('DC_BTTS_N', 0),
            'over_2_5': probs.get('DC_OU_2_5_O', 0),
            'under_2_5': probs.get('DC_OU_2_5_U', 0),
            'over_1_5': probs.get('DC_OU_1_5_O', 0),
            'under_1_5': probs.get('DC_OU_1_5_U', 0),
            'over_3_5': probs.get('DC_OU_3_5_O', 0),
            'under_3_5': probs.get('DC_OU_3_5_U', 0),
            'home_win': probs.get('DC_1X2_H', 0),
            'draw': probs.get('DC_1X2_D', 0),
            'away_win': probs.get('DC_1X2_A', 0),
            # Expected goals
            'expected_home_goals': probs.get('expected_home_goals', 0),
            'expected_away_goals': probs.get('expected_away_goals', 0),
            'expected_total': probs.get('expected_total_goals', 0),
            # Features used
            'features': features,
            'h2h_meetings': h2h_features.get('h2h_meetings', 0),
        }
        
        return result
    
    def predict_date(self, date: str = None) -> List[Dict]:
        """
        Predict all fixtures for a date
        
        Args:
            date: Date string (YYYY-MM-DD) or None for today
        
        Returns:
            List of prediction dicts
        """
        if date:
            target_date = pd.to_datetime(date)
        else:
            target_date = pd.Timestamp.now().normalize()
        
        predictions = []
        
        # Check if we have fixtures in database for this date
        with sqlite3.connect(DB_PATH) as conn:
            fixtures = pd.read_sql_query(f"""
                SELECT 
                    fixture_id,
                    league_code,
                    home_team,
                    away_team,
                    home_team_id,
                    away_team_id,
                    date,
                    league_type
                FROM fixtures
                WHERE date = '{target_date.strftime('%Y-%m-%d')}'
                AND home_goals IS NULL
                ORDER BY league_code, date
            """, conn)
        
        if fixtures.empty:
            logger.warning(f"No fixtures found for {target_date.date()}")
            return predictions
        
        logger.info(f"Predicting {len(fixtures)} fixtures for {target_date.date()}")
        
        for _, row in fixtures.iterrows():
            pred = self.predict_fixture(
                home=row['home_team'],
                away=row['away_team'],
                league=row['league_code'],
                match_date=target_date,
                home_team_id=row['home_team_id'],
                away_team_id=row['away_team_id'],
                is_cup=row['league_type'] == 'cup'
            )
            pred['fixture_id'] = row['fixture_id']
            predictions.append(pred)
        
        return predictions
    
    def predict_upcoming(self, days: int = 7, leagues: List[str] = None) -> pd.DataFrame:
        """
        Predict upcoming fixtures using API
        
        Args:
            days: Number of days ahead to predict
            leagues: List of league codes to include
        
        Returns:
            DataFrame with predictions
        """
        if not self.client:
            logger.error("API client not available. Provide api_key.")
            return pd.DataFrame()
        
        leagues = leagues or list(LEAGUES.keys())
        all_predictions = []

        from_date = datetime.now().strftime('%Y-%m-%d')
        to_date = (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')

        # Determine current season
        current_month = datetime.now().month
        current_year = datetime.now().year
        season = current_year if current_month >= 8 else current_year - 1

        for league_code in leagues:
            league_info = LEAGUES.get(league_code)
            if not league_info:
                continue

            logger.info(f"Fetching fixtures for {league_code}...")

            try:
                # Get upcoming fixtures from API
                fixtures = self.client.get_fixtures(
                    league_id=league_info['id'],
                    season=season,  # REQUIRED: API needs season parameter
                    from_date=from_date,
                    to_date=to_date,
                    status='NS'  # Not started
                )
            except Exception as e:
                logger.error(f"API Error: {e}")
                continue
            
            for fixture in fixtures:
                f = fixture['fixture']
                teams = fixture['teams']
                
                pred = self.predict_fixture(
                    home=teams['home']['name'],
                    away=teams['away']['name'],
                    league=league_code,
                    match_date=pd.to_datetime(f['date'][:10]),
                    home_team_id=teams['home']['id'],
                    away_team_id=teams['away']['id'],
                    is_cup=league_info['type'] == 'cup'
                )
                
                pred['fixture_id'] = f['id']
                pred['kickoff'] = f['date']
                all_predictions.append(pred)
        
        df = pd.DataFrame(all_predictions)
        
        if not df.empty:
            # Sort by date and expected value
            df = df.sort_values(['date', 'btts_yes'], ascending=[True, False])
        
        return df
    
    def save_predictions(self, predictions: pd.DataFrame, output_path: Path = None):
        """Save predictions to file"""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        if output_path is None:
            date_str = datetime.now().strftime('%Y%m%d_%H%M')
            output_path = OUTPUT_DIR / f"predictions_{date_str}.csv"
        
        predictions.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to {output_path}")
        
        # Also save JSON for web consumption
        json_path = output_path.with_suffix('.json')
        predictions.to_json(json_path, orient='records', indent=2)
        
        return output_path


def format_prediction(pred: Dict) -> str:
    """Format a prediction for display"""
    if 'error' in pred:
        return f"{pred.get('home', '?')} vs {pred.get('away', '?')}: {pred['error']}"
    
    lines = [
        f"\n{'='*60}",
        f"{pred['home']} vs {pred['away']}",
        f"League: {pred['league']} {'(CUP)' if pred.get('is_cup') else ''}",
        f"Date: {pred.get('date', 'N/A')}",
        f"{'='*60}",
        f"\nExpected Goals: {pred['expected_home_goals']:.2f} - {pred['expected_away_goals']:.2f}",
        f"Expected Total: {pred['expected_total']:.2f}",
        f"\nBTTS:",
        f"  Yes: {pred['btts_yes']:.1%}",
        f"  No:  {pred['btts_no']:.1%}",
        f"\nOver/Under 2.5:",
        f"  Over:  {pred['over_2_5']:.1%}",
        f"  Under: {pred['under_2_5']:.1%}",
        f"\n1X2:",
        f"  Home: {pred['home_win']:.1%}",
        f"  Draw: {pred['draw']:.1%}",
        f"  Away: {pred['away_win']:.1%}",
    ]
    
    if pred.get('h2h_meetings', 0) > 0:
        lines.append(f"\nH2H Meetings: {pred['h2h_meetings']}")
    
    return '\n'.join(lines)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Predict match outcomes")
    parser.add_argument("--api-key", default=os.environ.get("API_FOOTBALL_KEY"),
                        help="API key (or set API_FOOTBALL_KEY env var)")
    parser.add_argument("--date", help="Predict fixtures for specific date (YYYY-MM-DD)")
    parser.add_argument("--fixture", type=int, help="Predict specific fixture ID")
    parser.add_argument("--home", help="Home team name")
    parser.add_argument("--away", help="Away team name")
    parser.add_argument("--league", default="E0", help="League code")
    parser.add_argument("--upcoming", type=int, default=7, help="Days ahead to predict")
    parser.add_argument("--output", type=Path, help="Output file path")
    
    args = parser.parse_args()
    
    predictor = MatchPredictor(api_key=args.api_key)
    
    if args.home and args.away:
        # Single match prediction
        pred = predictor.predict_fixture(
            home=args.home,
            away=args.away,
            league=args.league,
            match_date=pd.to_datetime(args.date) if args.date else None
        )
        print(format_prediction(pred))
        
    elif args.date:
        # Predict all fixtures for a date
        preds = predictor.predict_date(args.date)
        for pred in preds:
            print(format_prediction(pred))
        
        if preds:
            df = pd.DataFrame(preds)
            predictor.save_predictions(df, args.output)
    
    elif args.api_key:
        # Predict upcoming using API
        df = predictor.predict_upcoming(days=args.upcoming)
        
        if not df.empty:
            print(f"\nTop 10 BTTS Yes predictions:")
            print("-"*60)
            top_btts = df.nlargest(10, 'btts_yes')
            for _, row in top_btts.iterrows():
                print(f"{row['home']} vs {row['away']}: {row['btts_yes']:.1%} ({row['league']})")
            
            print(f"\nTop 10 Over 2.5 predictions:")
            print("-"*60)
            top_over = df.nlargest(10, 'over_2_5')
            for _, row in top_over.iterrows():
                print(f"{row['home']} vs {row['away']}: {row['over_2_5']:.1%} ({row['league']})")
            
            predictor.save_predictions(df, args.output)
        else:
            print("No predictions generated")
    
    else:
        print("Usage examples:")
        print("  python predict_api.py --home Arsenal --away Chelsea --league E0")
        print("  python predict_api.py --date 2024-01-15")
        print("  python predict_api.py --api-key YOUR_KEY --upcoming 7")
