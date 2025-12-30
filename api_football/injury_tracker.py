# injury_tracker.py
"""
Injury Impact Tracker
Calculates the impact of injuries on team performance

Features:
- Track current injuries by team
- Estimate impact on attack/defence
- Identify key player absences
- Historical injury patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "football_api.db"


class InjuryTracker:
    """
    Track and analyze injury impacts on team performance
    """
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._load_data()
    
    def _get_conn(self):
        return sqlite3.connect(self.db_path)
    
    def _load_data(self):
        """Load injury and player data from database"""
        try:
            with self._get_conn() as conn:
                # Load injuries
                self.injuries_df = pd.read_sql_query("""
                    SELECT * FROM injuries 
                    ORDER BY date DESC
                """, conn)
                
                if not self.injuries_df.empty:
                    self.injuries_df['date'] = pd.to_datetime(self.injuries_df['date'])
                
                # Load player stats if available
                try:
                    self.player_stats_df = pd.read_sql_query("""
                        SELECT 
                            player_id,
                            player_name,
                            team_id,
                            SUM(goals) as total_goals,
                            SUM(assists) as total_assists,
                            AVG(rating) as avg_rating,
                            COUNT(*) as matches_played
                        FROM player_fixture_stats
                        GROUP BY player_id, player_name, team_id
                    """, conn)
                except:
                    self.player_stats_df = pd.DataFrame()
                
                logger.info(f"Loaded {len(self.injuries_df)} injury records")
                
        except Exception as e:
            logger.warning(f"Could not load injury data: {e}")
            self.injuries_df = pd.DataFrame()
            self.player_stats_df = pd.DataFrame()
    
    def get_team_injuries(self, team_id: int, as_of_date: datetime = None,
                          lookback_days: int = 30) -> List[Dict]:
        """
        Get current injuries for a team
        
        Args:
            team_id: Team ID
            as_of_date: Date to check injuries for (default: now)
            lookback_days: How far back to consider injuries
        
        Returns:
            List of injury records
        """
        if self.injuries_df.empty:
            return []
        
        as_of_date = as_of_date or datetime.now()
        cutoff_date = as_of_date - timedelta(days=lookback_days)
        
        mask = (
            (self.injuries_df['team_id'] == team_id) &
            (self.injuries_df['date'] >= cutoff_date) &
            (self.injuries_df['date'] <= as_of_date)
        )
        
        injuries = self.injuries_df[mask].to_dict('records')
        return injuries
    
    def get_injury_count(self, team_id: int, as_of_date: datetime = None) -> int:
        """Get count of current injuries for team"""
        injuries = self.get_team_injuries(team_id, as_of_date)
        return len(injuries)
    
    def is_key_player_injured(self, team_id: int, as_of_date: datetime = None) -> Tuple[bool, List[str]]:
        """
        Check if any key players are injured
        
        Key players defined as:
        - Top 3 goal scorers
        - Top 3 by assists
        - Players with avg rating > 7.0
        
        Returns:
            Tuple of (has_key_injury, list of injured key player names)
        """
        injuries = self.get_team_injuries(team_id, as_of_date)
        
        if not injuries or self.player_stats_df.empty:
            return False, []
        
        # Get team's key players
        team_stats = self.player_stats_df[self.player_stats_df['team_id'] == team_id]
        
        if team_stats.empty:
            return False, []
        
        # Top scorers
        top_scorers = set(team_stats.nlargest(3, 'total_goals')['player_id'].values)
        
        # Top assists
        top_assists = set(team_stats.nlargest(3, 'total_assists')['player_id'].values)
        
        # High rated players
        high_rated = set(team_stats[team_stats['avg_rating'] > 7.0]['player_id'].values)
        
        key_players = top_scorers | top_assists | high_rated
        
        # Check injuries
        injured_key = []
        for injury in injuries:
            if injury['player_id'] in key_players:
                injured_key.append(injury['player_name'])
        
        return len(injured_key) > 0, injured_key
    
    def calculate_injury_impact(self, team_id: int, as_of_date: datetime = None) -> Dict[str, float]:
        """
        Calculate estimated impact of injuries on team performance
        
        Returns:
            Dict with:
            - attack_impact: Multiplier for attack (1.0 = no impact, 0.8 = 20% reduction)
            - defence_impact: Multiplier for defence
            - overall_impact: Combined impact
            - injury_count: Number of injuries
            - key_injuries: Number of key players injured
        """
        injuries = self.get_team_injuries(team_id, as_of_date)
        
        result = {
            'attack_impact': 1.0,
            'defence_impact': 1.0,
            'overall_impact': 1.0,
            'injury_count': len(injuries),
            'key_injuries': 0,
            'injured_players': []
        }
        
        if not injuries:
            return result
        
        # Base impact from injury count
        if len(injuries) >= 5:
            result['overall_impact'] *= 0.90
        elif len(injuries) >= 3:
            result['overall_impact'] *= 0.95
        
        # Check for key player injuries
        has_key, key_names = self.is_key_player_injured(team_id, as_of_date)
        result['key_injuries'] = len(key_names)
        result['injured_players'] = [i['player_name'] for i in injuries]
        
        if len(key_names) >= 2:
            result['attack_impact'] *= 0.85
        elif len(key_names) >= 1:
            result['attack_impact'] *= 0.90
        
        # Combine impacts
        result['overall_impact'] *= result['attack_impact'] * result['defence_impact']
        
        return result
    
    def get_fixture_injury_features(self, home_team_id: int, away_team_id: int,
                                     fixture_date: datetime) -> Dict[str, float]:
        """
        Get injury-related features for a fixture
        
        Returns dict ready to pass to price_match()
        """
        home_impact = self.calculate_injury_impact(home_team_id, fixture_date)
        away_impact = self.calculate_injury_impact(away_team_id, fixture_date)
        
        return {
            'home_injuries_count': home_impact['injury_count'],
            'away_injuries_count': away_impact['injury_count'],
            'home_key_injuries': home_impact['key_injuries'],
            'away_key_injuries': away_impact['key_injuries'],
            'home_injury_attack_mult': home_impact['attack_impact'],
            'away_injury_attack_mult': away_impact['attack_impact'],
        }
    
    def get_team_injury_history(self, team_id: int, season: int = None) -> pd.DataFrame:
        """Get historical injury data for a team"""
        if self.injuries_df.empty:
            return pd.DataFrame()
        
        mask = self.injuries_df['team_id'] == team_id
        
        if season:
            mask &= self.injuries_df['season'] == season
        
        return self.injuries_df[mask].sort_values('date', ascending=False)
    
    def analyze_injury_patterns(self, team_id: int) -> Dict:
        """
        Analyze injury patterns for a team
        
        Returns:
            Stats about injury frequency, common types, etc.
        """
        history = self.get_team_injury_history(team_id)
        
        if history.empty:
            return {
                'total_injuries': 0,
                'avg_per_season': 0,
                'most_common_type': None,
                'injury_prone_months': []
            }
        
        # Most common injury types
        if 'injury_type' in history.columns and history['injury_type'].notna().any():
            most_common = history['injury_type'].mode()
            most_common_type = most_common.iloc[0] if len(most_common) > 0 else None
        else:
            most_common_type = None
        
        # Injuries by month
        if 'date' in history.columns:
            history['month'] = history['date'].dt.month
            monthly = history.groupby('month').size()
            injury_prone_months = monthly.nlargest(3).index.tolist()
        else:
            injury_prone_months = []
        
        # Seasons
        if 'season' in history.columns:
            seasons = history['season'].nunique()
            avg_per_season = len(history) / max(seasons, 1)
        else:
            avg_per_season = 0
        
        return {
            'total_injuries': len(history),
            'avg_per_season': avg_per_season,
            'most_common_type': most_common_type,
            'injury_prone_months': injury_prone_months
        }


# =============================================================================
# FEATURE GENERATION HELPERS
# =============================================================================

def add_injury_features_to_df(df: pd.DataFrame, tracker: InjuryTracker = None) -> pd.DataFrame:
    """
    Add injury features to a fixtures DataFrame
    
    Args:
        df: DataFrame with HomeTeamID, AwayTeamID, Date columns
        tracker: InjuryTracker instance (creates one if not provided)
    
    Returns:
        DataFrame with injury columns added
    """
    if tracker is None:
        tracker = InjuryTracker()
    
    df = df.copy()
    
    # Initialize columns
    df['home_injuries_count'] = 0
    df['away_injuries_count'] = 0
    df['home_key_injuries'] = 0
    df['away_key_injuries'] = 0
    
    for idx in df.index:
        row = df.loc[idx]
        
        home_id = row.get('HomeTeamID')
        away_id = row.get('AwayTeamID')
        match_date = row.get('Date')
        
        if pd.isna(home_id) or pd.isna(away_id) or pd.isna(match_date):
            continue
        
        if isinstance(match_date, str):
            match_date = pd.to_datetime(match_date)
        
        # Get injury features
        features = tracker.get_fixture_injury_features(
            int(home_id), int(away_id), match_date
        )
        
        df.loc[idx, 'home_injuries_count'] = features['home_injuries_count']
        df.loc[idx, 'away_injuries_count'] = features['away_injuries_count']
        df.loc[idx, 'home_key_injuries'] = features['home_key_injuries']
        df.loc[idx, 'away_key_injuries'] = features['away_key_injuries']
    
    return df


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Injury Tracker")
    parser.add_argument("--team", type=int, help="Team ID to analyze")
    parser.add_argument("--season", type=int, help="Season to filter")
    
    args = parser.parse_args()
    
    tracker = InjuryTracker()
    
    if args.team:
        print(f"\nInjury Analysis for Team ID {args.team}")
        print("="*50)
        
        # Current injuries
        injuries = tracker.get_team_injuries(args.team)
        print(f"\nCurrent injuries: {len(injuries)}")
        for inj in injuries[:5]:
            print(f"  - {inj['player_name']}: {inj.get('injury_type', 'Unknown')}")
        
        # Impact
        impact = tracker.calculate_injury_impact(args.team)
        print(f"\nImpact:")
        print(f"  Attack impact: {impact['attack_impact']:.2f}")
        print(f"  Key injuries: {impact['key_injuries']}")
        
        # Patterns
        patterns = tracker.analyze_injury_patterns(args.team)
        print(f"\nPatterns:")
        print(f"  Total injuries: {patterns['total_injuries']}")
        print(f"  Avg per season: {patterns['avg_per_season']:.1f}")
        print(f"  Most common: {patterns['most_common_type']}")
    else:
        print("Use --team <team_id> to analyze a specific team")
        print(f"Total injury records: {len(tracker.injuries_df)}")
