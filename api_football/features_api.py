# features_api.py
"""
Enhanced Feature Engineering with API-Football Data
Includes: xG, injuries, lineups, advanced statistics, cup competitions

Replaces: features.py (enhanced version)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "football_api.db"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"

# Feature flags
USE_XG = True
USE_INJURIES = True
USE_FORMATIONS = True
USE_ADVANCED_STATS = True


# =============================================================================
# DATABASE HELPERS
# =============================================================================

def get_db_connection():
    """Get database connection"""
    return sqlite3.connect(DB_PATH)


def load_fixtures_from_db() -> pd.DataFrame:
    """Load fixtures from database with statistics (deduplicated)"""
    query = """
        SELECT DISTINCT
            f.fixture_id,
            f.date as Date,
            f.league_code as League,
            f.league_name as LeagueName,
            f.league_type as LeagueType,
            f.season as Season,
            f.round as Round,
            f.home_team as HomeTeam,
            f.away_team as AwayTeam,
            f.home_team_id as HomeTeamID,
            f.away_team_id as AwayTeamID,
            f.home_goals as FTHG,
            f.away_goals as FTAG,
            CASE
                WHEN f.home_goals > f.away_goals THEN 'H'
                WHEN f.home_goals < f.away_goals THEN 'A'
                ELSE 'D'
            END as FTR,
            f.ht_home_goals as HTHG,
            f.ht_away_goals as HTAG,
            f.home_xG,
            f.away_xG,
            f.referee as Referee,
            f.venue_name as Venue,
            -- Home stats (use MAX to aggregate duplicates)
            MAX(CASE WHEN hs.is_home = 1 THEN hs.shots_total END) as HS,
            MAX(CASE WHEN hs.is_home = 1 THEN hs.shots_on_target END) as HST,
            MAX(CASE WHEN hs.is_home = 1 THEN hs.shots_inside_box END) as HS_inside,
            MAX(CASE WHEN hs.is_home = 1 THEN hs.corners END) as HC,
            MAX(CASE WHEN hs.is_home = 1 THEN hs.fouls END) as HF,
            MAX(CASE WHEN hs.is_home = 1 THEN hs.yellow_cards END) as HY,
            MAX(CASE WHEN hs.is_home = 1 THEN hs.red_cards END) as HR,
            MAX(CASE WHEN hs.is_home = 1 THEN hs.ball_possession END) as HPoss,
            MAX(CASE WHEN hs.is_home = 1 THEN hs.passes_total END) as HPasses,
            MAX(CASE WHEN hs.is_home = 1 THEN hs.passes_accurate END) as HPassesAcc,
            MAX(CASE WHEN hs.is_home = 1 THEN hs.tackles END) as HTackles,
            MAX(CASE WHEN hs.is_home = 1 THEN hs.interceptions END) as HInterceptions,
            MAX(CASE WHEN hs.is_home = 1 THEN hs.saves END) as HSaves,
            MAX(CASE WHEN hs.is_home = 1 THEN hs.expected_goals END) as HxG_stat,
            -- Away stats (use MAX to aggregate duplicates)
            MAX(CASE WHEN aws.is_home = 0 THEN aws.shots_total END) as AShots,
            MAX(CASE WHEN aws.is_home = 0 THEN aws.shots_on_target END) as AST,
            MAX(CASE WHEN aws.is_home = 0 THEN aws.shots_inside_box END) as AS_inside,
            MAX(CASE WHEN aws.is_home = 0 THEN aws.corners END) as AC,
            MAX(CASE WHEN aws.is_home = 0 THEN aws.fouls END) as AF,
            MAX(CASE WHEN aws.is_home = 0 THEN aws.yellow_cards END) as AY,
            MAX(CASE WHEN aws.is_home = 0 THEN aws.red_cards END) as AR,
            MAX(CASE WHEN aws.is_home = 0 THEN aws.ball_possession END) as APoss,
            MAX(CASE WHEN aws.is_home = 0 THEN aws.passes_total END) as APasses,
            MAX(CASE WHEN aws.is_home = 0 THEN aws.passes_accurate END) as APassesAcc,
            MAX(CASE WHEN aws.is_home = 0 THEN aws.tackles END) as ATackles,
            MAX(CASE WHEN aws.is_home = 0 THEN aws.interceptions END) as AInterceptions,
            MAX(CASE WHEN aws.is_home = 0 THEN aws.saves END) as ASaves,
            MAX(CASE WHEN aws.is_home = 0 THEN aws.expected_goals END) as AxG_stat
        FROM fixtures f
        LEFT JOIN fixture_statistics hs
            ON f.fixture_id = hs.fixture_id
        LEFT JOIN fixture_statistics aws
            ON f.fixture_id = aws.fixture_id
        WHERE f.status = 'FT' AND f.home_goals IS NOT NULL
        GROUP BY f.fixture_id, f.date, f.league_code, f.league_name, f.league_type,
                 f.season, f.round, f.home_team, f.away_team, f.home_team_id, f.away_team_id,
                 f.home_goals, f.away_goals, f.ht_home_goals, f.ht_away_goals,
                 f.home_xG, f.away_xG, f.referee, f.venue_name
        ORDER BY f.date, f.league_code
    """

    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn)

    df['Date'] = pd.to_datetime(df['Date'])
    logger.info(f"Loaded {len(df)} unique fixtures from database")
    return df


def load_injuries_from_db() -> pd.DataFrame:
    """Load injuries from database"""
    query = """
        SELECT 
            team_id,
            team_name,
            player_id,
            player_name,
            injury_type,
            injury_reason,
            date,
            league_id,
            season
        FROM injuries
        ORDER BY date DESC
    """
    
    with get_db_connection() as conn:
        return pd.read_sql_query(query, conn)


def load_lineups_from_db() -> pd.DataFrame:
    """Load lineups from database"""
    query = """
        SELECT 
            fixture_id,
            team_id,
            team_name,
            formation,
            player_id,
            player_name,
            position,
            is_starter
        FROM lineups
    """
    
    with get_db_connection() as conn:
        return pd.read_sql_query(query, conn)


# =============================================================================
# XG FEATURES (CRITICAL)
# =============================================================================

def add_xg_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add xG-based features - THE MOST IMPORTANT ENHANCEMENT
    
    Features:
    - Rolling xG averages (attack strength proxy)
    - xG overperformance (regression candidate)
    - xG differential
    """
    df = df.sort_values(['League', 'Date']).copy()

    # Combine xG from multiple sources (fixture level or statistics level)
    # Set pandas option to avoid FutureWarning about downcasting
    with pd.option_context('future.no_silent_downcasting', True):
        df['home_xG_combined'] = df['home_xG'].fillna(df['HxG_stat'])
        df['away_xG_combined'] = df['away_xG'].fillna(df['AxG_stat'])
    
    # Initialize columns
    xg_cols = [
        'home_xG_for_ma5', 'home_xG_for_ma10', 'home_xG_against_ma5', 'home_xG_against_ma10',
        'away_xG_for_ma5', 'away_xG_for_ma10', 'away_xG_against_ma5', 'away_xG_against_ma10',
        'home_xG_diff', 'away_xG_diff',
        'home_xG_overperformance', 'away_xG_overperformance',
        'home_xG_overperf_ma5', 'away_xG_overperf_ma5'
    ]
    for col in xg_cols:
        df[col] = np.nan
    
    for league in df['League'].unique():
        league_mask = df['League'] == league
        league_df = df[league_mask].copy()
        
        # Track xG for each team
        team_xg_for = {}  # Team -> list of xG scored
        team_xg_against = {}  # Team -> list of xG conceded
        team_overperf = {}  # Team -> list of (goals - xG)
        
        for idx in league_df.index:
            row = df.loc[idx]
            home = row['HomeTeam']
            away = row['AwayTeam']
            
            # Initialize if needed
            for team in [home, away]:
                if team not in team_xg_for:
                    team_xg_for[team] = []
                    team_xg_against[team] = []
                    team_overperf[team] = []
            
            # Calculate features BEFORE this match (no leakage)
            if len(team_xg_for[home]) >= 5:
                df.loc[idx, 'home_xG_for_ma5'] = np.mean(team_xg_for[home][-5:])
                df.loc[idx, 'home_xG_against_ma5'] = np.mean(team_xg_against[home][-5:])
                df.loc[idx, 'home_xG_overperf_ma5'] = np.mean(team_overperf[home][-5:])
            
            if len(team_xg_for[home]) >= 10:
                df.loc[idx, 'home_xG_for_ma10'] = np.mean(team_xg_for[home][-10:])
                df.loc[idx, 'home_xG_against_ma10'] = np.mean(team_xg_against[home][-10:])
            
            if len(team_xg_for[away]) >= 5:
                df.loc[idx, 'away_xG_for_ma5'] = np.mean(team_xg_for[away][-5:])
                df.loc[idx, 'away_xG_against_ma5'] = np.mean(team_xg_against[away][-5:])
                df.loc[idx, 'away_xG_overperf_ma5'] = np.mean(team_overperf[away][-5:])
            
            if len(team_xg_for[away]) >= 10:
                df.loc[idx, 'away_xG_for_ma10'] = np.mean(team_xg_for[away][-10:])
                df.loc[idx, 'away_xG_against_ma10'] = np.mean(team_xg_against[away][-10:])
            
            # xG differential
            if pd.notna(df.loc[idx, 'home_xG_for_ma5']) and pd.notna(df.loc[idx, 'home_xG_against_ma5']):
                df.loc[idx, 'home_xG_diff'] = df.loc[idx, 'home_xG_for_ma5'] - df.loc[idx, 'home_xG_against_ma5']
            
            if pd.notna(df.loc[idx, 'away_xG_for_ma5']) and pd.notna(df.loc[idx, 'away_xG_against_ma5']):
                df.loc[idx, 'away_xG_diff'] = df.loc[idx, 'away_xG_for_ma5'] - df.loc[idx, 'away_xG_against_ma5']
            
            # Update histories with THIS match's data
            home_xg = row['home_xG_combined']
            away_xg = row['away_xG_combined']
            
            if pd.notna(home_xg) and pd.notna(away_xg):
                # Home team
                team_xg_for[home].append(home_xg)
                team_xg_against[home].append(away_xg)
                team_overperf[home].append(row['FTHG'] - home_xg)
                
                # Away team
                team_xg_for[away].append(away_xg)
                team_xg_against[away].append(home_xg)
                team_overperf[away].append(row['FTAG'] - away_xg)
                
                # Keep last 20 only
                for team in [home, away]:
                    if len(team_xg_for[team]) > 20:
                        team_xg_for[team] = team_xg_for[team][-20:]
                        team_xg_against[team] = team_xg_against[team][-20:]
                        team_overperf[team] = team_overperf[team][-20:]
    
    # Calculate overperformance based on rolling averages
    df['home_xG_overperformance'] = df['home_xG_overperf_ma5']
    df['away_xG_overperformance'] = df['away_xG_overperf_ma5']
    
    logger.info(f"   xG features: {df['home_xG_for_ma5'].notna().sum()} matches with xG data")
    
    return df


# =============================================================================
# INJURY IMPACT FEATURES
# =============================================================================

def add_injury_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add injury impact features
    
    Features:
    - Number of injured players
    - Key players missing (estimated)
    - Defensive injuries count
    """
    df = df.copy()
    
    # Initialize columns
    df['home_injuries_count'] = 0
    df['away_injuries_count'] = 0
    df['home_key_injuries'] = 0
    df['away_key_injuries'] = 0
    
    try:
        injuries_df = load_injuries_from_db()
        
        if injuries_df.empty:
            logger.warning("No injury data available")
            return df
        
        injuries_df['date'] = pd.to_datetime(injuries_df['date'])
        
        # Create team injury counts by date
        for idx in df.index:
            row = df.loc[idx]
            match_date = row['Date']
            home_id = row.get('HomeTeamID')
            away_id = row.get('AwayTeamID')
            
            # Count injuries within 30 days before match
            recent_mask = (
                (injuries_df['date'] >= match_date - timedelta(days=30)) &
                (injuries_df['date'] < match_date)
            )
            
            # Home team injuries
            if home_id:
                home_injuries = injuries_df[recent_mask & (injuries_df['team_id'] == home_id)]
                df.loc[idx, 'home_injuries_count'] = len(home_injuries)
                # Rough estimate of key injuries (would need player importance data)
                df.loc[idx, 'home_key_injuries'] = min(len(home_injuries) // 3, 2)
            
            # Away team injuries
            if away_id:
                away_injuries = injuries_df[recent_mask & (injuries_df['team_id'] == away_id)]
                df.loc[idx, 'away_injuries_count'] = len(away_injuries)
                df.loc[idx, 'away_key_injuries'] = min(len(away_injuries) // 3, 2)
        
        logger.info(f"   Injury features: Avg injuries home={df['home_injuries_count'].mean():.1f}")
        
    except Exception as e:
        logger.warning(f"Could not load injury data: {e}")
    
    return df


# =============================================================================
# FORMATION FEATURES
# =============================================================================

# Formation attack ratings (higher = more attacking)
FORMATION_RATINGS = {
    '4-3-3': 0.85, '4-2-3-1': 0.80, '3-4-3': 0.90,
    '4-4-2': 0.70, '4-4-1-1': 0.65, '4-5-1': 0.55,
    '5-4-1': 0.45, '5-3-2': 0.50, '3-5-2': 0.65,
    '4-1-4-1': 0.60, '3-4-2-1': 0.75, '4-2-2-2': 0.70,
    '4-1-2-1-2': 0.65, '4-3-1-2': 0.70, '3-4-1-2': 0.70,
}

def add_formation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add formation-based features
    
    Features:
    - Formation attack rating
    - Formation matchup indicator
    """
    df = df.copy()
    
    df['home_formation'] = None
    df['away_formation'] = None
    df['home_formation_attack'] = 0.70
    df['away_formation_attack'] = 0.70
    df['formation_matchup_goals_mult'] = 1.0
    
    try:
        lineups_df = load_lineups_from_db()
        
        if lineups_df.empty:
            logger.warning("No lineup data available")
            return df
        
        # Get formation for each fixture/team
        formations = lineups_df.groupby(['fixture_id', 'team_id'])['formation'].first().reset_index()
        
        for idx in df.index:
            fixture_id = df.loc[idx, 'fixture_id']
            home_id = df.loc[idx, 'HomeTeamID']
            away_id = df.loc[idx, 'AwayTeamID']
            
            # Home formation
            home_form = formations[
                (formations['fixture_id'] == fixture_id) & 
                (formations['team_id'] == home_id)
            ]['formation'].values
            
            if len(home_form) > 0 and home_form[0]:
                df.loc[idx, 'home_formation'] = home_form[0]
                df.loc[idx, 'home_formation_attack'] = FORMATION_RATINGS.get(home_form[0], 0.70)
            
            # Away formation
            away_form = formations[
                (formations['fixture_id'] == fixture_id) & 
                (formations['team_id'] == away_id)
            ]['formation'].values
            
            if len(away_form) > 0 and away_form[0]:
                df.loc[idx, 'away_formation'] = away_form[0]
                df.loc[idx, 'away_formation_attack'] = FORMATION_RATINGS.get(away_form[0], 0.70)
            
            # Formation matchup - attacking vs attacking = more goals
            if df.loc[idx, 'home_formation_attack'] > 0.75 and df.loc[idx, 'away_formation_attack'] > 0.75:
                df.loc[idx, 'formation_matchup_goals_mult'] = 1.08
            elif df.loc[idx, 'home_formation_attack'] < 0.55 and df.loc[idx, 'away_formation_attack'] < 0.55:
                df.loc[idx, 'formation_matchup_goals_mult'] = 0.92
        
        logger.info(f"   Formation features: {df['home_formation'].notna().sum()} matches with formation data")
        
    except Exception as e:
        logger.warning(f"Could not load formation data: {e}")
    
    return df


# =============================================================================
# ADVANCED STATISTICS FEATURES
# =============================================================================

def add_advanced_stats_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add advanced statistics features
    
    Features from match stats: shots, possession, corners, passes, etc.
    """
    df = df.sort_values(['League', 'Date']).copy()
    
    # Statistics to track as rolling averages
    stats_to_roll = {
        'shots_for': ('HS', 'AShots'),
        'shots_on_target_for': ('HST', 'AST'),
        'corners_for': ('HC', 'AC'),
        'possession': ('HPoss', 'APoss'),
        'passes_for': ('HPasses', 'APasses'),
    }
    
    # Initialize output columns
    for stat_name in stats_to_roll.keys():
        df[f'home_{stat_name}_ma5'] = np.nan
        df[f'away_{stat_name}_ma5'] = np.nan
    
    # Also add derived stats
    df['home_shot_accuracy_ma5'] = np.nan
    df['away_shot_accuracy_ma5'] = np.nan
    df['home_attack_quality'] = np.nan
    df['away_attack_quality'] = np.nan
    
    for league in df['League'].unique():
        league_mask = df['League'] == league
        league_df = df[league_mask].copy()
        
        # Track stats per team
        team_stats = {}
        
        for idx in league_df.index:
            row = df.loc[idx]
            home = row['HomeTeam']
            away = row['AwayTeam']
            
            # Initialize teams
            for team in [home, away]:
                if team not in team_stats:
                    team_stats[team] = {k: [] for k in stats_to_roll.keys()}
                    team_stats[team]['shot_accuracy'] = []
            
            # Apply rolling averages BEFORE this match
            for stat_name in stats_to_roll.keys():
                if len(team_stats[home][stat_name]) >= 5:
                    df.loc[idx, f'home_{stat_name}_ma5'] = np.mean(team_stats[home][stat_name][-5:])
                if len(team_stats[away][stat_name]) >= 5:
                    df.loc[idx, f'away_{stat_name}_ma5'] = np.mean(team_stats[away][stat_name][-5:])
            
            # Shot accuracy
            if len(team_stats[home]['shot_accuracy']) >= 5:
                df.loc[idx, 'home_shot_accuracy_ma5'] = np.mean(team_stats[home]['shot_accuracy'][-5:])
            if len(team_stats[away]['shot_accuracy']) >= 5:
                df.loc[idx, 'away_shot_accuracy_ma5'] = np.mean(team_stats[away]['shot_accuracy'][-5:])
            
            # Attack quality composite
            home_sot = df.loc[idx, 'home_shots_on_target_for_ma5']
            home_poss = df.loc[idx, 'home_possession_ma5']
            home_corn = df.loc[idx, 'home_corners_for_ma5']
            if pd.notna(home_sot) and pd.notna(home_poss):
                df.loc[idx, 'home_attack_quality'] = (
                    (home_sot or 0) * 0.4 +
                    (home_poss or 50) * 0.01 +
                    (home_corn or 0) * 0.1
                )
            
            away_sot = df.loc[idx, 'away_shots_on_target_for_ma5']
            away_poss = df.loc[idx, 'away_possession_ma5']
            away_corn = df.loc[idx, 'away_corners_for_ma5']
            if pd.notna(away_sot) and pd.notna(away_poss):
                df.loc[idx, 'away_attack_quality'] = (
                    (away_sot or 0) * 0.4 +
                    (away_poss or 50) * 0.01 +
                    (away_corn or 0) * 0.1
                )
            
            # Update histories
            for stat_name, (home_col, away_col) in stats_to_roll.items():
                home_val = row.get(home_col)
                away_val = row.get(away_col)
                
                if pd.notna(home_val):
                    team_stats[home][stat_name].append(home_val)
                if pd.notna(away_val):
                    team_stats[away][stat_name].append(away_val)
            
            # Shot accuracy
            if pd.notna(row.get('HS')) and row.get('HS', 0) > 0:
                team_stats[home]['shot_accuracy'].append(
                    (row.get('HST', 0) or 0) / row['HS']
                )
            if pd.notna(row.get('AShots')) and row.get('AShots', 0) > 0:
                team_stats[away]['shot_accuracy'].append(
                    (row.get('AST', 0) or 0) / row['AShots']
                )
            
            # Keep last 20
            for team in [home, away]:
                for stat_name in team_stats[team]:
                    if len(team_stats[team][stat_name]) > 20:
                        team_stats[team][stat_name] = team_stats[team][stat_name][-20:]
    
    logger.info(f"   Advanced stats: {df['home_shots_for_ma5'].notna().sum()} matches with stats")
    
    return df


# =============================================================================
# REST DAYS (ACCURATE WITH CUPS)
# =============================================================================

def add_accurate_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate accurate rest days including ALL competitions
    
    Now that we have cup matches in the database, rest days are accurate!
    """
    df = df.sort_values(['Date']).copy()
    
    df['home_rest_days'] = 14
    df['away_rest_days'] = 14
    df['home_had_cup_midweek'] = 0
    df['away_had_cup_midweek'] = 0
    
    # Track all matches per team (not per league)
    team_last_match = {}
    team_last_match_type = {}
    
    for idx in df.index:
        row = df.loc[idx]
        home = row['HomeTeam']
        away = row['AwayTeam']
        match_date = row['Date']
        match_type = row.get('LeagueType', 'league')
        
        # Home team
        if home in team_last_match:
            last_date, last_type = team_last_match[home], team_last_match_type[home]
            rest = (match_date - last_date).days
            df.loc[idx, 'home_rest_days'] = max(1, min(rest, 14))
            
            # Check if midweek cup match
            if rest <= 4 and last_type == 'cup':
                df.loc[idx, 'home_had_cup_midweek'] = 1
        
        # Away team
        if away in team_last_match:
            last_date, last_type = team_last_match[away], team_last_match_type[away]
            rest = (match_date - last_date).days
            df.loc[idx, 'away_rest_days'] = max(1, min(rest, 14))
            
            if rest <= 4 and last_type == 'cup':
                df.loc[idx, 'away_had_cup_midweek'] = 1
        
        # Update trackers
        team_last_match[home] = match_date
        team_last_match[away] = match_date
        team_last_match_type[home] = match_type
        team_last_match_type[away] = match_type
    
    # Add rest bands
    df['home_rest_band'] = pd.cut(
        df['home_rest_days'], 
        bins=[0, 3, 6, 100], 
        labels=['short', 'medium', 'long']
    )
    df['away_rest_band'] = pd.cut(
        df['away_rest_days'], 
        bins=[0, 3, 6, 100], 
        labels=['short', 'medium', 'long']
    )
    
    logger.info(f"   Rest days: Avg home={df['home_rest_days'].mean():.1f}, away={df['away_rest_days'].mean():.1f}")
    logger.info(f"   Midweek cup: {df['home_had_cup_midweek'].sum()} home, {df['away_had_cup_midweek'].sum()} away")
    
    return df


# =============================================================================
# H2H FEATURES (uses API data)
# =============================================================================

def add_h2h_features(df: pd.DataFrame, min_meetings: int = 3) -> pd.DataFrame:
    """
    Add H2H features (enhanced version using full history)
    """
    df = df.sort_values(['Date']).copy()
    
    df['h2h_total_goals_avg'] = np.nan
    df['h2h_btts_rate'] = np.nan
    df['h2h_home_win_rate'] = np.nan
    df['h2h_over25_rate'] = np.nan
    df['h2h_meetings'] = 0
    
    # Track all H2H history globally (across leagues for international teams)
    h2h_history = {}
    
    for idx in df.index:
        row = df.loc[idx]
        home = row['HomeTeam']
        away = row['AwayTeam']
        match_date = row['Date']
        
        pairing = tuple(sorted([home, away]))
        
        if pairing in h2h_history:
            past = [m for m in h2h_history[pairing] if m['date'] < match_date]
            
            if len(past) >= min_meetings:
                total_goals = [m['total_goals'] for m in past]
                btts = [m['btts'] for m in past]
                home_wins = [m['home_win'] for m in past if m['home_team'] == home]
                over25 = [m['over25'] for m in past]
                
                df.loc[idx, 'h2h_total_goals_avg'] = np.mean(total_goals)
                df.loc[idx, 'h2h_btts_rate'] = np.mean(btts)
                df.loc[idx, 'h2h_home_win_rate'] = np.mean(home_wins) if home_wins else 0.5
                df.loc[idx, 'h2h_over25_rate'] = np.mean(over25)
                df.loc[idx, 'h2h_meetings'] = len(past)
        
        if pairing not in h2h_history:
            h2h_history[pairing] = []
        
        # Add this match to history
        if pd.notna(row['FTHG']) and pd.notna(row['FTAG']):
            h2h_history[pairing].append({
                'date': match_date,
                'home_team': home,
                'home_goals': row['FTHG'],
                'away_goals': row['FTAG'],
                'total_goals': row['FTHG'] + row['FTAG'],
                'btts': 1 if (row['FTHG'] > 0 and row['FTAG'] > 0) else 0,
                'home_win': 1 if row['FTHG'] > row['FTAG'] else 0,
                'over25': 1 if (row['FTHG'] + row['FTAG']) > 2.5 else 0
            })
    
    # Fill missing with league averages
    for league in df['League'].unique():
        league_mask = df['League'] == league
        no_h2h_mask = league_mask & (df['h2h_meetings'] < min_meetings)
        
        league_total_goals = df[league_mask]['FTHG'].mean() + df[league_mask]['FTAG'].mean()
        league_btts = ((df[league_mask]['FTHG'] > 0) & (df[league_mask]['FTAG'] > 0)).mean()
        
        df.loc[no_h2h_mask, 'h2h_total_goals_avg'] = league_total_goals
        df.loc[no_h2h_mask, 'h2h_btts_rate'] = league_btts
        df.loc[no_h2h_mask, 'h2h_home_win_rate'] = 0.45
        df.loc[no_h2h_mask, 'h2h_over25_rate'] = ((df[league_mask]['FTHG'] + df[league_mask]['FTAG']) > 2.5).mean()
    
    logger.info(f"   H2H: {(df['h2h_meetings'] >= min_meetings).sum()} matches with H2H data")
    
    return df


# =============================================================================
# CUP COMPETITION FEATURES
# =============================================================================

def add_cup_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cup-specific features
    """
    df = df.copy()
    
    df['is_cup_match'] = (df['LeagueType'] == 'cup').astype(int)
    df['is_european_cup'] = df['League'].isin(['UCL', 'UEL', 'UECL']).astype(int)
    df['is_knockout'] = df['Round'].str.contains(
        'Final|Semi|Quarter|Round of', 
        case=False, 
        na=False
    ).astype(int)
    
    # Cup matches tend to have different characteristics
    # Home advantage often reduced in cups
    df['cup_home_advantage_factor'] = np.where(
        df['is_cup_match'] == 1,
        0.85,  # Reduced home advantage
        1.0
    )
    
    logger.info(f"   Cup features: {df['is_cup_match'].sum()} cup matches, {df['is_european_cup'].sum()} European")
    
    return df


# =============================================================================
# TARGETS
# =============================================================================

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add target variables"""
    df = df.copy()
    
    # BTTS
    df['y_BTTS'] = np.where(
        (df['FTHG'] > 0) & (df['FTAG'] > 0),
        'Y', 'N'
    )
    
    # Over/Under lines
    for line in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
        col = f"y_OU_{str(line).replace('.', '_')}"
        total = df['FTHG'] + df['FTAG']
        df[col] = np.where(total > line, 'O', 'U')
    
    return df


# =============================================================================
# ROLLING FORM FEATURES (from original features.py)
# =============================================================================

def add_rolling_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling form features (goals, points, corners, shots, cards)
    CRITICAL: These are required by the Dixon-Coles model

    Uses index-based approach to avoid cartesian product in merges
    """
    logger.info("   Calculating rolling form features...")

    df = df.sort_values(['League', 'Date']).copy()

    # Add match ID to prevent cartesian products
    df['_match_id'] = range(len(df))

    # Helper to create team-side view WITH MATCH ID
    def team_side_view(df, side):
        out = df.copy()
        if side == 'Home':
            out['Team'] = out['HomeTeam']
            out['GoalsFor'] = out['FTHG']
            out['GoalsAgainst'] = out['FTAG']
            out['Win'] = (out['FTR'] == 'H').astype(int)
            out['Draw'] = (out['FTR'] == 'D').astype(int)
            out['Shots'] = out.get('HS', 0)
            out['ShotsT'] = out.get('HST', 0)
            out['Corners'] = out.get('HC', 0)
            out['CardsY'] = out.get('HY', 0)
        else:
            out['Team'] = out['AwayTeam']
            out['GoalsFor'] = out['FTAG']
            out['GoalsAgainst'] = out['FTHG']
            out['Win'] = (out['FTR'] == 'A').astype(int)
            out['Draw'] = (out['FTR'] == 'D').astype(int)
            out['Shots'] = out.get('AShots', 0)
            out['ShotsT'] = out.get('AST', 0)
            out['Corners'] = out.get('AC', 0)
            out['CardsY'] = out.get('AY', 0)
        return out[['_match_id', 'League', 'Date', 'Team', 'GoalsFor', 'GoalsAgainst', 'Win', 'Draw', 'Shots', 'ShotsT', 'Corners', 'CardsY']]

    # Calculate rolling stats per team
    home_df = team_side_view(df, 'Home')
    away_df = team_side_view(df, 'Away')
    team_df = pd.concat([home_df, away_df])

    # Calculate moving averages per team
    rolling_features = []
    for (league, team), group in team_df.groupby(['League', 'Team']):
        group = group.sort_values('Date').copy()

        # 5-game moving average (shifted to avoid leakage)
        rolled = group.shift(1).rolling(window=5, min_periods=1)
        group['GF_ma5'] = rolled['GoalsFor'].mean()
        group['GA_ma5'] = rolled['GoalsAgainst'].mean()
        group['GD_ma5'] = group['GF_ma5'] - group['GA_ma5']
        group['Pts_ma5'] = (rolled['Win'].sum() * 3 + rolled['Draw'].sum()) / 5
        group['Corners_ma5'] = rolled['Corners'].mean()
        group['ShotsT_ma5'] = rolled['ShotsT'].mean()
        group['CardsY_ma5'] = rolled['CardsY'].mean()

        # 10-game exponential weighted average
        ew = group.shift(1).ewm(span=10, adjust=False)
        group['GF_ewm10'] = ew['GoalsFor'].mean()
        group['GA_ewm10'] = ew['GoalsAgainst'].mean()
        group['FormPts_ewm10'] = ew['Win'].mean() * 3

        rolling_features.append(group)

    team_rolling = pd.concat(rolling_features)

    # Create home view: rename Team to HomeTeam and prefix columns
    home_rolling = team_rolling.copy()
    home_rolling = home_rolling.rename(columns={'Team': 'HomeTeam'})
    home_rolling.columns = ['Home_' + c if c not in ['_match_id', 'League', 'Date', 'HomeTeam'] else c for c in home_rolling.columns]

    # Create away view: rename Team to AwayTeam and prefix columns
    away_rolling = team_rolling.copy()
    away_rolling = away_rolling.rename(columns={'Team': 'AwayTeam'})
    away_rolling.columns = ['Away_' + c if c not in ['_match_id', 'League', 'Date', 'AwayTeam'] else c for c in away_rolling.columns]

    # Merge using MATCH ID + Team name to ensure 1:1 relationship
    df = df.merge(home_rolling[['_match_id', 'HomeTeam', 'Home_GF_ma5', 'Home_GA_ma5', 'Home_GD_ma5',
                                  'Home_Pts_ma5', 'Home_Corners_ma5', 'Home_ShotsT_ma5', 'Home_CardsY_ma5',
                                  'Home_GF_ewm10', 'Home_GA_ewm10', 'Home_FormPts_ewm10']],
                  on=['_match_id', 'HomeTeam'], how='left')

    df = df.merge(away_rolling[['_match_id', 'AwayTeam', 'Away_GF_ma5', 'Away_GA_ma5', 'Away_GD_ma5',
                                  'Away_Pts_ma5', 'Away_Corners_ma5', 'Away_ShotsT_ma5', 'Away_CardsY_ma5',
                                  'Away_GF_ewm10', 'Away_GA_ewm10', 'Away_FormPts_ewm10']],
                  on=['_match_id', 'AwayTeam'], how='left')

    # Clean up match ID
    df = df.drop(columns=['_match_id'])

    logger.info(f"   Rolling form: Added 22 features (GF/GA/GD/Pts/Corners/ShotsT/CardsY)")
    return df


def add_match_number_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add match_number for seasonal pattern adjustments
    CRITICAL: Required by Dixon-Coles model for early-season volatility

    Optimized vectorized implementation for speed - uses index to avoid merge issues
    """
    logger.info("   Calculating match numbers...")

    df = df.sort_values(['League', 'Date']).copy()

    # Add match counts directly using groupby cumcount
    # For home matches
    df['home_match_count'] = df.groupby(['League', 'HomeTeam']).cumcount()

    # For away matches
    df['away_match_count'] = df.groupby(['League', 'AwayTeam']).cumcount()

    # Average of both teams' match counts
    df['match_number'] = ((df['home_match_count'] + df['away_match_count']) / 2).astype(int)

    # Clean up temporary columns
    df = df.drop(columns=['home_match_count', 'away_match_count'])

    logger.info(f"   Match numbers: 0-{df['match_number'].max()} (early season = 0-10)")
    return df


def add_elo_ratings(df: pd.DataFrame, base_rating: float = 1500, k: float = 32, home_adv: float = 100) -> pd.DataFrame:
    """
    Add ELO rating system for dynamic team strength
    CRITICAL: Provides accurate pre-match strength differential
    """
    logger.info("   Calculating ELO ratings...")

    df = df.sort_values(['League', 'Date']).copy()

    # Initialize ELO ratings per league/team
    elo_state = {}

    home_elos = []
    away_elos = []

    for idx, row in df.iterrows():
        league = row['League']
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        key_h = (league, home_team)
        key_a = (league, away_team)

        # Get current ratings
        elo_h = elo_state.get(key_h, base_rating)
        elo_a = elo_state.get(key_a, base_rating)

        # Store pre-match ELOs
        home_elos.append(elo_h)
        away_elos.append(elo_a)

        # Update ELOs based on result
        ftr = row.get('FTR')
        if pd.notna(ftr):
            if ftr == 'H': score = 1.0
            elif ftr == 'D': score = 0.5
            else: score = 0.0

            # Expected score (with home advantage)
            elo_h_eff = elo_h + home_adv
            expected_h = 1 / (1 + 10 ** ((elo_a - elo_h_eff) / 400))

            # Update ratings
            elo_h_new = elo_h + k * (score - expected_h)
            elo_a_new = elo_a + k * ((1 - score) - (1 - expected_h))

            elo_state[key_h] = elo_h_new
            elo_state[key_a] = elo_a_new

    df['EloHome_pre'] = home_elos
    df['EloAway_pre'] = away_elos
    df['EloDiff_pre'] = df['EloHome_pre'] - df['EloAway_pre']

    logger.info(f"   ELO ratings: {len(elo_state)} team ratings (base={base_rating})")
    return df


# =============================================================================
# MAIN BUILD FUNCTION
# =============================================================================

def build_features(force: bool = False, from_parquet: Path = None) -> Path:
    """
    Build all features
    
    Args:
        force: Rebuild even if features file exists
        from_parquet: Load from parquet instead of database
    
    Returns:
        Path to features parquet file
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    if FEATURES_PARQUET.exists() and not force:
        logger.info(f"Features file exists at {FEATURES_PARQUET}. Use --force to rebuild.")
        return FEATURES_PARQUET
    
    logger.info("="*60)
    logger.info("BUILDING ENHANCED FEATURES")
    logger.info("="*60)
    
    # Load data
    if from_parquet and from_parquet.exists():
        logger.info(f"Loading from parquet: {from_parquet}")
        df = pd.read_parquet(from_parquet)
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        logger.info("Loading from database...")
        df = load_fixtures_from_db()
    
    logger.info(f"Loaded {len(df)} matches from {df['Date'].min()} to {df['Date'].max()}")
    logger.info(f"Leagues: {df['League'].nunique()} ({', '.join(df['League'].unique()[:10])}...)")
    
    # Apply enhancements
    logger.info("\n[1/10] Adding xG features...")
    if USE_XG:
        df = add_xg_features(df)

    logger.info("\n[2/10] Adding injury features...")
    if USE_INJURIES:
        df = add_injury_features(df)

    logger.info("\n[3/10] Adding formation features...")
    if USE_FORMATIONS:
        df = add_formation_features(df)

    logger.info("\n[4/10] Adding advanced stats features...")
    if USE_ADVANCED_STATS:
        df = add_advanced_stats_features(df)

    logger.info("\n[5/10] Adding accurate rest days...")
    df = add_accurate_rest_days(df)

    logger.info("\n[6/10] Adding H2H features...")
    df = add_h2h_features(df)

    logger.info("\n[7/10] Adding cup features...")
    df = add_cup_features(df)

    # Add MAXIMUM ACCURACY features (rolling form, match number, ELO)
    logger.info("\n[8/10] Adding rolling form features...")
    df = add_rolling_form_features(df)

    logger.info("\n[9/10] Adding match number...")
    df = add_match_number_feature(df)

    logger.info("\n[10/10] Adding ELO ratings...")
    df = add_elo_ratings(df)

    # Add targets
    logger.info("\nAdding target variables...")
    df = add_targets(df)
    
    # Save
    df.to_parquet(FEATURES_PARQUET, index=False)
    
    logger.info("\n" + "="*60)
    logger.info(f"FEATURES COMPLETE: {len(df)} matches, {len(df.columns)} columns")
    logger.info(f"Saved to: {FEATURES_PARQUET}")
    logger.info("="*60)
    
    # Summary
    logger.info("\nKey features added:")
    new_features = [
        'home_xG_for_ma5', 'home_xG_overperformance',
        'home_injuries_count', 'home_key_injuries',
        'home_formation_attack', 'formation_matchup_goals_mult',
        'home_attack_quality', 'home_shot_accuracy_ma5',
        'is_cup_match', 'is_european_cup', 'home_had_cup_midweek',
        'Home_GF_ma5', 'Home_Pts_ma5', 'Home_GF_ewm10',
        'match_number', 'EloHome_pre', 'EloAway_pre', 'EloDiff_pre'
    ]
    for f in new_features:
        if f in df.columns:
            if df[f].dtype in ['float64', 'int64']:
                non_null = df[f].notna().sum()
                logger.info(f"   {f}: mean={df[f].mean():.3f} ({non_null} non-null)")
    
    return FEATURES_PARQUET


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build enhanced features")
    parser.add_argument("--force", action="store_true", help="Force rebuild")
    parser.add_argument("--from-parquet", type=Path, help="Load from parquet file")
    
    args = parser.parse_args()
    
    output = build_features(force=args.force, from_parquet=args.from_parquet)
    print(f"\nOutput: {output}")
