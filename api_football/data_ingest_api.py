# data_ingest_api.py
"""
Data Ingestion from API-Football
Builds historical database with xG, injuries, lineups, and all competitions

Replaces: download_football_data.py + data_ingest.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
import json
import logging

from api_football_client import (
    APIFootballClient, LEAGUES, get_league_id, get_league_code
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw_api"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "football_api.db"

# Seasons to download (adjust based on your needs)
DEFAULT_SEASONS = [2021, 2022, 2023, 2024]

# Default leagues to download
DEFAULT_LEAGUES = [
    # England
    'E0', 'E1', 'E2', 'E3', 'EC',  # Premier League, Championship, League One, League Two, National League

    # Germany
    'D1', 'D2',                      # Bundesliga, 2. Bundesliga

    # Spain
    'SP1', 'SP2',                    # La Liga, Segunda

    # Italy
    'I1', 'I2',                      # Serie A, Serie B

    # France
    'F1', 'F2',                      # Ligue 1, Ligue 2

    # Other Major Leagues
    'N1',                            # Netherlands: Eredivisie
    'B1',                            # Belgium: Jupiler Pro League
    'P1',                            # Portugal: Primeira Liga
    'G1',                            # Greece: Super League

    # Scotland
    'SC0', 'SC1', 'SC2', 'SC3',      # Premiership, Championship, League One, League Two

    # Turkey
    'T1'                             # Super Lig
]


# =============================================================================
# DATABASE SCHEMA
# =============================================================================

SCHEMA = """
-- Core fixtures table
CREATE TABLE IF NOT EXISTS fixtures (
    fixture_id INTEGER PRIMARY KEY,
    league_code TEXT,
    league_id INTEGER,
    league_name TEXT,
    league_type TEXT,  -- 'league' or 'cup'
    season INTEGER,
    round TEXT,
    date DATE,
    time TEXT,
    venue_id INTEGER,
    venue_name TEXT,
    referee TEXT,
    home_team_id INTEGER,
    home_team TEXT,
    away_team_id INTEGER,
    away_team TEXT,
    home_goals INTEGER,
    away_goals INTEGER,
    ht_home_goals INTEGER,
    ht_away_goals INTEGER,
    status TEXT,
    -- xG data
    home_xG REAL,
    away_xG REAL,
    -- Odds implied
    home_odds REAL,
    draw_odds REAL,
    away_odds REAL,
    btts_yes_odds REAL,
    over25_odds REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Match statistics
CREATE TABLE IF NOT EXISTS fixture_statistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id INTEGER,
    team_id INTEGER,
    team_name TEXT,
    is_home INTEGER,
    -- Shots
    shots_total INTEGER,
    shots_on_target INTEGER,
    shots_off_target INTEGER,
    shots_blocked INTEGER,
    shots_inside_box INTEGER,
    shots_outside_box INTEGER,
    -- Possession & Passing
    ball_possession REAL,
    passes_total INTEGER,
    passes_accurate INTEGER,
    passes_pct REAL,
    -- Set pieces
    corners INTEGER,
    free_kicks INTEGER,
    -- Defensive
    fouls INTEGER,
    tackles INTEGER,
    interceptions INTEGER,
    blocks INTEGER,
    clearances INTEGER,
    -- Cards
    yellow_cards INTEGER,
    red_cards INTEGER,
    -- Goalkeeper
    saves INTEGER,
    -- xG (if available per team)
    expected_goals REAL,
    FOREIGN KEY (fixture_id) REFERENCES fixtures(fixture_id)
);

-- Team info cache
CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    name TEXT,
    short_name TEXT,
    country TEXT,
    logo_url TEXT,
    venue_id INTEGER,
    venue_name TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Injuries tracking
CREATE TABLE IF NOT EXISTS injuries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER,
    player_name TEXT,
    team_id INTEGER,
    team_name TEXT,
    fixture_id INTEGER,
    league_id INTEGER,
    season INTEGER,
    injury_type TEXT,
    injury_reason TEXT,
    date DATE,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Lineups
CREATE TABLE IF NOT EXISTS lineups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id INTEGER,
    team_id INTEGER,
    team_name TEXT,
    formation TEXT,
    player_id INTEGER,
    player_name TEXT,
    position TEXT,
    position_grid TEXT,
    is_starter INTEGER,
    jersey_number INTEGER,
    FOREIGN KEY (fixture_id) REFERENCES fixtures(fixture_id)
);

-- Player statistics per fixture
CREATE TABLE IF NOT EXISTS player_fixture_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id INTEGER,
    team_id INTEGER,
    player_id INTEGER,
    player_name TEXT,
    position TEXT,
    minutes_played INTEGER,
    rating REAL,
    -- Attacking
    goals INTEGER,
    assists INTEGER,
    shots_total INTEGER,
    shots_on_target INTEGER,
    -- Passing
    passes_total INTEGER,
    passes_accurate INTEGER,
    passes_key INTEGER,
    -- Defensive
    tackles INTEGER,
    interceptions INTEGER,
    duels_total INTEGER,
    duels_won INTEGER,
    -- Discipline
    yellow_cards INTEGER,
    red_cards INTEGER,
    fouls_committed INTEGER,
    fouls_drawn INTEGER,
    -- Dribbling
    dribbles_attempts INTEGER,
    dribbles_success INTEGER,
    FOREIGN KEY (fixture_id) REFERENCES fixtures(fixture_id)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_fixtures_league ON fixtures(league_code, season);
CREATE INDEX IF NOT EXISTS idx_fixtures_date ON fixtures(date);
CREATE INDEX IF NOT EXISTS idx_fixtures_teams ON fixtures(home_team_id, away_team_id);
CREATE INDEX IF NOT EXISTS idx_stats_fixture ON fixture_statistics(fixture_id);
CREATE INDEX IF NOT EXISTS idx_injuries_team ON injuries(team_id, season);
CREATE INDEX IF NOT EXISTS idx_lineups_fixture ON lineups(fixture_id);
"""


# =============================================================================
# DATA INGESTION CLASS
# =============================================================================

class APIFootballIngestor:
    """
    Ingest data from API-Football into local database
    """
    
    def __init__(self, api_key: str):
        self.client = APIFootballClient(api_key)
        self._init_dirs()
        self._init_db()
    
    def _init_dirs(self):
        """Create data directories"""
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    def _init_db(self):
        """Initialize SQLite database"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.executescript(SCHEMA)
        logger.info(f"Database initialized at {DB_PATH}")
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection"""
        return sqlite3.connect(DB_PATH)
    
    # =========================================================================
    # FIXTURE INGESTION
    # =========================================================================
    
    def ingest_fixtures(self, league_code: str, season: int, 
                        include_stats: bool = True) -> int:
        """
        Ingest all fixtures for a league/season
        
        Args:
            league_code: Our league code (e.g., 'E0')
            season: Season year (e.g., 2024)
            include_stats: Whether to also fetch match statistics
        
        Returns:
            Number of fixtures ingested
        """
        league_info = LEAGUES.get(league_code)
        if not league_info:
            logger.error(f"Unknown league code: {league_code}")
            return 0
        
        league_id = league_info['id']
        logger.info(f"Ingesting {league_code} ({league_info['name']}) season {season}...")
        
        # Fetch fixtures
        fixtures = self.client.get_fixtures(league_id=league_id, season=season)
        
        if not fixtures:
            logger.warning(f"No fixtures found for {league_code} {season}")
            return 0
        
        count = 0
        with self._get_conn() as conn:
            for fixture in fixtures:
                try:
                    self._insert_fixture(conn, fixture, league_code, league_info)
                    count += 1
                    
                    # Fetch statistics for completed matches
                    if include_stats and fixture['fixture']['status']['short'] == 'FT':
                        fixture_id = fixture['fixture']['id']
                        self._ingest_fixture_statistics(conn, fixture_id)
                        
                except Exception as e:
                    logger.error(f"Error inserting fixture {fixture.get('fixture', {}).get('id')}: {e}")
            
            conn.commit()
        
        logger.info(f"Ingested {count} fixtures for {league_code} {season}")
        return count
    
    def _insert_fixture(self, conn: sqlite3.Connection, fixture: Dict, 
                        league_code: str, league_info: Dict):
        """Insert single fixture into database"""
        f = fixture['fixture']
        teams = fixture['teams']
        goals = fixture['goals']
        score = fixture.get('score', {})
        
        # Extract xG if available (may be in statistics)
        home_xg = None
        away_xg = None
        
        # Try to get from fixture data first
        if 'statistics' in fixture:
            for stat in fixture.get('statistics', []):
                if stat.get('team', {}).get('id') == teams['home']['id']:
                    for s in stat.get('statistics', []):
                        if s['type'] == 'expected_goals':
                            home_xg = s['value']
                elif stat.get('team', {}).get('id') == teams['away']['id']:
                    for s in stat.get('statistics', []):
                        if s['type'] == 'expected_goals':
                            away_xg = s['value']
        
        conn.execute("""
            INSERT OR REPLACE INTO fixtures (
                fixture_id, league_code, league_id, league_name, league_type,
                season, round, date, time, venue_id, venue_name, referee,
                home_team_id, home_team, away_team_id, away_team,
                home_goals, away_goals, ht_home_goals, ht_away_goals,
                status, home_xG, away_xG, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f['id'],
            league_code,
            league_info['id'],
            league_info['name'],
            league_info['type'],
            fixture['league']['season'],
            fixture['league'].get('round'),
            f['date'][:10] if f.get('date') else None,
            f['date'][11:16] if f.get('date') and len(f['date']) > 11 else None,
            f.get('venue', {}).get('id'),
            f.get('venue', {}).get('name'),
            f.get('referee'),
            teams['home']['id'],
            teams['home']['name'],
            teams['away']['id'],
            teams['away']['name'],
            goals.get('home'),
            goals.get('away'),
            score.get('halftime', {}).get('home'),
            score.get('halftime', {}).get('away'),
            f['status']['short'],
            home_xg,
            away_xg,
            datetime.now().isoformat()
        ))
        
        # Also cache team info
        self._cache_team(conn, teams['home'])
        self._cache_team(conn, teams['away'])
    
    def _cache_team(self, conn: sqlite3.Connection, team: Dict):
        """Cache team information"""
        conn.execute("""
            INSERT OR IGNORE INTO teams (team_id, name, logo_url)
            VALUES (?, ?, ?)
        """, (team['id'], team['name'], team.get('logo')))
    
    def _ingest_fixture_statistics(self, conn: sqlite3.Connection, fixture_id: int):
        """Ingest detailed statistics for a fixture"""
        stats = self.client.get_fixture_statistics(fixture_id)
        
        if not stats:
            return
        
        for team_stats in stats:
            team_id = team_stats['team']['id']
            team_name = team_stats['team']['name']
            
            # Parse statistics into dict
            stat_dict = {}
            for stat in team_stats.get('statistics', []):
                stat_type = stat['type'].lower().replace(' ', '_').replace('%', 'pct')
                stat_dict[stat_type] = stat['value']
            
            # Determine if home team
            is_home = self._is_home_team(conn, fixture_id, team_id)
            
            conn.execute("""
                INSERT OR REPLACE INTO fixture_statistics (
                    fixture_id, team_id, team_name, is_home,
                    shots_total, shots_on_target, shots_off_target, shots_blocked,
                    shots_inside_box, shots_outside_box,
                    ball_possession, passes_total, passes_accurate, passes_pct,
                    corners, fouls, yellow_cards, red_cards,
                    tackles, interceptions, blocks, clearances, saves,
                    expected_goals
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fixture_id, team_id, team_name, is_home,
                self._parse_stat(stat_dict.get('shots_on_goal')) or self._parse_stat(stat_dict.get('total_shots')),
                self._parse_stat(stat_dict.get('shots_on_goal')),
                self._parse_stat(stat_dict.get('shots_off_goal')),
                self._parse_stat(stat_dict.get('blocked_shots')),
                self._parse_stat(stat_dict.get('shots_insidebox')),
                self._parse_stat(stat_dict.get('shots_outsidebox')),
                self._parse_stat(stat_dict.get('ball_possession')),
                self._parse_stat(stat_dict.get('total_passes')),
                self._parse_stat(stat_dict.get('passes_accurate')),
                self._parse_stat(stat_dict.get('passes_pct')),
                self._parse_stat(stat_dict.get('corner_kicks')),
                self._parse_stat(stat_dict.get('fouls')),
                self._parse_stat(stat_dict.get('yellow_cards')),
                self._parse_stat(stat_dict.get('red_cards')),
                self._parse_stat(stat_dict.get('tackles')),
                self._parse_stat(stat_dict.get('interceptions')),
                self._parse_stat(stat_dict.get('blocks')),
                self._parse_stat(stat_dict.get('clearances')),
                self._parse_stat(stat_dict.get('goalkeeper_saves')),
                self._parse_stat(stat_dict.get('expected_goals'))
            ))
    
    def _is_home_team(self, conn: sqlite3.Connection, fixture_id: int, team_id: int) -> int:
        """Check if team is home team for fixture"""
        cursor = conn.execute(
            "SELECT home_team_id FROM fixtures WHERE fixture_id = ?",
            (fixture_id,)
        )
        row = cursor.fetchone()
        return 1 if row and row[0] == team_id else 0
    
    def _parse_stat(self, value) -> Optional[float]:
        """Parse statistic value to number"""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # Remove % sign if present
            clean = value.replace('%', '').strip()
            try:
                return float(clean)
            except ValueError:
                return None
        return None
    
    # =========================================================================
    # INJURIES INGESTION
    # =========================================================================
    
    def ingest_injuries(self, league_code: str, season: int) -> int:
        """Ingest injury data for a league/season"""
        league_info = LEAGUES.get(league_code)
        if not league_info:
            return 0
        
        league_id = league_info['id']
        injuries = self.client.get_injuries(league_id=league_id, season=season)
        
        if not injuries:
            return 0
        
        count = 0
        with self._get_conn() as conn:
            for injury in injuries:
                try:
                    player = injury['player']
                    team = injury['team']
                    fixture = injury.get('fixture', {})
                    
                    conn.execute("""
                        INSERT OR REPLACE INTO injuries (
                            player_id, player_name, team_id, team_name,
                            fixture_id, league_id, season, injury_type, injury_reason, date
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        player['id'],
                        player['name'],
                        team['id'],
                        team['name'],
                        fixture.get('id'),
                        league_id,
                        season,
                        injury.get('type'),
                        injury.get('reason'),
                        fixture.get('date', '')[:10] if fixture.get('date') else None
                    ))
                    count += 1
                except Exception as e:
                    logger.error(f"Error inserting injury: {e}")
            
            conn.commit()
        
        logger.info(f"Ingested {count} injuries for {league_code} {season}")
        return count
    
    # =========================================================================
    # LINEUPS INGESTION
    # =========================================================================
    
    def ingest_lineups(self, fixture_id: int) -> int:
        """Ingest lineup data for a specific fixture"""
        lineups = self.client.get_fixture_lineups(fixture_id)
        
        if not lineups:
            return 0
        
        count = 0
        with self._get_conn() as conn:
            for team_lineup in lineups:
                team_id = team_lineup['team']['id']
                team_name = team_lineup['team']['name']
                formation = team_lineup.get('formation')
                
                # Starting XI
                for player in team_lineup.get('startXI', []):
                    p = player['player']
                    conn.execute("""
                        INSERT OR REPLACE INTO lineups (
                            fixture_id, team_id, team_name, formation,
                            player_id, player_name, position, position_grid,
                            is_starter, jersey_number
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        fixture_id, team_id, team_name, formation,
                        p['id'], p['name'], p.get('pos'), p.get('grid'),
                        1, p.get('number')
                    ))
                    count += 1
                
                # Substitutes
                for player in team_lineup.get('substitutes', []):
                    p = player['player']
                    conn.execute("""
                        INSERT OR REPLACE INTO lineups (
                            fixture_id, team_id, team_name, formation,
                            player_id, player_name, position, position_grid,
                            is_starter, jersey_number
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        fixture_id, team_id, team_name, formation,
                        p['id'], p['name'], p.get('pos'), None,
                        0, p.get('number')
                    ))
                    count += 1
            
            conn.commit()
        
        return count
    
    # =========================================================================
    # BULK INGESTION
    # =========================================================================
    
    def ingest_all(self, leagues: List[str] = None, seasons: List[int] = None,
                   include_stats: bool = True, include_injuries: bool = True):
        """
        Bulk ingest all data
        
        Args:
            leagues: List of league codes (default: DEFAULT_LEAGUES)
            seasons: List of seasons (default: DEFAULT_SEASONS)
            include_stats: Include match statistics
            include_injuries: Include injury data
        """
        leagues = leagues or DEFAULT_LEAGUES
        seasons = seasons or DEFAULT_SEASONS
        
        total_fixtures = 0
        total_injuries = 0
        
        for league_code in leagues:
            for season in seasons:
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing {league_code} - Season {season}")
                logger.info(f"{'='*60}")
                
                # Fixtures
                fixtures = self.ingest_fixtures(league_code, season, include_stats)
                total_fixtures += fixtures
                
                # Injuries
                if include_injuries:
                    injuries = self.ingest_injuries(league_code, season)
                    total_injuries += injuries
        
        logger.info(f"\n{'='*60}")
        logger.info(f"INGESTION COMPLETE")
        logger.info(f"Total fixtures: {total_fixtures}")
        logger.info(f"Total injuries: {total_injuries}")
        logger.info(f"{'='*60}")
    
    # =========================================================================
    # EXPORT TO PARQUET (for compatibility with existing code)
    # =========================================================================
    
    def export_to_parquet(self, output_path: Path = None) -> Path:
        """
        Export database to parquet format for DC model compatibility
        
        Creates a DataFrame matching the expected format of historical_results.parquet
        """
        output_path = output_path or PROCESSED_DIR / "historical_results.parquet"
        
        with self._get_conn() as conn:
            # Main query with statistics joined
            query = """
                SELECT 
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
                    -- Home stats
                    hs.shots_total as HS,
                    hs.shots_on_target as HST,
                    hs.corners as HC,
                    hs.fouls as HF,
                    hs.yellow_cards as HY,
                    hs.red_cards as HR,
                    hs.ball_possession as HPoss,
                    hs.passes_total as HPasses,
                    hs.passes_accurate as HPassesAcc,
                    hs.expected_goals as HxG,
                    -- Away stats  
                    aws.shots_total as AShots,
                    aws.shots_on_target as AST,
                    aws.corners as AC,
                    aws.fouls as AF,
                    aws.yellow_cards as AY,
                    aws.red_cards as AR,
                    aws.ball_possession as APoss,
                    aws.passes_total as APasses,
                    aws.passes_accurate as APassesAcc,
                    aws.expected_goals as AxG
                FROM fixtures f
                LEFT JOIN fixture_statistics hs 
                    ON f.fixture_id = hs.fixture_id AND hs.is_home = 1
                LEFT JOIN fixture_statistics aws 
                    ON f.fixture_id = aws.fixture_id AND aws.is_home = 0
                WHERE f.status = 'FT'
                ORDER BY f.date, f.league_code
            """
            
            df = pd.read_sql_query(query, conn)
        
        # Convert date
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Fill NaN for stats columns
        stat_cols = ['HS', 'HST', 'HC', 'HF', 'HY', 'HR', 'HPoss', 
                     'AShots', 'AST', 'AC', 'AF', 'AY', 'AR', 'APoss']
        for col in stat_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Save
        df.to_parquet(output_path, index=False)
        logger.info(f"Exported {len(df)} matches to {output_path}")
        
        return output_path


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_ingest(api_key: str, leagues: List[str] = None, seasons: List[int] = None):
    """Quick ingestion helper"""
    ingestor = APIFootballIngestor(api_key)
    ingestor.ingest_all(leagues=leagues, seasons=seasons)
    return ingestor.export_to_parquet()


def update_recent(api_key: str, days: int = 7, leagues: List[str] = None):
    """Update only recent fixtures"""
    ingestor = APIFootballIngestor(api_key)

    # Get recent fixtures
    from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')

    # Determine current season (API uses the year the season STARTED)
    current_month = datetime.now().month
    current_year = datetime.now().year
    # Football seasons typically start in August
    season = current_year if current_month >= 8 else current_year - 1

    # Use specified leagues or DEFAULT_LEAGUES
    leagues_to_update = leagues or DEFAULT_LEAGUES

    for league_code in leagues_to_update:
        league_info = LEAGUES.get(league_code)
        if not league_info:
            logger.warning(f"Unknown league code: {league_code}")
            continue

        logger.info(f"Updating {league_code}...")

        try:
            fixtures = ingestor.client.get_fixtures(
                league_id=league_info['id'],
                season=season,  # REQUIRED: API needs season even with date range
                from_date=from_date,
                to_date=to_date
            )
        except Exception as e:
            logger.error(f"API Error: {e}")
            continue

        with ingestor._get_conn() as conn:
            for fixture in fixtures:
                ingestor._insert_fixture(conn, fixture, league_code, league_info)

                if fixture['fixture']['status']['short'] == 'FT':
                    fixture_id = fixture['fixture']['id']
                    ingestor._ingest_fixture_statistics(conn, fixture_id)

            conn.commit()

    return ingestor.export_to_parquet()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Ingest data from API-Football")
    parser.add_argument("--api-key", default=os.environ.get("API_FOOTBALL_KEY"),
                        help="API key (or set API_FOOTBALL_KEY env var)")
    parser.add_argument("--leagues", nargs="+", default=DEFAULT_LEAGUES,
                        help="League codes to ingest")
    parser.add_argument("--seasons", nargs="+", type=int, default=DEFAULT_SEASONS,
                        help="Seasons to ingest")
    parser.add_argument("--update-recent", type=int, metavar="DAYS",
                        help="Only update recent N days")
    parser.add_argument("--no-stats", action="store_true",
                        help="Skip match statistics")
    parser.add_argument("--no-injuries", action="store_true",
                        help="Skip injury data")
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("ERROR: API key required. Set API_FOOTBALL_KEY or use --api-key")
        exit(1)
    
    if args.update_recent:
        output = update_recent(args.api_key, args.update_recent)
    else:
        ingestor = APIFootballIngestor(args.api_key)
        ingestor.ingest_all(
            leagues=args.leagues,
            seasons=args.seasons,
            include_stats=not args.no_stats,
            include_injuries=not args.no_injuries
        )
        output = ingestor.export_to_parquet()
    
    print(f"\nOutput: {output}")
