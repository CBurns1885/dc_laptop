# api_football_client.py
"""
API-Football Client Wrapper
Handles all API interactions with rate limiting and caching

API Documentation: https://www.api-football.com/documentation-v3
"""

import requests
import time
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """API Configuration"""
    api_key: str
    base_url: str = "https://v3.football.api-sports.io"
    requests_per_minute: int = 30  # Free plan limit
    cache_dir: Path = Path("data/api_cache")
    cache_expiry_hours: int = 24


class RateLimiter:
    """Handle API rate limiting"""
    
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.request_times: List[float] = []
    
    def wait_if_needed(self):
        """Wait if we've hit rate limit"""
        now = time.time()
        minute_ago = now - 60
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if t > minute_ago]
        
        if len(self.request_times) >= self.requests_per_minute:
            # Wait until oldest request is >1 minute old
            sleep_time = self.request_times[0] - minute_ago + 0.1
            if sleep_time > 0:
                logger.info(f"Rate limit reached, waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
        
        self.request_times.append(time.time())


class CacheManager:
    """SQLite-based cache for API responses"""
    
    def __init__(self, cache_dir: Path, expiry_hours: int = 24):
        self.cache_dir = cache_dir
        self.expiry_hours = expiry_hours
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = cache_dir / "api_cache.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize cache database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    cache_key TEXT PRIMARY KEY,
                    response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON cache(created_at)")
    
    def get(self, key: str) -> Optional[Dict]:
        """Get cached response if not expired"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT response, created_at FROM cache WHERE cache_key = ?",
                (key,)
            )
            row = cursor.fetchone()
            
            if row:
                response, created_at = row
                created = datetime.fromisoformat(created_at)
                if datetime.now() - created < timedelta(hours=self.expiry_hours):
                    return json.loads(response)
                else:
                    # Expired, delete it
                    conn.execute("DELETE FROM cache WHERE cache_key = ?", (key,))
        return None
    
    def set(self, key: str, response: Dict):
        """Cache response"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (cache_key, response, created_at) VALUES (?, ?, ?)",
                (key, json.dumps(response), datetime.now().isoformat())
            )
    
    def clear_expired(self):
        """Remove expired cache entries"""
        cutoff = datetime.now() - timedelta(hours=self.expiry_hours)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache WHERE created_at < ?", (cutoff.isoformat(),))


class APIFootballClient:
    """
    Main API-Football client
    
    Usage:
        client = APIFootballClient(api_key="your_key")
        fixtures = client.get_fixtures(league_id=39, season=2024)
    """
    
    def __init__(self, api_key: str, config: APIConfig = None):
        self.api_key = api_key
        self.config = config or APIConfig(api_key=api_key)
        self.headers = {
            "x-apisports-key": api_key
        }
        self.rate_limiter = RateLimiter(self.config.requests_per_minute)
        self.cache = CacheManager(self.config.cache_dir, self.config.cache_expiry_hours)
        self.request_count = 0
    
    def _make_request(self, endpoint: str, params: Dict = None, use_cache: bool = True) -> Dict:
        """Make API request with caching and rate limiting"""
        params = params or {}
        cache_key = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
        
        # Check cache first
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                logger.debug(f"Cache hit: {endpoint}")
                return cached
        
        # Rate limit
        self.rate_limiter.wait_if_needed()
        
        # Make request
        url = f"{self.config.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            self.request_count += 1
            
            # Check for API errors
            if data.get("errors"):
                logger.error(f"API Error: {data['errors']}")
                return {"response": [], "errors": data["errors"]}
            
            # Cache successful response
            if use_cache and data.get("response"):
                self.cache.set(cache_key, data)
            
            logger.info(f"API request #{self.request_count}: {endpoint} ({len(data.get('response', []))} results)")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return {"response": [], "errors": [str(e)]}
    
    # =========================================================================
    # LEAGUES & COMPETITIONS
    # =========================================================================
    
    def get_leagues(self, country: str = None, season: int = None, 
                    league_type: str = None) -> List[Dict]:
        """
        Get available leagues
        
        Args:
            country: Filter by country name (e.g., "England")
            season: Filter by season year
            league_type: "league" or "cup"
        
        Returns:
            List of league objects
        """
        params = {}
        if country:
            params["country"] = country
        if season:
            params["season"] = season
        if league_type:
            params["type"] = league_type
        
        return self._make_request("leagues", params).get("response", [])
    
    def get_league_seasons(self, league_id: int) -> List[int]:
        """Get available seasons for a league"""
        leagues = self._make_request("leagues", {"id": league_id}).get("response", [])
        if leagues:
            return leagues[0].get("seasons", [])
        return []
    
    # =========================================================================
    # FIXTURES
    # =========================================================================
    
    def get_fixtures(self, league_id: int = None, season: int = None,
                     team_id: int = None, date: str = None,
                     from_date: str = None, to_date: str = None,
                     last: int = None, next_n: int = None,
                     status: str = None) -> List[Dict]:
        """
        Get fixtures/matches
        
        Args:
            league_id: League ID
            season: Season year (e.g., 2024)
            team_id: Team ID (gets all matches for team)
            date: Specific date (YYYY-MM-DD)
            from_date: Start date range
            to_date: End date range
            last: Last N matches for team
            next_n: Next N matches for team
            status: Match status (NS, FT, etc.)
        
        Returns:
            List of fixture objects with full details
        """
        params = {}
        if league_id:
            params["league"] = league_id
        if season:
            params["season"] = season
        if team_id:
            params["team"] = team_id
        if date:
            params["date"] = date
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        if last:
            params["last"] = last
        if next_n:
            params["next"] = next_n
        if status:
            params["status"] = status
        
        return self._make_request("fixtures", params).get("response", [])
    
    def get_fixture_by_id(self, fixture_id: int) -> Optional[Dict]:
        """Get single fixture by ID"""
        fixtures = self._make_request("fixtures", {"id": fixture_id}).get("response", [])
        return fixtures[0] if fixtures else None
    
    def get_fixture_statistics(self, fixture_id: int) -> List[Dict]:
        """
        Get detailed match statistics
        
        Returns stats like: shots, possession, corners, fouls, cards, passes, etc.
        """
        return self._make_request("fixtures/statistics", {"fixture": fixture_id}).get("response", [])
    
    def get_fixture_events(self, fixture_id: int) -> List[Dict]:
        """Get match events (goals, cards, subs)"""
        return self._make_request("fixtures/events", {"fixture": fixture_id}).get("response", [])
    
    def get_fixture_lineups(self, fixture_id: int) -> List[Dict]:
        """Get team lineups and formations"""
        return self._make_request("fixtures/lineups", {"fixture": fixture_id}).get("response", [])
    
    def get_fixture_players(self, fixture_id: int) -> List[Dict]:
        """Get player statistics for a fixture"""
        return self._make_request("fixtures/players", {"fixture": fixture_id}).get("response", [])
    
    # =========================================================================
    # HEAD TO HEAD
    # =========================================================================
    
    def get_h2h(self, team1_id: int, team2_id: int, last: int = 20) -> List[Dict]:
        """
        Get head-to-head history between two teams
        
        Args:
            team1_id: First team ID
            team2_id: Second team ID  
            last: Number of past meetings to return
        
        Returns:
            List of past fixtures between the teams
        """
        params = {
            "h2h": f"{team1_id}-{team2_id}",
            "last": last
        }
        return self._make_request("fixtures/headtohead", params).get("response", [])
    
    # =========================================================================
    # TEAMS
    # =========================================================================
    
    def get_teams(self, league_id: int = None, season: int = None,
                  team_id: int = None, country: str = None) -> List[Dict]:
        """Get teams"""
        params = {}
        if league_id:
            params["league"] = league_id
        if season:
            params["season"] = season
        if team_id:
            params["id"] = team_id
        if country:
            params["country"] = country
        
        return self._make_request("teams", params).get("response", [])
    
    def get_team_statistics(self, team_id: int, league_id: int, season: int) -> Dict:
        """Get team statistics for a season"""
        params = {
            "team": team_id,
            "league": league_id,
            "season": season
        }
        response = self._make_request("teams/statistics", params).get("response", {})
        return response
    
    # =========================================================================
    # PLAYERS
    # =========================================================================
    
    def get_players(self, team_id: int = None, league_id: int = None,
                    season: int = None, player_id: int = None) -> List[Dict]:
        """Get player information and statistics"""
        params = {}
        if team_id:
            params["team"] = team_id
        if league_id:
            params["league"] = league_id
        if season:
            params["season"] = season
        if player_id:
            params["id"] = player_id
        
        return self._make_request("players", params).get("response", [])
    
    def get_squad(self, team_id: int) -> List[Dict]:
        """Get current team squad"""
        return self._make_request("players/squads", {"team": team_id}).get("response", [])
    
    def get_top_scorers(self, league_id: int, season: int) -> List[Dict]:
        """Get league top scorers"""
        params = {"league": league_id, "season": season}
        return self._make_request("players/topscorers", params).get("response", [])
    
    # =========================================================================
    # INJURIES & SUSPENSIONS
    # =========================================================================
    
    def get_injuries(self, league_id: int = None, season: int = None,
                     team_id: int = None, fixture_id: int = None,
                     player_id: int = None) -> List[Dict]:
        """
        Get injury information
        
        Args:
            league_id: Filter by league
            season: Filter by season
            team_id: Filter by team
            fixture_id: Get injuries for specific fixture
            player_id: Get injuries for specific player
        
        Returns:
            List of injury records
        """
        params = {}
        if league_id:
            params["league"] = league_id
        if season:
            params["season"] = season
        if team_id:
            params["team"] = team_id
        if fixture_id:
            params["fixture"] = fixture_id
        if player_id:
            params["player"] = player_id
        
        return self._make_request("injuries", params).get("response", [])
    
    def get_sidelined(self, player_id: int = None, team_id: int = None) -> List[Dict]:
        """Get sidelined/injured players"""
        params = {}
        if player_id:
            params["player"] = player_id
        if team_id:
            params["team"] = team_id
        
        return self._make_request("sidelined", params).get("response", [])
    
    # =========================================================================
    # STANDINGS
    # =========================================================================
    
    def get_standings(self, league_id: int, season: int) -> List[Dict]:
        """Get league standings/table"""
        params = {"league": league_id, "season": season}
        return self._make_request("standings", params).get("response", [])
    
    # =========================================================================
    # ODDS & PREDICTIONS
    # =========================================================================
    
    def get_odds(self, fixture_id: int = None, league_id: int = None,
                 season: int = None, bookmaker_id: int = None) -> List[Dict]:
        """Get pre-match odds"""
        params = {}
        if fixture_id:
            params["fixture"] = fixture_id
        if league_id:
            params["league"] = league_id
        if season:
            params["season"] = season
        if bookmaker_id:
            params["bookmaker"] = bookmaker_id
        
        return self._make_request("odds", params).get("response", [])
    
    def get_predictions(self, fixture_id: int) -> Dict:
        """
        Get API predictions for a fixture
        
        Returns predictions for: winner, goals, advice, comparison stats
        """
        response = self._make_request("predictions", {"fixture": fixture_id}).get("response", [])
        return response[0] if response else {}
    
    # =========================================================================
    # COACHES & REFEREES
    # =========================================================================
    
    def get_coaches(self, team_id: int = None, coach_id: int = None) -> List[Dict]:
        """Get coach information"""
        params = {}
        if team_id:
            params["team"] = team_id
        if coach_id:
            params["id"] = coach_id
        
        return self._make_request("coachs", params).get("response", [])
    
    # =========================================================================
    # VENUES
    # =========================================================================
    
    def get_venues(self, venue_id: int = None, country: str = None) -> List[Dict]:
        """Get venue information"""
        params = {}
        if venue_id:
            params["id"] = venue_id
        if country:
            params["country"] = country
        
        return self._make_request("venues", params).get("response", [])
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_status(self) -> Dict:
        """Get API account status (requests remaining, etc.)"""
        return self._make_request("status", use_cache=False)
    
    def get_timezone(self) -> List[str]:
        """Get available timezones"""
        return self._make_request("timezone").get("response", [])


# =============================================================================
# LEAGUE CONFIGURATION
# =============================================================================

# Major leagues with their API-Football IDs
LEAGUES = {
    # Top 5 European Leagues
    'E0': {'id': 39, 'name': 'Premier League', 'country': 'England', 'type': 'league'},
    'E1': {'id': 40, 'name': 'Championship', 'country': 'England', 'type': 'league'},
    'E2': {'id': 41, 'name': 'League One', 'country': 'England', 'type': 'league'},
    'E3': {'id': 42, 'name': 'League Two', 'country': 'England', 'type': 'league'},
    'D1': {'id': 78, 'name': 'Bundesliga', 'country': 'Germany', 'type': 'league'},
    'D2': {'id': 79, 'name': '2. Bundesliga', 'country': 'Germany', 'type': 'league'},
    'SP1': {'id': 140, 'name': 'La Liga', 'country': 'Spain', 'type': 'league'},
    'SP2': {'id': 141, 'name': 'Segunda Division', 'country': 'Spain', 'type': 'league'},
    'I1': {'id': 135, 'name': 'Serie A', 'country': 'Italy', 'type': 'league'},
    'I2': {'id': 136, 'name': 'Serie B', 'country': 'Italy', 'type': 'league'},
    'F1': {'id': 61, 'name': 'Ligue 1', 'country': 'France', 'type': 'league'},
    'F2': {'id': 62, 'name': 'Ligue 2', 'country': 'France', 'type': 'league'},
    
    # Other European Leagues
    'N1': {'id': 88, 'name': 'Eredivisie', 'country': 'Netherlands', 'type': 'league'},
    'P1': {'id': 94, 'name': 'Primeira Liga', 'country': 'Portugal', 'type': 'league'},
    'B1': {'id': 144, 'name': 'Jupiler Pro League', 'country': 'Belgium', 'type': 'league'},
    'T1': {'id': 203, 'name': 'Super Lig', 'country': 'Turkey', 'type': 'league'},
    'G1': {'id': 197, 'name': 'Super League', 'country': 'Greece', 'type': 'league'},
    
    # Scottish - Full pyramid
    'SC0': {'id': 179, 'name': 'Premiership', 'country': 'Scotland', 'type': 'league'},
    'SC1': {'id': 180, 'name': 'Championship', 'country': 'Scotland', 'type': 'league'},
    'SC2': {'id': 181, 'name': 'League One', 'country': 'Scotland', 'type': 'league'},
    'SC3': {'id': 182, 'name': 'League Two', 'country': 'Scotland', 'type': 'league'},

    # English National League
    'EC': {'id': 43, 'name': 'National League', 'country': 'England', 'type': 'league'},
    
    # ==========================================================================
    # CUP COMPETITIONS (NEW!)
    # ==========================================================================
    
    # English Cups
    'FA_CUP': {'id': 45, 'name': 'FA Cup', 'country': 'England', 'type': 'cup'},
    'EFL_CUP': {'id': 48, 'name': 'EFL Cup', 'country': 'England', 'type': 'cup'},
    'COMMUNITY_SHIELD': {'id': 528, 'name': 'Community Shield', 'country': 'England', 'type': 'cup'},
    
    # German Cups
    'DFB_POKAL': {'id': 81, 'name': 'DFB Pokal', 'country': 'Germany', 'type': 'cup'},
    
    # Spanish Cups
    'COPA_DEL_REY': {'id': 143, 'name': 'Copa del Rey', 'country': 'Spain', 'type': 'cup'},
    
    # Italian Cups
    'COPPA_ITALIA': {'id': 137, 'name': 'Coppa Italia', 'country': 'Italy', 'type': 'cup'},
    
    # French Cups
    'COUPE_DE_FRANCE': {'id': 66, 'name': 'Coupe de France', 'country': 'France', 'type': 'cup'},
    
    # ==========================================================================
    # EUROPEAN COMPETITIONS (NEW!)
    # ==========================================================================
    
    'UCL': {'id': 2, 'name': 'UEFA Champions League', 'country': 'World', 'type': 'cup'},
    'UEL': {'id': 3, 'name': 'UEFA Europa League', 'country': 'World', 'type': 'cup'},
    'UECL': {'id': 848, 'name': 'UEFA Conference League', 'country': 'World', 'type': 'cup'},
    'UEFA_SUPER_CUP': {'id': 531, 'name': 'UEFA Super Cup', 'country': 'World', 'type': 'cup'},
    
    # ==========================================================================
    # INTERNATIONAL (NEW!)
    # ==========================================================================
    
    'WORLD_CUP': {'id': 1, 'name': 'World Cup', 'country': 'World', 'type': 'cup'},
    'EUROS': {'id': 4, 'name': 'Euro Championship', 'country': 'World', 'type': 'cup'},
    'NATIONS_LEAGUE': {'id': 5, 'name': 'UEFA Nations League', 'country': 'World', 'type': 'cup'},
}

# Reverse lookup: API ID -> our code
LEAGUE_ID_TO_CODE = {v['id']: k for k, v in LEAGUES.items()}


def get_league_id(code: str) -> Optional[int]:
    """Convert league code to API ID"""
    return LEAGUES.get(code, {}).get('id')


def get_league_code(api_id: int) -> Optional[str]:
    """Convert API ID to league code"""
    return LEAGUE_ID_TO_CODE.get(api_id)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Test with environment variable or placeholder
    import os
    api_key = os.environ.get("API_FOOTBALL_KEY", "YOUR_API_KEY_HERE")
    
    if api_key == "YOUR_API_KEY_HERE":
        print("Set API_FOOTBALL_KEY environment variable to test")
        print("\nAvailable leagues configured:")
        for code, info in LEAGUES.items():
            print(f"  {code}: {info['name']} ({info['type']})")
    else:
        client = APIFootballClient(api_key)
        
        # Test status
        status = client.get_status()
        print(f"API Status: {status}")
        
        # Test fixtures
        fixtures = client.get_fixtures(league_id=39, season=2024)
        print(f"Premier League 2024 fixtures: {len(fixtures)}")
