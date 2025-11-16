# auto_fixture_downloader.py
"""
Auto-download upcoming fixtures from both football-data.org API and API-Football
Combines domestic leagues + Champions League into single upcoming_fixtures.csv
"""

import os
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

# API Keys (set as environment variables or in config)
FOOTBALL_DATA_ORG_TOKEN = os.environ.get("FOOTBALL_DATA_ORG_TOKEN", "")
API_FOOTBALL_KEY = os.environ.get("API_FOOTBALL_KEY", "")

# API Endpoints
FOOTBALL_DATA_ORG_BASE = "https://api.football-data.org/v4"
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

# Competition IDs for football-data.org
FOOTBALL_DATA_COMPETITIONS = {
    'PL': 2021,   # Premier League
    'ELC': 2016,  # Championship
    'BL1': 2002,  # Bundesliga
    'PD': 2014,   # La Liga
    'SA': 2019,   # Serie A
    'FL1': 2015,  # Ligue 1
    'DED': 2003,  # Eredivisie
    'PPL': 2017,  # Primeira Liga
    'CL': 2001,   # Champions League
}

# API-Football Competition IDs
API_FOOTBALL_COMPETITIONS = {
    'CL': 2,      # Champions League
    'EL': 3,      # Europa League
    'ECL': 848,   # Conference League
}

# League code mapping (for normalization)
LEAGUE_CODE_MAPPING = {
    'PL': 'E0',
    'ELC': 'E1',
    'BL1': 'D1',
    'PD': 'SP1',
    'SA': 'I1',
    'FL1': 'F1',
    'DED': 'N1',
    'PPL': 'P1',
    'CL': 'CL',
    'EL': 'EL',
    'ECL': 'ECL',
}

# Team name normalization (API-Football → Standard names)
TEAM_NAME_MAPPING = {
    'Manchester City': 'Man City',
    'Manchester United': 'Man United',
    'Tottenham Hotspur': 'Tottenham',
    'Newcastle United': 'Newcastle',
    'West Ham United': 'West Ham',
    'Wolverhampton Wanderers': 'Wolves',
    'Brighton & Hove Albion': 'Brighton',
    'Nottingham Forest': "Nott'm Forest",
    'AFC Bournemouth': 'Bournemouth',
    'Paris Saint Germain': 'Paris SG',
    'Bayern Munich': 'Bayern Munich',
    'Borussia Dortmund': 'Dortmund',
    'RB Leipzig': 'RB Leipzig',
    'Bayer Leverkusen': 'Leverkusen',
}

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# FOOTBALL-DATA.ORG FETCHER
# ============================================================================

class FootballDataOrgFetcher:
    """Fetch fixtures from football-data.org API"""
    
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.base_url = FOOTBALL_DATA_ORG_BASE
        self.session = requests.Session()
        self.session.headers.update({'X-Auth-Token': api_token})
    
    def get_fixtures(self, competition_id: int, date_from: str, date_to: str) -> List[Dict]:
        """Fetch fixtures for a competition"""
        url = f"{self.base_url}/competitions/{competition_id}/matches"
        params = {
            'dateFrom': date_from,
            'dateTo': date_to,
            'status': 'SCHEDULED'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('matches', [])
        except Exception as e:
            print(f"⚠️ Error fetching {competition_id}: {e}")
            return []
    
    def process_fixtures(self, fixtures: List[Dict], league_code: str) -> pd.DataFrame:
        """Process fixtures into standard format"""
        processed = []
        
        for match in fixtures:
            try:
                processed.append({
                    'Date': match['utcDate'][:10],  # YYYY-MM-DD
                    'League': league_code,
                    'HomeTeam': self._normalize_team_name(match['homeTeam']['name']),
                    'AwayTeam': self._normalize_team_name(match['awayTeam']['name']),
                    'Source': 'football-data.org'
                })
            except KeyError:
                continue
        
        return pd.DataFrame(processed)
    
    def _normalize_team_name(self, name: str) -> str:
        """Normalize team names to match historical data"""
        return TEAM_NAME_MAPPING.get(name, name)

# ============================================================================
# API-FOOTBALL FETCHER
# ============================================================================

class APIFootballFetcher:
    """Fetch fixtures from API-Football"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = API_FOOTBALL_BASE
        self.session = requests.Session()
        self.session.headers.update({'x-apisports-key': api_key})
    
    def get_fixtures(self, league_id: int, season: int) -> List[Dict]:
        """Fetch upcoming fixtures for a league"""
        url = f"{self.base_url}/fixtures"
        params = {
            'league': league_id,
            'season': season,
            'status': 'NS',  # Not Started
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('response', [])
        except Exception as e:
            print(f"⚠️ Error fetching league {league_id}: {e}")
            return []
    
    def process_fixtures(self, fixtures: List[Dict], league_code: str) -> pd.DataFrame:
        """Process API-Football fixtures into standard format"""
        processed = []
        
        for match in fixtures:
            try:
                fixture = match['fixture']
                teams = match['teams']
                
                # Parse date (API-Football gives full datetime)
                match_date = datetime.fromisoformat(fixture['date'].replace('Z', '+00:00'))
                
                processed.append({
                    'Date': match_date.strftime('%Y-%m-%d'),
                    'League': league_code,
                    'HomeTeam': self._normalize_team_name(teams['home']['name']),
                    'AwayTeam': self._normalize_team_name(teams['away']['name']),
                    'Source': 'API-Football'
                })
            except (KeyError, ValueError):
                continue
        
        return pd.DataFrame(processed)
    
    def _normalize_team_name(self, name: str) -> str:
        """Normalize team names"""
        return TEAM_NAME_MAPPING.get(name, name)

# ============================================================================
# COMBINED FIXTURE DOWNLOADER
# ============================================================================

class CombinedFixtureDownloader:
    """Download and combine fixtures from multiple sources"""
    
    def __init__(self):
        self.fd_fetcher = None
        self.af_fetcher = None
        
        # Initialize fetchers if API keys available
        if FOOTBALL_DATA_ORG_TOKEN:
            self.fd_fetcher = FootballDataOrgFetcher(FOOTBALL_DATA_ORG_TOKEN)
            print("✅ football-data.org API initialized")
        
        if API_FOOTBALL_KEY:
            self.af_fetcher = APIFootballFetcher(API_FOOTBALL_KEY)
            print("✅ API-Football initialized")
    
    def download_all_fixtures(self, days_ahead: int = 14) -> pd.DataFrame:
        """Download fixtures from all sources"""
        date_from = datetime.now().strftime('%Y-%m-%d')
        date_to = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        current_season = datetime.now().year if datetime.now().month >= 7 else datetime.now().year - 1
        
        all_fixtures = []
        
        # Download from football-data.org
        if self.fd_fetcher:
            print(f"\n📡 Downloading from football-data.org ({date_from} to {date_to})")
            
            for comp_name, comp_id in FOOTBALL_DATA_COMPETITIONS.items():
                league_code = LEAGUE_CODE_MAPPING.get(comp_name, comp_name)
                print(f"   • {comp_name} ({league_code})...", end='')
                
                fixtures = self.fd_fetcher.get_fixtures(comp_id, date_from, date_to)
                if fixtures:
                    df = self.fd_fetcher.process_fixtures(fixtures, league_code)
                    all_fixtures.append(df)
                    print(f" {len(df)} matches")
                else:
                    print(" no matches")
        
        # Download from API-Football (Champions League, Europa League, etc.)
        if self.af_fetcher:
            print(f"\n📡 Downloading from API-Football (season {current_season})")
            
            for comp_name, comp_id in API_FOOTBALL_COMPETITIONS.items():
                league_code = LEAGUE_CODE_MAPPING.get(comp_name, comp_name)
                print(f"   • {comp_name} ({league_code})...", end='')
                
                fixtures = self.af_fetcher.get_fixtures(comp_id, current_season)
                if fixtures:
                    df = self.af_fetcher.process_fixtures(fixtures, league_code)
                    # Filter to next 14 days
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df[(df['Date'] >= date_from) & (df['Date'] <= date_to)]
                    if not df.empty:
                        all_fixtures.append(df)
                        print(f" {len(df)} matches")
                    else:
                        print(" no matches in range")
                else:
                    print(" no matches")
        
        # Combine all fixtures
        if not all_fixtures:
            print("\n⚠️ No fixtures downloaded from any source!")
            return pd.DataFrame(columns=['Date', 'League', 'HomeTeam', 'AwayTeam'])
        
        combined = pd.concat(all_fixtures, ignore_index=True)
        
        # Remove duplicates (same match from multiple sources)
        combined = combined.drop_duplicates(subset=['Date', 'League', 'HomeTeam', 'AwayTeam'])
        
        # Sort by date
        combined['Date'] = pd.to_datetime(combined['Date'])
        combined = combined.sort_values(['Date', 'League'])
        combined['Date'] = combined['Date'].dt.strftime('%Y-%m-%d')
        
        # Final cleanup - ensure column order
        combined = combined[['Date', 'League', 'HomeTeam', 'AwayTeam']]
        
        return combined
    
    def save_fixtures(self, df: pd.DataFrame, filename: str = "upcoming_fixtures.csv"):
        """Save fixtures to CSV"""
        output_path = OUTPUT_DIR / filename
        df.to_csv(output_path, index=False)
        print(f"\n✅ Saved {len(df)} fixtures to {output_path}")
        
        # Also save as XLSX for manual review
        xlsx_path = OUTPUT_DIR / filename.replace('.csv', '.xlsx')
        df.to_excel(xlsx_path, index=False)
        print(f"✅ Saved Excel version to {xlsx_path}")
        
        return output_path
    
    def generate_summary(self, df: pd.DataFrame):
        """Print summary of downloaded fixtures"""
        print(f"\n{'='*60}")
        print("📊 FIXTURE DOWNLOAD SUMMARY")
        print(f"{'='*60}")
        print(f"Total fixtures: {len(df)}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"\nFixtures by league:")
        for league, count in df['League'].value_counts().items():
            print(f"   • {league}: {count} matches")
        print(f"{'='*60}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def download_upcoming_fixtures(days_ahead: int = 14) -> Path:
    """
    Main function to download all upcoming fixtures
    
    Args:
        days_ahead: Number of days to look ahead (default 14)
    
    Returns:
        Path to generated CSV file
    """
    print("🔄 AUTO-FIXTURE DOWNLOAD SYSTEM")
    print("="*60)
    
    # Check API keys
    if not FOOTBALL_DATA_ORG_TOKEN and not API_FOOTBALL_KEY:
        print("⚠️ No API keys configured!")
        print("Set environment variables:")
        print("  • FOOTBALL_DATA_ORG_TOKEN")
        print("  • API_FOOTBALL_KEY")
        print("\nFalling back to manual fixture file...")
        return None
    
    # Initialize downloader
    downloader = CombinedFixtureDownloader()
    
    # Download fixtures
    fixtures_df = downloader.download_all_fixtures(days_ahead)
    
    if fixtures_df.empty:
        print("❌ No fixtures downloaded!")
        return None
    
    # Save to file
    output_path = downloader.save_fixtures(fixtures_df)
    
    # Generate summary
    downloader.generate_summary(fixtures_df)
    
    return output_path

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download upcoming football fixtures")
    parser.add_argument('--days', type=int, default=14, help='Days ahead to download (default: 14)')
    parser.add_argument('--output', type=str, default='upcoming_fixtures.csv', help='Output filename')
    
    args = parser.parse_args()
    
    result = download_upcoming_fixtures(args.days)
    
    if result:
        print(f"\n✅ SUCCESS! Fixtures saved to {result}")
        print("\n💡 Next step: Run your prediction system with this file")
    else:
        print("\n❌ Download failed - check API keys and try again")
