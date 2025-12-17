# days_since_match_fetcher.py
"""
Fetch days since last match using Claude API
Handles cup games and midweek fixtures not in league data
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
import anthropic
from typing import Dict, List, Optional

class DaysSinceMatchFetcher:
    """Use Claude API to get complete match schedules including cup games"""

    def __init__(self, api_key: str = None):
        """
        Initialize with Anthropic API key

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.cache_file = Path("data/match_schedule_cache.json")
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """Load cached match schedules"""
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """Save match schedules to cache"""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)

    def get_days_since_last_match(self,
                                   team: str,
                                   match_date: str,
                                   league: str = None) -> Optional[int]:
        """
        Get days since last match for a team (including cups)

        Args:
            team: Team name (e.g., "Arsenal", "Man City")
            match_date: Match date (YYYY-MM-DD)
            league: League code (e.g., "E0" for Premier League) - helps with context

        Returns:
            Days since last match, or None if not found
        """
        cache_key = f"{team}_{match_date}"

        # Check cache first
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            # Cache expires after 7 days
            cache_date = datetime.fromisoformat(cached['cached_at'])
            if datetime.now() - cache_date < timedelta(days=7):
                return cached['days_since_last_match']

        # Query Claude API
        days_since = self._query_claude(team, match_date, league)

        # Cache result
        if days_since is not None:
            self.cache[cache_key] = {
                'days_since_last_match': days_since,
                'cached_at': datetime.now().isoformat()
            }
            self._save_cache()

        return days_since

    def _query_claude(self, team: str, match_date: str, league: str = None) -> Optional[int]:
        """
        Query Claude API for match schedule

        Args:
            team: Team name
            match_date: Match date (YYYY-MM-DD)
            league: League code (optional, for context)

        Returns:
            Days since last match, or None if error
        """
        league_context = ""
        if league:
            league_map = {
                'E0': 'Premier League',
                'E1': 'Championship',
                'SP1': 'La Liga',
                'D1': 'Bundesliga',
                'I1': 'Serie A',
                'F1': 'Ligue 1'
            }
            league_name = league_map.get(league, league)
            league_context = f" (they play in the {league_name})"

        prompt = f"""I need to know when {team}{league_context} last played a competitive match before {match_date}.

Please find their most recent match (league OR cup competition) before {match_date} and calculate the number of days between that match and {match_date}.

IMPORTANT:
- Include ALL competitive matches (league, domestic cups, European competitions)
- Do NOT include friendlies or pre-season games
- The match must be BEFORE {match_date}
- If you can't find reliable information, respond with "UNKNOWN"

Respond in this exact format:
LAST_MATCH_DATE: YYYY-MM-DD
DAYS_SINCE: <number>

For example:
LAST_MATCH_DATE: 2024-11-26
DAYS_SINCE: 3"""

        try:
            message = self.client.messages.create(
                model="claude-3-5-haiku-20241022",  # Fast, cheap model
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            response_text = message.content[0].text.strip()

            # Parse response
            if "UNKNOWN" in response_text:
                return None

            # Extract DAYS_SINCE
            for line in response_text.split('\n'):
                if line.startswith('DAYS_SINCE:'):
                    days_str = line.split(':')[1].strip()
                    return int(days_str)

            return None

        except Exception as e:
            print(f"Error querying Claude for {team} on {match_date}: {e}")
            return None

    def enrich_dataframe(self,
                         df: pd.DataFrame,
                         home_col: str = 'HomeTeam',
                         away_col: str = 'AwayTeam',
                         date_col: str = 'Date',
                         league_col: str = 'League') -> pd.DataFrame:
        """
        Add days_since_last_match columns to dataframe

        Args:
            df: Input dataframe with matches
            home_col: Column name for home team
            away_col: Column name for away team
            date_col: Column name for match date
            league_col: Column name for league (optional)

        Returns:
            DataFrame with added columns:
                - Home_DaysSinceLast
                - Away_DaysSinceLast
        """
        df = df.copy()

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])

        home_days = []
        away_days = []

        total = len(df)
        print(f"Fetching days since last match for {total} matches...")

        for idx, row in df.iterrows():
            if idx % 10 == 0:
                print(f"  Progress: {idx}/{total} ({idx/total*100:.1f}%)")

            match_date_str = row[date_col].strftime('%Y-%m-%d')
            league = row[league_col] if league_col in df.columns else None

            # Get home team days
            home_days_val = self.get_days_since_last_match(
                row[home_col],
                match_date_str,
                league
            )
            home_days.append(home_days_val)

            # Get away team days
            away_days_val = self.get_days_since_last_match(
                row[away_col],
                match_date_str,
                league
            )
            away_days.append(away_days_val)

        df['Home_DaysSinceLast'] = home_days
        df['Away_DaysSinceLast'] = away_days

        print(f"\nEnrichment complete:")
        print(f"  Home team coverage: {df['Home_DaysSinceLast'].notna().sum()}/{total} ({df['Home_DaysSinceLast'].notna().sum()/total*100:.1f}%)")
        print(f"  Away team coverage: {df['Away_DaysSinceLast'].notna().sum()}/{total} ({df['Away_DaysSinceLast'].notna().sum()/total*100:.1f}%)")

        return df


def add_days_since_features(df: pd.DataFrame, use_api: bool = True) -> pd.DataFrame:
    """
    Convenience function to add days since last match features

    Args:
        df: Input dataframe
        use_api: If True, use Claude API. If False, use only league data

    Returns:
        DataFrame with days since last match columns
    """
    if use_api:
        try:
            fetcher = DaysSinceMatchFetcher()
            return fetcher.enrich_dataframe(df)
        except ValueError as e:
            print(f"Warning: {e}")
            print("Falling back to league-only data")
            return _add_days_since_from_league_data(df)
    else:
        return _add_days_since_from_league_data(df)


def _add_days_since_from_league_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback: Calculate days since last match using only league data
    (Will miss cup games and midweek fixtures)
    """
    df = df.copy()
    df = df.sort_values('Date')

    home_days = []
    away_days = []

    for idx, row in df.iterrows():
        current_date = row['Date']
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        # Find last match for home team
        home_prev = df[
            ((df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team)) &
            (df['Date'] < current_date)
        ].sort_values('Date', ascending=False)

        if len(home_prev) > 0:
            last_home_date = home_prev.iloc[0]['Date']
            home_days.append((current_date - last_home_date).days)
        else:
            home_days.append(None)

        # Find last match for away team
        away_prev = df[
            ((df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)) &
            (df['Date'] < current_date)
        ].sort_values('Date', ascending=False)

        if len(away_prev) > 0:
            last_away_date = away_prev.iloc[0]['Date']
            away_days.append((current_date - last_away_date).days)
        else:
            away_days.append(None)

    df['Home_DaysSinceLast'] = home_days
    df['Away_DaysSinceLast'] = away_days

    print("Warning: Using league-only data (missing cup games)")
    print(f"  Coverage: {df['Home_DaysSinceLast'].notna().sum()}/{len(df)} matches")

    return df


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python days_since_match_fetcher.py <team> <date> [league]")
        print("Example: python days_since_match_fetcher.py Arsenal 2024-11-30 E0")
        sys.exit(1)

    team = sys.argv[1]
    date = sys.argv[2]
    league = sys.argv[3] if len(sys.argv) > 3 else None

    fetcher = DaysSinceMatchFetcher()
    days = fetcher.get_days_since_last_match(team, date, league)

    if days is not None:
        print(f"\n{team} last played {days} days before {date}")
    else:
        print(f"\nCould not determine days since last match for {team} before {date}")
