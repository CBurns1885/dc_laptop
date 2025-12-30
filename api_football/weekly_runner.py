# weekly_runner.py
"""
Weekly Prediction Runner
Run this once per week to:
1. Update recent match data
2. Rebuild features
3. Generate predictions for upcoming matches
4. Export results for betting

Usage:
    python weekly_runner.py
    python weekly_runner.py --days 7
    python weekly_runner.py --leagues E0 D1 SP1
    python weekly_runner.py --dry-run
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "predictions"
DB_PATH = DATA_DIR / "football_api.db"

# Create directories
for d in [DATA_DIR, PROCESSED_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Default configuration
DEFAULT_API_KEY = "0f17fdba78d15a625710f7244a1cc770"

# Leagues to track - adjust based on your betting focus
DEFAULT_LEAGUES = [
    # Top 5 leagues
    'E0',       # Premier League
    'E1',       # Championship  
    'D1',       # Bundesliga
    'SP1',      # La Liga
    'I1',       # Serie A
    'F1',       # Ligue 1
    # English lower
    'E2',       # League One
    'E3',       # League Two
    # Cups (if you bet on these)
    'FA_CUP',
    'EFL_CUP',
]

# League ID mapping for API
LEAGUE_IDS = {
    'E0': 39,    # Premier League
    'E1': 40,    # Championship
    'E2': 41,    # League One
    'E3': 42,    # League Two
    'D1': 78,    # Bundesliga
    'D2': 79,    # Bundesliga 2
    'SP1': 140,  # La Liga
    'SP2': 141,  # La Liga 2
    'I1': 135,   # Serie A
    'I2': 136,   # Serie B
    'F1': 61,    # Ligue 1
    'F2': 62,    # Ligue 2
    'N1': 88,    # Eredivisie
    'P1': 94,    # Primeira Liga
    'B1': 144,   # Jupiler Pro League
    'SC0': 179,  # Scottish Premiership
    'FA_CUP': 45,
    'EFL_CUP': 48,
    'UCL': 2,
    'UEL': 3,
}


def check_api_status(api_key: str) -> dict:
    """Check API status and remaining requests"""
    import requests
    
    response = requests.get(
        "https://v3.football.api-sports.io/status",
        headers={"x-apisports-key": api_key},
        timeout=10
    )
    
    data = response.json()
    resp = data.get("response", {})
    
    if isinstance(resp, dict):
        requests_info = resp.get("requests", {})
        return {
            "current": requests_info.get("current", 0),
            "limit": requests_info.get("limit_day", 0),
            "remaining": requests_info.get("limit_day", 0) - requests_info.get("current", 0)
        }
    return {"current": 0, "limit": 0, "remaining": 0}


def update_recent_data(api_key: str, days: int = 14, leagues: list = None):
    """
    Update database with recent match data
    
    Args:
        api_key: API-Football key
        days: Days of history to update
        leagues: League codes to update
    """
    from data_ingest_api import APIFootballIngestor
    
    leagues = leagues or DEFAULT_LEAGUES
    
    logger.info(f"Updating recent data for {len(leagues)} leagues, last {days} days")
    
    ingestor = APIFootballIngestor(api_key, db_path=DB_PATH)
    
    # Get recent fixtures for each league
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    total_fixtures = 0
    for league_code in leagues:
        league_id = LEAGUE_IDS.get(league_code)
        if not league_id:
            logger.warning(f"Unknown league code: {league_code}")
            continue
        
        try:
            # Get finished fixtures
            fixtures = ingestor.client.get_fixtures(
                league_id=league_id,
                season=2024,
                from_date=start_date.strftime('%Y-%m-%d'),
                to_date=end_date.strftime('%Y-%m-%d')
            )
            
            if fixtures.get("response"):
                count = len(fixtures["response"])
                total_fixtures += count
                logger.info(f"  {league_code}: {count} fixtures")
                
                # Process each fixture
                for fixture in fixtures["response"]:
                    ingestor._process_fixture(fixture, league_code)
        
        except Exception as e:
            logger.error(f"  {league_code}: Error - {e}")
    
    logger.info(f"Updated {total_fixtures} total fixtures")
    return total_fixtures


def get_upcoming_fixtures(api_key: str, days: int = 7, leagues: list = None) -> pd.DataFrame:
    """
    Get upcoming fixtures from API
    
    Args:
        api_key: API-Football key
        days: Days ahead to fetch
        leagues: League codes
    
    Returns:
        DataFrame of upcoming fixtures
    """
    import requests
    
    leagues = leagues or DEFAULT_LEAGUES
    
    all_fixtures = []
    
    for league_code in leagues:
        league_id = LEAGUE_IDS.get(league_code)
        if not league_id:
            continue
        
        try:
            response = requests.get(
                "https://v3.football.api-sports.io/fixtures",
                headers={"x-apisports-key": api_key},
                params={
                    "league": league_id,
                    "season": 2024,
                    "next": 50  # Get next 50 fixtures
                },
                timeout=30
            )
            
            data = response.json()
            
            for fixture in data.get("response", []):
                fixture_date = datetime.fromisoformat(
                    fixture["fixture"]["date"].replace("Z", "+00:00")
                )
                
                # Only include fixtures within our window
                if fixture_date <= datetime.now(fixture_date.tzinfo) + timedelta(days=days):
                    all_fixtures.append({
                        "fixture_id": fixture["fixture"]["id"],
                        "date": fixture_date,
                        "league_code": league_code,
                        "league_name": fixture["league"]["name"],
                        "round": fixture["league"].get("round", ""),
                        "home_team": fixture["teams"]["home"]["name"],
                        "home_team_id": fixture["teams"]["home"]["id"],
                        "away_team": fixture["teams"]["away"]["name"],
                        "away_team_id": fixture["teams"]["away"]["id"],
                        "venue": fixture["fixture"].get("venue", {}).get("name", ""),
                    })
        
        except Exception as e:
            logger.warning(f"Error fetching {league_code}: {e}")
    
    df = pd.DataFrame(all_fixtures)
    if not df.empty:
        df = df.sort_values("date")
    
    logger.info(f"Found {len(df)} upcoming fixtures")
    return df


def build_features():
    """Build feature set from database"""
    from features_api import build_features as _build_features
    
    logger.info("Building features...")
    output = _build_features(force=True)
    logger.info(f"Features saved to {output}")
    return output


def generate_predictions(api_key: str, days: int = 7, leagues: list = None) -> pd.DataFrame:
    """
    Generate predictions for upcoming matches
    
    Returns:
        DataFrame with predictions
    """
    from predict_api import MatchPredictor
    
    leagues = leagues or DEFAULT_LEAGUES
    
    logger.info(f"Generating predictions for next {days} days...")
    
    # Initialize predictor
    predictor = MatchPredictor(api_key=api_key)
    
    # Get upcoming fixtures
    upcoming = get_upcoming_fixtures(api_key, days=days, leagues=leagues)
    
    if upcoming.empty:
        logger.warning("No upcoming fixtures found")
        return pd.DataFrame()
    
    # Generate predictions for each fixture
    predictions = []
    
    for _, fixture in upcoming.iterrows():
        try:
            pred = predictor.predict_fixture(
                home=fixture["home_team"],
                away=fixture["away_team"],
                league=fixture["league_code"],
                match_date=fixture["date"]
            )
            
            if "error" not in pred:
                predictions.append({
                    "fixture_id": fixture["fixture_id"],
                    "date": fixture["date"],
                    "kickoff": fixture["date"].strftime("%a %H:%M"),
                    "league": fixture["league_code"],
                    "league_name": fixture["league_name"],
                    "home": fixture["home_team"],
                    "away": fixture["away_team"],
                    # Probabilities
                    "btts_yes": pred.get("btts_yes", 0.5),
                    "btts_no": pred.get("btts_no", 0.5),
                    "over_1_5": pred.get("over_1_5", 0.5),
                    "over_2_5": pred.get("over_2_5", 0.5),
                    "over_3_5": pred.get("over_3_5", 0.5),
                    "under_2_5": pred.get("under_2_5", 0.5),
                    # 1X2
                    "home_win": pred.get("home_win", 0.33),
                    "draw": pred.get("draw", 0.33),
                    "away_win": pred.get("away_win", 0.33),
                    # Expected goals
                    "exp_home": pred.get("expected_home_goals", 1.3),
                    "exp_away": pred.get("expected_away_goals", 1.1),
                    "exp_total": pred.get("expected_total_goals", 2.4),
                })
        
        except Exception as e:
            logger.debug(f"Could not predict {fixture['home_team']} vs {fixture['away_team']}: {e}")
    
    df = pd.DataFrame(predictions)
    
    if not df.empty:
        df = df.sort_values(["date", "league"])
    
    logger.info(f"Generated {len(df)} predictions")
    return df


def filter_value_bets(predictions: pd.DataFrame, 
                      min_prob: float = 0.55,
                      markets: list = None) -> pd.DataFrame:
    """
    Filter predictions to find value bets
    
    Args:
        predictions: Full predictions DataFrame
        min_prob: Minimum probability threshold
        markets: Markets to check (default: BTTS, O2.5)
    
    Returns:
        Filtered DataFrame with best bets
    """
    markets = markets or ["btts_yes", "over_2_5", "under_2_5"]
    
    value_bets = []
    
    for _, row in predictions.iterrows():
        for market in markets:
            if market in row and row[market] >= min_prob:
                # Calculate implied odds needed for value
                implied_odds = 1 / row[market]
                
                value_bets.append({
                    "date": row["date"],
                    "kickoff": row["kickoff"],
                    "league": row["league"],
                    "match": f"{row['home']} vs {row['away']}",
                    "market": market.upper().replace("_", " "),
                    "probability": row[market],
                    "min_odds": round(implied_odds, 2),
                    "confidence": "HIGH" if row[market] >= 0.60 else "MEDIUM",
                    "exp_goals": f"{row['exp_home']:.1f} - {row['exp_away']:.1f}",
                })
    
    df = pd.DataFrame(value_bets)
    
    if not df.empty:
        df = df.sort_values(["date", "probability"], ascending=[True, False])
    
    return df


def export_predictions(predictions: pd.DataFrame, value_bets: pd.DataFrame):
    """Export predictions to files"""
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Full predictions
    full_path = OUTPUT_DIR / f"predictions_{timestamp}.csv"
    predictions.to_csv(full_path, index=False)
    logger.info(f"Full predictions: {full_path}")
    
    # Value bets
    if not value_bets.empty:
        value_path = OUTPUT_DIR / f"value_bets_{timestamp}.csv"
        value_bets.to_csv(value_path, index=False)
        logger.info(f"Value bets: {value_path}")
    
    # Quick summary for console
    return full_path, value_path if not value_bets.empty else None


def print_summary(predictions: pd.DataFrame, value_bets: pd.DataFrame):
    """Print summary to console"""
    print("\n" + "="*70)
    print("WEEKLY PREDICTION SUMMARY")
    print("="*70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Total fixtures: {len(predictions)}")
    print(f"Value bets found: {len(value_bets)}")
    
    if not predictions.empty:
        print(f"\nLeagues covered:")
        for league in predictions["league"].unique():
            count = len(predictions[predictions["league"] == league])
            print(f"  {league}: {count} matches")
    
    if not value_bets.empty:
        print("\n" + "-"*70)
        print("TOP VALUE BETS")
        print("-"*70)
        
        # Group by date
        for date in value_bets["date"].unique():
            day_bets = value_bets[value_bets["date"] == date]
            print(f"\n{pd.to_datetime(date).strftime('%A %d %B')}:")
            
            for _, bet in day_bets.head(10).iterrows():
                conf_emoji = "🔥" if bet["confidence"] == "HIGH" else "✓"
                print(f"  {conf_emoji} {bet['match']}")
                print(f"      {bet['market']}: {bet['probability']:.0%} (min odds {bet['min_odds']})")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Weekly Prediction Runner")
    parser.add_argument("--api-key", default=os.environ.get("API_FOOTBALL_KEY", DEFAULT_API_KEY),
                        help="API-Football key")
    parser.add_argument("--days", type=int, default=7, help="Days ahead to predict")
    parser.add_argument("--leagues", nargs="+", help="Leagues to include")
    parser.add_argument("--min-prob", type=float, default=0.55, help="Minimum probability for value bets")
    parser.add_argument("--update-days", type=int, default=14, help="Days of history to update")
    parser.add_argument("--skip-update", action="store_true", help="Skip data update")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature rebuild")
    parser.add_argument("--dry-run", action="store_true", help="Check API only, don't run")
    parser.add_argument("--output", type=Path, help="Output directory")
    
    args = parser.parse_args()
    
    if args.output:
        global OUTPUT_DIR
        OUTPUT_DIR = args.output
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    leagues = args.leagues or DEFAULT_LEAGUES
    
    print("\n" + "="*70)
    print("WEEKLY PREDICTION RUNNER")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Leagues: {leagues}")
    print(f"Days ahead: {args.days}")
    
    # Check API status
    print("\n[1/5] Checking API status...")
    try:
        status = check_api_status(args.api_key)
        print(f"  Requests today: {status['current']} / {status['limit']}")
        print(f"  Remaining: {status['remaining']}")
        
        if status["remaining"] < 100:
            print("  ⚠️ Warning: Low API requests remaining!")
    except Exception as e:
        print(f"  Error checking status: {e}")
        return 1
    
    if args.dry_run:
        print("\n[DRY RUN] Stopping here.")
        return 0
    
    # Update recent data
    if not args.skip_update:
        print(f"\n[2/5] Updating recent data (last {args.update_days} days)...")
        try:
            update_recent_data(args.api_key, days=args.update_days, leagues=leagues)
        except Exception as e:
            logger.error(f"Data update failed: {e}")
            print(f"  ⚠️ Data update failed, continuing with existing data...")
    else:
        print("\n[2/5] Skipping data update")
    
    # Build features
    if not args.skip_features:
        print("\n[3/5] Building features...")
        try:
            build_features()
        except Exception as e:
            logger.error(f"Feature build failed: {e}")
            print(f"  ⚠️ Feature build failed: {e}")
    else:
        print("\n[3/5] Skipping feature rebuild")
    
    # Generate predictions
    print(f"\n[4/5] Generating predictions...")
    try:
        predictions = generate_predictions(args.api_key, days=args.days, leagues=leagues)
    except Exception as e:
        logger.error(f"Prediction generation failed: {e}")
        print(f"  ❌ Prediction generation failed: {e}")
        return 1
    
    if predictions.empty:
        print("  No predictions generated. Check logs.")
        return 1
    
    # Find value bets
    print(f"\n[5/5] Finding value bets (min {args.min_prob:.0%})...")
    value_bets = filter_value_bets(predictions, min_prob=args.min_prob)
    
    # Export
    export_predictions(predictions, value_bets)
    
    # Print summary
    print_summary(predictions, value_bets)
    
    print(f"\n✅ Complete! Files saved to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
