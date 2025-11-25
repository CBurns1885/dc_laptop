#!/usr/bin/env python3
"""
cup_fixtures_addon_FIXED_API.py

FIXED: API-Football requires 'season' parameter for cup competitions.
This version includes the current season in all API calls.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import requests
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

API_KEY = os.environ.get("API_FOOTBALL_KEY", "0f17fdba78d15a625710f7244a1cc770")
BASE_URL = "https://v3.football.api-sports.io"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Get current season (e.g., 2024 for 2024/25 season)
def get_current_season() -> int:
    """Get current football season year (starts in August)"""
    now = datetime.now()
    # If before August, previous year's season
    # If August or later, current year's season
    return now.year if now.month >= 8 else now.year - 1

CURRENT_SEASON = get_current_season()

# ============================================================================
# TEAM TO DOMESTIC LEAGUE MAPPING
# ============================================================================

TEAM_DOMESTIC_LEAGUE = {
    # ENGLAND (E0) - Premier League
    "Arsenal": "E0", "Aston Villa": "E0", "Bournemouth": "E0",
    "Brentford": "E0", "Brighton": "E0", "Brighton & Hove Albion": "E0",
    "Chelsea": "E0", "Crystal Palace": "E0", "Everton": "E0",
    "Fulham": "E0", "Ipswich": "E0", "Leicester": "E0",
    "Leicester City": "E0", "Liverpool": "E0", "Man City": "E0",
    "Manchester City": "E0", "Man United": "E0", "Manchester United": "E0",
    "Newcastle": "E0", "Newcastle United": "E0", "Nottingham Forest": "E0",
    "Nott'm Forest": "E0", "Southampton": "E0", "Tottenham": "E0",
    "Tottenham Hotspur": "E0", "West Ham": "E0", "West Ham United": "E0",
    "Wolves": "E0", "Wolverhampton Wanderers": "E0",
    
    # ENGLAND (E1) - Championship
    "Blackburn": "E1", "Bristol City": "E1", "Burnley": "E1",
    "Cardiff": "E1", "Coventry": "E1", "Derby": "E1",
    "Hull": "E1", "Leeds": "E1", "Leeds United": "E1",
    "Luton": "E1", "Middlesbrough": "E1", "Millwall": "E1",
    "Norwich": "E1", "Oxford United": "E1", "Plymouth": "E1",
    "Portsmouth": "E1", "Preston": "E1", "QPR": "E1",
    "Queens Park Rangers": "E1", "Sheffield United": "E1",
    "Sheffield Weds": "E1", "Sheffield Wednesday": "E1",
    "Stoke": "E1", "Sunderland": "E1", "Swansea": "E1",
    "Watford": "E1", "West Brom": "E1", "West Bromwich Albion": "E1",
    
    # SPAIN (SP1) - La Liga
    "Alaves": "SP1", "Deportivo Alaves": "SP1", "Ath Bilbao": "SP1",
    "Athletic Club": "SP1", "Ath Madrid": "SP1", "Atletico Madrid": "SP1",
    "Barcelona": "SP1", "Betis": "SP1", "Real Betis": "SP1",
    "Celta": "SP1", "Celta Vigo": "SP1", "Espanyol": "SP1",
    "Getafe": "SP1", "Girona": "SP1", "Las Palmas": "SP1",
    "Leganes": "SP1", "Mallorca": "SP1", "Osasuna": "SP1",
    "Rayo Vallecano": "SP1", "Vallecano": "SP1", "Real Madrid": "SP1",
    "Real Sociedad": "SP1", "Sevilla": "SP1", "Valencia": "SP1",
    "Valladolid": "SP1", "Villarreal": "SP1",
    
    # GERMANY (D1) - Bundesliga
    "Augsburg": "D1", "Bayern Munich": "D1", "Bayer Leverkusen": "D1",
    "Leverkusen": "D1", "Bochum": "D1", "Dortmund": "D1",
    "Borussia Dortmund": "D1", "Eintracht Frankfurt": "D1",
    "Frankfurt": "D1", "Freiburg": "D1", "Heidenheim": "D1",
    "Hoffenheim": "D1", "TSG Hoffenheim": "D1", "Holstein Kiel": "D1",
    "FC Koln": "D1", "Cologne": "D1", "Koln": "D1",
    "RB Leipzig": "D1", "Leipzig": "D1", "Mainz": "D1",
    "M'gladbach": "D1", "Borussia Monchengladbach": "D1",
    "St Pauli": "D1", "Stuttgart": "D1", "Union Berlin": "D1",
    "Werder Bremen": "D1", "Bremen": "D1", "Wolfsburg": "D1",
    
    # ITALY (I1) - Serie A
    "Atalanta": "I1", "Bologna": "I1", "Cagliari": "I1",
    "Como": "I1", "Empoli": "I1", "Fiorentina": "I1",
    "Genoa": "I1", "Inter": "I1", "Inter Milan": "I1",
    "Juventus": "I1", "Lazio": "I1", "Lecce": "I1",
    "Milan": "I1", "AC Milan": "I1", "Monza": "I1",
    "Napoli": "I1", "Parma": "I1", "Roma": "I1",
    "AS Roma": "I1", "Torino": "I1", "Udinese": "I1",
    "Venezia": "I1", "Verona": "I1", "Hellas Verona": "I1",
    
    # FRANCE (F1) - Ligue 1
    "Angers": "F1", "Auxerre": "F1", "Brest": "F1",
    "Le Havre": "F1", "Lens": "F1", "Lille": "F1",
    "Lyon": "F1", "Olympique Lyonnais": "F1", "Marseille": "F1",
    "Olympique Marseille": "F1", "Monaco": "F1", "AS Monaco": "F1",
    "Montpellier": "F1", "Nantes": "F1", "Nice": "F1",
    "Paris SG": "F1", "Paris Saint Germain": "F1",
    "Paris Saint-Germain": "F1", "PSG": "F1",
    "Reims": "F1", "Rennes": "F1", "St Etienne": "F1",
    "Saint-Etienne": "F1", "Strasbourg": "F1", "Toulouse": "F1",
    
    # PORTUGAL (P1) - Primeira Liga
    "AVS": "P1", "Arouca": "P1", "Benfica": "P1",
    "Boavista": "P1", "Braga": "P1", "Casa Pia": "P1",
    "Estoril": "P1", "Estrela": "P1", "Farense": "P1",
    "Famalicao": "P1", "Gil Vicente": "P1", "Moreirense": "P1",
    "Nacional": "P1", "Porto": "P1", "Rio Ave": "P1",
    "Santa Clara": "P1", "Sporting CP": "P1", "Vitoria SC": "P1",
    
    # NETHERLANDS (N1) - Eredivisie
    "Ajax": "N1", "AZ Alkmaar": "N1", "Alkmaar": "N1",
    "Feyenoord": "N1", "Fortuna Sittard": "N1", "Go Ahead Eagles": "N1",
    "Groningen": "N1", "Heerenveen": "N1", "Heracles": "N1",
    "NEC Nijmegen": "N1", "NAC Breda": "N1", "PEC Zwolle": "N1",
    "PSV": "N1", "PSV Eindhoven": "N1", "RKC Waalwijk": "N1",
    "Sparta Rotterdam": "N1", "Twente": "N1", "FC Twente": "N1",
    "Utrecht": "N1", "Willem II": "N1",
    
    # BELGIUM (B1) - Pro League
    "Anderlecht": "B1", "Antwerp": "B1", "Royal Antwerp": "B1",
    "Beerschot VA": "B1", "Cercle Brugge": "B1", "Charleroi": "B1",
    "Club Brugge": "B1", "Club Brugge KV": "B1", "Dender": "B1",
    "Genk": "B1", "Gent": "B1", "KV Kortrijk": "B1",
    "Kortrijk": "B1", "Leuven": "B1", "Oud-Heverlee Leuven": "B1",
    "Mechelen": "B1", "KV Mechelen": "B1", "RAAL La Louviere": "B1",
    "St Truiden": "B1", "Standard": "B1", "Standard Liege": "B1",
    "Union SG": "B1", "Westerlo": "B1",
    
    # SCOTLAND (SC0) - Premiership
    "Aberdeen": "SC0", "Celtic": "SC0", "Dundee": "SC0",
    "Dundee United": "SC0", "Hearts": "SC0", "Hibernian": "SC0",
    "Kilmarnock": "SC0", "Motherwell": "SC0", "Rangers": "SC0",
    "Ross County": "SC0", "St Johnstone": "SC0", "St Mirren": "SC0",
    
    # TURKEY (T1) - Super Lig
    "Adana Demirspor": "T1", "Alanyaspor": "T1", "Antalyaspor": "T1",
    "Besiktas": "T1", "Bodrum FK": "T1", "Eyupspor": "T1",
    "Fenerbahce": "T1", "Galatasaray": "T1", "Gaziantep FK": "T1",
    "Goztepe": "T1", "Hatayspor": "T1", "Kasimpasa": "T1",
    "Kayserispor": "T1", "Konyaspor": "T1", "Rizespor": "T1",
    "Samsunspor": "T1", "Sivasspor": "T1", "Trabzonspor": "T1",
}

# ============================================================================
# TEAM NAME STANDARDIZATION
# ============================================================================

TEAM_NAME_MAP = {
    "Manchester United": "Man United", "Manchester City": "Man City",
    "Newcastle United": "Newcastle", "Tottenham Hotspur": "Tottenham",
    "Brighton & Hove Albion": "Brighton", "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves", "Leicester City": "Leicester",
    "Leeds United": "Leeds", "Nottingham Forest": "Nott'm Forest",
    "Sheffield Wednesday": "Sheffield Weds", "Queens Park Rangers": "QPR",
    "West Bromwich Albion": "West Brom",
    
    "Athletic Club": "Ath Bilbao", "Atletico Madrid": "Ath Madrid",
    "Real Betis": "Betis", "Deportivo Alaves": "Alaves",
    "Rayo Vallecano": "Vallecano", "Celta Vigo": "Celta",
    
    "Borussia Dortmund": "Dortmund", "Borussia Monchengladbach": "M'gladbach",
    "Bayer Leverkusen": "Leverkusen", "TSG Hoffenheim": "Hoffenheim",
    "FC Koln": "Koln", "Eintracht Frankfurt": "Frankfurt",
    "Werder Bremen": "Bremen",
    
    "Inter Milan": "Inter", "AC Milan": "Milan",
    "AS Roma": "Roma", "Hellas Verona": "Verona",
    
    "Paris Saint Germain": "Paris SG", "Paris Saint-Germain": "Paris SG",
    "PSG": "Paris SG", "Olympique Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon", "AS Monaco": "Monaco",
    "Saint-Etienne": "St Etienne",
    
    "Club Brugge KV": "Club Brugge", "Royal Antwerp": "Antwerp",
    "Standard Liege": "Standard", "Oud-Heverlee Leuven": "Leuven",
    "KV Kortrijk": "Kortrijk", "KV Mechelen": "Mechelen",
    
    "PSV Eindhoven": "PSV", "FC Twente": "Twente", "AZ Alkmaar": "Alkmaar",
}

# ============================================================================
# CUP COMPETITIONS
# ============================================================================

CUP_COMPETITIONS = {
    2: "Champions League", 3: "Europa League", 848: "Conference League",
    45: "FA Cup", 48: "League Cup",
    143: "Copa del Rey", 81: "DFB Pokal", 137: "Coppa Italia",
    66: "Coupe de France", 94: "Taça de Portugal", 36: "KNVB Beker",
    38: "Belgian Cup", 51: "Scottish Cup", 52: "Scottish League Cup",
}

# ============================================================================
# FUNCTIONS
# ============================================================================

def fetch_fixtures(league_id: int, season: int, days_ahead: int = 14) -> list:
    """Fetch fixtures for a specific league and season"""
    today = datetime.now()
    date_from = today.strftime("%Y-%m-%d")
    date_to = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    
    url = f"{BASE_URL}/fixtures"
    headers = {"x-apisports-key": API_KEY}
    params = {
        "league": league_id,
        "season": season,  # CRITICAL: Season is required!
        "from": date_from,
        "to": date_to,
        "timezone": "UTC"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("errors"):
            print(f"   API errors: {data['errors']}")
            return []
        
        return data.get("response", [])
    except Exception as e:
        print(f"   Error: {e}")
        return []

def get_team_league(team_name: str) -> str:
    """Get domestic league code for a team"""
    standardized = TEAM_NAME_MAP.get(team_name, team_name)
    league = TEAM_DOMESTIC_LEAGUE.get(standardized)
    if not league:
        league = TEAM_DOMESTIC_LEAGUE.get(team_name)
    return league or "E0"

def normalize_team_name(name: str) -> str:
    """Standardize team name"""
    return TEAM_NAME_MAP.get(name, name)

def convert_fixture(fixture_data: dict) -> list:
    """Convert API fixture to CSV rows (1 or 2 rows for cross-league)"""
    fixture = fixture_data.get("fixture", {})
    teams = fixture_data.get("teams", {})
    
    date_str = fixture.get("date", "")
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        match_date = dt.strftime("%d/%m/%Y")
        match_time = dt.strftime("%H:%M")
    except:
        match_date = ""
        match_time = ""
    
    home_name_raw = teams.get("home", {}).get("name", "")
    away_name_raw = teams.get("away", {}).get("name", "")
    
    home_team = normalize_team_name(home_name_raw)
    away_team = normalize_team_name(away_name_raw)
    
    home_league = get_team_league(home_team)
    away_league = get_team_league(away_team)
    
    rows = []
    
    # Always create row with home team's league
    rows.append({
        "League": home_league,
        "Date": match_date,
        "Time": match_time,
        "HomeTeam": home_team,
        "AwayTeam": away_team,
        "Referee": "",
    })
    
    # If cross-league, create second row
    if home_league != away_league:
        rows.append({
            "League": away_league,
            "Date": match_date,
            "Time": match_time,
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "Referee": "",
        })
    
    return rows

def download_and_merge_cups(existing_fixtures_path: Path, days_ahead: int = 14) -> Path:
    """Download cups and merge with existing fixtures"""
    
    print("="*70)
    print(" CUP FIXTURES ADD-ON (DUAL-LEAGUE + API SEASON FIX)")
    print("="*70)
    print(f"Current season: {CURRENT_SEASON}")
    print(f"Fetching cup fixtures for next {days_ahead} days...")
    
    if not existing_fixtures_path.exists():
        print(f" File not found: {existing_fixtures_path}")
        return existing_fixtures_path
    
    existing_df = pd.read_csv(existing_fixtures_path)
    print(f" Loaded {len(existing_df)} existing fixtures")
    
    print("\n Downloading cup fixtures...")
    all_cup_fixtures = []
    
    for league_id, comp_name in CUP_COMPETITIONS.items():
        print(f"   {comp_name} (ID {league_id})...", end=" ", flush=True)
        
        fixtures = fetch_fixtures(league_id, CURRENT_SEASON, days_ahead)
        
        if fixtures:
            print(f" {len(fixtures)} matches")
            for fixture_data in fixtures:
                rows = convert_fixture(fixture_data)
                all_cup_fixtures.extend(rows)
        else:
            print(" None")
        
        # Rate limiting: wait 6 seconds between requests (10 per minute max)
        time.sleep(6)
    
    if not all_cup_fixtures:
        print("\n No cup fixtures found")
        return existing_fixtures_path
    
    cup_df = pd.DataFrame(all_cup_fixtures)
    print(f"\n Created {len(cup_df)} prediction rows from cup fixtures")
    
    # Match columns
    for col in existing_df.columns:
        if col not in cup_df.columns:
            cup_df[col] = ""
    cup_df = cup_df[existing_df.columns]
    
    # Merge
    merged_df = pd.concat([existing_df, cup_df], ignore_index=True)
    before = len(merged_df)
    merged_df = merged_df.drop_duplicates(keep="first")
    after = len(merged_df)
    
    if before > after:
        print(f"   Removed {before - after} exact duplicates")
    
    # Sort
    merged_df["_sort"] = pd.to_datetime(merged_df["Date"], format="%d/%m/%Y", errors="coerce")
    merged_df = merged_df.sort_values(["_sort", "League"]).drop(columns=["_sort"])
    
    # Save
    merged_path = OUTPUT_DIR / "upcoming_fixtures_with_cups.csv"
    merged_df.to_csv(merged_path, index=False)
    
    print(f"\n MERGED: {merged_path}")
    print(f"   Total prediction rows: {len(merged_df)}")
    print(f"   Added: {len(merged_df) - len(existing_df)} cup prediction rows")
    
    print("\n League breakdown:")
    for league in sorted(merged_df["League"].unique()):
        count = len(merged_df[merged_df["League"] == league])
        print(f"   {league}: {count} prediction rows")
    
    return merged_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="upcoming_fixtures.csv")
    parser.add_argument("--days", type=int, default=14)
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        input_path = OUTPUT_DIR / args.input
    
    merged_path = download_and_merge_cups(input_path, args.days)
    print(f"\n Use this file: {merged_path}")