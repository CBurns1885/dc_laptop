#!/usr/bin/env python3
"""
download_all_fixtures.py - BEST OF BOTH WORLDS

Sources:
✅ football-data.co.uk scrape → Domestic leagues (FREE, perfect match)
✅ API-Football → European cups only (FREE 100/day)

Result: ~200 fixtures, no formatting issues, minimal API usage
"""

import os
import sys
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from typing import Optional, List, Dict
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

# football-data.co.uk
FOOTBALL_DATA_URL = "https://www.football-data.co.uk/matches.php"

# API-Football (for European cups only)
API_FOOTBALL_KEY = os.environ.get("API_FOOTBALL_KEY", "")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

EUROPEAN_COMPETITIONS = {
    'CL': 2,       # Champions League
    'EL': 3,       # Europa League
    'ECL': 848,    # Conference League
}

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Team name normalization for API-Football responses
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
    'Leicester City': 'Leicester',
    'Bayern Munich': 'Bayern Munich',
    'Borussia Dortmund': 'Dortmund',
    'RB Leipzig': 'RB Leipzig',
    'Bayer Leverkusen': 'Leverkusen',
    'Eintracht Frankfurt': 'Ein Frankfurt',
    'VfL Wolfsburg': 'Wolfsburg',
    'Borussia Monchengladbach': "M'gladbach",
    'Real Madrid': 'Real Madrid',
    'FC Barcelona': 'Barcelona',
    'Atletico Madrid': 'Ath Madrid',
    'Athletic Bilbao': 'Ath Bilbao',
    'Real Sociedad': 'Sociedad',
    'Real Betis': 'Betis',
    'Inter Milan': 'Inter',
    'AC Milan': 'Milan',
    'Juventus': 'Juventus',
    'AS Roma': 'Roma',
    'Lazio': 'Lazio',
    'Napoli': 'Napoli',
    'Atalanta': 'Atalanta',
    'Paris Saint Germain': 'Paris SG',
    'Olympique Marseille': 'Marseille',
    'Olympique Lyon': 'Lyon',
    'AS Monaco': 'Monaco',
}

# ============================================================================
# PART 1: SCRAPE DOMESTIC FROM FOOTBALL-DATA.CO.UK
# ============================================================================

def scrape_domestic_fixtures() -> pd.DataFrame:
    """Scrape domestic league fixtures from football-data.co.uk"""
    
    print("\n📡 SCRAPING DOMESTIC FIXTURES")
    print("="*60)
    print(f"Source: {FOOTBALL_DATA_URL}")
    
    try:
        # Fetch page
        response = requests.get(FOOTBALL_DATA_URL, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Parse fixtures from tables
        fixtures = []
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            
            for row in rows:
                cells = row.find_all('td')
                
                if len(cells) >= 4:
                    try:
                        text_cells = [cell.get_text(strip=True) for cell in cells]
                        
                        # Look for date (DD/MM/YY or DD/MM/YYYY)
                        date_idx = None
                        for i, text in enumerate(text_cells):
                            if re.match(r'\d{2}/\d{2}/\d{2,4}', text):
                                date_idx = i
                                break
                        
                        if date_idx is not None:
                            date_str = text_cells[date_idx]
                            
                            # Find league code (E0, D1, etc.)
                            league = None
                            for text in text_cells:
                                if re.match(r'^[A-Z]{1,3}\d{1,2}$', text):
                                    league = text
                                    break
                            
                            # Get team names (remaining non-date, non-league cells)
                            remaining = [c for c in text_cells if c != date_str and c != league and c]
                            
                            if len(remaining) >= 2 and league:
                                home_team = remaining[0]
                                away_team = remaining[1]
                                
                                # Parse date
                                try:
                                    if len(date_str) == 10:
                                        date_obj = datetime.strptime(date_str, '%d/%m/%Y')
                                    else:
                                        date_obj = datetime.strptime(date_str, '%d/%m/%y')
                                    
                                    fixtures.append({
                                        'Date': date_obj.strftime('%Y-%m-%d'),
                                        'League': league,
                                        'HomeTeam': home_team,
                                        'AwayTeam': away_team,
                                        'Source': 'football-data.co.uk'
                                    })
                                except:
                                    continue
                    except:
                        continue
        
        df = pd.DataFrame(fixtures)
        
        if not df.empty:
            df = df.drop_duplicates(subset=['Date', 'League', 'HomeTeam', 'AwayTeam'])
            df = df.sort_values('Date').reset_index(drop=True)
            print(f"✅ Scraped {len(df)} domestic fixtures")
            
            # Show summary
            for league, count in df['League'].value_counts().items():
                print(f"   • {league}: {count} matches")
        else:
            print("⚠️ No fixtures found - website structure may have changed")
        
        return df
        
    except Exception as e:
        print(f"❌ Scraping failed: {e}")
        return pd.DataFrame(columns=['Date', 'League', 'HomeTeam', 'AwayTeam'])

# ============================================================================
# PART 2: API-FOOTBALL FOR EUROPEAN CUPS ONLY
# ============================================================================

def download_european_cups(days_ahead: int = 14) -> pd.DataFrame:
    """Download European cup fixtures from API-Football"""
    
    print("\n📡 DOWNLOADING EUROPEAN CUPS")
    print("="*60)
    
    if not API_FOOTBALL_KEY:
        print("⚠️ API-Football key not set - skipping European cups")
        print("   (This is optional - you can add it later)")
        return pd.DataFrame(columns=['Date', 'League', 'HomeTeam', 'AwayTeam'])
    
    current_season = datetime.now().year if datetime.now().month >= 7 else datetime.now().year - 1
    date_from = datetime.now()
    date_to = datetime.now() + timedelta(days=days_ahead)
    
    session = requests.Session()
    session.headers.update({'x-apisports-key': API_FOOTBALL_KEY})
    
    all_fixtures = []
    
    for comp_name, league_id in EUROPEAN_COMPETITIONS.items():
        print(f"   🏆 {comp_name} (ID: {league_id})", end='')
        
        try:
            response = session.get(
                f"{API_FOOTBALL_BASE}/fixtures",
                params={'league': league_id, 'season': current_season, 'status': 'NS'},
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            fixtures = data.get('response', [])
            
            if fixtures:
                for match in fixtures:
                    try:
                        fixture = match['fixture']
                        teams = match['teams']
                        
                        match_date = datetime.fromisoformat(fixture['date'].replace('Z', '+00:00'))
                        
                        if date_from <= match_date <= date_to:
                            home_team = TEAM_NAME_MAPPING.get(teams['home']['name'], teams['home']['name'])
                            away_team = TEAM_NAME_MAPPING.get(teams['away']['name'], teams['away']['name'])
                            
                            all_fixtures.append({
                                'Date': match_date.strftime('%Y-%m-%d'),
                                'League': comp_name,
                                'HomeTeam': home_team,
                                'AwayTeam': away_team,
                                'Source': 'API-Football'
                            })
                    except:
                        continue
                
                count = len([f for f in all_fixtures if f['League'] == comp_name])
                print(f" → {count} matches")
            else:
                print(" → no matches")
                
        except Exception as e:
            print(f" → error: {e}")
    
    if all_fixtures:
        df = pd.DataFrame(all_fixtures)
        df = df.drop_duplicates(subset=['Date', 'League', 'HomeTeam', 'AwayTeam'])
        print(f"✅ Downloaded {len(df)} European cup fixtures")
        return df
    
    return pd.DataFrame(columns=['Date', 'League', 'HomeTeam', 'AwayTeam'])

# ============================================================================
# PART 3: COMBINE BOTH SOURCES
# ============================================================================

def download_all_fixtures(days_ahead: int = 14) -> Optional[Path]:
    """
    Download ALL fixtures from both sources
    
    Returns:
        Path to combined CSV file
    """
    
    print("\n⚽ COMPLETE FIXTURE DOWNLOAD SYSTEM")
    print("="*60)
    print("📋 Strategy:")
    print("   1. Scrape domestic leagues from football-data.co.uk (FREE)")
    print("   2. Download European cups from API-Football (100/day limit)")
    print("="*60)
    
    all_fixtures = []
    
    # Part 1: Scrape domestic leagues
    domestic_df = scrape_domestic_fixtures()
    if not domestic_df.empty:
        all_fixtures.append(domestic_df)
    
    # Part 2: Download European cups
    european_df = download_european_cups(days_ahead)
    if not european_df.empty:
        all_fixtures.append(european_df)
    
    # Combine
    if not all_fixtures:
        print("\n❌ No fixtures downloaded from any source!")
        return None
    
    combined = pd.concat(all_fixtures, ignore_index=True)
    combined = combined.drop_duplicates(subset=['Date', 'League', 'HomeTeam', 'AwayTeam'])
    combined = combined.sort_values(['Date', 'League']).reset_index(drop=True)
    
    # Drop Source column (not needed in final output)
    combined = combined[['Date', 'League', 'HomeTeam', 'AwayTeam']]
    
    # Print final summary
    print("\n" + "="*60)
    print("✅ DOWNLOAD COMPLETE")
    print("="*60)
    print(f"📊 Total fixtures: {len(combined)}")
    print(f"📅 Date range: {combined['Date'].min()} to {combined['Date'].max()}")
    
    print(f"\n🏟️ Fixtures by league:")
    for league, count in combined['League'].value_counts().items():
        print(f"   • {league}: {count} matches")
    
    print(f"\n📅 Fixtures by date:")
    for date, count in combined.groupby('Date').size().items():
        print(f"   • {date}: {count} matches")
    
    # Save files
    csv_path = OUTPUT_DIR / "upcoming_fixtures.csv"
    xlsx_path = OUTPUT_DIR / "upcoming_fixtures.xlsx"
    
    combined.to_csv(csv_path, index=False)
    print(f"\n💾 Saved to: {csv_path}")
    
    combined.to_excel(xlsx_path, index=False)
    print(f"💾 Saved to: {xlsx_path}")
    
    print("="*60)
    print("\n🎉 SUCCESS! Ready for predictions")
    print("💡 Next step: python RUN_WEEKLY.py")
    
    return csv_path

# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download all upcoming football fixtures")
    parser.add_argument('--days', type=int, default=14, help='Days ahead (default: 14)')
    args = parser.parse_args()
    
    result = download_all_fixtures(args.days)
    
    if not result:
        sys.exit(1)
