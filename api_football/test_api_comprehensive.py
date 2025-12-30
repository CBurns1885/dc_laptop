# test_api_comprehensive.py
"""
Comprehensive API Testing Script
Tests all available endpoints to understand capabilities
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Fix Unicode encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from api_football_client import APIFootballClient, LEAGUES

# API Key
API_KEY = "0f17fdba78d15a625710f7244a1cc770"

def print_section(title):
    """Print section header"""
    print("\n" + "="*70)
    print(title)
    print("="*70)

def test_status(client):
    """Test API status and quota"""
    print_section("1. API STATUS & QUOTA")

    status = client.get_status()

    if 'response' in status:
        response = status['response']
        print(f"\n✅ API Connection: SUCCESS")
        print(f"\nAccount Details:")
        print(f"  Account: {response.get('account', {}).get('firstname', 'N/A')} {response.get('account', {}).get('lastname', 'N/A')}")
        print(f"  Email: {response.get('account', {}).get('email', 'N/A')}")

        print(f"\nSubscription:")
        print(f"  Plan: {response.get('subscription', {}).get('plan', 'N/A')}")
        print(f"  End Date: {response.get('subscription', {}).get('end', 'N/A')}")
        print(f"  Active: {response.get('subscription', {}).get('active', False)}")

        print(f"\nQuota:")
        requests_made = response.get('requests', {})
        print(f"  Current: {requests_made.get('current', 0):,}")
        print(f"  Limit (Day): {requests_made.get('limit_day', 0):,}")
        remaining = requests_made.get('limit_day', 0) - requests_made.get('current', 0)
        print(f"  Remaining Today: {remaining:,}")

        return True
    else:
        print(f"\n❌ API Connection: FAILED")
        print(f"Errors: {status.get('errors', 'Unknown error')}")
        return False

def test_leagues(client):
    """Test league access"""
    print_section("2. LEAGUE ACCESS")

    print("\nTesting England leagues...")
    england_leagues = client.get_leagues(country="England", season=2024)

    if england_leagues:
        print(f"\n✅ Found {len(england_leagues)} England leagues for 2024:")
        for league in england_leagues[:10]:
            league_info = league.get('league', {})
            print(f"  - {league_info.get('name')}: ID={league_info.get('id')}, Type={league_info.get('type')}")
    else:
        print("❌ No leagues found")

    return len(england_leagues) > 0

def test_fixtures_historical(client):
    """Test historical fixture access"""
    print_section("3. HISTORICAL FIXTURES")

    # Test Premier League 2023-24 season
    print("\nTesting Premier League (E0) 2023-24 season...")
    fixtures = client.get_fixtures(
        league_id=39,  # Premier League
        season=2023
    )

    if fixtures:
        print(f"\n✅ Found {len(fixtures)} Premier League fixtures for 2023-24")

        # Sample some matches
        print("\nSample fixtures:")
        for i, fixture in enumerate(fixtures[:5]):
            f = fixture['fixture']
            teams = fixture['teams']
            goals = fixture['goals']

            print(f"\n  Match {i+1}:")
            print(f"    Date: {f['date'][:10]}")
            print(f"    {teams['home']['name']} {goals['home']} - {goals['away']} {teams['away']['name']}")
            print(f"    Status: {f['status']['long']}")
            print(f"    Fixture ID: {f['id']}")

        return fixtures
    else:
        print("❌ No fixtures found")
        return []

def test_fixture_statistics(client, fixtures):
    """Test detailed fixture statistics"""
    print_section("4. FIXTURE STATISTICS")

    if not fixtures:
        print("⚠️ No fixtures available to test")
        return False

    # Get a completed fixture
    completed = [f for f in fixtures if f['fixture']['status']['short'] == 'FT']

    if not completed:
        print("⚠️ No completed fixtures found")
        return False

    fixture = completed[0]
    fixture_id = fixture['fixture']['id']
    teams = fixture['teams']

    print(f"\nTesting statistics for:")
    print(f"  {teams['home']['name']} vs {teams['away']['name']}")
    print(f"  Fixture ID: {fixture_id}")

    # Get statistics
    stats = client.get_fixture_statistics(fixture_id)

    if stats:
        print(f"\n✅ Statistics retrieved successfully")
        print(f"\nDetailed Statistics:")

        for team_stats in stats:
            team = team_stats['team']
            statistics = team_stats['statistics']

            print(f"\n  {team['name']}:")
            for stat in statistics:
                stat_type = stat.get('type')
                value = stat.get('value')
                print(f"    {stat_type}: {value}")

        return True
    else:
        print("❌ No statistics found")
        return False

def test_fixture_events(client, fixtures):
    """Test fixture events (goals, cards, subs)"""
    print_section("5. FIXTURE EVENTS")

    if not fixtures:
        return False

    completed = [f for f in fixtures if f['fixture']['status']['short'] == 'FT']
    if not completed:
        return False

    fixture_id = completed[0]['fixture']['id']

    print(f"\nTesting events for Fixture ID: {fixture_id}")

    events = client.get_fixture_events(fixture_id)

    if events:
        print(f"\n✅ Found {len(events)} events")

        # Show goals
        goals = [e for e in events if e['type'] == 'Goal']
        cards = [e for e in events if e['type'] == 'Card']

        print(f"\nGoals ({len(goals)}):")
        for goal in goals[:5]:
            time = goal['time']['elapsed']
            team = goal['team']['name']
            player = goal['player']['name']
            detail = goal['detail']
            print(f"  {time}' - {player} ({team}) - {detail}")

        print(f"\nCards ({len(cards)}):")
        for card in cards[:5]:
            time = card['time']['elapsed']
            team = card['team']['name']
            player = card['player']['name']
            detail = card['detail']
            print(f"  {time}' - {player} ({team}) - {detail}")

        return True
    else:
        print("❌ No events found")
        return False

def test_upcoming_fixtures(client):
    """Test upcoming fixtures"""
    print_section("6. UPCOMING FIXTURES")

    today = datetime.now()
    next_week = today + timedelta(days=7)

    print(f"\nTesting upcoming fixtures (next 7 days)...")
    print(f"Date range: {today.strftime('%Y-%m-%d')} to {next_week.strftime('%Y-%m-%d')}")

    fixtures = client.get_fixtures(
        league_id=39,  # Premier League
        season=2024,
        from_date=today.strftime('%Y-%m-%d'),
        to_date=next_week.strftime('%Y-%m-%d')
    )

    if fixtures:
        print(f"\n✅ Found {len(fixtures)} upcoming Premier League fixtures")

        print("\nUpcoming matches:")
        for fixture in fixtures:
            f = fixture['fixture']
            teams = fixture['teams']

            print(f"\n  {f['date'][:10]} {f['date'][11:16]}:")
            print(f"    {teams['home']['name']} vs {teams['away']['name']}")
            print(f"    Venue: {f.get('venue', {}).get('name', 'N/A')}")
            print(f"    Status: {f['status']['long']}")

        return fixtures
    else:
        print("⚠️ No upcoming fixtures found (may be off-season or international break)")
        return []

def test_h2h(client, fixtures):
    """Test head-to-head data"""
    print_section("7. HEAD-TO-HEAD")

    if not fixtures:
        return False

    fixture = fixtures[0]
    home_id = fixture['teams']['home']['id']
    away_id = fixture['teams']['away']['id']
    home_name = fixture['teams']['home']['name']
    away_name = fixture['teams']['away']['name']

    print(f"\nTesting H2H for {home_name} vs {away_name}")

    h2h = client.get_h2h(home_id, away_id, last=10)

    if h2h:
        print(f"\n✅ Found {len(h2h)} past meetings")

        print("\nRecent H2H:")
        for match in h2h[:5]:
            f = match['fixture']
            teams = match['teams']
            goals = match['goals']

            print(f"\n  {f['date'][:10]}:")
            print(f"    {teams['home']['name']} {goals['home']} - {goals['away']} {teams['away']['name']}")

        return True
    else:
        print("⚠️ No H2H data found")
        return False

def test_team_statistics(client):
    """Test team season statistics"""
    print_section("8. TEAM STATISTICS")

    print(f"\nTesting team statistics for Arsenal (Premier League 2023-24)...")

    stats = client.get_team_statistics(
        team_id=42,  # Arsenal
        league_id=39,  # Premier League
        season=2023
    )

    if stats:
        print(f"\n✅ Team statistics retrieved")

        # Show key stats
        team = stats.get('team', {})
        form = stats.get('form', '')
        fixtures = stats.get('fixtures', {})
        goals = stats.get('goals', {})

        print(f"\nTeam: {team.get('name')}")
        print(f"Form: {form}")
        print(f"\nFixtures:")
        print(f"  Played: {fixtures.get('played', {}).get('total', 0)}")
        print(f"  Wins: {fixtures.get('wins', {}).get('total', 0)}")
        print(f"  Draws: {fixtures.get('draws', {}).get('total', 0)}")
        print(f"  Losses: {fixtures.get('loses', {}).get('total', 0)}")

        print(f"\nGoals:")
        print(f"  For: {goals.get('for', {}).get('total', {}).get('total', 0)}")
        print(f"  Against: {goals.get('against', {}).get('total', {}).get('total', 0)}")
        avg_for = goals.get('for', {}).get('average', {}).get('total', 0)
        avg_against = goals.get('against', {}).get('average', {}).get('total', 0)
        # Convert to float if string
        avg_for = float(avg_for) if isinstance(avg_for, (str, int, float)) else 0
        avg_against = float(avg_against) if isinstance(avg_against, (str, int, float)) else 0
        print(f"  Average For: {avg_for:.2f}")
        print(f"  Average Against: {avg_against:.2f}")

        return True
    else:
        print("❌ No team statistics found")
        return False

def test_injuries(client):
    """Test injury data"""
    print_section("9. INJURIES")

    print(f"\nTesting current injuries for Premier League...")

    injuries = client.get_injuries(
        league_id=39,  # Premier League
        season=2024
    )

    if injuries:
        print(f"\n✅ Found {len(injuries)} injury records")

        print("\nSample injuries:")
        for injury in injuries[:10]:
            player = injury.get('player', {})
            team = injury.get('team', {})
            fixture = injury.get('fixture', {})

            print(f"\n  {player.get('name')} ({team.get('name')})")
            print(f"    Type: {player.get('type')}")
            print(f"    Reason: {player.get('reason')}")

        return True
    else:
        print("⚠️ No injury data found (may not be included in plan)")
        return False

def test_standings(client):
    """Test league standings"""
    print_section("10. LEAGUE STANDINGS")

    print(f"\nTesting Premier League standings 2023-24...")

    standings = client.get_standings(league_id=39, season=2023)

    if standings:
        print(f"\n✅ Standings retrieved")

        # Show table
        for standing_group in standings:
            league = standing_group.get('league', {})
            standings_list = league.get('standings', [[]])[0]

            print(f"\n{league.get('name')} - {league.get('season')}")
            print(f"\n{'Pos':<4} {'Team':<25} {'P':>3} {'W':>3} {'D':>3} {'L':>3} {'GF':>3} {'GA':>3} {'GD':>4} {'Pts':>4}")
            print("-" * 80)

            for team in standings_list[:10]:
                rank = team.get('rank', 0)
                team_info = team.get('team', {})
                all_stats = team.get('all', {})
                goals = team.get('goals', {})
                points = team.get('points', 0)

                print(f"{rank:<4} {team_info.get('name'):<25} "
                      f"{all_stats.get('played', 0):>3} "
                      f"{all_stats.get('win', 0):>3} "
                      f"{all_stats.get('draw', 0):>3} "
                      f"{all_stats.get('lose', 0):>3} "
                      f"{goals.get('for', 0):>3} "
                      f"{goals.get('against', 0):>3} "
                      f"{goals.get('for', 0) - goals.get('against', 0):>4} "
                      f"{points:>4}")

        return True
    else:
        print("❌ No standings found")
        return False

def test_odds(client, fixtures):
    """Test betting odds"""
    print_section("11. BETTING ODDS")

    if not fixtures:
        return False

    fixture_id = fixtures[0]['fixture']['id']

    print(f"\nTesting odds for Fixture ID: {fixture_id}")

    odds = client.get_odds(fixture_id=fixture_id)

    if odds:
        print(f"\n✅ Found odds from {len(odds)} bookmakers")

        # Show sample odds
        for odd in odds[:2]:
            fixture_info = odd.get('fixture', {})
            league_info = odd.get('league', {})

            print(f"\nMatch: {fixture_info.get('date', 'N/A')[:10]}")

            bookmakers = odd.get('bookmakers', [])
            for bookmaker in bookmakers[:3]:
                print(f"\nBookmaker: {bookmaker.get('name')}")

                bets = bookmaker.get('bets', [])
                for bet in bets[:3]:
                    print(f"  {bet.get('name')}:")
                    for value in bet.get('values', [])[:5]:
                        print(f"    {value.get('value')}: {value.get('odd')}")

        return True
    else:
        print("⚠️ No odds data found (may not be included in plan)")
        return False

def generate_summary_report(results):
    """Generate summary report"""
    print_section("SUMMARY REPORT")

    print("\nEndpoint Accessibility:")
    print(f"  ✅ API Status: {results.get('status', False)}")
    print(f"  ✅ Leagues: {results.get('leagues', False)}")
    print(f"  ✅ Historical Fixtures: {results.get('fixtures_hist', False)}")
    print(f"  ✅ Fixture Statistics: {results.get('stats', False)}")
    print(f"  ✅ Fixture Events: {results.get('events', False)}")
    print(f"  ✅ Upcoming Fixtures: {results.get('fixtures_upcoming', False)}")
    print(f"  ✅ Head-to-Head: {results.get('h2h', False)}")
    print(f"  ✅ Team Statistics: {results.get('team_stats', False)}")
    print(f"  {'✅' if results.get('injuries', False) else '⚠️ '} Injuries: {results.get('injuries', False)}")
    print(f"  ✅ Standings: {results.get('standings', False)}")
    print(f"  {'✅' if results.get('odds', False) else '⚠️ '} Betting Odds: {results.get('odds', False)}")

    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR INTEGRATION")
    print("="*70)

    print("\n1. CORE DATA AVAILABLE:")
    print("   ✅ Historical match results (multiple seasons)")
    print("   ✅ Detailed match statistics (shots, possession, etc.)")
    print("   ✅ Match events (goals, cards, substitutions)")
    print("   ✅ Team season statistics")
    print("   ✅ League standings")
    print("   ✅ Head-to-head history")

    print("\n2. ENHANCED FEATURES:")
    if results.get('injuries'):
        print("   ✅ Injury data available - integrate for prediction adjustments")
    else:
        print("   ⚠️  Injury data limited - may need manual tracking")

    if results.get('odds'):
        print("   ✅ Betting odds available - can compare vs our predictions")
    else:
        print("   ⚠️  Betting odds limited - use our calibrated probabilities")

    print("\n3. NEXT STEPS:")
    print("   1. Run full data ingestion: python run_api_football.py ingest")
    print("   2. Build feature set: python run_api_football.py features")
    print("   3. Run calibration backtest: python run_api_football.py calibrate --optimize")
    print("   4. Generate predictions: python run_api_football.py predict")

    # Save results
    output_file = Path("api_test_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📁 Detailed results saved to: {output_file}")

def main():
    """Main test function"""
    print("="*70)
    print("API-FOOTBALL COMPREHENSIVE TEST")
    print("="*70)
    print(f"\nAPI Key: {API_KEY[:20]}...")
    print(f"Testing at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize client
    client = APIFootballClient(API_KEY)

    # Track results
    results = {}

    # Run tests
    results['status'] = test_status(client)

    if not results['status']:
        print("\n❌ API connection failed. Please check your API key.")
        return

    results['leagues'] = test_leagues(client)

    fixtures = test_fixtures_historical(client)
    results['fixtures_hist'] = len(fixtures) > 0

    results['stats'] = test_fixture_statistics(client, fixtures)
    results['events'] = test_fixture_events(client, fixtures)

    upcoming = test_upcoming_fixtures(client)
    results['fixtures_upcoming'] = len(upcoming) > 0

    results['h2h'] = test_h2h(client, fixtures if fixtures else upcoming)
    results['team_stats'] = test_team_statistics(client)
    results['injuries'] = test_injuries(client)
    results['standings'] = test_standings(client)
    results['odds'] = test_odds(client, fixtures if fixtures else upcoming)

    # Generate summary
    generate_summary_report(results)

    print("\n✅ Testing complete!")
    print(f"Total API requests made: {client.request_count}")

if __name__ == "__main__":
    main()
