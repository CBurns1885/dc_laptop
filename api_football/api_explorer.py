# api_explorer.py
"""
API-Football Explorer
Discovers and displays all available endpoints, parameters, and data columns

Usage:
    python api_explorer.py --api-key YOUR_KEY
    python api_explorer.py --api-key YOUR_KEY --league 39
    python api_explorer.py --api-key YOUR_KEY --fixture 1234567
"""

import requests
import json
from typing import Dict, List, Any
from datetime import datetime, timedelta
import time


class APIExplorer:
    """Explore API-Football endpoints and data structure"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://v3.football.api-sports.io"
        self.headers = {
            "x-apisports-key": api_key,
            "x-rapidapi-host": "v3.football.api-sports.io"
        }
        self.request_count = 0
    
    def _request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request"""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params or {}, timeout=30)
            self.request_count += 1
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error {response.status_code}: {response.text}")
                return {"response": [], "errors": [response.text]}
        except Exception as e:
            print(f"Request failed: {e}")
            return {"response": [], "errors": [str(e)]}
    
    def check_status(self):
        """Check API account status"""
        print("\n" + "="*70)
        print("API ACCOUNT STATUS")
        print("="*70)
        
        data = self._request("status")
        
        # Handle different response formats
        response = data.get("response", {})
        if isinstance(response, list):
            response = response[0] if response else {}
        
        if response:
            account = response.get("account", {})
            subscription = response.get("subscription", {})
            requests_info = response.get("requests", {})
            
            print(f"\nAccount:")
            print(f"  Name: {account.get('firstname', 'N/A')} {account.get('lastname', '')}")
            print(f"  Email: {account.get('email', 'N/A')}")
            
            print(f"\nSubscription:")
            print(f"  Plan: {subscription.get('plan', 'N/A')}")
            print(f"  End: {subscription.get('end', 'N/A')}")
            print(f"  Active: {subscription.get('active', 'N/A')}")
            
            print(f"\nRequests:")
            print(f"  Current: {requests_info.get('current', 0)}")
            print(f"  Limit (daily): {requests_info.get('limit_day', 0)}")
            remaining = requests_info.get('limit_day', 0) - requests_info.get('current', 0)
            print(f"  Remaining: {remaining}")
        else:
            print("\nCould not retrieve account status")
            print(f"Response: {data}")
        
        return data
    
    def explore_leagues(self, country: str = None, limit: int = 20):
        """Explore available leagues"""
        print("\n" + "="*70)
        print("AVAILABLE LEAGUES")
        print("="*70)
        
        params = {}
        if country:
            params["country"] = country
        
        data = self._request("leagues", params)
        
        if data.get("response"):
            leagues = data["response"][:limit]
            
            print(f"\nFound {len(data['response'])} leagues" + 
                  (f" (showing {limit})" if len(data['response']) > limit else ""))
            print(f"\n{'ID':<8} {'Name':<35} {'Country':<15} {'Type':<10}")
            print("-"*70)
            
            for item in leagues:
                league = item.get("league", {})
                country_info = item.get("country", {})
                print(f"{league.get('id', 'N/A'):<8} {league.get('name', 'N/A'):<35} "
                      f"{country_info.get('name', 'N/A'):<15} {league.get('type', 'N/A'):<10}")
            
            # Show sample structure
            if leagues:
                print("\n" + "-"*70)
                print("LEAGUE DATA STRUCTURE:")
                print("-"*70)
                self._print_structure(leagues[0], indent=2)
        
        return data
    
    def explore_fixture(self, fixture_id: int = None, league_id: int = None):
        """Explore fixture data structure"""
        print("\n" + "="*70)
        print("FIXTURE DATA STRUCTURE")
        print("="*70)
        
        if fixture_id:
            data = self._request("fixtures", {"id": fixture_id})
        elif league_id:
            # Get a recent completed fixture
            data = self._request("fixtures", {
                "league": league_id,
                "season": 2024,
                "status": "FT",
                "last": 1
            })
        else:
            # Default: get a Premier League fixture
            data = self._request("fixtures", {
                "league": 39,
                "season": 2024,
                "status": "FT",
                "last": 1
            })
        
        if data.get("response"):
            fixture = data["response"][0]
            
            print("\nSample Fixture:")
            print(f"  {fixture['teams']['home']['name']} vs {fixture['teams']['away']['name']}")
            print(f"  Date: {fixture['fixture']['date']}")
            print(f"  Score: {fixture['goals']['home']} - {fixture['goals']['away']}")
            
            print("\n" + "-"*70)
            print("FULL FIXTURE STRUCTURE:")
            print("-"*70)
            self._print_structure(fixture, indent=2)
            
            return fixture
        
        return None
    
    def explore_fixture_statistics(self, fixture_id: int):
        """Explore match statistics"""
        print("\n" + "="*70)
        print("FIXTURE STATISTICS")
        print("="*70)
        
        data = self._request("fixtures/statistics", {"fixture": fixture_id})
        
        if data.get("response"):
            for team_stats in data["response"]:
                team = team_stats.get("team", {})
                stats = team_stats.get("statistics", [])
                
                print(f"\n{team.get('name', 'Unknown')}:")
                print("-"*40)
                
                for stat in stats:
                    stat_type = stat.get("type", "Unknown")
                    value = stat.get("value", "N/A")
                    print(f"  {stat_type:<25} {value}")
            
            print("\n" + "-"*70)
            print("AVAILABLE STATISTICS TYPES:")
            print("-"*70)
            if data["response"]:
                stat_types = [s["type"] for s in data["response"][0].get("statistics", [])]
                for i, st in enumerate(stat_types, 1):
                    print(f"  {i:2}. {st}")
        
        return data
    
    def explore_fixture_events(self, fixture_id: int):
        """Explore match events (goals, cards, subs)"""
        print("\n" + "="*70)
        print("FIXTURE EVENTS")
        print("="*70)
        
        data = self._request("fixtures/events", {"fixture": fixture_id})
        
        if data.get("response"):
            events = data["response"]
            
            print(f"\nFound {len(events)} events:")
            print(f"\n{'Time':<8} {'Team':<20} {'Type':<15} {'Player':<25} {'Detail':<20}")
            print("-"*90)
            
            for event in events[:20]:
                time_info = event.get("time", {})
                elapsed = time_info.get("elapsed", 0)
                extra = time_info.get("extra")
                time_str = f"{elapsed}'" + (f"+{extra}" if extra else "")
                
                team = event.get("team", {}).get("name", "N/A")[:18]
                event_type = event.get("type", "N/A")
                player = event.get("player", {}).get("name", "N/A")[:23]
                detail = event.get("detail", "N/A")[:18]
                
                print(f"{time_str:<8} {team:<20} {event_type:<15} {player:<25} {detail:<20}")
            
            print("\n" + "-"*70)
            print("EVENT TYPES:")
            event_types = set(e.get("type") for e in events)
            for et in sorted(event_types):
                print(f"  - {et}")
        
        return data
    
    def explore_lineups(self, fixture_id: int):
        """Explore lineup data"""
        print("\n" + "="*70)
        print("FIXTURE LINEUPS")
        print("="*70)
        
        data = self._request("fixtures/lineups", {"fixture": fixture_id})
        
        if data.get("response"):
            for team_lineup in data["response"]:
                team = team_lineup.get("team", {})
                formation = team_lineup.get("formation", "N/A")
                coach = team_lineup.get("coach", {})
                
                print(f"\n{team.get('name', 'Unknown')} ({formation})")
                print(f"Coach: {coach.get('name', 'N/A')}")
                print("-"*50)
                
                print("\nStarting XI:")
                for player in team_lineup.get("startXI", [])[:11]:
                    p = player.get("player", {})
                    print(f"  {p.get('number', '?'):>2}. {p.get('name', 'N/A'):<25} ({p.get('pos', 'N/A')})")
                
                subs = team_lineup.get("substitutes", [])
                if subs:
                    print(f"\nSubstitutes ({len(subs)}):")
                    for player in subs[:5]:
                        p = player.get("player", {})
                        print(f"  {p.get('number', '?'):>2}. {p.get('name', 'N/A'):<25} ({p.get('pos', 'N/A')})")
                    if len(subs) > 5:
                        print(f"  ... and {len(subs) - 5} more")
        
        return data
    
    def explore_injuries(self, league_id: int = 39, season: int = 2024):
        """Explore injury data"""
        print("\n" + "="*70)
        print("INJURY DATA")
        print("="*70)
        
        data = self._request("injuries", {"league": league_id, "season": season})
        
        if data.get("response"):
            injuries = data["response"][:20]
            
            print(f"\nFound {len(data['response'])} injury records (showing 20)")
            print(f"\n{'Player':<25} {'Team':<20} {'Type':<15} {'Reason':<20}")
            print("-"*80)
            
            for injury in injuries:
                player = injury.get("player", {}).get("name", "N/A")[:23]
                team = injury.get("team", {}).get("name", "N/A")[:18]
                inj_type = injury.get("player", {}).get("type", "N/A")[:13]
                reason = injury.get("player", {}).get("reason", "N/A")[:18]
                
                print(f"{player:<25} {team:<20} {inj_type:<15} {reason:<20}")
            
            if injuries:
                print("\n" + "-"*70)
                print("INJURY DATA STRUCTURE:")
                print("-"*70)
                self._print_structure(injuries[0], indent=2)
        
        return data
    
    def explore_predictions(self, fixture_id: int):
        """Explore API predictions"""
        print("\n" + "="*70)
        print("API PREDICTIONS")
        print("="*70)
        
        data = self._request("predictions", {"fixture": fixture_id})
        
        if data.get("response"):
            pred = data["response"][0] if data["response"] else {}
            
            predictions = pred.get("predictions", {})
            teams = pred.get("teams", {})
            
            print(f"\n{teams.get('home', {}).get('name', 'Home')} vs {teams.get('away', {}).get('name', 'Away')}")
            print("-"*50)
            
            print(f"\nWinner: {predictions.get('winner', {}).get('name', 'N/A')}")
            print(f"Advice: {predictions.get('advice', 'N/A')}")
            
            goals = predictions.get("goals", {})
            print(f"\nGoals prediction:")
            print(f"  Home: {goals.get('home', 'N/A')}")
            print(f"  Away: {goals.get('away', 'N/A')}")
            
            percent = predictions.get("percent", {})
            print(f"\nWin probabilities:")
            print(f"  Home: {percent.get('home', 'N/A')}")
            print(f"  Draw: {percent.get('draw', 'N/A')}")
            print(f"  Away: {percent.get('away', 'N/A')}")
            
            print("\n" + "-"*70)
            print("FULL PREDICTION STRUCTURE:")
            print("-"*70)
            self._print_structure(pred, indent=2, max_depth=3)
        
        return data
    
    def explore_odds(self, fixture_id: int):
        """Explore odds data"""
        print("\n" + "="*70)
        print("ODDS DATA")
        print("="*70)
        
        data = self._request("odds", {"fixture": fixture_id})
        
        if data.get("response"):
            odds_data = data["response"][0] if data["response"] else {}
            bookmakers = odds_data.get("bookmakers", [])
            
            if bookmakers:
                print(f"\nFound {len(bookmakers)} bookmakers")
                
                # Show first bookmaker's odds
                bookie = bookmakers[0]
                print(f"\n{bookie.get('name', 'Unknown')}:")
                print("-"*50)
                
                for bet in bookie.get("bets", []):
                    bet_name = bet.get("name", "Unknown")
                    print(f"\n  {bet_name}:")
                    for value in bet.get("values", []):
                        print(f"    {value.get('value', 'N/A')}: {value.get('odd', 'N/A')}")
                
                print("\n" + "-"*70)
                print("AVAILABLE BET TYPES:")
                print("-"*70)
                bet_types = set()
                for bookie in bookmakers:
                    for bet in bookie.get("bets", []):
                        bet_types.add(bet.get("name"))
                
                for i, bt in enumerate(sorted(bet_types), 1):
                    print(f"  {i:2}. {bt}")
        
        return data
    
    def explore_h2h(self, team1_id: int, team2_id: int):
        """Explore head-to-head data"""
        print("\n" + "="*70)
        print("HEAD TO HEAD")
        print("="*70)
        
        data = self._request("fixtures/headtohead", {"h2h": f"{team1_id}-{team2_id}", "last": 10})
        
        if data.get("response"):
            matches = data["response"]
            
            print(f"\nFound {len(matches)} past meetings:")
            print(f"\n{'Date':<12} {'Home':<20} {'Score':^7} {'Away':<20}")
            print("-"*65)
            
            for match in matches:
                date = match.get("fixture", {}).get("date", "")[:10]
                home = match.get("teams", {}).get("home", {}).get("name", "N/A")[:18]
                away = match.get("teams", {}).get("away", {}).get("name", "N/A")[:18]
                home_goals = match.get("goals", {}).get("home", "?")
                away_goals = match.get("goals", {}).get("away", "?")
                score = f"{home_goals} - {away_goals}"
                
                print(f"{date:<12} {home:<20} {score:^7} {away:<20}")
        
        return data
    
    def explore_team(self, team_id: int, league_id: int = 39, season: int = 2024):
        """Explore team statistics"""
        print("\n" + "="*70)
        print("TEAM STATISTICS")
        print("="*70)
        
        data = self._request("teams/statistics", {
            "team": team_id,
            "league": league_id,
            "season": season
        })
        
        if data.get("response"):
            stats = data["response"]
            team = stats.get("team", {})
            league = stats.get("league", {})
            
            print(f"\n{team.get('name', 'Unknown')} - {league.get('name', 'Unknown')} {league.get('season', '')}")
            print("-"*50)
            
            fixtures = stats.get("fixtures", {})
            print(f"\nFixtures:")
            print(f"  Played: Home={fixtures.get('played', {}).get('home', 0)}, Away={fixtures.get('played', {}).get('away', 0)}")
            print(f"  Wins: Home={fixtures.get('wins', {}).get('home', 0)}, Away={fixtures.get('wins', {}).get('away', 0)}")
            
            goals = stats.get("goals", {})
            print(f"\nGoals:")
            print(f"  Scored: Home={goals.get('for', {}).get('total', {}).get('home', 0)}, Away={goals.get('for', {}).get('total', {}).get('away', 0)}")
            print(f"  Conceded: Home={goals.get('against', {}).get('total', {}).get('home', 0)}, Away={goals.get('against', {}).get('total', {}).get('away', 0)}")
            
            clean_sheets = stats.get("clean_sheet", {})
            print(f"\nClean Sheets:")
            print(f"  Home: {clean_sheets.get('home', 0)}, Away: {clean_sheets.get('away', 0)}")
            
            print("\n" + "-"*70)
            print("FULL TEAM STATS STRUCTURE:")
            print("-"*70)
            self._print_structure(stats, indent=2, max_depth=3)
        
        return data
    
    def explore_players(self, team_id: int, season: int = 2024):
        """Explore player data"""
        print("\n" + "="*70)
        print("PLAYER DATA")
        print("="*70)
        
        data = self._request("players", {"team": team_id, "season": season})
        
        if data.get("response"):
            players = data["response"][:15]
            
            print(f"\nFound {len(data['response'])} players (showing 15)")
            print(f"\n{'Name':<25} {'Age':<5} {'Position':<12} {'Goals':<6} {'Assists':<8}")
            print("-"*60)
            
            for item in players:
                player = item.get("player", {})
                stats = item.get("statistics", [{}])[0] if item.get("statistics") else {}
                goals_data = stats.get("goals", {})
                
                name = player.get("name", "N/A")[:23]
                age = player.get("age", "N/A")
                position = stats.get("games", {}).get("position", "N/A")[:10]
                goals = goals_data.get("total") or 0
                assists = goals_data.get("assists") or 0
                
                print(f"{name:<25} {str(age):<5} {position:<12} {goals:<6} {assists:<8}")
            
            if players:
                print("\n" + "-"*70)
                print("PLAYER STATISTICS STRUCTURE:")
                print("-"*70)
                self._print_structure(players[0], indent=2, max_depth=3)
        
        return data
    
    def _print_structure(self, obj: Any, indent: int = 0, max_depth: int = 4, current_depth: int = 0):
        """Recursively print data structure"""
        prefix = " " * indent
        
        if current_depth >= max_depth:
            print(f"{prefix}...")
            return
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, dict):
                    print(f"{prefix}{key}: {{")
                    self._print_structure(value, indent + 2, max_depth, current_depth + 1)
                    print(f"{prefix}}}")
                elif isinstance(value, list):
                    if len(value) > 0:
                        print(f"{prefix}{key}: [  # {len(value)} items")
                        if isinstance(value[0], dict):
                            self._print_structure(value[0], indent + 2, max_depth, current_depth + 1)
                        else:
                            print(f"{prefix}  {type(value[0]).__name__}: {str(value[0])[:50]}")
                        print(f"{prefix}]")
                    else:
                        print(f"{prefix}{key}: []")
                else:
                    val_str = str(value)[:50] if value else "null"
                    print(f"{prefix}{key}: {val_str}")
        elif isinstance(obj, list) and len(obj) > 0:
            self._print_structure(obj[0], indent, max_depth, current_depth)
    
    def run_full_exploration(self, league_id: int = 39):
        """Run comprehensive exploration"""
        print("\n" + "="*70)
        print("API-FOOTBALL FULL EXPLORATION")
        print("="*70)
        print(f"League ID: {league_id}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check status
        self.check_status()
        time.sleep(0.5)
        
        # Get a sample fixture
        print("\n\nFetching sample fixture...")
        fixture_data = self._request("fixtures", {
            "league": league_id,
            "season": 2024,
            "status": "FT",
            "last": 1
        })
        
        if fixture_data.get("response"):
            fixture = fixture_data["response"][0]
            fixture_id = fixture["fixture"]["id"]
            home_team_id = fixture["teams"]["home"]["id"]
            away_team_id = fixture["teams"]["away"]["id"]
            
            print(f"Sample fixture: {fixture['teams']['home']['name']} vs {fixture['teams']['away']['name']}")
            print(f"Fixture ID: {fixture_id}")
            
            # Explore fixture structure
            self.explore_fixture(fixture_id=fixture_id)
            time.sleep(0.5)
            
            # Explore statistics
            self.explore_fixture_statistics(fixture_id)
            time.sleep(0.5)
            
            # Explore events
            self.explore_fixture_events(fixture_id)
            time.sleep(0.5)
            
            # Explore lineups
            self.explore_lineups(fixture_id)
            time.sleep(0.5)
            
            # Explore predictions
            # Get an upcoming fixture for predictions
            upcoming = self._request("fixtures", {
                "league": league_id,
                "season": 2024,
                "status": "NS",
                "next": 1
            })
            if upcoming.get("response"):
                upcoming_id = upcoming["response"][0]["fixture"]["id"]
                self.explore_predictions(upcoming_id)
                time.sleep(0.5)
                self.explore_odds(upcoming_id)
            
            time.sleep(0.5)
            
            # Explore H2H
            self.explore_h2h(home_team_id, away_team_id)
            time.sleep(0.5)
            
            # Explore team stats
            self.explore_team(home_team_id, league_id)
            time.sleep(0.5)
            
            # Explore players
            self.explore_players(home_team_id)
            time.sleep(0.5)
            
            # Explore injuries
            self.explore_injuries(league_id)
        
        print("\n" + "="*70)
        print(f"EXPLORATION COMPLETE")
        print(f"Total API requests: {self.request_count}")
        print("="*70)


def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Explore API-Football endpoints")
    parser.add_argument("--api-key", default=os.environ.get("API_FOOTBALL_KEY"),
                        help="API key (or set API_FOOTBALL_KEY)")
    parser.add_argument("--league", type=int, default=39, help="League ID (default: 39 = Premier League)")
    parser.add_argument("--fixture", type=int, help="Specific fixture ID to explore")
    parser.add_argument("--team", type=int, help="Team ID to explore")
    parser.add_argument("--full", action="store_true", help="Run full exploration")
    parser.add_argument("--status-only", action="store_true", help="Just check API status")
    parser.add_argument("--list-leagues", action="store_true", help="List available leagues")
    parser.add_argument("--country", help="Filter leagues by country")
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("ERROR: API key required. Set API_FOOTBALL_KEY or use --api-key")
        return 1
    
    explorer = APIExplorer(args.api_key)
    
    if args.status_only:
        explorer.check_status()
    elif args.list_leagues:
        explorer.explore_leagues(country=args.country, limit=50)
    elif args.fixture:
        explorer.explore_fixture(fixture_id=args.fixture)
        explorer.explore_fixture_statistics(args.fixture)
        explorer.explore_fixture_events(args.fixture)
        explorer.explore_lineups(args.fixture)
    elif args.team:
        explorer.explore_team(args.team, args.league)
        explorer.explore_players(args.team)
    elif args.full:
        explorer.run_full_exploration(args.league)
    else:
        # Default: show status and sample fixture
        explorer.check_status()
        explorer.explore_fixture(league_id=args.league)
        explorer.explore_fixture_statistics
    
    return 0


if __name__ == "__main__":
    exit(main())
