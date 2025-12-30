# api_test.py
"""
API-Football Key Tester
Tests your API key against both possible API hosts
"""

import requests

API_KEY = "7c836ddd7b00863b3c8c6068e39ebb90"

# Two possible API hosts
HOSTS = [
    {
        "name": "RapidAPI (api-football-v1.p.rapidapi.com)",
        "url": "https://api-football-v1.p.rapidapi.com/v3/status",
        "headers": {
            "x-rapidapi-key": API_KEY,
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com"
        }
    },
    {
        "name": "API-Sports (v3.football.api-sports.io)",
        "url": "https://v3.football.api-sports.io/status",
        "headers": {
            "x-apisports-key": API_KEY
        }
    },
    {
        "name": "API-Sports with rapidapi header",
        "url": "https://v3.football.api-sports.io/status",
        "headers": {
            "x-rapidapi-key": API_KEY
        }
    }
]

print("="*60)
print("API-FOOTBALL KEY TESTER")
print("="*60)
print(f"Testing key: {API_KEY[:8]}...{API_KEY[-4:]}")
print()

for host in HOSTS:
    print(f"\nTesting: {host['name']}")
    print("-"*50)
    
    try:
        response = requests.get(
            host["url"],
            headers=host["headers"],
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        data = response.json()
        
        if data.get("errors"):
            print(f"Errors: {data['errors']}")
        
        if data.get("response"):
            resp = data["response"]
            if isinstance(resp, dict):
                account = resp.get("account", {})
                requests_info = resp.get("requests", {})
                
                print(f"✅ SUCCESS!")
                print(f"   Account: {account.get('email', 'N/A')}")
                print(f"   Plan: {resp.get('subscription', {}).get('plan', 'N/A')}")
                print(f"   Requests today: {requests_info.get('current', 0)} / {requests_info.get('limit_day', 0)}")
            elif isinstance(resp, list) and len(resp) > 0:
                print(f"✅ Got response with {len(resp)} items")
            else:
                print(f"Response: {resp}")
        else:
            print(f"Full response: {data}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "="*60)
print("If one of these worked, update api_explorer.py with that host/header format")
print("="*60)
