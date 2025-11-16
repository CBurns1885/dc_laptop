#!/usr/bin/env python3
"""
api_football_weekly_fixtures.py

Standalone script to download ALL fixtures for the NEXT 7 DAYS from API-FOOTBALL
(including European cups) and save them as:
  - a combined CSV in ./outputs/
  - per-day CSVs in ./outputs/daily/

It also includes a --diagnose mode to print why you might be getting empty results.
Supports BOTH the native API-FOOTBALL host and the RapidAPI mirror.

USAGE (examples)
---------------
# Normal run (7 days from today, using native host):
python api_football_weekly_fixtures.py

# Normal run with RapidAPI (if your key is RapidAPI-based):
python api_football_weekly_fixtures.py --provider rapidapi

# Diagnose connectivity/key/plan issues:
python api_football_weekly_fixtures.py --diagnose
python api_football_weekly_fixtures.py --diagnose --provider rapidapi
"""
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
import re
API_FOOTBALL_KEY = "0f17fdba78d15a625710f7244a1cc770"

import requests
import pandas as pd

# ---------- Providers ----------
NATIVE_BASE = "https://v3.football.api-sports.io"         # requires header: x-apisports-key
RAPIDAPI_BASE = "https://api-football-v1.p.rapidapi.com/v3"  # requires headers: X-RapidAPI-Key + X-RapidAPI-Host

# Read keys from env or inline (inline provided for convenience here)
API_FOOTBALL_KEY = os.environ.get("API_FOOTBALL_KEY") or "0f17fdba78d15a625710f7244a1cc770"
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")  # optional, only if you use RapidAPI

# Output locations
OUTPUT_DIR = Path("outputs")
DAILY_DIR = OUTPUT_DIR / "daily"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
DAILY_DIR.mkdir(exist_ok=True, parents=True)

# -------- Config you can tweak --------
DAYS_AHEAD = 7            # Next 7 days (inclusive of today)
MAX_RETRIES = 5
BACKOFF_SECONDS = 2       # base backoff, will multiply each retry
REQUEST_TIMEOUT = 25
TIMEZONE = "UTC"          # change to "Europe/London" if you prefer
# -------------------------------------


def iso_date(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def build_session(provider: str) -> Tuple[requests.Session, str]:
    provider = provider.lower().strip()
    sess = requests.Session()
    if provider == "rapidapi":
        if not RAPIDAPI_KEY and not API_FOOTBALL_KEY:
            raise RuntimeError("No RapidAPI key set. Set RAPIDAPI_KEY env var or pass --provider native with API_FOOTBALL_KEY.")
        key = RAPIDAPI_KEY or API_FOOTBALL_KEY  # allow fallback for convenience
        sess.headers.update({
            "X-RapidAPI-Key": key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        })
        return sess, RAPIDAPI_BASE
    else:
        # native is default
        if not API_FOOTBALL_KEY:
            raise RuntimeError("API_FOOTBALL_KEY is not set. Set env var or edit the script.")
        sess.headers.update({"x-apisports-key": API_FOOTBALL_KEY})
        return sess, NATIVE_BASE



def is_uefa_cup(league_name: Optional[str]) -> bool:
    """Return True if the league name indicates UCL/UEL/UECL (robust to minor naming differences)."""
    if not league_name:
        return False
    n = league_name.lower()
    targets = [
        "uefa champions league",
        "champions league",
        "uefa europa league",
        "europa league",
        "uefa europa conference league",
        "europa conference league",
    ]
    return any(t in n for t in targets)


def parse_league_ids(arg: Optional[str]) -> List[int]:
    """Parse a comma/space-separated list of league IDs into ints."""
    if not arg:
        return []
    parts = re.split(r"[,\s]+", arg.strip())
    out: List[int] = []
    for p_ in parts:
        if not p_:
            continue
        try:
            out.append(int(p_))
        except ValueError:
            pass
    return out


def request_json(sess: requests.Session, url: str, params: Dict[str, Any]) -> Dict[str, Any]:

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = sess.get(url, params=params, timeout=REQUEST_TIMEOUT)
            status = resp.status_code
            if status == 429 or 500 <= status < 600:
                wait = BACKOFF_SECONDS * attempt
                print(f"   HTTP {status} from {url}; retrying in {wait}s (attempt {attempt}/{MAX_RETRIES})...")
                time.sleep(wait)
                continue
            # Raise for 4xx except 429 (handled above)
            resp.raise_for_status()
            try:
                return resp.json()
            except Exception:
                raise RuntimeError(f"Non-JSON response from {url}: {resp.text[:200]}")
        except requests.HTTPError as e:
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                # Print the error payload to help diagnose auth/plan issues
                try:
                    err = resp.json()
                except Exception:
                    err = {"raw": resp.text[:300]}
                raise RuntimeError(f"HTTP {resp.status_code} error from {url}\nParams: {params}\nBody: {err}") from e
        except requests.RequestException as e:
            wait = BACKOFF_SECONDS * attempt
            print(f"   Network error calling {url}: {e}; retrying in {wait}s (attempt {attempt}/{MAX_RETRIES})...")
            time.sleep(wait)
    raise RuntimeError(f"Failed to call {url} after {MAX_RETRIES} retries.")


def fetch_fixtures_for_date(sess: requests.Session, base: str, date_str: str, league_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """Fetch fixtures for a single date (YYYY-MM-DD); handles pagination via 'paging'."""
    all_rows: List[Dict[str, Any]] = []
    page = 1
    # If league_ids provided, we'll loop leagues sequentially to keep params simple (one league per request)
    leagues_to_fetch = league_ids or [None]
    for lg in leagues_to_fetch:
        page = 1
        while True:
            # First request mirrors diagnostics EXACTLY (no 'page' param)
            params = {"date": date_str, "timezone": TIMEZONE}
            if lg is not None:
                params["league"] = lg
            if page > 1:
                params["page"] = page
            data = request_json(sess, f"{base}/fixtures", params=params)
        # Log any API-provided errors to help debug empty responses
        if data.get("errors"):
            print(f"   {date_str}: API errors -> {data['errors']}")
        response = data.get("response", [])
        paging = data.get("paging", {}) or data.get("pagination", {})
        total_pages = int(paging.get("total", 1)) if isinstance(paging.get("total", 1), int) else 1
        current_page = int(paging.get("current", page)) if isinstance(paging.get("current", page), int) else page

        for item in response:
            fixture = item.get("fixture", {})
            league = item.get("league", {})
            teams = item.get("teams", {})
            goals = item.get("goals", {})
            raw_date = fixture.get("date")
            try:
                dt = datetime.fromisoformat((raw_date or "").replace("Z", "+00:00"))
            except Exception:
                dt = None

            all_rows.append({
                "fixture_id": fixture.get("id"),
                "kickoff_utc": dt.strftime("%Y-%m-%d %H:%M:%S") if dt else raw_date,
                "kickoff_date": iso_date(dt) if dt else date_str,
                "status": (fixture.get("status") or {}).get("short"),
                "venue": (fixture.get("venue") or {}).get("name"),
                "referee": fixture.get("referee"),
                "league_id": league.get("id"),
                "league_name": league.get("name"),
                "league_country": league.get("country"),
                "league_season": league.get("season"),
                "league_round": league.get("round"),
                "home_id": (teams.get("home") or {}).get("id"),
                "home_name": (teams.get("home") or {}).get("name"),
                "away_id": (teams.get("away") or {}).get("id"),
                "away_name": (teams.get("away") or {}).get("name"),
                "goals_home": goals.get("home"),
                "goals_away": goals.get("away"),
                "source": "API-FOOTBALL",
            })

        if current_page == 1:
            print(f"   {date_str}: page {current_page}/{total_pages} -> {len(response)} fixtures")
        else:
            print(f"   {date_str}: page {current_page}/{total_pages} -> +{len(response)} fixtures (cum {len(all_rows)})")

        if current_page >= total_pages or total_pages == 0:
            break
        page += 1
    return all_rows



def save_master(rows: List[Dict[str, Any]], master_path: Path) -> Path:
    """Append to a master CSV (creating if needed) and drop duplicates by fixture_id."""
    import pandas as pd
    new_df = pd.DataFrame(rows)
    if master_path.exists():
        old_df = pd.read_csv(master_path)
        merged = pd.concat([old_df, new_df], ignore_index=True)
    else:
        merged = new_df
    # Drop duplicates by fixture_id + kickoff_utc to be extra safe
    merged = merged.drop_duplicates(subset=["fixture_id", "kickoff_utc"], keep="last")
    merged = merged.sort_values(["kickoff_date", "league_country", "league_name", "kickoff_utc", "home_name"])
    master_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(master_path, index=False)
    print(f"📦 Updated master CSV: {master_path.resolve()} ({len(merged)} rows)")
    return master_path


def save_csvs(rows: List[Dict[str, Any]], date_from: str, date_to: str) -> Path:

    if not rows:
        raise ValueError("No fixtures returned to save.")
    df = pd.DataFrame(rows)
    df["kickoff_date"] = pd.to_datetime(df["kickoff_date"], errors="coerce")
    df = df.sort_values(["kickoff_date", "league_country", "league_name", "kickoff_utc", "home_name"])
    combined_name = f"upcoming_fixtures_{date_from}_to_{date_to}.csv"
    combined_path = OUTPUT_DIR / combined_name
    df.to_csv(combined_path, index=False)
    print(f"✅ Saved combined CSV: {combined_path.resolve()} ({len(df)} rows)")
    for day, df_day in df.groupby(df["kickoff_date"].dt.strftime("%Y-%m-%d")):
        day_path = DAILY_DIR / f"{day}.csv"
        df_day.to_csv(day_path, index=False)
        print(f"   └─ Saved {len(df_day)} fixtures for {day}: {day_path.resolve()}")
    return combined_path


def diagnose(sess: requests.Session, base: str) -> None:
    """Run quick checks to explain empty responses."""
    print("\n--- Diagnostics ---")
    # 1) Status
    try:
        status = request_json(sess, f"{base}/status", {})
        print("Status:", status.get("response", {}))
    except Exception as e:
        print("Status check failed:", e)

    # 2) Timezone list (confirms auth + shows default)
    try:
        tz = request_json(sess, f"{base}/timezone", {})
        tz_list = tz.get("response", [])
        print(f"Timezones available: {len(tz_list)} (showing first 5): {tz_list[:5]}")
    except Exception as e:
        print("Timezone check failed:", e)

    # 3) Current leagues (helps reveal plan coverage)
    try:
        leagues = request_json(sess, f"{base}/leagues", {"current": "true"})
        lresp = leagues.get("response", [])
        print(f"Current leagues accessible: {len(lresp)} (showing first 3 names): {[ (x.get('league') or {}).get('name') for x in lresp[:3] ]}")
    except Exception as e:
        print("Leagues check failed:", e)

    # 4) Today's fixtures as a smoke test
    try:
        today_utc = datetime.now(timezone.utc).date()
        date_str = today_utc.strftime("%Y-%m-%d")
        fx = request_json(sess, f"{base}/fixtures", {"date": date_str, "timezone": TIMEZONE})
        print(f"Today's fixtures (UTC date={date_str}):", len(fx.get("response", [])))
        if fx.get("errors"):
            print("Fixture errors:", fx["errors"])
    except Exception as e:
        print("Fixtures smoke test failed:", e)
    print("--- End diagnostics ---\n")


def main(days_ahead: Optional[int], provider: str, run_diagnose: bool, master: Optional[str], stop_on_free_window: bool, uefa_only: bool, league_ids_arg: Optional[str]) -> int:
    sess, base = build_session(provider)

    if run_diagnose:
        diagnose(sess, base)

    league_ids = parse_league_ids(league_ids_arg)
    if league_ids:
        print(f"Server-side filtering by league IDs: {league_ids}")
    if uefa_only and not league_ids:
        print("Client-side UEFA-only filter is ON (Champions/Europa/Conference Leagues).")

    today_utc = datetime.now(timezone.utc).date()
    start_dt = datetime(today_utc.year, today_utc.month, today_utc.day, tzinfo=timezone.utc)
    date_from = iso_date(start_dt)
    end_dt = start_dt + timedelta(days=(days_ahead or DAYS_AHEAD) - 1)  # inclusive
    date_to = iso_date(end_dt)

    print("=" * 70)
    print("API-FOOTBALL — Weekly Fixture Downloader")
    print("=" * 70)
    print(f"Provider: {provider} | Date range (UTC): {date_from} → {date_to} (days: {(days_ahead or DAYS_AHEAD)})")
    if provider == "rapidapi":
        print(f"Auth header: X-RapidAPI-Key ({'env' if os.environ.get('RAPIDAPI_KEY') else 'inline/fallback'}) ✅")
    else:
        print(f"Auth header: x-apisports-key ({'env' if os.environ.get('API_FOOTBALL_KEY') else 'inline'}) ✅")
    print("Fetching by day to include ALL leagues & cups (UCL/UEL/UECL etc.)...\n")

    all_rows: List[Dict[str, Any]] = []
    cur = start_dt
    free_window_hit = False
    while cur <= end_dt:
        day = iso_date(cur)
        try:
            day_rows = fetch_fixtures_for_date(sess, base, day, league_ids=league_ids if league_ids else None)
        except RuntimeError as e:
            # bubble up hard errors
            raise
        # Detect the free-plan window message echoed by the API (printed in fetch)
        # If the API returned 0 with a 'plan' error, gracefully stop if requested
        # We infer it by checking if no rows came back while later dates are in the future
        if len(day_rows) == 0 and (cur.date() > start_dt.date()):
            free_window_hit = True
        # Optional client-side UEFA filter
        if uefa_only and not league_ids:
            day_rows = [r for r in day_rows if is_uefa_cup(r.get("league_name"))]
        all_rows.extend(day_rows)
        if free_window_hit and stop_on_free_window:
            print("⛔ Reached the Free plan date window; stopping early. Run the script again tomorrow to build up a master CSV.")
            break
        cur += timedelta(days=1)

    if not all_rows:
        print("⚠️ No fixtures returned from API-FOOTBALL in this window.")
        return 2

    combined_path = save_csvs(all_rows, date_from, date_to)
    if uefa_only:
        # Rename combined to include '-uefa' suffix
        from pathlib import Path as _P
        new_path = _P(str(combined_path).replace('.csv', '-uefa.csv'))
        _P(combined_path).rename(new_path)
        combined_path = new_path
    if master:
        from pathlib import Path as _P
        save_master(all_rows, _P(master))
    print("\n🎉 Done! You can now use the CSV(s) for your workflow.")
    print(f"Combined file: {combined_path.resolve()}")
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download fixtures from API-FOOTBALL.")
    parser.add_argument("--days", type=int, default=DAYS_AHEAD, help="Number of days ahead to fetch (default: 7, inclusive of today)")
    parser.add_argument("--provider", choices=["native", "rapidapi"], default="native", help="API host to use (default: native)")
    parser.add_argument("--diagnose", action="store_true", help="Run diagnostics before fetching")
    parser.add_argument("--master", type=str, default=None, help="Path to a rolling master CSV to append & de-duplicate")
    parser.add_argument("--stop-on-free-window", action="store_true", help="Stop fetching once the API reports the free-plan date window limitation")
    parser.add_argument("--uefa-only", action="store_true", help="Filter results to UEFA Champions/Europa/Conference League only (client-side)")
    parser.add_argument("--league-ids", type=str, default=None, help="Comma/space-separated league IDs to fetch server-side (e.g. '2,3,848')")
    args = parser.parse_args()
    sys.exit(main(args.days, args.provider, args.diagnose, args.master, args.stop_on_free_window, args.uefa_only, args.league_ids))
    sys.exit(main(
        days_ahead=2,
        provider="native",
        run_diagnose=False,
        master="outputs/uefa_master.csv",
        stop_on_free_window=True,
        uefa_only=True,
        league_ids_arg=None
    ))