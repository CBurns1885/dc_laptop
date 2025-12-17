# update_results.py
"""
Update accuracy database with actual match results
"""

import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

def update_accuracy_database():
    """
    Update predictions database with actual match results

    Returns:
        bool: True if results were updated, False otherwise
    """
    db_path = Path("outputs/accuracy_database.db")

    if not db_path.exists():
        print(" No accuracy database found (no predictions logged yet)")
        return False

    try:
        conn = sqlite3.connect(db_path)

        # Get predictions awaiting results
        query = """
            SELECT prediction_id, match_date, home_team, away_team,
                   market, predicted_outcome
            FROM predictions
            WHERE actual_outcome IS NULL
            AND match_date <= date('now')
        """

        pending_df = pd.read_sql_query(query, conn)

        if len(pending_df) == 0:
            conn.close()
            return False

        # Here you would fetch actual results from API/scraping
        # For now, we'll just mark as pending
        print(f" Found {len(pending_df)} predictions awaiting results")
        print(" (Actual result fetching not implemented yet)")

        conn.close()
        return False

    except Exception as e:
        print(f" Error updating accuracy database: {e}")
        return False

def show_recent_performance(weeks: int = 2):
    """
    Show recent prediction performance

    Args:
        weeks: Number of recent weeks to show
    """
    db_path = Path("outputs/accuracy_database.db")

    if not db_path.exists():
        print(" No accuracy database found")
        return

    try:
        conn = sqlite3.connect(db_path)

        # Get recent performance
        cutoff_date = (datetime.now() - timedelta(weeks=weeks)).strftime('%Y-%m-%d')

        query = f"""
            SELECT
                market,
                COUNT(*) as total,
                SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct,
                AVG(CASE WHEN correct = 1 THEN 100.0 ELSE 0.0 END) as accuracy
            FROM predictions
            WHERE prediction_date >= '{cutoff_date}'
            AND actual_outcome IS NOT NULL
            GROUP BY market
            ORDER BY accuracy DESC
        """

        df = pd.read_sql_query(query, conn)

        if len(df) == 0:
            print(f" No completed predictions in last {weeks} weeks")
        else:
            print(f"\n Performance (last {weeks} weeks):")
            for _, row in df.iterrows():
                print(f"   {row['market']:10} {row['accuracy']:.1f}% ({row['correct']:.0f}/{row['total']:.0f})")

        conn.close()

    except Exception as e:
        print(f" Error showing performance: {e}")

if __name__ == "__main__":
    print("Updating accuracy database...")
    success = update_accuracy_database()

    if success:
        print("\nRecent performance:")
        show_recent_performance(weeks=4)
    else:
        print("No updates available")
