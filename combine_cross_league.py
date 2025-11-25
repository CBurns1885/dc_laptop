#!/usr/bin/env python3
"""
combine_cross_league.py

Post-processing script that combines duplicate cross-league match predictions
by averaging their probabilities.

Usage:
    python combine_cross_league.py
    
    Or from another script:
    from combine_cross_league import combine_predictions
    combine_predictions("weekly_bets.csv")
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

def combine_predictions(input_csv: str, output_csv: str = None) -> Path:
    """
    Combine cross-league predictions by averaging duplicates.
    
    Args:
        input_csv: Path to weekly_bets.csv
        output_csv: Optional output path (default: combined_predictions.csv)
    
    Returns:
        Path to combined predictions file
    """
    
    print("="*70)
    print(" CROSS-LEAGUE PREDICTION COMBINER")
    print("="*70)
    
    # Load predictions
    input_path = Path(input_csv)
    if not input_path.exists():
        print(f" Error: File not found: {input_path}")
        return None
    
    df = pd.read_csv(input_path)
    print(f"\n Loaded: {input_path}")
    print(f"   Total predictions: {len(df)}")
    
    # Identify duplicates (same Date, HomeTeam, AwayTeam)
    df['_match_key'] = df['Date'].astype(str) + '|' + df['HomeTeam'].astype(str) + '|' + df['AwayTeam'].astype(str)
    duplicate_mask = df.duplicated(subset=['_match_key'], keep=False)
    
    num_duplicates = duplicate_mask.sum()
    num_unique_matches = df[duplicate_mask]['_match_key'].nunique()
    
    print(f"\n Found {num_duplicates} rows from {num_unique_matches} cross-league matches")
    
    if num_unique_matches == 0:
        print("\n No cross-league matches found - all predictions are single-league")
        print(f"   Output: {input_path} (unchanged)")
        return input_path
    
    # Show which matches will be combined
    print("\n Cross-league matches to combine:")
    for match_key in df[duplicate_mask]['_match_key'].unique():
        match_rows = df[df['_match_key'] == match_key]
        leagues = match_rows['League'].tolist()
        home = match_rows['HomeTeam'].iloc[0]
        away = match_rows['AwayTeam'].iloc[0]
        date = match_rows['Date'].iloc[0]
        print(f"   {date}: {home} vs {away} ({' + '.join(leagues)})")
    
    # Combine predictions
    combined_rows = []
    processed_keys = set()
    
    for idx, row in df.iterrows():
        match_key = row['_match_key']
        
        # If already processed this match, skip
        if match_key in processed_keys:
            continue
        
        # Get all rows for this match
        match_rows = df[df['_match_key'] == match_key]
        
        if len(match_rows) == 1:
            # Single prediction - keep as-is
            combined_rows.append(row.to_dict())
        else:
            # Multiple predictions - average them
            avg_row = {}
            
            # Copy non-numeric columns from first row
            for col in ['Date', 'Time', 'HomeTeam', 'AwayTeam', 'Referee']:
                if col in match_rows.columns:
                    avg_row[col] = match_rows[col].iloc[0]
            
            # Mark as combined
            leagues = match_rows['League'].tolist()
            avg_row['League'] = f"COMBINED({'+'.join(leagues)})"
            
            # Average all numeric columns (predictions, probabilities, etc.)
            numeric_cols = match_rows.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                avg_row[col] = match_rows[col].mean()
            
            combined_rows.append(avg_row)
        
        processed_keys.add(match_key)
    
    # Create combined DataFrame
    combined_df = pd.DataFrame(combined_rows)
    
    # Remove temporary column
    if '_match_key' in combined_df.columns:
        combined_df = combined_df.drop(columns=['_match_key'])
    
    # Sort by date
    if 'Date' in combined_df.columns:
        combined_df['_date_sort'] = pd.to_datetime(combined_df['Date'], format='%d/%m/%Y', errors='coerce')
        combined_df = combined_df.sort_values('_date_sort').drop(columns=['_date_sort'])
    
    # Save combined predictions
    if output_csv is None:
        output_path = input_path.parent / "combined_predictions.csv"
    else:
        output_path = Path(output_csv)
    
    combined_df.to_csv(output_path, index=False)
    
    # Summary
    print(f"\n Combined predictions saved: {output_path}")
    print(f"   Original rows: {len(df)}")
    print(f"   Combined rows: {len(combined_df)}")
    print(f"   Reduced by: {len(df) - len(combined_df)} rows")
    
    # Show example of combined prediction
    if num_unique_matches > 0:
        print("\n Example combined prediction:")
        combined_match = combined_df[combined_df['League'].str.contains('COMBINED', na=False)].iloc[0]
        print(f"   Match: {combined_match.get('HomeTeam', 'N/A')} vs {combined_match.get('AwayTeam', 'N/A')}")
        print(f"   League: {combined_match.get('League', 'N/A')}")
        
        if 'P_1X2_H' in combined_match:
            print(f"   Home Win: {combined_match['P_1X2_H']:.1%}")
            print(f"   Draw: {combined_match['P_1X2_D']:.1%}")
            print(f"   Away Win: {combined_match['P_1X2_A']:.1%}")
    
    print("\n" + "="*70)
    return output_path


def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Combine cross-league match predictions by averaging"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/weekly_bets_lite.csv",
        help="Input CSV file (default: outputs/weekly_bets_lite.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file (default: combined_predictions.csv)"
    )
    
    args = parser.parse_args()
    
    result = combine_predictions(args.input, args.output)
    
    if result:
        print(f"\n Success! Use this file for your bets:")
        print(f"   {result}")
        return 0
    else:
        print("\n Failed to combine predictions")
        return 1


if __name__ == "__main__":
    sys.exit(main())
