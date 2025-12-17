#!/usr/bin/env python3
"""
Excel Output Generator - Multi-sheet weekly predictions
Creates structured Excel workbook with:
- Sheet 1: Top 10 from each market
- Sheets 2-8: All predictions per market (BTTS, OU 0.5, OU 1.5, etc.)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List

def generate_excel_report(predictions_csv: Path, output_dir: Path = None) -> Path:
    """
    Generate multi-sheet Excel workbook from predictions

    Args:
        predictions_csv: Path to weekly_bets.csv with all predictions
        output_dir: Output directory (defaults to same as input)

    Returns:
        Path to generated Excel file
    """
    if output_dir is None:
        output_dir = predictions_csv.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    df = pd.read_csv(predictions_csv)

    print("\n Generating Multi-Sheet Excel Report")
    print("="*60)

    # Create Excel writer
    date_str = datetime.now().strftime('%Y-%m-%d')
    excel_path = output_dir / f"predictions_{date_str}.xlsx"

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

        # ==================================================================
        # SHEET 1: Top 10 from each market
        # ==================================================================
        print(" Sheet 1: Top 10 from each market...")

        # Define markets to extract
        markets = {
            'BTTS': ['P_BTTS_Y', 'P_BTTS_N'],
            'OU 0.5': ['P_OU_0_5_O', 'P_OU_0_5_U'],
            'OU 1.5': ['P_OU_1_5_O', 'P_OU_1_5_U'],
            'OU 2.5': ['P_OU_2_5_O', 'P_OU_2_5_U'],
            'OU 3.5': ['P_OU_3_5_O', 'P_OU_3_5_U'],
            'OU 4.5': ['P_OU_4_5_O', 'P_OU_4_5_U'],
            'OU 5.5': ['P_OU_5_5_O', 'P_OU_5_5_U'],
        }

        top10_rows = []

        for market_name, prob_cols in markets.items():
            # Find columns that exist in the dataframe
            available_cols = [c for c in prob_cols if c in df.columns]
            if not available_cols:
                continue

            # Get best probability for this market
            df_market = df.copy()
            df_market['Market'] = market_name
            df_market['BestProb'] = df_market[available_cols].max(axis=1)
            df_market['BestSelection'] = df_market[available_cols].idxmax(axis=1)

            # Map column names to readable selections
            selection_map = {}
            for col in available_cols:
                if 'BTTS_Y' in col:
                    selection_map[col] = 'Yes'
                elif 'BTTS_N' in col:
                    selection_map[col] = 'No'
                elif col.endswith('_O'):
                    selection_map[col] = 'Over'
                elif col.endswith('_U'):
                    selection_map[col] = 'Under'

            df_market['Selection'] = df_market['BestSelection'].map(selection_map)

            # Get top 10 for this market
            top10 = df_market.nlargest(10, 'BestProb')

            # Select and rename columns
            top10_clean = top10[[
                'Date', 'League', 'HomeTeam', 'AwayTeam',
                'Market', 'Selection', 'BestProb'
            ]].copy()
            top10_clean.rename(columns={'BestProb': 'Probability'}, inplace=True)
            top10_clean['Probability'] = top10_clean['Probability'].apply(lambda x: f"{x:.1%}")

            top10_rows.append(top10_clean)

        # Combine all top 10s
        if top10_rows:
            top10_df = pd.concat(top10_rows, ignore_index=True)
            top10_df.to_excel(writer, sheet_name='Top 10 All Markets', index=False)
            print(f"    Added {len(top10_df)} top predictions")

        # ==================================================================
        # SHEETS 2-8: Individual market sheets
        # ==================================================================
        print("\n Individual market sheets...")

        for market_name, prob_cols in markets.items():
            available_cols = [c for c in prob_cols if c in df.columns]
            if not available_cols:
                continue

            # Create market-specific dataframe
            df_market = df.copy()

            # Calculate probabilities for each selection
            market_data = []
            for idx, row in df_market.iterrows():
                for col in available_cols:
                    # Parse selection
                    if 'BTTS_Y' in col:
                        selection = 'Yes'
                    elif 'BTTS_N' in col:
                        selection = 'No'
                    elif col.endswith('_O'):
                        selection = 'Over'
                    elif col.endswith('_U'):
                        selection = 'Under'
                    else:
                        selection = col

                    market_data.append({
                        'Date': row['Date'],
                        'League': row['League'],
                        'HomeTeam': row['HomeTeam'],
                        'AwayTeam': row['AwayTeam'],
                        'Selection': selection,
                        'Probability': row[col],
                    })

            market_df = pd.DataFrame(market_data)

            # Sort by probability descending
            market_df = market_df.sort_values('Probability', ascending=False)

            # Format probability as percentage
            market_df['Probability'] = market_df['Probability'].apply(lambda x: f"{x:.1%}")

            # Write to sheet (sheet name max 31 chars)
            sheet_name = market_name[:31]
            market_df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"    {market_name}: {len(market_df)} predictions")

        # ==================================================================
        # Format sheets
        # ==================================================================
        print("\n Formatting sheets...")

        # Auto-adjust column widths
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

            # Freeze top row
            worksheet.freeze_panes = 'A2'

    print(f"\n Excel report saved: {excel_path}")
    print(f"    {len(writer.sheets)} sheets created")
    return excel_path


if __name__ == "__main__":
    from config import OUTPUT_DIR

    # Test with weekly_bets.csv
    input_csv = OUTPUT_DIR / "weekly_bets.csv"

    if input_csv.exists():
        excel_file = generate_excel_report(input_csv)
        print(f"\n Report ready: {excel_file}")
    else:
        print(f" Input file not found: {input_csv}")
        print("   Run predict.py first to generate predictions")
