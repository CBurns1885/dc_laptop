# generate_outputs_from_actual.py
"""
Generate outputs using DC (Dixon-Coles) predictions only
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_high_confidence_bets(csv_path, threshold=0.90):
    """Extract high confidence bets using DC columns"""

    df = pd.read_csv(csv_path)

    print(f"\n Analyzing {len(df)} predictions...")

    high_conf_bets = []

    # Markets using DC predictions
    markets = {
        '1X2': {'probs': ['DC_1X2_H', 'DC_1X2_D', 'DC_1X2_A'],
                'outcomes': ['H', 'D', 'A']},
        'BTTS': {'probs': ['DC_BTTS_Y', 'DC_BTTS_N'],
                 'outcomes': ['Yes', 'No']},
        'OU_1_5': {'probs': ['DC_OU_1_5_O', 'DC_OU_1_5_U'],
                   'outcomes': ['Over', 'Under']},
        'OU_2_5': {'probs': ['DC_OU_2_5_O', 'DC_OU_2_5_U'],
                   'outcomes': ['Over', 'Under']},
        'OU_3_5': {'probs': ['DC_OU_3_5_O', 'DC_OU_3_5_U'],
                   'outcomes': ['Over', 'Under']},
        'OU_4_5': {'probs': ['DC_OU_4_5_O', 'DC_OU_4_5_U'],
                   'outcomes': ['Over', 'Under']},
    }

    for idx, row in df.iterrows():
        match_info = {
            'Date': row['Date'],
            'League': row.get('League', 'Unknown'),
            'HomeTeam': row['HomeTeam'],
            'AwayTeam': row['AwayTeam']
        }

        for market, config in markets.items():
            # Get DC probabilities
            probs = []
            outcomes = []

            for prob_col, outcome in zip(config['probs'], config['outcomes']):
                if prob_col in df.columns and pd.notna(row[prob_col]):
                    probs.append(row[prob_col])
                    outcomes.append(outcome)

            if not probs:
                continue

            # Find max probability
            max_idx = np.argmax(probs)
            max_prob = probs[max_idx]

            if max_prob >= threshold:
                bet = {
                    **match_info,
                    'Market': market,
                    'Prediction': outcomes[max_idx],
                    'Confidence': round(max_prob * 100, 1)
                }
                high_conf_bets.append(bet)

    if not high_conf_bets:
        print(f" No bets over {threshold*100:.0f}% confidence")
        return None

    bets_df = pd.DataFrame(high_conf_bets)
    bets_df = bets_df.sort_values('Confidence', ascending=False)

    # Save
    output_dir = Path("outputs")
    csv_out = output_dir / "high_confidence_bets.csv"
    html_out = output_dir / "high_confidence_bets.html"

    bets_df.to_csv(csv_out, index=False)

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>High Confidence Bets ({threshold*100:.0f}%+)</title>
    <style>
        body {{ font-family: Arial; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #2c3e50; }}
        .summary {{ background: white; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; background: white; margin-top: 20px; }}
        th {{ background: #2c3e50; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        .conf-95 {{ background: #d4edda; font-weight: bold; }}
        .conf-90 {{ background: #fff3cd; }}
        tr:hover {{ background: #f8f9fa; }}
    </style>
</head>
<body>
    <h1>High Confidence Bets ({threshold*100:.0f}%+)</h1>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Bets:</strong> {len(bets_df)}</p>
        <p><strong>Confidence Range:</strong> {bets_df['Confidence'].min():.1f}% - {bets_df['Confidence'].max():.1f}%</p>
        <p><strong>Markets:</strong> {', '.join(bets_df['Market'].unique())}</p>
    </div>

    <table>
        <tr>
            <th>Date</th>
            <th>League</th>
            <th>Match</th>
            <th>Market</th>
            <th>Prediction</th>
            <th>Confidence</th>
        </tr>
"""

    for _, row in bets_df.iterrows():
        conf_class = 'conf-95' if row['Confidence'] >= 95 else 'conf-90'
        html += f"""
        <tr class="{conf_class}">
            <td>{row['Date']}</td>
            <td>{row['League']}</td>
            <td>{row['HomeTeam']} vs {row['AwayTeam']}</td>
            <td><strong>{row['Market']}</strong></td>
            <td><strong>{row['Prediction']}</strong></td>
            <td><strong>{row['Confidence']:.1f}%</strong></td>
        </tr>
"""

    html += """
    </table>

    <div style="margin-top: 20px; padding: 15px; background: white; border-radius: 5px;">
        <h3>Color Guide</h3>
        <p><span style="background: #d4edda; padding: 5px;">Green = 95%+ confidence</span></p>
        <p><span style="background: #fff3cd; padding: 5px;">Yellow = 90-94% confidence</span></p>
    </div>
</body>
</html>
"""

    with open(html_out, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n > Generated {len(bets_df)} high-confidence bets")
    print(f"   Range: {bets_df['Confidence'].min():.1f}% - {bets_df['Confidence'].max():.1f}%")
    print(f"   Markets: {bets_df['Market'].nunique()} unique")
    print(f"   Saved: {csv_out.name}, {html_out.name}")

    return bets_df

def generate_ou_accumulators(csv_path):
    """Generate O/U accumulators using DC predictions"""

    df = pd.read_csv(csv_path)

    print(f"\n Analyzing O/U predictions...")

    # Check if this is already a simplified format (high_confidence_bets.csv)
    # or the full weekly_bets_lite.csv format
    if 'Market' in df.columns and 'Prediction' in df.columns and 'Confidence' in df.columns:
        # Already in simplified format (high_confidence_bets.csv)
        print(f" Using high confidence bets format")

        # Filter for O/U bets only (lines 1.5, 3.5, 4.5)
        ou_markets = ['OU_1_5', 'OU_3_5', 'OU_4_5']
        bets_df = df[df['Market'].isin(ou_markets)].copy()

        # Add Match column
        bets_df['Match'] = bets_df['HomeTeam'] + ' vs ' + bets_df['AwayTeam']

        # Add Time column if not present
        if 'Time' not in bets_df.columns:
            bets_df['Time'] = ''

        # Convert confidence to 0-1 range if it's in percentage (0-100)
        if bets_df['Confidence'].max() > 1:
            bets_df['Confidence'] = bets_df['Confidence'] / 100

    else:
        # Full format (weekly_bets_lite.csv) - extract O/U bets from DC columns
        print(f" Using weekly bets lite format")
        ou_bets = []

        for line in ['1_5', '3_5', '4_5']:
            prob_o = f'DC_OU_{line}_O'
            prob_u = f'DC_OU_{line}_U'

            if prob_o not in df.columns:
                continue

            for idx, row in df.iterrows():
                if pd.notna(row[prob_o]):
                    prob_over = row[prob_o]
                    prob_under = row[prob_u]

                    # Get time if available
                    time_str = row.get('Time', '')
                    if pd.isna(time_str):
                        time_str = ''

                    # Take whichever is higher
                    if prob_over > prob_under:
                        ou_bets.append({
                            'idx': idx,
                            'Date': row['Date'],
                            'Time': time_str,
                            'League': row.get('League', 'Unknown'),
                            'HomeTeam': row['HomeTeam'],
                            'AwayTeam': row['AwayTeam'],
                            'Market': f'OU_{line}',
                            'Prediction': 'Over',
                            'Confidence': prob_over,
                            'Match': f"{row['HomeTeam']} vs {row['AwayTeam']}"
                        })
                    else:
                        ou_bets.append({
                            'idx': idx,
                            'Date': row['Date'],
                            'Time': time_str,
                            'League': row.get('League', 'Unknown'),
                            'HomeTeam': row['HomeTeam'],
                            'AwayTeam': row['AwayTeam'],
                            'Market': f'OU_{line}',
                            'Prediction': 'Under',
                            'Confidence': prob_under,
                            'Match': f"{row['HomeTeam']} vs {row['AwayTeam']}"
                        })

        if not ou_bets:
            print(" No O/U bets found")
            return None

        bets_df = pd.DataFrame(ou_bets)

    # Filter for 90%+ confidence only
    bets_df = bets_df[bets_df['Confidence'] >= 0.90].copy()

    if len(bets_df) == 0:
        print(" No O/U bets found with 90%+ confidence")
        return None

    # Remove duplicates: Keep only the BEST bet per match (highest confidence)
    print(f" Found {len(bets_df)} O/U bets before deduplication")
    bets_df = bets_df.sort_values('Confidence', ascending=False)
    bets_df = bets_df.drop_duplicates(subset='Match', keep='first')

    # Sort by Date and Time for chronological grouping
    bets_df['DateTime'] = pd.to_datetime(bets_df['Date'] + ' ' + bets_df['Time'].fillna(''),
                                          errors='coerce', format='mixed')
    bets_df = bets_df.sort_values('DateTime')

    print(f" Found {len(bets_df)} unique matches with 90%+ confidence")
    print(f"   Confidence range: {bets_df['Confidence'].min()*100:.1f}% - {bets_df['Confidence'].max()*100:.1f}%")

    # Generate accumulators - make as many 4-folds as possible
    accumulators = []

    print(f"\n Building 4-fold accumulators (sorted by date/time)...")

    # Create accumulators from consecutive groups of 4 matches
    num_accumulators = len(bets_df) // 4

    for i in range(num_accumulators):
        start_idx = i * 4
        end_idx = start_idx + 4
        combo_bets = bets_df.iloc[start_idx:end_idx]

        # Double-check we have exactly 4 unique matches
        if len(combo_bets) != 4 or combo_bets['Match'].nunique() != 4:
            continue

        combined_prob = combo_bets['Confidence'].prod()
        combined_conf = combo_bets['Confidence'].min()
        implied_odds = 1 / combined_prob if combined_prob > 0 else 1.0

        accumulators.append({
            'Fold': '4-Fold',
            'Combined_Confidence': combined_conf,
            'Combined_Probability': combined_prob,
            'Implied_Odds': implied_odds,
            'Legs': len(combo_bets),
            'Bets': combo_bets.to_dict('records')
        })

    # Handle remaining bets (less than 4) - create smaller accumulators
    remaining_start = num_accumulators * 4
    remaining_bets = bets_df.iloc[remaining_start:]

    if len(remaining_bets) >= 2:  # Create 2-fold or 3-fold with remaining
        combined_prob = remaining_bets['Confidence'].prod()
        combined_conf = remaining_bets['Confidence'].min()
        implied_odds = 1 / combined_prob if combined_prob > 0 else 1.0

        fold_type = f"{len(remaining_bets)}-Fold"
        accumulators.append({
            'Fold': fold_type,
            'Combined_Confidence': combined_conf,
            'Combined_Probability': combined_prob,
            'Implied_Odds': implied_odds,
            'Legs': len(remaining_bets),
            'Bets': remaining_bets.to_dict('records')
        })

    # Save CSV
    csv_rows = []
    for acca in accumulators:
        legs_text = " | ".join([
            f"{bet['Market']} {bet['Prediction']} ({bet['HomeTeam']} vs {bet['AwayTeam']})"
            for bet in acca['Bets']
        ])
        csv_rows.append({
            'Fold': acca['Fold'],
            'Combined_Confidence_%': round(acca['Combined_Confidence'] * 100, 1),
            'Combined_Probability_%': round(acca['Combined_Probability'] * 100, 1),
            'Implied_Odds': round(acca['Implied_Odds'], 2),
            'Legs': acca['Legs'],
            'Selections': legs_text
        })

    output_dir = Path("outputs")
    csv_out = output_dir / "ou_accumulators.csv"
    html_out = output_dir / "ou_accumulators.html"

    pd.DataFrame(csv_rows).to_csv(csv_out, index=False)

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>O/U Accumulators (DC Model)</title>
    <style>
        body {{ font-family: Arial; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #2c3e50; }}
        .summary {{ background: white; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .acca {{ background: white; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #3498db; }}
        .acca-header {{ display: flex; justify-content: space-between; margin-bottom: 10px; }}
        .acca-title {{ font-size: 18px; font-weight: bold; }}
        .acca-odds {{ font-size: 24px; color: #e67e22; font-weight: bold; }}
        .leg {{ padding: 8px; background: #ecf0f1; margin: 5px 0; border-radius: 3px; }}
        .prob-high {{ color: #27ae60; font-weight: bold; }}
        .prob-medium {{ color: #f39c12; }}
    </style>
</head>
<body>
    <h1>O/U Accumulators (90%+ Confidence)</h1>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Accumulators:</strong> {len(accumulators)}</p>
        <p><strong>Total Matches:</strong> {len(bets_df)}</p>
        <p><strong>Markets:</strong> O/U 1.5, O/U 3.5, O/U 4.5</p>
        <p><strong>Strategy:</strong> Maximize 4-fold accumulators, sorted chronologically by date/time</p>
        <p><strong>Minimum confidence per leg:</strong> 90%</p>
        <p><strong>Note:</strong> Each match appears only once (best O/U line selected)</p>
        <p><strong>Grouping:</strong> Matches grouped by date/time for better scheduling</p>
    </div>
"""

    for i, acca in enumerate(accumulators, 1):
        prob_pct = acca['Combined_Probability'] * 100
        conf_pct = acca['Combined_Confidence'] * 100
        prob_class = "prob-high" if prob_pct >= 70 else ("prob-medium" if prob_pct >= 50 else "")

        html += f"""
    <div class="acca">
        <div class="acca-header">
            <div class="acca-title">#{i}: {acca['Fold']}</div>
            <div class="acca-odds">@{acca['Implied_Odds']:.2f}</div>
        </div>
        <p><strong>Combined Probability:</strong> <span class="{prob_class}">{prob_pct:.1f}%</span> |
           <strong>Min Confidence:</strong> {conf_pct:.1f}% |
           <strong>Legs:</strong> {acca['Legs']}</p>
        <div class="legs">
"""

        for bet in acca['Bets']:
            # Format date and time for display
            date_str = bet.get('Date', '')
            time_str = bet.get('Time', '')
            datetime_display = f"{date_str} {time_str}".strip() if time_str else date_str

            html += f"""
            <div class="leg">
                <strong>{bet['Market']} {bet['Prediction']}</strong> - {bet['HomeTeam']} vs {bet['AwayTeam']}
                <br><small>{datetime_display} | {bet['League']}</small>
                <br>Confidence: {bet['Confidence']*100:.1f}%
            </div>
"""

        html += """
        </div>
    </div>
"""

    html += """
    <div style="margin-top: 20px; padding: 15px; background: white; border-radius: 5px;">
        <h3>Tips</h3>
        <ul>
            <li><strong>Green probability (70%+):</strong> High chance of winning</li>
            <li><strong>Yellow probability (50-70%):</strong> Medium risk</li>
            <li>Accumulators sorted by combined probability (most likely to win first)</li>
            <li>Min confidence shows the weakest leg</li>
        </ul>
    </div>
</body>
</html>
"""

    with open(html_out, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f" > Generated {len(accumulators)} accumulators")
    print(f"   Saved: {csv_out.name}, {html_out.name}")

    return accumulators

if __name__ == "__main__":
    csv_path = Path("outputs/weekly_bets_lite.csv")

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        print("Run GENERATE_NEW_OUTPUTS.bat to copy the file first")
        exit(1)

    print("="*70)
    print(" GENERATING OUTPUTS (DC MODEL ONLY)")
    print("="*70)

    # Generate high confidence bets
    high_conf = generate_high_confidence_bets(csv_path, threshold=0.90)

    # Generate accumulators
    accas = generate_ou_accumulators(csv_path)

    print("\n" + "="*70)
    print(" COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - outputs/high_confidence_bets.html")
    print("  - outputs/high_confidence_bets.csv")
    print("  - outputs/ou_accumulators.html")
    print("  - outputs/ou_accumulators.csv")
    print("\n" + "="*70)
