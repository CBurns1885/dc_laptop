#!/usr/bin/env python3
"""
BET FINDER - DC-ONLY: BTTS and Over/Under Markets
Automatically find high-quality betting opportunities using Dixon-Coles model
Filters weekly_bets.csv based on:
- Probability thresholds
- Confidence scores
- Dixon-Coles validation
ONLY BTTS AND OVER/UNDER (0.5-5.5) MARKETS
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_get(row, column, default=0.0):
    """
    Safely get value from row, handling NaN and missing columns

    Args:
        row: pandas Series (row from DataFrame)
        column: column name
        default: default value if missing or NaN

    Returns:
        float: value or default if NaN/missing
    """
    try:
        value = row.get(column, default)
        # Check if value is NaN
        if pd.isna(value):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

# ============================================================================
# CONFIGURATION
# ============================================================================
today = datetime.now().strftime("%Y-%m-%d")
OUTPUT_DIR = Path(f"outputs/{today}")
WEEKLY_BETS_FILE = OUTPUT_DIR / "weekly_bets_lite.csv"

# Betting Criteria - DC-only BTTS and O/U
CONFIG = {
    # Probability thresholds (minimum confidence in prediction)
    'min_prob_btts': 0.65,      # 65% for both teams score
    'min_prob_ou': 0.65,        # 65% for over/under (all lines)

    # Confidence thresholds (model certainty)
    'min_confidence': 0.60,     # 60% confidence minimum

    # Agreement thresholds (model consensus)
    'min_agreement': 70.0,      # 70% agreement between models

    # Dixon-Coles validation
    'dc_validation': True,      # Check DC model agrees
    'dc_max_diff': 0.15,        # DC can differ by max 15% from main model

    # Value betting
    'calculate_value': True,    # Calculate implied odds and value
    'min_value_threshold': 0.05, # 5% minimum edge for value bets
}

# ============================================================================
# BET FINDER CLASS
# ============================================================================

class BetFinder:
    def __init__(self, config=CONFIG):
        self.config = config
        self.df = None
        self.quality_bets = []

    def load_data(self, filepath=WEEKLY_BETS_FILE):
        """Load weekly_bets.csv"""
        print(f"Loading {filepath}...")

        if not filepath.exists():
            print(f"❌ Error: {filepath} not found!")
            print(f"   Make sure you've run the weekly pipeline first.")
            return False

        self.df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(self.df)} matches")
        print(f"   Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
        return True

    # ========================================================================
    # BTTS MARKETS
    # ========================================================================

    def check_btts_yes(self, row):
        """Check if BTTS Yes meets criteria"""
        prob = safe_get(row, 'P_BTTS_Y', 0.0)
        conf = safe_get(row, 'CONF_BTTS_Y', 0.0)
        agree = safe_get(row, 'AGREE_BTTS_Y', 0.0)
        dc_prob = safe_get(row, 'DC_BTTS_Y', 0.0)

        if prob < self.config['min_prob_btts']:
            return None
        if conf < self.config['min_confidence']:
            return None
        if agree < self.config['min_agreement']:
            return None

        if self.config['dc_validation'] and dc_prob > 0:
            if abs(prob - dc_prob) > self.config['dc_max_diff']:
                return None

        # Logic check: High BTTS should correlate with higher goal totals
        ou25_over = safe_get(row, 'P_OU_2_5_O', 0.0)
        if prob > 0.7 and ou25_over < 0.5:
            return None

        return {
            'Market': 'BTTS',
            'Selection': 'Yes',
            'Probability': prob,
            'Confidence': conf,
            'Agreement': agree,
            'DC_Probability': dc_prob if dc_prob > 0 else None,
            'ImpliedOdds': round(1 / prob, 2) if prob > 0 else None,
        }

    def check_btts_no(self, row):
        """Check if BTTS No meets criteria"""
        prob = safe_get(row, 'P_BTTS_N', 0.0)
        conf = safe_get(row, 'CONF_BTTS_N', 0.0)
        agree = safe_get(row, 'AGREE_BTTS_N', 0.0)
        dc_prob = safe_get(row, 'DC_BTTS_N', 0.0)

        if prob < self.config['min_prob_btts']:
            return None
        if conf < self.config['min_confidence']:
            return None
        if agree < self.config['min_agreement']:
            return None

        if self.config['dc_validation'] and dc_prob > 0:
            if abs(prob - dc_prob) > self.config['dc_max_diff']:
                return None

        return {
            'Market': 'BTTS',
            'Selection': 'No',
            'Probability': prob,
            'Confidence': conf,
            'Agreement': agree,
            'DC_Probability': dc_prob if dc_prob > 0 else None,
            'ImpliedOdds': round(1 / prob, 2) if prob > 0 else None,
        }

    # ========================================================================
    # OVER/UNDER MARKETS (0.5-5.5 Lines)
    # ========================================================================

    def check_ou_market(self, row, line, over_under):
        """Generic checker for any O/U line (0.5-5.5)"""
        ou_type = 'O' if over_under == 'Over' else 'U'
        prob = safe_get(row, f'P_OU_{line}_{ou_type}', 0.0)

        # Confidence and agreement
        conf_col = f'CONF_OU_{line}_{ou_type}'
        agree_col = f'AGREE_OU_{line}_{ou_type}'
        conf = safe_get(row, conf_col, prob)  # Use prob as fallback
        agree = safe_get(row, agree_col, prob * 100)  # Use prob as fallback

        if prob < self.config['min_prob_ou']:
            return None

        if conf_col in row.index:
            if conf < self.config['min_confidence']:
                return None
            if agree < self.config['min_agreement']:
                return None

        dc_prob = safe_get(row, f'DC_OU_{line}_{ou_type}', 0.0)
        if self.config['dc_validation'] and dc_prob > 0:
            if abs(prob - dc_prob) > self.config['dc_max_diff']:
                return None

        return {
            'Market': f'O/U {line.replace("_", ".")}',
            'Selection': over_under,
            'Probability': prob,
            'Confidence': conf,
            'Agreement': agree,
            'DC_Probability': dc_prob if dc_prob > 0 else None,
            'ImpliedOdds': round(1 / prob, 2) if prob > 0 else None,
        }

    # ========================================================================
    # FIND QUALITY BETS
    # ========================================================================

    def find_quality_bets(self):
        """Scan all matches and find quality betting opportunities"""
        print("\n" + "="*60)
        print("🔍 SCANNING FOR QUALITY BETS (BTTS & O/U 0.5-5.5)")
        print("="*60)

        self.quality_bets = []

        for idx, row in self.df.iterrows():
            match_info = {
                'League': row['League'],
                'Date': row['Date'],
                'HomeTeam': row['HomeTeam'],
                'AwayTeam': row['AwayTeam'],
            }

            checkers = []

            # BTTS Markets
            checkers.extend([
                self.check_btts_yes,
                self.check_btts_no,
            ])

            # Over/Under Markets (0.5-5.5)
            for line in ['0_5', '1_5', '2_5', '3_5', '4_5', '5_5']:
                checkers.append(lambda r, l=line: self.check_ou_market(r, l, 'Over'))
                checkers.append(lambda r, l=line: self.check_ou_market(r, l, 'Under'))

            # Run all checkers
            for checker in checkers:
                try:
                    bet = checker(row)
                    if bet:
                        bet.update(match_info)
                        self.quality_bets.append(bet)
                except Exception as e:
                    # Silently skip any checker errors
                    pass

        print(f"✅ Found {len(self.quality_bets)} quality betting opportunities")
        return self.quality_bets

    def calculate_kelly_stake(self, probability, odds):
        """Calculate Kelly Criterion stake percentage"""
        if odds <= 1.0:
            return 0.0

        kelly = (probability * odds - 1) / (odds - 1)
        return max(0, kelly * 0.25)  # Quarter Kelly for safety

    def generate_report(self):
        """Generate detailed betting report"""
        if not self.quality_bets:
            print("\n❌ No quality bets found with current criteria.")
            print("💡 Try adjusting the CONFIG thresholds to be less strict.")
            return

        # Convert to DataFrame
        bets_df = pd.DataFrame(self.quality_bets)

        # Sort by confidence and probability
        bets_df['Score'] = (bets_df['Probability'] * 0.5 +
                            bets_df['Confidence'] * 0.3 +
                            bets_df['Agreement'] / 100 * 0.2)
        bets_df = bets_df.sort_values('Score', ascending=False)

        # Add Kelly stakes
        if self.config['calculate_value']:
            bets_df['Kelly%'] = bets_df.apply(
                lambda x: self.calculate_kelly_stake(x['Probability'], x['ImpliedOdds']),
                axis=1
            )

        # Sort by Date first for better readability
        if 'Date' in bets_df.columns:
            bets_df['Date'] = pd.to_datetime(bets_df['Date'], errors='coerce')
            bets_df = bets_df.sort_values(['Date', 'Score'], ascending=[True, False])
            print("✅ Sorted quality bets by Date, then by Score")

        # Display summary
        print("\n" + "="*60)
        print("📊 BETTING SUMMARY (DC-ONLY: BTTS & O/U)")
        print("="*60)
        print(f"Total quality bets: {len(bets_df)}")
        print(f"Matches covered: {bets_df[['HomeTeam', 'AwayTeam']].drop_duplicates().shape[0]}")
        print(f"\nBreakdown by market:")
        print(bets_df['Market'].value_counts().to_string())

        # Save to CSV
        output_file = OUTPUT_DIR / f"quality_bets_dc_{datetime.now().strftime('%Y%m%d')}.csv"
        bets_df.to_csv(output_file, index=False)
        print(f"\n💾 Saved to: {output_file}")

        # Save to HTML
        html_file = OUTPUT_DIR / f"quality_bets_dc_{datetime.now().strftime('%Y%m%d')}.html"
        self.generate_html_report(bets_df, html_file)
        print(f"💾 Saved HTML: {html_file}")

        # Display top 20
        print("\n" + "="*60)
        print("🏆 TOP 20 QUALITY BETS (DIXON-COLES)")
        print("="*60)

        display_cols = ['Date', 'HomeTeam', 'AwayTeam', 'Market', 'Selection',
                       'Probability', 'Confidence', 'Agreement', 'ImpliedOdds']
        if 'Kelly%' in bets_df.columns:
            display_cols.append('Kelly%')

        top20 = bets_df.head(20)[display_cols]

        # Format for display
        top20_display = top20.copy()
        top20_display['Probability'] = top20_display['Probability'].apply(lambda x: f"{x:.1%}")
        top20_display['Confidence'] = top20_display['Confidence'].apply(lambda x: f"{x:.1%}")
        top20_display['Agreement'] = top20_display['Agreement'].apply(lambda x: f"{x:.0f}%")
        if 'Kelly%' in top20_display.columns:
            top20_display['Kelly%'] = top20_display['Kelly%'].apply(lambda x: f"{x:.1%}")

        print(top20_display.to_string(index=False))

        return bets_df

    def generate_html_report(self, df, output_path):
        """Generate beautiful HTML report"""

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Dixon-Coles Quality Bets - {datetime.now().strftime('%Y-%m-%d')}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #667eea;
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-card h3 {{
            margin: 0;
            font-size: 2em;
        }}
        .stat-card p {{
            margin: 5px 0 0 0;
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 0.9em;
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #eee;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .prob-high {{ color: #10b981; font-weight: bold; }}
        .prob-medium {{ color: #f59e0b; font-weight: bold; }}
        .conf-high {{ background: #d1fae5; color: #065f46; padding: 4px 8px; border-radius: 4px; }}
        .conf-medium {{ background: #fef3c7; color: #92400e; padding: 4px 8px; border-radius: 4px; }}
        .market {{
            background: #f3f4f6;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.85em;
        }}
        .odds {{
            background: #dbeafe;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
        }}
        .kelly {{
            background: #fce7f3;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #eee;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>⚽ Dixon-Coles Quality Bets: BTTS & O/U (0.5-5.5)</h1>
        <p class="subtitle">Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>

        <div class="stats">
            <div class="stat-card">
                <h3>{len(df)}</h3>
                <p>Quality Bets Found</p>
            </div>
            <div class="stat-card">
                <h3>{df[['HomeTeam', 'AwayTeam']].drop_duplicates().shape[0]}</h3>
                <p>Matches Covered</p>
            </div>
            <div class="stat-card">
                <h3>{df['Probability'].mean():.0%}</h3>
                <p>Avg Probability</p>
            </div>
            <div class="stat-card">
                <h3>{len(df['Market'].unique())}</h3>
                <p>Different Markets</p>
            </div>
        </div>

        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Match</th>
                    <th>League</th>
                    <th>Market</th>
                    <th>Selection</th>
                    <th>Probability</th>
                    <th>Confidence</th>
                    <th>Agreement</th>
                    <th>Implied Odds</th>
                    {'<th>Kelly %</th>' if 'Kelly%' in df.columns else ''}
                </tr>
            </thead>
            <tbody>
"""

        for _, row in df.iterrows():
            prob_class = 'prob-high' if row['Probability'] > 0.70 else 'prob-medium'
            conf_class = 'conf-high' if row['Confidence'] > 0.70 else 'conf-medium'

            kelly_cell = f"<td><span class='kelly'>{row['Kelly%']:.1%}</span></td>" if 'Kelly%' in df.columns else ''

            html += f"""
                <tr>
                    <td>{row['Date']}</td>
                    <td><strong>{row['HomeTeam']}</strong> vs <strong>{row['AwayTeam']}</strong></td>
                    <td>{row['League']}</td>
                    <td><span class="market">{row['Market']}</span></td>
                    <td><strong>{row['Selection']}</strong></td>
                    <td class="{prob_class}">{row['Probability']:.1%}</td>
                    <td><span class="{conf_class}">{row['Confidence']:.1%}</span></td>
                    <td>{row['Agreement']:.0f}%</td>
                    <td><span class="odds">{row['ImpliedOdds']:.2f}</span></td>
                    {kelly_cell}
                </tr>
            """

        html += """
            </tbody>
        </table>

        <div class="footer">
            <p><strong>⚠️ Betting Disclaimer:</strong> These are predictions based on Dixon-Coles statistical model.
            Past performance does not guarantee future results. Bet responsibly.</p>
            <p><strong>💡 How to use:</strong> Look for bets with high probability, high confidence,
            and high agreement. Compare implied odds with bookmaker odds to find value.</p>
            <p><strong>📊 Dixon-Coles Markets:</strong> BTTS (Both Teams To Score) and Over/Under goal lines (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)</p>
        </div>
    </div>
</body>
</html>
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

    def run(self):
        """Main execution flow"""
        print("="*60)
        print("⚽ BET FINDER - Dixon-Coles: BTTS & O/U (0.5-5.5)")
        print("="*60)

        if not self.load_data():
            return

        self.find_quality_bets()

        if self.quality_bets:
            self.generate_report()

            print("\n" + "="*60)
            print("✅ BET FINDER COMPLETE")
            print("="*60)
            print("\n💡 Next steps:")
            print("   1. Review the HTML report in your browser")
            print("   2. Compare implied odds with bookmaker odds")
            print("   3. Look for value bets (bookmaker odds > implied odds)")
            print("   4. Use Kelly % for stake sizing (bet conservatively!)")
            print("   5. Track your bets in a spreadsheet")
        else:
            print("\n💡 Tips to find more bets:")
            print("   - Lower min_prob thresholds in CONFIG")
            print("   - Lower min_confidence to 0.55")
            print("   - Lower min_agreement to 65")
            print("   - Set dc_validation to False")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    finder = BetFinder(config=CONFIG)
    finder.run()
