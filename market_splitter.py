# market_splitter.py
"""
Split predictions into separate files per market
Cleaner output structure - one file per market type
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List
from config import OUTPUT_DIR


class MarketSplitter:
    """Split weekly predictions into market-specific files"""

    def __init__(self, predictions_file: Path):
        self.predictions_file = predictions_file
        self.df = None

    def load_predictions(self) -> bool:
        """Load predictions CSV"""
        if not self.predictions_file.exists():
            print(f"❌ Predictions file not found: {self.predictions_file}")
            return False

        try:
            self.df = pd.read_csv(self.predictions_file)
            print(f"✓ Loaded {len(self.df)} predictions from {self.predictions_file.name}")
            return True
        except Exception as e:
            print(f"❌ Error loading predictions: {e}")
            return False

    def split_by_market(self) -> Dict[str, pd.DataFrame]:
        """
        Split predictions into separate DataFrames per market type

        Returns:
            Dictionary: {market_name: dataframe}
        """
        markets = {}

        # Define markets and their probability columns (using DC predictions)
        market_configs = {
            '1X2': {
                'prob_cols': ['DC_1X2_H', 'DC_1X2_D', 'DC_1X2_A'],
                'display_name': '1X2 (Match Result)'
            },
            'BTTS': {
                'prob_cols': ['DC_BTTS_Y', 'DC_BTTS_N'],
                'display_name': 'Both Teams To Score'
            },
            'OU_0_5': {
                'prob_cols': ['DC_OU_0_5_O', 'DC_OU_0_5_U'],
                'display_name': 'Over/Under 0.5 Goals'
            },
            'OU_1_5': {
                'prob_cols': ['DC_OU_1_5_O', 'DC_OU_1_5_U'],
                'display_name': 'Over/Under 1.5 Goals'
            },
            'OU_2_5': {
                'prob_cols': ['DC_OU_2_5_O', 'DC_OU_2_5_U'],
                'display_name': 'Over/Under 2.5 Goals'
            },
            'OU_3_5': {
                'prob_cols': ['DC_OU_3_5_O', 'DC_OU_3_5_U'],
                'display_name': 'Over/Under 3.5 Goals'
            },
            'OU_4_5': {
                'prob_cols': ['DC_OU_4_5_O', 'DC_OU_4_5_U'],
                'display_name': 'Over/Under 4.5 Goals'
            },
            'OU_5_5': {
                'prob_cols': ['DC_OU_5_5_O', 'DC_OU_5_5_U'],
                'display_name': 'Over/Under 5.5 Goals'
            },
        }

        # Base columns to include in all market files
        base_cols = ['Date', 'League', 'HomeTeam', 'AwayTeam', 'Time']

        for market_key, config in market_configs.items():
            prob_cols = config['prob_cols']

            # Check if these columns exist
            available_cols = [col for col in prob_cols if col in self.df.columns]

            if not available_cols:
                continue  # Skip market if no probability columns found

            # Select base columns + probability columns for this market
            cols_to_include = base_cols + available_cols

            # Only include columns that actually exist
            cols_to_include = [col for col in cols_to_include if col in self.df.columns]

            # Create market-specific DataFrame
            market_df = self.df[cols_to_include].copy()

            # Rename DC_ columns to P_ for consistency and CSV output
            rename_map = {col: col.replace('DC_', 'P_') for col in available_cols}
            market_df = market_df.rename(columns=rename_map)

            # Update available_cols to use new names
            renamed_cols = [col.replace('DC_', 'P_') for col in available_cols]

            # Add predicted outcome column
            market_df = self._add_prediction_column(market_df, renamed_cols, market_key)

            # Add confidence column (max probability)
            if renamed_cols:
                market_df['Confidence'] = market_df[renamed_cols].max(axis=1)
                market_df['Confidence_%'] = (market_df['Confidence'] * 100).round(1)

            # Sort by date, then league, then confidence (highest first)
            sort_cols = []
            if 'Date' in market_df.columns:
                market_df['Date'] = pd.to_datetime(market_df['Date'], errors='coerce')
                sort_cols.append('Date')
            if 'League' in market_df.columns:
                sort_cols.append('League')
            if 'Confidence' in market_df.columns:
                sort_cols.append('Confidence')
                market_df = market_df.sort_values(sort_cols, ascending=[True, True, False])
            elif sort_cols:
                market_df = market_df.sort_values(sort_cols)

            # Store
            markets[market_key] = {
                'df': market_df,
                'display_name': config['display_name']
            }

        print(f"✓ Split into {len(markets)} markets")
        return markets

    def _add_prediction_column(self, df: pd.DataFrame, prob_cols: List[str], market_key: str) -> pd.DataFrame:
        """Add a 'Prediction' column showing the most likely outcome"""
        if not prob_cols:
            return df

        # Get column with highest probability
        df['_pred_col'] = df[prob_cols].idxmax(axis=1, skipna=True)

        # Extract outcome from column name
        # E.g., 'P_1X2_H' -> 'H', 'P_BTTS_Y' -> 'Y', 'P_OU_2_5_O' -> 'O'
        outcome_map = {}
        for col in prob_cols:
            # Extract last part after final underscore
            outcome = col.split('_')[-1]
            outcome_map[col] = outcome

        df['Prediction'] = df['_pred_col'].map(outcome_map)
        df = df.drop(columns=['_pred_col'])

        # Make prediction more readable
        df = self._format_prediction_display(df, market_key)

        return df

    def _format_prediction_display(self, df: pd.DataFrame, market_key: str) -> pd.DataFrame:
        """Format prediction to be more readable"""
        if 'Prediction' not in df.columns:
            return df

        # Map short codes to readable text
        display_maps = {
            '1X2': {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'},
            'BTTS': {'Y': 'Yes (Both Score)', 'N': 'No (One/Neither)'},
            'OU_0_5': {'O': 'Over 0.5', 'U': 'Under 0.5'},
            'OU_1_5': {'O': 'Over 1.5', 'U': 'Under 1.5'},
            'OU_2_5': {'O': 'Over 2.5', 'U': 'Under 2.5'},
            'OU_3_5': {'O': 'Over 3.5', 'U': 'Under 3.5'},
            'OU_4_5': {'O': 'Over 4.5', 'U': 'Under 4.5'},
            'OU_5_5': {'O': 'Over 5.5', 'U': 'Under 5.5'},
        }

        if market_key in display_maps:
            df['Prediction_Text'] = df['Prediction'].map(display_maps[market_key])

        return df

    def save_market_files(self, markets: Dict, output_dir: Path = OUTPUT_DIR):
        """Save each market to separate CSV and HTML files"""
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []

        for market_key, market_data in markets.items():
            market_df = market_data['df']
            display_name = market_data['display_name']

            if market_df.empty:
                continue

            # Save CSV
            csv_filename = f"predictions_{market_key.lower()}.csv"
            csv_path = output_dir / csv_filename
            market_df.to_csv(csv_path, index=False)
            saved_files.append(csv_path)

            # Save HTML
            html_filename = f"predictions_{market_key.lower()}.html"
            html_path = output_dir / html_filename
            self._generate_html(market_df, display_name, market_key, html_path)
            saved_files.append(html_path)

        return saved_files

    def _generate_html(self, df: pd.DataFrame, display_name: str, market_key: str, output_path: Path):
        """Generate HTML report for a market"""
        # Summary stats
        total_matches = len(df)
        high_confidence = len(df[df.get('Confidence', 0) >= 0.75]) if 'Confidence' in df.columns else 0
        elite_confidence = len(df[df.get('Confidence', 0) >= 0.85]) if 'Confidence' in df.columns else 0

        # Group by league
        league_counts = df.groupby('League').size().to_dict() if 'League' in df.columns else {}

        # HTML template
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{display_name} - Predictions</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 20px auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 32px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #666;
            font-size: 14px;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #eee;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        .confidence-elite {{
            background: #22c55e;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
        }}
        .confidence-high {{
            background: #3b82f6;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
        }}
        .confidence-medium {{
            background: #f59e0b;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
        }}
        .confidence-low {{
            background: #ef4444;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
        }}
        .prediction {{
            font-weight: bold;
            color: #1e40af;
        }}
        .league-tag {{
            background: #e0e7ff;
            color: #4338ca;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⚽ {display_name}</h1>
        <p>Market: {market_key} | Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{total_matches}</div>
            <div class="stat-label">Total Matches</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{elite_confidence}</div>
            <div class="stat-label">Elite (≥85%)</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{high_confidence}</div>
            <div class="stat-label">High Confidence (≥75%)</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(league_counts)}</div>
            <div class="stat-label">Leagues</div>
        </div>
    </div>

    <table>
        <thead>
            <tr>
                <th>Date</th>
                <th>League</th>
                <th>Match</th>
                <th>Prediction</th>
                <th>Confidence</th>
                <th>Probabilities</th>
            </tr>
        </thead>
        <tbody>
"""

        # Add rows
        for idx, row in df.iterrows():
            date_str = row.get('Date', '').strftime('%a %d %b') if pd.notna(row.get('Date')) else ''
            league = row.get('League', '')
            home = row.get('HomeTeam', '')
            away = row.get('AwayTeam', '')
            match = f"{home} vs {away}"

            prediction = row.get('Prediction_Text', row.get('Prediction', ''))
            confidence = row.get('Confidence_%', row.get('Confidence', 0))

            # Confidence badge
            if confidence >= 85:
                conf_class = 'confidence-elite'
            elif confidence >= 75:
                conf_class = 'confidence-high'
            elif confidence >= 65:
                conf_class = 'confidence-medium'
            else:
                conf_class = 'confidence-low'

            # Build probabilities string
            prob_cols = [col for col in df.columns if col.startswith('P_')]
            probs = []
            for col in prob_cols:
                val = row.get(col, 0)
                if pd.notna(val):
                    outcome = col.split('_')[-1]
                    probs.append(f"{outcome}: {val*100:.1f}%")
            prob_str = " | ".join(probs)

            html += f"""
            <tr>
                <td>{date_str}</td>
                <td><span class="league-tag">{league}</span></td>
                <td>{match}</td>
                <td class="prediction">{prediction}</td>
                <td><span class="{conf_class}">{confidence:.1f}%</span></td>
                <td style="font-size: 12px; color: #666;">{prob_str}</td>
            </tr>
"""

        html += """
        </tbody>
    </table>
</body>
</html>
"""

        output_path.write_text(html, encoding='utf-8')
        print(f"  ✓ {output_path.name}")


def split_predictions(predictions_file: Path = None, output_dir: Path = OUTPUT_DIR):
    """
    Main function to split predictions by market

    Args:
        predictions_file: Path to weekly_bets_lite.csv (or similar)
        output_dir: Where to save market-specific files
    """
    if predictions_file is None:
        predictions_file = OUTPUT_DIR / "weekly_bets_lite.csv"

    print("\n" + "="*60)
    print(" SPLITTING PREDICTIONS BY MARKET")
    print("="*60)

    splitter = MarketSplitter(predictions_file)

    if not splitter.load_predictions():
        return False

    markets = splitter.split_by_market()

    if not markets:
        print("❌ No markets found in predictions")
        return False

    saved_files = splitter.save_market_files(markets, output_dir)

    print(f"\n✓ Created {len(saved_files)} files:")
    for file in saved_files:
        print(f"  • {file.name}")

    print("\n" + "="*60)
    return True


if __name__ == "__main__":
    split_predictions()
