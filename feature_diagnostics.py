"""
Feature diagnostics and sanity checking output
Generates a summary of features being used for predictions
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def analyze_features(features_path: Path, output_path: Path):
    """Analyze and report on features used for predictions"""

    print("="*70)
    print("FEATURE DIAGNOSTICS - SANITY CHECKS")
    print("="*70)

    # Load features
    df = pd.read_parquet(features_path)

    print(f"\nDataset: {len(df)} matches")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Leagues: {df['League'].nunique()} ({', '.join(df['League'].unique())})")

    # Separate features from metadata
    metadata_cols = ["fixture_id", "Date", "League", "HomeTeam", "AwayTeam",
                     "LeagueName", "LeagueType", "Season", "Round",
                     "HomeTeamID", "AwayTeamID", "Referee", "Venue",
                     "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR"]

    feature_cols = [c for c in df.columns
                    if not c.startswith("y_")
                    and c not in metadata_cols]

    print(f"\nFeatures available: {len(feature_cols)}")

    # Categorize features
    rolling_form = [c for c in feature_cols if 'ma5' in c or 'ewm' in c]
    elo_features = [c for c in feature_cols if 'Elo' in c]
    xg_features = [c for c in feature_cols if 'xG' in c or 'xg' in c.lower()]
    stat_features = [c for c in feature_cols if c.startswith(('H', 'A')) and c not in rolling_form + elo_features + xg_features]
    other_features = [c for c in feature_cols if c not in rolling_form + elo_features + xg_features + stat_features]

    print(f"\nFeature breakdown:")
    print(f"  Rolling form (ma5/ewm): {len(rolling_form)}")
    print(f"  ELO ratings: {len(elo_features)}")
    print(f"  xG features: {len(xg_features)}")
    print(f"  Match statistics: {len(stat_features)}")
    print(f"  Other: {len(other_features)}")

    # Check for missing data
    print(f"\n" + "="*70)
    print("DATA QUALITY CHECKS")
    print("="*70)

    null_pct = (df[feature_cols].isnull().sum() / len(df) * 100).sort_values(ascending=False)
    high_null = null_pct[null_pct > 50]

    if len(high_null) > 0:
        print(f"\nWARNING: {len(high_null)} features with >50% missing data:")
        for feat, pct in high_null.head(10).items():
            print(f"  {feat}: {pct:.1f}% missing")
    else:
        print("\n[OK] No features with excessive missing data")

    # Check for constant features
    constant_features = []
    for col in feature_cols:
        if df[col].dtype in ['float64', 'int64']:
            if df[col].nunique() == 1:
                constant_features.append(col)

    if constant_features:
        print(f"\nWARNING: {len(constant_features)} constant features (no variation):")
        for feat in constant_features[:10]:
            print(f"  {feat}: value={df[feat].iloc[0]}")
    else:
        print("\n[OK] All features have variation")

    # Feature statistics
    print(f"\n" + "="*70)
    print("KEY FEATURE STATISTICS")
    print("="*70)

    key_features = ['Home_GF_ma5', 'Away_GF_ma5', 'Home_EloHome_pre', 'Away_EloAway_pre',
                    'home_xG', 'away_xG', 'match_number']

    stats_list = []
    for feat in key_features:
        if feat in df.columns:
            col_data = df[feat]
            stats_list.append({
                'Feature': feat,
                'Mean': f"{col_data.mean():.2f}" if pd.notna(col_data.mean()) else "N/A",
                'Std': f"{col_data.std():.2f}" if pd.notna(col_data.std()) else "N/A",
                'Min': f"{col_data.min():.2f}" if pd.notna(col_data.min()) else "N/A",
                'Max': f"{col_data.max():.2f}" if pd.notna(col_data.max()) else "N/A",
                'Missing%': f"{col_data.isnull().sum() / len(df) * 100:.1f}%"
            })

    stats_df = pd.DataFrame(stats_list)
    print(f"\n{stats_df.to_string(index=False)}")

    # Feature correlation analysis
    print(f"\n" + "="*70)
    print("FEATURE SANITY - TEAM-SPECIFIC VALUES")
    print("="*70)

    # Check if Home_ and Away_ features differ appropriately
    home_away_pairs = [
        ('Home_GF_ma5', 'Away_GF_ma5'),
        ('Home_GA_ma5', 'Away_GA_ma5'),
        ('Home_EloHome_pre', 'Away_EloAway_pre')
    ]

    for home_feat, away_feat in home_away_pairs:
        if home_feat in df.columns and away_feat in df.columns:
            correlation = df[home_feat].corr(df[away_feat])
            same_value_pct = (df[home_feat] == df[away_feat]).sum() / len(df) * 100

            print(f"\n{home_feat} vs {away_feat}:")
            print(f"  Correlation: {correlation:.3f}")
            print(f"  Identical values: {same_value_pct:.1f}% of matches")

            if same_value_pct > 10:
                print(f"  [WARNING] High percentage of identical home/away values!")

    # Save detailed report
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("FEATURE DIAGNOSTICS REPORT\n")
        f.write("="*70 + "\n\n")

        f.write(f"Dataset: {len(df)} matches\n")
        f.write(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}\n")
        f.write(f"Total features: {len(feature_cols)}\n\n")

        f.write("FEATURE LIST:\n")
        f.write("-"*70 + "\n")

        for i, feat in enumerate(sorted(feature_cols), 1):
            dtype = df[feat].dtype
            null_pct = df[feat].isnull().sum() / len(df) * 100
            f.write(f"{i}. {feat} ({dtype}) - {null_pct:.1f}% missing\n")

        f.write("\n" + "="*70 + "\n")
        f.write("KEY FEATURE STATISTICS\n")
        f.write("="*70 + "\n\n")
        f.write(stats_df.to_string(index=False))

    print(f"\n[OK] Detailed report saved to: {output_path}")


if __name__ == "__main__":
    features_file = Path("../data/processed/features.parquet")
    output_file = Path("outputs/feature_diagnostics.txt")

    if not features_file.exists():
        print(f"ERROR: Features file not found at {features_file}")
        sys.exit(1)

    analyze_features(features_file, output_file)
