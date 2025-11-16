#!/usr/bin/env python3
"""
Enhanced Prediction Module - Improved BTTS/OU/1X2 accuracy
Adds probability sharpening, confidence weighting, and market correlation analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from models import load_trained_targets, predict_proba
from features import _load_features, _feature_columns
from config import OU_LINES, AH_LINES, MODEL_ARTIFACTS_DIR, OUTPUT_DIR

def _collect_market_columns() -> List[str]:
    """Return all expected probability column names."""
    cols = ["P_1X2_H","P_1X2_D","P_1X2_A","P_BTTS_Y","P_BTTS_N"]
    
    # Over/Under lines
    for l in OU_LINES: 
        cols += [f"P_OU_{l}_O", f"P_OU_{l}_U"]
    
    # Asian Handicap lines
    for l in AH_LINES: 
        cols += [f"P_AH_{l}_H", f"P_AH_{l}_A", f"P_AH_{l}_P"]
    
    # Goal ranges
    cols += [f"P_GR_{k}" for k in ["0","1","2","3","4","5+"]]
    
    # Half time
    cols += ["P_HT_H","P_HT_D","P_HT_A"]
    
    # Half time / Full time
    cols += [f"P_HTFT_{a}_{b}" for a in ["H","D","A"] for b in ["H","D","A"]]
    
    # Team goals
    for l in ["0_5","1_5","2_5","3_5"]:
        cols += [f"P_HomeTG_{l}_O",f"P_HomeTG_{l}_U",f"P_AwayTG_{l}_O",f"P_AwayTG_{l}_U"]
    
    # Cards bands
    cols += [f"P_HomeCardsY_{b}" for b in ["0-2","3","4-5","6+"]]
    cols += [f"P_AwayCardsY_{b}" for b in ["0-2","3","4-5","6+"]]
    
    # Corners bands
    cols += [f"P_HomeCorners_{b}" for b in ["0-3","4-5","6-7","8-9","10+"]]
    cols += [f"P_AwayCorners_{b}" for b in ["0-3","4-5","6-7","8-9","10+"]]
    
    # Correct scores
    for i in range(6):
        for j in range(6):
            cols.append(f"P_CS_{i}_{j}")
    cols.append("P_CS_Other")
    
    return cols

def sharpen_probability(prob: float, temperature: float = 0.8) -> float:
    """
    NEW: Sharpen probability distribution using temperature scaling.
    Lower temperature = more confident predictions
    """
    if prob <= 0 or prob >= 1:
        return prob
    
    # Convert to logit, scale, convert back
    logit = np.log(prob / (1 - prob))
    scaled_logit = logit / temperature
    sharpened = 1 / (1 + np.exp(-scaled_logit))
    
    return float(sharpened)

def apply_market_correlations(probs: Dict[str, float]) -> Dict[str, float]:
    """
    NEW: Adjust predictions based on known market correlations
    E.g., if BTTS=Y is high, Under 0.5 should be very low
    """
    adjusted = probs.copy()
    
    # BTTS and goal correlations
    if 'BTTS_Y' in probs and probs['BTTS_Y'] > 0.7:
        # High BTTS means unlikely Under 0.5
        if 'OU_0_5_U' in adjusted:
            adjusted['OU_0_5_U'] = min(adjusted['OU_0_5_U'], 0.05)
        # Likely Over 1.5
        if 'OU_1_5_O' in adjusted:
            adjusted['OU_1_5_O'] = max(adjusted['OU_1_5_O'], 0.85)
    
    # Strong home win correlations
    if '1X2_H' in probs and probs['1X2_H'] > 0.75:
        # Reduce draw probability
        if '1X2_D' in adjusted:
            adjusted['1X2_D'] = min(adjusted['1X2_D'], 0.15)
    
    # Under 2.5 and BTTS No correlation
    if 'OU_2_5_U' in probs and probs['OU_2_5_U'] > 0.7:
        if 'BTTS_N' in adjusted:
            adjusted['BTTS_N'] = max(adjusted['BTTS_N'], 0.65)
    
    return adjusted

def calculate_confidence_weights(df: pd.DataFrame) -> Dict[str, float]:
    """
    NEW: Calculate confidence weights based on feature quality
    Better features = higher confidence in predictions
    """
    weights = {}
    
    for idx, row in df.iterrows():
        match_key = f"{row.get('HomeTeam', '')}_{row.get('AwayTeam', '')}"
        
        # Base confidence
        confidence = 1.0
        
        # Check for missing key features
        key_features = ['home_xG_MA5', 'away_xG_MA5', 'home_elo', 'away_elo']
        missing_count = sum(1 for f in key_features if pd.isna(row.get(f, np.nan)))
        confidence -= (missing_count * 0.1)
        
        # Boost confidence if we have good recent form data
        if not pd.isna(row.get('home_form_MA5', np.nan)):
            confidence += 0.05
        if not pd.isna(row.get('away_form_MA5', np.nan)):
            confidence += 0.05
        
        # Cap confidence between 0.5 and 1.2
        weights[match_key] = max(0.5, min(1.2, confidence))
    
    return weights

def _map_preds_to_columns(models, preds: dict, fixtures_df: pd.DataFrame = None) -> Tuple[List[dict], List[str]]:
    """Enhanced prediction mapping with sharpening and correlations."""
    out_cols = _collect_market_columns()
    n_rows = next(iter(preds.values())).shape[0] if preds else 0
    rows = []
    
    # Calculate confidence weights if fixtures provided
    confidence_weights = {}
    if fixtures_df is not None:
        confidence_weights = calculate_confidence_weights(fixtures_df)
    
    # Create class mappings
    class_maps = {t: list(m.classes_) for t, m in models.items()}
    
    def labmap(t): 
        return {lab: i for i, lab in enumerate(class_maps.get(t, []))}
    
    def pick(p, t, label):
        """Extract probability for specific label from prediction array."""
        if t not in class_maps:
            return 0.0
        m = labmap(t)
        return float(p[m[label]]) if label in m else 0.0
    
    for i in range(n_rows):
        row = {}
        raw_probs = {}
        
        # Initialize all columns to 0.0
        for col in out_cols:
            row[col] = 0.0
        
        # Collect raw predictions first
        # 1X2 - Main result
        if "y_1X2" in preds:
            p = preds["y_1X2"][i]
            raw_probs['1X2_H'] = pick(p, "y_1X2", "H")
            raw_probs['1X2_D'] = pick(p, "y_1X2", "D")
            raw_probs['1X2_A'] = pick(p, "y_1X2", "A")
        
        # BTTS - Both teams to score
        if "y_BTTS" in preds:
            p = preds["y_BTTS"][i]
            raw_probs['BTTS_Y'] = pick(p, "y_BTTS", "Y")
            raw_probs['BTTS_N'] = pick(p, "y_BTTS", "N")
        
        # Over/Under lines
        for l in OU_LINES:
            key = f"y_OU_{l}"
            if key in preds:
                p = preds[key][i]
                raw_probs[f'OU_{l}_O'] = pick(p, key, "O")
                raw_probs[f'OU_{l}_U'] = pick(p, key, "U")
        
        # NEW: Apply correlation adjustments
        adjusted_probs = apply_market_correlations(raw_probs)
        
        # NEW: Apply sharpening to high-confidence predictions
        for key, prob in adjusted_probs.items():
            if prob > 0.65:  # Only sharpen confident predictions
                adjusted_probs[key] = sharpen_probability(prob, temperature=0.85)
        
        # NEW: Apply confidence weight if available
        match_key = None
        if fixtures_df is not None and i < len(fixtures_df):
            fixture_row = fixtures_df.iloc[i]
            match_key = f"{fixture_row.get('HomeTeam', '')}_{fixture_row.get('AwayTeam', '')}"
            if match_key in confidence_weights:
                weight = confidence_weights[match_key]
                for key in adjusted_probs:
                    # Adjust toward 0.5 if low confidence
                    adjusted_probs[key] = 0.5 + (adjusted_probs[key] - 0.5) * weight
        
        # Map adjusted probabilities back to row
        for key, value in adjusted_probs.items():
            col_name = f"P_{key}"
            if col_name in out_cols:
                row[col_name] = value
        
        # Handle other markets (AH, GR, etc.) - existing code
        # Asian Handicap lines
        for l in AH_LINES:
            key = f"y_AH_{l}"
            if key in preds:
                p = preds[key][i]
                row[f"P_AH_{l}_H"] = pick(p, key, "H")
                row[f"P_AH_{l}_A"] = pick(p, key, "A")
                row[f"P_AH_{l}_P"] = pick(p, key, "P")
        
        # Goal ranges
        if "y_GR" in preds:
            p = preds["y_GR"][i]
            for gr in ["0","1","2","3","4","5+"]:
                row[f"P_GR_{gr}"] = pick(p, "y_GR", gr)
        
        # Half time result
        if "y_HT" in preds:
            p = preds["y_HT"][i]
            row["P_HT_H"] = pick(p, "y_HT", "H")
            row["P_HT_D"] = pick(p, "y_HT", "D")
            row["P_HT_A"] = pick(p, "y_HT", "A")
        
        # HT/FT
        if "y_HTFT" in preds:
            p = preds["y_HTFT"][i]
            for a in ["H","D","A"]:
                for b in ["H","D","A"]:
                    row[f"P_HTFT_{a}_{b}"] = pick(p, "y_HTFT", f"{a}/{b}")
        
        # Team goals
        for tm in ["Home","Away"]:
            for l in ["0_5","1_5","2_5","3_5"]:
                key = f"y_{tm}TG_{l}"
                if key in preds:
                    p = preds[key][i]
                    row[f"P_{tm}TG_{l}_O"] = pick(p, key, "O")
                    row[f"P_{tm}TG_{l}_U"] = pick(p, key, "U")
        
        # Cards
        for tm in ["Home","Away"]:
            key = f"y_{tm}CardsY"
            if key in preds:
                p = preds[key][i]
                for band in ["0-2","3","4-5","6+"]:
                    row[f"P_{tm}CardsY_{band}"] = pick(p, key, band)
        
        # Corners
        for tm in ["Home","Away"]:
            key = f"y_{tm}Corners"
            if key in preds:
                p = preds[key][i]
                for band in ["0-3","4-5","6-7","8-9","10+"]:
                    row[f"P_{tm}Corners_{band}"] = pick(p, key, band)
        
        # Correct score
        if "y_CS" in preds:
            p = preds["y_CS"][i]
            for h_goals in range(6):
                for a_goals in range(6):
                    score_label = f"{h_goals}-{a_goals}"
                    row[f"P_CS_{h_goals}_{a_goals}"] = pick(p, "y_CS", score_label)
            row["P_CS_Other"] = pick(p, "y_CS", "Other")
        
        rows.append(row)
    
    return rows, out_cols

def predict_week(fixtures_csv: Path, models_dir: Path = MODEL_ARTIFACTS_DIR) -> pd.DataFrame:
    """Generate predictions for upcoming fixtures with enhancements."""
    
    # Load fixtures
    df_fixtures = pd.read_csv(fixtures_csv)
    if df_fixtures.empty:
        return pd.DataFrame()
    
    # Load historical features for context
    hist_features = _load_features()
    
    # Build features for fixtures
    fixture_rows = []
    for _, row in df_fixtures.iterrows():
        # Get latest features for each team (simplified)
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        # Get most recent features for these teams
        home_features = hist_features[
            (hist_features['HomeTeam'] == home_team) | 
            (hist_features['AwayTeam'] == home_team)
        ].tail(1)
        
        away_features = hist_features[
            (hist_features['HomeTeam'] == away_team) | 
            (hist_features['AwayTeam'] == away_team)
        ].tail(1)
        
        # Create feature row (simplified - use your actual feature engineering)
        feature_row = {}
        for col in _feature_columns():
            if not home_features.empty and col in home_features.columns:
                feature_row[col] = home_features[col].iloc[0]
            elif not away_features.empty and col in away_features.columns:
                feature_row[col] = away_features[col].iloc[0]
            else:
                feature_row[col] = 0  # Default value
        
        fixture_rows.append(feature_row)
    
    df_future = pd.DataFrame(fixture_rows)
    
    # Load models
    models = load_trained_targets(models_dir)
    if not models:
        print("No trained models found!")
        return pd.DataFrame()
    
    # Generate predictions
    preds = predict_proba(models, df_future)
    
    # Map predictions to columns with enhancements
    rows, out_cols = _map_preds_to_columns(models, preds, df_fixtures)
    
    # Create output dataframe
    df_out = pd.DataFrame(rows, columns=out_cols)
    
    # Add fixture info
    for col in ['League', 'Date', 'HomeTeam', 'AwayTeam']:
        if col in df_fixtures.columns:
            df_out[col] = df_fixtures[col].values
    
    # Reorder columns
    info_cols = ['League', 'Date', 'HomeTeam', 'AwayTeam']
    pred_cols = [c for c in df_out.columns if c not in info_cols]
    df_out = df_out[info_cols + pred_cols]
    
    return df_out

if __name__ == "__main__":
    # Test with upcoming fixtures
    fixtures_file = OUTPUT_DIR / "upcoming_fixtures.csv"
    if fixtures_file.exists():
        predictions = predict_week(fixtures_file)
        output_file = OUTPUT_DIR / "weekly_predictions_enhanced.csv"
        predictions.to_csv(output_file, index=False)
        print(f"Enhanced predictions saved to {output_file}")
    else:
        print("No fixtures file found!")
