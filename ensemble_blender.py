#!/usr/bin/env python3
"""
Advanced Ensemble Blender
Intelligently combines predictions from multiple models using dynamic weighting
Tracks model performance and adjusts blend weights accordingly
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sqlite3
from datetime import datetime, timedelta
from scipy.special import softmax

class SmartBlender:
    """
    NEW: Dynamic model blending based on recent performance
    """
    
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or Path("outputs/accuracy_database.db")
        self.model_weights = {}
        self.market_weights = {}
        self.load_historical_performance()
    
    def load_historical_performance(self):
        """
        Load historical accuracy data to calculate model weights
        """
        if not self.db_path.exists():
            # Default weights if no history
            self.model_weights = {
                'P_': 0.4,   # Base ML model
                'DC_': 0.35, # Dixon-Coles
                'BLEND_': 0.25  # Previous blend
            }
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get recent accuracy by model type
            query = """
            SELECT 
                CASE 
                    WHEN market LIKE 'P_%' THEN 'ML'
                    WHEN market LIKE 'DC_%' THEN 'DC'
                    WHEN market LIKE 'BLEND_%' THEN 'BLEND'
                END as model_type,
                COUNT(*) as total,
                SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct,
                AVG(CASE WHEN correct = 1 THEN 1.0 ELSE 0.0 END) as accuracy
            FROM predictions
            WHERE actual_outcome IS NOT NULL
            AND prediction_date > date('now', '-90 days')
            GROUP BY model_type
            """
            
            results = pd.read_sql_query(query, conn)
            conn.close()
            
            if not results.empty:
                # Calculate weights based on accuracy
                for _, row in results.iterrows():
                    if row['total'] > 50:  # Need sufficient samples
                        # Weight = accuracy normalized
                        base_weight = row['accuracy']
                        if row['model_type'] == 'ML':
                            self.model_weights['P_'] = base_weight
                        elif row['model_type'] == 'DC':
                            self.model_weights['DC_'] = base_weight
                        elif row['model_type'] == 'BLEND':
                            self.model_weights['BLEND_'] = base_weight
                
                # Normalize weights to sum to 1
                total = sum(self.model_weights.values())
                if total > 0:
                    for key in self.model_weights:
                        self.model_weights[key] /= total
            
        except Exception as e:
            print(f"Could not load historical performance: {e}")
            # Use default weights
            self.model_weights = {'P_': 0.4, 'DC_': 0.35, 'BLEND_': 0.25}
    
    def calculate_market_specific_weights(self, market: str) -> Dict[str, float]:
        """
        NEW: Get market-specific weights (some models better for certain markets)
        """
        # Default to global weights
        weights = self.model_weights.copy()
        
        # Market-specific adjustments based on empirical observations
        if 'BTTS' in market:
            # Dixon-Coles often good for BTTS
            weights['DC_'] *= 1.2
        elif 'OU_2_5' in market:
            # ML models typically better for O/U 2.5
            weights['P_'] *= 1.15
        elif '1X2' in market:
            # Blend usually best for match result
            weights['BLEND_'] *= 1.1
        elif 'OU_0_5' in market or 'OU_4_5' in market:
            # Extreme lines benefit from DC model
            weights['DC_'] *= 1.25
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            for key in weights:
                weights[key] /= total
        
        return weights
    
    def blend_predictions(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        NEW: Smart blending of P_, DC_, and existing BLEND_ columns
        Creates new SUPERBLEND_ columns with optimized weights
        """
        blended = predictions_df.copy()
        
        # Track all markets we're blending
        markets_blended = set()
        
        # Find all unique markets (e.g., 1X2_H, BTTS_Y, OU_2_5_O)
        for col in predictions_df.columns:
            if col.startswith('P_'):
                market = col[2:]  # Remove 'P_' prefix
                markets_blended.add(market)
        
        # Blend each market
        for market in markets_blended:
            p_col = f'P_{market}'
            dc_col = f'DC_{market}'
            blend_col = f'BLEND_{market}'
            
            # Skip if base column doesn't exist
            if p_col not in predictions_df.columns:
                continue
            
            # Get market-specific weights
            weights = self.calculate_market_specific_weights(market)
            
            # Create superblend
            superblend = np.zeros(len(predictions_df))
            weight_sum = 0
            
            if p_col in predictions_df.columns:
                superblend += predictions_df[p_col].fillna(0) * weights.get('P_', 0.4)
                weight_sum += weights.get('P_', 0.4)
            
            if dc_col in predictions_df.columns:
                superblend += predictions_df[dc_col].fillna(0) * weights.get('DC_', 0.35)
                weight_sum += weights.get('DC_', 0.35)
            
            if blend_col in predictions_df.columns:
                superblend += predictions_df[blend_col].fillna(0) * weights.get('BLEND_', 0.25)
                weight_sum += weights.get('BLEND_', 0.25)
            
            # Normalize if we had any inputs
            if weight_sum > 0:
                superblend /= weight_sum
            
            # Store as SUPERBLEND column
            blended[f'SUPERBLEND_{market}'] = superblend
        
        # Ensure probability constraints for paired markets
        blended = self.enforce_probability_constraints(blended)
        
        return blended
    
    def enforce_probability_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        NEW: Ensure probabilities sum to 1 for mutually exclusive events
        """
        # 1X2 markets
        if all(col in df.columns for col in ['SUPERBLEND_1X2_H', 'SUPERBLEND_1X2_D', 'SUPERBLEND_1X2_A']):
            for idx in df.index:
                probs = [
                    df.at[idx, 'SUPERBLEND_1X2_H'],
                    df.at[idx, 'SUPERBLEND_1X2_D'],
                    df.at[idx, 'SUPERBLEND_1X2_A']
                ]
                # Normalize using softmax for smooth distribution
                normalized = softmax(probs)
                df.at[idx, 'SUPERBLEND_1X2_H'] = normalized[0]
                df.at[idx, 'SUPERBLEND_1X2_D'] = normalized[1]
                df.at[idx, 'SUPERBLEND_1X2_A'] = normalized[2]
        
        # BTTS markets
        if all(col in df.columns for col in ['SUPERBLEND_BTTS_Y', 'SUPERBLEND_BTTS_N']):
            for idx in df.index:
                yes = df.at[idx, 'SUPERBLEND_BTTS_Y']
                no = df.at[idx, 'SUPERBLEND_BTTS_N']
                total = yes + no
                if total > 0:
                    df.at[idx, 'SUPERBLEND_BTTS_Y'] = yes / total
                    df.at[idx, 'SUPERBLEND_BTTS_N'] = no / total
        
        # Over/Under markets
        for line in ['0_5', '1_5', '2_5', '3_5', '4_5']:
            over_col = f'SUPERBLEND_OU_{line}_O'
            under_col = f'SUPERBLEND_OU_{line}_U'
            if all(col in df.columns for col in [over_col, under_col]):
                for idx in df.index:
                    over = df.at[idx, over_col]
                    under = df.at[idx, under_col]
                    total = over + under
                    if total > 0:
                        df.at[idx, over_col] = over / total
                        df.at[idx, under_col] = under / total
        
        return df
    
    def calculate_confidence_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        NEW: Add confidence scores based on model agreement
        """
        df = df.copy()
        
        for idx, row in df.iterrows():
            # Check each market type
            for market_base in ['1X2_H', '1X2_D', '1X2_A', 'BTTS_Y', 'BTTS_N', 
                               'OU_2_5_O', 'OU_2_5_U', 'OU_1_5_O', 'OU_1_5_U']:
                
                predictions = []
                
                # Collect all predictions for this market
                for prefix in ['P_', 'DC_', 'BLEND_', 'SUPERBLEND_']:
                    col = f'{prefix}{market_base}'
                    if col in row and pd.notna(row[col]):
                        predictions.append(row[col])
                
                if len(predictions) >= 2:
                    # Calculate standard deviation as inverse confidence
                    std_dev = np.std(predictions)
                    # Low std = high agreement = high confidence
                    confidence = 1 - min(std_dev * 2, 1)  # Scale and cap at 1
                    
                    # Store confidence score
                    df.at[idx, f'CONF_{market_base}'] = confidence
                    
                    # Boost SUPERBLEND if high confidence
                    if f'SUPERBLEND_{market_base}' in df.columns and confidence > 0.8:
                        current = df.at[idx, f'SUPERBLEND_{market_base}']
                        # Push probability away from 0.5 when confident
                        if current > 0.5:
                            df.at[idx, f'SUPERBLEND_{market_base}'] = min(current * 1.05, 0.95)
                        else:
                            df.at[idx, f'SUPERBLEND_{market_base}'] = max(current * 0.95, 0.05)
        
        return df


def create_superblend_predictions(input_file: Path, output_file: Path):
    """
    Main function to create enhanced superblend predictions
    """
    print("Creating SUPERBLEND predictions with advanced ensemble techniques...")
    
    # Load predictions
    predictions = pd.read_csv(input_file)
    
    # Initialize blender
    blender = SmartBlender()
    
    # Show model weights being used
    print("\nModel Weights (based on historical performance):")
    for model, weight in blender.model_weights.items():
        print(f"  {model}: {weight:.2%}")
    
    # Create superblend
    superblended = blender.blend_predictions(predictions)
    
    # Add confidence scores
    superblended = blender.calculate_confidence_scores(superblended)
    
    # Sort columns for better readability
    info_cols = ['League', 'Date', 'HomeTeam', 'AwayTeam']
    superblend_cols = [c for c in superblended.columns if c.startswith('SUPERBLEND_')]
    conf_cols = [c for c in superblended.columns if c.startswith('CONF_')]
    other_cols = [c for c in superblended.columns if c not in info_cols + superblend_cols + conf_cols]
    
    # Reorder columns
    ordered_cols = info_cols + superblend_cols + conf_cols + other_cols
    superblended = superblended[[c for c in ordered_cols if c in superblended.columns]]
    
    # Save enhanced predictions
    superblended.to_csv(output_file, index=False)
    print(f"\nSUPERBLEND predictions saved to {output_file}")
    
    # Show top confident predictions
    print("\nTop 10 Most Confident Predictions:")
    
    for market in ['1X2_H', 'BTTS_Y', 'OU_2_5_O']:
        if f'SUPERBLEND_{market}' in superblended.columns and f'CONF_{market}' in superblended.columns:
            top_picks = superblended.nlargest(5, f'CONF_{market}')
            
            print(f"\n{market}:")
            for _, pick in top_picks.iterrows():
                print(f"  {pick['HomeTeam']} vs {pick['AwayTeam']}: "
                      f"{pick[f'SUPERBLEND_{market}']:.1%} "
                      f"(Confidence: {pick[f'CONF_{market}']:.1%})")
    
    return superblended


if __name__ == "__main__":
    from pathlib import Path
    
    OUTPUT_DIR = Path("outputs")
    input_file = OUTPUT_DIR / "weekly_bets_lite.csv"
    output_file = OUTPUT_DIR / "weekly_bets_lite_superblend.csv"
    
    if input_file.exists():
        create_superblend_predictions(input_file, output_file)
    else:
        print("No input file found!")