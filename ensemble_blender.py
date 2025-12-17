# ensemble_blender.py - STUB for DC-ONLY system
"""
Minimal stub for ensemble blending - not used in DC-only system
DC-only uses pure Dixon-Coles probabilities without ensemble blending
"""

import pandas as pd
from pathlib import Path

def create_superblend_predictions(input_csv: Path, output_csv: Path):
    """
    Stub function - just copies input to output
    In DC-only system, no blending is needed
    """
    df = pd.read_csv(input_csv)
    df.to_csv(output_csv, index=False)
    print("ℹ Ensemble blending not used in DC-only system (predictions copied as-is)")
