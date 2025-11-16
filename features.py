# features.py
# Leak-free historical feature engineering for multi-market football modeling.
# Uses only factual, historical data from football-data.co.uk ingests.

from __future__ import annotations
from dataclasses import dataclass
from pickletools import string1
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    PROCESSED_DIR, FEATURES_PARQUET, HISTORICAL_PARQUET,
    TRAIN_SEASONS_BACK, USE_ELO, USE_ROLLING_FORM, USE_MARKET_FEATURES,
    log_header
)

# -----------------------------
# Utilities
# -----------------------------

RESULT_MAP = {"H": 1, "D": 0, "A": -1}

def _points_from_ftr(ftr: pd.Series) -> pd.Series:
    # 3 for win, 1 draw, 0 loss (home perspective)
    return ftr.map({"H": 3, "D": 1, "A": 0}).fillna(0).astype(int)

def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

# -----------------------------
# Elo rating (simple, configurable)
# -----------------------------

@dataclass
class EloConfig:
    base_rating: float = 1500.0
    k_base: float = 18.0
    home_adv: float = 65.0  # typical home edge in Elo points

def _expected_score(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(ra - rb) / 400.0))

def _update_elo(ra: float, rb: float, score_a: float, cfg: EloConfig) -> Tuple[float, float]:
    ea = _expected_score(ra, rb)
    eb = 1.0 - ea
    ra_new = ra + cfg.k_base * (score_a - ea)
    rb_new = rb + cfg.k_base * ((1.0 - score_a) - eb)
    return ra_new, rb_new

def _elo_by_league(df: pd.DataFrame, cfg: EloConfig) -> pd.DataFrame:
    # Compute league-specific Elo ratings, time-ordered to avoid leakage.
    df = df.sort_values(["League","Date"]).copy()
    # Initialize ratings per (League, Team)
    teams = pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]]))
    # We maintain state dict per league separately
    state: Dict[Tuple[str,str], float] = {}

    home_elos = []
    away_elos = []

    for idx, row in df.iterrows():
        lg = row["League"]
        ht = row["HomeTeam"]
        at = row["AwayTeam"]
        key_h = (lg, ht); key_a = (lg, at)
        ra = state.get(key_h, cfg.base_rating)
        rb = state.get(key_a, cfg.base_rating)

        # Pre-match elos (features)
        home_elos.append(ra)
        away_elos.append(rb)

        # Match outcome as score for home
        ftr = row.get("FTR")
        if pd.isna(ftr):
            # For fixtures without result, do not update
            continue
        if ftr == "H": score_home = 1.0
        elif ftr == "D": score_home = 0.5
        else: score_home = 0.0

        # Apply home advantage offset for expectation (as Elo points)
        ra_eff = ra + cfg.home_adv
        rb_eff = rb

        ra_new, rb_new = _update_elo(ra_eff, rb_eff, score_home, cfg)
        # Remove the home advantage when storing back
        state[key_h] = ra_new - cfg.home_adv
        state[key_a] = rb_new

    out = df.copy()
    out["EloHome_pre"] = home_elos
    out["EloAway_pre"] = away_elos
    out["EloDiff_pre"] = out["EloHome_pre"] - out["EloAway_pre"]
    return out

# -----------------------------
# Rolling team form & stats (leak-free via shift)
# -----------------------------

def _add_team_side(df: pd.DataFrame, side: str) -> pd.DataFrame:
    # side: 'Home' or 'Away' to create unified columns (Team, Opp, GoalsFor, GoalsAgainst, Shots, ShotsT, Corners, CardsY, CardsR)
    out = df.copy()
    if side == "Home":
        out["Team"] = out["HomeTeam"]
        out["Opp"] = out["AwayTeam"]
        out["GoalsFor"] = out["FTHG"]
        out["GoalsAgainst"] = out["FTAG"]
        out["Win"] = (out["FTR"] == "H").astype(int)
        out["Draw"] = (out["FTR"] == "D").astype(int)
        out["Loss"] = (out["FTR"] == "A").astype(int)
        # shots/SoT/corners/cards (optional in source)
        out = _ensure_cols(out, ["HS","HST","HC","HY","HR"])
        out["Shots"] = out["HS"]
        out["ShotsT"] = out["HST"]
        out["Corners"] = out["HC"]
        out["CardsY"] = out["HY"]
        out["CardsR"] = out["HR"]
    else:
        out["Team"] = out["AwayTeam"]
        out["Opp"] = out["HomeTeam"]
        out["GoalsFor"] = out["FTAG"]
        out["GoalsAgainst"] = out["FTHG"]
        out["Win"] = (out["FTR"] == "A").astype(int)
        out["Draw"] = (out["FTR"] == "D").astype(int)
        out["Loss"] = (out["FTR"] == "H").astype(int)
        out = _ensure_cols(out, ["AS","AST","AC","AY","AR"])
        out["Shots"] = out["AS"]
        out["ShotsT"] = out["AST"]
        out["Corners"] = out["AC"]
        out["CardsY"] = out["AY"]
        out["CardsR"] = out["AR"]
    out["Side"] = side
    return out[["League","Date","Team","Opp","Side","GoalsFor","GoalsAgainst","Win","Draw","Loss","Shots","ShotsT","Corners","CardsY","CardsR"]]

def _rolling_stats(team_df: pd.DataFrame, windows: List[int] = [5,10,20]) -> pd.DataFrame:
    team_df = team_df.sort_values("Date").copy()
    # shift to prevent leakage (only past info)
    for w in windows:
        rolled = team_df.shift(1).rolling(window=w, min_periods=1)
        team_df[f"GF_ma{w}"] = rolled["GoalsFor"].mean()
        team_df[f"GA_ma{w}"] = rolled["GoalsAgainst"].mean()
        team_df[f"GD_ma{w}"] = team_df[f"GF_ma{w}"] - team_df[f"GA_ma{w}"]
        team_df[f"Pts_ma{w}"] = (rolled["Win"].sum() * 3 + rolled["Draw"].sum() * 1) / w
        for col in ["Shots","ShotsT","Corners","CardsY","CardsR"]:
            team_df[f"{col}_ma{w}"] = rolled[col].mean()
    # recency-weighted form (EWMA)
    ew = team_df.shift(1).ewm(span=10, adjust=False)
    team_df["GF_ewm10"] = ew["GoalsFor"].mean()
    team_df["GA_ewm10"] = ew["GoalsAgainst"].mean()
    team_df["FormPts_ewm10"] = (ew["Win"].mean()*3 + ew["Draw"].mean()*1)
    return team_df

def _build_side_features(df: pd.DataFrame) -> pd.DataFrame:
    home = _add_team_side(df, "Home")
    away = _add_team_side(df, "Away")
    long = pd.concat([home, away], ignore_index=True)
    # compute per (League, Team)
    parts = []
    for (lg, tm), g in long.groupby(["League","Team"], sort=False):
        parts.append(_rolling_stats(g))
    long_feats = pd.concat(parts, ignore_index=True)
    return long_feats

def _pivot_back(match_df: pd.DataFrame, side_feats: pd.DataFrame) -> pd.DataFrame:
    # Join side features back to match rows as Home_* and Away_*
    # Start with unique key
    key_cols = ["League","Date","HomeTeam","AwayTeam"]
    out = match_df[key_cols + ["FTHG","FTAG","FTR","B365H","B365D","B365A","PSCH","PSCD","PSCA"]].copy()
    # home join
    hf = side_feats.query("Side == 'Home'")[ ["League","Date","Team"] + [c for c in side_feats.columns if c not in ["League","Date","Team","Opp","Side","GoalsFor","GoalsAgainst","Win","Draw","Loss"]] ].copy()
    hf = hf.rename(columns={c: f"Home_{c}" for c in hf.columns if c not in ["League","Date","Team"]})
    out = out.merge(hf, left_on=["League","Date","HomeTeam"], right_on=["League","Date","Team"], how="left").drop(columns=["Team"])    
    # away join
    af = side_feats.query("Side == 'Away'")[ ["League","Date","Team"] + [c for c in side_feats.columns if c not in ["League","Date","Team","Opp","Side","GoalsFor","GoalsAgainst","Win","Draw","Loss"]] ].copy()
    af = af.rename(columns={c: f"Away_{c}" for c in af.columns if c not in ["League","Date","Team"]})
    out = out.merge(af, left_on=["League","Date","AwayTeam"], right_on=["League","Date","Team"], how="left").drop(columns=["Team"])    
    return out

# -----------------------------
# Targets for multiple markets
# -----------------------------

OU_LINES = [0.5, 1.5, 2.5, 3.5, 4.5]
AH_LINES = [-1.0, -0.5, 0.0, 0.5, 1.0]

def _target_1x2(df: pd.DataFrame) -> pd.Series:
    return df["FTR"].astype(str);

def _target_btts(df: pd.DataFrame) -> pd.Series:
    total = df["FTHG"].fillna(0) + df["FTAG"].fillna(0)
    both = (df["FTHG"].fillna(0) > 0) & (df["FTAG"].fillna(0) > 0)
    return np.where(both, "Y", "N")

def _target_ou(df: pd.DataFrame, line: float) -> pd.Series:
    tot = df["FTHG"].fillna(0) + df["FTAG"].fillna(0)
    return np.where(tot > line, "O", "U")

def _target_ah_home(df: pd.DataFrame, line: float) -> pd.Series:
    # Asian handicap result from home perspective: "H", "A", "P" (push)
    diff = df["FTHG"].fillna(0) - df["FTAG"].fillna(0) - line
    res = np.where(diff > 0, "H", np.where(diff < 0, "A", "P"))
    return res

def _target_goal_ranges(df: pd.DataFrame) -> pd.Series:
    tot = (df["FTHG"].fillna(0) + df["FTAG"].fillna(0)).astype(int)
    bins = pd.cut(tot, bins=[-1,0,1,2,3,4,100], labels=["0","1","2","3","4","5+"], right=True)
    return bins.astype(str)

def _target_correct_score(df: pd.DataFrame, max_goal: int = 5) -> pd.Series:
    # Map to 'x-y' for x,y <= max_goal, otherwise 'Other'
    x = df["FTHG"].astype("Int64").fillna(-1)
    y = df["FTAG"].astype("Int64").fillna(-1)
    lab = x.astype(str) + "-" + y.astype(str)
    mask = (x <= max_goal) & (y <= max_goal) & (x >= 0) & (y >= 0)
    out = np.where(mask, lab, "Other")
    return pd.Series(out).astype(str)

# Corners and Cards totals/bands depend on columns availability; if missing, we skip targets.
def _maybe_sum_cols(df: pd.DataFrame, home_col: str, away_col: str) -> pd.Series:
    if home_col in df.columns and away_col in df.columns:
        return df[home_col].astype("float").fillna(0) + df[away_col].astype("float").fillna(0)
    return pd.Series(np.nan, index=df.index)

def _target_bands(total: pd.Series, bands: List[Tuple[int,int]], labels: List[str]) -> pd.Series:
    lab = pd.Series(np.nan, index=total.index, dtype="object")
    for (lo, hi), name in zip(bands, labels):
        lab = np.where((total >= lo) & (total <= hi), name, lab)
    # any above last hi -> 'High+'
    return pd.Series(lab).astype(str)

# -----------------------------
# Main build function
# -----------------------------

def build_features(force: bool = False) -> Path:
    out_path = FEATURES_PARQUET
    if out_path.exists() and not force:
        log_header(f"Features parquet already exists at {out_path}. Skipping.")
        return out_path

    hist_path = HISTORICAL_PARQUET
    if not hist_path.exists():
        raise FileNotFoundError(f"Historical parquet not found at {hist_path}. Run data ingestion first.")

    df = pd.read_parquet(hist_path)
    df = df.dropna(subset=["Date","HomeTeam","AwayTeam"]).copy()
    df = df.sort_values(["League","Date"]).reset_index(drop=True)

    # Optional Elo
    if USE_ELO:
        df = _elo_by_league(df, EloConfig())
    else:
        df["EloHome_pre"] = np.nan
        df["EloAway_pre"] = np.nan
        df["EloDiff_pre"] = np.nan

    # Rolling form/stats
    if USE_ROLLING_FORM:
        side_feats = _build_side_features(df)
        df = _pivot_back(df, side_feats)
    else:
        # Minimal structure if disabled
        for prefix in ["Home","Away"]:
            for stat in ["GF_ma5","GA_ma5","GD_ma5","Pts_ma5","Corners_ma5","CardsY_ma5","ShotsT_ma5","GF_ewm10","GA_ewm10","FormPts_ewm10"]:
                df[f"{prefix}_{stat}"] = np.nan

    # Market features (closing odds if present)
    if not USE_MARKET_FEATURES:
        for c in ["B365H","B365D","B365A","PSCH","PSCD","PSCA"]:
            if c in df.columns: df[c] = np.nan

    # Targets
    df["y_1X2"] = _target_1x2(df)
    df["y_BTTS"] = _target_btts(df).astype(str)
    for line in OU_LINES:
        df[f"y_OU_{str(line).replace('.','_')}"] = _target_ou(df, line).astype(str)
    for line in AH_LINES:
        df[f"y_AH_{str(line).replace('.','_')}"] = _target_ah_home(df, line).astype(str)
    df["y_GOAL_RANGE"] = _target_goal_ranges(df)
    df["y_CS"] = _target_correct_score(df, max_goal=5)

    # Corners & Cards targets if source columns exist
    df = _ensure_cols(df, ["HC","AC","HY","AY"])  # totals exist on many leagues
    tot_corners = _maybe_sum_cols(df, "HC", "AC")
    tot_cards_y = _maybe_sum_cols(df, "HY", "AY")
    df["CornersTotal"] = tot_corners
    df["CardsYTotal"] = tot_cards_y

    # Bands (example): 0-7, 8-10, 11-13, 14+
    if tot_corners.notna().any():
        corners_bands = [(0,7),(8,10),(11,13)]
        corners_labels = ["0-7","8-10","11-13"]
        df["y_CORNERS_BAND"] = _target_bands(tot_corners, corners_bands, corners_labels)
    else:
        df["y_CORNERS_BAND"] = pd.Series(pd.NA, index=df.index, dtype=str)

    # Cards bands: 0-3, 4-5, 6-7, 8+ (yellow-equivalent counts; reds ~ 2Y but we use Y only for simplicity)
    if tot_cards_y.notna().any():
        cards_bands = [(0,3),(4,5),(6,7)]
        cards_labels = ["0-3","4-5","6-7"]
        df["y_CARDSY_BAND"] = _target_bands(tot_cards_y, cards_bands, cards_labels)
    else:
        df["y_CARDSY_BAND"] = pd.Series(pd.NA, index=df.index, dtype=str)

    # Final ordering & save
    df.to_parquet(out_path, index=False)
    log_header(f"Wrote features to {out_path} with {len(df):,} rows and {df.shape[1]} columns.")
    return out_path

def get_feature_columns() -> List[str]:
    """Return list of feature columns (not targets or metadata)"""
    exclude = ['Date', 'League', 'HomeTeam', 'AwayTeam', 'Referee', 
               'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR']
    
    # Get all columns from features file
    from config import FEATURES_PARQUET
    import pandas as pd
    
    df = pd.read_parquet(FEATURES_PARQUET)
    
    feature_cols = [col for col in df.columns 
                    if col not in exclude 
                    and not col.startswith('y_')
                    and not col.startswith('FT')
                    and not col.startswith('HT')]
    
    return feature_cols

    
if __name__ == "__main__":
    build_features(force=False)
