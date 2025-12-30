# models_dc_xg.py
"""
xG-Integrated Dixon-Coles Model
Enhanced with: xG, injuries, formations, cup competitions

Key improvements over standard DC:
1. Uses xG instead of raw goals for expected goals calculation
2. Adjusts for injuries and key players missing
3. Formation-aware goal expectations
4. Cup competition adjustments (reduced home advantage)
5. League-specific rho optimization
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

MAX_GOALS = 12
RECENT_FORM_WINDOW = 5

# League-specific parameters
LEAGUE_CONFIG = {
    # England - Full pyramid
    'E0': {'rho_init': 0.03, 'rho_bounds': (-0.03, 0.12), 'decay_days': 365, 'xg_weight': 0.7},
    'E1': {'rho_init': 0.04, 'rho_bounds': (-0.05, 0.15), 'decay_days': 320, 'xg_weight': 0.6},
    'E2': {'rho_init': 0.04, 'rho_bounds': (-0.05, 0.15), 'decay_days': 310, 'xg_weight': 0.5},
    'E3': {'rho_init': 0.05, 'rho_bounds': (-0.05, 0.16), 'decay_days': 300, 'xg_weight': 0.5},
    'EC': {'rho_init': 0.05, 'rho_bounds': (-0.05, 0.16), 'decay_days': 290, 'xg_weight': 0.4},

    # Germany
    'D1': {'rho_init': 0.02, 'rho_bounds': (-0.05, 0.10), 'decay_days': 380, 'xg_weight': 0.7},
    'D2': {'rho_init': 0.03, 'rho_bounds': (-0.05, 0.12), 'decay_days': 340, 'xg_weight': 0.6},

    # Spain
    'SP1': {'rho_init': 0.05, 'rho_bounds': (-0.02, 0.15), 'decay_days': 390, 'xg_weight': 0.7},
    'SP2': {'rho_init': 0.06, 'rho_bounds': (-0.02, 0.16), 'decay_days': 350, 'xg_weight': 0.6},

    # Italy
    'I1': {'rho_init': 0.08, 'rho_bounds': (0.00, 0.18), 'decay_days': 370, 'xg_weight': 0.7},
    'I2': {'rho_init': 0.07, 'rho_bounds': (0.00, 0.16), 'decay_days': 340, 'xg_weight': 0.6},

    # France
    'F1': {'rho_init': 0.04, 'rho_bounds': (-0.02, 0.12), 'decay_days': 360, 'xg_weight': 0.7},
    'F2': {'rho_init': 0.05, 'rho_bounds': (-0.02, 0.15), 'decay_days': 330, 'xg_weight': 0.6},

    # Netherlands
    'N1': {'rho_init': 0.05, 'rho_bounds': (-0.02, 0.15), 'decay_days': 350, 'xg_weight': 0.6},

    # Belgium
    'B1': {'rho_init': 0.05, 'rho_bounds': (-0.02, 0.15), 'decay_days': 350, 'xg_weight': 0.6},

    # Portugal
    'P1': {'rho_init': 0.07, 'rho_bounds': (0.00, 0.16), 'decay_days': 360, 'xg_weight': 0.6},

    # Greece
    'G1': {'rho_init': 0.07, 'rho_bounds': (0.00, 0.16), 'decay_days': 350, 'xg_weight': 0.5},

    # Scotland - Full divisions
    'SC0': {'rho_init': 0.04, 'rho_bounds': (-0.05, 0.15), 'decay_days': 330, 'xg_weight': 0.5},
    'SC1': {'rho_init': 0.05, 'rho_bounds': (-0.05, 0.15), 'decay_days': 320, 'xg_weight': 0.5},
    'SC2': {'rho_init': 0.05, 'rho_bounds': (-0.05, 0.16), 'decay_days': 310, 'xg_weight': 0.4},
    'SC3': {'rho_init': 0.06, 'rho_bounds': (-0.05, 0.16), 'decay_days': 300, 'xg_weight': 0.4},

    # Turkey
    'T1': {'rho_init': 0.06, 'rho_bounds': (-0.01, 0.15), 'decay_days': 360, 'xg_weight': 0.5},

    # Cup competitions - different characteristics!
    'FA_CUP': {'rho_init': 0.04, 'rho_bounds': (-0.05, 0.15), 'decay_days': 300, 'xg_weight': 0.5, 'home_adv_mult': 0.85},
    'EFL_CUP': {'rho_init': 0.04, 'rho_bounds': (-0.05, 0.15), 'decay_days': 300, 'xg_weight': 0.5, 'home_adv_mult': 0.80},
    'DFB_POKAL': {'rho_init': 0.03, 'rho_bounds': (-0.05, 0.12), 'decay_days': 300, 'xg_weight': 0.5, 'home_adv_mult': 0.85},
    'COPA_DEL_REY': {'rho_init': 0.05, 'rho_bounds': (-0.02, 0.15), 'decay_days': 300, 'xg_weight': 0.5, 'home_adv_mult': 0.85},
    'COPPA_ITALIA': {'rho_init': 0.07, 'rho_bounds': (0.00, 0.16), 'decay_days': 300, 'xg_weight': 0.5, 'home_adv_mult': 0.85},
    'COUPE_DE_FRANCE': {'rho_init': 0.04, 'rho_bounds': (-0.02, 0.15), 'decay_days': 300, 'xg_weight': 0.5, 'home_adv_mult': 0.85},

    # European competitions - most variable, lower xG weight
    'UCL': {'rho_init': 0.05, 'rho_bounds': (-0.03, 0.15), 'decay_days': 400, 'xg_weight': 0.6, 'home_adv_mult': 0.90},
    'UEL': {'rho_init': 0.05, 'rho_bounds': (-0.03, 0.15), 'decay_days': 380, 'xg_weight': 0.5, 'home_adv_mult': 0.90},
    'UECL': {'rho_init': 0.05, 'rho_bounds': (-0.03, 0.15), 'decay_days': 360, 'xg_weight': 0.5, 'home_adv_mult': 0.90},
}

DEFAULT_CONFIG = {'rho_init': 0.05, 'rho_bounds': (-0.05, 0.15), 'decay_days': 365, 'xg_weight': 0.5, 'home_adv_mult': 1.0}

def get_league_config(league: str) -> Dict:
    """Get configuration for a league"""
    return LEAGUE_CONFIG.get(league, DEFAULT_CONFIG)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DCParamsXG:
    """
    Enhanced Dixon-Coles parameters with xG integration
    """
    # Core DC parameters
    attack: Dict[str, float]
    defence: Dict[str, float]
    home_adv: float
    rho: float
    league: str
    
    # xG-based adjustments
    xg_attack: Dict[str, float] = field(default_factory=dict)  # xG-based attack ratings
    xg_defence: Dict[str, float] = field(default_factory=dict)  # xG-based defence ratings
    
    # Recent form (goal-based)
    recent_attack: Dict[str, float] = field(default_factory=dict)
    recent_defence: Dict[str, float] = field(default_factory=dict)
    
    # Recent form (xG-based) - more stable than goals
    recent_xg_attack: Dict[str, float] = field(default_factory=dict)
    recent_xg_defence: Dict[str, float] = field(default_factory=dict)
    
    # Overperformance tracking (for regression)
    overperformance: Dict[str, float] = field(default_factory=dict)
    
    # League characteristics
    league_avg_goals: float = 2.5
    league_avg_xg: float = 2.5
    league_btts_rate: float = 0.50
    
    # Competition type
    is_cup: bool = False
    home_adv_multiplier: float = 1.0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _time_weights(dates: pd.Series, league: str = None) -> np.ndarray:
    """Calculate time-based weights with league-specific decay"""
    config = get_league_config(league)
    half_life = config.get('decay_days', 365)
    
    dates = pd.to_datetime(dates, errors="coerce")
    maxd = dates.max()
    age = (maxd - dates).dt.days.values.astype(float)
    
    return np.exp(-np.log(2.0) * age / half_life)


def _dc_corr(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles correlation adjustment for low scores"""
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    elif x == 0 and y == 1:
        return 1 + lam * rho
    elif x == 1 and y == 0:
        return 1 + mu * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


def _neg_loglik(theta, n, home_idx, away_idx, hg, ag, w, rho_bounds=None):
    """Negative log-likelihood for DC model"""
    att = theta[:n] - np.mean(theta[:n])  # Mean-center attack
    deff = theta[n:2*n]
    home = theta[-2]
    rho = theta[-1]
    
    if rho_bounds:
        rho = np.clip(rho, rho_bounds[0], rho_bounds[1])
    
    lam = np.exp(att[home_idx] - deff[away_idx] + home)
    mu = np.exp(att[away_idx] - deff[home_idx])
    
    # Poisson log-likelihood
    logp = (
        -lam + hg * np.log(lam + 1e-12) - gammaln(hg + 1)
        - mu + ag * np.log(mu + 1e-12) - gammaln(ag + 1)
    )
    
    # DC correlation adjustment
    corr = np.array([_dc_corr(int(x), int(y), L, M, rho) 
                     for x, y, L, M in zip(hg, ag, lam, mu)])
    logp += np.log(np.maximum(corr, 1e-12))
    
    return -np.sum(w * logp)


def _score_grid(lam: float, mu: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Generate probability matrix for score combinations"""
    P = np.zeros((max_goals + 1, max_goals + 1))
    
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            base = np.exp(
                -lam + i * np.log(lam + 1e-12) - gammaln(i + 1)
                - mu + j * np.log(mu + 1e-12) - gammaln(j + 1)
            )
            P[i, j] = base * _dc_corr(i, j, lam, mu, rho)
    
    P /= P.sum()
    return P


# =============================================================================
# XG-BASED FORM CALCULATION
# =============================================================================

def _calculate_xg_form(df: pd.DataFrame, window: int = RECENT_FORM_WINDOW) -> Dict:
    """
    Calculate xG-based form (more stable than goal-based form)
    """
    df = df.sort_values('Date')
    teams = pd.unique(pd.concat([df['HomeTeam'], df['AwayTeam']]))
    
    xg_attack = {}
    xg_defence = {}
    overperf = {}
    
    league_avg_xg = df['home_xG_combined'].mean() if 'home_xG_combined' in df.columns else 1.3
    if pd.isna(league_avg_xg):
        league_avg_xg = 1.3
    
    for team in teams:
        # Get recent matches
        home_matches = df[df['HomeTeam'] == team].tail(window * 2)
        away_matches = df[df['AwayTeam'] == team].tail(window * 2)
        
        if len(home_matches) + len(away_matches) < 3:
            xg_attack[team] = 1.0
            xg_defence[team] = 1.0
            overperf[team] = 0.0
            continue
        
        # xG scored and conceded
        xg_for_home = home_matches['home_xG_combined'].dropna().tail(window)
        xg_for_away = away_matches['away_xG_combined'].dropna().tail(window)
        xg_against_home = home_matches['away_xG_combined'].dropna().tail(window)
        xg_against_away = away_matches['home_xG_combined'].dropna().tail(window)
        
        xg_for = pd.concat([xg_for_home, xg_for_away])
        xg_against = pd.concat([xg_against_home, xg_against_away])
        
        # Goals for comparison
        goals_for_home = home_matches['FTHG'].tail(window)
        goals_for_away = away_matches['FTAG'].tail(window)
        goals_for = pd.concat([goals_for_home, goals_for_away])
        
        if len(xg_for) > 0 and xg_for.mean() > 0:
            xg_attack[team] = xg_for.mean() / league_avg_xg
            xg_defence[team] = league_avg_xg / max(xg_against.mean(), 0.3)
            
            # Overperformance = goals - xG (positive = scoring above xG)
            if len(goals_for) > 0:
                overperf[team] = goals_for.mean() - xg_for.mean()
            else:
                overperf[team] = 0.0
        else:
            xg_attack[team] = 1.0
            xg_defence[team] = 1.0
            overperf[team] = 0.0
        
        # Clamp to reasonable ranges
        xg_attack[team] = max(0.6, min(1.5, xg_attack[team]))
        xg_defence[team] = max(0.6, min(1.5, xg_defence[team]))
        overperf[team] = max(-0.8, min(0.8, overperf[team]))
    
    return {
        'xg_attack': xg_attack,
        'xg_defence': xg_defence,
        'overperformance': overperf
    }


def _calculate_goal_form(df: pd.DataFrame, window: int = RECENT_FORM_WINDOW) -> Dict:
    """Calculate traditional goal-based form"""
    df = df.sort_values('Date')
    teams = pd.unique(pd.concat([df['HomeTeam'], df['AwayTeam']]))
    
    attack = {}
    defence = {}
    
    league_avg = (df['FTHG'].mean() + df['FTAG'].mean()) / 2
    
    for team in teams:
        home_matches = df[df['HomeTeam'] == team].tail(window)
        away_matches = df[df['AwayTeam'] == team].tail(window)
        
        if len(home_matches) + len(away_matches) < 3:
            attack[team] = 1.0
            defence[team] = 1.0
            continue
        
        goals_for = home_matches['FTHG'].sum() + away_matches['FTAG'].sum()
        goals_against = home_matches['FTAG'].sum() + away_matches['FTHG'].sum()
        matches = len(home_matches) + len(away_matches)
        
        attack[team] = (goals_for / matches) / max(league_avg, 0.5)
        defence[team] = max(league_avg, 0.5) / max(goals_against / matches, 0.3)
        
        attack[team] = max(0.6, min(1.5, attack[team]))
        defence[team] = max(0.6, min(1.5, defence[team]))
    
    return {'attack': attack, 'defence': defence}


# =============================================================================
# MODEL FITTING
# =============================================================================

def fit_league_xg(df_league: pd.DataFrame, use_xg: bool = True) -> DCParamsXG:
    """
    Fit enhanced DC model with xG integration
    
    Args:
        df_league: DataFrame with league data
        use_xg: Whether to incorporate xG into form calculations
    
    Returns:
        DCParamsXG with fitted parameters
    """
    df = df_league.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Date'])
    
    if len(df) < 30:
        raise ValueError(f"Insufficient data: {len(df)} matches (need 30+)")
    
    # Get league configuration
    league_name = df['League'].iloc[0] if 'League' in df.columns else 'Unknown'
    config = get_league_config(league_name)
    
    # Determine if cup competition
    is_cup = df['LeagueType'].iloc[0] == 'cup' if 'LeagueType' in df.columns else False
    
    # Team mapping
    teams = pd.unique(pd.concat([df['HomeTeam'], df['AwayTeam']]))
    t2i = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    
    # Map data
    home_idx = df['HomeTeam'].map(t2i).values
    away_idx = df['AwayTeam'].map(t2i).values
    hg = df['FTHG'].astype(int).values
    ag = df['FTAG'].astype(int).values
    w = _time_weights(df['Date'], league_name)
    
    # Initial parameters
    theta0 = np.zeros(2 * n + 2)
    theta0[-2] = 0.2  # Home advantage
    theta0[-1] = config['rho_init']
    
    # Optimize
    res = minimize(
        _neg_loglik,
        theta0,
        args=(n, home_idx, away_idx, hg, ag, w, config['rho_bounds']),
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-10}
    )
    
    # Extract parameters
    th = res.x
    att = th[:n] - np.mean(th[:n])
    deff = th[n:2*n]
    fitted_rho = np.clip(th[-1], config['rho_bounds'][0], config['rho_bounds'][1])
    fitted_home = th[-2]
    
    # Apply home advantage multiplier for cups
    home_adv_mult = config.get('home_adv_mult', 1.0)
    
    # League statistics
    league_avg_goals = df['FTHG'].mean() + df['FTAG'].mean()
    league_btts = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).mean()
    
    # xG statistics if available
    if 'home_xG_combined' in df.columns:
        league_avg_xg = df['home_xG_combined'].mean() + df['away_xG_combined'].mean()
    else:
        league_avg_xg = league_avg_goals
    
    # Create params object
    params = DCParamsXG(
        attack={t: float(att[i]) for t, i in t2i.items()},
        defence={t: float(deff[i]) for t, i in t2i.items()},
        home_adv=float(fitted_home),
        rho=float(fitted_rho),
        league=league_name,
        league_avg_goals=league_avg_goals,
        league_avg_xg=league_avg_xg if not pd.isna(league_avg_xg) else league_avg_goals,
        league_btts_rate=league_btts,
        is_cup=is_cup,
        home_adv_multiplier=home_adv_mult
    )
    
    # Calculate xG-based form
    if use_xg and 'home_xG_combined' in df.columns:
        xg_form = _calculate_xg_form(df)
        params.recent_xg_attack = xg_form['xg_attack']
        params.recent_xg_defence = xg_form['xg_defence']
        params.overperformance = xg_form['overperformance']
    
    # Calculate goal-based form
    goal_form = _calculate_goal_form(df)
    params.recent_attack = goal_form['attack']
    params.recent_defence = goal_form['defence']
    
    return params


def fit_all_xg(df: pd.DataFrame, use_xg: bool = True) -> Dict[str, DCParamsXG]:
    """
    Fit DC model to all leagues/competitions
    
    Args:
        df: DataFrame with all leagues
        use_xg: Whether to use xG features
    
    Returns:
        Dictionary mapping league code to DCParamsXG
    """
    params_by_league = {}
    
    for league, group in df.groupby('League', sort=False):
        try:
            params = fit_league_xg(group, use_xg=use_xg)
            params_by_league[league] = params
            
            config = get_league_config(league)
            cup_str = " (CUP)" if params.is_cup else ""
            logger.info(f"  Fitted {league}{cup_str}: {len(group)} matches, "
                       f"home_adv={params.home_adv:.3f}, "
                       f"rho={params.rho:.4f}")
        except Exception as e:
            logger.warning(f"  [WARN] DC fit skipped for {league}: {e}")
    
    return params_by_league


# =============================================================================
# ENHANCED MATCH PRICING
# =============================================================================

def price_match_xg(
    params: DCParamsXG,
    home: str,
    away: str,
    use_xg_form: bool = True,
    max_goals: int = MAX_GOALS,
    # xG features
    home_xG_for_ma5: Optional[float] = None,
    away_xG_for_ma5: Optional[float] = None,
    home_xG_overperformance: Optional[float] = None,
    away_xG_overperformance: Optional[float] = None,
    # Rest days
    home_rest_days: Optional[int] = None,
    away_rest_days: Optional[int] = None,
    home_had_cup_midweek: Optional[int] = None,
    away_had_cup_midweek: Optional[int] = None,
    # Injuries
    home_injuries_count: Optional[int] = None,
    away_injuries_count: Optional[int] = None,
    home_key_injuries: Optional[int] = None,
    away_key_injuries: Optional[int] = None,
    # Formations
    home_formation_attack: Optional[float] = None,
    away_formation_attack: Optional[float] = None,
    formation_matchup_goals_mult: Optional[float] = None,
    # H2H
    h2h_total_goals_avg: Optional[float] = None,
    h2h_btts_rate: Optional[float] = None,
    # Advanced stats
    home_attack_quality: Optional[float] = None,
    away_attack_quality: Optional[float] = None,
    # Cup
    is_cup_match: Optional[int] = None,
    is_knockout: Optional[int] = None,
    # Streaks
    home_scoring_streak: Optional[int] = None,
    away_conceding_streak: Optional[int] = None,
) -> Dict[str, float]:
    """
    Price a match with full xG integration and all features
    
    Returns:
        Dictionary of probabilities for all markets
    """
    if home not in params.attack or away not in params.attack:
        return {}
    
    # Get league config
    config = get_league_config(params.league)
    xg_weight = config.get('xg_weight', 0.5)
    
    # ==========================================================================
    # BASE EXPECTED GOALS (from fitted DC parameters)
    # ==========================================================================
    
    base_lam = np.exp(params.attack[home] - params.defence[away] + params.home_adv)
    base_mu = np.exp(params.attack[away] - params.defence[home])
    
    # ==========================================================================
    # XG-BASED FORM ADJUSTMENT (Key improvement!)
    # ==========================================================================
    
    lam = base_lam
    mu = base_mu
    
    if use_xg_form:
        # Prefer xG-based form over goal-based
        if params.recent_xg_attack:
            xg_att_h = params.recent_xg_attack.get(home, 1.0)
            xg_att_a = params.recent_xg_attack.get(away, 1.0)
            xg_def_h = params.recent_xg_defence.get(home, 1.0)
            xg_def_a = params.recent_xg_defence.get(away, 1.0)
            
            # Blend xG and goal-based form
            goal_att_h = params.recent_attack.get(home, 1.0)
            goal_att_a = params.recent_attack.get(away, 1.0)
            
            form_mult_home_att = xg_weight * xg_att_h + (1 - xg_weight) * goal_att_h
            form_mult_away_att = xg_weight * xg_att_a + (1 - xg_weight) * goal_att_a
            form_mult_home_def = xg_weight * xg_def_h + (1 - xg_weight) * params.recent_defence.get(home, 1.0)
            form_mult_away_def = xg_weight * xg_def_a + (1 - xg_weight) * params.recent_defence.get(away, 1.0)
            
            lam *= form_mult_home_att * (1 / form_mult_away_def)
            mu *= form_mult_away_att * (1 / form_mult_home_def)
        
        elif params.recent_attack:
            # Fallback to goal-based form
            lam *= params.recent_attack.get(home, 1.0) / params.recent_defence.get(away, 1.0)
            mu *= params.recent_attack.get(away, 1.0) / params.recent_defence.get(home, 1.0)
    
    # ==========================================================================
    # REGRESSION TO MEAN (Overperformance adjustment)
    # ==========================================================================
    
    if params.overperformance and home_xG_overperformance is None:
        # Use stored overperformance
        home_overperf = params.overperformance.get(home, 0)
        away_overperf = params.overperformance.get(away, 0)
    else:
        home_overperf = home_xG_overperformance or 0
        away_overperf = away_xG_overperformance or 0
    
    # Teams overperforming xG will regress - reduce their expected
    regression_strength = 0.3
    lam *= (1 - home_overperf * regression_strength * 0.5)
    mu *= (1 - away_overperf * regression_strength * 0.5)
    
    # ==========================================================================
    # REST DAYS ADJUSTMENT
    # ==========================================================================
    
    if home_rest_days is not None:
        if home_rest_days < 4:
            lam *= 0.88
        elif home_rest_days < 7:
            lam *= 0.95
    
    if away_rest_days is not None:
        if away_rest_days < 4:
            mu *= 0.88
        elif away_rest_days < 7:
            mu *= 0.95
    
    # Extra fatigue if midweek cup
    if home_had_cup_midweek == 1:
        lam *= 0.92
    if away_had_cup_midweek == 1:
        mu *= 0.92
    
    # ==========================================================================
    # INJURY ADJUSTMENT
    # ==========================================================================
    
    if home_injuries_count is not None and home_injuries_count > 3:
        lam *= 0.92
    if home_key_injuries is not None and home_key_injuries >= 1:
        lam *= 0.90  # Key player missing = significant impact
    
    if away_injuries_count is not None and away_injuries_count > 3:
        mu *= 0.92
    if away_key_injuries is not None and away_key_injuries >= 1:
        mu *= 0.90
    
    # ==========================================================================
    # FORMATION ADJUSTMENT
    # ==========================================================================
    
    if home_formation_attack is not None:
        # More attacking formation = slightly higher expected goals
        formation_adj = 0.7 + home_formation_attack * 0.3
        lam *= formation_adj
    
    if away_formation_attack is not None:
        formation_adj = 0.7 + away_formation_attack * 0.3
        mu *= formation_adj
    
    if formation_matchup_goals_mult is not None:
        lam *= np.sqrt(formation_matchup_goals_mult)
        mu *= np.sqrt(formation_matchup_goals_mult)
    
    # ==========================================================================
    # H2H ADJUSTMENT
    # ==========================================================================
    
    h2h_btts_adjustment = 1.0
    
    if h2h_total_goals_avg is not None:
        expected_total = lam + mu
        h2h_ratio = h2h_total_goals_avg / max(expected_total, 1.5)
        h2h_goals_adj = 0.75 + 0.25 * h2h_ratio  # 25% weight to H2H
        lam *= np.sqrt(h2h_goals_adj)
        mu *= np.sqrt(h2h_goals_adj)
    
    if h2h_btts_rate is not None:
        h2h_btts_adjustment = h2h_btts_rate / max(params.league_btts_rate, 0.3)
        h2h_btts_adjustment = max(0.85, min(1.15, h2h_btts_adjustment))
    
    # ==========================================================================
    # CUP COMPETITION ADJUSTMENTS
    # ==========================================================================
    
    if is_cup_match == 1 or params.is_cup:
        # Reduced home advantage in cups
        home_factor = params.home_adv_multiplier
        # Recalculate with reduced home advantage
        lam_adj = lam / np.exp(params.home_adv) * np.exp(params.home_adv * home_factor)
        lam = lam_adj
    
    if is_knockout == 1:
        # Knockout games tend to be tighter
        lam *= 0.95
        mu *= 0.95
    
    # ==========================================================================
    # ENSURE REASONABLE BOUNDS
    # ==========================================================================
    
    lam = max(0.3, min(4.0, lam))
    mu = max(0.2, min(3.5, mu))
    
    # ==========================================================================
    # GENERATE SCORE GRID
    # ==========================================================================
    
    P = _score_grid(lam, mu, params.rho, max_goals)
    
    out = {}
    
    # 1X2 probabilities
    out['DC_1X2_H'] = float(np.tril(P, -1).sum())
    out['DC_1X2_D'] = float(np.trace(P))
    out['DC_1X2_A'] = float(np.triu(P, 1).sum())
    
    # ==========================================================================
    # BTTS WITH STREAK ADJUSTMENT
    # ==========================================================================
    
    base_btts_yes = float(P[1:, 1:].sum())
    btts_mult = 1.0
    
    if home_scoring_streak is not None and home_scoring_streak >= 3:
        btts_mult *= 1.05
    if away_conceding_streak is not None and away_conceding_streak >= 3:
        btts_mult *= 1.05
    
    btts_adjusted = base_btts_yes * btts_mult * h2h_btts_adjustment
    btts_adjusted = max(0.10, min(0.90, btts_adjusted))
    
    out['DC_BTTS_Y'] = btts_adjusted
    out['DC_BTTS_N'] = 1 - btts_adjusted
    
    # ==========================================================================
    # OVER/UNDER
    # ==========================================================================
    
    S = np.add.outer(np.arange(P.shape[0]), np.arange(P.shape[1]))
    
    for line in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
        line_str = str(line).replace('.', '_')
        over_prob = float(P[S > line].sum())
        out[f'DC_OU_{line_str}_O'] = over_prob
        out[f'DC_OU_{line_str}_U'] = 1 - over_prob
    
    # Store expected goals for reference
    out['expected_home_goals'] = lam
    out['expected_away_goals'] = mu
    out['expected_total_goals'] = lam + mu
    
    return out


# =============================================================================
# CONVENIENCE ALIASES
# =============================================================================

# Backward compatibility
fit_league = fit_league_xg
fit_all = fit_all_xg
price_match = price_match_xg


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    
    # Create sample data
    teams = ['Arsenal', 'Chelsea', 'Liverpool', 'Man City', 'Man United',
             'Tottenham', 'Newcastle', 'Aston Villa']
    
    data = []
    for i in range(200):
        home_idx = i % len(teams)
        away_idx = (i + 3) % len(teams)
        
        home_goals = np.random.poisson(1.5)
        away_goals = np.random.poisson(1.2)
        home_xg = home_goals + np.random.normal(0, 0.3)
        away_xg = away_goals + np.random.normal(0, 0.3)
        
        data.append({
            'Date': pd.Timestamp('2023-01-01') + pd.Timedelta(days=i*2),
            'League': 'E0',
            'LeagueType': 'league',
            'HomeTeam': teams[home_idx],
            'AwayTeam': teams[away_idx],
            'FTHG': min(home_goals, 5),
            'FTAG': min(away_goals, 4),
            'home_xG_combined': max(0.1, home_xg),
            'away_xG_combined': max(0.1, away_xg),
        })
    
    df = pd.DataFrame(data)
    
    print("Testing xG-integrated DC model...")
    print("="*60)
    
    # Fit model
    params = fit_league_xg(df, use_xg=True)
    
    print(f"\nFitted parameters:")
    print(f"  Home advantage: {params.home_adv:.4f}")
    print(f"  Rho: {params.rho:.4f}")
    print(f"  League avg goals: {params.league_avg_goals:.2f}")
    print(f"  League avg xG: {params.league_avg_xg:.2f}")
    
    # Price a match
    probs = price_match_xg(
        params, 'Arsenal', 'Chelsea',
        use_xg_form=True,
        home_rest_days=4,
        away_rest_days=7,
        home_injuries_count=2,
        is_cup_match=0
    )
    
    print(f"\nArsenal vs Chelsea:")
    print(f"  Expected goals: {probs['expected_home_goals']:.2f} - {probs['expected_away_goals']:.2f}")
    print(f"  BTTS Yes: {probs['DC_BTTS_Y']:.1%}")
    print(f"  Over 2.5: {probs['DC_OU_2_5_O']:.1%}")
    print(f"  1X2: H={probs['DC_1X2_H']:.1%}, D={probs['DC_1X2_D']:.1%}, A={probs['DC_1X2_A']:.1%}")
