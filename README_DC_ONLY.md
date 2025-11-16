# DC Laptop - Dixon-Coles Only: BTTS & Over/Under (0.5-5.5)

## Overview

This is a **streamlined, Dixon-Coles only** football prediction system focused exclusively on:
- **BTTS (Both Teams To Score)** - Yes/No predictions
- **Over/Under Goal Lines** - 0.5, 1.5, 2.5, 3.5, 4.5, 5.5 goals

All other markets (1X2, Asian Handicaps, Correct Scores, Half Time, Cards, Corners, etc.) have been removed to create a **pure Dixon-Coles implementation** optimized for goal-based markets.

## What is Dixon-Coles?

The Dixon-Coles model is a statistical framework specifically designed for football prediction. It models goal scoring as independent Poisson processes while accounting for low-scoring game correlations.

### Key Features:
- **Team Strength Modeling**: Attack and defense parameters for each team
- **Home Advantage**: League-specific home field advantage
- **Time Decay**: Recent matches weighted more heavily
- **Low-Score Correlation**: Adjusts for dependency in 0-0, 1-0, 0-1, 1-1 scorelines
- **Form Adjustments**: Recent form multipliers for attack/defense

## Supported Markets

### 1. Both Teams To Score (BTTS)
- **BTTS Yes**: Both teams score at least 1 goal
- **BTTS No**: At least one team fails to score

### 2. Over/Under Goal Lines
- **O/U 0.5**: Over 0.5 goals (at least 1 goal) / Under 0.5 (no goals)
- **O/U 1.5**: Over/Under 1.5 total goals
- **O/U 2.5**: Over/Under 2.5 total goals (most popular market)
- **O/U 3.5**: Over/Under 3.5 total goals
- **O/U 4.5**: Over/Under 4.5 total goals
- **O/U 5.5**: Over/Under 5.5 total goals

## System Architecture

### Core Components

#### 1. **models_dc.py**
Dixon-Coles model implementation with:
- League-specific time decay (adaptive half-life)
- Score probability grid calculation
- Enhanced correlation for low scores
- Recent form adjustments

#### 2. **features.py**
Feature engineering focused on DC inputs:
- Historical results (FTHG, FTAG)
- Elo ratings (attack/defense strength proxies)
- Rolling form features
- Only generates y_BTTS and y_OU_X_X targets

#### 3. **models.py**
Simplified to **DC-only**:
- No ensemble models (RF, XGB, LightGBM removed)
- No stacking or calibration
- Direct Dixon-Coles probability output
- Supports only y_BTTS and y_OU_{line} targets

#### 4. **predict.py**
Prediction engine:
- Fits DC model per league
- Generates probabilities for BTTS and O/U markets
- League-specific calibration
- Outputs weekly predictions

#### 5. **bet_finder.py**
Simplified bet finder for DC markets only:
- Scans BTTS Yes/No opportunities
- Scans all O/U lines (0.5-5.5)
- DC probability validation
- Kelly Criterion stake sizing
- HTML report generation

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Historical Data
```bash
python download_football_data.py
```

### 3. Generate Features (First Time Setup)
```bash
python features.py
```

### 4. Run Weekly Predictions
```bash
python run_weekly.py
```

This will:
- Download latest fixtures
- Fit DC models per league (incremental update)
- Generate BTTS and O/U predictions
- Output to `outputs/YYYY-MM-DD/weekly_bets_lite.csv`

### 5. Find Quality Bets
```bash
python bet_finder.py
```

Outputs:
- `quality_bets_dc_YYYYMMDD.csv` - All quality bets
- `quality_bets_dc_YYYYMMDD.html` - Beautiful HTML report

## Configuration

### config.py
```python
# DC-only configuration
PRIMARY_TARGETS = ["BTTS", "OU"]
USE_ELO = True  # Helps DC model
USE_ROLLING_FORM = True  # Recent form adjustments
USE_MARKET_FEATURES = False  # DC doesn't need bookmaker odds
TRAIN_SEASONS_BACK = 8  # Historical data depth
```

### Bet Finder Thresholds (bet_finder.py)
```python
CONFIG = {
    'min_prob_btts': 0.65,      # 65% minimum for BTTS
    'min_prob_ou': 0.65,        # 65% minimum for O/U
    'min_confidence': 0.60,     # 60% model confidence
    'min_agreement': 70.0,      # 70% model agreement
    'dc_validation': True,      # Validate with DC probabilities
    'dc_max_diff': 0.15,        # Max 15% difference from DC
}
```

## Model Parameters

### League-Specific Time Decay
Different leagues have different half-lives for time weighting:
- **Premier League (E0)**: 365 days (stable, top quality)
- **Championship (E1)**: 320 days (more volatile)
- **Bundesliga (D1)**: 380 days (very stable)
- **La Liga (SP1)**: 390 days
- **Default**: 400 days for other leagues

### Home Advantage
Fitted per league during model training. Typically:
- England: 0.10-0.13 (moderate)
- Spain/Portugal: 0.14-0.16 (strong)
- Germany/Netherlands: 0.08-0.10 (weaker)

### Rho Parameter
Correlation adjustment for low-scoring games (0-0, 1-0, 0-1, 1-1).
- Typically: 0.03-0.08
- Negative values indicate these scores are less likely than independent Poisson would suggest

## Incremental Training

The system is designed for **weekly incremental updates**:

1. **First Run**: Full training from scratch (8+ seasons)
2. **Weekly Updates**: Only processes new results
   - Features are appended to existing parquet file
   - DC model refits quickly (lightweight)
   - No expensive ML model retraining needed

### Incremental Training Command
```bash
python incremental_trainer.py
```

This:
- Fetches only new results since last run
- Appends to `features.parquet`
- Refits DC parameters (fast, ~seconds per league)

## Accuracy Optimization

### For Maximum DC Accuracy:

1. **Use More Historical Data**
   - Set `TRAIN_SEASONS_BACK = 10` or more
   - More data = better parameter estimates

2. **League-Specific Tuning**
   - Review league profiles in `predict.py`
   - Adjust decay half-lives in `models_dc.py`

3. **Form Window Tuning**
   - Default: Last 5 matches
   - Try 3-8 match windows in `models_dc.py`

4. **Max Goals Parameter**
   - Default: 12 goals maximum in probability grid
   - Increase for high-scoring leagues

5. **Rho Constraints**
   - Constrain rho to [-0.2, 0.0] for stability
   - Or allow free optimization for flexibility

## Output Files

### Weekly Predictions
`outputs/YYYY-MM-DD/weekly_bets_lite.csv`:
```
League,Date,HomeTeam,AwayTeam,
P_BTTS_Y,P_BTTS_N,
P_OU_0_5_O,P_OU_0_5_U,
P_OU_1_5_O,P_OU_1_5_U,
...
DC_BTTS_Y,DC_BTTS_N,
DC_OU_0_5_O,DC_OU_0_5_U,
...
```

### Quality Bets
`outputs/YYYY-MM-DD/quality_bets_dc_YYYYMMDD.csv`:
```
League,Date,HomeTeam,AwayTeam,Market,Selection,
Probability,Confidence,Agreement,DC_Probability,ImpliedOdds,Kelly%
```

### DC Model Parameters
Stored in memory during run, not persisted.
To save:
```python
import joblib
from models_dc import fit_all
params = fit_all(historical_df)
joblib.dump(params, 'models/dc_params.joblib')
```

## Supported Leagues

### England
- E0: Premier League
- E1: Championship
- E2: League One
- E3: League Two
- EC: National League

### Major European
- SP1, SP2: Spain (La Liga, Segunda)
- I1, I2: Italy (Serie A, Serie B)
- D1, D2: Germany (Bundesliga, 2. Bundesliga)
- F1, F2: France (Ligue 1, Ligue 2)

### Others
- N1: Netherlands (Eredivisie)
- B1: Belgium (Pro League)
- P1: Portugal (Primeira Liga)
- SC0-SC3: Scotland
- T1: Turkey (Super Lig)
- G1: Greece

## Backtest Examples

To validate the DC model historically:
```bash
python backtest.py --markets BTTS OU_2_5 --lookback-days 365
```

This will:
- Test last year of predictions
- Calculate accuracy, calibration, profit/loss
- Generate backtest report

Expected performance (BTTS & O/U 2.5):
- **Accuracy**: 55-60% (above random)
- **Log Loss**: 0.60-0.65 (well-calibrated)
- **ROI**: -2% to +5% (market-dependent, bookmaker margin)

## Files Removed from Original System

The following files were removed to create the DC-only system:
- `bet_finder_all_markets.py` - Replaced with `bet_finder.py`
- `model_binary.py`, `model_multiclass.py`, `model_ordinal.py` - No ML models
- `calibration.py`, `ordinal.py` - No calibration needed
- `ensemble_blender.py`, `blending.py` - No ensemble
- `tuning.py` - No hyperparameter tuning
- `model_enhancer.py` - No model stacking
- `accumulator_finder.py`, `acc_builder.py` - Multi-market features

## Advantages of DC-Only Approach

1. **Speed**: DC model fits in seconds (vs hours for ML ensembles)
2. **Interpretability**: Clear parameters (attack, defense, home advantage)
3. **Incremental Learning**: Easy to update with new data
4. **No Overfitting**: Statistical model, not data-driven ML
5. **Goal-Market Focused**: Optimized specifically for BTTS and O/U
6. **No Feature Engineering**: Only needs historical scores
7. **Proven Foundation**: Academic research-backed (Dixon & Coles, 1997)

## Limitations

1. **No 1X2 Betting**: Doesn't predict match winner (by design)
2. **No Exotic Markets**: No cards, corners, etc.
3. **Assumes Poisson**: May underperform for very low/high scoring teams
4. **No In-Play**: Pre-match predictions only
5. **League-Level Only**: Doesn't model head-to-head history

## Future Enhancements

- [ ] Bayesian inference for uncertainty quantification
- [ ] Player-level impact adjustments (injuries, transfers)
- [ ] Weather/pitch condition features
- [ ] Head-to-head modifiers
- [ ] In-play live model updates
- [ ] Multi-league correlation modeling

## References

- **Original Paper**: Dixon, M. J., & Coles, S. G. (1997). "Modelling Association Football Scores and Inefficiencies in the Football Betting Market". Journal of the Royal Statistical Society.
- **Implementation Guide**: Baio, G., & Blangiardo, M. (2010). "Bayesian hierarchical model for the prediction of football results".

## Support

For issues, questions, or improvements:
1. Check the `QUICK_REFERENCE.md` for common problems
2. Review `PROJECT_ANALYSIS.md` for architecture details
3. See `XGBOOST_VS_GAUSSIAN_GUIDE.md` for technical comparisons

## License

MIT License - See LICENSE file

---

**Disclaimer**: This is a statistical modeling tool for educational and research purposes. Betting involves risk. Always bet responsibly and within your means. Past performance is not indicative of future results.
