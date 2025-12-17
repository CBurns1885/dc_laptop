# Enhancement #2: Home/Away Attack Strength Split

## Overview
Modifies the Dixon-Coles model to fit **separate attack parameters** for home vs away performances, rather than assuming a team's attack strength is constant regardless of venue.

## Current Limitation
Standard DC model:
```
Expected home goals = exp(attack[home] - defence[away] + home_advantage)
Expected away goals = exp(attack[away] - defence[home])
```

Problem: `attack[team]` is the same whether team plays home or away. But real data shows massive asymmetry!

## Real-World Examples

| Team | Home Goals/Game | Away Goals/Game | Difference |
|------|----------------|-----------------|------------|
| Burnley 22/23 | 1.2 | 0.6 | **+100%** |
| Atletico Madrid | 1.8 | 1.2 | **+50%** |
| Man City | 2.8 | 2.5 | +12% |
| Brighton | 1.6 | 1.8 | **-11%** (better away!) |

## Proposed Solution
Split attack into home and away components:
```
Expected home goals = exp(attack_home[home] - defence_away[away] + global_home_adv)
Expected away goals = exp(attack_away[away] - defence_home[home])
```

Now we have **2× attack parameters** (attack_home AND attack_away per team), plus **2× defence parameters** (defence_home AND defence_away).

## Implementation Complexity
⚠️ **This is a MAJOR change** to the DC fitting logic.

### Current Parameter Count
For a 20-team league:
- 20 attack parameters
- 20 defence parameters
- 1 home advantage
- 1 rho
- **Total: 42 parameters**

### New Parameter Count
For a 20-team league:
- 20 attack_home parameters
- 20 attack_away parameters
- 20 defence_home parameters
- 20 defence_away parameters
- 1 global home advantage (or remove, since it's baked into attack_home vs attack_away difference)
- 1 rho
- **Total: 81 or 82 parameters**

### Optimization Challenge
- Nearly **2× the parameters** to fit
- Requires **more data** per league (risk of overfitting with small leagues)
- Longer optimization time (2-3× slower)

## Benefits

### Accuracy Gains
Expected improvements:
- **1X2 predictions**: +3-5% (if we were predicting match results, which we're not)
- **BTTS predictions**: +1-2% (teams that score more at home also concede more at home)
- **O/U predictions**: +2-4% (total goals more accurate when home/away split)

### Better Team Profiling
Can now identify:
- **Home Fortresses**: Teams that dramatically overperform at home
- **Road Warriors**: Teams that perform better away (rare but exists)
- **Consistent Teams**: Teams with minimal home/away split (Man City, Liverpool)

## Implementation Steps

### 1. Modify DCParams Dataclass
```python
@dataclass
class DCParams:
    attack_home: Dict[str, float]  # NEW
    attack_away: Dict[str, float]  # NEW
    defence_home: Dict[str, float]  # NEW
    defence_away: Dict[str, float]  # NEW
    home_adv: float  # Might be reduced or removed
    rho: float
    league: str
```

### 2. Modify Negative Log-Likelihood
```python
def _neg_loglik_asymmetric(theta, n, home_idx, away_idx, hg, ag, w):
    # Extract 4n + 2 parameters instead of 2n + 2
    att_home = theta[:n]
    att_away = theta[n:2*n]
    def_home = theta[2*n:3*n]
    def_away = theta[3*n:4*n]
    home = theta[-2]
    rho = theta[-1]

    # Expected goals with asymmetric parameters
    lam = np.exp(att_home[home_idx] - def_away[away_idx] + home)
    mu = np.exp(att_away[away_idx] - def_home[home_idx])

    # ... rest of log-likelihood calculation
```

### 3. Update Optimization
```python
def fit_league_asymmetric(df_league):
    n = len(teams)
    theta0 = np.zeros(4 * n + 2)  # 4n instead of 2n

    # Initialize with symmetric values, then optimize
    res = minimize(_neg_loglik_asymmetric, theta0, ...)
```

### 4. Update price_match()
```python
def price_match(params, home, away):
    lam = np.exp(params.attack_home[home] - params.defence_away[away] + params.home_adv)
    mu = np.exp(params.attack_away[away] - params.defence_home[home])
```

## Risks & Mitigations

### Risk 1: Overfitting
**Problem**: Doubling parameters with same data = overfitting risk

**Mitigation**:
- Require minimum 100 matches per league (vs 50 currently)
- Add L2 regularization: `penalty = lambda * (attack_home² + attack_away²)`
- Cross-validate: Check if asymmetric model actually improves out-of-sample

### Risk 2: Small Sample Sizes
**Problem**: Newly promoted teams have only ~5 home games and ~5 away games in sample

**Mitigation**:
- Hierarchical priors: Pull attack_home and attack_away toward league mean
- Start with symmetric model, gradually allow asymmetry as data accumulates
- Use Bayesian approach with informative priors

### Risk 3: Computation Time
**Problem**: 2× parameters = slower fitting

**Mitigation**:
- Use better optimization algorithm (L-BFGS-B with good initial guess)
- Parallelize across leagues
- Cache fitted parameters (only refit when new data available)

## Decision: Should We Implement?

### ✅ Arguments FOR
1. **Huge real-world effect**: Some teams have 50-100% home/away split
2. **Academic support**: Modern DC implementations use this
3. **Better predictions**: Especially for O/U markets

### ❌ Arguments AGAINST
1. **Complexity**: Significant code changes required
2. **Data requirements**: Need more matches per league
3. **Diminishing returns**: We're focused on BTTS & O/U, not 1X2

### 🎯 Recommendation
**Implement as OPTIONAL enhancement**:
- Add `use_asymmetric=False` parameter to `fit_league()`
- Default to symmetric (current model) for stability
- Enable asymmetric for major leagues with lots of data (E0, D1, SP1, I1, F1)
- Compare backtests to validate improvement

## Validation Plan

1. **Backtest both models** (symmetric vs asymmetric)
2. **Compare on major leagues** (E0, D1, SP1) with 1000+ matches
3. **Metrics to compare**:
   - Log loss (calibration)
   - Accuracy (% correct predictions)
   - ROI (profitability)
4. **If asymmetric wins by >2% log loss**: implement for all leagues
5. **If marginal (<1% improvement)**: keep symmetric for simplicity

## Sample Code Snippet

```python
# In models_dc.py
def fit_league(df_league, use_asymmetric=False):
    if use_asymmetric:
        return _fit_asymmetric(df_league)
    else:
        return _fit_symmetric(df_league)  # Current implementation

def _fit_asymmetric(df_league):
    # ... 4n parameter optimization ...
    pass
```

---

**Status**: 📋 Planned (Not Yet Implemented)
**Impact**: ⭐⭐⭐⭐ (High)
**Effort**: ⭐⭐⭐⭐ (High - Major refactor)
**Testing**: Requires extensive backtesting before production

**Next Step**: Implement as experimental feature with `use_asymmetric` flag, backtest on E0/D1/SP1 to validate improvement justifies complexity.
