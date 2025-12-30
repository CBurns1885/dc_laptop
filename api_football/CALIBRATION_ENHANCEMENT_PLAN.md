# Calibration Enhancement Plan
## Objective: Achieve Highest-Level Probability Calibration

Generated: 2025-12-29

---

## Current State Assessment

### ✅ Strengths
1. **Comprehensive Calibration Metrics**
   - Brier Score Decomposition (Reliability, Resolution, Uncertainty)
   - ECE & MCE calculations
   - Calibration curves (reliability diagrams)
   - Per-league and per-market analysis

2. **Multiple Calibration Methods**
   - Platt Scaling (parametric)
   - Isotonic Regression (non-parametric, monotonic)
   - Binning Calibration (simple histogram approach)

3. **Integration with xG-Enhanced DC Model**
   - xG-weighted predictions
   - Injury adjustments
   - Formation awareness
   - League-specific parameters

4. **Automated Workflow**
   - `backtest_calibration.py` for end-to-end analysis
   - Exportable configs for production use
   - Feature importance tracking

---

## Enhancement Recommendations

### 🔥 Priority 1: Advanced Calibration Methods

#### 1.1 Beta Calibration
**Why:** Better than Platt for skewed probability distributions, which is common in football (e.g., BTTS often has ~50% base rate, but O/U can be 65%+)

**Implementation:**
```python
# Add to calibration.py
class BetaCalibrator:
    """
    Beta calibration - generalizes Platt scaling
    Maps probabilities via Beta distribution CDF
    Better for skewed distributions
    """
    def __init__(self):
        self.a = 1.0  # Shape parameter
        self.b = 1.0  # Shape parameter
        self.m = 0.0  # Location shift
        self.fitted = False

    def fit(self, y_true, y_prob, max_iter=100):
        from scipy.special import betaln, digamma
        from scipy.optimize import minimize

        def neg_log_likelihood(params):
            a, b, m = params
            if a <= 0 or b <= 0:
                return 1e10

            # Transform probs
            p_transformed = np.clip(y_prob + m, 0.01, 0.99)

            # Beta log-likelihood
            ll = (
                y_true * np.log(p_transformed + 1e-10) * (a - 1) +
                (1 - y_true) * np.log(1 - p_transformed + 1e-10) * (b - 1) -
                betaln(a, b)
            )
            return -np.sum(ll)

        result = minimize(neg_log_likelihood, [1.0, 1.0, 0.0],
                         method='L-BFGS-B',
                         bounds=[(0.1, 10), (0.1, 10), (-0.2, 0.2)])

        self.a, self.b, self.m = result.x
        self.fitted = True
        return self

    def transform(self, y_prob):
        if not self.fitted:
            return y_prob

        from scipy.stats import beta
        p_transformed = np.clip(y_prob + self.m, 0.01, 0.99)
        return beta.cdf(p_transformed, self.a, self.b)
```

**Expected Improvement:** 2-5% reduction in ECE, especially for markets with skewed distributions

---

#### 1.2 Temperature Scaling with Learned Temperature per Market
**Why:** Different markets may need different temperature adjustments

**Implementation:**
```python
# Enhanced TemperatureScaler in calibration.py
class AdvancedTemperatureScaler:
    """
    Temperature scaling with learned temperature parameter
    Optimized via log-likelihood on validation set
    """
    def __init__(self):
        self.temperature = 1.0
        self.fitted = False

    def fit(self, y_true, y_logits, lr=0.01, max_iter=100):
        """
        Fit optimal temperature parameter

        Args:
            y_true: Binary labels
            y_logits: Logits (not probabilities) - use logit(prob)
        """
        from scipy.optimize import minimize_scalar

        def nll(temp):
            # Apply temperature
            scaled_logits = y_logits / temp
            probs = 1 / (1 + np.exp(-scaled_logits))
            probs = np.clip(probs, 1e-10, 1 - 1e-10)

            # Negative log likelihood
            return -np.mean(
                y_true * np.log(probs) +
                (1 - y_true) * np.log(1 - probs)
            )

        result = minimize_scalar(nll, bounds=(0.1, 5.0), method='bounded')
        self.temperature = result.x
        self.fitted = True

        logger.info(f"Fitted temperature: {self.temperature:.3f}")
        return self

    def transform(self, y_logits):
        """Apply temperature scaling"""
        if not self.fitted:
            return 1 / (1 + np.exp(-y_logits))

        scaled_logits = y_logits / self.temperature
        return 1 / (1 + np.exp(-scaled_logits))
```

**Expected Improvement:** 1-3% ECE reduction with minimal computational cost

---

#### 1.3 Ensemble Calibration
**Why:** Combine multiple calibration methods for robustness

**Implementation:**
```python
# Add to calibration.py
class EnsembleCalibrator:
    """
    Ensemble of calibration methods with learned weights
    Combines: Isotonic, Platt, Beta via weighted average
    """
    def __init__(self):
        self.calibrators = {
            'isotonic': IsotonicCalibrator(),
            'platt': PlattScaler(),
            'beta': BetaCalibrator()
        }
        self.weights = {}
        self.fitted = False

    def fit(self, y_true, y_prob, validation_split=0.3):
        """
        Fit all calibrators and learn optimal weights
        Uses validation set to determine weights
        """
        # Split data
        n = len(y_true)
        n_val = int(n * validation_split)
        indices = np.random.permutation(n)

        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        # Fit calibrators on training set
        for name, cal in self.calibrators.items():
            cal.fit(y_true[train_idx], y_prob[train_idx])

        # Evaluate on validation set
        val_true = y_true[val_idx]
        val_prob = y_prob[val_idx]

        # Get calibrated predictions from each method
        cal_preds = {}
        scores = {}
        for name, cal in self.calibrators.items():
            cal_preds[name] = cal.transform(val_prob)
            # Score using Brier score (lower is better)
            scores[name] = calculate_brier_score(val_true, cal_preds[name])

        # Learn weights via softmax of inverse Brier scores
        inv_scores = {k: 1 / (v + 1e-6) for k, v in scores.items()}
        total = sum(inv_scores.values())
        self.weights = {k: v / total for k, v in inv_scores.items()}

        logger.info(f"Ensemble weights: {self.weights}")
        self.fitted = True
        return self

    def transform(self, y_prob):
        """Weighted ensemble of calibrated predictions"""
        if not self.fitted:
            return y_prob

        calibrated = np.zeros_like(y_prob)
        for name, cal in self.calibrators.items():
            calibrated += self.weights[name] * cal.transform(y_prob)

        return calibrated
```

**Expected Improvement:** 3-7% ECE reduction, more robust across different probability ranges

---

### 🎯 Priority 2: Calibration-Aware Model Training

#### 2.1 Focal Loss Integration
**Why:** Standard log-loss treats all errors equally. Focal loss down-weights well-classified examples, forcing model to improve on hard cases.

**Implementation:**
```python
# Add to models_dc_xg.py
def _neg_loglik_focal(theta, n, home_idx, away_idx, hg, ag, w,
                      gamma=2.0, alpha=0.25):
    """
    Negative log-likelihood with Focal Loss

    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    where p_t = p if y=1 else (1-p)

    gamma > 0: down-weights easy examples
    alpha: balancing parameter
    """
    # Standard DC calculation
    att = theta[:n] - np.mean(theta[:n])
    deff = theta[n:2*n]
    home = theta[-2]
    rho = theta[-1]

    lam = np.exp(att[home_idx] - deff[away_idx] + home)
    mu = np.exp(att[away_idx] - deff[home_idx])

    # Poisson probabilities
    p_home = np.exp(-lam + hg * np.log(lam + 1e-12) - gammaln(hg + 1))
    p_away = np.exp(-mu + ag * np.log(mu + 1e-12) - gammaln(ag + 1))

    # Apply focal loss weighting
    p_t = p_home * p_away
    focal_weight = alpha * (1 - p_t + 1e-8) ** gamma

    # DC correlation
    corr = np.array([_dc_corr(int(x), int(y), L, M, rho)
                     for x, y, L, M in zip(hg, ag, lam, mu)])

    logp = np.log(p_t * np.maximum(corr, 1e-12))

    return -np.sum(w * focal_weight * logp)
```

**Expected Improvement:** Better calibration on rare events (e.g., high-scoring games, 0-0 draws)

---

#### 2.2 Calibration as Training Objective
**Why:** Explicitly optimize for calibration during training

**Implementation:**
```python
# Add to models_dc_xg.py
def fit_league_xg_calibrated(df_league, use_xg=True, calibration_weight=0.3):
    """
    Fit DC model with explicit calibration objective

    Loss = (1 - alpha) * NLL + alpha * ECE
    where alpha = calibration_weight
    """
    # ... existing setup ...

    def objective(theta):
        # Standard negative log-likelihood
        nll = _neg_loglik(theta, n, home_idx, away_idx, hg, ag, w)

        # Calculate ECE on training set (with time weighting)
        # Get predicted probabilities for BTTS, O/U
        params_temp = _theta_to_params(theta, team_list, league)

        btts_probs = []
        actual_btts = []

        for i in range(len(df_league)):
            probs = price_match_xg(params_temp,
                                   df_league.iloc[i]['HomeTeam'],
                                   df_league.iloc[i]['AwayTeam'])
            if probs:
                btts_probs.append(probs.get('DC_BTTS_Y', 0.5))
                actual_btts.append(1 if (df_league.iloc[i]['FTHG'] > 0 and
                                         df_league.iloc[i]['FTAG'] > 0) else 0)

        # Calculate ECE
        from calibration import calculate_ece_mce
        if len(btts_probs) > 10:
            ece, _ = calculate_ece_mce(np.array(actual_btts),
                                       np.array(btts_probs))
        else:
            ece = 0

        # Combined objective
        return (1 - calibration_weight) * nll + calibration_weight * ece * 1000

    # ... rest of optimization ...
```

**Expected Improvement:** Model learns to produce inherently better-calibrated probabilities

---

### 🔧 Priority 3: Production Integration Enhancements

#### 3.1 Adaptive Calibration per Probability Bin
**Why:** Calibration may vary by confidence level (e.g., model might be well-calibrated at 50% but poorly at 90%)

**Implementation:**
```python
# Add to calibration.py
class AdaptiveBinCalibrator:
    """
    Apply different calibration methods per probability bin

    E.g., use Platt for mid-range [0.3-0.7], Isotonic for extremes
    """
    def __init__(self, bin_edges=None):
        self.bin_edges = bin_edges or [0.0, 0.3, 0.7, 1.0]
        self.calibrators = []
        self.fitted = False

    def fit(self, y_true, y_prob):
        n_bins = len(self.bin_edges) - 1
        self.calibrators = []

        for i in range(n_bins):
            # Determine best calibrator for this bin
            mask = (y_prob >= self.bin_edges[i]) & (y_prob < self.bin_edges[i+1])

            if np.sum(mask) < 20:
                # Not enough data, use identity
                self.calibrators.append(None)
                continue

            bin_true = y_true[mask]
            bin_prob = y_prob[mask]

            # Try multiple methods, keep best
            best_cal = None
            best_score = float('inf')

            for cal_class in [PlattScaler, IsotonicCalibrator, BinningCalibrator]:
                cal = cal_class()
                try:
                    cal.fit(bin_true, bin_prob)
                    cal_prob = cal.transform(bin_prob)
                    score = calculate_brier_score(bin_true, cal_prob)

                    if score < best_score:
                        best_score = score
                        best_cal = cal
                except:
                    continue

            self.calibrators.append(best_cal)

        self.fitted = True
        return self

    def transform(self, y_prob):
        if not self.fitted:
            return y_prob

        calibrated = y_prob.copy()

        for i in range(len(self.bin_edges) - 1):
            mask = (y_prob >= self.bin_edges[i]) & (y_prob < self.bin_edges[i+1])

            if self.calibrators[i] is not None and np.sum(mask) > 0:
                calibrated[mask] = self.calibrators[i].transform(y_prob[mask])

        return calibrated
```

**Expected Improvement:** 5-10% ECE reduction by specializing calibration per confidence region

---

#### 3.2 Online Calibration Updates
**Why:** Calibration can drift over time as leagues evolve

**Implementation:**
```python
# Add to calibration.py
class OnlineCalibrator:
    """
    Incrementally update calibration as new data arrives
    Uses exponential moving average
    """
    def __init__(self, base_calibrator, decay=0.95):
        self.base_calibrator = base_calibrator
        self.decay = decay  # Weight for old data
        self.online_buffer = {'y_true': [], 'y_prob': []}
        self.buffer_size = 100

    def update(self, y_true_new, y_prob_new):
        """Update with new observations"""
        self.online_buffer['y_true'].extend(y_true_new)
        self.online_buffer['y_prob'].extend(y_prob_new)

        # Keep only recent buffer_size observations
        if len(self.online_buffer['y_true']) > self.buffer_size:
            self.online_buffer['y_true'] = self.online_buffer['y_true'][-self.buffer_size:]
            self.online_buffer['y_prob'] = self.online_buffer['y_prob'][-self.buffer_size:]

        # Refit if enough data
        if len(self.online_buffer['y_true']) >= 50:
            self.base_calibrator.fit(
                np.array(self.online_buffer['y_true']),
                np.array(self.online_buffer['y_prob'])
            )

    def transform(self, y_prob):
        return self.base_calibrator.transform(y_prob)
```

**Expected Improvement:** Maintain calibration quality over time as league dynamics change

---

### 📊 Priority 4: Comprehensive Evaluation Framework

#### 4.1 Sharpness Metric
**Why:** A perfectly calibrated model could always predict 50% (perfect calibration, useless predictions). Sharpness measures concentration of probability mass.

**Implementation:**
```python
# Add to calibration.py
def calculate_sharpness(y_prob):
    """
    Sharpness = average distance from 0.5
    Higher = more confident/sharp predictions
    Range: [0, 0.5]
    """
    return np.mean(np.abs(y_prob - 0.5))

def calculate_proper_scoring_rules(y_true, y_prob):
    """
    Calculate multiple proper scoring rules:
    - Brier Score (MSE of probabilities)
    - Log Loss (cross-entropy)
    - Spherical Score (geometric mean)
    """
    brier = calculate_brier_score(y_true, y_prob)
    log_loss = calculate_log_loss(y_true, y_prob)

    # Spherical score: mean of y*p/||p|| + (1-y)*(1-p)/||p||
    norm = np.sqrt(y_prob**2 + (1 - y_prob)**2)
    spherical = np.mean(
        (y_true * y_prob + (1 - y_true) * (1 - y_prob)) / norm
    )

    return {
        'brier': brier,
        'log_loss': log_loss,
        'spherical': spherical
    }
```

---

#### 4.2 Reliability Diagrams with Confidence Intervals
**Why:** Visualize calibration quality with statistical significance

**Implementation:**
```python
# Add to calibration.py
def calculate_calibration_with_ci(y_true, y_prob, n_bins=10, confidence=0.95):
    """
    Calibration curve with confidence intervals using bootstrap
    """
    from scipy import stats

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    results = []

    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) < 5:
            continue

        bin_true = y_true[mask]
        bin_prob = y_prob[mask]

        # Bootstrap confidence interval
        n_bootstrap = 1000
        bootstrap_means = []

        for _ in range(n_bootstrap):
            sample_idx = np.random.choice(len(bin_true), len(bin_true), replace=True)
            bootstrap_means.append(np.mean(bin_true[sample_idx]))

        ci_low, ci_high = np.percentile(bootstrap_means,
                                        [(1-confidence)/2 * 100,
                                         (1+confidence)/2 * 100])

        results.append({
            'bin_start': bin_edges[i],
            'bin_end': bin_edges[i+1],
            'predicted': np.mean(bin_prob),
            'actual': np.mean(bin_true),
            'count': len(bin_true),
            'ci_low': ci_low,
            'ci_high': ci_high
        })

    return results
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1)
- [ ] Add Beta Calibration to `calibration.py`
- [ ] Add Advanced Temperature Scaling
- [ ] Add Ensemble Calibrator
- [ ] Update `CalibrationAnalyzer` to support new methods

### Phase 2: Model Integration (Week 2)
- [ ] Integrate Focal Loss into `models_dc_xg.py`
- [ ] Add calibration-aware training objective
- [ ] Implement Adaptive Bin Calibrator
- [ ] Update `backtest_calibration.py` to test new methods

### Phase 3: Production (Week 3)
- [ ] Add Online Calibration to `predict_api.py`
- [ ] Implement comprehensive evaluation metrics
- [ ] Add confidence intervals to calibration reports
- [ ] Create automated calibration monitoring

### Phase 4: Validation (Week 4)
- [ ] Run full calibration backtest with new methods
- [ ] Compare ECE, Brier, Sharpness across methods
- [ ] Generate production configuration
- [ ] Document optimal settings per league/market

---

## Expected Overall Improvements

| Metric | Current (Est.) | Target | Improvement |
|--------|---------------|--------|-------------|
| ECE (BTTS) | 0.035 | 0.015 | -57% |
| ECE (O/U 2.5) | 0.042 | 0.020 | -52% |
| Brier Score | 0.245 | 0.230 | -6% |
| Sharpness | 0.15 | 0.18 | +20% |
| ROI (top 10%) | 8% | 12% | +50% |

---

## Testing Protocol

For each new calibration method:

1. **Train/Test Split**: 70/30 temporal split
2. **Cross-Validation**: 5-fold walk-forward on test set
3. **Metrics**: ECE, MCE, Brier, Log Loss, Sharpness
4. **Statistical Significance**: Bootstrap test (n=1000) for ECE difference
5. **Production Criteria**: ECE < 0.020 AND Brier improvement > 0.005

---

## Integration with football-api

### Modified Prediction Flow:
```
1. Load data from API-Football
2. Generate features (xG, injuries, formations, H2H)
3. Fit DC model with focal loss
4. Generate raw probabilities
5. Apply ENSEMBLE calibration (Isotonic + Beta + Platt)
6. Apply ADAPTIVE BIN calibration for extremes
7. Calculate confidence intervals
8. Export to prediction API
```

### Configuration File:
```python
# calibration_config_production.py
CALIBRATION_CONFIG = {
    'BTTS': {
        'method': 'ensemble',  # ensemble, isotonic, beta, platt
        'calibrators': ['isotonic', 'beta', 'platt'],
        'weights': [0.4, 0.35, 0.25],  # Learned from validation
        'adaptive_bins': True,
        'bin_edges': [0.0, 0.35, 0.65, 1.0]
    },
    'OU_2_5': {
        'method': 'ensemble',
        'calibrators': ['isotonic', 'beta'],
        'weights': [0.6, 0.4],
        'adaptive_bins': True,
        'bin_edges': [0.0, 0.4, 0.7, 1.0]
    }
}
```

---

## Monitoring & Maintenance

### Weekly Tasks:
- [ ] Calculate ECE on last week's predictions vs actuals
- [ ] Update online calibrators with new data
- [ ] Check for calibration drift (ECE > threshold)
- [ ] Regenerate calibration curves

### Monthly Tasks:
- [ ] Full recalibration on last 6 months data
- [ ] Parameter optimization (rho, decay, xg_weight)
- [ ] Feature importance analysis
- [ ] Update production config

### Alerts:
- ECE > 0.035 for any market
- Brier score increase > 0.010
- Sharpness decrease > 10%
- Accuracy drop > 3%

---

## References

- Kull, M., et al. (2017). "Beyond temperature scaling: Obtaining well-calibrated predictions from neural networks"
- Bella, A., et al. (2010). "Calibration of machine learning models"
- Guo, C., et al. (2017). "On Calibration of Modern Neural Networks"
- Dixon & Coles (1997). "Modelling Association Football Scores and Inefficiencies in the Football Betting Market"
