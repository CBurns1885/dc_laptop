# Football-API Integration Guide with Enhanced Calibration

## Quick Start

### 1. Setup API Key
```bash
# Windows
set API_FOOTBALL_KEY=your_api_key_here

# Linux/Mac
export API_FOOTBALL_KEY=your_api_key_here
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Check System Status
```bash
cd api_football
python run_api_football.py status
```

---

## Complete Workflow

### Option A: Full Pipeline (Recommended for First Run)
```bash
# Download data, build features, run backtest, generate predictions
python run_api_football.py full --api-key YOUR_KEY
```

### Option B: Step-by-Step

#### Step 1: Download Data from API-Football
```bash
# Download historical data for all leagues
python run_api_football.py ingest --api-key YOUR_KEY

# Or download specific leagues
python run_api_football.py ingest --api-key YOUR_KEY --leagues E0 D1 SP1

# Or just update recent matches (faster)
python run_api_football.py ingest --api-key YOUR_KEY --update-only --days 7
```

#### Step 2: Build Features
```bash
# Build complete feature set (xG, injuries, formations, H2H, etc.)
python run_api_football.py features
```

#### Step 3: Run Calibration Backtest (CRITICAL FOR HIGHEST ACCURACY!)
```bash
# Full calibration analysis with all enhancements
python run_api_football.py calibrate --optimize --features data/processed/features.parquet

# This will:
# - Test all calibration methods (Isotonic, Platt, Beta, Ensemble, Adaptive)
# - Optimize DC parameters (rho, decay, xg_weight) per league
# - Calculate feature importance
# - Generate optimal thresholds per market
# - Export production-ready config
```

Output files:
- `calibration_results_{timestamp}.json` - Full metrics
- `calibration_config_{timestamp}.py` - Python config for production
- `calibration_predictions_{timestamp}.csv` - Backtest predictions

#### Step 4: Run Standard Backtest (Optional)
```bash
# Standard backtest with feature ablation
python run_api_football.py backtest --ablation

# League-specific backtest
python run_api_football.py backtest --leagues E0 D1 SP1
```

#### Step 5: Generate Predictions
```bash
# Predict upcoming fixtures (next 7 days)
python run_api_football.py predict --api-key YOUR_KEY --days 7

# Predict specific match
python run_api_football.py predict --home Arsenal --away Chelsea --league E0 --date 2025-01-05

# Predict all matches on a specific date
python run_api_football.py predict --date 2025-01-05
```

---

## Using Enhanced Calibration in Production

### Method 1: Apply Calibrated Config (Recommended)

After running calibration backtest, you'll have a config file. Use it:

```python
# predict_with_calibration.py
from predict_api import MatchPredictor
from calibration_enhanced import EnhancedCalibrationSystem
import pandas as pd
import numpy as np

# Load calibration config
from calibration_config_20250101_1200 import (
    OPTIMAL_THRESHOLDS,
    CALIBRATION_METHODS
)

# Initialize predictor
predictor = MatchPredictor(api_key="YOUR_KEY")

# Initialize calibration systems per market
calibrators = {}
for market in ['BTTS', 'OU_2_5', 'OU_1_5', 'OU_3_5']:
    calibrators[market] = EnhancedCalibrationSystem()
    # Load pre-fitted calibrator from disk (you'll save this during backtest)

# Predict upcoming fixtures
df = predictor.predict_upcoming(days=7)

# Apply calibration
for market in calibrators.keys():
    prob_col = f'{market.lower()}_yes' if market == 'BTTS' else f'over_{market.split("_")[1]}'

    if prob_col in df.columns:
        df[f'{prob_col}_calibrated'] = calibrators[market].transform(
            df[prob_col].values
        )

# Apply optimal thresholds
for league in df['league'].unique():
    if league in OPTIMAL_THRESHOLDS:
        for market, threshold in OPTIMAL_THRESHOLDS[league].items():
            # Mark as "recommended" if above optimal threshold
            mask = df['league'] == league
            prob_col = f'{market.lower()}_yes' if market == 'BTTS' else f'over_{market.split("_")[1]}'

            if f'{prob_col}_calibrated' in df.columns:
                df.loc[mask, f'{market}_recommended'] = (
                    df.loc[mask, f'{prob_col}_calibrated'] >= threshold
                )

# Export predictions
df.to_csv('predictions_calibrated.csv', index=False)
df.to_excel('predictions_calibrated.xlsx', index=False)
```

### Method 2: Real-Time Calibration

For live predictions with calibration:

```python
# live_prediction_system.py
from api_football_client import APIFootballClient, LEAGUES
from predict_api import MatchPredictor
from calibration_enhanced import EnhancedCalibrationSystem
import pandas as pd
from datetime import datetime, timedelta

class LivePredictionSystem:
    """Real-time prediction system with calibration"""

    def __init__(self, api_key: str):
        self.client = APIFootballClient(api_key)
        self.predictor = MatchPredictor(api_key)

        # Load pre-fitted calibrators
        self.calibrators = self._load_calibrators()

    def _load_calibrators(self):
        """Load pre-fitted calibration systems"""
        # Load from backtest results
        import pickle

        calibrators = {}
        for market in ['BTTS', 'OU_2_5']:
            with open(f'calibrators/{market}_calibrator.pkl', 'rb') as f:
                calibrators[market] = pickle.load(f)

        return calibrators

    def predict_today(self):
        """Predict all fixtures for today with calibration"""
        today = datetime.now().strftime('%Y-%m-%d')

        # Get predictions
        predictions = self.predictor.predict_date(today)

        if not predictions:
            print(f"No fixtures found for {today}")
            return None

        # Apply calibration
        df = pd.DataFrame(predictions)

        for market, calibrator in self.calibrators.items():
            prob_col = f'{market.lower()}_yes' if market == 'BTTS' else 'over_2_5'

            if prob_col in df.columns:
                df[f'{prob_col}_raw'] = df[prob_col]
                df[prob_col] = calibrator.transform(df[prob_col].values)

        return df

    def get_top_picks(self, min_prob: float = 0.65, n: int = 10):
        """Get top calibrated picks for today"""
        df = self.predict_today()

        if df is None or df.empty:
            return None

        # Filter by minimum probability (calibrated)
        btts_picks = df[df['btts_yes'] >= min_prob].nlargest(n, 'btts_yes')
        over_picks = df[df['over_2_5'] >= min_prob].nlargest(n, 'over_2_5')

        return {
            'btts': btts_picks[['home', 'away', 'league', 'btts_yes', 'expected_total']],
            'over_2_5': over_picks[['home', 'away', 'league', 'over_2_5', 'expected_total']]
        }

# Usage
if __name__ == "__main__":
    system = LivePredictionSystem(api_key="YOUR_KEY")

    picks = system.get_top_picks(min_prob=0.65, n=10)

    if picks:
        print("="*70)
        print("TOP PICKS FOR TODAY (CALIBRATED)")
        print("="*70)

        print("\nBTTS Yes:")
        print(picks['btts'].to_string(index=False))

        print("\nOver 2.5:")
        print(picks['over_2_5'].to_string(index=False))
```

---

## Calibration Workflow Details

### Understanding Calibration Metrics

After running calibration backtest, you'll see metrics like:

```
CALIBRATION METRICS:

BTTS:
  Accuracy:         54.2%
  Brier Score:      0.2387    <- Lower is better (MSE of probabilities)
  Log Loss:         0.6821    <- Lower is better (cross-entropy)
  ECE:              0.0234    <- Expected Calibration Error (lower = better calibrated)
  MCE:              0.0812    <- Maximum Calibration Error

  Brier Decomposition:
    Reliability:    0.0087    <- Calibration quality (lower = better)
    Resolution:     0.0234    <- Discrimination (higher = better)
    Uncertainty:    0.2500    <- Base rate variance (fixed)

  Optimal Threshold: 0.53 (accuracy: 56.1%)
```

**Key Metrics:**
- **ECE < 0.030**: Excellent calibration
- **ECE < 0.050**: Good calibration
- **ECE > 0.080**: Poor calibration (needs fixing)

**What This Means:**
- If ECE = 0.0234: When your model says 60%, it happens ~59.6% of the time (excellent!)
- If Brier = 0.2387: Your probability errors average to 0.2387² per prediction
- If Optimal Threshold = 0.53: Use 53% as cutoff instead of 50% for best accuracy

### Calibration Curve Interpretation

```
Calibration Curve:

Bin          Predicted     Actual      Gap      Count
0.0-0.1         0.052      0.045     -0.007       82
0.1-0.2         0.154      0.167     +0.013      124
0.2-0.3         0.248      0.255     +0.007      156
0.3-0.4         0.346      0.342     -0.004      189
0.4-0.5         0.451      0.457     +0.006      213
0.5-0.6         0.547      0.541     -0.006      198
0.6-0.7         0.648      0.654     +0.006      167
0.7-0.8         0.743      0.721     -0.022      134
0.8-0.9         0.846      0.802     -0.044       89   <- OVERCONFIDENT!
0.9-1.0         0.932      0.878     -0.054       42   <- VERY OVERCONFIDENT!
```

**Interpretation:**
- Negative gap: Model is OVERCONFIDENT (predicts higher than actual)
- Positive gap: Model is UNDERCONFIDENT (predicts lower than actual)
- Gap close to 0: Well calibrated

**In this example:**
- Model is well-calibrated for probabilities 0.1 to 0.7 (gaps < ±0.01)
- Model is OVERCONFIDENT for high probabilities (0.8-1.0)
  - When it says 93%, it actually happens 87.8% of the time
- **Solution:** Apply calibration (Isotonic, Beta, or Ensemble) to fix this

---

## Advanced Usage

### Custom League Configuration

Edit `api_football_client.py` to add new leagues:

```python
LEAGUES = {
    # Add your custom league
    'CUSTOM_LEAGUE': {
        'id': 123,  # API-Football league ID
        'name': 'Custom League Name',
        'country': 'Country',
        'type': 'league'  # or 'cup'
    }
}
```

Then add league-specific DC parameters in `models_dc_xg.py`:

```python
LEAGUE_CONFIG = {
    'CUSTOM_LEAGUE': {
        'rho_init': 0.05,
        'rho_bounds': (-0.05, 0.15),
        'decay_days': 365,
        'xg_weight': 0.6
    }
}
```

### Feature Engineering Customization

Modify `features_api.py` to add custom features:

```python
def add_custom_features(df):
    """Add your custom features"""

    # Example: Weather impact
    df['is_winter'] = df['Date'].dt.month.isin([12, 1, 2])

    # Example: Derby matches
    df['is_derby'] = (
        (df['HomeTeam'].str.contains('Arsenal') & df['AwayTeam'].str.contains('Tottenham')) |
        (df['HomeTeam'].str.contains('Liverpool') & df['AwayTeam'].str.contains('Everton'))
        # ... etc
    )

    return df
```

### Custom Calibration Strategy

Create your own calibration logic:

```python
from calibration_enhanced import EnhancedCalibrationSystem
import numpy as np

class CustomCalibrationStrategy:
    """Custom calibration with domain-specific rules"""

    def __init__(self):
        self.base_calibrator = EnhancedCalibrationSystem()
        self.adjustments = {}

    def fit(self, df: pd.DataFrame):
        """Fit with DataFrame containing predictions and features"""

        # Fit base calibrator
        y_true = (df['actual_btts'] == 'Y').astype(int)
        y_prob = df['prob_btts']
        self.base_calibrator.fit(y_true.values, y_prob.values)

        # Learn adjustments for specific scenarios
        # E.g., Top 6 teams tend to have higher BTTS
        top6_teams = ['Arsenal', 'Chelsea', 'Liverpool', 'Man City', 'Man United', 'Tottenham']

        top6_mask = df['HomeTeam'].isin(top6_teams) | df['AwayTeam'].isin(top6_teams)

        if top6_mask.sum() > 50:
            top6_base_rate = df.loc[top6_mask, 'actual_btts'].apply(lambda x: 1 if x == 'Y' else 0).mean()
            overall_base_rate = y_true.mean()

            # Adjustment factor
            self.adjustments['top6'] = top6_base_rate / overall_base_rate

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply calibration with adjustments"""

        # Base calibration
        probs = self.base_calibrator.transform(df['prob_btts'].values)

        # Apply adjustments
        if 'top6' in self.adjustments:
            top6_teams = ['Arsenal', 'Chelsea', 'Liverpool', 'Man City', 'Man United', 'Tottenham']
            top6_mask = df['HomeTeam'].isin(top6_teams) | df['AwayTeam'].isin(top6_teams)

            probs[top6_mask] *= self.adjustments['top6']

        return np.clip(probs, 0.0, 1.0)
```

---

## Troubleshooting

### Issue: "API key not found"
```bash
# Make sure API key is set
echo %API_FOOTBALL_KEY%  # Windows
echo $API_FOOTBALL_KEY   # Linux/Mac

# Or pass directly
python run_api_football.py predict --api-key YOUR_KEY
```

### Issue: "No features file found"
```bash
# Build features first
python run_api_football.py features

# Or download and build in one go
python run_api_football.py full --api-key YOUR_KEY
```

### Issue: "High ECE (> 0.08)"
This means poor calibration. Solutions:

1. Run calibration backtest to fit calibrators:
```bash
python run_api_football.py calibrate --optimize
```

2. Use ensemble calibration:
```python
from calibration_enhanced import EnhancedCalibrationSystem

system = EnhancedCalibrationSystem()
system.fit(y_true, y_prob, force_method='ensemble')
```

3. Check for data leakage in features

### Issue: "ImportError: No module named X"
```bash
# Install missing dependencies
pip install -r requirements.txt

# Or specific package
pip install scipy pandas numpy scikit-learn
```

---

## Next Steps

1. **Run Full Calibration Backtest**:
   ```bash
   python run_api_football.py calibrate --optimize --features data/processed/features.parquet
   ```

2. **Analyze Results**:
   - Check ECE for each market (target: < 0.030)
   - Review calibration curves for over/underconfidence
   - Note optimal thresholds per league

3. **Apply in Production**:
   - Use generated config file
   - Apply calibrated probabilities
   - Use optimal thresholds for recommendations

4. **Monitor Performance**:
   - Track weekly ECE on actual results
   - Recalibrate monthly with new data
   - Adjust parameters based on performance

---

## Questions?

Check the documentation:
- [CALIBRATION_ENHANCEMENT_PLAN.md](CALIBRATION_ENHANCEMENT_PLAN.md) - Detailed calibration strategy
- [README_DC_ONLY.md](../README_DC_ONLY.md) - Original DC model docs
- [README_ADAPTIVE_DC.md](../README_ADAPTIVE_DC.md) - Adaptive features

Or examine the code:
- [calibration_enhanced.py](calibration_enhanced.py) - Enhanced calibration methods
- [backtest_calibration.py](backtest_calibration.py) - Calibration backtest system
- [models_dc_xg.py](models_dc_xg.py) - xG-enhanced Dixon-Coles model
