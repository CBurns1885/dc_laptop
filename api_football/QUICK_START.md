# Quick Start Guide
## Get Up and Running in 5 Minutes

---

## Prerequisites

✅ API Key: `0f17fdba78d15a625710f7244a1cc770`
✅ Python 3.8+
✅ ~2GB disk space for data

---

## Step 1: Install Dependencies (30 seconds)

```bash
cd "d:\Users\Ian\Desktop\Chris Code\dc_laptop"
pip install -r requirements.txt
```

If you get errors, install individually:
```bash
pip install pandas numpy scipy requests openpyxl scikit-learn
```

---

## Step 2: Download Historical Data (10-30 minutes)

### Option A: Quick Test (Premier League only, ~2 minutes)
```bash
cd api_football

python run_api_football.py ingest \
    --api-key 0f17fdba78d15a625710f7244a1cc770 \
    --leagues E0 \
    --seasons 2023 2024
```

### Option B: Full Dataset (Recommended, ~30 minutes)
```bash
python run_api_football.py ingest \
    --api-key 0f17fdba78d15a625710f7244a1cc770 \
    --leagues E0 E1 D1 D2 SP1 I1 F1 \
    --seasons 2023 2024
```

**What this does**:
- Downloads all fixtures for selected leagues
- Fetches detailed statistics (including **xG**)
- Gets match events (goals, cards, subs)
- Retrieves current injuries
- Stores everything in `data/football_api.db`

**Expected output**:
```
[INFO] Downloading Premier League 2023-24...
[INFO] Found 380 fixtures
[INFO] Fetching statistics... (380/380)
[INFO] Fetching events... (380/380)
[INFO] Downloaded 380 fixtures with full stats
```

---

## Step 3: Build Features (2-5 minutes)

```bash
python run_api_football.py features
```

**What this does**:
- Calculates xG moving averages
- Computes team form (L5, L10)
- Extracts H2H history
- Calculates rest days
- Adds injury features
- Outputs `data/processed/features.parquet`

**Expected output**:
```
[INFO] Loading fixtures from database...
[INFO] Building xG features...
[INFO] Building form features...
[INFO] Building H2H features...
[INFO] Building injury features...
[INFO] Saved 380 matches with 87 features
```

---

## Step 4: Run Calibration Backtest (5-15 minutes)

```bash
python run_api_football.py calibrate \
    --optimize \
    --features data/processed/features.parquet
```

**What this does**:
- Fits Dixon-Coles model with xG
- Tests all calibration methods (Isotonic, Platt, Beta, Ensemble)
- Optimizes parameters per league
- Calculates optimal thresholds
- Generates production config

**Expected output**:
```
CALIBRATION BACKTEST REPORT
================================================================

BTTS:
  Accuracy:         54.2%
  Brier Score:      0.2387
  ECE:              0.0234    <- Target: <0.030 ✅
  Optimal Threshold: 0.53

Recommendations:
  ✅ BTTS: Apply isotonic calibration (Brier +0.0047)
  🎯 BTTS: Use threshold 0.53 instead of 0.50

Results saved to: calibration_results/calibration_results_20251229_1200.json
```

---

## Step 5: Generate Predictions (1 minute)

```bash
python run_api_football.py predict \
    --api-key 0f17fdba78d15a625710f7244a1cc770 \
    --days 7
```

**What this does**:
- Fetches upcoming fixtures from API
- Applies xG-enhanced DC model
- Applies calibration
- Uses optimal thresholds
- Exports predictions

**Expected output**:
```
TOP BTTS YES PREDICTIONS
====================================================================
  Arsenal vs Chelsea: 67.3% [E0]
  Liverpool vs Man City: 64.8% [E0]
  Bayern vs Dortmund: 62.1% [D1]

TOP OVER 2.5 PREDICTIONS
====================================================================
  Real Madrid vs Barcelona: 71.2% [SP1]
  PSG vs Marseille: 68.5% [F1]

Saved 47 predictions to: outputs/predictions_20251229_1205.csv
```

---

## Full Pipeline (One Command)

```bash
python run_api_football.py full \
    --api-key 0f17fdba78d15a625710f7244a1cc770 \
    --leagues E0 D1 SP1 \
    --seasons 2023 2024
```

**This runs everything**:
1. ✅ Data ingestion
2. ✅ Feature engineering
3. ✅ Calibration backtest
4. ✅ Prediction generation

**Total time**: ~45 minutes for 3 leagues

---

## Daily Update Workflow

After initial setup, run this daily:

```bash
# Update recent results + generate new predictions (1-2 minutes)
python run_api_football.py full \
    --api-key 0f17fdba78d15a625710f7244a1cc770 \
    --update-only \
    --skip-backtest
```

**What this does**:
- Updates last 7 days of results
- Rebuilds features
- Generates predictions for next 7 days
- Uses ~50-100 API calls

---

## Viewing Results

### Method 1: CSV File
```bash
# Open in Excel
outputs/predictions_YYYYMMDD_HHMM.csv
```

Columns:
- `home`, `away`, `league`, `date`
- `btts_yes`, `over_2_5`, `under_2_5`
- `expected_home_goals`, `expected_away_goals`
- `btts_confidence`, `ou_confidence`

### Method 2: JSON File
```bash
# For programmatic access
outputs/predictions_YYYYMMDD_HHMM.json
```

### Method 3: Excel File
```bash
# Auto-formatted with color coding
outputs/predictions_YYYYMMDD_HHMM.xlsx
```

---

## Understanding the Output

### Calibrated Probabilities
```
BTTS Yes: 67.3%
```
**Meaning**: When the model says 67.3%, it happens ~67% of the time (calibrated!)

### Confidence Scores
```
BTTS Confidence: 0.89
```
**Meaning**: High confidence = tight calibration, more reliable prediction
- **>0.85**: Very reliable
- **0.70-0.85**: Reliable
- **<0.70**: Less reliable

### Optimal Thresholds
```
Optimal Threshold: 0.53
```
**Meaning**: For best accuracy, treat >53% as "Yes" instead of >50%

---

## Monitoring Performance

### Check Calibration Quality

```bash
python -c "
import pandas as pd
df = pd.read_csv('outputs/predictions_latest.csv')
actual = pd.read_csv('data/results_latest.csv')

# Merge predictions with actuals
merged = df.merge(actual, on='fixture_id')

# Calculate Brier score
from calibration import calculate_brier_score
import numpy as np

y_true = (merged['btts_actual'] == 'Y').astype(int)
y_prob = merged['btts_yes']

brier = calculate_brier_score(y_true.values, y_prob.values)
print(f'Brier Score: {brier:.4f}')  # Target: <0.235
"
```

### Track API Usage

```bash
python run_api_football.py status
```

Shows:
- Requests used today
- Requests remaining
- Database size
- Feature file info

---

## Troubleshooting

### "ModuleNotFoundError: No module named X"
```bash
pip install X
```

### "API Error: 401 Unauthorized"
Check API key is correct:
```bash
python -c "print('0f17fdba78d15a625710f7244a1cc770')"
```

### "No fixtures found"
May be off-season. Try different dates:
```bash
python run_api_football.py predict --date 2025-01-15
```

### "Low xG coverage"
Normal for cup competitions. xG should be >95% for leagues.

### "High ECE (>0.05)"
Run calibration backtest:
```bash
python run_api_football.py calibrate --optimize
```

---

## Advanced Usage

### Custom League Configuration

Edit `api_football_client.py`:
```python
LEAGUES = {
    'MY_LEAGUE': {
        'id': 123,  # API-Football league ID
        'name': 'My League',
        'country': 'Country',
        'type': 'league'
    }
}
```

### Custom Features

Edit `features_api.py` to add custom features:
```python
def add_custom_features(df):
    # Add your features here
    df['is_derby'] = ...
    df['weather_impact'] = ...
    return df
```

### Custom Calibration

Use the enhanced calibration system:
```python
from calibration_enhanced import EnhancedCalibrationSystem

system = EnhancedCalibrationSystem()
system.fit(y_true, y_prob)
calibrated = system.transform(y_prob_new)
```

---

## Best Practices

### 1. Recalibrate Monthly
```bash
# At start of each month
python run_api_football.py calibrate --optimize
```

### 2. Track Performance
Keep a log of predictions vs actuals:
```bash
predictions_YYYYMMDD.csv -> archive/
results_YYYYMMDD.csv -> archive/
```

### 3. Monitor ECE
Target: ECE < 0.030 for all markets
```bash
# Weekly check
python -c "
from calibration import calculate_ece_mce
import pandas as pd

# Load last week's data
df = pd.read_csv('archive/last_week_results.csv')
ece, mce = calculate_ece_mce(df['actual'], df['predicted'])
print(f'ECE: {ece:.4f}')  # Should be <0.030
"
```

### 4. Use Optimal Thresholds
Don't use 50% as cutoff. Use league-specific thresholds:
```python
# Load from calibration config
from calibration_config_20251229_1200 import OPTIMAL_THRESHOLDS

# For Premier League BTTS
threshold = OPTIMAL_THRESHOLDS['E0']['BTTS']  # e.g., 0.53
recommended = predictions[predictions['btts_yes'] >= threshold]
```

---

## Performance Targets

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| Brier Score | <0.245 | <0.235 | <0.228 |
| ECE | <0.040 | <0.030 | <0.020 |
| Accuracy | >52% | >54% | >56% |
| ROI (top 10%) | >5% | >10% | >15% |

---

## Support

### Documentation
- [CALIBRATION_ENHANCEMENT_PLAN.md](CALIBRATION_ENHANCEMENT_PLAN.md) - Full calibration strategy
- [API_INTEGRATION_SUMMARY.md](API_INTEGRATION_SUMMARY.md) - API capabilities
- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - Detailed integration guide

### Test Scripts
```bash
# Test API access
python test_api_comprehensive.py

# Test calibration
python calibration_enhanced.py

# Test prediction pipeline
python test_end_to_end.py
```

---

## Quick Reference

```bash
# Daily workflow
python run_api_football.py full --api-key KEY --update-only --skip-backtest

# Full rebuild
python run_api_football.py full --api-key KEY --leagues E0 D1 SP1

# Just predictions
python run_api_football.py predict --api-key KEY --days 7

# Calibration check
python run_api_football.py calibrate --features data/processed/features.parquet

# System status
python run_api_football.py status
```

---

## What to Expect

### First Run (Full Dataset)
- **Time**: 45-60 minutes
- **API Calls**: ~3,000
- **Disk Space**: ~500MB
- **Output**: Database + Features + Calibrated Model

### Daily Updates
- **Time**: 1-2 minutes
- **API Calls**: ~50-100
- **Disk Space**: +10MB/day
- **Output**: New predictions

### Calibration Quality
- **Without calibration**: ECE ~0.045, Brier ~0.245
- **With calibration**: ECE ~0.022, Brier ~0.230
- **Improvement**: ~50% better calibration, 7% better Brier

---

## Next Steps After Setup

1. **Run backtest** to validate performance
2. **Track predictions** for 2 weeks to verify calibration
3. **Tune parameters** per league if needed
4. **Deploy automated daily updates**
5. **Build dashboard** for visualization (optional)

---

🎉 **You're ready to go!** Start with Step 1 and you'll have predictions in ~45 minutes.
