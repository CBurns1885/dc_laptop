# Adaptive Dixon-Coles Backtesting

## Overview

This module tests whether **recalibrating Dixon-Coles model parameters** improves prediction accuracy.

### Two Approaches Compared:

1. **STATIC DC** (Current Production Method)
   - Fit DC parameters **once** using all historical data
   - Never update parameters during backtest
   - Simulates current `dc_predict.py` behavior

2. **ADAPTIVE DC** (Walk-Forward Recalibration)
   - **Re-fit** DC parameters each test period
   - Uses only data available at that point in time
   - Tests if updating attack/defence weights improves accuracy

## What Gets Recalibrated?

When adaptive mode re-fits, it updates:
- ✅ **Team attack strengths** - How many goals teams score
- ✅ **Team defence strengths** - How many goals teams concede
- ✅ **Home advantage** - League-specific home edge
- ✅ **Rho parameter** - Low-score correlation
- ✅ **Recent form multipliers** - Last 5 games boost

## Files

- **`backtest_adaptive_dc.py`** - Main backtest engine
- **`backtest_adaptive_visualizer.py`** - Creates comparison charts
- **`README_ADAPTIVE_DC.md`** - This file

## Usage

### Quick Start

```bash
python backtest_adaptive_dc.py
```

Interactive prompts will guide you through:
1. Date range (default: last 6 months)
2. Refit frequency (weekly/biweekly/monthly)

### Example Run

```
ADAPTIVE DC BACKTEST
======================================================================

This compares:
  1. STATIC: Fit DC once on all data (current production)
  2. ADAPTIVE: Re-fit DC parameters each test period

Default period: Last 6 months
   From: 2024-05-25
   To: 2024-11-25

Use default? (y/n, default=y): y

Refit frequency options:
  1. Weekly (refit every week)
  2. Biweekly (refit every 2 weeks)
  3. Monthly (refit every 4 weeks)

Select (1-3, default=1): 1

[Runs backtest comparing both methods...]
```

## Output Files

After running, check `outputs/` folder:

### 1. Summary CSVs
- `backtest_dc_static_summary.csv` - Static DC performance
- `backtest_dc_adaptive_summary.csv` - Adaptive DC performance
- `backtest_dc_comparison.csv` - Head-to-head comparison

### 2. Comparison CSV Format

```csv
Market,Static_Acc_%,Adaptive_Acc_%,Acc_Diff_%,Static_Brier,Adaptive_Brier,Brier_Diff,Winner
OU_2_5,58.23,59.87,+1.64,0.2341,0.2298,-0.0043,✓ ADAPTIVE
BTTS,56.12,55.98,-0.14,0.2456,0.2461,+0.0005,≈ TIE
OU_1_5,61.34,60.89,-0.45,0.2198,0.2219,+0.0021,✓ STATIC
...
```

## Interpreting Results

### Accuracy Difference
- **+0.5% or more** → Adaptive wins (meaningful improvement)
- **-0.5% or less** → Static wins (recalibration hurts)
- **Between ±0.5%** → Tie (no significant difference)

### Brier Score
- **Lower is better** (calibration quality)
- Negative difference = Adaptive better calibrated

### Verdict Section

The engine prints recommendations:

```
VERDICT
======================================================================
Adaptive wins: 5
Static wins: 2
Ties: 1

Average accuracy improvement: +1.23%
Average Brier improvement: +0.0034

✓ RECOMMENDATION: Use ADAPTIVE DC fitting
   Refit frequency: weekly
```

## Refit Frequency Options

### Weekly (Default)
- Re-fit every test period (7 days)
- Most responsive to recent form
- Higher computation cost

### Biweekly
- Re-fit every 2 weeks
- Balanced approach
- Moderate computation

### Monthly
- Re-fit every 4 weeks
- Most stable parameters
- Lowest computation cost

**Recommendation**: Start with weekly, then test others if performance is similar.

## Visualizations

Generate charts from results:

```bash
python backtest_adaptive_visualizer.py
```

Creates:
- `backtest_adaptive_improvement_heatmap.png` - Which markets improved?

## Implementation Guide

If adaptive DC performs better, update production code:

### Current Production (`dc_predict.py`)
```python
# Fits on ALL historical data
params = fit_all(base)
```

### Switch to Adaptive (Rolling Window)
```python
# Use only recent data (e.g., last 365 days)
cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=365)
recent_data = base[base['Date'] >= cutoff_date]
params = fit_all(recent_data)
```

### Automated Weekly Refitting
Add to your weekly prediction pipeline:
```python
# In your weekly workflow
def generate_predictions():
    # 1. Download latest results
    download_latest_results()

    # 2. Re-fit DC parameters (adaptive)
    params = refit_dc_parameters()

    # 3. Generate predictions with updated params
    predictions = predict_with_dc(params)
```

## Technical Details

### Walk-Forward Validation
- Each test period uses **only** data from before that period
- No data leakage (matches in test set not used for training)
- Realistic simulation of production usage

### Time Weighting
DC models already use exponential time decay:
```python
# In models_dc.py
LEAGUE_DECAY_DAYS = {
    'E0': 365,  # Premier League
    'E1': 320,  # Championship
    ...
}
```

Adaptive refitting **combines** with time weighting:
- Old matches have lower weight (via decay)
- Very old matches excluded entirely (via rolling window)

### Form Multipliers
Both methods use recent form (last 5 games):
```python
# Applied during prediction
lam *= form_mult_home_att * (1 / form_mult_away_def)
```

Adaptive refitting updates the **base attack/defence** more frequently.

## Troubleshooting

### "Insufficient data for static fit"
- Need at least 100 matches before backtest start
- Solution: Choose later start date or build more historical data

### "No test matches - skipping"
- Some weeks have no matches (international breaks, off-season)
- This is normal, engine skips these periods

### Slow performance
- Fitting DC parameters is computationally expensive
- Try biweekly or monthly refit frequency
- Consider shorter backtest period for testing

## Next Steps

1. **Run the backtest** on your data (6+ months recommended)
2. **Check the comparison CSV** - which method wins?
3. **If adaptive is better**:
   - Update `dc_predict.py` to use rolling window
   - Schedule weekly DC refitting in production
4. **If static is better**:
   - Keep current approach (no changes needed)
   - DC model already well-optimized

## Questions?

- **Why might adaptive be worse?** Over-fitting to recent noise, teams' true strength hasn't changed
- **Why might adaptive be better?** Teams change (injuries, form, tactics), recent data more relevant
- **What if results are similar?** Keep static (simpler is better when performance equal)

## Related Files

- `models_dc.py` - Core Dixon-Coles implementation
- `dc_predict.py` - Production DC prediction pipeline
- `backtest.py` - General backtest engine (uses full ML pipeline)
- `backtest_config.py` - Backtest configuration options
