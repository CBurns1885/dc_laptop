# Testing Guide for All Updates

## Quick Start

### Option 1: Run Everything (Automated)

```bash
# 1. Run backtest (4 weeks)
python backtest_config.py

# 2. Run test suite
python test_all_updates.py
```

Or on Windows:
```bash
RUN_TESTS.bat
```

### Option 2: Manual Testing

Run these in order:

```bash
# Test 1: Backtest with all fixes
python backtest_config.py

# Test 2: Weekly pipeline with enrichment
python run_weekly.py

# Test 3: Verify all updates
python test_all_updates.py
```

## What Gets Tested

### 1. Backtest Updates ✅

**Tests**:
- Conservative ROI calculation (with bookmaker margins)
- 1X2 market included
- League-specific breakdowns
- Proper column names (Brier_Score not Avg_Brier_Score)

**Expected Files**:
```
outputs/
├── backtest_summary.csv           ← ROI with margins
├── backtest_detailed.csv          ← Period breakdowns
├── backtest_league_breakdown.csv  ← Per-league performance
```

**Good Results**:
- ROI is **negative** (e.g., -2% to -8%) = conservative ✓
- 1X2 market appears in results
- League breakdown CSV exists and has data
- Brier scores < 0.25

**Bad Results**:
- ROI > 10% (margins not applied!)
- No 1X2 market
- Empty league breakdown file
- KeyError: 'Avg_Brier_Score' (should be fixed)

### 2. Historical Data Enrichment ✅

**Tests**:
- `Home_DaysSinceLast` column exists
- `Away_DaysSinceLast` column exists
- Coverage > 80% (most matches have data)

**Location**: `data/processed/historical_matches.parquet`

**Good Results**:
```
✓ Home_DaysSinceLast: 12,450/13,000 (95.8% coverage)
  Average: 6.8 days
✓ Away_DaysSinceLast: 12,450/13,000 (95.8% coverage)
  Average: 6.8 days
```

**Bad Results**:
- Columns missing (run Step 5 in run_weekly.py)
- Coverage < 50% (API issues or fallback to league-only)

### 3. Fixtures Enrichment ✅

**Tests**:
- Weekly fixtures have days-since-last-match columns
- Data includes cup games (not just league)

**Location**: `outputs/weekly_bets_lite.csv`

**Good Results**:
```
✓ Home_DaysSinceLast: 150/150 (100% coverage)
✓ Away_DaysSinceLast: 150/150 (100% coverage)
```

**Bad Results**:
- Columns missing (run Step 9 in run_weekly.py)
- All values same (fallback to league-only)

### 4. Market Splitting ✅

**Tests**:
- Separate HTML/CSV files per market
- All 8 markets generated (1X2, BTTS, 6 O/U lines)

**Expected Files**:
```
outputs/
├── predictions_1x2.csv/html
├── predictions_btts.csv/html
├── predictions_ou_0_5.csv/html
├── predictions_ou_1_5.csv/html
├── predictions_ou_2_5.csv/html
├── predictions_ou_3_5.csv/html
├── predictions_ou_4_5.csv/html
└── predictions_ou_5_5.csv/html
```

**Good Results**:
```
✓ 1X2: 150 predictions
✓ BTTS: 150 predictions
✓ OU_2_5: 150 predictions
...
Found 8/8 markets
```

**Bad Results**:
- Files missing (run Step 11 in run_weekly.py)
- Empty files
- < 6 markets found

### 5. Cache System ✅

**Tests**:
- Cache file created
- Entries have correct structure
- Reduces API calls on subsequent runs

**Location**: `data/match_schedule_cache.json`

**Good Results**:
```
✓ Cache loaded: 245 entries

Sample entry:
  Team/Date: Arsenal_2024-11-30
  Days since last match: 4
  Cached at: 2024-11-29T15:30:00
```

**Note**: Only exists if Claude API is configured. Skipped otherwise.

## Interpreting Results

### Test Output Example

```
======================================================================
 TEST 1: BACKTEST UPDATES
======================================================================
✓ backtest_summary.csv: Summary with conservative ROI
    Rows: 8, Columns: ['Total_Matches', 'Correct', 'Accuracy_%', 'Brier_Score', 'ROI_%']
    Average ROI: -3.2% (negative = conservative ✓)
    Average Brier Score: 0.187 (<0.25 = well-calibrated)
✓ backtest_detailed.csv: Period-by-period breakdown
    Rows: 32, Columns: ['period_start', 'period_end', 'league', 'market', ...]
✓ backtest_league_breakdown.csv: League-specific performance
    Rows: 24, Columns: ['league', 'market', 'total', 'correct', 'accuracy_%']

======================================================================
 TEST 2: HISTORICAL DATA ENRICHMENT
======================================================================
✓ Historical data loaded: 12,543 matches
  Columns: 87
  ✓ Home_DaysSinceLast: 11,958/12,543 (95.3% coverage)
    Average: 6.8 days
  ✓ Away_DaysSinceLast: 11,958/12,543 (95.3% coverage)
    Average: 6.8 days

... [more tests]

======================================================================
 TEST SUMMARY
======================================================================

Results:
  ✓ Passed: 5
  ✗ Failed: 0
  ⊘ Skipped: 0

Detailed:
  ✓ PASS: backtest
  ✓ PASS: historical
  ✓ PASS: fixtures
  ✓ PASS: markets
  ✓ PASS: cache

======================================================================
 RECOMMENDATIONS
======================================================================

✅ ALL TESTS PASSED!
   All updates are working correctly
```

## Troubleshooting

### Test 1 Fails (Backtest)

**Problem**: Backtest files missing

**Solution**:
```bash
python backtest_config.py
```

### Test 2 Fails (Historical Enrichment)

**Problem**: Columns missing from historical data

**Solution**:
```bash
# Run full pipeline (Step 5 will enrich)
python run_weekly.py

# Or just enrich historical data
python -c "
from days_since_match_fetcher import add_days_since_features
import pandas as pd
from pathlib import Path

hist_path = Path('data/processed/historical_matches.parquet')
df = pd.read_parquet(hist_path)
df = add_days_since_features(df, use_api=True)
df.to_parquet(hist_path, index=False)
print('Historical data enriched!')
"
```

### Test 3 Fails (Fixtures Enrichment)

**Problem**: weekly_bets_lite.csv doesn't have days columns

**Solution**:
```bash
# Run weekly pipeline
python run_weekly.py
```

This runs Step 9 which enriches fixtures.

### Test 4 Fails (Market Splitting)

**Problem**: Market files not generated

**Solution**:
```bash
# Run market splitter manually
python market_splitter.py

# Or run full pipeline
python run_weekly.py
```

### Test 5 Shows "Skipped" (Cache)

**Not a Problem**: This means Claude API isn't configured.

**Optional**: Set up API (see SETUP_CLAUDE_API.md)
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-api-YOUR-KEY"
pip install anthropic
```

**Alternative**: System uses league-only data (works but less accurate)

## Common Issues

### "ROI is positive and high (>10%)"

**Cause**: Bookmaker margins not applied

**Check**:
```python
import pandas as pd
df = pd.read_csv('outputs/backtest_summary.csv')
print(df[['Accuracy_%', 'ROI_%']])
```

If ROI ≈ (Accuracy - 50) * 2, margins aren't applied.

**Fix**: Should be fixed in latest code. If not, update backtest.py.

### "KeyError: 'Avg_Brier_Score'"

**Cause**: Old column name

**Fix**: Already fixed in backtest_config.py (changed to 'Brier_Score')

### "Coverage < 50% for days-since-last-match"

**Cause**: Either:
1. API not configured (fallback to league-only)
2. API errors
3. Data quality issues

**Check**:
```bash
# Test API directly
python days_since_match_fetcher.py "Arsenal" "2024-11-30" "E0"
```

Should print: `Arsenal last played X days before 2024-11-30`

If error, check ANTHROPIC_API_KEY is set.

### "Market files empty"

**Cause**: No predictions generated yet

**Fix**: Run predictions first:
```bash
python run_weekly.py
```

Then market splitter runs automatically (Step 11).

## Success Criteria

All tests should show:

✅ **Backtest**:
- ROI negative (conservative)
- 1X2 market present
- League breakdown exists
- Brier < 0.25

✅ **Historical Data**:
- Days columns exist
- Coverage > 80%

✅ **Fixtures**:
- Days columns exist
- Coverage > 90%

✅ **Markets**:
- 6-8 market files generated
- HTML + CSV for each

✅ **Cache**:
- File exists (if API configured)
- OR skipped (if no API - this is fine)

## Next Steps After Testing

### If All Tests Pass

Great! Your system is fully updated:

1. **Use the backtest** to validate performance:
   ```bash
   python backtest_config.py
   ```

2. **Run weekly predictions** with all enhancements:
   ```bash
   python run_weekly.py
   ```

3. **Check market-specific files** in outputs/:
   - `predictions_ou_2_5.html` for O/U 2.5 bets
   - `predictions_1x2.html` for match results
   - etc.

### If Some Tests Fail

Follow recommendations in test output:
- Run backtest if backtest tests failed
- Run run_weekly.py if enrichment/market tests failed
- Set up Claude API for better coverage (optional)

## Files Reference

**Test Scripts**:
- `test_all_updates.py` - Comprehensive test suite
- `RUN_TESTS.bat` - Windows batch runner

**Documentation**:
- `TESTING_GUIDE.md` (this file)
- `SETUP_CLAUDE_API.md` - API setup instructions
- `BACKTEST_FIXES.md` - Details of backtest updates
- `README_DAYS_SINCE_MATCH.md` - Days-since-last-match docs
- `README_MARKET_OUTPUTS.md` - Market splitting docs
- `ROI_EXPLAINED.md` - Conservative ROI calculation

**Main Scripts**:
- `backtest_config.py` - Run backtest
- `run_weekly.py` - Full pipeline (21 steps)
- `market_splitter.py` - Split by market
- `days_since_match_fetcher.py` - Enrich with days

---

**Quick Test**: `python test_all_updates.py`
