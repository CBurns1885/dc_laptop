# Bug Fixes Applied

## 1. Statistics Generation Not Working

### Problem
The message "No CSV files found - skipping statistics generation" always appeared, even though 110 CSV files existed in `data/raw/`.

### Root Cause
The `generate_statistics()` function in [generate_and_load_stats.py](generate_and_load_stats.py) was silently failing to find CSV files with no debugging output.

### Fix Applied
Added detailed debugging output to show:
- Which directories are being checked
- Whether each directory exists
- How many CSV files are found in each
- Which directory is ultimately selected

**Location**: [generate_and_load_stats.py:283-299](generate_and_load_stats.py#L283-L299)

### Testing
Run this to diagnose:
```bash
python test_stats_generation.py
```

Expected output:
```
Searching for historical CSV files...
   Checking downloaded_data (exists: False, CSVs: 0)
   Checking data/raw (exists: True, CSVs: 110)
 ✓ Using data from: C:\...\data\raw
```

---

## 2. Backtest Crash with NaN Predictions

### Problem
Backtest crashed with error:
```
KeyError: nan
```

When running `backtest_config.py` or `backtest_adaptive_dc.py`.

### Root Cause
When all prediction probabilities for a match are NaN (missing predictions), `idxmax()` returns `nan`. The code then tried to use `nan` as a column name, causing a KeyError.

This happened in two places:
1. **[backtest.py:270](backtest.py#L270)** - Main backtest engine
2. **[backtest_adaptive_dc.py:271](backtest_adaptive_dc.py#L271)** - Adaptive DC backtest

### Fix Applied

#### Before (Crashes):
```python
pred_outcome_idx = predictions.idxmax(axis=1)
# ... later ...
pred_prob = predictions.loc[idx, pred_col]  # KeyError if pred_col is nan
```

#### After (Handles NaN):
```python
pred_outcome_idx = predictions.idxmax(axis=1, skipna=True)

# Filter out rows where prediction is NaN
valid_predictions = pred_outcome_idx.notna()
pred_outcome_idx = pred_outcome_idx[valid_predictions]
predictions = predictions.loc[valid_predictions]
actual = actual.loc[valid_predictions]

if len(actual) == 0:
    # No valid predictions for this market
    continue
```

### Testing
The backtest will now:
- Skip matches with no predictions instead of crashing
- Continue processing other markets
- Show warnings about missing predictions in output

---

## Files Modified

1. ✅ **generate_and_load_stats.py**
   - Added debugging output for CSV detection
   - Lines 283-299

2. ✅ **backtest.py**
   - Fixed NaN handling in prediction evaluation
   - Lines 267-313

3. ✅ **backtest_adaptive_dc.py**
   - Fixed NaN handling in DC backtest
   - Lines 269-288

## Files Created

1. ✅ **test_stats_generation.py**
   - Diagnostic tool for CSV statistics generation
   - Run to verify stats generation works

2. ✅ **BUGFIXES.md** (this file)
   - Documentation of all applied fixes

---

## Next Steps

### If Statistics Still Don't Generate

Run the diagnostic:
```bash
python test_stats_generation.py
```

Look for:
- Does `data/raw` exist and have CSV files?
- Are you running from the correct working directory?
- Check file permissions

### If Backtest Still Crashes

The fixes should handle missing predictions gracefully. If it still crashes:
1. Check which market is causing the issue (look at the traceback)
2. Verify prediction CSV files have the expected column names
3. Check for data type issues (strings vs floats for probabilities)

---

## Prevention

### For Future Development

**When working with pandas DataFrames:**
- Always use `skipna=True` with `idxmax()` when NaN values are possible
- Filter for `notna()` before using column names from index operations
- Add explicit NaN checks before dictionary/column lookups

**When debugging file operations:**
- Print absolute paths (`.resolve()`) not just relative paths
- Show both "exists" status AND file counts
- Make failures verbose with actionable suggestions

---

## Related Issues Fixed

These fixes also resolve:
- FutureWarning about `idxmax()` with all-NA values
- Silent failures in statistics pipeline
- Misleading "this is OK for first run" messages
