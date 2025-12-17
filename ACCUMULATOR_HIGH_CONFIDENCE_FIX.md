# Accumulator Generator - High Confidence Bets Support

## Problem
The accumulator generator was designed to work with `weekly_bets_lite.csv` which has DC prediction columns (DC_OU_1_5_O, DC_OU_1_5_U, etc.), but couldn't read the simpler `high_confidence_bets.csv` format.

## Solution
Updated `generate_ou_accumulators()` to automatically detect and handle both file formats:

### Format 1: high_confidence_bets.csv (Simple)
```csv
Date,League,HomeTeam,AwayTeam,Market,Prediction,Confidence
2025-12-13,N1,PSV Eindhoven,Heracles,OU_1_5,Over,99.9
```

### Format 2: weekly_bets_lite.csv (Full)
```csv
Date,HomeTeam,AwayTeam,DC_OU_1_5_O,DC_OU_1_5_U,DC_OU_3_5_O,DC_OU_3_5_U,...
```

## Changes Made

### File: generate_outputs_from_actual.py

**Lines 165-182:** Auto-detect file format
```python
if 'Market' in df.columns and 'Prediction' in df.columns and 'Confidence' in df.columns:
    # High confidence format - already has Market, Prediction, Confidence
    bets_df = df[df['Market'].isin(['OU_1_5', 'OU_3_5', 'OU_4_5'])].copy()
    bets_df['Match'] = bets_df['HomeTeam'] + ' vs ' + bets_df['AwayTeam']

    # Convert confidence from percentage (0-100) to decimal (0-1)
    if bets_df['Confidence'].max() > 1:
        bets_df['Confidence'] = bets_df['Confidence'] / 100
else:
    # Weekly bets lite format - extract from DC columns
    # [existing logic...]
```

**Line 254:** Fixed date parsing warning
```python
# Changed from dayfirst=True to format='mixed'
bets_df['DateTime'] = pd.to_datetime(bets_df['Date'] + ' ' + bets_df['Time'].fillna(''),
                                      errors='coerce', format='mixed')
```

## Test Results

### From high_confidence_bets.csv:
```
Total bets: 119 O/U bets
Unique matches: 94 matches (after deduplication)
Confidence range: 90.1% - 99.9%
Accumulators created: 24 total
  - 23 four-folds (92 matches)
  - 1 two-fold (2 remaining matches)
```

### Sample Accumulator:
```
4-Fold Accumulator #1
├─ Combined Confidence: 90.1%
├─ Combined Probability: 77.6%
└─ Implied Odds: @1.29
```

## Files Updated

1. **run_weekly.py** (Line 566)
   - Changed from `weekly_bets_lite.csv` to `high_confidence_bets.csv`

2. **regenerate_accumulators.bat**
   - Updated to use `high_confidence_bets.csv`

3. **generate_outputs_from_actual.py**
   - Auto-detect file format
   - Handle both simple and full formats
   - Fixed date parsing warning

## Usage

**Run the batch file:**
```bash
.\regenerate_accumulators.bat
```

**Or in PowerShell:**
```powershell
py -c "from generate_outputs_from_actual import generate_ou_accumulators; generate_ou_accumulators('outputs/high_confidence_bets.csv')"
```

**Output files:**
- `outputs/ou_accumulators.csv`
- `outputs/ou_accumulators.html`

## Benefits

✅ Works with both file formats automatically
✅ Uses high confidence bets (90%+ only)
✅ Creates maximum number of accumulators (24 from 94 matches)
✅ Each match appears only once (best O/U line selected)
✅ Sorted chronologically by date/time
✅ No more date parsing warnings

---
Fixed: 2025-12-13
