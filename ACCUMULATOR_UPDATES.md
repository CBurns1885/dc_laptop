# O/U Accumulator Updates - Enhanced Grouping

## Changes Made

### 1. Maximize Accumulators
**Before:** Only created 4-folds if exactly 4 bets were available
**After:**
- Creates as many 4-folds as possible
- Uses remaining bets (2-3) to create smaller accumulators
- No bets left behind!

### 2. Date/Time Sorting
**Before:** Sorted by confidence only
**After:**
- Sorts matches chronologically by date and time
- Groups matches that play at similar times
- Better for tracking and betting

### 3. Enhanced Display
**Before:** Only showed match and confidence
**After:** Each leg now shows:
- Match details (Home vs Away)
- **Date and Time**
- **League**
- Confidence percentage

## Example Output

### Accumulator #1 (4-Fold)
```
OU_4.5 Under - Pisa vs Parma
  12/12/2025 | I2
  Confidence: 96.1%

OU_4.5 Under - Padova vs Cesena
  12/12/2025 | I2
  Confidence: 98.1%

OU_4.5 Under - Guimaraes vs Gil Vicente
  12/12/2025 | P1
  Confidence: 97.2%

OU_1.5 Over - Avellino vs Venezia
  12/12/2025 | I2
  Confidence: 96.1%

Combined Probability: 89.7%
Implied Odds: @1.11
```

## Test Results

From 10 unique matches with 90%+ confidence:
- **2 four-folds** created (8 matches)
- **1 two-fold** created (2 remaining matches)
- **Total: 3 accumulators** using all 10 matches

## How to Regenerate

Run the batch file:
```bash
regenerate_accumulators.bat
```

Or run directly:
```bash
py -c "from generate_outputs_from_actual import generate_ou_accumulators; generate_ou_accumulators('outputs/weekly_bets_lite.csv')"
```

## Files Updated

1. **generate_outputs_from_actual.py**
   - Added Time field extraction (line 179-181)
   - Added DateTime sorting (line 229-232)
   - Maximized 4-fold creation (line 243-265)
   - Handle remaining bets (line 267-284)
   - Enhanced HTML display with date/time/league (line 357-367)

2. **regenerate_accumulators.bat** (NEW)
   - Quick batch file to regenerate accumulators

## Key Features

✅ Each match appears only once (best O/U line)
✅ Sorted chronologically by date/time
✅ Maximum number of accumulators created
✅ Remaining matches used in smaller accumulators
✅ Date/Time/League shown for each leg
✅ All matches with 90%+ confidence included

---
Updated: 2025-12-13
