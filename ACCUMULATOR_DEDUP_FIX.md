# O/U Accumulator Deduplication Fix

## Problem
The O/U accumulator generator was creating multiple bets for the same match across different O/U lines (1.5, 3.5, 4.5), which meant:
- The same match could appear in multiple accumulators
- Arsenal vs Chelsea might appear as "OU_1.5 Over" in one accumulator AND "OU_3.5 Under" in another
- This created redundancy and potential conflicts

## Solution
Added deduplication logic to keep only the **BEST** bet per match (highest confidence).

## Changes Made

### File: generate_outputs_from_actual.py

**Lines 217-220 (NEW):**
```python
# Remove duplicates: Keep only the BEST bet per match (highest confidence)
print(f" Found {len(bets_df)} O/U bets before deduplication")
bets_df = bets_df.sort_values('Confidence', ascending=False)
bets_df = bets_df.drop_duplicates(subset='Match', keep='first')
```

**How it works:**
1. Sort all bets by confidence (highest first)
2. Drop duplicate matches, keeping only the first (highest confidence)
3. Result: Each match appears only once with its best O/U line

**Updated HTML message (Line 304):**
```html
<p><strong>Note:</strong> Each match appears only once (best O/U line selected)</p>
```

## Example

### Before Fix:
```
Arsenal vs Chelsea - OU_1.5 Over (92%)
Arsenal vs Chelsea - OU_3.5 Under (94%)  <- Best
Arsenal vs Chelsea - OU_4.5 Under (91%)
```
All three could be used in different accumulators.

### After Fix:
```
Arsenal vs Chelsea - OU_3.5 Under (94%)  <- Only this one kept (highest confidence)
```
Only the best bet is kept and used in accumulators.

## Testing

Run `py test_accumulator_dedup.py` to verify the logic:

```
BEFORE Deduplication:
Total bets: 6
Unique matches: 3

AFTER Deduplication:
Total bets: 3
Unique matches: 3

VERIFICATION:
[OK] SUCCESS: Each match appears exactly once
[OK] Arsenal vs Chelsea: Kept best bet (94.0%)
[OK] Liverpool vs Man Utd: Kept best bet (95.0%)
[OK] Man City vs Spurs: Kept best bet (96.0%)
```

## Impact

- Each match now appears in **at most one accumulator**
- Always uses the highest confidence O/U line for each match
- Eliminates redundancy and conflicting predictions
- Cleaner, more focused accumulator recommendations

## Next Steps

When you run `py run_weekly.py`, Step 16 (O/U ACCUMULATORS) will now:
1. Find all O/U bets with 90%+ confidence
2. Deduplicate by match (keep best)
3. Build 4-fold accumulators with unique matches only

---
Fixed: 2025-12-13
