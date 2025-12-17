# Date Format Fix - UK Date Support

## Problem
The prediction pipeline was failing with error:
```
time data "13/12/2025" doesn't match format "%m/%d/%Y"
```

This happened because:
- Fixture files use UK date format: **DD/MM/YYYY** (13/12/2025 = 13th December)
- Python's pandas was defaulting to US format: **MM/DD/YYYY**
- This caused parsing errors for dates like 13/12/2025 (no 13th month!)

## Solution
Added `dayfirst=True` parameter to all `pd.to_datetime()` calls that parse fixture dates.

## Files Modified

### 1. predict.py (2 locations)
**Line 255:**
```python
# Before:
fx["Date"] = pd.to_datetime(fx["Date"])

# After:
fx["Date"] = pd.to_datetime(fx["Date"], dayfirst=True, errors='coerce')
```

**Line 928:**
```python
# Before:
fx["Date"] = pd.to_datetime(fx["Date"])

# After:
fx["Date"] = pd.to_datetime(fx["Date"], dayfirst=True, errors='coerce')
```

### 2. prep_fixtures.py (1 location)
**Line 15:**
```python
# Before:
df["Date"] = pd.to_datetime(df["Date"])

# After:
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors='coerce')
```

### 3. run_weekly.py (1 location)
**Line 269:**
```python
# Before:
fixtures_df['Date'] = pd.to_datetime(fixtures_df['Date'], errors='coerce')

# After:
fixtures_df['Date'] = pd.to_datetime(fixtures_df['Date'], dayfirst=True, errors='coerce')
```

## What This Does

- `dayfirst=True`: Tells pandas to interpret dates as DD/MM/YYYY instead of MM/DD/YYYY
- `errors='coerce'`: Converts invalid dates to NaT (Not a Time) instead of crashing
- Now handles UK date formats correctly: 13/12/2025, 31/01/2025, etc.

## Testing
Run `py test_date_fix.py` to verify the fix works correctly.

## Impact
This fix resolves:
- Step 10 (GENERATE PREDICTIONS) error
- All downstream steps (11, 14, 15, 16, 18) should now work

## Next Steps
Run the full pipeline again:
```bash
py run_weekly.py
```

The date parsing errors should be resolved!

---
Fixed: 2025-12-12
