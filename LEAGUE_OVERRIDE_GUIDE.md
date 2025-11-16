# League Override Configuration Guide

## Quick Reference

Open `run_weekly.py` and find this section at the top (around line 31):

```python
# ============================================================================
# LEAGUE OVERRIDE OPTION
# ============================================================================
# Set to True to use ALL leagues regardless of fixtures file
# Set to False (default) to auto-detect leagues from fixtures
USE_ALL_LEAGUES = False
```

## Usage

### Option 1: Auto-Detect Leagues (Default - Recommended)
**Set:** `USE_ALL_LEAGUES = False`

**Behavior:**
- Reads your `upcoming_fixtures.csv` file
- Automatically detects which leagues have matches
- Downloads and trains only on those leagues
- **Much faster** if you only need a few leagues

**Example:**
```
Fixtures have: E1, E2, EC, SP2 (4 leagues)
→ Downloads only those 4 leagues
→ Trains on ~4,500 matches instead of 25,000
→ 80% faster! (3 min vs 15 min)
```

**Best For:**
- Weekly predictions on specific leagues
- Quick testing
- When you know which leagues you want

---

### Option 2: Force All Leagues
**Set:** `USE_ALL_LEAGUES = True`

**Behavior:**
- Ignores what's in fixtures file
- Downloads ALL 23 supported leagues
- Trains on all historical data (~25,000 matches)
- Slower but comprehensive

**Example:**
```
Fixtures have: E1, E2 (2 leagues)
→ But downloads all 23 leagues anyway
→ Trains on all ~25,000 matches
→ Takes full 15-20 min
```

**Best For:**
- Building comprehensive models once per season
- When you want predictions available for any league
- Initial model training at start of season

---

## How It Works

### When `USE_ALL_LEAGUES = False` (Auto-Detect):

**Step 1: Download**
```
Reading fixtures...
   Found leagues: ['E1', 'E2', 'EC', 'SP2']
📋 Leagues found in fixtures: ['E1', 'E2', 'EC', 'SP2']
📅 Downloading 2021-2025...
   Leagues: ['E1', 'E2', 'EC', 'SP2']  ← Only 4 leagues
   Years: 2021-2025 (5 seasons)
```

**Step 5: Feature Building**
```
🔧 Filtering to leagues: ['E1', 'E2', 'EC', 'SP2']
   Filtered: 22 leagues → 4 leagues
   Matches: 25,501 → 4,511 (17.7%)
✅ Historical data filtered to fixture leagues only
```

**Step 6: Training**
```
Fitted E1: 836 matches
Fitted E2: 1819 matches
Fitted EC: 507 matches
Fitted SP2: 1517 matches
⏱️ Time: 3 min (80% faster!)
```

---

### When `USE_ALL_LEAGUES = True` (Force All):

**Step 1: Download**
```
🔧 USE_ALL_LEAGUES=True - Using all 23 leagues
📅 Downloading 2021-2025...
   Leagues: ['B1', 'D1', 'D2', 'E0', 'E1', ... all 23]  ← All leagues
   Years: 2021-2025 (5 seasons)
```

**Step 2: Build Database**
```
📋 Building database for all leagues (USE_ALL_LEAGUES=True)
```

**Step 5: Feature Building**
```
Building features for 25,000 matches across 23 leagues
⏱️ Time: 3 min
```

**Step 6: Training**
```
Fitted B1: 1034 matches
Fitted D1: 999 matches
Fitted E0: 1240 matches
... (all 23 leagues)
⏱️ Time: 15 min
```

---

## Performance Comparison

| Setting | Leagues | Matches | Download Time | Training Time | Total Time |
|---------|---------|---------|---------------|---------------|------------|
| `False` (4 leagues) | 4 | ~4,500 | ~2 min | ~3 min | **~8 min** ✅ |
| `True` (all) | 23 | ~25,000 | ~8 min | ~15 min | **~25 min** |

**Savings with Auto-Detect:** ~17 minutes (68% faster)

---

## Incremental Training Impact

### With Auto-Detect (`USE_ALL_LEAGUES = False`):
```
Monday: Train on ['E1', 'E2'] → Save models (3 min)
Tuesday: Same leagues → Load models (30 sec) ⚡
Wednesday: Different leagues ['SP1', 'SP2'] → Retrain (3 min)
```

### With All Leagues (`USE_ALL_LEAGUES = True`):
```
Monday: Train on all 23 → Save models (15 min)
Tuesday: Same leagues → Load models (30 sec) ⚡
Wednesday: Same leagues → Load models (30 sec) ⚡
```

**Trade-off:**
- Auto-detect: Faster per run, but retrains if leagues change
- All leagues: Slower first run, but never needs retraining

---

## Recommended Strategy

### For Weekly Predictions:
```python
USE_ALL_LEAGUES = False  # Auto-detect from fixtures
```
**Why:** Much faster, you only predict specific leagues anyway

### For Seasonal Model Building:
```python
USE_ALL_LEAGUES = True  # Build comprehensive models
```
**Why:** Train once at start of season, then use incremental loading

### For Testing/Development:
```python
USE_ALL_LEAGUES = False  # Fast iterations
```
**Why:** Quick feedback, easy testing

---

## Configuration Summary Display

When you run the script, you'll see:

### With Auto-Detect:
```
============================================================
FOOTBALL PREDICTION SYSTEM - WEEKLY RUN
============================================================

📋 Configuration:
   Training Years: 2021-2025
   Optuna Trials: 0
   League Mode: AUTO-DETECT FROM FIXTURES
============================================================
```

### With All Leagues:
```
============================================================
FOOTBALL PREDICTION SYSTEM - WEEKLY RUN
============================================================

📋 Configuration:
   Training Years: 2021-2025
   Optuna Trials: 0
   League Mode: ALL LEAGUES
   Leagues: 23 leagues (E0, E1, D1, SP1, etc.)
============================================================
```

---

## Troubleshooting

### Problem: Still downloading all leagues when `USE_ALL_LEAGUES = False`

**Check:**
1. Is the setting actually `False` (not commented out)?
2. Does your fixtures file actually have fewer leagues?
3. Look for this message: `📋 Leagues found in fixtures: [...]`

**Debug:**
```python
# In run_weekly.py, around line 31, check:
USE_ALL_LEAGUES = False  # ← Should be False, not True
```

### Problem: Want to use fixtures but add extra leagues

**Solution:** Manually edit your `upcoming_fixtures.csv` to include dummy rows for extra leagues, or set `USE_ALL_LEAGUES = True`

---

## Files Modified

1. ✅ `run_weekly.py` - Added `USE_ALL_LEAGUES` flag (line 33)
2. ✅ `run_weekly.py` - Step 1 checks override (line 189)
3. ✅ `run_weekly.py` - Step 2 checks override (line 239)
4. ✅ `run_weekly.py` - Configuration display (line 48)

---

## Example Use Cases

### Use Case 1: "I only care about English leagues this week"
```python
USE_ALL_LEAGUES = False
```
→ Put only English leagues in fixtures → Fast 3-minute run

### Use Case 2: "I want models ready for any league"
```python
USE_ALL_LEAGUES = True
```
→ Train once with all leagues → Use incremental loading

### Use Case 3: "Testing new features"
```python
USE_ALL_LEAGUES = False
```
→ Create minimal fixtures file → Super fast iterations

---

## Summary

**Default Behavior (Recommended):**
```python
USE_ALL_LEAGUES = False  # Auto-detect from fixtures
```
✅ Fast, efficient, only processes what you need

**Override (When Needed):**
```python
USE_ALL_LEAGUES = True  # Force all leagues
```
✅ Comprehensive, but slower

**Simply change this one line at the top of `run_weekly.py` to switch modes!**
