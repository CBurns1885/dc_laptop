# New Streamlined Outputs

## What Changed

**OLD** (too many files):
- ❌ weekly_bets.csv
- ❌ top50_weighted.html/csv
- ❌ ou_analysis.html/csv
- ❌ accumulators_safe.html
- ❌ accumulators_mixed.html
- ❌ accumulators_aggressive.html
- ❌ quality_bets.html
- ❌ 16+ market-specific files (predictions_1x2.html, etc.)

**NEW** (just 2 main files + market splits):
- ✅ **high_confidence_bets.html/csv** - ALL predictions over 90% (any market)
- ✅ **ou_accumulators.html/csv** - 2-4 fold accumulators (O/U 1.5/3.5/4.5 only)
- ✅ Market splits (16 files, optional: predictions_1x2.html, predictions_ou_2_5.html, etc.)

## Output 1: High Confidence Bets (90%+)

**File**: `outputs/high_confidence_bets.html`

**What it shows**:
- ALL predictions over 90% confidence
- Covers ALL markets (1X2, BTTS, O/U 0.5-5.5)
- Sorted by confidence (highest first)

**Format**:
```
Date  | League | Match              | Market  | Prediction | Confidence | Probabilities
------|--------|-------------------|---------|------------|------------|---------------
Dec 8 | E0     | Arsenal vs Chelsea | OU_2_5  | Over       | 95.2%      | Over: 95.2%, Under: 4.8%
Dec 8 | E0     | Man City vs Spurs  | BTTS    | Yes        | 93.1%      | Yes: 93.1%, No: 6.9%
Dec 9 | SP1    | Real vs Barca      | OU_3_5  | Over       | 91.8%      | Over: 91.8%, Under: 8.2%
...
```

**Color Coding**:
- 🟢 Green background = 95%+ confidence
- 🟡 Yellow background = 90-94% confidence

**Use case**: Your highest confidence bets across all markets. Start here!

## Output 2: O/U Accumulators (2-4 Fold)

**File**: `outputs/ou_accumulators.html`

**What it shows**:
- 2-fold, 3-fold, 4-fold accumulators
- O/U 1.5, 3.5, 4.5 markets ONLY
- Sorted by combined probability (most likely to win first)
- Top 50 accumulators

**Why these markets?**:
- O/U 1.5: High hit rate (usually Over)
- O/U 3.5: Medium goals (balanced)
- O/U 4.5: High goals (specific matches)
- Avoids O/U 2.5 (too mainstream) and 0.5/5.5 (too extreme)

**Format**:
```
#1: 4-Fold @8.50
Combined Probability: 72.3% | Min Confidence: 85.1%

Legs:
  - OU_1_5 Over (Arsenal vs Chelsea) 92.5%
  - OU_3_5 Over (Man City vs Spurs) 88.3%
  - OU_4_5 Under (Burnley vs Luton) 85.1%
  - OU_1_5 Over (Liverpool vs Brighton) 90.7%

---

#2: 3-Fold @4.20
Combined Probability: 78.5% | Min Confidence: 87.2%

Legs:
  - OU_1_5 Over (Real vs Barca) 91.8%
  - OU_3_5 Over (Bayern vs Dortmund) 89.4%
  - OU_1_5 Over (PSG vs Monaco) 87.2%
```

**Color Coding**:
- 🟢 Green = 70%+ combined probability (high chance)
- 🟡 Yellow = 50-70% combined probability (medium)
- 🔴 Red = <50% combined probability (risky, higher odds)

**Strategy**: Start with highest probability (top of list), work down as you want more risk/reward.

## Pipeline Integration

New streamlined steps in `run_weekly.py`:

**OLD** (Steps 15-18):
- Step 15: Generate weighted Top 50
- Step 16: O/U Analysis
- Step 17: Build Accumulators (3 files)
- Step 18: Find Quality Bets

**NEW** (Steps 15-16):
- **Step 15**: HIGH CONFIDENCE BETS (90%+) - Single file, all markets
- **Step 16**: O/U ACCUMULATORS (2-4 FOLD) - Single file, focused markets

**Result**: Reduced from 19 steps to 19 steps but cleaner output (fewer files)

## How to Test

### Quick Test
```bash
python test_new_outputs.py
```

This will:
1. Generate high_confidence_bets.html/csv
2. Generate ou_accumulators.html/csv
3. Show summary of what was created

### Full Pipeline
```bash
python run_weekly.py
```

Will generate everything including the 2 new files.

## What You Get

### Scenario: 150 Weekly Predictions

**Before** (messy):
- 20+ output files
- Accumulators scattered across 3 files (safe/mixed/aggressive)
- High-confidence bets mixed in with everything else
- Hard to find the "best bets"

**After** (clean):
- **high_confidence_bets.html**: 15 bets over 90% (your absolute best picks)
- **ou_accumulators.html**: 50 accumulator combinations (sorted by probability)
- Market splits: Optional (16 files if you want to filter by market)

## Examples

### High Confidence Bets

**Example Output**:
```
⭐ High Confidence Bets (90%+ Confidence)

Summary
Total Bets: 18
Confidence Range: 90.2% - 96.5%
Markets: OU_2_5, BTTS, OU_1_5, 1X2, OU_3_5
Leagues: E0, SP1, D1

Top Bets:
1. 96.5% - OU_2_5 Over (Arsenal vs Chelsea)
2. 94.8% - BTTS Yes (Man City vs Liverpool)
3. 93.2% - OU_1_5 Over (Real vs Barca)
4. 92.7% - 1X2 Home (Bayern vs Dortmund)
...
```

### O/U Accumulators

**Example Output**:
```
📊 O/U Accumulators (2-4 Fold)

Summary
Total Accumulators: 50
Markets: O/U 1.5, O/U 3.5, O/U 4.5
Max Fold: 4-fold
Strategy: Highest probability combinations

Top Accumulators:
#1: 4-Fold @8.20 (Combined: 75.3%)
#2: 3-Fold @5.10 (Combined: 79.8%)
#3: 4-Fold @9.50 (Combined: 71.2%)
...
```

## Benefits

### 1. Cleaner
- 2 main files instead of 8+
- Easy to find your best bets
- Less clutter in outputs/

### 2. Focused
- High confidence = 90%+ only (elite picks)
- Accumulators = O/U 1.5/3.5/4.5 only (proven markets)
- No noise from lower confidence bets

### 3. Actionable
- High confidence: Just bet these straight
- Accumulators: Pick your risk level (top = safer, bottom = riskier)
- Sorted by probability = easy decision making

## Migration

**Old files** (still generated but less useful):
- weekly_bets_lite.csv - Master file (still needed for other steps)
- Market splits - Still generated (Step 11) for detailed analysis

**New files** (what you should use):
- **high_confidence_bets.html** - Start here every week
- **ou_accumulators.html** - Check for accumulator opportunities

**What to ignore**:
- Top 50 weighted (removed)
- O/U analysis (replaced by high confidence)
- Multiple accumulator strategies (replaced by single sorted list)
- Quality bets finder (redundant with high confidence)

## Testing

```bash
# 1. Generate outputs
python run_weekly.py

# 2. Test new outputs
python test_new_outputs.py

# 3. Open in browser
outputs/high_confidence_bets.html
outputs/ou_accumulators.html
```

## Summary

**Before**: 20+ files, hard to find best bets, scattered information
**After**: 2 focused files, sorted by confidence/probability, actionable

Your workflow now:
1. Open `high_confidence_bets.html` → Bet these straight
2. Open `ou_accumulators.html` → Build your accas (top = safer)
3. Done! ✅

---

**Note**: Market splits (16 files) are still generated in Step 11 if you want to deep-dive into specific markets. But for weekly betting, the 2 new files are all you need.
