# Backtest Improvements Applied

## Fixed Issues

### 1. ✅ Added 1X2 Market Testing

**Problem**: Backtest only tested BTTS and O/U markets, not 1X2 (match result)

**Fix**: Added 1X2 to markets dict in [backtest.py:206-211](backtest.py#L206-L211)

```python
'1X2': {
    'actual': 'y_1X2',
    'pred_cols': ['P_1X2_H', 'P_1X2_D', 'P_1X2_A'],
    'outcomes': ['H', 'D', 'A']
},
```

Now the backtest will test all markets including match results!

---

### 2. ✅ Fixed ROI Calculation (CONSERVATIVE)

**Problem**: ROI used simplified formula `((accuracy - 0.5) * 100)` assuming 50/50 odds

**Old Method** ([backtest.py:337](backtest.py#L337)):
```python
roi = ((correct / total) - 0.5) * 100  # Wrong!
```

This assumed:
- All bets at even money (2.0 odds)
- 50% accuracy = breakeven
- Doesn't consider actual predicted probabilities

**New Method** ([backtest.py:342-389](backtest.py#L342-L389)):
```python
# For each prediction:
# 1. Get predicted probability
# 2. Calculate fair odds = 1 / probability
# 3. Apply bookmaker margin (4-8% depending on market)
# 4. If correct: profit = (bookmaker_odds - 1)
# 5. If wrong: profit = -1
# 6. ROI = total_profit / total_bets * 100
```

**Bookmaker Margins Applied (Conservative)**:
- **O/U 2.5**: 4% margin (most liquid market)
- **O/U 0.5, 1.5, 3.5**: 5% margin
- **1X2**: 6% margin (competitive)
- **O/U 4.5**: 6% margin
- **O/U 5.5**: 7% margin (lower liquidity)
- **BTTS**: 8% margin

**Example**:
- Old: 70% accuracy → ROI = 20% (unrealistic!)
- Fair odds: 70% accuracy at 70% confidence → odds = 1.43 → ROI = 0% (breakeven)
- **New (Conservative)**: 70% at 70% → fair odds 1.43 → bookmaker odds 1.37 (4% margin) → **ROI = -4.3%**

This now properly accounts for:
- Betting at your predicted probabilities
- **Realistic bookmaker margins** (conservative estimate)
- Market-specific odds reduction
- Real-world betting conditions

---

### 3. ✅ Improved League Breakdown

**Problem**: `analyze_by_league()` did nothing (just printed a message)

**Fix**: Now actually analyzes league performance ([backtest.py:636-670](backtest.py#L636-L670))

**Features**:
- Reads `backtest_detailed.csv`
- Groups by League + Market
- Shows best market per league
- Saves to `backtest_league_breakdown.csv`

**Example Output**:
```
Top Markets by League:
   E0: OU_2_5 (68.5% accuracy)
      123/180 correct
   SP1: OU_3_5 (71.2% accuracy)
      89/125 correct
```

**Note**: This currently works at the period level. For per-match league tracking, would need to modify `evaluate_predictions()` to track league per prediction.

---

## ROI Calculation Deep Dive

### Why The Old Method Was Wrong

**Scenario**: You predict Home Win at 80% confidence

**Old calculation**:
- 80% accuracy
- ROI = (0.80 - 0.50) * 100 = **30%**
- Assumes you're getting 2.0 odds (implied 50% probability)

**But reality**:
- If you bet at 80% confidence, fair odds = 1.25
- Bookmaker offers 1.25 odds (fair market)
- If you win: get back 1.25x stake = 0.25 profit
- If you lose: lose 1x stake = -1.00 loss
- Over 100 bets: 80 wins × 0.25 = +20 units, 20 losses × -1 = -20 units
- **Total profit = 0 units** (breakeven)

**New calculation** (correctly):
- Profit = 80 × (1.25 - 1) - 20 × 1 = 20 - 20 = **0%** ROI ✓

### When Do You Make Money?

**Only when accuracy > predicted probability**

If you predict 70% but you're actually 75% accurate:
- Fair odds = 1 / 0.70 = 1.43
- Over 100 bets: 75 wins × (1.43-1) = +32.1, 25 losses = -25
- Profit = +7.1 units = **7.1% ROI** ✓

This is edge betting - you found mispriced odds!

### Realistic ROI Expectations

| Accuracy | Confidence | Fair Odds | ROI |
|----------|-----------|-----------|-----|
| 60% | 60% | 1.67 | 0% (breakeven) |
| 65% | 60% | 1.67 | +8.3% (value!) |
| 70% | 70% | 1.43 | 0% (breakeven) |
| 75% | 70% | 1.43 | +7.1% (value!) |
| 80% | 75% | 1.33 | +6.7% (value!) |

**Key insight**: ROI depends on being BETTER than your predictions suggest.

---

## Testing The Fixes

Run backtest again to see:

1. **1X2 results** (new market in output)
2. **Realistic ROI** (probably much lower, possibly negative)
3. **League breakdown** (new CSV file generated)

**Expected changes**:
- ROI will likely be **much lower** (more realistic)
- You might see **negative ROI** even with good accuracy (means well-calibrated)
- **1X2 market** will appear in results
- **League breakdown** will show per-league performance

---

## Understanding Your Results

### If ROI is negative but accuracy is good:
✅ **This is actually good!** It means your probabilities are well-calibrated. You're honest about uncertainty.

### If ROI is positive:
🎯 **You found value!** Your model is better than it thinks it is. Either:
- Increase bet sizing on high-confidence bets
- Or recalibrate to be more confident

### If accuracy is high but ROI is zero:
⚖️ **Perfect calibration.** Your probabilities exactly match reality. Hard to beat!

---

## Next Steps

1. **Run backtest** with these fixes
2. **Check ROI** - expect it to be lower
3. **Review league breakdown** - find which leagues you're best at
4. **Test 1X2 performance** - see if match result predictions are any good

## Files Modified

- ✅ [backtest.py](backtest.py) - Added 1X2, fixed ROI, improved league analysis
- ✅ [backtest_config.py](backtest_config.py) - Fixed column name handling

## New Output Files

When you run backtest now, you'll get:
- `backtest_summary.csv` - Overall performance (with realistic ROI)
- `backtest_detailed.csv` - Period-by-period (with league data)
- `backtest_league_breakdown.csv` - **NEW!** Per-league performance
- `backtest_best_doubles.csv` - Best double combinations
- `backtest_best_trebles.csv` - Best treble combinations
