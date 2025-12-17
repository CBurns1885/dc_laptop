# Conservative ROI Calculation Explained

## Overview

The backtest now uses **conservative, realistic ROI calculations** that account for actual bookmaker margins per market type.

## How It Works

### 1. Calculate Fair Odds

For each prediction, we start with the predicted probability:

```
Fair Odds = 1 / Predicted_Probability
```

**Example**: If you predict "Over 2.5 Goals" at 70% confidence:
- Fair Odds = 1 / 0.70 = **1.43**

### 2. Apply Bookmaker Margin

Bookmakers don't offer fair odds - they reduce payouts to guarantee profit. We apply realistic margins:

```
Bookmaker Odds = Fair Odds × (1 - Margin)
```

**Market-Specific Margins**:
| Market | Margin | Why? |
|--------|--------|------|
| O/U 2.5 | 4% | Most liquid market, high volume |
| O/U 0.5, 1.5, 3.5 | 5% | High volume markets |
| 1X2 | 6% | Competitive but 3-way market |
| O/U 4.5 | 6% | Medium liquidity |
| O/U 5.5 | 7% | Lower liquidity |
| BTTS | 8% | Two-way market, variable |

**Example (O/U 2.5 with 4% margin)**:
- Fair Odds = 1.43
- Bookmaker Odds = 1.43 × (1 - 0.04) = 1.43 × 0.96 = **1.37**

### 3. Calculate Profit Per Bet

For each bet (stake = 1 unit):

- **If Correct**: Profit = (Bookmaker_Odds - 1)
- **If Wrong**: Profit = -1

**Example (70% prediction at bookmaker odds 1.37)**:
- 100 bets at 1 unit each
- 70 correct: 70 × (1.37 - 1) = 70 × 0.37 = **+25.9 units**
- 30 wrong: 30 × (-1) = **-30 units**
- **Total Profit = -4.1 units**

### 4. Calculate ROI

```
ROI = (Total Profit / Total Bets) × 100
```

**Example**:
- Total Profit = -4.1 units
- Total Bets = 100
- **ROI = -4.1%**

## What This Means

### Negative ROI is Normal!

If you're well-calibrated (your 70% predictions hit 70% of the time), you'll have **negative ROI** because:
- Bookmaker margins eat your edge
- You're betting at fair value, not finding value

**Example ROI by Accuracy (at predicted 70% confidence)**:
| Accuracy | Fair Odds | Bookmaker Odds (4% margin) | ROI |
|----------|-----------|---------------------------|-----|
| 65% | 1.43 | 1.37 | -8.5% |
| 70% | 1.43 | 1.37 | -4.1% |
| 75% | 1.43 | 1.37 | +0.4% ✅ |
| 80% | 1.43 | 1.37 | +4.8% ✅ |

### When Do You Profit?

You need **accuracy > predicted probability** to overcome the margin:

**Break-even Formula**:
```
Required Accuracy = Predicted_Prob + (Predicted_Prob × Margin)
```

For 70% prediction with 4% margin:
```
Break-even = 0.70 + (0.70 × 0.04) = 0.70 + 0.028 = 72.8%
```

You need **72.8% accuracy** to break even when predicting at 70% confidence.

## Different Scenarios

### Scenario 1: Well-Calibrated Model
```
Prediction: 70% confidence
Actual: 70% accuracy
Odds: 1.37 (after 4% margin)
ROI: -4.1%
```

**Interpretation**: Your model is honest but doesn't find value. Conservative estimate shows expected loss from margins.

### Scenario 2: Overconfident Model
```
Prediction: 80% confidence
Actual: 70% accuracy
Fair Odds: 1.25
Bookmaker Odds: 1.20 (4% margin)
ROI: -16%
```

**Interpretation**: You're predicting too confidently, getting poor odds, and underperforming. Very bad!

### Scenario 3: Finding Value
```
Prediction: 70% confidence
Actual: 75% accuracy
Odds: 1.37 (based on your 70% prediction)
ROI: +0.4%
```

**Interpretation**: Your model is better than it thinks! You're finding undervalued bets. This is profit!

### Scenario 4: Elite Model
```
Prediction: 75% confidence
Actual: 80% accuracy
Odds: 1.28 (4% margin)
ROI: +8.0%
```

**Interpretation**: Excellent calibration + finding value. This is sustainable long-term profit.

## Market Comparison

### Why Different Margins Matter

**O/U 2.5 (4% margin)**:
- Most liquid market
- Tight odds
- Hard to beat but lower cost
- Break-even at ~73% for 70% prediction

**BTTS (8% margin)**:
- Higher margins
- More expensive to bet
- Need bigger edge to profit
- Break-even at ~75.6% for 70% prediction

**1X2 (6% margin)**:
- Three-way market complexity
- Moderate margins
- Harder to predict, moderate cost
- Break-even at ~74.2% for 70% prediction

## How to Use This

### 1. Check Accuracy vs Confidence

Compare backtest accuracy to your predicted probabilities:
- **Accuracy ≈ Confidence**: Well-calibrated, expect negative ROI
- **Accuracy > Confidence**: Finding value, potential profit
- **Accuracy < Confidence**: Overconfident, expect losses

### 2. Identify Best Markets

Look for markets where:
- **ROI is least negative** (closer to 0%)
- **Accuracy exceeds predictions** most
- **Low margin markets** (O/U 2.5, O/U 1.5)

### 3. Focus on High Confidence

Higher confidence = lower odds = smaller margin impact:

**Example: 85% prediction with 4% margin**:
- Fair Odds = 1.18
- Bookmaker Odds = 1.13
- Break-even = 88.4%
- If you hit 88%+ accuracy → **profitable!**

## Real-World Context

### Conservative = Realistic

These margins are **typical averages**. Some bookmakers:
- **Better odds**: 2-3% margins (Pinnacle, exchanges)
- **Worse odds**: 8-12% margins (recreational books)

Our estimates assume **middle-ground bookmakers** - realistic for most bettors.

### What Positive ROI Means

If backtest shows **+2% ROI** after margins:
- ✅ You're finding value
- ✅ Model beats bookmaker prices
- ✅ Sustainable long-term edge
- ✅ Consider increasing stakes on this market

If backtest shows **-2% ROI**:
- ⚠️ Well-calibrated but no edge
- ⚠️ Break-even at best (with perfect execution)
- ⚠️ Bookmaker margin eats all profit
- ⚠️ Need to improve accuracy or find better odds

## Adjusting Margins

To test different bookmaker environments, edit [backtest.py:346-355](backtest.py#L346-L355):

```python
market_margins = {
    '1X2': 0.06,      # Change to 0.03 for Pinnacle, 0.10 for bad books
    'BTTS': 0.08,     # Adjust based on your bookmaker
    'OU_2_5': 0.04,   # Premium bookmakers: 0.02, recreational: 0.08
    # etc.
}
```

**Recommendation**: Use conservative (higher) margins for planning. Better to be pleasantly surprised than disappointed!

---

## Summary

**Conservative ROI shows realistic betting expectations**:
- Accounts for bookmaker margins
- Different margins per market type
- Negative ROI is normal for well-calibrated models
- Positive ROI indicates real edge
- Use this to set realistic profit expectations

**Key Takeaway**: You need to beat your own predictions by enough to overcome bookmaker margins. A well-calibrated model alone isn't enough - you need to find value!
