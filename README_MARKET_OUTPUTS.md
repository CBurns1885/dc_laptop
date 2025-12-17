# Market-Specific Output Files

## Overview

The prediction system now generates **one file per market type**, making it much easier to find the predictions you care about.

## New Output Structure

Instead of one giant CSV with all markets mixed together, you now get:

### Market Files (Auto-generated)

```
outputs/
├── predictions_1x2.html          ← Match results (Home/Draw/Away)
├── predictions_1x2.csv
├── predictions_btts.html         ← Both Teams To Score
├── predictions_btts.csv
├── predictions_ou_0_5.html       ← Over/Under 0.5 goals
├── predictions_ou_0_5.csv
├── predictions_ou_1_5.html       ← Over/Under 1.5 goals
├── predictions_ou_1_5.csv
├── predictions_ou_2_5.html       ← Over/Under 2.5 goals ⭐ Most popular
├── predictions_ou_2_5.csv
├── predictions_ou_3_5.html       ← Over/Under 3.5 goals
├── predictions_ou_3_5.csv
├── predictions_ou_4_5.html       ← Over/Under 4.5 goals
├── predictions_ou_4_5.csv
├── predictions_ou_5_5.html       ← Over/Under 5.5 goals
└── predictions_ou_5_5.csv
```

## What Each File Contains

### HTML Files
Beautiful, easy-to-read reports with:
- **Summary stats** (total matches, high confidence predictions)
- **Color-coded confidence levels**:
  - 🟢 Green (≥85%) - Elite confidence
  - 🔵 Blue (≥75%) - High confidence
  - 🟠 Orange (≥65%) - Medium confidence
  - 🔴 Red (<65%) - Low confidence
- **Sortable tables** by date, league, confidence
- **All probabilities displayed** for each outcome

### CSV Files
Clean data for spreadsheet analysis:
- Date, League, HomeTeam, AwayTeam
- Prediction (e.g., "Over 2.5", "Home Win", "Yes")
- Confidence % (max probability)
- Individual probabilities for each outcome

## How It Works

### Automatic Generation

When you run `run_weekly.py`, Step 9 splits the predictions:

```
STEP 9/19 (47%): SPLIT BY MARKET
=====================================================
✓ Loaded 150 predictions from weekly_bets_lite.csv
✓ Split into 8 markets
✓ Created 16 files:
  • predictions_1x2.csv
  • predictions_1x2.html
  • predictions_btts.csv
  • predictions_btts.html
  ...
```

### Manual Generation

Run anytime:
```bash
python market_splitter.py
```

This reads `outputs/weekly_bets_lite.csv` and creates market-specific files.

## File Format Examples

### 1X2 (Match Result)
```csv
Date,League,HomeTeam,AwayTeam,Prediction,Prediction_Text,Confidence_%,P_1X2_H,P_1X2_D,P_1X2_A
2025-11-30,E0,Arsenal,Chelsea,H,Home Win,78.3,0.783,0.167,0.050
```

### BTTS (Both Teams To Score)
```csv
Date,League,HomeTeam,AwayTeam,Prediction,Prediction_Text,Confidence_%,P_BTTS_Y,P_BTTS_N
2025-11-30,E0,Man City,Liverpool,Y,Yes (Both Score),82.1,0.821,0.179
```

### O/U 2.5 Goals
```csv
Date,League,HomeTeam,AwayTeam,Prediction,Prediction_Text,Confidence_%,P_OU_2_5_O,P_OU_2_5_U
2025-11-30,SP1,Real Madrid,Barcelona,O,Over 2.5,75.4,0.754,0.246
```

## Sorting & Filtering

### Default Sort Order
1. **Date** (earliest first)
2. **League** (alphabetical)
3. **Confidence** (highest first within each league/date)

This means:
- Saturday's Premier League matches appear first
- Within Premier League, highest confidence bets are at the top
- Then Sunday's matches, etc.

### Finding High-Value Bets

**Elite confidence (≥85%)**
```python
import pandas as pd
df = pd.read_csv('outputs/predictions_ou_2_5.csv')
elite = df[df['Confidence_%'] >= 85]
```

**Premier League only**
```python
epl = df[df['League'] == 'E0']
```

**Weekend matches**
```python
df['Date'] = pd.to_datetime(df['Date'])
weekend = df[df['Date'].dt.dayofweek.isin([5, 6])]  # Sat=5, Sun=6
```

## HTML Features

### Visual Highlights
- **League tags** - Color-coded by league
- **Confidence badges** - Green/Blue/Orange/Red
- **Prediction highlighting** - Bold, blue text
- **Hover effects** - Row highlights on mouseover

### Summary Cards
Each HTML file shows:
- Total matches for that market
- Number of elite predictions (≥85%)
- Number of high-confidence predictions (≥75%)
- Number of leagues covered

## Integration with Existing Workflow

### Before (Old Way)
```
1. Open weekly_bets_lite.csv
2. Scroll through all 150+ matches
3. Filter by market columns manually
4. Sort by confidence
5. Find the bets you want
```

### After (New Way)
```
1. Open predictions_ou_2_5.html
2. See only O/U 2.5 predictions
3. Already sorted by date/league/confidence
4. Click to check details
```

## Comparison with Legacy Files

| File | Purpose | Status |
|------|---------|--------|
| `weekly_bets_lite.csv` | Master file (all markets) | ✅ Still generated |
| `ou_analysis.html` | O/U specific analyzer | ✅ Still generated (different purpose) |
| `top50_weighted.html` | Best 50 picks overall | ✅ Still generated |
| `predictions_*.html` | Per-market organized view | ✅ NEW! |

The new files **complement** existing files, not replace them.

## Use Cases

### Scenario 1: "I only bet O/U 2.5"
```
→ Open predictions_ou_2_5.html
→ See all O/U 2.5 predictions
→ Sorted by date and confidence
```

### Scenario 2: "What's the best 1X2 bet this weekend?"
```
→ Open predictions_1x2.html
→ Look at confidence badges (green = elite)
→ Check top matches for Saturday/Sunday
```

### Scenario 3: "BTTS bets for Premier League"
```
→ Open predictions_btts.html
→ Filter to League = E0
→ Check Yes predictions with high confidence
```

### Scenario 4: "I want to analyze all markets"
```
→ Use weekly_bets_lite.csv (master file)
→ Or loop through predictions_*.csv programmatically
```

## Technical Details

### How Splitting Works

The `MarketSplitter` class:
1. Loads `weekly_bets_lite.csv`
2. Groups by market type (1X2, BTTS, O/U lines)
3. Extracts relevant probability columns
4. Adds prediction and confidence columns
5. Sorts by date → league → confidence
6. Generates CSV + HTML for each market

### Customization

Edit [market_splitter.py:48-79](market_splitter.py#L48-L79) to:
- Add new markets
- Change sort order
- Modify HTML styling
- Adjust confidence thresholds

### Performance

- **Processing time**: ~2 seconds for 150 matches
- **File sizes**: ~10-50 KB per CSV, ~50-200 KB per HTML
- **No impact** on prediction generation (happens after)

## Troubleshooting

### "No market files generated"
- Check `outputs/weekly_bets_lite.csv` exists
- Verify it has probability columns (P_1X2_H, P_BTTS_Y, etc.)
- Run `python market_splitter.py` manually to see errors

### "Market missing from output"
- That market might not have predictions this week
- Check column names in `weekly_bets_lite.csv`
- Verify market config in `market_splitter.py`

### "Confidence always 100%"
- This means all probabilities for that match are missing
- Check prediction generation step
- Look for NaN values in probability columns

## Future Enhancements

Potential additions:
- ✅ League-specific files (e.g., `predictions_epl_ou_2_5.html`)
- ✅ Confidence-filtered files (e.g., `predictions_elite_only.html`)
- ✅ Time-based splits (e.g., `predictions_saturday.html`)
- ✅ Accumulator suggestions per market

Vote for features in the GitHub issues!

## Related Files

- **[market_splitter.py](market_splitter.py)** - Main splitting logic
- **[run_weekly.py:400-411](run_weekly.py#L400-L411)** - Integration into pipeline
- **[ou_analyzer.py](ou_analyzer.py)** - Similar concept, O/U specific

---

**Summary**: One HTML + CSV file per market type, sorted by date/league/confidence, making it super easy to find the bets you care about! 🎯
