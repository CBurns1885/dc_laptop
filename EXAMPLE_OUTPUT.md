# Example Output - New Streamlined Files

## 1. High Confidence Bets (90%+)

**File**: `outputs/high_confidence_bets.html`

### Summary Section
```
⭐ High Confidence Bets (90%+ Confidence)

Summary
  Total Bets: 22
  Confidence Range: 90.1% - 96.5%
  Markets: BTTS, OU_1_5, OU_3_5, OU_4_5, OU_2_5
  Leagues: E0, SP1, D1, F1, I1
```

### Bets Table
| Date    | League | Match                   | Market  | Prediction | Confidence | Probabilities            |
|---------|--------|-------------------------|---------|------------|------------|--------------------------|
| Dec 14  | E0     | Arsenal vs Chelsea      | BTTS    | Yes        | **96.5%**  | Yes: 96.5%, No: 3.5%     |
| Dec 14  | SP1    | Real vs Barca           | OU_1_5  | Over       | **95.2%**  | Over: 95.2%, Under: 4.8% |
| Dec 15  | D1     | Bayern vs Dortmund      | OU_3_5  | Over       | **94.8%**  | Over: 94.8%, Under: 5.2% |
| Dec 14  | E0     | Man City vs Liverpool   | BTTS    | Yes        | **93.7%**  | Yes: 93.7%, No: 6.3%     |
| Dec 15  | F1     | PSG vs Monaco           | OU_1_5  | Over       | **92.9%**  | Over: 92.9%, Under: 7.1% |
| Dec 15  | I1     | Inter vs Juventus       | OU_4_5  | Under      | **91.4%**  | Over: 8.6%, Under: 91.4% |
| Dec 14  | E0     | Tottenham vs Brighton   | OU_1_5  | Over       | **90.8%**  | Over: 90.8%, Under: 9.2% |
| Dec 15  | E0     | Newcastle vs Man Utd    | BTTS    | Yes        | **90.1%**  | Yes: 90.1%, No: 9.9%     |

### Color Guide
- 🟢 **Green rows**: 95%+ confidence (elite)
- 🟡 **Yellow rows**: 90-94% confidence (high)

---

## 2. O/U Accumulators (2-4 Fold)

**File**: `outputs/ou_accumulators.html`

### Summary Section
```
📊 O/U Accumulators (2-4 Fold)

Summary
  Total Accumulators: 50
  Markets: O/U 1.5, O/U 3.5, O/U 4.5
  Max Fold: 4-fold
  Strategy: Highest probability combinations (most likely to win first)
```

### Top Accumulators

#### #1: 4-Fold @8.20 (Combined: 75.3%)
```
Combined Probability: 75.3% | Min Confidence: 90.8%
Legs: 4

Legs:
  OU_1_5 Over - Arsenal vs Chelsea (96.5%)
  OU_3_5 Over - Bayern vs Dortmund (94.8%)
  OU_1_5 Over - PSG vs Monaco (92.9%)
  OU_1_5 Over - Tottenham vs Brighton (90.8%)
```

**Odds**: 8.20
**Return on £10 stake**: £82.00

---

#### #2: 3-Fold @5.10 (Combined: 79.8%)
```
Combined Probability: 79.8% | Min Confidence: 92.9%
Legs: 3

Legs:
  OU_1_5 Over - Real vs Barca (95.2%)
  OU_3_5 Over - Bayern vs Dortmund (94.8%)
  OU_1_5 Over - PSG vs Monaco (92.9%)
```

**Odds**: 5.10
**Return on £10 stake**: £51.00

---

#### #3: 4-Fold @9.50 (Combined: 71.2%)
```
Combined Probability: 71.2% | Min Confidence: 88.3%
Legs: 4

Legs:
  OU_1_5 Over - Arsenal vs Chelsea (96.5%)
  OU_3_5 Over - Bayern vs Dortmund (94.8%)
  OU_4_5 Under - Inter vs Juventus (91.4%)
  OU_1_5 Over - Man City vs Liverpool (88.3%)
```

**Odds**: 9.50
**Return on £10 stake**: £95.00

---

#### #4: 2-Fold @2.80 (Combined: 91.5%)
```
Combined Probability: 91.5% | Min Confidence: 95.2%
Legs: 2

Legs:
  BTTS Yes - Arsenal vs Chelsea (96.5%)
  OU_1_5 Over - Real vs Barca (95.2%)
```

**Odds**: 2.80
**Return on £10 stake**: £28.00

---

#### #5: 3-Fold @4.85 (Combined: 81.3%)
```
Combined Probability: 81.3% | Min Confidence: 91.4%
Legs: 3

Legs:
  OU_1_5 Over - Arsenal vs Chelsea (96.5%)
  OU_3_5 Over - Bayern vs Dortmund (94.8%)
  OU_4_5 Under - Inter vs Juventus (91.4%)
```

**Odds**: 4.85
**Return on £10 stake**: £48.50

---

### Probability Color Coding
- 🟢 **Green**: 70%+ combined probability (high chance)
- 🟡 **Yellow**: 50-70% combined probability (medium)
- 🔴 **Red**: <50% combined probability (risky, higher odds)

### Tips Section
```
💡 Tips
  • Green probability (70%+): High chance of winning
  • Yellow probability (50-70%): Medium risk
  • Red probability (<50%): Higher risk, higher reward
  • Accumulators are sorted by combined probability (most likely to win first)
  • Min confidence shows the weakest leg
```

---

## File Sizes (Typical)

```
outputs/
├── high_confidence_bets.html     ~15 KB (nicely formatted)
├── high_confidence_bets.csv      ~3 KB  (import to Excel)
├── ou_accumulators.html          ~45 KB (50 accumulators)
└── ou_accumulators.csv           ~12 KB (import to Excel)
```

---

## How to Use

### Daily Workflow

1. **Open `high_confidence_bets.html`**
   - Look at green rows (95%+) first
   - These are your safest single bets
   - Bet straight on these

2. **Open `ou_accumulators.html`**
   - Start at #1 (highest probability)
   - Check if all legs make sense
   - Build accumulator or skip to next

3. **Done!**
   - No need to check 20 other files
   - Everything you need in 2 clicks

### Risk Levels

**Conservative Strategy**:
- High confidence bets: Only bet green rows (95%+)
- Accumulators: Top 10 (75%+ probability)

**Balanced Strategy**:
- High confidence bets: Bet all rows (90%+)
- Accumulators: Top 25 (65%+ probability)

**Aggressive Strategy**:
- High confidence bets: Bet all rows, increase stakes on green
- Accumulators: Top 50 (50%+ probability, higher odds)

---

## CSV Format

Both files have corresponding CSV files for importing to Excel/Google Sheets:

### high_confidence_bets.csv
```csv
Date,League,HomeTeam,AwayTeam,Market,Prediction,Confidence,P_Yes,P_No
2024-12-14,E0,Arsenal,Chelsea,BTTS,Yes,96.5,96.5,3.5
2024-12-14,SP1,Real Madrid,Barcelona,OU_1_5,Over,95.2,95.2,4.8
...
```

### ou_accumulators.csv
```csv
Fold,Combined_Confidence_%,Combined_Probability_%,Implied_Odds,Legs,Selections
4-Fold,90.8,75.3,8.20,4,"OU_1_5 Over (Arsenal vs Chelsea) | OU_3_5 Over (Bayern vs Dortmund) | ..."
3-Fold,92.9,79.8,5.10,3,"OU_1_5 Over (Real vs Barca) | OU_3_5 Over (Bayern vs Dortmund) | ..."
...
```

---

## Comparison: Old vs New

### OLD System (Messy)
```
outputs/
├── weekly_bets.csv
├── weekly_bets_lite.csv
├── top50_weighted.html
├── top50_weighted.csv
├── ou_analysis.html
├── ou_analysis.csv
├── accumulators_safe.html
├── accumulators_mixed.html
├── accumulators_aggressive.html
├── quality_bets.html
├── quality_bets.csv
└── 16 market-specific files...

Total: 26+ files
Time to review: 15-20 minutes
```

### NEW System (Clean)
```
outputs/
├── high_confidence_bets.html    ← Start here!
├── high_confidence_bets.csv
├── ou_accumulators.html         ← Then check this
├── ou_accumulators.csv
└── 16 market-specific files (optional)

Total: 4 main files (+ 16 optional)
Time to review: 2-3 minutes
```

---

## Real-World Example

**Saturday morning, you have 10 minutes before kickoff**

### OLD way:
1. Open top50_weighted.html
2. Open ou_analysis.html
3. Open accumulators_safe.html
4. Compare across files
5. Manually build your accumulator
6. Check quality_bets.html for other ideas
7. **Result**: Confused, missed kickoff

### NEW way:
1. Open high_confidence_bets.html → Bet top 3 straight
2. Open ou_accumulators.html → Pick #1 accumulator
3. **Result**: Done in 2 minutes, placed bets

---

**This is what you get when you run the new system!**
