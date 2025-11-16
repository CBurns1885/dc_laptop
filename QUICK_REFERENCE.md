# ⚡ QUICK REFERENCE CARD

## 🎯 ONE-LINE SUMMARY
Run `python run_weeklyOU.py` weekly → Wait 15-60 min → Check `outputs/` folder → Profit! 🚀

---

## 📅 WEEKLY WORKFLOW

### Monday Morning:
```bash
python run_weeklyOU.py
```
↓ Select Option 4 (No tuning - fastest)  
↓ Wait ~15 minutes  
↓ Check outputs/ folder  

### Files to Review:
1. **top50_weighted.html** ← Best individual picks
2. **ou_analysis.html** ← Over/Under opportunities  
3. **accumulators_*.html** ← Multi-leg options

### After Weekend Matches:
```bash
python update_results.py
```
↓ Updates accuracy database  
↓ Improves next week's weights  

---

## 🎰 BETTING STRATEGY

### Conservative (Safest):
- Check `top50_weighted.html` → Pick top 10
- Focus on probabilities > 70%
- Use `accumulators_safe.html` for 4-folds

### Moderate (Balanced):
- Top 20 from weighted list
- Probabilities > 65%
- Use `accumulators_mixed.html` for 5-folds

### Aggressive (Higher Risk):
- Top 30-50 picks
- Probabilities > 60%
- Use `accumulators_aggressive.html` for 6-folds

### Over/Under Specialist:
- Check `ou_analysis.html`
- Focus on Elite (85%+) picks
- Look for O/U 2.5 value

---

## 📊 UNDERSTANDING OUTPUTS

### weekly_bets.csv
- **ALL predictions** with probabilities
- Filter by probability for your risk level
- Markets: 1X2, BTTS, O/U 1.5/2.5/3.5/4.5

### top50_weighted.html  
- Top picks **weighted by historical accuracy**
- Pre-filtered for quality
- Sorted by expected value

### ou_analysis.html
- Over/Under specialist picks
- Categories: Elite (85%+), High (75%+), Medium (65%+)
- Shows best value O/U bets

### accumulators_*.html
- Pre-built multi-leg bets
- Safe = 4-fold (70%+ each leg)
- Mixed = 5-fold (65%+ each leg)
- Aggressive = 6-fold (60%+ each leg)

---

## 🔧 QUICK FIXES

### No fixtures found?
Download from football-data.co.uk/matches.php → Save as `upcoming_fixtures.csv`

### Module errors?
```bash
pip install -r requirements.txt
```

### Slow run?
Choose Option 4 when prompted (no tuning)

### Email not working?
Edit email settings in `run_weeklyOU.py` lines 18-24

---

## 📈 PERFORMANCE TRACKING

### Check Accuracy:
Open `outputs/accuracy_database.db` in SQLite viewer  
OR  
Check `outputs/accuracy_report.csv`

### Improve Results:
1. Run weekly consistently
2. Always run `update_results.py` after matches
3. System learns from past performance
4. Weights improve over time

---

## 🎯 BEST PRACTICES

### DO:
✅ Run every Monday morning  
✅ Update results after weekend  
✅ Bet responsibly within budget  
✅ Focus on weighted top picks  
✅ Track your own results  

### DON'T:
❌ Chase losses  
❌ Bet more than you can afford  
❌ Ignore probabilities  
❌ Skip result updates  
❌ Bet on all predictions  

---

## 💡 PRO TIPS

1. **Combine Markets**: Look for matches with high confidence in BOTH 1X2 AND BTTS
2. **Value Betting**: 65% probability at 2.00 odds = value bet
3. **Accumulators**: Mix safe + moderate picks for better odds
4. **Track ROI**: Record your bets to see what works
5. **Trust the Weights**: Historical accuracy matters!

---

## 📞 QUICK COMMANDS

| Task | Command |
|------|---------|
| Weekly run | `python run_weeklyOU.py` |
| Update results | `python update_results.py` |
| Check version | Check README.md |

---

## 🎓 PROBABILITY → ODDS GUIDE

| Probability | Implied Odds | Bookmaker Odds |
|-------------|--------------|----------------|
| 90% | 1.11 | 1.05-1.15 |
| 80% | 1.25 | 1.15-1.30 |
| 70% | 1.43 | 1.35-1.50 |
| 65% | 1.54 | 1.45-1.60 |
| 60% | 1.67 | 1.55-1.75 |
| 55% | 1.82 | 1.70-1.90 |
| 50% | 2.00 | 1.90-2.10 |

**Value Bet Example:**  
System says 70% (implied odds 1.43)  
Bookmaker offers 1.70  
→ VALUE BET! 🎯

---

## 🏆 SUCCESS METRICS

### Good Performance:
- 60%+ accuracy on 1X2
- 65%+ accuracy on BTTS
- 70%+ accuracy on O/U 2.5
- Positive ROI over 100+ bets

### Excellent Performance:
- 65%+ accuracy on 1X2
- 70%+ accuracy on BTTS  
- 75%+ accuracy on O/U 2.5
- 10%+ ROI sustained

---

## ⚠️ RESPONSIBLE GAMBLING

🎲 Only bet what you can afford to lose  
📊 Track all bets and outcomes  
🛑 Set weekly/monthly limits  
⏸️ Take breaks if losing  
🆘 Seek help if needed: BeGambleAware.org

---

**Remember:** This is a PREDICTION system, not a guarantee.  
Past performance ≠ Future results.  
Bet smart, stay disciplined, track results! 📈
