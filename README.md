# ⚽ Football Prediction System - Clean Version

## 🚀 QUICK START - ONE CLICK RUN

### First Time Setup:
1. **Install Python 3.8+** (if not already installed)
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the system:**
   ```bash
   python run_weeklyOU.py
   ```

### Weekly Usage:
1. **Run once per week** (preferably Monday morning)
2. **Select tuning mode** when prompted:
   - Option 4 (No tuning) - Fastest, ~15 min [RECOMMENDED]
   - Option 3 (Fast mode) - ~30 min
   - Option 2 (Quick tuning) - ~1 hour
   - Option 1 (Full tuning) - ~2+ hours
3. **Wait for completion**
4. **Check outputs/** folder for results

### After Matches Played:
1. Run `python update_results.py` to update accuracy tracking
2. This improves next week's weighted predictions

---

## 📊 OUTPUT FILES

The system generates these files in the `outputs/` folder:

### Main Predictions:
- **weekly_bets.csv** - All predictions with probabilities
- **top50_weighted.html** - Top 50 picks (weighted by historical accuracy)
- **top50_weighted.csv** - CSV version of top picks

### Specialized Analysis:
- **ou_analysis.html** - Over/Under analysis and high-confidence picks
- **accumulators_safe.html** - Conservative 4-fold accumulators
- **accumulators_mixed.html** - Balanced 5-fold accumulators  
- **accumulators_aggressive.html** - High-risk 6-fold accumulators

### Tracking:
- **accuracy_database.db** - SQLite database tracking all predictions and results
- **accuracy_report.csv** - Model accuracy summary by market

---

## 📁 FILE DESCRIPTIONS

### Core Runner:
- **run_weeklyOU.py** - MAIN weekly runner (14-step pipeline)

### Data Pipeline:
- **download_football_data.py** - Downloads historical match data
- **simple_fixture_downloader.py** - Downloads upcoming fixtures
- **data_ingest.py** - Builds historical database
- **ingest_local_run.py** - Validates data integrity
- **prep_fixtures.py** - Converts XLSX → CSV

### Feature Engineering:
- **features.py** - Creates 50-80 predictive features per match
- **generate_and_load_stats.py** - Team/referee statistics

### Machine Learning:
- **models.py** - 11 model types (Random Forest, XGBoost, LightGBM, etc.)
- **incremental_trainer.py** - Smart training (only retrains when needed)
- **train_evaluate.py** - Training and evaluation logic

### Prediction & Analysis:
- **predict.py** - Generates predictions for all markets
- **accuracy_tracker.py** - Logs predictions to database
- **weighted_top50.py** - Creates weighted HTML reports
- **ou_analyzer.py** - Over/Under analysis
- **acc_builder.py** - Accumulator builder

### Utilities:
- **config.py** - Configuration and settings
- **progress_utils.py** - Progress tracking
- **email_sender.py** - Email results (optional)
- **update_results.py** - Update accuracy after matches

---

## 🎯 MARKETS COVERED

The system predicts probabilities for:

1. **1X2** (Home Win / Draw / Away Win)
2. **BTTS** (Both Teams To Score - Yes/No)
3. **O/U 1.5** (Over/Under 1.5 goals)
4. **O/U 2.5** (Over/Under 2.5 goals)
5. **O/U 3.5** (Over/Under 3.5 goals)
6. **O/U 4.5** (Over/Under 4.5 goals)

---

## 🔧 TROUBLESHOOTING

### "No fixtures file found"
- Download fixtures manually from football-data.co.uk/matches.php
- Save as `upcoming_fixtures.csv` in project folder

### "Module not found"
- Run: `pip install -r requirements.txt`

### Slow first run
- First run downloads historical data (2017-present)
- Subsequent runs are much faster (~15 min)

### Want faster runs?
- Use Option 4 (No tuning) when prompted
- Models will load from disk if already trained

---

## 📚 DOCUMENTATION

- **PROJECT_ANALYSIS.md** - Complete system analysis
- **ALL_PHASES_COMPLETE.docx** - Detailed technical documentation
- **system_overview.pdf** - System architecture overview

---

## ⚡ PERFORMANCE TIPS

### For Speed:
1. Select "No tuning" option (Option 4)
2. Reduce training years in code: `TRAINING_START_YEAR = 2023`
3. Keep trained models (they reload automatically)

### For Accuracy:
1. Run weekly and update results with `update_results.py`
2. Use more tuning (Options 1-2)
3. Keep more historical data (2017+)

---

## 📧 EMAIL CONFIGURATION (Optional)

To receive results by email, edit these in `run_weeklyOU.py`:
```python
os.environ["EMAIL_SMTP_SERVER"] = "smtp-mail.outlook.com"
os.environ["EMAIL_SMTP_PORT"] = "587"
os.environ["EMAIL_SENDER"] = "your_email@example.com"
os.environ["EMAIL_PASSWORD"] = "your_password"
os.environ["EMAIL_RECIPIENT"] = "recipient@example.com"
```

---

## ✅ CHANGELOG

### v1.0 - Clean Release
- Fixed accumulator import bug in run_weeklyOU.py
- Removed 25+ redundant/duplicate files
- Streamlined to 21 essential files
- Added comprehensive documentation
- Ready for one-click weekly operation

---

## 🤝 SUPPORT

If you encounter issues:
1. Check the troubleshooting section above
2. Review PROJECT_ANALYSIS.md for detailed flow
3. Check ALL_PHASES_COMPLETE.docx for technical details

---

## 📝 LICENSE

This is a personal betting prediction system. Use at your own risk.
Gambling can be addictive - please bet responsibly.

---

**Last Updated:** October 2025  
**Version:** 1.0 Clean Release
