# ⚽ Football Prediction System - Project Analysis

## 📌 MAIN RUNNER: `run_weeklyOU.py`

This is your **ONE-CLICK weekly runner** that orchestrates the entire pipeline.

---

## 🔄 COMPLETE WORKFLOW (14 Steps)

### Pipeline Flow:
```
run_weeklyOU.py (MAIN)
├── Step 0: Download Fixtures → simple_fixture_downloader.py
├── Step 1: Download Historical Data → download_football_data.py
├── Step 2: Build Database → data_ingest.py
├── Step 3: Validate Data → ingest_local_run.py
├── Step 4: Generate Statistics → generate_and_load_stats.py
├── Step 5: Build Features → features.py
├── Step 6: Train/Load Models → incremental_trainer.py → models.py
├── Step 7: Prepare Fixtures → prep_fixtures.py
├── Step 8: Generate Predictions → predict.py
├── Step 9: Log Predictions → accuracy_tracker.py
├── Step 10: Weighted Top 50 → weighted_top50.py
├── Step 11: O/U Analysis → ou_analyzer.py
├── Step 12: Build Accumulators → (MISSING - should be acc_builder.py)
├── Step 13: Email Results → email_sender.py
└── Step 14: Open Outputs Folder
```

---

## ✅ ESSENTIAL FILES (KEEP - CURRENTLY USED)

### Core Runner
- **run_weeklyOU.py** - MAIN weekly runner (ONE-CLICK)

### Configuration
- **config.py** - Paths, settings, logging utilities

### Data Pipeline
- **download_football_data.py** - Downloads historical CSVs from football-data.co.uk
- **simple_fixture_downloader.py** - Downloads upcoming fixtures
- **data_ingest.py** - Builds historical_results.parquet database
- **ingest_local_run.py** - Validates local CSV data
- **prep_fixtures.py** - Converts XLSX fixtures to CSV

### Feature Engineering
- **features.py** - Builds 50-80 features per match (form, H2H, stats, etc.)
- **generate_and_load_stats.py** - Generates team/referee statistics

### Machine Learning
- **models.py** - Defines all 11 model types (RF, XGB, LGBM, etc.)
- **incremental_trainer.py** - Smart training (only retrains if needed)
- **train_evaluate.py** - Training and evaluation logic

### Prediction & Output
- **predict.py** - Generates predictions for upcoming fixtures
- **accuracy_tracker.py** - Logs predictions to SQLite database
- **weighted_top50.py** - Creates weighted Top 50 HTML report
- **ou_analyzer.py** - Over/Under analysis and HTML report

### Communication
- **email_sender.py** - Emails results to user

### Utilities
- **progress_utils.py** - Timer and heartbeat utilities

### Requirements
- **requirements.txt** - Python package dependencies

---

## ⚠️ BROKEN/INCOMPLETE FILES (FIX REQUIRED)

### Missing Import in run_weeklyOU.py
- **Step 12** imports `accumulator_builder` but file doesn't exist
- **Solution**: Should import from `acc_builder.py` OR `accumulator_finder.py`
- **Line 287**: `from accumulator_builder import AccumulatorFinder`
- **Line 291**: Uses `AccumulatorBuilder` (wrong class name)

### Recommendation:
```python
# Fix Step 12 in run_weeklyOU.py:
from acc_builder import AccumulatorFinder  # or accumulator_finder
builder = AccumulatorFinder(str(csv_path))  # Fix class name
```

---

## 🗑️ UNUSED/REDUNDANT FILES (CAN DELETE)

### Duplicate/Old Versions
- **updated_run_weekly.py** - Old version of runner
- **updated_run_weekly__1_.py** - Another old version
- **run_pipeline.py** - Alternative runner (not used)
- **ONE_CLICK_RUN.py** - Duplicate/alternative runner
- **modelsOLD.py** - Old model definitions
- **load_existing_stats__1_.py** - Duplicate with `__1_` suffix
- **weighted_html_output__1_.py** - Duplicate with `__1_` suffix

### Specialized/Optional Files (Not in main weekly flow)
- **realistic_backtest.py** - Backtesting (separate tool)
- **backtest.py** - Backtesting (separate tool)
- **backtest_config.py** - Backtesting configuration
- **backtest_engine.py** - Backtesting engine
- **backtest_error_handling.py** - Backtesting utilities
- **backtest_visualizer.py** - Backtesting visualization
- **enhanced_predict.py** - Enhanced prediction (not used in weekly)
- **dc_predict.py** - Dixon-Coles prediction only
- **models_dc.py** - Dixon-Coles models only
- **update_results.py** - Manual results update (run separately after matches)

### Utility/Experimental Files
- **auto_fixture_downloader.py** - Alternative fixture downloader
- **api_football_weekly_fixtures.py** - API-based fixture download (not used)
- **combined_fixture_downloader.py** - Combined downloader (not used)
- **cup_fixtures_addon_FINAL.py** - Cup fixtures addon (not in main flow)
- **combine_cross_league.py** - Cross-league analysis
- **btts_ou_optimizer.py** - BTTS/O/U optimizer
- **blending.py** - Model blending experiments
- **ensemble_blender.py** - Ensemble blending experiments
- **model_enhancer.py** - Model enhancement experiments
- **model_binary.py** - Binary classification models
- **model_multiclass.py** - Multiclass models
- **model_ordinal.py** - Ordinal regression models
- **ordinal.py** - Ordinal utilities
- **calibration.py** - Calibration utilities
- **tuning.py** - Hyperparameter tuning
- **features_patch.py** - Feature patching
- **load_existing_stats.py** - Load stats (duplicate)
- **setup_structure.py** - One-time setup script

### Jupyter Notebooks (Development/Testing)
- **Notebook.ipynb**
- **Untitled-1.ipynb**
- **football_end_to_end.ipynb**
- **europe_cup_test.ipynb**

### Trivial/Empty Files
- **pip_install_numpy_pandas_scikit-learn_xg.py** - Just pip install command

---

## 📊 DATA FILES (KEEP)

### Input Data
- **upcoming_fixtures.csv** - Current week's fixtures
- **upcoming_fixtures.xlsx** - Current week's fixtures (Excel)

### Output Data
- **weekly_bets.csv** - All predictions for the week

---

## 📄 DOCUMENTATION (KEEP)

- **ALL_PHASES_COMPLETE.docx** - Complete system documentation
- **system_overview.pdf** - System overview
- **VALIDATION_CHECKLIST_md.pdf** - Validation checklist
- **footballdata_co_uk_notes_txt.pdf** - Data source notes
- **📁____Football_Prediction_System.docx** - Additional documentation

---

## 🎯 FILES FOR ONE-CLICK WEEKLY RUN

### Absolute Minimum (17 files):
```
1.  run_weeklyOU.py              ← MAIN RUNNER
2.  config.py
3.  download_football_data.py
4.  simple_fixture_downloader.py
5.  data_ingest.py
6.  ingest_local_run.py
7.  features.py
8.  generate_and_load_stats.py
9.  models.py
10. incremental_trainer.py
11. train_evaluate.py
12. prep_fixtures.py
13. predict.py
14. accuracy_tracker.py
15. weighted_top50.py
16. ou_analyzer.py
17. progress_utils.py
```

### Recommended (Add these for full functionality):
```
18. email_sender.py              ← Email results
19. acc_builder.py               ← Accumulators (FIX import in run_weeklyOU.py)
20. requirements.txt             ← Dependencies
```

### Optional Extras:
```
21. update_results.py            ← Run after matches to update accuracy
22. backtest*.py files           ← For backtesting (separate from weekly run)
```

---

## 🐛 CRITICAL FIX NEEDED

### In `run_weeklyOU.py` - Step 12 (Lines 285-310):

**Current (BROKEN):**
```python
from accumulator_builder import AccumulatorFinder  # ❌ File doesn't exist
builder = AccumulatorBuilder(str(csv_path))        # ❌ Wrong class name
```

**Fix Option 1 (Use acc_builder.py):**
```python
from acc_builder import AccumulatorFinder  # ✅
builder = AccumulatorFinder(str(csv_path)) # ✅
```

**Fix Option 2 (Use accumulator_finder.py):**
```python
from accumulator_finder import AccumulatorFinder  # ✅
builder = AccumulatorFinder(str(csv_path))        # ✅
```

---

## 📦 RECOMMENDED FILE STRUCTURE

### For Clean Laptop Version:
```
football_prediction/
├── run_weeklyOU.py                 ← ONE-CLICK RUN THIS
├── config.py
├── requirements.txt
│
├── core/                           ← Core pipeline files
│   ├── download_football_data.py
│   ├── simple_fixture_downloader.py
│   ├── data_ingest.py
│   ├── ingest_local_run.py
│   ├── prep_fixtures.py
│   ├── features.py
│   ├── generate_and_load_stats.py
│   ├── models.py
│   ├── incremental_trainer.py
│   ├── train_evaluate.py
│   ├── predict.py
│   └── progress_utils.py
│
├── analysis/                       ← Output generators
│   ├── accuracy_tracker.py
│   ├── weighted_top50.py
│   ├── ou_analyzer.py
│   └── acc_builder.py
│
├── utils/                          ← Utilities
│   └── email_sender.py
│
├── models/                         ← Trained models (lite versions)
│   ├── y_1X2.pkl
│   ├── y_BTTS.pkl
│   └── y_OU_2_5.pkl
│
├── data/                           ← Generated data
│   ├── historical_results.parquet
│   └── features.pkl
│
├── outputs/                        ← Weekly outputs
│   ├── weekly_bets.csv
│   ├── top50_weighted.html
│   ├── ou_analysis.html
│   └── accuracy_database.db
│
└── docs/                           ← Documentation
    ├── ALL_PHASES_COMPLETE.docx
    └── system_overview.pdf
```

---

## 🚀 USAGE - ONE CLICK RUN

### Weekly Workflow:
1. **Double-click** `run_weeklyOU.py` (or run from terminal)
2. **Select tuning mode** (recommend option 4: No tuning - fastest)
3. **Wait 15-60 minutes** (depending on mode)
4. **Check outputs/** folder for results:
   - `weekly_bets.csv` - All predictions
   - `top50_weighted.html` - Best picks
   - `ou_analysis.html` - O/U opportunities
   - `accumulators_*.html` - Accumulator suggestions

### After Matches:
1. **Run** `update_results.py` to update accuracy database
2. **Next week**: Run `run_weeklyOU.py` again (will use updated weights)

---

## ⚡ OPTIMIZATION RECOMMENDATIONS

### For Faster Runs:
1. **Use trained models**: Set `OPTUNA_TRIALS=0` (no tuning)
2. **Include lite models**: Add pre-trained `.pkl` files to models/ folder
3. **Reduce training years**: Set `TRAINING_START_YEAR=2023` (not 2017)

### For Better Accuracy:
1. **More tuning**: Set `OPTUNA_TRIALS=50`
2. **More training data**: Set `TRAINING_START_YEAR=2017`
3. **Update weekly**: Run `update_results.py` after each week

---

## 🎯 SUMMARY

### Files to DEFINITELY DELETE (25 files):
- updated_run_weekly.py
- updated_run_weekly__1_.py
- modelsOLD.py
- load_existing_stats__1_.py
- weighted_html_output__1_.py
- ONE_CLICK_RUN.py (duplicate runner)
- run_pipeline.py (alternative runner)
- All 4 .ipynb notebooks
- All backtest*.py files (6 files) - unless you want backtesting
- enhanced_predict.py, dc_predict.py, models_dc.py
- auto_fixture_downloader.py, api_football_weekly_fixtures.py
- combined_fixture_downloader.py, cup_fixtures_addon_FINAL.py
- combine_cross_league.py, btts_ou_optimizer.py
- blending.py, ensemble_blender.py, model_enhancer.py
- model_binary.py, model_multiclass.py, model_ordinal.py
- ordinal.py, calibration.py, tuning.py
- features_patch.py, setup_structure.py
- pip_install_numpy_pandas_scikit-learn_xg.py

### Files to KEEP (20 files):
- run_weeklyOU.py ← MAIN
- config.py, requirements.txt
- Core pipeline (14 files)
- Analysis outputs (4 files)
- email_sender.py
- Optional: update_results.py

### Files to FIX:
- run_weeklyOU.py - Line 287 & 291 (accumulator import)

---

## ✅ READY FOR ONE-CLICK OPERATION

Once you:
1. **Fix** the accumulator import in run_weeklyOU.py
2. **Delete** the 25 redundant files
3. **Add** your pre-trained models to models/ folder
4. **Test** one run to ensure everything works

You'll have a clean, production-ready system that runs with ONE CLICK each week!
