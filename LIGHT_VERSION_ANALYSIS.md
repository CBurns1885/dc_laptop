# Light Version Analysis - dc_laptop

## Requirements Status

All required packages are installed:
- numpy 2.3.2
- pandas 2.3.1
- scikit-learn 1.7.1
- scipy 1.16.1
- joblib 1.5.1
- xgboost 3.0.3
- lightgbm 4.6.0
- catboost 1.2.8
- optuna 4.5.0
- torch 2.8.0+cpu
- requests 2.32.4
- openpyxl 3.1.5
- matplotlib 3.10.5
- seaborn 0.13.2

## File Usage Analysis

### Summary
- **Total Python files**: 59
- **Essential files (used)**: 29
- **Unused files**: 30
- **Space saving potential**: ~50%

## ESSENTIAL FILES (Keep These - 29 files)

These are actively imported and used by the main workflows:

### Core Workflow (Main Entry Points)
1. run_weekly.py - Main weekly prediction pipeline
2. dc_predict.py - DC-specific predictions
3. backtest.py - Backtesting system
4. train_evaluate.py - Model training and evaluation

### Configuration & Setup
5. config.py - Central configuration

### Data Management
6. data_ingest.py - Historical data ingestion
7. ingest_local_run.py - Local CSV validation
8. download_football_data.py - Historical data downloader
9. simple_fixture_downloader.py - Fixture downloader
10. generate_and_load_stats.py - Statistics generation

### Feature Engineering
11. features.py - Feature building
12. days_since_match_fetcher.py - Days since last match enrichment

### Modeling
13. models.py - General models
14. models_dc.py - DC-specific models
15. ordinal.py - Ordinal regression
16. calibration.py - Model calibration
17. tuning.py - Hyperparameter tuning
18. incremental_trainer.py - Smart training/loading

### Prediction & Analysis
19. predict.py - Core prediction engine
20. prep_fixtures.py - Fixture preparation
21. market_splitter.py - Split predictions by market
22. btts_ou_optimizer.py - BTTS/O/U optimization
23. ensemble_blender.py - Ensemble blending
24. generate_outputs_from_actual.py - Output generation

### Tracking & Reporting
25. accuracy_tracker.py - Performance tracking
26. update_results.py - Result updates
27. excel_generator.py - Excel reports
28. email_sender.py - Email notifications
29. progress_utils.py - Progress displays

---

## UNUSED FILES (Can Delete - 30 files)

These files are NOT imported by the main workflows and can be safely removed:

### Testing Files (9 files)
- test_adaptive_quick.py
- test_all_updates.py
- test_claude_api.py
- test_dc_only.py
- test_new_outputs.py
- test_outputs_sample.py
- test_stats_generation.py
- validate_fixes.py
- pip install numpy pandas scikit-learn xg.py (invalid filename)

### Alternative/Legacy Implementations (10 files)
- backtest_adaptive_dc.py
- backtest_adaptive_visualizer.py
- backtest_config.py
- backtest_engine.py
- backtest_error_handling.py
- backtest_visualizer.py
- realistic_backtest.py
- enhanced_predict.py
- weighted_top50.py
- setup_structure.py

### Alternative Downloaders (4 files)
- api_football_weekly_fixtures.py
- auto_fixture_downloader.py
- combined_fixture_downloader.py
- cup_fixtures_addon_FINAL.py

### Standalone Utilities (7 files)
- accumulator_finder.py
- bet_finder.py
- high_confidence_bets.py
- ou_accumulators.py
- ou_analyzer.py
- combine_cross_league.py
- load_existing_stats.py

---

## Recommendations for Light Version

### Keep (29 files)
All files in the "ESSENTIAL FILES" section above.

### Delete (30 files)
All files in the "UNUSED FILES" section above.

### Data Folders to Keep
- data/raw/ - Historical CSV data
- data/processed/ - Processed parquets
- models/ - Trained models
- outputs/ - Prediction outputs

### Optional: Archive Testing Files
Instead of deleting, you could move test files to a `/tests/` subfolder for future reference:
- All test_*.py files
- validate_fixes.py

---

## Space Savings

Removing 30 unused Python files will:
- Reduce clutter by ~50%
- Make the codebase easier to navigate
- Reduce maintenance burden
- No impact on functionality (files are not used)

---

## Next Steps

1. **Backup first**: Create a copy of the entire folder before deletion
2. **Delete unused files**: Remove the 30 files listed above
3. **Test**: Run `py run_weekly.py` to verify everything works
4. **Optional**: Move test files to `/tests/` folder instead of deleting

---

Generated: 2025-12-08
Analysis method: Static import tracing from main entry points
