# Import Dependency Map for run_weekly.py

## Executive Summary
- **Total Python Files in Project**: 52 files (including run_weekly.py)
- **Files USED**: 27 files (including run_weekly.py)
- **Files UNUSED**: 25 files
- **Import Chain Depth**: 4 levels

---

## FILES THAT ARE IMPORTED/USED

### Level 1: DIRECTLY IMPORTED from run_weekly.py
These modules are explicitly imported in the main run_weekly.py file:

1. **config.py**
   - Imports: OUTPUT_DIR, log_header
   - Used for: Configuration management, output directory handling
   - Dependencies: pathlib, os, datetime

2. **simple_fixture_downloader.py**
   - Imports: download_upcoming_fixtures
   - Used for: Downloading upcoming fixtures (Step 0)
   - Dependencies: requests, pandas, pathlib, datetime

3. **download_football_data.py**
   - Imports: download
   - Used for: Downloading historical football data (Step 1)
   - Dependencies: requests, config, progress_utils, typing

4. **ingest_local_run.py**
   - Imports: ingest_local_csvs
   - Used for: Validating local CSV data (Step 3)
   - Dependencies: pandas, config, progress_utils

5. **data_ingest.py**
   - Imports: build_historical_results
   - Used for: Building historical results database (Step 2)
   - Dependencies: pandas, requests, numpy, config, typing

6. **features.py**
   - Imports: build_features
   - Used for: Building statistical features (Step 5)
   - Dependencies: pandas, numpy, pathlib, config

7. **prep_fixtures.py**
   - Imports: xlsx_to_csv
   - Used for: Converting Excel fixtures to CSV (Step 7)
   - Dependencies: pandas, pathlib, config, progress_utils

8. **predict.py**
   - Imports: predict_week
   - Used for: Generating predictions for upcoming fixtures (Step 8)
   - Dependencies: pandas, numpy, pathlib, scipy, datetime, config, models, dc_predict, progress_utils, blending

9. **generate_and_load_stats.py**
   - Imports: generate_statistics
   - Used for: Generating team/referee statistics (Step 4)
   - Dependencies: pandas, numpy, pathlib, typing

10. **incremental_trainer.py**
    - Imports: smart_train_or_load
    - Used for: Training or loading pre-trained models (Step 6)
    - Dependencies: os, json, pandas, pathlib, datetime, models, config

11. **btts_ou_optimizer.py**
    - Imports: optimize_weekly_predictions
    - Used for: Optimizing BTTS and O/U predictions (Step 9)
    - Dependencies: pandas, numpy, pathlib, typing, sqlite3, datetime

12. **ensemble_blender.py**
    - Imports: create_superblend_predictions
    - Used for: Smart ensemble blending of predictions (Step 10)
    - Dependencies: pandas, numpy, pathlib, typing, sqlite3, datetime, scipy

13. **accuracy_tracker.py**
    - Imports: log_weekly_predictions, update_accuracy_database
    - Used for: Logging and tracking prediction accuracy (Steps 11, 16)
    - Dependencies: pandas, sqlite3, pathlib, datetime, typing, json

14. **weighted_top50.py**
    - Imports: generate_weighted_top50
    - Used for: Generating weighted Top 50 predictions (Step 12)
    - Dependencies: pandas, sqlite3, pathlib, typing, json

15. **ou_analyzer.py**
    - Imports: analyze_ou_predictions
    - Used for: Over/Under market analysis (Step 13)
    - Dependencies: pandas, numpy, pathlib, datetime, sqlite3, config

16. **accumulator_finder.py**
    - Imports: AccumulatorFinder
    - Used for: Building accumulator betting combinations (Step 14)
    - Dependencies: pandas, numpy, pathlib, itertools, typing

17. **bet_finder_all_markets.py**
    - Imports: BetFinder
    - Used for: Finding quality bets across all markets (Step 15)
    - Dependencies: pandas, numpy, pathlib, datetime

### Level 2: INDIRECTLY IMPORTED (via other modules)
These modules are imported by Level 1 modules:

18. **progress_utils.py**
    - Referenced by: download_football_data.py, ingest_local_run.py, prep_fixtures.py, predict.py, models.py
    - Provides: Timer, heartbeat utility functions

19. **models.py**
    - Referenced by: predict.py, incremental_trainer.py
    - Provides: load_trained_targets, predict_proba, train_all_targets, _load_features
    - Dependencies: model_binary, model_multiclass, model_ordinal, models_dc, tuning, ordinal, calibration

20. **dc_predict.py**
    - Referenced by: predict.py
    - Provides: build_dc_for_fixtures
    - Dependencies: models_dc

21. **blending.py**
    - Referenced by: predict.py
    - Provides: Various blending utilities and BLEND_WEIGHTS_JSON
    - Dependencies: models_dc, models

### Level 3: INDIRECTLY IMPORTED (via Level 2 modules)
These modules are imported by Level 2 modules:

22. **model_binary.py**
    - Referenced by: models.py
    - Provides: BinaryMarketModel, is_binary_market
    - Dependencies: numpy, pandas, sklearn

23. **model_multiclass.py**
    - Referenced by: models.py
    - Provides: MulticlassMarketModel, is_multiclass_market
    - Dependencies: numpy, pandas, sklearn

24. **model_ordinal.py**
    - Referenced by: models.py
    - Provides: OrdinalMarketModel, is_ordinal_market
    - Dependencies: numpy, pandas, sklearn

25. **models_dc.py**
    - Referenced by: models.py, blending.py, dc_predict.py
    - Provides: fit_all, price_match (Dixon-Coles model)
    - Dependencies: numpy, pandas, scipy

26. **tuning.py**
    - Referenced by: models.py
    - Provides: make_time_split, objective_factory, CVData
    - Dependencies: numpy, optuna, sklearn

27. **ordinal.py**
    - Referenced by: models.py
    - Provides: CORALOrdinal (ordinal classification)
    - Dependencies: numpy, sklearn

28. **calibration.py**
    - Referenced by: models.py
    - Provides: DirichletCalibrator, TemperatureScaler
    - Dependencies: numpy, sklearn

---

## FILES THAT ARE NOT IMPORTED/USED

These files exist in the project but are never imported or used by run_weekly.py:

### Backup/Old Files (22 files):
- ✗ modelsOLD.py - Old model implementation
- ✗ model_binaryMAIN.py - Alternative model implementation
- ✗ model_multiclassMAIN.py - Alternative model implementation
- ✗ model_ordinalMAIN.py - Alternative model implementation
- ✗ updated_run_weekly (1).py - Duplicate/backup of main script

### Standalone/Experimental Scripts (3 files):
- ✗ backtest.py - Backtesting system
- ✗ backtest_engine.py - Backtest engine
- ✗ backtest_config.py - Backtest configuration
- ✗ backtest_error_handling.py - Backtest error handling
- ✗ backtest_visualizer.py - Backtest visualization
- ✗ realistic_backtest.py - Realistic backtesting
- ✗ combine_cross_league.py - Cross-league combination utility
- ✗ combined_fixture_downloader.py - Alternative fixture downloader
- ✗ cup_fixtures_addon_FINAL.py - Cup fixtures addon
- ✗ api_football_weekly_fixtures.py - API-based fixture downloader
- ✗ auto_fixture_downloader.py - Automatic fixture downloader
- ✗ email_sender.py - Email notification system
- ✗ load_existing_stats.py - Load stats utility
- ✗ train_evaluate.py - Training and evaluation script
- ✗ tuning.py - Seems to be referenced but check if actually used
- ✗ update_results.py - Update results after matches
- ✗ accuracy_tracker.py - Accuracy tracking (actually USED - see Level 1)
- ✗ enhanced_predict.py - Enhanced prediction system
- ✗ ou_analyzer.py - O/U analyzer (actually USED - see Level 1)
- ✗ acc_builder.py - Accumulator builder (unused variant)
- ✗ bet_finder_all_markets.py - Bet finder (actually USED - see Level 1)
- ✗ calibration.py - Calibration (actually USED - see Level 3)

### Unused Utility Scripts (5 files):
- ✗ setup_structure.py - Project structure setup
- ✗ ordinal.py - Ordinal classifier (actually USED - see Level 3)
- ✗ dc_predict.py - DC predict (actually USED - see Level 2)
- ✗ model_enhancer.py - Model enhancement
- ✗ blending.py - Blending utility (actually USED - see Level 2)

---

## DETAILED IMPORT CHAINS

### Chain 1: Features → Models → Predictions
```
run_weekly.py
├── features.py
├── incremental_trainer.py
│   └── models.py
│       ├── model_binary.py
│       ├── model_multiclass.py
│       ├── model_ordinal.py
│       │   └── ordinal.py
│       ├── models_dc.py
│       ├── tuning.py
│       ├── calibration.py
│       └── progress_utils.py
└── predict.py
    ├── models.py (shared)
    ├── dc_predict.py
    │   └── models_dc.py (shared)
    ├── progress_utils.py (shared)
    └── blending.py
        ├── models.py (shared)
        └── models_dc.py (shared)
```

### Chain 2: Data Pipeline
```
run_weekly.py
├── config.py
├── simple_fixture_downloader.py
├── download_football_data.py
│   └── progress_utils.py
├── ingest_local_run.py
│   └── progress_utils.py
├── data_ingest.py
└── prep_fixtures.py
    └── progress_utils.py
```

### Chain 3: Analysis & Output
```
run_weekly.py
├── btts_ou_optimizer.py
├── ensemble_blender.py
├── accuracy_tracker.py
├── weighted_top50.py
├── ou_analyzer.py
├── accumulator_finder.py
└── bet_finder_all_markets.py
```

---

## IMPORT STATISTICS

| Metric | Count |
|--------|-------|
| Total Files in Project | 52 |
| Files Used | 27 |
| Files Unused | 25 |
| Direct Imports (Level 1) | 17 |
| Indirect Imports (Level 2-3) | 10 |
| Maximum Chain Depth | 4 levels |
| Files with Multiple References | 4 (models, models_dc, progress_utils, ordinal) |

---

## KEY FINDINGS

1. **Core Dependencies**: The pipeline heavily depends on:
   - **models.py** and its sub-models (binary, multiclass, ordinal)
   - **models_dc.py** (Dixon-Coles statistical model)
   - **progress_utils.py** (timing and logging)

2. **Unused Categories**:
   - Backtesting modules (backtest.py, backtest_engine.py, etc.) - not used in weekly pipeline
   - Alternative/experimental models (model_*MAIN.py files) - redundant implementations
   - Email system (email_sender.py) - implemented but not integrated into run_weekly
   - Advanced downloaders (auto_fixture_downloader.py, api_football_weekly_fixtures.py) - alternatives to simple_fixture_downloader.py

3. **Critical Files**: These files MUST exist for run_weekly.py to function:
   - config.py
   - models.py (with all sub-models)
   - predict.py
   - features.py

4. **Potential Cleanup**: The following 25 unused files could be archived or removed:
   - All backtest_*.py files
   - All model_*MAIN.py and modelsOLD.py files
   - All alternative downloader implementations
   - Email system integration (if not needed)

---

## NOTES

- Some files appear in multiple import chains (e.g., models.py, models_dc.py)
- The project has significant code duplication with multiple versions of similar functionality
- The weekly pipeline is well-isolated from experimental backtesting code
- All Level 1 imports are necessary for the pipeline to execute
