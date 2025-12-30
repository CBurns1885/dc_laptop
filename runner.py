#!/usr/bin/env python3
"""
RUNNER.PY - API-Football Enhanced Weekly Runner
Primary data source: API-Football (with xG, injuries, formations)
Model architecture: Original Dixon-Coles with all enhancements
Outputs: Same as run_weekly.py (market splits, accumulators, quality bets, etc.)
"""

import sys
import os
from pathlib import Path
import datetime
from config import OUTPUT_DIR

# Fix Unicode encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ============================================================================
# CONFIGURATION
# ============================================================================

# API-Football Configuration
API_KEY = os.environ.get("API_FOOTBALL_KEY", "0f17fdba78d15a625710f7244a1cc770")

# Model Configuration (same as run_weekly.py)
os.environ["DISABLE_XGB"] = "0"  # Enable XGBoost
os.environ["OPTUNA_TRIALS"] = "0"  # Fast incremental (no hyperparameter tuning)
os.environ["N_ESTIMATORS"] = "150"

# Email Configuration (optional)
os.environ["EMAIL_SMTP_SERVER"] = "smtp-mail.outlook.com"
os.environ["EMAIL_SMTP_PORT"] = "587"
os.environ["EMAIL_SENDER"] = "christopher_burns@live.co.uk"
os.environ["EMAIL_PASSWORD"] = ""
os.environ["EMAIL_RECIPIENT"] = "christopher_burns@live.co.uk"

# Training Configuration - TEST WITH E0 ONLY
TRAINING_START_YEAR = 2025 #mtch original system
TRAINING_SEASONS = [ 2025]  # Full history for proper training

# League Configuration - START WITH E0 FOR TESTING
LEAGUES_TO_USE = ['E0']  # Premier League only for test

# After successful test, expand to all leagues:
# LEAGUES_TO_USE = ['E0', 'E1', 'E2', 'E3', 'EC', 'D1', 'D2', 'SP1', 'SP2',
#                   'I1', 'I2', 'F1', 'F2', 'N1', 'B1', 'P1', 'G1',
#                   'SC0', 'SC1', 'T1']

print("="*70)
print("API-FOOTBALL ENHANCED PREDICTION SYSTEM")
print("="*70)
print(f"\nConfiguration:")
print(f"  Data Source: API-Football Pro (xG, injuries, formations)")
print(f"  Training: {TRAINING_START_YEAR}-{datetime.datetime.now().year}")
print(f"  Leagues: {LEAGUES_TO_USE}")
print(f"  Model: Dixon-Coles + XGBoost + Ensemble + Calibration")
print(f"  Outputs: Full pipeline (same as run_weekly.py)")
print("="*70)

# ============================================================================
# STEP -1: UPDATE ACCURACY FROM LAST WEEK
# ============================================================================

print("\n[-1] UPDATING ACCURACY FROM LAST WEEK")
print("="*70)

try:
    accuracy_db = Path("outputs/accuracy_database.db")

    if accuracy_db.exists():
        print("Found accuracy database")

        import sqlite3
        conn = sqlite3.connect(accuracy_db)
        cursor = conn.cursor()
        pending = cursor.execute("""
            SELECT COUNT(*) FROM predictions WHERE actual_outcome IS NULL
        """).fetchone()[0]
        conn.close()

        if pending > 0:
            print(f"Found {pending} predictions awaiting results")
            print("Updating with latest match results...")

            from update_results import update_accuracy_database, show_recent_performance

            success = update_accuracy_database()

            if success:
                print("\nRecent Performance:")
                print("-" * 70)
                show_recent_performance(weeks=2)
                print("-" * 70)
            else:
                print("No new results available yet")
        else:
            print("All predictions up to date")
    else:
        print("No accuracy database yet (first run)")

except Exception as e:
    print(f"Accuracy update skipped: {e}")

# ============================================================================
# STEP 0: API-FOOTBALL DATA INGESTION
# ============================================================================

print("\n[0] DATA INGESTION - API-FOOTBALL")
print("="*70)

try:
    # Import API-Football modules without polluting global path
    api_football_dir = Path(__file__).parent / "api_football"

    # Temporarily add to path for import
    sys.path.insert(0, str(api_football_dir))
    from data_ingest_api import APIFootballIngestor
    from features_api import build_features as build_api_features
    from predict_api import MatchPredictor
    # Remove from path to avoid conflicts with old system
    sys.path.remove(str(api_football_dir))

    print(f"Downloading historical data from API-Football...")
    print(f"  Leagues: {LEAGUES_TO_USE}")
    print(f"  Seasons: {TRAINING_SEASONS}")
    print(f"  This will download all historical matches with xG, injuries, etc.")

    ingestor = APIFootballIngestor(API_KEY)
    ingestor.ingest_all(
        leagues=LEAGUES_TO_USE,
        seasons=TRAINING_SEASONS,
        include_stats=True,
        include_injuries=True
    )

    # Export to parquet for features step
    parquet_path = ingestor.export_to_parquet()
    print(f"Data exported to: {parquet_path}")

except Exception as e:
    print(f"ERROR: Data ingestion failed: {e}")
    import traceback
    traceback.print_exc()
    print("\nCannot continue without data. Exiting.")
    sys.exit(1)

# ============================================================================
# STEP 1: BUILD FEATURES (API-FOOTBALL ENHANCED)
# ============================================================================

print("\n[1] FEATURE ENGINEERING - API ENHANCED")
print("="*70)

try:
    # Use the imported function
    features_path = build_api_features(force=True)

    if features_path and features_path.exists():
        print(f"Features built: {features_path}")

        # Check feature quality
        import pandas as pd
        df = pd.read_parquet(features_path)
        print(f"\nFeature Summary:")
        print(f"  Total matches: {len(df):,}")
        print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

        # Check xG coverage
        if 'home_xG' in df.columns:
            xg_coverage = df['home_xG'].notna().mean()
            print(f"  xG coverage: {xg_coverage:.1%}")
    else:
        print("ERROR: Feature building failed")
        sys.exit(1)

except Exception as e:
    print(f"ERROR: Feature engineering failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 2: CONVERT API FEATURES TO STANDARD FORMAT
# ============================================================================

print("\n[2] CONVERTING TO STANDARD FORMAT")
print("="*70)

try:
    import pandas as pd

    # Load API features
    api_features = pd.read_parquet(features_path)

    # Map to standard format expected by old system
    # The old system expects: data/processed/historical_matches.parquet

    standard_path = Path("data/processed/historical_matches.parquet")
    standard_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure required columns exist
    required_cols = ['Date', 'League', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']

    if all(col in api_features.columns for col in required_cols):
        # Save in standard location
        api_features.to_parquet(standard_path, index=False)
        print(f"Converted features saved to: {standard_path}")
        print(f"  Rows: {len(api_features):,}")
    else:
        missing = [col for col in required_cols if col not in api_features.columns]
        print(f"ERROR: Missing required columns: {missing}")
        sys.exit(1)

except Exception as e:
    print(f"ERROR: Format conversion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 3: VERIFY FEATURES FILE (DON'T OVERWRITE API DATA)
# ============================================================================

print("\n[3] VERIFY FEATURES FILE")
print("="*70)

try:
    # The API features are already in the correct format and location
    # DO NOT call old build_features() - it would overwrite with CSV data lacking HTHG/HTAG
    features_file = Path("data/processed/historical_matches.parquet")

    if features_file.exists():
        import pandas as pd
        df = pd.read_parquet(features_file)
        print(f"Features verified: {len(df):,} matches, {len(df.columns)} columns")
        print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        print(f"  Leagues: {df['League'].unique().tolist()}")

        # Check critical columns from API
        api_columns = {
            'Core Match Data': ['Date', 'League', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'],
            'Half-Time Scores': ['HTHG', 'HTAG'],
            'xG Data': ['home_xG', 'away_xG', 'HxG_stat', 'AxG_stat'],
            'Basic Stats': ['HS', 'HST', 'HC', 'HY', 'HR', 'AS_inside', 'AST', 'AC', 'AY', 'AR'],
            'Advanced Stats': ['HPoss', 'APoss', 'HPasses', 'APasses', 'HSaves', 'ASaves'],
            'Injury Data': ['home_injuries_count', 'away_injuries_count', 'home_key_injuries', 'away_key_injuries'],
            'Rest Days': ['home_rest_days', 'away_rest_days', 'home_rest_band', 'away_rest_band'],
            'xG Features': ['home_xG_for_ma5', 'away_xG_for_ma5', 'home_xG_overperformance', 'away_xG_overperformance'],
            'Formation': ['home_formation_attack', 'away_formation_attack', 'formation_matchup_goals_mult'],
            'H2H Features': ['h2h_total_goals_avg', 'h2h_btts_rate', 'h2h_home_win_rate', 'h2h_meetings'],
            'Cup Features': ['is_cup_match', 'is_european_cup', 'cup_home_advantage_factor'],
            'Target Variables': ['y_BTTS', 'y_OU_2_5', 'y_OU_1_5', 'y_OU_3_5']
        }

        print("\nAPI Column Verification:")
        all_present = True
        for category, cols in api_columns.items():
            present = [c for c in cols if c in df.columns]
            missing = [c for c in cols if c not in df.columns]

            if missing:
                print(f"  {category}: {len(present)}/{len(cols)} present (MISSING: {', '.join(missing)})")
                all_present = False
            else:
                # Show data coverage for key columns
                if category == 'Half-Time Scores':
                    ht_coverage = (df['HTHG'].notna()).mean()
                    print(f"  {category}: ✓ Present ({ht_coverage:.1%} coverage)")
                elif category == 'xG Data':
                    xg_coverage = (df['home_xG'].notna()).mean()
                    print(f"  {category}: ✓ Present ({xg_coverage:.1%} coverage)")
                else:
                    print(f"  {category}: ✓ All present")

        if not all_present:
            print("\n⚠ WARNING: Some columns are missing!")
        else:
            print("\n✓ All API columns verified successfully")
    else:
        print("ERROR: Features file not found")
        sys.exit(1)

except Exception as e:
    print(f"ERROR: Feature verification failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 4: TRAIN/LOAD MODELS (ORIGINAL ARCHITECTURE)
# ============================================================================

print("\n[4] TRAIN/LOAD MODELS")
print("="*70)

try:
    from incremental_trainer import smart_train_or_load

    models = smart_train_or_load()

    if models:
        print("Models loaded/trained successfully")
    else:
        print("ERROR: Model training failed")
        sys.exit(1)

except Exception as e:
    print(f"ERROR: Model training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 5: GET UPCOMING FIXTURES FROM API
# ============================================================================

print("\n[5] FETCHING UPCOMING FIXTURES")
print("="*70)

try:
    # Use the imported class
    predictor = MatchPredictor(api_key=API_KEY)

    # Get upcoming fixtures
    print("Fetching next 7 days of fixtures from API...")
    upcoming_df = predictor.predict_upcoming(days=7, leagues=LEAGUES_TO_USE)

    if not upcoming_df.empty:
        print(f"Found {len(upcoming_df)} upcoming fixtures")

        # Save to standard location for compatibility
        fixtures_csv = OUTPUT_DIR / "upcoming_fixtures.csv"

        # Convert API format to standard format
        standard_fixtures = pd.DataFrame({
            'Date': pd.to_datetime(upcoming_df['date']).dt.strftime('%d/%m/%Y'),
            'Time': pd.to_datetime(upcoming_df['date']).dt.strftime('%H:%M'),
            'League': upcoming_df['league'],
            'HomeTeam': upcoming_df['home'],
            'AwayTeam': upcoming_df['away']
        })

        standard_fixtures.to_csv(fixtures_csv, index=False)
        print(f"Fixtures saved to: {fixtures_csv}")
    else:
        print("Warning: No upcoming fixtures found")
        # Create dummy file so pipeline doesn't crash
        fixtures_csv = OUTPUT_DIR / "upcoming_fixtures.csv"
        pd.DataFrame(columns=['Date', 'Time', 'League', 'HomeTeam', 'AwayTeam']).to_csv(fixtures_csv, index=False)

except Exception as e:
    print(f"Warning: Could not fetch API fixtures: {e}")
    print("Will try to use existing fixtures file...")
    fixtures_csv = OUTPUT_DIR / "upcoming_fixtures.csv"
    if not fixtures_csv.exists():
        print("ERROR: No fixtures available")
        sys.exit(1)

# ============================================================================
# STEP 6: GENERATE PREDICTIONS (ORIGINAL SYSTEM)
# ============================================================================

print("\n[6] GENERATE PREDICTIONS")
print("="*70)

try:
    from predict import predict_week

    predict_week(fixtures_csv)
    print("Predictions generated successfully")

except Exception as e:
    print(f"ERROR: Prediction generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 7-19: RUN ALL ENHANCEMENT STEPS (ORIGINAL PIPELINE)
# ============================================================================

TOTAL_STEPS = 19
errors = []

def run_step(step_num, step_name, func, *args, **kwargs):
    """Run a step with error recovery"""
    import time

    percentage = int((step_num / TOTAL_STEPS) * 100)
    print(f"\n{'='*70}")
    print(f"STEP {step_num}/{TOTAL_STEPS} ({percentage}%): {step_name}")
    print('='*70)

    step_start = time.time()
    try:
        result = func(*args, **kwargs)
        step_elapsed = time.time() - step_start
        print(f"Step {step_num} complete ({step_elapsed:.1f}s)")
        return result, None
    except Exception as e:
        step_elapsed = time.time() - step_start
        error_msg = f"Step {step_num} ({step_name}): {str(e)}"
        errors.append(error_msg)
        print(f"Step {step_num} failed: {e} ({step_elapsed:.1f}s)")
        print("Continuing to next step...")
        return None, error_msg

try:
    import datetime as dt

    # Step 7: Split Predictions by Market
    def step7():
        from market_splitter import split_predictions
        csv_path = OUTPUT_DIR / "weekly_bets_lite.csv"

        if csv_path.exists():
            split_predictions(csv_path, OUTPUT_DIR)
            print("Predictions split into market-specific files")
        else:
            raise FileNotFoundError("weekly_bets_lite.csv not found")

    run_step(7, "SPLIT BY MARKET", step7)

    # Step 8: Optimize BTTS/O/U Predictions
    def step8_optimize():
        from btts_ou_optimizer import optimize_weekly_predictions
        csv_path = OUTPUT_DIR / "weekly_bets_lite.csv"
        hist_path = Path("data/historical_results.parquet")
        output_path = OUTPUT_DIR / "weekly_bets_lite_optimized.csv"

        if csv_path.exists():
            import pandas as pd
            hist_df = pd.read_parquet(hist_path) if hist_path.exists() else None
            optimize_weekly_predictions(csv_path, hist_df, output_path)

            import shutil
            shutil.copy(output_path, csv_path)
            print("BTTS/O/U optimized with historical patterns")
        else:
            raise FileNotFoundError("weekly_bets_lite.csv not found")

    run_step(8, "OPTIMIZE BTTS/O/U", step8_optimize)

    # Step 9: Smart Ensemble Blending
    def step9_blend():
        from ensemble_blender import create_superblend_predictions
        input_file = OUTPUT_DIR / "weekly_bets_lite.csv"
        output_file = OUTPUT_DIR / "weekly_bets_lite_blended.csv"

        if input_file.exists():
            create_superblend_predictions(input_file, output_file)

            import shutil
            shutil.copy(output_file, input_file)
            print("Ensemble weights optimized")
        else:
            raise FileNotFoundError("weekly_bets_lite.csv not found")

    run_step(9, "SMART ENSEMBLE BLENDING", step9_blend)

    # Step 10: Log predictions
    def step10_log():
        from accuracy_tracker import log_weekly_predictions
        week_id = dt.datetime.now().strftime('%Y-W%W')
        csv_path = OUTPUT_DIR / "weekly_bets_lite.csv"

        if csv_path.exists():
            log_weekly_predictions(csv_path, week_id)
            print(f"Logged predictions (Week {week_id})")
        else:
            raise FileNotFoundError("weekly_bets_lite.csv not found")

    run_step(10, "LOG PREDICTIONS", step10_log)

    # Step 11: High Confidence Bets
    def step11_highconf():
        from generate_outputs_from_actual import generate_high_confidence_bets
        csv_path = OUTPUT_DIR / "weekly_bets_lite.csv"

        if csv_path.exists():
            generate_high_confidence_bets(csv_path, threshold=0.90)
        else:
            raise FileNotFoundError("weekly_bets_lite.csv not found")

    run_step(11, "HIGH CONFIDENCE BETS (90%+)", step11_highconf)

    # Step 12: O/U Accumulators
    def step12_acca():
        from generate_outputs_from_actual import generate_ou_accumulators
        csv_path = OUTPUT_DIR / "high_confidence_bets.csv"

        if csv_path.exists():
            generate_ou_accumulators(csv_path)
        else:
            raise FileNotFoundError("high_confidence_bets.csv not found")

    run_step(12, "O/U ACCUMULATORS (4-FOLD, 90%+)", step12_acca)

    # Step 13: Update accuracy database
    def step13_track():
        from accuracy_tracker import log_weekly_predictions
        csv_path = OUTPUT_DIR / "weekly_bets_lite.csv"

        if csv_path.exists():
            log_weekly_predictions(csv_path)
            print("Predictions logged for accuracy tracking")
        else:
            print("weekly_bets_lite.csv not found, skipping")

    run_step(13, "UPDATE ACCURACY DATABASE", step13_track)

    # Step 14: Generate Excel Report
    def step14_excel():
        from excel_generator import generate_excel_report
        from email_sender import send_weekly_predictions

        csv_path = OUTPUT_DIR / "weekly_bets.csv"

        if csv_path.exists():
            excel_file = generate_excel_report(csv_path, OUTPUT_DIR)

            html_report = OUTPUT_DIR / "top50_weighted.html"
            if not html_report.exists():
                html_report = None

            send_weekly_predictions(excel_file, html_report)

            print(f"Excel report generated: {excel_file.name}")
        else:
            raise FileNotFoundError("weekly_bets.csv not found")

    run_step(14, "GENERATE EXCEL & SEND EMAIL", step14_excel)

    # Step 15: Archive outputs
    def step15_archive():
        """Copy all output files to dated archive folder"""
        import shutil
        from datetime import datetime
        import glob

        date_str = datetime.now().strftime('%Y-%m-%d')
        archive_dir = Path("archives") / date_str
        archive_dir.mkdir(parents=True, exist_ok=True)

        files_to_archive = [
            "weekly_bets_lite.csv",
            "top50_weighted.html",
            "top50_weighted.csv",
            "ou_analysis.html",
            "ou_analysis.csv",
            "ou_analysis.xlsx",
            "accumulators_safe.html",
            "accumulators_mixed.html",
            "accumulators_aggressive.html",
        ]

        quality_bet_files = glob.glob(str(OUTPUT_DIR / "quality_bets_*.csv"))
        quality_bet_files += glob.glob(str(OUTPUT_DIR / "quality_bets_*.html"))

        archived_count = 0
        for filename in files_to_archive:
            source = OUTPUT_DIR / filename
            if source.exists():
                name_parts = filename.rsplit('.', 1)
                if len(name_parts) == 2:
                    archived_name = f"{name_parts[0]}_{date_str}.{name_parts[1]}"
                else:
                    archived_name = f"{filename}_{date_str}"

                dest = archive_dir / archived_name
                shutil.copy2(source, dest)
                archived_count += 1

        for filepath in quality_bet_files:
            filename = Path(filepath).name
            dest = archive_dir / filename
            shutil.copy2(filepath, dest)
            archived_count += 1

        print(f"Archived {archived_count} files to {archive_dir}")

    run_step(15, "ARCHIVE OUTPUTS", step15_archive)

except Exception as e:
    print(f"ERROR in enhancement pipeline: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# SUCCESS SUMMARY
# ============================================================================

print("\n" + "="*70)
print("PIPELINE COMPLETE!")
print("="*70)

if errors:
    print(f"\n{len(errors)} step(s) had errors:")
    for error in errors:
        print(f"  - {error}")
    print("\nCheck outputs folder - some files may still be generated")
else:
    print("\nAll steps completed successfully!")

print("\nData Source: API-Football Pro")
print("  - xG data integrated")
print("  - Injury tracking enabled")
print("  - Enhanced accuracy")

print("\nMain Files:")
print("  - weekly_bets_lite.csv - All predictions (master file)")

print("\nMarket-Specific Files:")
print("  - predictions_1x2.html/csv")
print("  - predictions_btts.html/csv")
print("  - predictions_ou_2_5.html/csv")

print("\nSpecialized Reports:")
print("  - top50_weighted.html")
print("  - ou_analysis.html")
print("  - accumulators_*.html")
print(f"  - quality_bets_{datetime.datetime.now().strftime('%Y%m%d')}.html")

print("\nTracking:")
print("  - accuracy_database.db")

print("\n" + "="*70)
print("Next Steps:")
print("  1. Check predictions_*.html files")
print("  2. Review top50_weighted.html for best bets")
print("  3. Check quality_bets HTML for opportunities")
print("  4. Review accumulator files")
print("  5. After matches: run update_results.py")
print("="*70)

# Open outputs folder
try:
    import subprocess
    import platform

    outputs_dir = OUTPUT_DIR
    if platform.system() == "Windows":
        subprocess.run(["explorer", str(outputs_dir)], check=False)
    elif platform.system() == "Darwin":
        subprocess.run(["open", str(outputs_dir)], check=False)
    else:
        subprocess.run(["xdg-open", str(outputs_dir)], check=False)
except:
    pass
