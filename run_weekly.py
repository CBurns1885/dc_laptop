#!/usr/bin/env python3
"""
RUN_WEEKLY.py - Complete Weekly Runner
Includes: O/U Analyzer + Accumulator Builder + Simple Fixture Downloader + Quality Bet Finder
"""
#training year start is this file and league codes are used from config, this overrules run_weekly and the downloader
import sys
import os
from pathlib import Path
import datetime
from config import OUTPUT_DIR

# ============================================================================
# CONFIGURATION
# ============================================================================

os.environ["DISABLE_XGB"] = "0"  # Enable XGBoost for better accuracy (set to "1" to disable)
os.environ["OPTUNA_TRIALS"] = "0"  # Set to 0 for fast incremental updates (no hyperparameter tuning)
os.environ["N_ESTIMATORS"] = "150"
os.environ["EMAIL_SMTP_SERVER"] = "smtp-mail.outlook.com"
os.environ["EMAIL_SMTP_PORT"] = "587"
os.environ["EMAIL_SENDER"] = "christopher_burns@live.co.uk"
os.environ["EMAIL_PASSWORD"] = ""
os.environ["EMAIL_RECIPIENT"] = "christopher_burns@live.co.uk"

TRAINING_START_YEAR = 2021  # Start year for historical training data (2021-2025 = 5 years)

# ============================================================================
# LEAGUE OVERRIDE OPTION
# ============================================================================
# Set to True to use ALL leagues regardless of fixtures file
# Set to False (default) to auto-detect leagues from fixtures
USE_ALL_LEAGUES = False

# Default leagues (used as fallback if fixtures not available or if USE_ALL_LEAGUES=True)
DEFAULT_LEAGUES = ["E0", "E1", "E2", "E3", "EC",
                   "D1", "D2",
                   "SP1", "SP2",
                   "I1", "I2",
                   "F1", "F2",
                   "N1", "B1", "P1", "G1",
                   "SC0", "SC1", "SC2", "SC3",
                   "T1"]

print("="*60)
print("FOOTBALL PREDICTION SYSTEM - WEEKLY RUN")
print("="*60)
print(f"\n Configuration:")
print(f"   Training Years: {TRAINING_START_YEAR}-{datetime.datetime.now().year}")
print(f"   Optuna Trials: {os.environ.get('OPTUNA_TRIALS', '0')}")
print(f"   League Mode: {'ALL LEAGUES' if USE_ALL_LEAGUES else 'AUTO-DETECT FROM FIXTURES'}")
if USE_ALL_LEAGUES:
    print(f"   Leagues: {len(DEFAULT_LEAGUES)} leagues (E0, E1, D1, SP1, etc.)")
print("="*60)

# ============================================================================
# STEP -1: UPDATE ACCURACY FROM LAST WEEK (if applicable)
# ============================================================================

print("\n STEP -1: Updating Accuracy from Last Week's Predictions")
print("="*60)

try:
    from pathlib import Path
    accuracy_db = Path("outputs/accuracy_database.db")

    if accuracy_db.exists():
        print(" Found accuracy database")

        # Check if there are pending predictions
        import sqlite3
        conn = sqlite3.connect(accuracy_db)
        cursor = conn.cursor()
        pending = cursor.execute("""
            SELECT COUNT(*) FROM predictions WHERE actual_outcome IS NULL
        """).fetchone()[0]
        conn.close()

        if pending > 0:
            print(f" Found {pending} predictions awaiting results")
            print(" Updating with latest match results...")

            # Import and run update
            from update_results import update_accuracy_database, show_recent_performance

            success = update_accuracy_database()

            if success:
                print("\n Recent Performance:")
                print("-" * 60)
                show_recent_performance(weeks=2)
                print("-" * 60)
            else:
                print(" No new results available yet (matches may not have finished)")
        else:
            print(" No pending predictions (all up to date)")
    else:
        print("ℹ No accuracy database yet (will be created after first predictions)")

except Exception as e:
    print(f" Accuracy update skipped: {e}")
    print("   (This is normal for first run)")

print("\n" + "="*60)

# ============================================================================
# STEP 0: DOWNLOAD FIXTURES
# ============================================================================

print("\nSTEP 0: Downloading Fixtures")
print("="*40)

try:
    from simple_fixture_downloader import download_upcoming_fixtures
    print("Downloading upcoming fixtures from football-data.co.uk...")
    fixture_path = download_upcoming_fixtures()
    
    if fixture_path and fixture_path.exists():
        fixtures_file = fixture_path
        print(f"[OK] Downloaded: {fixtures_file}")
    else:
        print("[WARNING] Download failed, checking for manual file...")
        fixtures_file = None
except Exception as e:
    print(f"[WARNING] Auto-download error: {e}")
    fixtures_file = None

# Fallback to manual files
if not fixtures_file:
    fixtures_xlsx = Path("upcoming_fixtures.xlsx")
    fixtures_csv = Path("upcoming_fixtures.csv")
    outputs_csv = OUTPUT_DIR / "upcoming_fixtures.csv"
    
    if outputs_csv.exists():
        fixtures_file = outputs_csv
        print(f"[OK] Using: {fixtures_file}")
    elif fixtures_xlsx.exists():
        fixtures_file = fixtures_xlsx
        print(f"[OK] Using manual: {fixtures_file}")
    elif fixtures_csv.exists():
        fixtures_file = fixtures_csv
        print(f"[OK] Using manual: {fixtures_file}")
    else:
        print("[ERROR] No fixtures file found!")
        print("Options:")
        print("   1. Check internet connection for auto-download")
        print("   2. Manually download from football-data.co.uk/matches.php")
        print("   3. Save as upcoming_fixtures.csv in this folder")
        input("Press Enter to exit...")
        sys.exit(1)

# ============================================================================
# VALIDATE FIXTURES FILE
# ============================================================================

print("\nValidating fixtures file...")
try:
    import pandas as pd
    
    if fixtures_file.suffix.lower() == '.xlsx':
        df = pd.read_excel(fixtures_file)
    else:
        df = pd.read_csv(fixtures_file)
    
    # Fix common column issues
    if "Div" in df.columns and "League" not in df.columns:
        df = df.rename(columns={"Div": "League"})
        print("[OK] Fixed: Renamed 'Div' -> 'League'")
        df.to_csv(fixtures_file, index=False)
    
    required = ["Date", "League", "HomeTeam", "AwayTeam"]
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        print(f"   Required: {required}")
        input("Press Enter to exit...")
        sys.exit(1)

    print(f"[OK] Validated: {len(df)} matches found")
    print(f"   Leagues: {df['League'].unique().tolist()}")

except Exception as e:
    print(f"[ERROR] Error reading fixtures: {e}")
    input("Press Enter to exit...")
    sys.exit(1)

# ============================================================================
# USER OPTIONS
# ============================================================================

print("\n" + "="*60)
print(" CONFIGURATION")
print("="*60)

print(f"   Optuna trials: {os.environ['OPTUNA_TRIALS']} (0=no tuning for fast incremental updates)")
print(f"   N Estimators: {os.environ['N_ESTIMATORS']}")
print(f"   Training start year: {TRAINING_START_YEAR}")
print(f"   Training period: {TRAINING_START_YEAR}-{datetime.datetime.now().year}")
print(f"   Incremental training: ENABLED (reuses models when possible)")

# ============================================================================
# RUN PIPELINE WITH ERROR RECOVERY
# ============================================================================

TOTAL_STEPS = 18
errors = []

def run_step(step_num, step_name, func, *args, **kwargs):
    """Run a step with error recovery and timing"""
    import time

    percentage = int((step_num / TOTAL_STEPS) * 100)
    print(f"\n{'='*60}")
    print(f"STEP {step_num}/{TOTAL_STEPS} ({percentage}%): {step_name}")
    print('='*60)

    step_start = time.time()
    try:
        result = func(*args, **kwargs)
        step_elapsed = time.time() - step_start
        print(f" Step {step_num} complete ({step_elapsed:.1f}s)")
        return result, None
    except Exception as e:
        step_elapsed = time.time() - step_start
        error_msg = f"Step {step_num} ({step_name}): {str(e)}"
        errors.append(error_msg)
        print(f" Step {step_num} failed: {e} ({step_elapsed:.1f}s)")
        print("   Continuing to next step...")
        return None, error_msg

try:
    from config import log_header
    from download_football_data import download
    from ingest_local_run import ingest_local_csvs
    from data_ingest import build_historical_results
    from features import build_features
    from prep_fixtures import xlsx_to_csv
    from predict import predict_week
    import datetime as dt

    # Store detected leagues for use across steps (using dict to avoid nonlocal issues)
    league_context = {'detected_leagues': None}

    # Step 1: Download historical data
    def step1():
        # Dynamically determine which leagues to download based on fixtures
        leagues_to_download = DEFAULT_LEAGUES

        # Check if user wants to override with all leagues
        if USE_ALL_LEAGUES:
            print(f" USE_ALL_LEAGUES=True - Using all {len(DEFAULT_LEAGUES)} leagues")
            leagues_to_download = DEFAULT_LEAGUES
        elif fixtures_file and fixtures_file.exists():
            try:
                import pandas as pd
                if fixtures_file.suffix.lower() == '.xlsx':
                    fixtures_df = pd.read_excel(fixtures_file)
                else:
                    fixtures_df = pd.read_csv(fixtures_file)

                if 'League' in fixtures_df.columns:
                    fixture_leagues = fixtures_df['League'].dropna().unique().tolist()
                    leagues_to_download = sorted(set(fixture_leagues))  # Remove duplicates and sort
                    print(f" Leagues found in fixtures: {leagues_to_download}")

                    # Check fixture freshness
                    if 'Date' in fixtures_df.columns:
                        fixtures_df['Date'] = pd.to_datetime(fixtures_df['Date'], errors='coerce')
                        latest_fixture = fixtures_df['Date'].max()
                        days_ahead = (latest_fixture - pd.Timestamp.now()).days

                        if days_ahead < 0:
                            print(f" WARNING: Fixtures are {abs(days_ahead)} days old!")
                            print(f"   Latest fixture: {latest_fixture.strftime('%Y-%m-%d')}")
                            print(f"   Consider downloading fresh fixtures")
                        elif days_ahead > 14:
                            print(f"ℹ Fixtures cover next {days_ahead} days")
                        else:
                            print(f" Fixtures are current (next {days_ahead} days)")
                else:
                    print(f" No 'League' column found, using defaults: {DEFAULT_LEAGUES}")
            except Exception as e:
                print(f" Could not read fixtures for league detection: {e}")
                print(f"   Using default leagues: {DEFAULT_LEAGUES}")
        else:
            print(f" No fixtures file found, using default leagues: {DEFAULT_LEAGUES}")

        print(f" Downloading {TRAINING_START_YEAR}-{dt.datetime.now().year}...")
        print(f"   Leagues: {leagues_to_download}")

        # Download with better error handling per league
        years = list(range(TRAINING_START_YEAR, dt.datetime.now().year + 1))
        print(f"   Years: {min(years)}-{max(years)} ({len(years)} seasons)")
        download(leagues_to_download, years)
    
    run_step(1, "DOWNLOAD HISTORICAL DATA", step1)

    # Step 2: Build database (filtered to fixture leagues)
    def step2():
        if USE_ALL_LEAGUES:
            league_context['detected_leagues'] = DEFAULT_LEAGUES
            print(f" Building database for all leagues (USE_ALL_LEAGUES=True)")
        elif fixtures_file and fixtures_file.exists():
            try:
                import pandas as pd
                if fixtures_file.suffix.lower() == '.xlsx':
                    fixtures_df = pd.read_excel(fixtures_file)
                else:
                    fixtures_df = pd.read_csv(fixtures_file)

                if 'League' in fixtures_df.columns:
                    league_context['detected_leagues'] = sorted(set(fixtures_df['League'].dropna().unique().tolist()))
                    print(f" Building database for leagues: {league_context['detected_leagues']}")
            except Exception as e:
                print(f" Could not read fixtures for filtering: {e}")

        build_historical_results(force=True)

    run_step(2, "BUILD HISTORICAL DATABASE", step2)
    
    # Step 3: Validate data
    def step3():
        ingest_local_csvs()
    
    run_step(3, "VALIDATE LOCAL DATA", step3)
    
    # Step 4: Generate statistics
    def step4():
        from generate_and_load_stats import generate_statistics
        generate_statistics()
        print(" Team/referee statistics generated")
    
    run_step(4, "GENERATE STATISTICS", step4)
    
    # Step 5: Build features (filtered to detected leagues only)
    def step5():
        if league_context['detected_leagues']:
            print(f" Filtering to leagues: {league_context['detected_leagues']}")
            # Filter historical data to only detected leagues before building features
            hist_path = Path("data/processed/historical_matches.parquet")
            hist_backup = Path("data/processed/historical_matches_full.parquet")

            if hist_path.exists():
                import pandas as pd

                # Backup full data if not already backed up (for future runs)
                if not hist_backup.exists():
                    print(f" Creating backup of full historical data...")
                    df_full = pd.read_parquet(hist_path)
                    df_full.to_parquet(hist_backup, index=False)
                    print(f" Backed up {len(df_full):,} matches to {hist_backup.name}")

                # Load full data from backup
                df_hist = pd.read_parquet(hist_backup)
                original_count = len(df_hist)
                original_leagues = df_hist['League'].nunique()

                # Filter to only leagues we need
                df_hist = df_hist[df_hist['League'].isin(league_context['detected_leagues'])]
                filtered_count = len(df_hist)
                filtered_leagues = df_hist['League'].nunique()

                print(f"   Filtered: {original_leagues} leagues → {filtered_leagues} leagues")
                print(f"   Matches: {original_count:,} → {filtered_count:,} ({100*filtered_count/original_count:.1f}%)")

                # Save filtered data for feature building
                df_hist.to_parquet(hist_path, index=False)
                print(f" Historical data filtered (original preserved in backup)")

        build_features(force=True)

    run_step(5, "BUILD FEATURES", step5)
    
    # Step 6: Train/load models
    def step6():
        from incremental_trainer import smart_train_or_load
        return smart_train_or_load()
    
    models, err = run_step(6, "TRAIN/LOAD MODELS", step6)
    
    # Step 7: Prepare fixtures
    def step7():
        if fixtures_file.suffix.lower() == '.xlsx':
            return xlsx_to_csv(fixtures_file)
        return fixtures_file
    
    fixtures_csv_path, err = run_step(7, "PREPARE FIXTURES", step7)
    
    # Step 8: Generate predictions
    def step8():
        predict_week(fixtures_csv_path)
    
    run_step(8, "GENERATE PREDICTIONS", step8)
    
    # ========================================================================
    # ENHANCEMENT STEPS
    # ========================================================================
    
    # Step 9: Optimize BTTS/O/U Predictions
    def step9():
        try:
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
                print(" BTTS/O/U optimized with historical patterns")
            else:
                raise FileNotFoundError("weekly_bets_lite.csv not found")
        except Exception as e:
            print(f" BTTS/OU optimization skipped: {e}")
            print("   (Continuing with standard predictions)")
    
    run_step(9, "OPTIMIZE BTTS/O/U", step9)
    
    # Step 10: Smart Ensemble Blending
    def step10():
        try:
            from ensemble_blender import create_superblend_predictions
            input_file = OUTPUT_DIR / "weekly_bets_lite.csv"
            output_file = OUTPUT_DIR / "weekly_bets_lite_blended.csv"
            
            if input_file.exists():
                create_superblend_predictions(input_file, output_file)
                
                import shutil
                shutil.copy(output_file, input_file)
                print(" Ensemble weights optimized based on performance")
            else:
                raise FileNotFoundError("weekly_bets_lite.csv not found")
        except Exception as e:
            print(f" Ensemble blending skipped: {e}")
            print("   (Continuing with standard blending)")
    
    run_step(10, "SMART ENSEMBLE BLENDING", step10)
    
    # Step 11: Log predictions
    def step11():
        from accuracy_tracker import log_weekly_predictions
        week_id = dt.datetime.now().strftime('%Y-W%W')
        csv_path = OUTPUT_DIR / "weekly_bets_lite.csv"
        
        if csv_path.exists():
            log_weekly_predictions(csv_path, week_id)
            print(f" Logged predictions (Week {week_id})")
        else:
            raise FileNotFoundError("weekly_bets_lite.csv not found")
    
    run_step(11, "LOG PREDICTIONS", step11)
    
    # Step 12: Generate weighted Top 50
    def step12():
        from weighted_top50 import generate_weighted_top50
        csv_path = OUTPUT_DIR / "weekly_bets_lite.csv"
        
        if csv_path.exists():
            generate_weighted_top50(csv_path)
            print(" Weighted Top 50 generated")
        else:
            raise FileNotFoundError("weekly_bets_lite.csv not found")
    
    run_step(12, "GENERATE WEIGHTED TOP 50", step12)
    
    # Step 13: O/U Analysis
    def step13():
        from ou_analyzer import analyze_ou_predictions
        weekly_bets_FILE = OUTPUT_DIR / "weekly_bets_lite.csv"

        if weekly_bets_FILE.exists():
            df_ou = analyze_ou_predictions(min_confidence=0.65)

            if not df_ou.empty:
                print(f" O/U Analysis: {len(df_ou)} predictions")
                elite = len(df_ou[df_ou['Best_Prob'] >= 0.85])
                high = len(df_ou[df_ou['Best_Prob'] >= 0.75])
                print(f"   Elite (85%+): {elite}, High (75%+): {high}")
            else:
                print(" No high-confidence O/U predictions")
        else:
            raise FileNotFoundError("weekly_bets_lite.csv not found")
    
    run_step(13, "O/U ANALYSIS", step13)
    
    # Step 14: Build Accumulators
    def step14():
        from accumulator_finder import AccumulatorFinder
        csv_path = OUTPUT_DIR / "weekly_bets_lite.csv"
        
        if csv_path.exists():
            finder = AccumulatorFinder(csv_path)
            
            strategies = {
                'safe': {'min_prob': 0.75, 'min_legs': 4, 'max_legs': 4},
                'mixed': {'min_prob': 0.70, 'min_legs': 5, 'max_legs': 5},
                'aggressive': {'min_prob': 0.65, 'min_legs': 6, 'max_legs': 6}
            }
            
            acca_count = 0
            for strategy_name, params in strategies.items():
                accumulators = finder.build_accumulators(
                    min_legs=params['min_legs'],
                    max_legs=params['max_legs'],
                    min_prob=params['min_prob']
                )
                acca_path = OUTPUT_DIR / f"accumulators_{strategy_name}.html"
                finder.generate_html_report(accumulators, acca_path)
                acca_count += 1
            
            print(f" Generated {acca_count} accumulator strategies")
        else:
            raise FileNotFoundError("weekly_bets_lite.csv not found")
    
    run_step(14, "BUILD ACCUMULATORS", step14)
    
    # Step 15: Find Quality Bets (ALL MARKETS)
    def step15():
        from bet_finder import BetFinder
        csv_path = OUTPUT_DIR / "weekly_bets_lite.csv"

        if csv_path.exists():
            # Create bet finder instance
            finder = BetFinder()

            # Load data
            if finder.load_data(csv_path):
                # Find quality bets
                finder.find_quality_bets()

                # Generate report
                if finder.quality_bets:
                    finder.generate_report()
                    print(f" Quality bets report generated: {len(finder.quality_bets)} opportunities found")
                else:
                    print(" No quality bets found with current criteria")
        else:
            raise FileNotFoundError("weekly_bets_lite.csv not found")
    
    run_step(15, "FIND QUALITY BETS (ALL MARKETS)", step15)
    
    # Step 16: Log predictions for accuracy tracking
    def step16():
        from accuracy_tracker import log_weekly_predictions
        csv_path = OUTPUT_DIR / "weekly_bets_lite.csv"

        if csv_path.exists():
            log_weekly_predictions(csv_path)
            print(" Predictions logged for accuracy tracking")
        else:
            print(" weekly_bets_lite.csv not found, skipping accuracy tracking")
    
    run_step(16, "UPDATE ACCURACY DATABASE", step16)

    # Step 17: Generate Excel Report and Send Email
    def step17():
        from excel_generator import generate_excel_report
        from email_sender import send_weekly_predictions

        csv_path = OUTPUT_DIR / "weekly_bets.csv"

        if csv_path.exists():
            # Generate Excel file
            excel_file = generate_excel_report(csv_path, OUTPUT_DIR)

            # Try to send via email (optional - won't fail if not configured)
            html_report = OUTPUT_DIR / "top50_weighted.html"
            if not html_report.exists():
                html_report = None

            send_weekly_predictions(excel_file, html_report)

            print(f" Excel report generated: {excel_file.name}")
        else:
            raise FileNotFoundError("weekly_bets.csv not found")

    run_step(17, "GENERATE EXCEL & SEND EMAIL", step17)

    # Step 18: Archive outputs
    def step18():
        """Copy all output files to dated archive folder"""
        import shutil
        from datetime import datetime
        
        # Create archive folder with date
        date_str = datetime.now().strftime('%Y-%m-%d')
        archive_dir = Path("archives") / date_str
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Files to archive
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
        
        # Also archive quality bets files (with wildcard pattern)
        import glob
        quality_bet_files = glob.glob(str(OUTPUT_DIR / "quality_bets_*.csv"))
        quality_bet_files += glob.glob(str(OUTPUT_DIR / "quality_bets_*.html"))
        
        archived_count = 0
        for filename in files_to_archive:
            source = OUTPUT_DIR / filename
            if source.exists():
                # Add date to filename
                name_parts = filename.rsplit('.', 1)
                if len(name_parts) == 2:
                    archived_name = f"{name_parts[0]}_{date_str}.{name_parts[1]}"
                else:
                    archived_name = f"{filename}_{date_str}"
                
                dest = archive_dir / archived_name
                shutil.copy2(source, dest)
                archived_count += 1
        
        # Archive quality bets files
        for filepath in quality_bet_files:
            filename = Path(filepath).name
            dest = archive_dir / filename
            shutil.copy2(filepath, dest)
            archived_count += 1
        
        print(f" Archived {archived_count} files to {archive_dir}")

    run_step(18, "ARCHIVE OUTPUTS", step18)
    
    # ========================================================================
    # SUCCESS SUMMARY
    # ========================================================================
    
    print("\n" + "="*60)
    print(" PIPELINE COMPLETE!")
    print("="*60)
    
    # Show any errors that occurred
    if errors:
        print(f"\n {len(errors)} step(s) had errors:")
        for error in errors:
            print(f"   • {error}")
        print("\n Check outputs folder - some files may still be generated")
    else:
        print("\n All steps completed successfully!")
    
    print("\n Main Files:")
    print("   • weekly_bets_lite.csv - All predictions")
    print("   • top50_weighted.html - Top picks (weighted)")
    
    print("\n Specialized Reports:")
    print("   • ou_analysis.html - Over/Under analysis")
    print("   • accumulators_safe.html - Conservative 4-fold")
    print("   • accumulators_mixed.html - Balanced 5-fold")
    print("   • accumulators_aggressive.html - High-risk 6-fold")
    print(f"   • quality_bets_{datetime.datetime.now().strftime('%Y%m%d')}.html - Quality bets (ALL MARKETS)")
    
    print("\n Tracking:")
    print("   • accuracy_database.db - Performance database")
    
    print("\n" + "="*60)
    print(" Next Steps:")
    print("   1. Review top50_weighted.html for best individual bets")
    print("   2. Check quality_bets HTML for opportunities across ALL markets")
    print("   3. Check ou_analysis.html for O/U opportunities")
    print("   4. Review accumulator files for multi-leg options")
    print("   5. After matches: run update_results.py")
    print("="*60)
    
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

except ImportError as e:
    print(f" Missing required file: {e}")
    print(" Ensure all Python files are in the project folder")
    print(" Run: pip install -r requirements.txt")

except Exception as e:
    print(f" Error: {e}")
    import traceback
    traceback.print_exc()
