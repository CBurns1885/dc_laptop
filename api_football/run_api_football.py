# run_api_football.py
"""
Main Orchestration Script for API-Football Enhanced System

Commands:
    python run_api_football.py ingest       # Download data from API
    python run_api_football.py features     # Build feature set
    python run_api_football.py backtest     # Run backtests
    python run_api_football.py predict      # Generate predictions
    python run_api_football.py full         # Run full pipeline
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"
RESULTS_DIR = BASE_DIR / "backtest_results"

# Ensure directories exist
for d in [DATA_DIR, PROCESSED_DIR, OUTPUT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# COMMANDS
# =============================================================================

def cmd_ingest(args):
    """Download and ingest data from API-Football"""
    from data_ingest_api import APIFootballIngestor, DEFAULT_LEAGUES, DEFAULT_SEASONS
    
    if not args.api_key:
        logger.error("API key required. Set API_FOOTBALL_KEY or use --api-key")
        return 1
    
    ingestor = APIFootballIngestor(args.api_key)
    
    leagues = args.leagues or DEFAULT_LEAGUES
    seasons = args.seasons or DEFAULT_SEASONS
    
    logger.info("="*60)
    logger.info("DATA INGESTION")
    logger.info("="*60)
    logger.info(f"Leagues: {leagues}")
    logger.info(f"Seasons: {seasons}")
    
    if args.update_only:
        # Just update recent matches
        from data_ingest_api import update_recent
        output = update_recent(args.api_key, days=args.days or 7)
    else:
        ingestor.ingest_all(
            leagues=leagues,
            seasons=seasons,
            include_stats=not args.no_stats,
            include_injuries=not args.no_injuries
        )
        output = ingestor.export_to_parquet()
    
    logger.info(f"Output: {output}")
    return 0


def cmd_features(args):
    """Build feature set from ingested data"""
    from features_api import build_features, FEATURES_PARQUET
    
    logger.info("="*60)
    logger.info("FEATURE ENGINEERING")
    logger.info("="*60)
    
    output = build_features(
        force=args.force,
        from_parquet=args.from_parquet
    )
    
    logger.info(f"Output: {output}")
    return 0


def cmd_backtest(args):
    """Run backtests"""
    from backtest_api import Backtester, BacktestConfig, print_results
    
    logger.info("="*60)
    logger.info("BACKTESTING")
    logger.info("="*60)
    
    config = BacktestConfig(
        name=args.name or "backtest",
        leagues=args.leagues,
        include_cups=not args.no_cups,
        use_xg=not args.no_xg,
        use_injuries=not args.no_injuries,
        use_formations=not args.no_formations,
        use_h2h=not args.no_h2h,
    )
    
    backtester = Backtester(config)
    
    if args.ablation:
        # Run feature ablation study
        df = backtester.load_data()
        ablation_df = backtester.run_feature_ablation(df)
        
        print("\n" + "="*70)
        print("FEATURE ABLATION RESULTS")
        print("="*70)
        print(ablation_df.to_string())
        
        output_path = RESULTS_DIR / f"ablation_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        ablation_df.to_csv(output_path, index=False)
        logger.info(f"Saved to {output_path}")
    else:
        # Standard backtest
        predictions = backtester.run()
        results = backtester.evaluate(predictions)
        print_results(results)
        
        output_path = RESULTS_DIR / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        predictions.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
    
    return 0


def cmd_predict(args):
    """Generate predictions"""
    from predict_api import MatchPredictor, format_prediction
    import pandas as pd
    
    logger.info("="*60)
    logger.info("PREDICTIONS")
    logger.info("="*60)
    
    predictor = MatchPredictor(api_key=args.api_key)
    
    if args.home and args.away:
        # Single match
        pred = predictor.predict_fixture(
            home=args.home,
            away=args.away,
            league=args.league or 'E0',
            match_date=pd.to_datetime(args.date) if args.date else None
        )
        print(format_prediction(pred))
        
    elif args.date:
        # All fixtures for date
        preds = predictor.predict_date(args.date)
        for pred in preds:
            print(format_prediction(pred))
        
        if preds:
            df = pd.DataFrame(preds)
            output = predictor.save_predictions(df)
            logger.info(f"Saved to {output}")
    
    elif args.api_key:
        # Upcoming from API
        df = predictor.predict_upcoming(
            days=args.days or 7,
            leagues=args.leagues
        )
        
        if not df.empty:
            # Show top predictions
            print("\n" + "="*70)
            print("TOP BTTS YES PREDICTIONS")
            print("="*70)
            top_btts = df.nlargest(15, 'btts_yes')
            for _, row in top_btts.iterrows():
                cup_str = " (CUP)" if row.get('is_cup') else ""
                print(f"  {row['home']} vs {row['away']}: {row['btts_yes']:.1%} [{row['league']}{cup_str}]")
            
            print("\n" + "="*70)
            print("TOP OVER 2.5 PREDICTIONS")
            print("="*70)
            top_over = df.nlargest(15, 'over_2_5')
            for _, row in top_over.iterrows():
                cup_str = " (CUP)" if row.get('is_cup') else ""
                print(f"  {row['home']} vs {row['away']}: {row['over_2_5']:.1%} [{row['league']}{cup_str}]")
            
            output = predictor.save_predictions(df)
            logger.info(f"Saved {len(df)} predictions to {output}")
        else:
            print("No predictions generated")
    
    else:
        print("Use --home/--away, --date, or --api-key for predictions")
        return 1
    
    return 0


def cmd_full(args):
    """Run full pipeline"""
    logger.info("="*60)
    logger.info("FULL PIPELINE")
    logger.info("="*60)
    
    # Step 1: Ingest (if API key provided)
    if args.api_key and not args.skip_ingest:
        logger.info("\n[1/4] INGESTING DATA...")
        from data_ingest_api import APIFootballIngestor, DEFAULT_LEAGUES
        
        ingestor = APIFootballIngestor(args.api_key)
        
        if args.update_only:
            from data_ingest_api import update_recent
            update_recent(args.api_key, days=7)
        else:
            ingestor.ingest_all(
                leagues=args.leagues or DEFAULT_LEAGUES,
                seasons=args.seasons or [2023, 2024]
            )
            ingestor.export_to_parquet()
    else:
        logger.info("\n[1/4] SKIPPING INGESTION (no API key or --skip-ingest)")
    
    # Step 2: Features
    logger.info("\n[2/4] BUILDING FEATURES...")
    from features_api import build_features
    build_features(force=True)
    
    # Step 3: Backtest
    if not args.skip_backtest:
        logger.info("\n[3/4] RUNNING BACKTEST...")
        from backtest_api import Backtester, BacktestConfig, print_results
        
        config = BacktestConfig(include_cups=True)
        bt = Backtester(config)
        preds = bt.run()
        results = bt.evaluate(preds)
        print_results(results)
        
        preds.to_csv(RESULTS_DIR / "latest_backtest.csv", index=False)
    else:
        logger.info("\n[3/4] SKIPPING BACKTEST")
    
    # Step 4: Predictions
    if args.api_key:
        logger.info("\n[4/4] GENERATING PREDICTIONS...")
        from predict_api import MatchPredictor
        
        predictor = MatchPredictor(api_key=args.api_key)
        df = predictor.predict_upcoming(days=7)
        
        if not df.empty:
            predictor.save_predictions(df)
            logger.info(f"Generated {len(df)} predictions")
    else:
        logger.info("\n[4/4] SKIPPING PREDICTIONS (no API key)")
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*60)
    
    return 0


def cmd_calibrate(args):
    """Run calibration backtest"""
    from backtest_calibration import CalibrationBacktest, RESULTS_DIR
    from backtest_api import BacktestConfig
    
    logger.info("="*60)
    logger.info("CALIBRATION BACKTEST")
    logger.info("="*60)
    
    # Load features
    features_path = args.features or PROCESSED_DIR / "features.parquet"
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        return 1
    
    import pandas as pd
    df = pd.read_parquet(features_path)
    
    # Configure
    config = BacktestConfig(
        leagues=args.leagues,
        markets=args.markets or ['BTTS', 'OU_2_5', 'OU_1_5', 'OU_3_5']
    )
    
    # Run calibration backtest
    backtest = CalibrationBacktest(config)
    results = backtest.run(
        df,
        optimize_params=args.optimize,
        fit_calibrators=not args.no_calibrators,
        analyze_features=not args.no_features
    )
    
    # Print report
    backtest.print_report()
    
    # Export results
    output_dir = args.output or RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    backtest.export_full_results(output_dir / f"calibration_results_{timestamp}.json")
    backtest.export_python_config(output_dir / f"calibration_config_{timestamp}.py")
    
    if backtest.predictions_df is not None:
        backtest.predictions_df.to_csv(output_dir / f"calibration_predictions_{timestamp}.csv", index=False)
    
    logger.info(f"\nResults saved to {output_dir}")
    return 0


def cmd_status(args):
    """Check system status"""
    import sqlite3
    
    print("\n" + "="*60)
    print("SYSTEM STATUS")
    print("="*60)
    
    # Check database
    db_path = DATA_DIR / "football_api.db"
    if db_path.exists():
        with sqlite3.connect(db_path) as conn:
            fixtures = conn.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0]
            injuries = conn.execute("SELECT COUNT(*) FROM injuries").fetchone()[0]
            stats = conn.execute("SELECT COUNT(*) FROM fixture_statistics").fetchone()[0]
            
            date_range = conn.execute(
                "SELECT MIN(date), MAX(date) FROM fixtures WHERE status='FT'"
            ).fetchone()
            
            leagues = conn.execute(
                "SELECT league_code, COUNT(*) FROM fixtures GROUP BY league_code"
            ).fetchall()
        
        print(f"\nDatabase: {db_path}")
        print(f"  Fixtures: {fixtures:,}")
        print(f"  Statistics: {stats:,}")
        print(f"  Injuries: {injuries:,}")
        if date_range[0]:
            print(f"  Date range: {date_range[0]} to {date_range[1]}")
        
        print(f"\n  Leagues:")
        for league, count in sorted(leagues, key=lambda x: -x[1])[:10]:
            print(f"    {league}: {count:,}")
    else:
        print(f"\nDatabase: NOT FOUND ({db_path})")
    
    # Check features
    features_path = PROCESSED_DIR / "features.parquet"
    if features_path.exists():
        import pandas as pd
        df = pd.read_parquet(features_path)
        print(f"\nFeatures: {features_path}")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    else:
        print(f"\nFeatures: NOT FOUND ({features_path})")
    
    # Check API key
    api_key = os.environ.get("API_FOOTBALL_KEY")
    print(f"\nAPI Key: {'SET' if api_key else 'NOT SET'}")
    
    return 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="API-Football Enhanced Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline with API key
    python run_api_football.py full --api-key YOUR_KEY
    
    # Just update recent data and predict
    python run_api_football.py ingest --api-key YOUR_KEY --update-only
    python run_api_football.py predict --api-key YOUR_KEY
    
    # Run backtest with feature ablation
    python run_api_football.py backtest --ablation
    
    # Predict specific match
    python run_api_football.py predict --home Arsenal --away Chelsea --league E0
        """
    )
    
    parser.add_argument("--api-key", default=os.environ.get("API_FOOTBALL_KEY"),
                        help="API-Football key (or set API_FOOTBALL_KEY)")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest command
    ingest_p = subparsers.add_parser("ingest", help="Download data from API")
    ingest_p.add_argument("--leagues", nargs="+", help="Leagues to download")
    ingest_p.add_argument("--seasons", nargs="+", type=int, help="Seasons to download")
    ingest_p.add_argument("--update-only", action="store_true", help="Only update recent")
    ingest_p.add_argument("--days", type=int, default=7, help="Days to update")
    ingest_p.add_argument("--no-stats", action="store_true", help="Skip statistics")
    ingest_p.add_argument("--no-injuries", action="store_true", help="Skip injuries")
    
    # Features command
    features_p = subparsers.add_parser("features", help="Build features")
    features_p.add_argument("--force", action="store_true", help="Force rebuild")
    features_p.add_argument("--from-parquet", type=Path, help="Load from parquet")
    
    # Backtest command
    backtest_p = subparsers.add_parser("backtest", help="Run backtests")
    backtest_p.add_argument("--name", help="Backtest name")
    backtest_p.add_argument("--leagues", nargs="+", help="Leagues to test")
    backtest_p.add_argument("--no-cups", action="store_true", help="Exclude cups")
    backtest_p.add_argument("--no-xg", action="store_true", help="Disable xG features")
    backtest_p.add_argument("--no-injuries", action="store_true", help="Disable injuries")
    backtest_p.add_argument("--no-formations", action="store_true", help="Disable formations")
    backtest_p.add_argument("--no-h2h", action="store_true", help="Disable H2H")
    backtest_p.add_argument("--ablation", action="store_true", help="Run ablation study")
    
    # Predict command
    predict_p = subparsers.add_parser("predict", help="Generate predictions")
    predict_p.add_argument("--home", help="Home team")
    predict_p.add_argument("--away", help="Away team")
    predict_p.add_argument("--league", help="League code")
    predict_p.add_argument("--date", help="Match date (YYYY-MM-DD)")
    predict_p.add_argument("--days", type=int, default=7, help="Days ahead")
    predict_p.add_argument("--leagues", nargs="+", help="Leagues to predict")
    
    # Full pipeline command
    full_p = subparsers.add_parser("full", help="Run full pipeline")
    full_p.add_argument("--leagues", nargs="+", help="Leagues to include")
    full_p.add_argument("--seasons", nargs="+", type=int, help="Seasons")
    full_p.add_argument("--update-only", action="store_true", help="Only update recent")
    full_p.add_argument("--skip-ingest", action="store_true", help="Skip data ingestion")
    full_p.add_argument("--skip-backtest", action="store_true", help="Skip backtest")
    
    # Status command
    status_p = subparsers.add_parser("status", help="Check system status")
    
    # Calibration command
    calibrate_p = subparsers.add_parser("calibrate", help="Run calibration backtest")
    calibrate_p.add_argument("--features", type=Path, help="Features parquet file")
    calibrate_p.add_argument("--leagues", nargs="+", help="Leagues to analyze")
    calibrate_p.add_argument("--markets", nargs="+", help="Markets to analyze")
    calibrate_p.add_argument("--optimize", action="store_true", help="Run parameter optimization")
    calibrate_p.add_argument("--no-calibrators", action="store_true", help="Skip calibrator fitting")
    calibrate_p.add_argument("--no-features", action="store_true", help="Skip feature importance")
    calibrate_p.add_argument("--output", type=Path, help="Output directory")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    commands = {
        "ingest": cmd_ingest,
        "features": cmd_features,
        "backtest": cmd_backtest,
        "predict": cmd_predict,
        "full": cmd_full,
        "status": cmd_status,
        "calibrate": cmd_calibrate,
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
