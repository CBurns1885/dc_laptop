@echo off
REM Test all recent updates
REM Run this after backtest and weekly pipeline

echo ============================================================
echo TESTING ALL UPDATES
echo ============================================================
echo.

REM Run the comprehensive test
python test_all_updates.py

echo.
echo ============================================================
echo TEST COMPLETE
echo ============================================================
echo.
echo Next steps:
echo   1. If backtest tests failed: Run "python backtest_config.py"
echo   2. If enrichment/market tests failed: Run "python run_weekly.py"
echo   3. Check outputs folder for generated files
echo.

pause
