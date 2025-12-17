@echo off
REM Test new streamlined outputs with sample data

echo ============================================================
echo TESTING NEW OUTPUTS
echo ============================================================
echo.
echo This will:
echo   1. Create sample prediction data
echo   2. Generate high_confidence_bets.html/csv
echo   3. Generate ou_accumulators.html/csv
echo   4. Open the HTML files in your browser
echo.
echo ============================================================
echo.

REM Run the test
python test_outputs_sample.py

echo.
echo ============================================================
echo Opening HTML files in browser...
echo ============================================================

REM Open the HTML files
start outputs\high_confidence_bets.html
start outputs\ou_accumulators.html

echo.
echo Done! Check the browser windows.
echo.
pause
