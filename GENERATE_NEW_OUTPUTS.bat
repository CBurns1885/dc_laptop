@echo off
REM Generate new outputs from existing predictions

echo ============================================================
echo GENERATING NEW OUTPUTS
echo ============================================================
echo.

REM Copy latest predictions to outputs folder
echo Copying latest predictions...
copy "outputs\2025-12-08\weekly_bets_lite.csv" "outputs\weekly_bets_lite.csv" >nul

echo.
echo Generating outputs...
py generate_outputs_from_actual.py

echo.
echo ============================================================
echo Opening in browser...
echo ============================================================

start outputs\high_confidence_bets.html
timeout /t 1 /nobreak >nul
start outputs\ou_accumulators.html

echo.
echo Done! Check your browser.
pause
