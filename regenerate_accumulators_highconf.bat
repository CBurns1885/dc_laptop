@echo off
echo ============================================================
echo Regenerating O/U Accumulators from High Confidence Bets
echo ============================================================
echo.

py -c "from generate_outputs_from_actual import generate_ou_accumulators; generate_ou_accumulators('outputs/high_confidence_bets.csv')"

echo.
echo ============================================================
echo Done! Check outputs folder for:
echo   - ou_accumulators.csv
echo   - ou_accumulators.html
echo ============================================================
pause
