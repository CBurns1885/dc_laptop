@echo off
REM Setup Claude API Key
REM This sets the API key permanently in your user environment

echo ============================================================
echo SETTING UP CLAUDE API KEY
echo ============================================================
echo.

REM Set the API key as a user environment variable (permanent)
REM Replace YOUR-API-KEY-HERE with your actual API key
setx ANTHROPIC_API_KEY "YOUR-API-KEY-HERE"

echo.
echo ============================================================
echo API KEY SETUP COMPLETE
echo ============================================================
echo.
echo The API key has been set permanently in your user environment.
echo.
echo IMPORTANT: You need to close and reopen your terminal/command prompt
echo for the changes to take effect.
echo.
echo To verify:
echo   1. Close this window
echo   2. Open a new command prompt
echo   3. Run: echo %%ANTHROPIC_API_KEY%%
echo   4. You should see your API key
echo.
echo Next steps:
echo   1. Close and reopen terminal
echo   2. Run: python test_claude_api.py
echo   3. Run: python run_weekly.py
echo.
echo ============================================================

pause
