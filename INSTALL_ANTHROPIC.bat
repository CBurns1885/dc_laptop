@echo off
REM Install anthropic package for Claude API

echo ============================================================
echo INSTALLING ANTHROPIC PACKAGE
echo ============================================================
echo.
echo This is needed for the days-since-last-match feature.
echo.

pip install anthropic

echo.
echo ============================================================
echo INSTALLATION COMPLETE
echo ============================================================
echo.
echo The anthropic package has been installed.
echo.
echo Next steps:
echo   1. Set your API key (if not already set):
echo      Run: setup_api_key.bat
echo   2. Test the API:
echo      Run: python test_claude_api.py
echo   3. Run the pipeline:
echo      Run: python run_weekly.py
echo.
pause
