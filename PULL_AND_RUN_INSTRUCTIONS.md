# 📥 How to Pull and Run - DC-Only Football Prediction System

## ✅ FINAL VERIFICATION COMPLETE

**System Status:** ✅ DC-ONLY confirmed
- **Markets:** BTTS + Over/Under (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)
- **Model:** Dixon-Coles statistical model ONLY
- **No ML models:** All ensemble models removed
- **All syntax valid:** All Python files compile successfully

---

## 📥 Step 1: Pull from GitHub

### On Your Laptop:

```bash
# Navigate to your project folder (or create one)
cd ~/Desktop

# Clone the repository (first time only)
git clone https://github.com/CBurns1885/dc_laptop.git

# Navigate into the folder
cd dc_laptop

# Checkout the DC-only branch
git checkout claude/remove-non-goal-markets-01AGqXkzhXTY2F79YrFonzB5

# Pull latest changes
git pull origin claude/remove-non-goal-markets-01AGqXkzhXTY2F79YrFonzB5
```

### If You Already Have It:

```bash
# Navigate to your existing folder
cd ~/Desktop/dc_laptop

# Pull latest changes
git pull origin claude/remove-non-goal-markets-01AGqXkzhXTY2F79YrFonzB5
```

---

## 📦 Step 2: Install Dependencies

```bash
# Install required Python packages
pip3 install numpy pandas scipy scikit-learn joblib requests openpyxl

# Or use the minimal DC-only set:
pip3 install numpy pandas scipy requests openpyxl
```

**Note:** You don't need XGBoost, LightGBM, CatBoost, or PyTorch for DC-only!

---

## ⚙️ Step 3: Configure Email (Optional)

Edit `run_weekly.py` lines 20-24 OR add to your shell:

### For Outlook:
```bash
export EMAIL_SMTP_SERVER='smtp-mail.outlook.com'
export EMAIL_SMTP_PORT='587'
export EMAIL_SENDER='your-email@outlook.com'
export EMAIL_PASSWORD='your-password'
export EMAIL_RECIPIENT='your-email@outlook.com'
```

### For Gmail:
```bash
export EMAIL_SMTP_SERVER='smtp.gmail.com'
export EMAIL_SMTP_PORT='587'
export EMAIL_SENDER='your-email@gmail.com'
export EMAIL_PASSWORD='your-app-password'  # Generate at myaccount.google.com/apppasswords
export EMAIL_RECIPIENT='your-email@gmail.com'
```

**Skip this if you just want the Excel file without email.**

---

## 🚀 Step 4: First Run Setup

```bash
# Download historical data (takes ~5 minutes)
python3 download_football_data.py

# Build features with enhancements (takes ~10 minutes)
python3 features.py --force

# Train Dixon-Coles models (takes ~5 minutes)
python3 models.py
```

---

## 🎯 Step 5: Run Weekly Predictions

```bash
# Run the full weekly workflow
python3 run_weekly.py
```

This will:
1. Update accuracy from last week
2. Download new fixtures
3. Train Dixon-Coles models
4. Generate predictions
5. Find quality bets
6. **Create Excel workbook** with:
   - Sheet 1: Top 10 from each market
   - Sheets 2-8: All predictions per market
7. **Send email** (if configured)
8. Archive outputs

---

## 📊 Step 6: Check Outputs

Your predictions are in: `outputs/predictions_YYYY-MM-DD.xlsx`

**Excel Structure:**
- **Sheet 1:** Top 10 All Markets (best 10 from each)
- **Sheet 2:** BTTS (all predictions, both Yes/No)
- **Sheet 3:** OU 0.5 (all predictions, Over/Under)
- **Sheet 4:** OU 1.5 (all predictions, Over/Under)
- **Sheet 5:** OU 2.5 (all predictions, Over/Under)
- **Sheet 6:** OU 3.5 (all predictions, Over/Under)
- **Sheet 7:** OU 4.5 (all predictions, Over/Under)
- **Sheet 8:** OU 5.5 (all predictions, Over/Under)

---

## 🔬 Optional: Run Backtest

Test historical performance:

```bash
python3 backtest.py
```

Outputs:
- `outputs/backtest_summary.csv` - Overall accuracy by market
- `outputs/backtest_calibration.csv` - **Tiered accuracy analysis**
- `outputs/backtest_detailed.csv` - Period-by-period breakdown
- `outputs/backtest_best_doubles.csv` - Best 2-leg combinations
- `outputs/backtest_best_trebles.csv` - Best 3-leg combinations

---

## ✅ DC-ONLY CONFIRMATION

### What's Using Dixon-Coles:

✅ **models_dc.py** - Core DC implementation
- Fits attack/defence parameters per league
- Applies rest days penalty (-12% for <4 days)
- Applies seasonal patterns (+8-12% early season)
- Calculates BTTS and O/U probabilities

✅ **models.py** - Training orchestration
- Calls `dc_fit_all()` to fit DC parameters
- Only supports BTTS and O/U targets
- Skips any non-DC targets

✅ **predict.py** - Prediction generation
- Uses DC probabilities directly
- No ML ensemble blending
- Outputs P_BTTS_Y, P_OU_2_5_O, etc.

✅ **bet_finder.py** - Quality bet finder
- Only checks BTTS and O/U markets
- No 1X2, Asian Handicap, or Correct Score

### What's NOT Being Used:

❌ RandomForest, XGBoost, LightGBM, CatBoost
❌ Ensemble stacking/blending
❌ Calibration models (DC is inherently calibrated)
❌ 1X2, Asian Handicap, Correct Score markets
❌ Any ML training or tuning

**The ML imports exist in code for backwards compatibility, but they are NOT executed.**

---

## 🛠️ Troubleshooting

### "No module named numpy"
```bash
pip3 install numpy pandas scipy
```

### "Features file not found"
```bash
python3 features.py --force
```

### "No fixtures downloaded"
```bash
python3 download_fixtures.py
```

### Email not sending
- Check environment variables are set
- For Gmail: Use App Password, not your normal password
- For Outlook: Check your password is correct
- Email is optional - Excel file always generated

---

## 📞 Support

If you get errors:
1. Check Python version: `python3 --version` (need 3.8+)
2. Check dependencies: `pip3 list | grep -E "numpy|pandas|scipy"`
3. Check branch: `git branch` (should show `claude/remove-non-goal-markets-...`)

---

## 🎯 Summary

**One-time setup:**
```bash
git clone https://github.com/CBurns1885/dc_laptop.git
cd dc_laptop
git checkout claude/remove-non-goal-markets-01AGqXkzhXTY2F79YrFonzB5
pip3 install numpy pandas scipy requests openpyxl
python3 download_football_data.py
python3 features.py --force
python3 models.py
```

**Every Friday:**
```bash
cd dc_laptop
python3 run_weekly.py
```

**Check Excel file:**
```bash
open outputs/predictions_*.xlsx
```

That's it! 🎉

---

**System is 100% Dixon-Coles only. All tests passed. Ready to run!**
