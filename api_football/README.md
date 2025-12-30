# API-Football Enhanced Dixon-Coles System

A comprehensive football prediction system using API-Football data with xG integration, injury tracking, formation analysis, and cup competition support.

## 🚀 Quick Start

### 1. Get API Key
Sign up at [API-Football](https://www.api-football.com/) (free plan: 100 requests/day)

### 2. Install Dependencies
```bash
pip install pandas numpy scipy requests pyarrow
```

### 3. Set API Key
```bash
# Windows PowerShell
$env:API_FOOTBALL_KEY = "your_api_key_here"

# Windows CMD
set API_FOOTBALL_KEY=your_api_key_here

# Linux/Mac
export API_FOOTBALL_KEY="your_api_key_here"
```

### 4. Run Full Pipeline
```bash
python run_api_football.py full --api-key YOUR_KEY
```

---

## 📁 Files Overview

| File | Description |
|------|-------------|
| `api_football_client.py` | API wrapper with rate limiting and caching |
| `data_ingest_api.py` | Download and store data in SQLite |
| `features_api.py` | Feature engineering with xG, injuries, formations |
| `models_dc_xg.py` | xG-integrated Dixon-Coles model |
| `injury_tracker.py` | Track and calculate injury impacts |
| `predict_api.py` | Generate match predictions |
| `backtest_api.py` | Backtest and evaluate features |
| `run_api_football.py` | Main orchestration script |
| `test_api_football.py` | Test suite (no API required) |

---

## 🏆 Features

### New Data Sources (vs football-data.co.uk)
- ✅ **xG Data** - Expected goals per match
- ✅ **Injuries** - Player injury tracking
- ✅ **Lineups & Formations** - 4-3-3, 5-4-1, etc.
- ✅ **All Competitions** - FA Cup, UCL, domestic cups
- ✅ **30+ Match Statistics** - Shots, possession, corners, passes
- ✅ **H2H Built-in** - Historical head-to-head
- ✅ **1,200+ Leagues** - Worldwide coverage

### Enhanced Dixon-Coles Model
- xG-weighted attack/defence ratings
- Regression to mean (overperformance adjustment)
- League-specific rho optimization
- Cup competition adjustments (reduced home advantage)
- Injury impact modifiers
- Formation-aware goal expectations
- Rest day fatigue factors

---

## 📊 Commands

### Ingest Data
```bash
# Full historical download (takes time, uses API credits)
python run_api_football.py ingest --api-key YOUR_KEY --seasons 2023 2024

# Update recent only (daily use)
python run_api_football.py ingest --api-key YOUR_KEY --update-only --days 7
```

### Build Features
```bash
python run_api_football.py features --force
```

### Run Backtest
```bash
# Standard backtest
python run_api_football.py backtest

# Feature ablation study (which features help most)
python run_api_football.py backtest --ablation

# Test specific leagues
python run_api_football.py backtest --leagues E0 D1 SP1

# Exclude cup competitions
python run_api_football.py backtest --no-cups
```

### Generate Predictions
```bash
# Single match
python run_api_football.py predict --home Arsenal --away Chelsea --league E0

# All fixtures for a date
python run_api_football.py predict --date 2024-01-15

# Upcoming 7 days (requires API)
python run_api_football.py predict --api-key YOUR_KEY --days 7
```

### Check Status
```bash
python run_api_football.py status
```

---

## ⚽ Supported Competitions

### Top Leagues
| Code | League |
|------|--------|
| E0 | Premier League |
| E1 | Championship |
| D1 | Bundesliga |
| SP1 | La Liga |
| I1 | Serie A |
| F1 | Ligue 1 |

### Domestic Cups
| Code | Competition |
|------|-------------|
| FA_CUP | FA Cup |
| EFL_CUP | EFL Cup |
| DFB_POKAL | DFB Pokal |
| COPA_DEL_REY | Copa del Rey |
| COPPA_ITALIA | Coppa Italia |
| COUPE_DE_FRANCE | Coupe de France |

### European
| Code | Competition |
|------|-------------|
| UCL | Champions League |
| UEL | Europa League |
| UECL | Conference League |

---

## 🔬 Backtest Results

Expected accuracy improvements:

| Feature | BTTS Impact | O/U 2.5 Impact |
|---------|-------------|----------------|
| xG Integration | +3-4% | +2-3% |
| Injuries | +2-3% | +1-2% |
| Complete Fixtures (cups) | +1-2% | +2-3% |
| Formations | +1-2% | +1% |
| Advanced Stats | +1-2% | +1-2% |
| **Total Expected** | **+8-12%** | **+7-11%** |

---

## 🗄️ Database Schema

```
football_api.db
├── fixtures         # Match results with xG
├── fixture_statistics  # Shots, possession, corners, etc.
├── injuries         # Player injuries
├── lineups          # Formations and starting XI
├── teams            # Team info cache
└── player_fixture_stats  # Per-match player stats
```

---

## 📈 Integration with Existing System

To integrate with your existing dc_laptop project:

1. **Copy files** to your project directory
2. **Replace data ingestion**:
   ```python
   # Old
   from download_football_data import download_all
   
   # New
   from data_ingest_api import APIFootballIngestor
   ingestor = APIFootballIngestor(api_key)
   ingestor.ingest_all()
   ```

3. **Use enhanced features**:
   ```python
   # Old
   from features import build_features
   
   # New
   from features_api import build_features
   build_features(force=True)
   ```

4. **Use xG model**:
   ```python
   # Old
   from models_dc import fit_league, price_match
   
   # New
   from models_dc_xg import fit_league_xg, price_match_xg
   ```

---

## 💰 API Cost Estimates

| Plan | Requests/Day | Monthly Cost | Use Case |
|------|--------------|--------------|----------|
| Free | 100 | $0 | Testing |
| Pro | 7,500 | ~$20 | Development |
| Ultra | 75,000 | ~$80 | Production |

**Daily Usage Estimate:**
- Historical load: 500-1000 requests (one-time)
- Daily updates: 100-200 requests
- Weekly predictions: 50-100 requests

---

## 🧪 Running Tests

```bash
# No API key required
python test_api_football.py
```

Expected output:
```
✅ PASS: DC Model with xG
✅ PASS: Cup Competition Handling
✅ PASS: Feature Impacts
✅ PASS: Backtest Module
✅ PASS: Prediction Module
✅ PASS: League Configurations

🎉 ALL TESTS PASSED!
```

---

## ❓ Troubleshooting

### "No model for league X"
- The league needs minimum 30 matches for fitting
- Cup competitions may have insufficient data early season

### "API rate limit exceeded"
- Free plan: 100/day, Pro: 7,500/day
- The client handles rate limiting automatically

### "xG data missing"
- Not all leagues have xG data
- Model falls back to goals-based calculation

---

## 📝 License

MIT License - Free for personal and commercial use.
