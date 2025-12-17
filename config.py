# config.py - FIXED VERSION with Dated Output Folders
from pathlib import Path
import os
from datetime import date, datetime

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# NEW: Dated output directories
def get_dated_output_dir():
    """Creates a dated folder for outputs (e.g., outputs/2025-10-24)"""
    date_str = datetime.now().strftime("%Y-%m-%d")
    dated_dir = BASE_DIR / "outputs" / date_str
    dated_dir.mkdir(parents=True, exist_ok=True)
    return dated_dir

# Main output directory - NOW DATED!
OUTPUT_DIR = get_dated_output_dir()
MODEL_ARTIFACTS_DIR = BASE_DIR / "models"
MODELS_DIR = MODEL_ARTIFACTS_DIR  # Alias for compatibility

# API Keys
FOOTBALL_DATA_ORG_TOKEN = "0f17fdba78d15a625710f7244a1cc770"

# Ensure base directories exist
for p in [DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODEL_ARTIFACTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# --- Key File Paths ---
FEATURES_PARQUET = DATA_DIR / "processed" / "features.parquet"
HISTORICAL_PARQUET = PROCESSED_DIR / "historical_matches.parquet"
WEEKLY_OUTPUT_CSV = OUTPUT_DIR / "weekly_bets_lite.csv"
BLEND_WEIGHTS_JSON = MODEL_ARTIFACTS_DIR / "blend_weights.json"

# --- Dates ---
CURRENT_YEAR = date.today().year
CURRENT_MONTH = date.today().month
ACTIVE_SEASON_START = CURRENT_YEAR if CURRENT_MONTH >= 7 else CURRENT_YEAR - 1

SEASON_START_YEAR = int(os.environ.get("FOOTY_SEASON_START_YEAR", 2022))
SEASONS = [f"{str(y)[-2:]}{str(y+1)[-2:]}" for y in range(SEASON_START_YEAR, ACTIVE_SEASON_START + 1)]

# --- Coverage ---
LEAGUE_CODES = [
    "E0", "E1", "E2", "E3", "EC",      # England
    "D1", "D2",                         # Germany
    "SP1", "SP2",                       # Spain
    "I1", "I2",                         # Italy
    "F1", "F2",                         # France
    "N1",                               # Netherlands
    "B1",                               # Belgium
    "P1",                               # Portugal
    "G1",                               # Greece
    "SC0", "SC1", "SC2", "SC3",        # Scotland
    "T1",                               # Turkey
]

# --- Data sources ---
FOOTBALL_DATA_CSV_BASE = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

# Optional football-data.org (token required)  
FOOTBALL_DATA_ORG_TOKEN = os.environ.get("FOOTBALL_DATA_ORG_TOKEN", "").strip()
FOOTBALL_DATA_ORG_BASE = "https://api.football-data.org/v4"

# --- Randomness / Reproducibility ---
RANDOM_SEED = 42
GLOBAL_SEED = RANDOM_SEED

# --- Modeling - DC-ONLY with BTTS and O/U (0.5-5.5) ---
PRIMARY_TARGETS = ["BTTS", "OU"]  # Dixon-Coles only: BTTS and Over/Under
USE_ELO = True
USE_ROLLING_FORM = True
USE_MARKET_FEATURES = False  # DC doesn't need market odds features

TRAIN_SEASONS_BACK = int(os.environ.get("FOOTY_TRAIN_SEASONS_BACK", 8))

def season_code(year_start: int) -> str:
    return f"{str(year_start)[-2:]}{str(year_start + 1)[-2:]}"

def log_header(msg: str) -> None:
    bar = "=" * max(20, len(msg) + 4)
    print(f"\n{bar}")
    print(f"  {msg}")
    print(f"{bar}")

# Email configuration for Outlook
EMAIL_SMTP_SERVER = os.environ.get("EMAIL_SMTP_SERVER", "smtp-mail.outlook.com")
EMAIL_SMTP_PORT = int(os.environ.get("EMAIL_SMTP_PORT", "587"))
EMAIL_SENDER = os.environ.get("EMAIL_SENDER", "")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")
EMAIL_RECIPIENT = os.environ.get("EMAIL_RECIPIENT", "")

print(f"Output directory: {OUTPUT_DIR}")