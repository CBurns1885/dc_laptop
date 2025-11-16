# setup_structure.py
from pathlib import Path

def setup(base: Path):
    for sub in ["data/raw","data/processed","models","outputs","logs"]:
        (base / sub).mkdir(parents=True, exist_ok=True)
    cfg = f"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_ARTIFACTS_DIR = BASE_DIR / "models"
FEATURES_PARQUET = DATA_DIR / "processed" / "features.parquet"
RANDOM_SEED = 42

def log_header(msg: str):
    print("\\n" + "="*60 + f"\\n{{msg}}\\n" + "="*60)
"""
    (base / "config.py").write_text(cfg.strip()+"\n", encoding="utf-8")
    print("Structure created and config.py written.")

if __name__ == "__main__":
    setup(Path(__file__).resolve().parent)
