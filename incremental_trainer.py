#incremental_trainer

# incremental_trainer.py
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from models import train_all_targets, load_trained_targets, _load_features
from config import MODEL_ARTIFACTS_DIR, log_header

def needs_retraining(models_dir: Path = MODEL_ARTIFACTS_DIR, days_threshold: int = 7) -> bool:
    """Check if models need retraining based on new data OR changed training settings"""
    if os.environ.get("FORCE_RETRAIN") == "1":
        print("Force retraining requested")
        return True
        
    if not models_dir.exists():
        print("No models directory found, training from scratch")
        return True
    
    # Check if training settings have changed
    settings_file = models_dir / "training_settings.json"

    # Get current leagues from features data
    try:
        df = _load_features()
        current_leagues = sorted(df['League'].unique().tolist()) if 'League' in df.columns else []
    except:
        current_leagues = []

    current_settings = {
        "optuna_trials": os.environ.get("OPTUNA_TRIALS", "0"),
        "n_estimators": os.environ.get("N_ESTIMATORS", "300"),
        "models_only": os.environ.get("MODELS_ONLY", ""),
        "leagues": current_leagues  # Track which leagues models were trained on
    }
    
    if settings_file.exists():
        try:
            old_settings = json.loads(settings_file.read_text())

            # Check if leagues changed
            old_leagues = set(old_settings.get("leagues", []))
            new_leagues = set(current_settings["leagues"])

            if old_leagues != new_leagues:
                print(f"Leagues changed: {old_leagues} → {new_leagues}")
                print("Retraining for new league set...")
                return True

            # Check if other settings changed
            settings_to_check = ["optuna_trials", "n_estimators", "models_only"]
            for key in settings_to_check:
                if old_settings.get(key) != current_settings.get(key):
                    print(f"Setting '{key}' changed: {old_settings.get(key)} → {current_settings.get(key)}")
                    print("Retraining...")
                    return True
        except Exception:
            print("Could not read previous training settings, retraining...")
            return True
    else:
        print("No previous training settings found, retraining...")
        return True
    
    # Check if manifest exists
    manifest_file = models_dir / "manifest.json"
    if not manifest_file.exists():
        print("No model manifest found, retraining...")
        return True
    
    # Check model age
    try:
        model_age = datetime.now() - datetime.fromtimestamp(manifest_file.stat().st_mtime)
        if model_age.days > days_threshold:
            print(f"Models are {model_age.days} days old, retraining...")
            return True
    except Exception:
        print("Could not check model age, retraining...")
        return True
    
    # Check for new data
    try:
        df = _load_features()
        if df.empty:
            print("No features data available")
            return True
        
        latest_data = df['Date'].max()
        cutoff_date = latest_data - timedelta(days=days_threshold)
        new_data_count = len(df[df['Date'] > cutoff_date])
        
        if new_data_count > 50:  # Adjust threshold as needed
            print(f"Found {new_data_count} new matches, retraining...")
            return True
    except Exception as e:
        print(f"Could not check for new data: {e}, retraining...")
        return True
    
    print(f"Models are compatible and recent, using existing models")
    return False

def smart_train_or_load():
    """Train models if needed, otherwise load existing ones"""
    if needs_retraining():
        log_header("TRAINING MODELS")
        models = train_all_targets()
        
        # Save training settings for future comparison
        try:
            settings = {
                "optuna_trials": os.environ.get("OPTUNA_TRIALS", "0"),
                "n_estimators": os.environ.get("N_ESTIMATORS", "300"),
                "models_only": os.environ.get("MODELS_ONLY", ""),
                "trained_at": datetime.now().isoformat()
            }
            settings_file = MODEL_ARTIFACTS_DIR / "training_settings.json"
            MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            settings_file.write_text(json.dumps(settings, indent=2))
            print(f"Saved training settings to {settings_file}")
        except Exception as e:
            print(f"Warning: Could not save training settings: {e}")
        
        return models
    else:
        log_header("LOADING EXISTING MODELS")
        models = load_trained_targets()
        if not models:
            log_header("NO MODELS FOUND - TRAINING FROM SCRATCH")
            return train_all_targets()
        print(f"Loaded {len(models)} existing models")
        return models