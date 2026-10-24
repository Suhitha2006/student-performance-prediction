import joblib
from pathlib import Path
from src.config import MODELS_DIR

def save_model(model, subject: str):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / f"{subject}_model.joblib")

def load_model(subject: str):
    model_path = MODELS_DIR / f"{subject}_model.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    raise FileNotFoundError(f"Model for {subject} not found at {model_path}")
