from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

DATASET_PATH = DATA_DIR / "StudentsPerformance.csv"
METRICS_PATH = MODELS_DIR / "evaluation_metrics.joblib"

# Model specific configurations
RANDOM_STATE = 42
TEST_SIZE = 0.2

EXPECTED_CATEGORICAL_COLUMNS = [
    "gender", 
    "race/ethnicity", 
    "parental level of education", 
    "lunch", 
    "test preparation course"
]

TARGET_COLUMNS = ["math score", "reading score", "writing score"]
