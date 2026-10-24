import pandas as pd
from pathlib import Path

def load_data(file_path: str = "data/StudentsPerformance.csv") -> pd.DataFrame:
    """Loads and validates the student performance dataset."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Please ensure the Kaggle CSV is present.")
    
    df = pd.read_csv(path)
    
    # Validate expected columns
    expected_cols = [
        "gender", "race/ethnicity", "parental level of education", 
        "lunch", "test preparation course", "math score", 
        "reading score", "writing score"
    ]
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
        
    return df

def validate_dataframe(df: pd.DataFrame) -> dict:
    """Checks for missing values and duplicates."""
    report = {
        "rows": len(df),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
        "math_range": (df["math score"].min(), df["math score"].max()),
        "reading_range": (df["reading score"].min(), df["reading score"].max()),
        "writing_range": (df["writing score"].min(), df["writing score"].max())
    }
    return report
