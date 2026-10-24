import pytest
import pandas as pd
from src.preprocessing import build_preprocessor, get_features_and_target

def test_build_preprocessor():
    preprocessor = build_preprocessor()
    assert preprocessor is not None

def test_get_features_and_target():
    df = pd.DataFrame({
        "gender": ["female", "male"],
        "race/ethnicity": ["group A", "group B"],
        "parental level of education": ["high school", "some college"],
        "lunch": ["standard", "free/reduced"],
        "test preparation course": ["none", "completed"],
        "math score": [70, 80],
        "reading score": [75, 85],
        "writing score": [72, 82]
    })
    
    X, y = get_features_and_target(df, "math score")
    
    # Target shouldn't be in X
    assert "math score" not in X.columns
    # Other scores shouldn't be in X (no target leakage)
    assert "reading score" not in X.columns
    assert "writing score" not in X.columns
    
    # Check length
    assert len(X) == 2
    assert len(y) == 2
