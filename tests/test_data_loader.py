import pytest
import pandas as pd
from pathlib import Path
from src.data_loader import load_data, validate_dataframe

def test_load_data_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(str(tmp_path / "nonexistent.csv"))

def test_validate_dataframe():
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
    
    report = validate_dataframe(df)
    assert report["rows"] == 2
    assert report["duplicates"] == 0
    assert report["math_range"] == (70, 80)
