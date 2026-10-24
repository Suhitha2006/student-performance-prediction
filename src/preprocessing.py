import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def build_preprocessor() -> ColumnTransformer:
    """Creates a scikit-learn ColumnTransformer for the categorical features."""
    categorical_features = [
        "gender", 
        "race/ethnicity", 
        "parental level of education", 
        "lunch", 
        "test preparation course"
    ]
    
    # We ignore unknown categories to support batch predictions robustly
    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop" # We only pass categorical features as inputs
    )
    
    return preprocessor

def get_features_and_target(df: pd.DataFrame, target_col: str):
    """Splits the dataframe into X (features) and y (target).
    Ensures target leakage is prevented by dropping other score columns.
    """
    score_cols = ["math score", "reading score", "writing score"]
    
    # Prevent target leakage
    X = df.drop(columns=score_cols)
    y = df[target_col]
    
    return X, y
