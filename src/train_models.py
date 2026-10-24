import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import load_data
from src.preprocessing import build_preprocessor, get_features_and_target

def evaluate_model(y_true, y_pred, name="Model"):
    """Calculates regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def train_subject(df: pd.DataFrame, subject: str, models: dict):
    """Trains and selects the best model for a specific subject."""
    X, y = get_features_and_target(df, target_col=f"{subject} score")
    
    # 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    best_model_name = None
    best_r2 = -float("inf")
    best_pipeline = None
    
    results = {}
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", model)
        ])
        
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")
        mean_cv_r2 = cv_scores.mean()
        
        # Fit on full training set
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        metrics = evaluate_model(y_test, y_pred, name)
        metrics["CV_R2_Mean"] = mean_cv_r2
        metrics["CV_R2_Std"] = cv_scores.std()
        
        results[name] = metrics
        
        if metrics["R2"] > best_r2:
            best_r2 = metrics["R2"]
            best_model_name = name
            best_pipeline = pipeline

    return {
        "subject": subject,
        "best_model_name": best_model_name,
        "best_pipeline": best_pipeline,
        "test_metrics": results[best_model_name],
        "all_results": results
    }

def main():
    print("Loading data...")
    df = load_data()
    
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }
    
    subjects = ["math", "reading", "writing"]
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    all_metrics = {}
    
    for subject in subjects:
        print(f"Training models for {subject}...")
        result = train_subject(df, subject, models)
        
        print(f"Best model for {subject}: {result['best_model_name']}")
        print(f"Test R2: {result['test_metrics']['R2']:.4f}")
        
        # Save model
        model_path = models_dir / f"{subject}_model.joblib"
        joblib.dump(result["best_pipeline"], model_path)
        print(f"Saved to {model_path}\n")
        
        all_metrics[subject] = result
        
    # Save metrics
    joblib.dump(all_metrics, models_dir / "evaluation_metrics.joblib")
    print("Training complete. Models saved in 'models/' directory.")

if __name__ == "__main__":
    main()
