from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="Student Performance API", version="1.0.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production (e.g., http://localhost:5173)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
models_dir = Path("../models")
try:
    math_model = joblib.load(models_dir / "math_model.joblib")
    reading_model = joblib.load(models_dir / "reading_model.joblib")
    writing_model = joblib.load(models_dir / "writing_model.joblib")
    MODELS_AVAILABLE = True
except Exception as e:
    MODELS_AVAILABLE = False
    print(f"Warning: Models not found or failed to load. {e}")

class StudentData(BaseModel):
    gender: str
    race_ethnicity: str
    parental_education: str
    lunch: str
    test_preparation_course: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": MODELS_AVAILABLE}

@app.post("/predict")
def predict_score(data: StudentData):
    if not MODELS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Models are not loaded.")

    # Convert to DataFrame matching training features
    input_df = pd.DataFrame([{
        "gender": data.gender,
        "race/ethnicity": data.race_ethnicity,
        "parental level of education": data.parental_education,
        "lunch": data.lunch,
        "test preparation course": data.test_preparation_course
    }])

    # Predict and clip bounds
    math_pred = max(0.0, min(100.0, float(math_model.predict(input_df)[0])))
    reading_pred = max(0.0, min(100.0, float(reading_model.predict(input_df)[0])))
    writing_pred = max(0.0, min(100.0, float(writing_model.predict(input_df)[0])))
    
    total = math_pred + reading_pred + writing_pred
    average = total / 3.0

    return {
        "math_score": round(math_pred, 1),
        "reading_score": round(reading_pred, 1),
        "writing_score": round(writing_pred, 1),
        "average_score": round(average, 1),
        "total_score": round(total, 1),
        "overall_status": "PASS" if average >= 40 else "FAIL"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="125.22.211.122", port=8000)
