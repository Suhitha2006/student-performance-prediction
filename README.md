# Student Performance Prediction & Personalized Academic Support System

## Abstract
This project uses explainable machine learning to predict student scores in mathematics, reading, and writing based on their profile and test-preparation information. It includes a complete Streamlit web application providing model explanations, what-if simulations, and academic support recommendations.

## Project Structure
- `app.py`: Main Streamlit entry point.
- `src/`: Core Python modules (data loading, preprocessing, training, prediction).
- `pages/`: Streamlit multipage application screens.
- `tests/`: Automated pytest suites.
- `data/`: Location for `StudentsPerformance.csv`.

## Setup Instructions

1. **Create Virtual Environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies**:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Data Requirements**:
   Download the Kaggle dataset [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams).
   Place the file `StudentsPerformance.csv` in the `data/` folder.

4. **Launch Application**:
   ```powershell
   streamlit run app.py
   ```
