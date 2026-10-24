# College Mini-Project Report

## 1. Introduction
This project presents a Student Performance Prediction & Personalized Academic Support System using Explainable Machine Learning. It provides an end-to-end web application that leverages predictive modeling to estimate student exam scores based on demographic and educational background data.

## 2. Literature/Background Overview
Historically, educational assessment relies solely on periodic examinations. Machine learning enables proactive interventions by identifying patterns in student profiles that correlate with academic outcomes.

## 3. Problem Statement
Students perform differently across mathematics, reading, and writing. Identifying factors that influence these variations can help educators and students target their efforts, but raw data is difficult to interpret without analytical tools.

## 4. Existing System Limitations
- Traditional systems are often reactive, evaluating students only after exams.
- They lack personalized "what-if" simulation capabilities.
- They do not clearly explain *why* a particular performance level might be expected.

## 5. Proposed System
The proposed system is an interactive Streamlit dashboard that:
1. Validates and processes the Kaggle "Students Performance in Exams" dataset.
2. Trains regression models to predict scores for math, reading, and writing independently.
3. Provides a What-If Simulator for scenario comparison.
4. Generates automated, general academic support recommendations based on predicted score bands.
5. Supports batch predictions via CSV upload.

## 6. Architecture and Workflow
- **Data Layer:** Pandas for data loading and cleaning.
- **Model Layer:** Scikit-learn pipelines with OneHotEncoding and various regressors (Linear Regression, Decision Tree, Random Forest, Gradient Boosting).
- **Presentation Layer:** Streamlit web application with multipage navigation.

## 7. Dataset Description
The dataset contains approximately 1,000 records with categorical inputs (gender, race/ethnicity, parental level of education, lunch, test preparation course) and numerical targets (math score, reading score, writing score).

## 8. Preprocessing
Categorical variables are transformed using `OneHotEncoder(handle_unknown="ignore")`. Targets are isolated for each subject to prevent target leakage (e.g., math models do not use reading or writing scores as inputs).

## 9. Implementation and Algorithms
Models are trained using an 80/20 train-test split. Cross-validation (cv=5) on the training set evaluates generalizability. The algorithm with the highest $R^2$ on cross-validation is selected and finally evaluated on the held-out test set.

## 10. Actual Experimental Results
The models were trained and evaluated across an 80/20 train-test split with 5-fold cross-validation on the training set:

- **Math Best Model:** Linear Regression (Test $R^2$: 0.1760, MAE: 11.27, RMSE: 14.16, 5-Fold CV $R^2$: 0.2392 ± 0.0811)
- **Reading Best Model:** Linear Regression (Test $R^2$: 0.1594, MAE: 10.83, RMSE: 13.79, 5-Fold CV $R^2$: 0.2122 ± 0.0422)
- **Writing Best Model:** Linear Regression (Test $R^2$: 0.2637, MAE: 10.19, RMSE: 13.32, 5-Fold CV $R^2$: 0.3196 ± 0.0574)

### Comparative Performance Table
| Subject | Model | Test MAE | Test RMSE | Test $R^2$ | 5-Fold CV $R^2$ Mean |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Math** | Linear Regression | **11.27** | **14.16** | **0.1760** | **0.2392** |
| | Gradient Boosting | 11.54 | 14.70 | 0.1123 | 0.1696 |
| | Random Forest | 12.32 | 15.68 | -0.0101 | 0.0209 |
| | Decision Tree | 12.69 | 16.14 | -0.0700 | -0.0922 |
| **Reading** | Linear Regression | **10.83** | **13.79** | **0.1594** | **0.2122** |
| | Gradient Boosting | 11.22 | 14.33 | 0.0923 | 0.1450 |
| | Random Forest | 11.82 | 15.21 | -0.0224 | -0.0118 |
| | Decision Tree | 12.30 | 15.72 | -0.0926 | -0.1242 |
| **Writing** | Linear Regression | **10.19** | **13.32** | **0.2637** | **0.3196** |
| | Gradient Boosting | 10.61 | 13.91 | 0.1976 | 0.2698 |
| | Random Forest | 11.22 | 14.73 | 0.1003 | 0.1227 |
| | Decision Tree | 11.49 | 15.04 | 0.0621 | 0.0370 |

## 11. Screenshots
[To be completed by the student: Insert dashboard screenshots here]

## 12. Conclusion
The system successfully integrates machine learning with an interactive interface to predict and explain potential student performance, demonstrating the value of predictive analytics in educational contexts.

## 13. Future Enhancements
- Integration of actual study hours and attendance data (if collected).
- Advanced interpretability using SHAP values.
- Deployment to a cloud platform (e.g., Streamlit Community Cloud or Heroku).

## 14. References
1. Kaggle Dataset: Students Performance in Exams. https://www.kaggle.com/datasets/spscientist/students-performance-in-exams
2. Scikit-learn Documentation: https://scikit-learn.org/
3. Streamlit Documentation: https://docs.streamlit.io/
