# ???? Churn Prediction for Telecom Customers

This project focuses on predicting customer churn for a telecom company using machine learning techniques. The goal is to build an end-to-end churn prediction system that can identify customers likely to churn and help businesses take proactive retention actions.

---

## ???? Project Objectives

- Build a robust churn prediction model using real customer data
- Improve recall to correctly identify as many churners as possible
- Perform advanced feature engineering and model explainability (SHAP)
- Compare multiple models: Logistic Regression, Random Forest, and LightGBM
- Select the best model and deploy it as an API with a user interface (Flask)
- Package the app using Docker for portable deployment

---

## ???? Key Highlights

- ???? Realistic customer data with over 30+ features
- ???? Advanced Feature Engineering & Encoding
- ?????? Addressed class imbalance using SMOTE and `scale_pos_weight`
- ??? Model Evaluation with Precision, Recall, AUC, and F1 Score
- ???? Threshold optimization to improve recall without hurting precision
- ???? Model Deployment using Flask API + Web UI + Docker container
- ???? Portfolio-ready project structure

---

## ???? Dataset Overview

The dataset contains customer-level information such as:

- **Demographics** (Gender, Senior Citizen, Partner, etc.)
- **Services** (Internet, Streaming, Device Protection, etc.)
- **Account Info** (Tenure, MonthlyCharges, Contract Type)
- **Target Variable**: `Churn` (Yes/No)

---

## ???? Project Structure

```
Churn_Prediction/
???
????????? data/
???   ????????? raw/                     # Original CSV file
???   ????????? processed/               # Cleaned and encoded data
???
????????? notebooks/
???   ????????? 01_data_preprocessing.ipynb        # Load and clean raw data
???   ????????? 02_feature_preparation.ipynb       # Feature engineering + encoding
???   ????????? 03_logistic_regression_modeling.ipynb
???   ????????? 04_random_forest_modeling.ipynb
???   ????????? 05_lightgbm_modeling.ipynb
???
????????? deployment/
???   ????????? final_logreg_model.pkl
???   ????????? final_rf_model.pkl
???   ????????? final_lightgbm_model.pkl
???   ????????? app.py                  # Flask app 
???   ????????? templates/
???   ???   ????????? index.html          # Frontend HTML UI 
???   ????????? Dockerfile              # Docker setup 
???
????????? src/                        # (To add any helper scripts later)
???
????????? requirements.txt
????????? README.md
```

---

## ???? Feature Engineering

Several new features were created:

- `is_long_term_contract`
- `has_bundle`
- `is_senior_alone`
- `low_tenure_high_charge`
- `is_tech_dependent`
- `total_services`: based on cleaned service columns

Also, `TotalCharges` was converted from object to numeric and cleaned.

---

## ???? Feature Selection

- For **Logistic Regression**: Recursive Feature Elimination with Cross Validation (RFECV)
- For **Random Forest** & **LightGBM**: SHAP values used to select top features
- Highly correlated features were removed based on SHAP importance

---

## ?????? Model Building Workflow

1. **Baseline Model**: Logistic Regression
2. **Model 2**: Random Forest (SHAP + grid search)
3. **Model 3**: LightGBM (SHAP + grid search)

All models evaluated using F1, Recall, Precision, AUC.

---

## ???? Model Evaluation Summary

| Model                | Precision | Recall | F1 Score | AUC     |
|---------------------|-----------|--------|----------|---------|
| Logistic Regression | 0.61      | 0.65   | 0.63     | 0.83    |
| Random Forest        | 0.49      | 0.83   | 0.62     | 0.83    |
| **LightGBM** ???      | **0.59**  | **0.81** | **0.69** | **0.83** |

> ??? **Final Model Selected: LightGBM** - Best balance between recall and precision

---

## ??????? Tools & Libraries

- Python 3.8+
- Pandas, NumPy
- Scikit-Learn
- LightGBM
- SHAP
- imbalanced-learn
- Matplotlib / Seaborn
- Flask (for deployment)
- Docker (for containerization)

---

## ???? Deployment Plan

The final LightGBM model will be deployed using:

- ??? **Flask API** to serve model predictions
- ??? **Frontend UI** (HTML) to input customer features and receive predictions
- ??? **Docker** for containerized deployment (portable & consistent)
- ?????? Hosting options: Render / Heroku / Railway / Localhost

Model files:
- `final_lightgbm_model.pkl`
- `final_rf_model.pkl`
- `final_logreg_model.pkl`

---

## ???? Future Improvements

- ??? Automate model retraining pipeline (with updated customer data)
- ???? Add real-time feedback loop for improved accuracy
- ???? Host the Flask API + UI using a cloud platform
- ???? Add unit tests and model drift detection
- ???? Integrate model into CRM dashboard for business use
- ???? Explore deep learning models like TabNet or CatBoost
- ???? Add interpretability dashboard using SHAP for business users

---

## ???? Author

**Sajid Ali**  
Data Scientist | Machine Learning | AI for Telecom  
???? alisajid@8030@gmail.com  


---

## ???? License

This project is open for learning, demonstration, and portfolio use only.
