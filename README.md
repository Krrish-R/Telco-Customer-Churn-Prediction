# Customer Churn Prediction

🔗 **Live Demo:** [telco-customer-churn-prediction-project.streamlit.app](https://telco-customer-churn-prediction-project.streamlit.app/)

A machine learning project to predict customer churn using the Telco dataset. It includes data cleaning, exploratory data analysis (EDA), feature engineering, model training/tuning, and a Streamlit web app for real-time predictions.

## Overview

Customer churn happens when subscribers cancel their service. This project analyzes what factors cause customers to leave and builds a machine learning model to predict churn probability for new or existing customers.

## Key Insights from EDA

- **Contract Type**: Month-to-month customers churn much more (~42.7%) than one-year (11.3%) or two-year (2.8%) customers.
- **Internet Service**: Fiber optic users have the highest churn rate (~41.9%).
- **Tech Support & Security**: Customers without tech support or online security are more likely to churn.
- **Payment Method**: Paying by electronic check has a higher churn rate (~45.3%) than automatic payment methods.

## Feature Engineering

Added 5 custom features:
- `Tenure Group`: Binned tenure into 4 buckets (`0-12`, `13-24`, `25-48`, `49-72` months).
- `Total Services`: Number of additional services subscribed (Security, Backup, Tech Support, Streaming TV/Movies, etc.).
- `Automatic Payment`: Flag for automatic vs manual payment methods.
- `Has Internet`: Flag for whether the customer has internet service.
- `Lives Alone`: Flag for customers with no partner and no dependents.

## Models Evaluated

Trained 8 models on an 80/20 train-test split:

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- | --- |
| Gradient Boosting | 79.99% | 0.8516 | 65.44% | 52.14% | 58.04% |
| Logistic Regression | 79.99% | 0.8511 | 63.69% | 57.22% | 60.28% |
| AdaBoost | 80.06% | 0.8474 | 65.98% | 51.34% | 57.74% |
| XGBoost | 78.00% | 0.8324 | 60.00% | 51.34% | 55.33% |
| Random Forest | 79.49% | 0.8317 | 64.12% | 51.60% | 57.19% |
| SVM | 80.34% | 0.8163 | 66.44% | 52.41% | 58.59% |
| KNN | 76.86% | 0.7827 | 56.59% | 55.08% | 55.83% |
| Decision Tree | 72.46% | 0.6533 | 48.21% | 50.27% | 49.21% |

### Final Model
After tuning hyperparameters with `RandomizedSearchCV`, **Gradient Boosting** was chosen as the best model:
- **Test Accuracy**: 80.70%
- **ROC-AUC**: 0.8545
- **Precision**: 67.47%
- **Recall**: 52.67%
- **F1 Score**: 59.16%

The final preprocessing pipeline and model are saved in `models/churn_model.pkl`.

## Project Structure

```
Customer Churn Prediction/
├── app/
│   └── app.py                   # Streamlit web app
├── data/
│   └── Telco_customer_churn.csv # Dataset
├── models/
│   └── churn_model.pkl          # Trained model pipeline
├── notebooks/
│   └── EDA.ipynb                # EDA and modeling notebook
├── .gitignore
├── requirements.txt
└── README.md
```

## How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Krrish-R/Telco-Customer-Churn-Prediction.git
   cd Telco-Customer-Churn-Prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit web app**:
   ```bash
   streamlit run app/app.py
   ```

4. **Run the notebook (optional)**:
   ```bash
   jupyter notebook notebooks/EDA.ipynb
   ```
