# Customer Churn Prediction

🔗 **Live Demo:** [telco-customer-churn-prediction-project.streamlit.app](https://telco-customer-churn-prediction-project.streamlit.app/)

A machine learning project to predict customer churn using the Telco dataset. It covers data cleaning, exploratory data analysis (EDA), feature engineering, hyperparameter tuning, model selection based on business recall requirements, and a Streamlit web app with automated retention recommendations.

## Overview

Customer churn happens when subscribers cancel their service. In business, **acquiring a new customer costs significantly more than retaining an existing one**:
- According to research published in the **Harvard Business Review** and studies by **Bain & Company**, acquiring a new customer is **5 to 25 times more expensive** than retaining an existing one.
- Furthermore, increasing customer retention rates by just 5% can increase company profits by 25% to 95%.
- In the telecom industry specifically, Customer Acquisition Cost (CAC) averages over **$300 per subscriber** (marketing, sales commissions, SIM/equipment subsidies), and a lost customer represents hundreds of dollars in lost annual recurring revenue. In contrast, offering a retention incentive (like a bill credit or free tech support) costs only $20 to $40.

Because of this steep cost asymmetry, missing an actual churner (False Negative) is far more damaging than offering a discount to a loyal customer (False Positive). This project analyzes churn drivers, optimizes for **Recall** to catch as many churners as possible, and deploys an interactive Streamlit web app with automated retention recommendations.

## What Makes This Project Different?

Most churn prediction tutorials train a default model, pick the one with the highest accuracy, and output a raw probability score. This project approaches churn from an industry engineering and business perspective:

1. **Business-Driven Metric Selection (Recall over Raw Accuracy)**:
   A standard model can achieve 80% accuracy simply by guessing "No Churn" for most customers, yet fail to catch more than half of the people actually leaving. By centering the evaluation on Customer Acquisition Cost (CAC), we chose a model that maximizes **Recall (80.75%)**, directly protecting customer revenue.

2. **Strict Data Leakage Prevention**:
   The raw Telco dataset contains fields like `Churn Score` and `Churn Reason`. These are post-event metrics generated only after a customer leaves. Including them gives artificial 99% accuracy in notebooks but fails completely in production. We identified and stripped these leakage columns before any modeling began.

3. **Production Pipeline Architecture (Zero Training-Serving Skew)**:
   All preprocessing (imputing zero-tenure charges, standardizing numericals, and one-hot encoding categories with unknown handling) is bundled inside a single Scikit-Learn `Pipeline`. The web app loads this serialized pipeline directly, guaranteeing identical data transformations during inference.

4. **Prescriptive Interventions (Not Just Risk Scores)**:
   Predicting churn probability is only useful if retention teams know what to do next. The application evaluates the specific triggers driving an individual customer's score and suggests targeted retention actions (e.g. 1-year contract incentives, complimentary tech support, or auto-pay billing credits).

5. **100% Deterministic & Zero-Latency (No External LLM Needed)**:
   The recommendations engine runs on pure Python conditional rules derived directly from our exploratory data analysis. It does not require API keys, token fees, or external cloud services, and executes instantly.

## Key Insights from EDA

- **Contract Type**: Month-to-month customers churn much more (~42.7%) than one-year (11.3%) or two-year (2.8%) customers.
- **Customer Tenure**: Churn is heavily concentrated in the first year (0-12 months) of customer lifespan.
- **Internet Service**: Fiber optic users have the highest churn rate (~41.9%), especially when lacking technical support.
- **Tech Support & Security**: Customers without tech support or online security churn at significantly higher rates.
- **Payment Method**: Paying by electronic check has a higher churn rate (~45.3%) compared to automatic payment methods (~15%).

## Feature Engineering

Added custom features to capture customer risk profiles:
- `Tenure Group`: Binned tenure into buckets (`0-12 Months`, `13-24 Months`, `25-48 Months`, `49-72 Months`).
- `Total Services`: Count of active add-on services (Online Security, Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies).
- `Automatic Payment`: Flag indicating whether the customer uses automatic payment (credit card or bank transfer) vs manual methods.
- `Has Internet`: Flag indicating whether the customer has active internet service.
- `Lives Alone`: Flag for customers with neither a partner nor dependents.

## Baseline Model Evaluation

Trained 8 classification models with an 80/20 stratified train-test split:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Gradient Boosting | 79.70% | 65.07% | 50.80% | 57.06% | 0.8433 |
| Logistic Regression | 73.88% | 50.51% | 78.88% | 61.59% | 0.8431 |
| AdaBoost | 79.99% | 67.29% | 47.86% | 55.94% | 0.8389 |
| SVM | 74.73% | 51.60% | 77.81% | 62.05% | 0.8256 |
| XGBoost | 76.30% | 54.26% | 68.18% | 60.43% | 0.8224 |
| Random Forest | 78.35% | 62.55% | 45.99% | 53.00% | 0.8190 |
| KNN | 77.08% | 57.57% | 51.87% | 54.57% | 0.7777 |
| Decision Tree | 73.03% | 49.21% | 50.00% | 49.60% | 0.6565 |

## Hyperparameter Tuning & Final Model Selection

We tuned the top-performing models (Gradient Boosting, Logistic Regression, and SVM) using `RandomizedSearchCV` with 5-fold cross-validation.

### Test Set Results After Tuning:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| **SVM** | 73.67% | 50.25% | **80.75%** | **61.95%** | 0.8348 |
| Logistic Regression | 73.95% | 50.60% | 79.14% | 61.73% | 0.8423 |
| Gradient Boosting | 79.84% | 66.07% | 49.47% | 56.57% | 0.8461 |

### Why SVM Was Chosen as the Final Model:
In customer churn, **Customer Acquisition Cost (CAC) is much higher than the cost of giving retention offers to loyal customers who are flagged incorrectly**. A False Negative (missing a customer who leaves permanently) is far costlier to the business than a False Positive (offering a discount to a customer who stays).

Because of this, we prioritized **Recall** when selecting the final model:
- **SVM** achieved the highest test recall at **80.75%**, successfully catching **302 out of 374** actual churners and leaving only 72 missed churners.
- In contrast, Gradient Boosting at default threshold only caught 185 churners (49.47% recall).
- **Confusion Matrix on Test Set (SVM)**:
  - Staying correctly predicted (TN): 736
  - Churners caught (TP): 302 (80.7% recall)
  - Unnecessarily targeted (FP): 299
  - Churners missed (FN): 72

The final trained pipeline is saved in `models/churn_model.pkl`.

## Data-Driven Retention Recommendations

The Streamlit web app does not use an external LLM or API keys. Instead, it runs a deterministic recommendation engine derived directly from our EDA findings:

- **Month-to-month Contract**: Suggests offering a 1-year contract with an introductory discount.
- **Tenure $\le$ 12 Months**: Triggers an onboarding customer support check-in.
- **Fiber Optic without Tech Support**: Offers 3 months of complimentary Tech Support.
- **Electronic Check Payment**: Suggests switching to automatic billing with a statement credit.

## Project Structure

```
Telco-Customer-Churn-Prediction/
├── app/
│   └── app.py                     # Streamlit web app
├── data/
│   └── Telco_customer_churn.csv   # Dataset
├── models/
│   └── churn_model.pkl            # Trained SVM pipeline
├── notebooks/
│   └── notebook.ipynb             # Complete EDA, feature engineering & model tuning
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

4. **Open the notebook**:
   ```bash
   jupyter notebook notebooks/notebook.ipynb
   ```
