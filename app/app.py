import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="wide"
)

# Load the saved model pipeline
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "churn_model.pkl")
    if not os.path.exists(model_path):
        model_path = "models/churn_model.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

st.title("Customer Churn Prediction")
st.write("Enter customer information to predict churn probability.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

with col2:
    st.subheader("Account Information")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    tenure_months = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=65.0, step=1.0)
    
    estimated_total = round(float(tenure_months * monthly_charges), 2)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=estimated_total, step=10.0)

with col3:
    st.subheader("Services")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"] if phone_service == "Yes" else ["No phone service"]
    )
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    
    if internet_service != "No":
        online_security = st.selectbox("Online Security", ["No", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
    else:
        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"

st.write("")
if st.button("Predict Churn", type="primary"):
    data = {
        'Gender': gender,
        'Senior Citizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'Tenure Months': tenure_months,
        'Phone Service': phone_service,
        'Multiple Lines': multiple_lines,
        'Internet Service': internet_service,
        'Online Security': online_security,
        'Online Backup': online_backup,
        'Device Protection': device_protection,
        'Tech Support': tech_support,
        'Streaming TV': streaming_tv,
        'Streaming Movies': streaming_movies,
        'Contract': contract,
        'Paperless Billing': paperless_billing,
        'Payment Method': payment_method,
        'Monthly Charges': monthly_charges,
        'Total Charges': total_charges,
    }
    
    df_input = pd.DataFrame([data])
    
    # Feature engineering matching EDA
    services = [
        'Online Security', 'Online Backup', 'Device Protection',
        'Tech Support', 'Streaming TV', 'Streaming Movies'
    ]
    df_input['Total Services'] = df_input[services].eq('Yes').sum(axis=1)
    df_input['Automatic Payment'] = df_input['Payment Method'].str.contains('automatic').map({True: 'Yes', False: 'No'})
    df_input['Has Internet'] = (df_input['Internet Service'] != 'No').map({True: 'Yes', False: 'No'})
    df_input['Lives Alone'] = ((df_input['Partner'] == 'No') & (df_input['Dependents'] == 'No')).map({True: 'Yes', False: 'No'})
    df_input['Tenure Group'] = pd.cut(
        df_input['Tenure Months'],
        bins=[0, 12, 24, 48, 72],
        labels=['0-12 Months', '13-24 Months', '25-48 Months', '49-72 Months'],
        include_lowest=True
    )
    
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0, 1]
    churn_percent = round(probability * 100, 1)
    
    st.divider()
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if prediction == 1:
            st.error(f"Prediction: Customer is likely to churn ({churn_percent}% risk)")
        else:
            st.success(f"Prediction: Customer is likely to stay ({churn_percent}% churn risk)")
        
        st.progress(float(probability))

    with res_col2:
        st.write("**Key Factors:**")
        factors = []
        if contract == "Month-to-month":
            factors.append("- Month-to-month contract increases churn risk.")
        if tenure_months <= 12:
            factors.append("- Low tenure (1 year or less).")
        if payment_method == "Electronic check":
            factors.append("- Paying by electronic check has higher churn rates.")
        if internet_service == "Fiber optic" and tech_support == "No":
            factors.append("- Fiber optic internet without tech support.")
        if df_input['Total Services'].iloc[0] == 0 and internet_service != "No":
            factors.append("- No additional add-on services.")
        
        if factors:
            for f in factors:
                st.write(f)
        else:
            st.write("- Account parameters indicate standard usage patterns.")
