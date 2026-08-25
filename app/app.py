import streamlit as st
import pandas as pd
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Customer Churn Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean, theme-adaptive CSS (works in BOTH Dark & Light modes)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .badge-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        border: 1px solid #EF4444;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }
    
    .badge-medium {
        background-color: rgba(245, 158, 11, 0.2);
        color: #F59E0B;
        border: 1px solid #F59E0B;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }
    
    .badge-low {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10B981;
        border: 1px solid #10B981;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }
    
    .tip-box {
        background: rgba(128, 128, 128, 0.08);
        border-left: 4px solid #3B82F6;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.6rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.92rem;
    }
    
    .tip-risk {
        border-left-color: #EF4444;
    }
    
    .tip-safe {
        border-left-color: #10B981;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 1. Model Loader
# -------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "churn_model.pkl")
    if not os.path.exists(model_path):
        model_path = "models/churn_model.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Failed to load model pipeline: {e}")
    st.stop()

# -------------------------------------------------------------
# 2. Feature Engineering Helper
# -------------------------------------------------------------
def preprocess_features(df_input):
    df = df_input.copy()
    services_list = [
        'Online Security', 'Online Backup', 'Device Protection',
        'Tech Support', 'Streaming TV', 'Streaming Movies'
    ]
    
    for s in services_list:
        if s not in df.columns:
            df[s] = 'No'
            
    df['Total Services'] = df[services_list].eq('Yes').sum(axis=1)
    df['Automatic Payment'] = df['Payment Method'].astype(str).str.contains('automatic', case=False).map({True: 'Yes', False: 'No'})
    df['Has Internet'] = (df['Internet Service'] != 'No').map({True: 'Yes', False: 'No'})
    df['Lives Alone'] = ((df['Partner'] == 'No') & (df['Dependents'] == 'No')).map({True: 'Yes', False: 'No'})
    df['Tenure Group'] = pd.cut(
        df['Tenure Months'],
        bins=[0, 12, 24, 48, 72],
        labels=['0-12 Months', '13-24 Months', '25-48 Months', '49-72 Months'],
        include_lowest=True
    )
    return df

# -------------------------------------------------------------
# 3. Theme-Adaptive Plotly Gauge Helper
# -------------------------------------------------------------
def create_gauge(prob):
    pct = prob * 100
    if pct >= 60:
        bar_color = "#EF4444"
    elif pct >= 35:
        bar_color = "#F59E0B"
    else:
        bar_color = "#10B981"
        
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={'suffix': "%", 'font': {'size': 38, 'family': 'Inter'}},
        title={'text': "Predicted Probability", 'font': {'size': 14, 'family': 'Inter'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': "rgba(128, 128, 128, 0.1)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 35], 'color': "rgba(16, 185, 129, 0.2)"},
                {'range': [35, 60], 'color': "rgba(245, 158, 11, 0.2)"},
                {'range': [60, 100], 'color': "rgba(239, 68, 68, 0.2)"}
            ],
            'threshold': {
                'line': {'color': bar_color, 'width': 4},
                'thickness': 0.8,
                'value': pct
            }
        }
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=35, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# -------------------------------------------------------------
# 4. Header
# -------------------------------------------------------------
st.title("Customer Churn Prediction & Analytics")
st.caption("Predict individual customer churn risk, analyze key drivers, and run batch predictions.")

tab1, tab2, tab3 = st.tabs(["Customer Risk Evaluator", "Batch CSV Scoring", "Model Feature Importance"])

# =============================================================
# TAB 1: Single Prediction
# =============================================================
with tab1:
    col_preset, _ = st.columns([2, 3])
    with col_preset:
        preset = st.selectbox(
            "Quick-Fill Preset Profile:",
            [
                "Custom (Manual Entry)",
                "High Risk (New Month-to-Month Fiber User)",
                "Low Risk (Long-term 2-Year Contract)",
                "Moderate Risk (Senior Citizen, 1-Year Plan)"
            ],
            help="Select a profile to auto-populate the inputs below."
        )

    # Presets definition
    if preset == "High Risk (New Month-to-Month Fiber User)":
        p_gender, p_senior, p_partner, p_dependents = "Female", "No", "No", "No"
        p_contract, p_tenure, p_pay = "Month-to-month", 3, "Electronic check"
        p_paperless, p_monthly = "Yes", 85.0
        p_phone, p_lines = "Yes", "No"
        p_internet = "Fiber optic"
        p_sec, p_back, p_dev, p_tech, p_tv, p_mov = "No", "No", "No", "No", "Yes", "No"
    elif preset == "Low Risk (Long-term 2-Year Contract)":
        p_gender, p_senior, p_partner, p_dependents = "Male", "No", "Yes", "Yes"
        p_contract, p_tenure, p_pay = "Two year", 48, "Credit card (automatic)"
        p_paperless, p_monthly = "No", 60.0
        p_phone, p_lines = "Yes", "Yes"
        p_internet = "DSL"
        p_sec, p_back, p_dev, p_tech, p_tv, p_mov = "Yes", "Yes", "Yes", "Yes", "No", "No"
    elif preset == "Moderate Risk (Senior Citizen, 1-Year Plan)":
        p_gender, p_senior, p_partner, p_dependents = "Female", "Yes", "Yes", "No"
        p_contract, p_tenure, p_pay = "One year", 18, "Bank transfer (automatic)"
        p_paperless, p_monthly = "Yes", 75.0
        p_phone, p_lines = "Yes", "Yes"
        p_internet = "Fiber optic"
        p_sec, p_back, p_dev, p_tech, p_tv, p_mov = "No", "Yes", "No", "No", "Yes", "Yes"
    else:
        p_gender, p_senior, p_partner, p_dependents = "Male", "No", "No", "No"
        p_contract, p_tenure, p_pay = "Month-to-month", 12, "Electronic check"
        p_paperless, p_monthly = "Yes", 70.0
        p_phone, p_lines = "Yes", "No"
        p_internet = "Fiber optic"
        p_sec, p_back, p_dev, p_tech, p_tv, p_mov = "No", "No", "No", "No", "No", "No"

    # Three-Column Input Layout with Native Themed Containers
    col_demo, col_account, col_services = st.columns(3)

    with col_demo:
        with st.container(border=True):
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(p_gender))
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"], index=["No", "Yes"].index(p_senior))
            partner = st.selectbox("Partner", ["No", "Yes"], index=["No", "Yes"].index(p_partner))
            dependents = st.selectbox("Dependents", ["No", "Yes"], index=["No", "Yes"].index(p_dependents))

    with col_account:
        with st.container(border=True):
            st.subheader("Contract & Billing")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(p_contract))
            tenure_months = st.slider("Tenure (Months)", min_value=0, max_value=72, value=p_tenure)
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(p_pay)
            )
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], index=["Yes", "No"].index(p_paperless))
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=p_monthly, step=1.0)
            estimated_total = round(float(tenure_months * monthly_charges), 2)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=estimated_total, step=10.0)

    with col_services:
        with st.container(border=True):
            st.subheader("Subscribed Services")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=["Yes", "No"].index(p_phone))
            lines_opts = ["No", "Yes", "No phone service"] if phone_service == "Yes" else ["No phone service"]
            p_lines_idx = lines_opts.index(p_lines) if p_lines in lines_opts else 0
            multiple_lines = st.selectbox("Multiple Lines", lines_opts, index=p_lines_idx)
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=["Fiber optic", "DSL", "No"].index(p_internet))
            
            if internet_service != "No":
                sub_c1, sub_c2 = st.columns(2)
                with sub_c1:
                    online_security = st.selectbox("Online Security", ["No", "Yes"], index=["No", "Yes"].index(p_sec))
                    online_backup = st.selectbox("Online Backup", ["No", "Yes"], index=["No", "Yes"].index(p_back))
                    device_protection = st.selectbox("Device Protection", ["No", "Yes"], index=["No", "Yes"].index(p_dev))
                with sub_c2:
                    tech_support = st.selectbox("Tech Support", ["No", "Yes"], index=["No", "Yes"].index(p_tech))
                    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"], index=["No", "Yes"].index(p_tv))
                    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"], index=["No", "Yes"].index(p_mov))
            else:
                online_security = online_backup = device_protection = "No internet service"
                tech_support = streaming_tv = streaming_movies = "No internet service"

    predict_btn = st.button("Evaluate Churn Risk", type="primary", use_container_width=True)

    if predict_btn or preset != "Custom (Manual Entry)":
        raw_data = {
            'Gender': gender, 'Senior Citizen': senior_citizen, 'Partner': partner,
            'Dependents': dependents, 'Tenure Months': tenure_months, 'Phone Service': phone_service,
            'Multiple Lines': multiple_lines, 'Internet Service': internet_service,
            'Online Security': online_security, 'Online Backup': online_backup,
            'Device Protection': device_protection, 'Tech Support': tech_support,
            'Streaming TV': streaming_tv, 'Streaming Movies': streaming_movies,
            'Contract': contract, 'Paperless Billing': paperless_billing,
            'Payment Method': payment_method, 'Monthly Charges': monthly_charges,
            'Total Charges': total_charges
        }
        df_single = preprocess_features(pd.DataFrame([raw_data]))
        
        pred = pipeline.predict(df_single)[0]
        prob = pipeline.predict_proba(df_single)[0, 1]
        pct = round(prob * 100, 1)

        st.divider()
        st.subheader("Prediction Results & Risk Diagnostics")
        res_left, res_mid, res_right = st.columns([1.2, 1.2, 1.6])

        with res_left:
            with st.container(border=True):
                st.plotly_chart(create_gauge(prob), use_container_width=True)

        with res_mid:
            with st.container(border=True):
                st.write("**Risk Classification**")
                if pct >= 60:
                    st.markdown('<span class="badge-high">HIGH RISK</span>', unsafe_allow_html=True)
                    st.write("Customer is **highly likely to churn**.")
                elif pct >= 35:
                    st.markdown('<span class="badge-medium">MODERATE RISK</span>', unsafe_allow_html=True)
                    st.write("Customer exhibits **moderate churn signals**.")
                else:
                    st.markdown('<span class="badge-low">LOW RISK (STABLE)</span>', unsafe_allow_html=True)
                    st.write("Customer exhibits **high retention stability**.")
                
                st.metric("Churn Probability", f"{pct}%")
                st.write(f"Active Add-on Services: **{df_single['Total Services'].iloc[0]} of 6**")

        with res_right:
            with st.container(border=True):
                st.write("**Key Drivers & Recommendations**")
                
                has_triggers = False
                if contract == "Month-to-month":
                    has_triggers = True
                    st.markdown('<div class="tip-box tip-risk"><strong>Month-to-month Contract:</strong> Offer a 1-year contract with a retention discount.</div>', unsafe_allow_html=True)
                if tenure_months <= 12:
                    has_triggers = True
                    st.markdown('<div class="tip-box tip-risk"><strong>Early Tenure (&le; 12 Months):</strong> Schedule onboarding customer support check-in.</div>', unsafe_allow_html=True)
                if internet_service == "Fiber optic" and tech_support == "No":
                    has_triggers = True
                    st.markdown('<div class="tip-box tip-risk"><strong>Fiber Optic without Tech Support:</strong> Provide 3 months of free Tech Support.</div>', unsafe_allow_html=True)
                if payment_method == "Electronic check":
                    has_triggers = True
                    st.markdown('<div class="tip-box tip-risk"><strong>Electronic Check:</strong> Incentivize switching to automatic payments with a billing credit.</div>', unsafe_allow_html=True)
                
                if not has_triggers:
                    st.markdown('<div class="tip-box tip-safe"><strong>Stable Customer:</strong> Account attributes indicate low churn risk.</div>', unsafe_allow_html=True)

# =============================================================
# TAB 2: Batch CSV Scoring
# =============================================================
with tab2:
    with st.container(border=True):
        st.subheader("Batch Customer Scoring via CSV")
        st.write("Upload a CSV file containing multiple customer records to calculate churn probabilities in bulk.")
        
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write(f"Loaded **{len(batch_df)}** customer records.")
                
                with st.spinner("Processing batch predictions..."):
                    processed_batch = preprocess_features(batch_df)
                    batch_preds = pipeline.predict(processed_batch)
                    batch_probs = pipeline.predict_proba(processed_batch)[:, 1]
                    
                    results_df = batch_df.copy()
                    results_df["Churn Prediction"] = ["Churn" if p == 1 else "Stay" for p in batch_preds]
                    results_df["Churn Probability (%)"] = (batch_probs * 100).round(2)
                    results_df["Risk Level"] = pd.cut(
                        results_df["Churn Probability (%)"],
                        bins=[0, 35, 60, 100],
                        labels=["Low Risk", "Moderate Risk", "High Risk"],
                        include_lowest=True
                    )
                
                b_col1, b_col2, b_col3 = st.columns(3)
                with b_col1:
                    st.metric("Total Records", len(results_df))
                with b_col2:
                    high_risk_n = (results_df["Risk Level"] == "High Risk").sum()
                    st.metric("High Risk Count", f"{high_risk_n} ({(high_risk_n/len(results_df))*100:.1f}%)")
                with b_col3:
                    avg_p = results_df["Churn Probability (%)"].mean()
                    st.metric("Avg Churn Risk", f"{avg_p:.1f}%")
                    
                st.dataframe(results_df, use_container_width=True)
                
                csv_bytes = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Scored Results (CSV)",
                    data=csv_bytes,
                    file_name="customer_churn_predictions.csv",
                    mime="text/csv",
                    type="primary"
                )
            except Exception as e:
                st.error(f"Error processing CSV: {e}")

# =============================================================
# TAB 3: Feature Importance
# =============================================================
with tab3:
    with st.container(border=True):
        st.subheader("Model Feature Importance")
        st.write("Top features influencing churn predictions in the trained Gradient Boosting model:")
        
        try:
            preproc = pipeline.named_steps['preprocessor']
            clf = pipeline.named_steps['classifier']
            
            raw_feat_names = preproc.get_feature_names_out()
            clean_feat_names = [f.replace('num__', '').replace('cat__', '').replace('remainder__', '') for f in raw_feat_names]
            
            feat_df = pd.DataFrame({
                'Feature': clean_feat_names,
                'Importance': clf.feature_importances_
            }).sort_values('Importance', ascending=True).tail(12)
            
            fig_bar = px.bar(
                feat_df,
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale=['#93C5FD', '#1E40AF']
            )
            fig_bar.update_layout(
                height=420,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.markdown("""
            **Key Drivers:**
            - **Tenure Months**: Shorter tenure is the single strongest indicator of customer churn.
            - **Fiber Optic Internet**: High billing with lack of complementary security/tech support drives cancellations.
            - **Payment Method & Contract**: Month-to-month contracts and electronic check payments show the highest churn rates.
            """)
        except Exception as e:
            st.error(f"Could not load feature importances: {e}")
