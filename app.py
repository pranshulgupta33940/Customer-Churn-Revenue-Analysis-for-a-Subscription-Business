import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="Churn & Revenue Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Model
@st.cache_resource
def load_model():
    return joblib.load('model.pkl'), joblib.load('features.pkl')

try:
    model, feature_names = load_model()
except:
    st.error("Model not found. Please run the training script first.")
    st.stop()

# --- HEADER ---
st.title("📊 Customer Churn & Revenue Analysis for a Subscription Business")
st.markdown("""
This dashboard allows you to analyze customer profiles and predict the likelihood of churn. 
Use the tabs below to simulate a customer scenario.
""")
st.markdown("---")

# --- INPUT SECTION (TABS) ---
tab1, tab2, tab3 = st.tabs(["👤 Customer Profile", "📞 Services Offered", "💰 Account & Billing"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    with col2:
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
    
    tenure = st.slider("Tenure (Months)", 0, 72, 12, help="How long the customer has been with the company.")

with tab2:
    col1, col2, col3 = st.columns(3)
    with col1:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multi_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    with col2:
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    with col3:
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    with col2:
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1000.0, step=10.0)

# Create DataFrame
input_data = {
    'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': partner, 'Dependents': dependents,
    'tenure': tenure, 'PhoneService': phone_service, 'MultipleLines': multi_lines,
    'InternetService': internet_service, 'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
    'DeviceProtection': device_protection, 'TechSupport': tech_support, 'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies, 'Contract': contract, 'PaperlessBilling': paperless,
    'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
}

df_input = pd.DataFrame([input_data])

# Feature Engineering
def tenure_group(t):
    if t <= 12: return '0-12'
    elif t <= 24: return '12-24'
    else: return '24+'

df_input['tenure_group'] = df_input['tenure'].apply(tenure_group)
avg_charges = 64.76 
df_input['high_value_customer'] = df_input['MonthlyCharges'].apply(lambda x: 1 if x > avg_charges else 0)
df_input['payment_risk_flag'] = df_input.apply(lambda x: 1 if x['PaymentMethod'] == 'Electronic check' and x['Contract'] == 'Month-to-month' else 0, axis=1)

# Encoding
domains = {
    'gender': ['Female', 'Male'],
    'Partner': ['No', 'Yes'],
    'Dependents': ['No', 'Yes'],
    'PhoneService': ['No', 'Yes'],
    'MultipleLines': ['No', 'No phone service', 'Yes'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['No', 'No internet service', 'Yes'],
    'OnlineBackup': ['No', 'No internet service', 'Yes'],
    'DeviceProtection': ['No', 'No internet service', 'Yes'],
    'TechSupport': ['No', 'No internet service', 'Yes'],
    'StreamingTV': ['No', 'No internet service', 'Yes'],
    'StreamingMovies': ['No', 'No internet service', 'Yes'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['No', 'Yes'],
    'PaymentMethod': ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'],
    'tenure_group': ['0-12', '12-24', '24+']
}

for col in domains:
    df_input[col] = df_input[col].map({v: i for i, v in enumerate(domains[col])})

df_final = df_input[feature_names]

# --- PREDICTION SECTION ---
st.markdown("### ⚡ Analysis Results")

if st.button("Predict Churn Risk", type="primary"):
    prediction = model.predict(df_final)[0]
    prob = model.predict_proba(df_final)[0][1]
    
    col1, col2, col3 = st.columns([1, 2, 2])
    
    with col1:
        st.metric("Risk Score", f"{prob:.1%}")
    
    with col2:
        if prediction == 1:
            st.error("🚨 **High Risk of Churn**")
        else:
            st.success("✅ **Loyal Customer**")
            
    with col3:
        st.progress(prob)
        st.caption("Probability of Churn")

    # Recommendations
    st.subheader("💡 Strategic Recommendations")
    rec_col1, rec_col2 = st.columns(2)
    
    recs = []
    if contract == 'Month-to-month':
        recs.append("📌 **Contract**: User is on Month-to-month. Offer a **5% discount** for switching to a 1-year contract.")
    if internet_service == 'Fiber optic' and monthly_charges > 80:
        recs.append("📌 **Pricing**: High Fiber Optic charges detected. Verify competitor pricing in user's region.")
    if tech_support == 'No':
        recs.append("📌 **Upsell**: Customer missing Tech Support. Bundle it for free for 3 months to increase stickiness.")
    if tenure < 6:
        recs.append("📌 **Onboarding**: New customer (< 6 months). Enroll in 'New User Concierge' program.")
        
    if not recs:
        recs.append("✅ Customer status looks healthy. Maintain current service levels.")
        
    for r in recs:
        st.info(r)

    # --- SHAP ---
    st.markdown("---")
    with st.expander("🔍 Deep Dive: Why this prediction?", expanded=False):
        st.write("The waterfall plot below breaks down how each feature pushed the risk score up (red) or down (blue).")
        try:
            import shap
            import matplotlib.pyplot as plt
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(df_final)
            fig, ax = plt.subplots(figsize=(10, 4))
            
            if len(shap_values.shape) == 3:
                explanation = shap_values[0, :, 1]
            else:
                explanation = shap_values[0]
                
            shap.plots.waterfall(explanation, show=False)
            st.pyplot(fig, bbox_inches='tight')
        except Exception as e:
            st.warning(f"Feature importance unavailable: {e}")
