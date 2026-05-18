
import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.base import BaseEstimator, TransformerMixin

# --- Custom Class Definition (REQUIRED FOR JOBLIB LOAD) ---
# This class must match the name and structure used during training exactly.
class ConnectTelFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Ensure numeric types to prevent division errors
        X['monthly_charges'] = pd.to_numeric(X['monthly_charges'], errors='coerce').fillna(0)
        X['total_charges'] = pd.to_numeric(X['total_charges'], errors='coerce').fillna(0)
        
        # 1. Friction Score
        X['friction_score'] = X.get('network_issues_3m', 0) + X.get('num_complaints_3m', 0)
        
        # 2. Charge Velocity (Bill Shock)
        X['charge_velocity'] = X['monthly_charges'] / (X['total_charges'] + 1)
        
        # 3. Usage Intensity
        X['usage_intensity'] = X['avg_data_gb_month'] / (X.get('tenure_months', 0) + 1)

        # 4. Silent Risk Score
        max_gb = X['avg_data_gb_month'].max() + 1
        max_logins = X['app_logins_30d'].max() + 1
        X['silent_risk_score'] = (X['avg_data_gb_month'] / max_gb) * (1 - (X['app_logins_30d'] / max_logins))
        return X

st.set_page_config(page_title='ConnectTel Retention Dashboard', layout='wide')

st.title('⚡ ConnectTel: Unified Retention Intelligence')
st.markdown('### Real-time Churn Prediction & Tactical CRM Mapping')

# Load Pipeline
@st.cache_resource
def load_pipeline():
    # Ensuring the path matches the repo root
    return joblib.load('connecttel_retention_pipeline.pkl')

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Critical Error loading model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload Customer Data (CSV)", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if st.button('Run Retention Analysis'):
        # The pipeline handles engineering and preprocessing internally
        probs = pipeline.predict_proba(data)[:, 1]
        data['Churn_Probability'] = probs

        def assign_strategy(row):
            p, t = row['Churn_Probability'], row.get('tenure_months', 0)
            if p > 0.70 or t > 72: return 'Dedicated Account Manager'
            if p > 0.35 and 48 <= t <= 72: return 'Exclusive Device Early Access'
            return 'Automated Retention Voucher' if p >= 0.15 else 'Standard Service'

        data['Retention_Strategy'] = data.apply(assign_strategy, axis=1)
        high_risk = data[data['Churn_Probability'] >= 0.15].sort_values('Churn_Probability', ascending=False)

        col1, col2 = st.columns(2)
        col1.metric("Total Targets Identified", len(high_risk))
        col2.metric("Avg Risk Probability", f"{high_risk['Churn_Probability'].mean():.2%}")

        st.dataframe(high_risk[['customer_id', 'Churn_Probability', 'Retention_Strategy']].head(50))

        st.download_button(
            label="Download CRM Action List",
            data=high_risk.to_csv(index=False).encode('utf-8'),
            file_name='ConnectTel_Action_List.csv',
            mime='text/csv',
        )
