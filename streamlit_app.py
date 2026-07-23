
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
from sklearn.base import BaseEstimator, TransformerMixin

# --- CRITICAL: Custom logic must be defined here for joblib to load the model ---
class ConnectTelFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['monthly_charges'] = pd.to_numeric(X['monthly_charges'], errors='coerce').fillna(0)
        X['total_charges'] = pd.to_numeric(X['total_charges'], errors='coerce').fillna(0)
        X['friction_score'] = X.get('network_issues_3m', 0) + X.get('num_complaints_3m', 0)
        X['charge_velocity'] = X['monthly_charges'] / (X['total_charges'] + 1)
        X['usage_intensity'] = X['avg_data_gb_month'] / (X.get('tenure_months', 0) + 1)
        max_gb = X['avg_data_gb_month'].max() + 1 if 'avg_data_gb_month' in X.columns else 1
        max_logins = X['app_logins_30d'].max() + 1 if 'app_logins_30d' in X.columns else 1
        X['silent_risk_score'] = (X.get('avg_data_gb_month', 0) / max_gb) * (1 - (X.get('app_logins_30d', 0) / max_logins))
        return X

st.set_page_config(page_title='ConnectTel Executive Hub', layout='wide')

@st.cache_resource
def load_production_artifacts():
    # Attempt to load the model file pushed to the repo
    try:
        model = joblib.load('connecttel_churn_model.pkl')
        return model
    except Exception as e:
        st.error(f'Model Load Error: {e}')
        return None

st.sidebar.title("💎 ConnectTel Suite")
view = st.sidebar.radio("Navigation", ["🛡️ Retention HQ", "📊 Data Science Lab", "📄 Project Governance"])

if view == "📄 Project Governance":
    st.title("📄 Executive Sign-off & Audit")
    report_file = 'ConnectTel_Executive_Signoff_Report.pdf'
    if os.path.exists(report_file):
        with open(report_file, "rb") as f:
            st.download_button("⏬ Download Executive Sign-off PDF", f, file_name=report_file)
        st.success("✅ Official project documentation is verified and ready for distribution.")
    else:
        st.error("❌ Report artifact missing.")

elif view == "🛡️ Retention HQ" or view == "📊 Data Science Lab":
    uploaded_file = st.file_uploader("Upload Manifest (telecom_churn.csv)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        model = load_production_artifacts()
        if model:
            # Run inference
            df['risk_score'] = model.predict_proba(df)[:, 1]
            if view == "🛡️ Retention HQ":
                st.title("🛡️ Retention Command")
                fig = px.scatter(df, x='tenure_months', y='monthly_charges', color='risk_score', color_continuous_scale='Oryel')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.title("📊 Predictive Insights")
                st.dataframe(df.sort_values('risk_score', ascending=False))
    else:
        st.info("Please upload data to begin.")
