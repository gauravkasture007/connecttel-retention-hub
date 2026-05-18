
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title='ConnectTel Retention Dashboard', layout='wide')

st.title('⚡ ConnectTel: Unified Retention Intelligence')
st.markdown('### Real-time Churn Prediction & Tactical CRM Mapping')

# Load Pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load('connecttel_retention_pipeline.pkl')

pipeline = load_pipeline()

uploaded_file = st.file_uploader("Upload Customer Data (CSV)", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    if st.button('Run Retention Analysis'):
        # Inference
        probs = pipeline.predict_proba(data)[:, 1]
        data['Churn_Probability'] = probs
        
        # Strategy Mapping
        def assign_strategy(row):
            p, t = row['Churn_Probability'], row.get('tenure_months', 0)
            if p > 0.70 or t > 72: return 'Dedicated Account Manager'
            if p > 0.35 and 48 <= t <= 72: return 'Exclusive Device Early Access'
            return 'Automated Retention Voucher' if p >= 0.15 else 'Standard Service'

        data['Retention_Strategy'] = data.apply(assign_strategy, axis=1)
        
        # Display Results
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
