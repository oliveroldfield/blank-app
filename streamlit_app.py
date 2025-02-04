import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Initialize session state
# Known fraudulent IBANs and suspicious websites
FRAUDULENT_IBANS = [
    'DE89370400440532013000',
    'GB29NWBK60161331926819',
    'FR1420041010050500013M02606'
]

SUSPICIOUS_WEBSITES = {
    'facebook.com': 0.8,
    'meta.com': 0.3,
    'instagram.com': 0.25,
    'whatsapp.com': 0.25,
    'telegram.org': 0.2
}

if 'model' not in st.session_state:
    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    hours = np.random.randint(0, 24, n_samples)
    suspicious_hours = ((hours >= 1) & (hours <= 5)).astype(int)  # Convert to binary feature
    amounts = np.random.lognormal(mean=7, sigma=1, size=n_samples)  # Log-normal for realistic amounts
    cross_border = np.random.binomial(1, 0.3, n_samples)
    high_risk_country = np.random.binomial(1, 0.2, n_samples)
    high_risk_merchant = np.random.binomial(1, 0.15, n_samples)
    
    # Create feature matrix
    X = np.column_stack([
        suspicious_hours,
        amounts,
        cross_border,
        high_risk_country,
        high_risk_merchant
    ])
    
    # Generate labels (fraud) based on a weighted combination of features
    fraud_prob = (
        0.3 * suspicious_hours.astype(float) +  # Suspicious hours
        0.4 * (amounts > np.percentile(amounts, 90)).astype(float) +  # High amounts
        0.2 * cross_border.astype(float) +  # Cross-border transactions
        0.5 * high_risk_country.astype(float) +  # High-risk countries
        0.3 * high_risk_merchant.astype(float)  # High-risk merchants
    )
    y = (fraud_prob > 0.5).astype(int)
    
    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)
    
    # Store in session state
    st.session_state['model'] = model
    st.session_state['scaler'] = scaler

st.title("ML-Enhanced Fraud Detection Demo")

with st.expander("ℹ️ About this demo", expanded=True):
    st.markdown("""
    This demo combines rule-based and ML approaches to detect potentially fraudulent transactions.
    
    **Example Low-Risk Transaction:**
    - Time: 2:00 PM
    - Amount: $500
    - US domestic transaction
    - Retail merchant
    
    **Example High-Risk Transaction:**
    - Time: 3:00 AM
    - Amount: $15,000
    - Nigeria → US transfer
    - Gambling merchant
    - Website: facebook.com
    
    **Known Fraudulent IBANs:**
    - DE89370400440532013000
    - GB29NWBK60161331926819
    - FR1420041010050500013M02606
    
    **Monitored Websites (Risk Score):**
    - facebook.com (0.8)
    - meta.com (0.3)
    - instagram.com (0.25)
    - whatsapp.com (0.25)
    - telegram.org (0.2)
    """)

# Input fields
col1, col2 = st.columns(2)

with col1:
    date_time = st.date_input("Transaction Date")
    iban = st.text_input("IBAN (if applicable)")
    website = st.text_input("Website (if applicable)")
    time = st.time_input("Transaction Time")
    transaction_type = st.selectbox(
        "Transaction Type",
        ["Bill Payment", "Fund Transfer", "ATM", "POS", "CBFT"]
    )
    payment_gateway = st.selectbox(
        "Payment Gateway",
        ["Apple Pay", "Google Pay", "POS", "ATM"]
    )

with col2:
    merchant_category = st.selectbox(
        "Merchant Category",
        ["Retail", "Travel", "Entertainment", "Gambling", "E-commerce", "Services"]
    )
    countries = ["United States", "United Kingdom", "Canada", "Australia", "Nigeria", 
                 "India", "United Arab Emirates", "Pakistan", "Nepal", "China", "Russia", "Brazil"]
    source_country = st.selectbox("Source Country", countries)
    dest_country = st.selectbox("Destination Country", countries)
    amount = st.number_input("Transaction Amount", min_value=0.0)

if st.button("Analyze Transaction"):
    datetime_combined = datetime.combine(date_time, time)
    
    # Rule-based risk assessment
    risk_score = 0
    risk_factors = []
    
    high_risk_countries = ["Nigeria", "Russia"]
    high_risk = int(source_country in high_risk_countries or dest_country in high_risk_countries)
    cross_border = int(source_country != dest_country)
    high_risk_merchant = int(merchant_category == "Gambling")
    
    if high_risk:
        risk_score += 30
        risk_factors.append("High-risk country involved in transaction")
    
    if cross_border:
        risk_score += 20
        risk_factors.append("Cross-border transaction")
    
    if amount > 10000:
        risk_score += 25
        risk_factors.append("Large transaction amount")
    
    if high_risk_merchant:
        risk_score += 15
        risk_factors.append("High-risk merchant category")
    
    if datetime_combined.hour >= 1 and datetime_combined.hour <= 5:
        risk_score += 10
        risk_factors.append("Transaction during suspicious hours")
    
    # ML-based probability
    features = np.array([[
        int(datetime_combined.hour >= 1 and datetime_combined.hour <= 5),  # Binary suspicious hour
        amount,
        cross_border,
        high_risk,
        high_risk_merchant
    ]])
    
    # Scale features
    features_scaled = st.session_state['scaler'].transform(features)
    
    # Get probability
    fraud_prob = st.session_state['model'].predict_proba(features_scaled)[0][1]
    
    # Display results
    st.header("Analysis Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Rule-Based Risk Score", f"{risk_score}/100")
    
    with col2:
        st.metric("ML-Based Fraud Probability", f"{fraud_prob:.1%}")
    
    # Check IBAN
    if iban and iban in FRAUDULENT_IBANS:
        st.error("⚠️ IMMEDIATE FRAUD ALERT: Known fraudulent IBAN detected")
        st.stop()
        
    # Check website
    website_risk = 0
    if website:
        website_risk = SUSPICIOUS_WEBSITES.get(website.lower(), 0)
        if website_risk > 0:
            risk_factors.append(f"Suspicious website ({website})")
            risk_score += website_risk * 100
    
    # Combined assessment
    combined_risk = (0.4 * (risk_score/100) + 0.6 * fraud_prob + website_risk * 0.3)
    
    if combined_risk >= 0.5:
        st.error(f"⚠️ High Risk Transaction (Combined Risk: {combined_risk:.1%})")
    else:
        st.success(f"✅ Low Risk Transaction (Combined Risk: {combined_risk:.1%})")
    
    # Risk factors
    if risk_factors:
        st.subheader("Risk Factors Identified:")
        for factor in risk_factors:
            st.write(f"• {factor}")
            
    # Feature importance
    st.subheader("Feature Weights")
    feature_names = ['Suspicious hours (1-5 AM)', 'Amount', 'Cross-border', 'High-risk country', 'High-risk merchant']
    weights = st.session_state['model'].coef_[0]
    
    # Create weight visualization
    weight_df = pd.DataFrame({
        'Feature': feature_names,
        'Weight': weights
    })
    st.bar_chart(weight_df.set_index('Feature'))