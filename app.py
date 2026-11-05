# import streamlit as st
# import pandas as pd
# import pickle
# import plotly.express as px
# import sqlite3

# # --- PAGE CONFIG ---
# st.set_page_config(page_title="Credit Risk Scorer", page_icon="🏦", layout="wide")

# # --- 1. LOAD THE SAVED MODEL ---
# @st.cache_resource
# def load_model():
#     with open('data/credit_model.pkl', 'rb') as f:
#         model = pickle.load(f)
#     return model

# model = load_model()

# # --- 2. HEADER ---
# st.title("🏦 Credit Risk Scoring Engine")
# st.markdown("Prototype for **M&T Bank** | Predicting probability of loan default.")

# # --- 3. SIDEBAR (User Inputs) ---
# st.sidebar.header("📝 Loan Application")
# st.sidebar.markdown("Enter applicant details below:")

# # Important: These inputs must match the order your model expects!
# # [Age, Credit amount, Duration]
# age = st.sidebar.number_input("Applicant Age", min_value=18, max_value=100, value=30)
# amount = st.sidebar.number_input("Credit Amount ($)", min_value=500, max_value=20000, value=5000)
# duration = st.sidebar.slider("Loan Duration (Months)", min_value=6, max_value=72, value=24)

# # --- 4. PREDICTION LOGIC ---
# if st.sidebar.button("Calculate Risk Score"):
#     # Create a DataFrame for the model
#     input_data = pd.DataFrame([[age, amount, duration]], 
#                               columns=['Age', 'Credit amount', 'Duration'])
    
#     # Get Prediction
#     prediction = model.predict(input_data)[0] # 0 or 1
#     probability = model.predict_proba(input_data)[0][1] # % chance of default
    
#     # Display Results
#     st.divider()
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("Risk Assessment")
#         if prediction == 1: # Bad Loan
#             st.error(f"⚠️ **HIGH RISK**: Application Recommended for Rejection")
#             st.metric("Default Probability", f"{probability:.1%}", delta="-High Risk", delta_color="inverse")
#         else: # Good Loan
#             st.success(f"✅ **LOW RISK**: Application Recommended for Approval")
#             st.metric("Default Probability", f"{probability:.1%}", delta="Safe", delta_color="normal")

# # --- 5. ANALYST DASHBOARD (The "Data Analyst" View) ---
# st.divider()
# st.subheader("📊 Portfolio Analysis (Data Analyst View)")

# # Load data from SQL for charts
# conn = sqlite3.connect('data/credit_risk.db')
# df_sql = pd.read_sql("SELECT * FROM loans", conn)
# conn.close()

# # Create 2 Columns for Charts
# chart1, chart2 = st.columns(2)

# with chart1:
#     st.markdown("**Loan Amounts by Purpose**")
#     # Interactive Bar Chart
#     fig = px.box(df_sql, x="Purpose", y="Credit amount", color="Purpose")
#     st.plotly_chart(fig, use_container_width=True)

# with chart2:
#     st.markdown("**Risk Distribution by Age**")
#     # Histogram
#     fig2 = px.histogram(df_sql, x="Age", color="Risk", barmode="overlay")
#     st.plotly_chart(fig2, use_container_width=True)

 





import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import sqlite3
import shap
from streamlit_shap import st_shap

# --- PAGE CONFIG ---
st.set_page_config(page_title="Credit Risk Scorer Pro", page_icon="🏦", layout="wide")

# --- 1. LOAD THE UPGRADED ASSETS ---
@st.cache_resource
def load_assets():
    # UPDATED: File names now match your analysis.ipynb exactly
    with open('data/credit_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('data/feature_columns.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    return model, feature_cols

try:
    model, feature_cols = load_assets()
except FileNotFoundError:
    st.error("⚠️ Model files not found. Please run your upgraded database_setup.py and analysis.ipynb first.")
    st.stop()

# --- 2. HEADER ---
st.title("🏦 Next-Gen Credit Risk Scoring Engine")
st.markdown("""
    **Institutional Prototype** | Leveraging XGBoost & SHAP Explainability for Transparent Lending.
""")

# --- 3. SIDEBAR (User Inputs) ---
st.sidebar.header("📝 Loan Application")

age = st.sidebar.number_input("Applicant Age", min_value=18, max_value=100, value=30)
# UPDATED: This variable 'amount' will be mapped to 'Credit_Amount' below
amount = st.sidebar.number_input("Credit Amount ($)", min_value=500, max_value=20000, value=5000)
duration = st.sidebar.slider("Loan Duration (Months)", min_value=6, max_value=72, value=24)

# Categorical Inputs (Matching your new database categories)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
purpose = st.sidebar.selectbox("Purpose", ["car", "education", "business", "housing"])

# --- 4. PREDICTION & EXPLAINABILITY LOGIC ---
if st.sidebar.button("Execute Risk Assessment"):
    # UPDATED: Column name must be 'Credit_Amount' to match your database_setup.py
    raw_input = pd.DataFrame([[age, amount, duration, sex, purpose]], 
                              columns=['Age', 'Credit_Amount', 'Duration', 'Sex', 'Purpose'])
    
    # One-Hot Encode and Reindex to match training columns
    input_df = pd.get_dummies(raw_input)
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    st.divider()
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.subheader("Decision Result")
        if prediction == 1:
            st.error(f"⚠️ **HIGH RISK**: Application Rejected")
            st.metric("Probability of Default", f"{probability:.1%}", delta_color="inverse")
        else:
            st.success(f"✅ **LOW RISK**: Application Approved")
            st.metric("Probability of Default", f"{probability:.1%}", delta_color="normal")

    with res_col2:
        st.subheader("🔍 Decision Transparency (SHAP)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], input_df.iloc[0,:]), height=250)

# --- 5. ANALYST DASHBOARD ---
st.divider()
st.subheader("📊 Institutional Portfolio Analysis")

conn = sqlite3.connect('data/credit_risk.db')
df_sql = pd.read_sql("SELECT * FROM loans", conn)
conn.close()

chart1, chart2 = st.columns(2)
with chart1:
    # UPDATED: Column name changed to 'Credit_Amount'
    fig = px.box(df_sql, x="Purpose", y="Credit_Amount", color="Purpose")
    st.plotly_chart(fig, use_container_width=True)

with chart2:
    fig2 = px.histogram(df_sql, x="Age", color="Risk", barmode="overlay")
    st.plotly_chart(fig2, use_container_width=True)