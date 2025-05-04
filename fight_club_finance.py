import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# --- Page Configuration ---
st.set_page_config(page_title="Fight Club Finance Regression", layout="wide")

# --- App Title ---
st.title("💥 Fight Club: Finance Regression Chaos")
st.image("https://media.giphy.com/media/d0NnEG1WnnXqg/giphy.gif", width=800)
st.markdown("""
Welcome to **Project Regression** — where financial chaos meets machine learning clarity.  
Upload your CSV, select features, and let the model fight for the truth.

> *"The first rule of finance modeling... you talk about predictions."*
""")

# --- File Upload ---
uploaded_file = st.sidebar.file_uploader("📄 Upload Your Financial CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.header("📊 Data Preview")
    st.dataframe(df.head())

    # --- Basic Cleaning ---
    st.subheader("🧼 Data Cleaning")
    st.write("Detecting missing values and patching up...")
    missing = df.isnull().sum()
    st.write(missing)

    if missing.sum() > 0:
        df = df.fillna(method="ffill").fillna(0)
        st.success("✅ Cleaned with forward fill + zero-fill.")
    else:
        st.info("No missing values found.")

    # --- Select Numeric Features Only ---
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        st.error("❌ Need at least 2 numeric columns.")
        st.stop()

    # --- Correlation Heatmap ---
    st.subheader("📌 Correlation Heatmap")
    corr = numeric_df.corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="reds", title="Feature Correlation")
    st.plotly_chart(fig_corr, use_container_width=True)

    # --- Feature Distributions ---
    st.subheader("📈 Feature Distributions")
    for col in numeric_df.columns:
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)

    # --- Select Features & Target ---
    st.header("🧠 Model Training: Linear Regression")
    target = st.selectbox("🎯 Select your target variable (what to predict):", numeric_df.columns)
    features = st.multiselect("📊 Select feature(s) for prediction:", [col for col in numeric_df.columns if col != target])

    if features:
        X = numeric_df[features]
        y = numeric_df[target]

        # --- Train-Test Split ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # --- Model Training ---
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # --- Evaluation Metrics ---
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        st.subheader("📉 Model Evaluation")
        st.metric(label="R² Score", value=f"{r2:.4f}")
        st.metric(label="Mean Squared Error", value=f"{mse:.2f}")

        # --- Coefficients ---
        st.subheader("📑 Model Coefficients")
        coef_df = pd.DataFrame({"Feature": features, "Impact": model.coef_})
        st.dataframe(coef_df)

        # --- Actual vs Predicted Scatter Plot ---
        st.subheader("🔍 Actual vs Predicted")
        results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        fig_pred = px.scatter(results_df, x="Actual", y="Predicted", trendline="ols",
                              title="📍 Actual vs Predicted Scatter")
        st.plotly_chart(fig_pred, use_container_width=True)

        # --- Residual Plot ---
        st.subheader("📉 Residuals Plot")
        residuals = y_test - y_pred
        fig_resid = px.histogram(residuals, nbins=30, title="Residuals Distribution")
        st.plotly_chart(fig_resid, use_container_width=True)

        st.success("🎯 Model trained and evaluated. Welcome to Fight Club, Data Edition.")
    else:
        st.info("Please select at least one feature to continue.")
else:
    st.info("📥 Upload a CSV file to get started.")
