import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Fight Club Theme Config
st.set_page_config(
    page_title="🧠 Fight Club Finance Regression",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title + Vibes
st.title("💥 Fight Club: Financial Regression Chaos")
st.image("https://media.giphy.com/media/d0NnEG1WnnXqg/giphy.gif", width=800)

st.markdown("""
Welcome to **Project Regression**. You are not your bank balance.  
You are the all-singing, all-predicting financial machine.  
Upload data. Build a model. Beat the market.

> 📌 *"The first rule of finance modeling... you talk about predictions."*
""")

# Upload Section
uploaded_file = st.sidebar.file_uploader("🧾 Upload your financial CSV data:", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.header("🔍 Your Data ")

    st.subheader("📂 Data Preview")
    st.dataframe(df.head())

    st.subheader("🧮 Stats Summary")
    st.write(df.describe())

    st.subheader("❓ Missing Data Check")
    missing = df.isnull().sum()
    st.write(missing)

    if missing.sum() > 0:
        st.warning("⚠️ Missing values detected — patching it up (forward fill, then zeros).")
        df = df.fillna(method='ffill').fillna(0)
    else:
        st.success("✅ No missing values. Your data is as clean as soap.")

    st.subheader("🧹 Data Type Info")
    st.write(df.dtypes)

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] < 2:
        st.error("❌ Need at least 2 numeric columns. You brought a knife to a gunfight.")
        st.stop()

    st.subheader("🔥 Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="rocket", ax=ax)
    st.pyplot(fig)

    if numeric_df.shape[1] <= 5:
        st.subheader("👀 Pairplot: Feature Relationships")
        sns.pairplot(numeric_df)
        st.pyplot(plt.gcf())

    st.subheader("📈 Distribution Plots")
    for col in numeric_df.columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    st.header("🎯 Linear Regression Setup")

    target = st.selectbox("📌 Choose your target variable:", numeric_df.columns)
    features = st.multiselect("💡 Select predictors (the pushers of the market):", [col for col in numeric_df.columns if col != target])

    if features:
        X = numeric_df[features]
        y = numeric_df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        st.success("🧠 Model Trained. You're in the Fight Club now.")

        st.subheader("📊 Evaluation Results")
        st.markdown(f"**R² Score:** `{r2:.4f}` — {'Masterful' if r2 >= 0.9 else 'Decent'}")
        st.markdown(f"**MSE:** `{mse:.4f}` — the lower the better, remember.")

        st.subheader("🧾 Coefficients")
        st.dataframe(pd.DataFrame({
            "Feature": features,
            "Impact": model.coef_
        }))

        st.subheader("🔍 Feature Effects")
        for feature, coef in zip(features, model.coef_):
            trend = "boosts" if coef > 0 else "crashes"
            st.markdown(f"- **{feature}** {trend} **{target}** by ~**{abs(coef):.2f}** units.")

        if len(features) == 1:
            st.subheader("📉 Actual vs Predicted (1D Visual)")

            plot_df = pd.DataFrame({
                features[0]: X_test[features[0]],
                "Actual": y_test,
                "Predicted": y_pred
            }).sort_values(by=features[0])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_df[features[0]], y=plot_df["Actual"], mode='markers', name="Actual"))
            fig.add_trace(go.Scatter(x=plot_df[features[0]], y=plot_df["Predicted"], mode='lines', name="Predicted", line=dict(color='red')))
            fig.update_layout(title="📉 Fight Outcome: Reality vs Prediction", xaxis_title=features[0], yaxis_title=target)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🧯 Plot only works with 1 feature. Simplify to see the truth.")

    else:
        st.info("👊 Pick at least one feature to get in the ring.")
else:
    st.info("📥 Upload a CSV to join the fight.")
