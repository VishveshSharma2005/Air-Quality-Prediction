# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="🌿 Air Quality Prediction App", layout="wide")

st.title("🌿 Air Quality Prediction in Urban Areas (Case Study 4)")
st.markdown("""
This AI model predicts Air Quality Index (AQI) using data like pollutant levels, weather, and traffic.
""")

# --- Step 1: Simulate dataset or upload real one
st.sidebar.header("📁 Dataset Options")
use_sample = st.sidebar.checkbox("Use sample simulated dataset", value=True)

if use_sample:
    st.info("Using simulated dataset for demonstration.")

    # Simulate dataset
    np.random.seed(42)
    date_range = pd.date_range(start="2022-01-01", end="2022-12-31", freq="h")
    data = pd.DataFrame({
        "datetime": date_range,
        "PM2.5": np.random.uniform(10, 250, len(date_range)),
        "PM10": np.random.uniform(20, 300, len(date_range)),
        "NO2": np.random.uniform(5, 100, len(date_range)),
        "CO": np.random.uniform(0.2, 3.0, len(date_range)),
        "Temperature": np.random.uniform(10, 40, len(date_range)),
        "Humidity": np.random.uniform(20, 90, len(date_range)),
        "WindSpeed": np.random.uniform(0, 15, len(date_range)),
        "Traffic_Volume": np.random.randint(100, 1000, len(date_range))
    })

    # Synthetic AQI target
    data["AQI"] = (
        0.4 * data["PM2.5"] +
        0.2 * data["PM10"] +
        0.15 * data["NO2"] +
        0.1 * data["CO"] * 100 + 
        0.05 * data["Traffic_Volume"] / 10 +
        np.random.normal(0, 10, len(data))
    )
else:
    uploaded = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])
    if uploaded:
        data = pd.read_csv(uploaded)
    else:
        st.warning("Please upload a dataset or use the sample.")
        st.stop()

# --- Step 2: Preview dataset
st.subheader("📊 Dataset Preview")
st.dataframe(data.head())

# --- Step 3: Feature selection & train-test split
features = ["PM2.5", "PM10", "NO2", "CO", "Temperature", "Humidity", "WindSpeed", "Traffic_Volume"]
target = "AQI"

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Step 4: Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Step 5: Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

st.subheader("✅ Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("RMSE", f"{rmse:.2f}")
col3.metric("R² Score", f"{r2:.2f}")

# --- Step 6: Visualization: Actual vs Predicted
st.subheader("📈 Actual vs Predicted AQI")
fig1, ax1 = plt.subplots(figsize=(8, 4))
sns.scatterplot(x=y_test, y=y_pred, ax=ax1, alpha=0.5)
ax1.set_xlabel("Actual AQI")
ax1.set_ylabel("Predicted AQI")
ax1.set_title("Actual vs Predicted AQI")
st.pyplot(fig1)

# --- Step 7: Feature importance
st.subheader("🔍 Feature Importance")
importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
fig2, ax2 = plt.subplots(figsize=(6, 4))
importance.plot(kind='barh', ax=ax2, color='skyblue')
ax2.set_title("Feature Importance")
st.pyplot(fig2)

# --- Step 8: Predict AQI for next 24 hours (demo)
st.subheader("🧪 Predict AQI for Next 24 Hours")
future_sample = X_test.head(24)
future_pred = model.predict(future_sample)

fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(range(1, 25), future_pred, marker='o', linestyle='-', color='green')
ax3.set_xlabel("Hour Ahead")
ax3.set_ylabel("Predicted AQI")
ax3.set_title("Predicted AQI for Next 24 Hours")
st.pyplot(fig3)

st.success("✅ Done! AI helps predict air quality and supports proactive measures to reduce pollution.")

st.caption("Case Study 4: AI/ML Green Skill Workshop - Streamlit App")
