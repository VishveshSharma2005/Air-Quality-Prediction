# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Air Quality Prediction in Urban Areas", layout="wide")

st.title("🌿 Air Quality Prediction (AQI) - Case Study 4")
st.write("""
AI can help predict air quality levels, enabling timely warnings and decisions to reduce public exposure and pollution.
This demo uses historical pollutant levels, weather and traffic data to predict AQI 24 hours ahead.
""")

# --- Upload or load dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    # Sample synthetic dataset
    st.info("Using sample dataset.")
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', periods=1000, freq='H')
    df = pd.DataFrame({
        'datetime': dates,
        'PM2.5': np.random.uniform(50, 200, len(dates)),
        'PM10': np.random.uniform(80, 300, len(dates)),
        'NO2': np.random.uniform(10, 80, len(dates)),
        'CO': np.random.uniform(0.5, 3.0, len(dates)),
        'temperature': np.random.uniform(15, 40, len(dates)),
        'humidity': np.random.uniform(20, 80, len(dates)),
        'wind_speed': np.random.uniform(0, 15, len(dates)),
        'vehicle_count': np.random.randint(50, 500, len(dates)),
        'AQI': np.random.uniform(100, 300, len(dates))
    })

# --- Data preview
st.subheader("📊 Dataset Overview")
st.dataframe(df.head())

# --- Feature engineering
st.subheader("⚙️ Data Preprocessing & Feature Engineering")
df = df.fillna(method='ffill')
features = ['PM2.5', 'PM10', 'NO2', 'CO', 'temperature', 'humidity', 'wind_speed', 'vehicle_count']
target = 'AQI'

# --- Split data
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# --- Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Predict
y_pred = model.predict(X_test)

# --- Evaluation
st.subheader("📈 Model Performance")
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# --- Plot Actual vs Predicted
fig, ax = plt.subplots(figsize=(8, 4))
sns.scatterplot(x=y_test, y=y_pred, ax=ax)
ax.set_xlabel("Actual AQI")
ax.set_ylabel("Predicted AQI")
ax.set_title("Actual vs Predicted AQI")
st.pyplot(fig)

# --- Feature importance
st.subheader("🔍 Feature Importance")
importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
st.bar_chart(importance)

# --- Predict AQI for new input
st.subheader("🧪 Predict AQI for Next 24 Hours (Example)")
sample_input = X_test.head(24)
future_pred = model.predict(sample_input)

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(range(24), future_pred, marker='o', label='Predicted AQI')
ax2.set_xlabel("Hour Ahead")
ax2.set_ylabel("AQI")
ax2.set_title("Predicted AQI for Next 24 Hours")
ax2.legend()
st.pyplot(fig2)

st.info("✅ This case study demonstrates how AI models can predict air quality and help manage pollution in cities.")
