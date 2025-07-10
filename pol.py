import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate a sample dataset for AQI prediction
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

# Create a synthetic AQI target using a weighted combination of features
data["AQI"] = (
    0.4 * data["PM2.5"] +
    0.2 * data["PM10"] +
    0.15 * data["NO2"] +
    0.1 * data["CO"] * 100 +  # scaled CO
    0.05 * data["Traffic_Volume"] / 10 +
    np.random.normal(0, 10, len(data))  # add noise
)

# Display the first few rows of the dataset
data.head()
