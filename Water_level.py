import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta


csv_file_path = "Water_Tank_Level_Prediction.csv"  
df = pd.read_csv(csv_file_path)

# Convert time to datetime object and extract features
df["Time"] = pd.to_datetime(df["Time"], format="%I:%M %p")  
df["Hour_of_Day"] = df["Time"].dt.hour

# Calculate water usage rate (liters per hour)
df["Usage_Rate"] = -df["Water Level (liters)"].diff()  # Negative because water level decreases
df["Usage_Rate"].fillna(method='bfill', inplace=True)  # Fill first value with backfill

refill_threshold = 150  

# Time remaining (in hours) until refill threshold is reached
df["Time_Until_Refill"] = (df["Water Level (liters)"] - refill_threshold) / df["Usage_Rate"]


# X = df[["Water Level (liters)", "Hour_of_Day", "Usage_Rate"]]
X = df[["Water Level (liters)"]]
y = df["Time_Until_Refill"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

import pickle as pkl
pkl.dump({'model': model}, open('model.pkl', 'wb'))
