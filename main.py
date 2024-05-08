import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression  # Example Model
import numpy as np

import pandas as pd

# Define Ticker Symbol
ticker = "AAPL"  # Replace with your desired stock ticker

# Download data for a specific period
data = yf.download(ticker, period="5y")

# Select specific columns (e.g., Close price)
data = data[["Close"]]

# Create a new dataframe for features (predictors) and target (predicted value)
features = pd.DataFrame()

# You can add more features here based on technical indicators etc.
# For this example, we'll use the previous day's closing price
features["prev_day_close"] = data["Close"].shift(1)

# Define the target variable (predict tomorrow's closing price)
target = data["Close"]

# Scale the features and target data (often required for machine learning models)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)
scaled_target = scaler.fit_transform(target.values.reshape(-1, 1))

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, scaled_target, test_size=0.2)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

from sklearn.impute import SimpleImputer

# Create an imputer to fill NaNs with the mean strategy
imputer = SimpleImputer(strategy="mean")

# Fit the imputer on the features (X_train and X_test)
imputer.fit(X_train)
imputer.fit(X_test)

# Transform the features to impute missing values
X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)

# create and fit the LSTM network
lstm_model = Sequential()
lstm_model.add(LSTM(4, input_shape=(1, 1)))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(X_train_imputed, y_train, epochs=10, batch_size=1, verbose=2) # epochs were 100

# Define hyperparameters (adjust based on data and experimentation)
timesteps = 60  # Number of past days to consider
features = 1  # Assuming you're using only closing prices

# LSTM v2

# Create the LSTM model
lstm_model = keras.Sequential()

# First LSTM layer with 50 units
lstm_model.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(timesteps, features)))

# Second LSTM layer with 50 units
lstm_model.add(keras.layers.LSTM(units=50))

# Dense layer for output (single value prediction)
lstm_model.add(keras.layers.Dense(units=1))

# Compile the model
lstm_model.compile(loss="mse", optimizer="adam")

# Train the model on your training data
lstm_model.fit(X_train_imputed, y_train, epochs=100, batch_size=32)

# Make predictions on the testing set
lstm_predictions = lstm_model.predict(X_test)

# Calculate the mean squared error (MSE) to evaluate model performance on unseen data
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, lstm_predictions)
print("Mean Squared Error:", mse) # Mean Squared Error: 0.00026602706894208246

# Get the most recent closing price
last_day_data = data.iloc[-1]
last_day_closing_price = last_day_data["Close"]

# Scale the new data point
new_data = pd.DataFrame({"prev_day_close": [last_day_closing_price]})
new_data_scaled = scaler.transform(new_data)

# Predict the closing price for the next day (or any future date)
lstm_predicted_price = lstm_model.predict(new_data_scaled)

# Invert scaling for the predicted value
lstm_predicted_price = scaler.inverse_transform(lstm_predicted_price.reshape(1, -1))
print("Predicted Price:", lstm_predicted_price[0][0]) # Predicted Price: 182.6942 USD for day 08/05/2024



# Train a Linear Regression model
model = LinearRegression()

# Use the imputed data for training
model.fit(X_train_imputed, y_train)

# Make predictions on the testing set
predictions = model.predict(X_test)

# Calculate the mean squared error (MSE) to evaluate model performance on unseen data
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse) #Mean Squared Error: 0.0002653659649066938

# Get the most recent closing price
last_day_data = data.iloc[-1]
last_day_closing_price = last_day_data["Close"]

# Scale the new data point
new_data = pd.DataFrame({"prev_day_close": [last_day_closing_price]})
new_data_scaled = scaler.transform(new_data)

# Predict the closing price for the next day (or any future date)
predicted_price = model.predict(new_data_scaled)

# Invert scaling for the predicted value
predicted_price = scaler.inverse_transform(predicted_price.reshape(1, -1))
print("Predicted Price:", predicted_price[0][0]) # Predicted Price: 182.63214724710943 USD for day 08/05/2024