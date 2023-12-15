"""
Bitcoin Price Prediction using LSTM

This script demonstrates the use of Long Short-Term Memory (LSTM) neural networks to predict Bitcoin prices.
The dataset used contains historical Bitcoin price data, and the LSTM model is trained to learn patterns and
make predictions.

this code is runned in windows WSL - Ubuntu 22.04

Dependencies:
- pip install numpy
- pip install pandas
- pip install keras
- pip install tensorflow[and-cuda] + CUDA driver (from 2.10 version and above Windows GPU training is not supported only in linux)
https://github.com/ashishpatel26/Cuda-installation-on-WSL2-Ubuntu-20.04-and-Windows11
- pip install matplotlib
- pip install scikit-learn

Note: Ensure the required libraries are installed before running the script.
"""

# Import necessary libraries
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import platform
from sklearn.preprocessing import MinMaxScaler

# Check if a GPU is available and set it up
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU is available.")
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
else:
    print("No GPU found, running on CPU.")

# handling different paths for Windows and WSL
if platform.system() == 'Windows':
    df = pd.read_csv("C:/Users/ati/Desktop/BTC-USD_price_data.csv")
else:
    matplotlib.use('TkAgg')
    df = pd.read_csv("/mnt/c/Users/ati/Desktop/BTC-USD_price_data.csv")

# Preprocess the data
df = df.drop("Date", axis=1)
df_data_percentage = 100   # in percentage (tweak this parametar if you want shorter dataframe so CPU trains model faster)
percentage = int(len(df) * (df_data_percentage / 100))
df = df[:percentage]

# Set parameters for data preparation and model
days_train_Y = []
days_to_predict = 120   
look_back = days_to_predict

# Function to generate input sequences and corresponding output values
def generate_sequences(dataset, look_back):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        # Reshape each sequence to be 2D
        sequence = dataset[i:(i + look_back)]  # Removed the second index
        X.append(sequence)
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)

# Split data into training and testing sets
train_data = df['Close'][:-look_back].to_numpy()
test_data = df['Close'][-look_back*2:].to_numpy().reshape(-1, 1)

# Scale the data to the range [-1, 1] just because tanh activation requires this, this code can be done without reshaping but you need relu activation
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).reshape(-1)
test_data_scaled = scaler.transform(test_data).reshape(-1, 1)

# Split data into training and testing sets for LSTM
train_X, train_Y = generate_sequences(train_data_scaled, look_back)
test_X, test_Y = generate_sequences(test_data_scaled, look_back)

# Generate time indices for plotting
for i in range(look_back):
    days_train_Y.append(len(train_Y) + i)

# Build LSTM model
model = Sequential()
model.add(LSTM(look_back, input_shape=(look_back, 1), activation='tanh')) #GPU can train LSTM model only with tanh activation, otherwise it will run CPU
model.add(Dense(units=look_back/2, activation='tanh'))
model.add(Dense(units=look_back/4, activation='tanh'))
model.add(Dense(units=look_back/8, activation='tanh'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', run_eagerly=True)

# Train the model
model.fit(train_X, train_Y, epochs=50, batch_size=32, verbose=1)

# Predict on the test set (known data)
prediction = model.predict(test_X)
prediction_flat = prediction.flatten()
difference = prediction_flat - test_Y

# Predict on the test set (unknown data)
# to be done

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(days_train_Y, test_Y, label='True Prices', linewidth=2)
plt.plot(days_train_Y, prediction_flat, label='Predicted Prices', linestyle='dashed', linewidth=2)
plt.plot(train_Y, label='True Prices', linewidth=2)
plt.title('Bitcoin Price Prediction using LSTM')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
