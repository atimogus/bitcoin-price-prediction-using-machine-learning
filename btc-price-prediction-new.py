import os
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def setup_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print("GPU is available.")
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    else:
        print("No GPU found, running on CPU.")

# Call the function to perform the GPU setup
setup_gpu()

# Load CSV data into a DataFrame
data_path = "/mnt/c/Users/ati/Desktop/ati-yd/BTC-USD_price_data.csv" if platform.system() != 'Windows' else "C:/Users/ati/Desktop/ati-yd/BTC-USD_price_data.csv"
df = pd.read_csv(data_path)

# Preprocess the data by dropping unnecessary columns and selecting a subset
df = df.drop("Date", axis=1)
df_percentage =  100
df_close = df['Close'][:int(len(df) * (df_percentage /  100))].to_numpy()

# Generate input sequences and corresponding output values
days_trained_Xval = []
seq_lenght =  7
num_of_days_to_predict = 60
histogram_density =  5

#scaler must be called only once because it scales every dataframe at its rate so if df seperated each df will have its scale
scaler = MinMaxScaler(feature_range=(-1,  1))
df_scaled = scaler.fit_transform(df_close.reshape(-1,  1)).reshape(-1)

train_data = df_scaled[:-(seq_lenght + num_of_days_to_predict)]
test_data = df_scaled[-(seq_lenght + num_of_days_to_predict):]


days_trained_Xval = list(range(len(train_data) +  1, len(train_data) +  1 + len(test_data)))


# # provjera podataka jel su dobro struktuirani
# plt.figure(figsize=(16,   8))
# plt.plot(train_data)
# plt.plot(days_trained_Xval, test_data)
# plt.title('Scaled Training and Test Data')
# plt.xlabel('Time Steps')
# plt.ylabel('Scaled Close Prices')
# plt.grid(True)
# plt.show()

def create_sequences(data, seq_lenght):
    X, Y = [], []
    for i in range(len(data)-seq_lenght):
        # Each sequence is of the form [x(t-look_back), ..., x(t-1)]
        sequence = data[i:i+seq_lenght]
        X.append(sequence)
        # The corresponding target value is x(t)
        Y.append(data[i+seq_lenght])
    return np.array(X), np.array(Y)

train_seq, train_Y = create_sequences(train_data , seq_lenght)
test_seq, test_Y = create_sequences(test_data , seq_lenght)


# Build and compile the LSTM model
model = Sequential()
model.add(LSTM(seq_lenght, input_shape=(seq_lenght,  1), activation='tanh'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', run_eagerly=True)

# Train the model
model.fit(train_seq, train_Y, epochs=30, batch_size=32, verbose=1)

# Predict using the model
prediction = model.predict(test_seq)
prediction_1d = np.array(prediction).flatten()

# Inverse transform the predictions
actual_prediction_price = scaler.inverse_transform(prediction).flatten()

test_data_2d = test_data.reshape(-1,  1)
test_data_rescaled = scaler.inverse_transform(test_data_2d)

# Calculate errors
errors = []
for i in range(len(actual_prediction_price)):
    temp = ((actual_prediction_price[i] - test_data_rescaled[i]) / test_data_rescaled[i]) *  100
    errors.append(abs(temp))
errors_1d = np.array(errors).flatten()
# Define the bin edges for the histogram
bin_edges = np.linspace(np.min(errors_1d), np.max(errors_1d), histogram_density+1)


# Uzimamo posljednju sekvencu testnih podataka za predikciju
last_sequence = test_seq[-1:]
# Uzimamo posljednju predviđenu cijenu
last_prediction = prediction_1d[-1]

# Inicijaliziramo novi array za ažuriranu sekvencu
new_sequence = np.append(last_sequence[0][1:], last_prediction).reshape(seq_lenght, 1)

# Inicijaliziramo array za pohranu novih predikcija
new_predictions = []

for _ in range(num_of_days_to_predict):
    # Predviđamo cijenu koristeći ažuriranu sekvencu
    new_pred = model.predict(new_sequence.reshape(1, seq_lenght, 1)).flatten()[0]
    # Dodajemo novu predikciju u listu predikcija
    new_predictions.append(new_pred)
    # Ažuriramo sekvencu za sljedeću iteraciju
    new_sequence = np.append(new_sequence[1:], new_pred).reshape(seq_lenght, 1)

# Ispisujemo nove predikcije
new_predictions_scaled = scaler.inverse_transform(np.array(new_predictions).reshape(-1, 1)).flatten()



fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(24,  6))
# Histogram of percentage errors
axes[0].hist(errors_1d, bins=bin_edges, edgecolor='black')
axes[0].set_xlabel('velicina greske u postotcima')
axes[0].set_ylabel('Frekvencija')

days_trained_X = list(range(len(train_data) , len(train_data) +  num_of_days_to_predict))

# Ažuriramo dane za nove predikcije
new_days_trained_X = list(range(days_trained_X[-1] + 1, days_trained_X[-1] + 1 + len(new_predictions)))

# Bitcoin price prediction plot
axes[1].plot(new_days_trained_X, new_predictions_scaled, label='Nove predikcije', linestyle='dotted', linewidth=2, color='red')
axes[1].plot(days_trained_X, actual_prediction_price, label='Predicted Prices', linestyle='dashed', linewidth=2)
axes[1].plot(df_close[:-len(test_data)], label='True Prices', linewidth=4)
axes[1].plot(df_close, label='FULL', linewidth=2)
axes[1].set_xlabel('Vrijeme u danima')
axes[1].set_ylabel('Cjena')
axes[1].legend()
axes[1].grid(True)

plt.show()


