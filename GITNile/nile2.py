import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM
from scipy.signal import savgol_filter

# Load the Excel file
file_path = 'E:\\HYDRO\\NILE2024\\export5_alldata.xlsx'
data = pd.read_excel(file_path)

# Convert date column to datetime
data['date'] = pd.to_datetime(data['date'])

# Rename columns for easier access
data.columns = ['date', 'runoff', 'precipitation', 'temperature']

# Feature Engineering
data['day_of_year'] = data['date'].dt.dayofyear
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year
data['cum_rainfall_3d'] = data['precipitation'].rolling(window=3).sum()
data['cum_rainfall_7d'] = data['precipitation'].rolling(window=7).sum()
data['temp_trend_7d'] = data['temperature'].rolling(window=7).mean()

# Drop rows with NaN values created by rolling windows
data = data.dropna()

# Define flood events based on the 95th percentile of runoff
flood_threshold = data['runoff'].quantile(0.95)
data['flood_event'] = (data['runoff'] > flood_threshold).astype(int)

# Define features and labels
X = data[['day_of_year', 'month', 'cum_rainfall_3d', 'cum_rainfall_7d', 'temp_trend_7d']]
y = data['flood_event']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a function to create and train an FNN model
def create_fnn_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the FNN model
fnn_model = create_fnn_model(X_train_scaled.shape[1])
fnn_history = fnn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Predict flood probabilities with the FNN model
y_prob_fnn = fnn_model.predict(X_test_scaled)

# Define a function to create and train a CNN model
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Reshape data for CNN
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# Train the CNN model
cnn_model = create_cnn_model((X_train_cnn.shape[1], X_train_cnn.shape[2]))
cnn_history = cnn_model.fit(X_train_cnn, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Predict flood probabilities with the CNN model
y_prob_cnn = cnn_model.predict(X_test_cnn)

# Define a function to create and train an LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Reshape data for LSTM
X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# Train the LSTM model
lstm_model = create_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
lstm_history = lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Predict flood probabilities with the LSTM model
y_prob_lstm = lstm_model.predict(X_test_lstm)

# Add predictions to the data
data['fnn_pred'] = np.nan
data['cnn_pred'] = np.nan
data['lstm_pred'] = np.nan

data.loc[y_test.index, 'fnn_pred'] = y_prob_fnn.flatten()
data.loc[y_test.index, 'cnn_pred'] = y_prob_cnn.flatten()
data.loc[y_test.index, 'lstm_pred'] = y_prob_lstm.flatten()

# Calculate annual average predictions
annual_avg_fnn = data.groupby('year')['fnn_pred'].mean().reset_index()
annual_avg_cnn = data.groupby('year')['cnn_pred'].mean().reset_index()
annual_avg_lstm = data.groupby('year')['lstm_pred'].mean().reset_index()

# Smooth the annual average predictions using Savitzky-Golay filter
window_size = 5  # You can adjust the window size
poly_order = 2   # You can adjust the polynomial order
annual_avg_fnn['smoothed'] = savgol_filter(annual_avg_fnn['fnn_pred'].fillna(0), window_size, poly_order)
annual_avg_cnn['smoothed'] = savgol_filter(annual_avg_cnn['cnn_pred'].fillna(0), window_size, poly_order)
annual_avg_lstm['smoothed'] = savgol_filter(annual_avg_lstm['lstm_pred'].fillna(0), window_size, poly_order)

# Plot the smoothed trend lines
plt.figure(figsize=(14, 7))
plt.plot(annual_avg_fnn['year'], annual_avg_fnn['smoothed'], color='red', label='FNN Predicted Flood Trend')
plt.plot(annual_avg_cnn['year'], annual_avg_cnn['smoothed'], color='green', label='CNN Predicted Flood Trend')
plt.plot(annual_avg_lstm['year'], annual_avg_lstm['smoothed'], color='orange', label='LSTM Predicted Flood Trend')

plt.xlabel('Year')
plt.ylabel('Average Predicted Flood Probability')
plt.title('Annual Average Predicted Flood Trends')
plt.legend()
plt.show()
