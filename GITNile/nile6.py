import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM

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
X = data[['day_of_year', 'month', 'cum_rainfall_3d', 'cum_rainfall_7d', 'temp_trend_7d', 'runoff']]
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
fnn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Make predictions with the FNN model
y_prob_fnn_test = fnn_model.predict(X_test_scaled)

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
cnn_model.fit(X_train_cnn, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Make predictions with the CNN model
y_prob_cnn_test = cnn_model.predict(X_test_cnn)

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
lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Make predictions with the LSTM model
y_prob_lstm_test = lstm_model.predict(X_test_lstm)

# Prepare data for the ensemble model
X_ensemble = np.vstack([y_prob_fnn_test.flatten(), y_prob_cnn_test.flatten(), y_prob_lstm_test.flatten()]).T
y_ensemble = y_test[:len(X_ensemble)]  # Ensure the target has the same length as the features

# Split the ensemble data into training and testing sets
X_ensemble_train, X_ensemble_test, y_ensemble_train, y_ensemble_test = train_test_split(X_ensemble, y_ensemble, test_size=0.2, random_state=42)

# Define the ensemble DNN model
def create_ensemble_dnn_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the ensemble DNN model
ensemble_model = create_ensemble_dnn_model(X_ensemble_train.shape[1])
ensemble_model.fit(X_ensemble_train, y_ensemble_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Make predictions with the ensemble model
y_prob_ensemble_test = ensemble_model.predict(X_ensemble_test)

# Use the initial models to generate predictions for 2023 synthetic data
data_2023 = pd.DataFrame({
    'day_of_year': range(1, 366),
    'month': np.repeat(range(1, 13), [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]),
    'cum_rainfall_3d': np.zeros(365),
    'cum_rainfall_7d': np.zeros(365),
    'temp_trend_7d': np.zeros(365),
    'runoff': np.zeros(365)
})

# Calculate historical averages and fill synthetic data
historical_avg = data.groupby('day_of_year').mean().reset_index()
data_2023['cum_rainfall_3d'] = historical_avg['cum_rainfall_3d']
data_2023['cum_rainfall_7d'] = historical_avg['cum_rainfall_7d']
data_2023['temp_trend_7d'] = historical_avg['temp_trend_7d']

X_2023 = scaler.transform(data_2023.values)

# Reshape for the models
X_2023_cnn = X_2023.reshape(X_2023.shape[0], X_2023.shape[1], 1)
X_2023_lstm = X_2023.reshape(X_2023.shape[0], 1, X_2023.shape[1])

# Generate predictions for 2023 synthetic data
y_prob_fnn_2023 = fnn_model.predict(X_2023)
y_prob_cnn_2023 = cnn_model.predict(X_2023_cnn)
y_prob_lstm_2023 = lstm_model.predict(X_2023_lstm)

# Prepare ensemble data for 2023
X_ensemble_2023 = np.vstack([y_prob_fnn_2023.flatten(), y_prob_cnn_2023.flatten(), y_prob_lstm_2023.flatten()]).T

# Predict flood probabilities for 2023 synthetic data using the ensemble model
y_prob_ensemble_2023 = ensemble_model.predict(X_ensemble_2023)

# Ensure the lengths of all predictions are the same for both synthetic and actual data
min_length = min(len(y_prob_fnn_test), len(y_prob_cnn_test), len(y_prob_lstm_test), len(y_prob_ensemble_test),
                 len(y_prob_fnn_2023), len(y_prob_cnn_2023), len(y_prob_lstm_2023), len(y_prob_ensemble_2023))

# Trim the predictions to the same length
y_prob_fnn_test = y_prob_fnn_test[:min_length]
y_prob_cnn_test = y_prob_cnn_test[:min_length]
y_prob_lstm_test = y_prob_lstm_test[:min_length]
y_prob_ensemble_test = y_prob_ensemble_test[:min_length]
y_prob_fnn_2023 = y_prob_fnn_2023[:min_length]
y_prob_cnn_2023 = y_prob_cnn_2023[:min_length]
y_prob_lstm_2023 = y_prob_lstm_2023[:min_length]
y_prob_ensemble_2023 = y_prob_ensemble_2023[:min_length]

# Create a DataFrame to hold the predicted probabilities for both synthetic and actual data
predictions_df = pd.DataFrame({
    'FNN (Actual)': y_prob_fnn_test.flatten(),
    'CNN (Actual)': y_prob_cnn_test.flatten(),
    'LSTM (Actual)': y_prob_lstm_test.flatten(),
    'Ensemble (Actual)': y_prob_ensemble_test.flatten(),
    'FNN (Synthetic)': y_prob_fnn_2023.flatten(),
    'CNN (Synthetic)': y_prob_cnn_2023.flatten(),
    'LSTM (Synthetic)': y_prob_lstm_2023.flatten(),
    'Ensemble (Synthetic)': y_prob_ensemble_2023.flatten()
})

# Create box plots to compare the predicted probabilities for both synthetic and actual data
plt.figure(figsize=(16, 10))
sns.boxplot(data=predictions_df)
plt.title('Comparison of Predicted Flood Probabilities by Different Models (Actual and Synthetic Data)')
plt.ylabel('Predicted Probability')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.show()
