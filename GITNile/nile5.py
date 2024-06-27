import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from scipy.signal import savgol_filter
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
cnn_model.fit(X_train_cnn, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Make predictions with the CNN model
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
lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Make predictions with the LSTM model
y_prob_lstm = lstm_model.predict(X_test_lstm)

# Prepare data for the ensemble model
X_ensemble = np.vstack([y_prob_fnn.flatten(), y_prob_cnn.flatten(), y_prob_lstm.flatten()]).T
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

# Generate predictions for 2023
y_prob_fnn_2023 = fnn_model.predict(X_2023)
y_prob_cnn_2023 = cnn_model.predict(X_2023_cnn)
y_prob_lstm_2023 = lstm_model.predict(X_2023_lstm)

# Prepare ensemble data for 2023
X_ensemble_2023 = np.vstack([y_prob_fnn_2023.flatten(), y_prob_cnn_2023.flatten(), y_prob_lstm_2023.flatten()]).T

# Predict flood probabilities for 2023 using the ensemble model
y_2023_prob = ensemble_model.predict(X_ensemble_2023)

# Load the actual 2023 data
file_path_2023 = 'E:\\HYDRO\\NILE2024\\export6_2023x.csv'
data_2023_actual = pd.read_csv(file_path_2023)

# Convert date column to datetime
data_2023_actual['date'] = pd.to_datetime(data_2023_actual['date'], format='%m/%d/%Y')

# Ensure all relevant columns are strings before replacing commas
data_2023_actual['runoff'] = data_2023_actual['runoff'].astype(str).str.replace(',', '').astype(float)
data_2023_actual['precipitation'] = data_2023_actual['precipitation'].astype(str).str.replace(',', '').astype(float)
data_2023_actual['temperature'] = data_2023_actual['temperature'].astype(float)

# Feature Engineering for 2023 data
data_2023_actual['day_of_year'] = data_2023_actual['date'].dt.dayofyear
data_2023_actual['month'] = data_2023_actual['date'].dt.month
data_2023_actual['cum_rainfall_3d'] = data_2023_actual['precipitation'].rolling(window=3).sum()
data_2023_actual['cum_rainfall_7d'] = data_2023_actual['precipitation'].rolling(window=7).sum()
data_2023_actual['temp_trend_7d'] = data_2023_actual['temperature'].rolling(window=7).mean()

# Drop rows with NaN values created by rolling windows
data_2023_actual = data_2023_actual.dropna()

# Normalize the 2023 data
X_2023_actual = data_2023_actual[['day_of_year', 'month', 'cum_rainfall_3d', 'cum_rainfall_7d', 'temp_trend_7d', 'runoff']].values
X_2023_actual_scaled = scaler.transform(X_2023_actual)

# Reshape for the models
X_2023_actual_cnn = X_2023_actual_scaled.reshape(X_2023_actual_scaled.shape[0], X_2023_actual_scaled.shape[1], 1)
X_2023_actual_lstm = X_2023_actual_scaled.reshape(X_2023_actual_scaled.shape[0], 1, X_2023_actual_scaled.shape[1])

# Generate predictions for actual 2023 data
y_prob_fnn_2023_actual = fnn_model.predict(X_2023_actual_scaled)
y_prob_cnn_2023_actual = cnn_model.predict(X_2023_actual_cnn)
y_prob_lstm_2023_actual = lstm_model.predict(X_2023_actual_lstm)

# Prepare ensemble data for actual 2023
X_ensemble_2023_actual = np.vstack([y_prob_fnn_2023_actual.flatten(), y_prob_cnn_2023_actual.flatten(), y_prob_lstm_2023_actual.flatten()]).T

# Predict flood probabilities for actual 2023 using the ensemble model
y_2023_prob_actual = ensemble_model.predict(X_ensemble_2023_actual)

# Adjust Savitzky-Golay filter parameters for smoother trends
window_length = 31  # Ensure the window length is odd and appropriate
polyorder = 3  # Polynomial order

# Smooth the data for actual 2023 predictions
y_2023_prob_actual_smooth = savgol_filter(y_2023_prob_actual.flatten(), window_length=window_length, polyorder=polyorder)

# Smooth the data for synthetic 2023 predictions
y_2023_prob_smooth = savgol_filter(y_2023_prob.flatten(), window_length=window_length, polyorder=polyorder)

# Visualize the predicted flood trend for 2023 compared to actual data
fig, ax1 = plt.subplots(figsize=(14, 7))

# Primary axis for synthetic data
ax1.plot(pd.date_range(start='2023-01-01', periods=len(y_2023_prob)), y_2023_prob_smooth, label='Predicted Flood Trend for 2023 (Synthetic Data)', color='red', linestyle='--')
ax1.set_xlabel('Date')
ax1.set_ylabel('Flood Probability (Synthetic Data)', color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.legend(loc='upper left')

# First secondary axis for actual data
ax2 = ax1.twinx()
ax2.plot(data_2023_actual['date'], y_2023_prob_actual_smooth, label='Predicted Flood Trend for 2023 (Actual Data)', color='blue', linestyle='-')
ax2.set_ylabel('Flood Probability (Actual Data)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.legend(loc='upper right')

# Second secondary axis for runoff, flood threshold, and flood event
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
ax3.plot(data_2023_actual['date'], data_2023_actual['runoff'], label='Runoff', color='green')
ax3.axhline(y=flood_threshold, color='purple', linestyle='--', label='Flood Threshold')
flood_days_2023 = data_2023_actual[data_2023_actual['runoff'] > flood_threshold]['date']
ax3.scatter(flood_days_2023, data_2023_actual[data_2023_actual['runoff'] > flood_threshold]['runoff'], color='orange', label='Flood Event')
ax3.set_ylabel('Runoff', color='green')
ax3.tick_params(axis='y', labelcolor='green')
ax3.legend(loc='lower left')

fig.tight_layout()
plt.title('Predicted Flood Trend for 2023')
plt.show()

# Correlation matrix for 2023 data
plt.figure(figsize=(10, 8))
correlation_matrix_2023 = data_2023_actual[['runoff', 'precipitation', 'temperature', 'cum_rainfall_3d', 'cum_rainfall_7d', 'temp_trend_7d']].corr()
sns.heatmap(correlation_matrix_2023, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix for 2023 Data')
plt.show()

# Distribution plot of runoff for 2023 data
plt.figure(figsize=(10, 6))
sns.histplot(data_2023_actual['runoff'], kde=True)
plt.axvline(x=flood_threshold, color='r', linestyle='--', label='Flood Threshold')
plt.xlabel('Runoff (m3/s)')
plt.title('Distribution of Runoff for 2023')
plt.legend()
plt.show()

# Scatter plot of runoff vs. precipitation for 2023 data
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_2023_actual['precipitation'], y=data_2023_actual['runoff'])
plt.xlabel('Precipitation (mm)')
plt.ylabel('Runoff (m3/s)')
plt.title('Runoff vs. Precipitation for 2023')
plt.show()

# Create a pivot table for the heatmap for 2023 data
pivot_table_2023 = data_2023_actual.pivot_table(index='date', values='runoff', aggfunc='mean')

# Plot the heatmap for 2023 data
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table_2023, annot=True, cmap='YlGnBu', fmt='.1f')
plt.title('Average Discharge During 2023 by Date')
plt.xlabel('Date')
plt.ylabel('Runoff (m3/s)')
plt.show()

# Visualize Runoff over Time with Flood Events Highlighted for 2023 data
plt.figure(figsize=(14, 7))
plt.plot(data_2023_actual['date'], data_2023_actual['runoff'], label='Runoff')
plt.axhline(y=flood_threshold, color='r', linestyle='--', label='Flood Threshold')
flood_days_2023 = data_2023_actual[data_2023_actual['runoff'] > flood_threshold]['date']
plt.scatter(flood_days_2023, [data_2023_actual[data_2023_actual['date'] == day]['runoff'].values[0] for day in flood_days_2023], color='red', label='Flood Event')
plt.xlabel('Date')
plt.ylabel('Runoff (m3/s)')
plt.title('Runoff Over Time with Flood Events Highlighted for 2023')
plt.legend()
plt.show()
