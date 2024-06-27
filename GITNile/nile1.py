import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
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
fnn_history = fnn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the FNN model
loss_fnn, accuracy_fnn = fnn_model.evaluate(X_test_scaled, y_test)
y_prob_fnn = fnn_model.predict(X_test_scaled)
y_pred_fnn = (y_prob_fnn > 0.5).astype(int)
classification_report_fnn = classification_report(y_test, y_pred_fnn)
roc_auc_fnn = roc_auc_score(y_test, y_prob_fnn)

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

# Evaluate the CNN model
loss_cnn, accuracy_cnn = cnn_model.evaluate(X_test_cnn, y_test)
y_prob_cnn = cnn_model.predict(X_test_cnn)
y_pred_cnn = (y_prob_cnn > 0.5).astype(int)
classification_report_cnn = classification_report(y_test, y_pred_cnn)
roc_auc_cnn = roc_auc_score(y_test, y_prob_cnn)

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

# Evaluate the LSTM model
loss_lstm, accuracy_lstm = lstm_model.evaluate(X_test_lstm, y_test)
y_prob_lstm = lstm_model.predict(X_test_lstm)
y_pred_lstm = (y_prob_lstm > 0.5).astype(int)
classification_report_lstm = classification_report(y_test, y_pred_lstm)
roc_auc_lstm = roc_auc_score(y_test, y_prob_lstm)

# Train a Random Forest model for feature importance
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Get feature importances
importances = rf_model.feature_importances_
features = X.columns
indices = np.argsort(importances)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# Visualize model training and validation loss for FNN
plt.figure(figsize=(10, 6))
plt.plot(fnn_history.history['loss'])
plt.plot(fnn_history.history['val_loss'])
plt.title('FNN Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Visualize model training and validation accuracy for FNN
plt.figure(figsize=(10, 6))
plt.plot(fnn_history.history['accuracy'])
plt.plot(fnn_history.history['val_accuracy'])
plt.title('FNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Visualize model training and validation loss for CNN
plt.figure(figsize=(10, 6))
plt.plot(cnn_history.history['loss'])
plt.plot(cnn_history.history['val_loss'])
plt.title('CNN Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Visualize model training and validation accuracy for CNN
plt.figure(figsize=(10, 6))
plt.plot(cnn_history.history['accuracy'])
plt.plot(cnn_history.history['val_accuracy'])
plt.title('CNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Visualize model training and validation loss for LSTM
plt.figure(figsize=(10, 6))
plt.plot(lstm_history.history['loss'])
plt.plot(lstm_history.history['val_loss'])
plt.title('LSTM Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Visualize model training and validation accuracy for LSTM
plt.figure(figsize=(10, 6))
plt.plot(lstm_history.history['accuracy'])
plt.plot(lstm_history.history['val_accuracy'])
plt.title('LSTM Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Compare model performance
print("FNN Classification Report:\n", classification_report_fnn)
print("FNN ROC AUC Score:", roc_auc_fnn)
print("\nCNN Classification Report:\n", classification_report_cnn)
print("CNN ROC AUC Score:", roc_auc_cnn)
print("\nLSTM Classification Report:\n", classification_report_lstm)
print("LSTM ROC AUC Score:", roc_auc_lstm)

# Summarize model performance
performance_summary = {
    'Model': ['FNN', 'CNN', 'LSTM'],
    'Accuracy': [accuracy_fnn, accuracy_cnn, accuracy_lstm],
    'ROC AUC': [roc_auc_fnn, roc_auc_cnn, roc_auc_lstm]
}
performance_df = pd.DataFrame(performance_summary)

# Print the performance summary
print("\nModel Performance Summary:")
print(performance_df)

# Plot model performance comparison
plt.figure(figsize=(10, 6))
plt.bar(performance_df['Model'], performance_df['Accuracy'], color=['blue', 'green', 'red'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(performance_df['Model'], performance_df['ROC AUC'], color=['blue', 'green', 'red'])
plt.xlabel('Model')
plt.ylabel('ROC AUC Score')
plt.title('Model ROC AUC Comparison')
plt.show()

# Plotting the Flood Events with Runoff over Time
plt.figure(figsize=(14, 7))
plt.plot(data['date'], data['runoff'], label='Runoff')
plt.axhline(y=flood_threshold, color='r', linestyle='--', label='Flood Threshold')
flood_days = data[data['flood_event'] == 1]['date']
plt.scatter(flood_days, [data[data['date'] == day]['runoff'].values[0] for day in flood_days], color='red', label='Flood Event')
plt.xlabel('Date')
plt.ylabel('Runoff (m3/s)')
plt.title('Runoff Over Time with Flood Events Highlighted')
plt.legend()
plt.show()

# Pair Plot of Features
sns.pairplot(data[['runoff', 'precipitation', 'temperature', 'cum_rainfall_3d', 'cum_rainfall_7d', 'temp_trend_7d', 'flood_event']], hue='flood_event')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = data[['runoff', 'precipitation', 'temperature', 'cum_rainfall_3d', 'cum_rainfall_7d', 'temp_trend_7d']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Distribution Plot of Runoff
plt.figure(figsize=(10, 6))
sns.histplot(data['runoff'], kde=True)
plt.axvline(x=flood_threshold, color='r', linestyle='--', label='Flood Threshold')
plt.xlabel('Runoff (m3/s)')
plt.title('Distribution of Runoff')
plt.legend()
plt.show()

# Scatter Plot of Runoff vs. Precipitation
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['precipitation'], y=data['runoff'], hue=data['flood_event'])
plt.xlabel('Precipitation (mm)')
plt.ylabel('Runoff (m3/s)')
plt.title('Runoff vs. Precipitation')
plt.legend(title='Flood Event')
plt.show()

# Create a pivot table for the heatmap
pivot_table = data[data['flood_event'] == 1].pivot_table(index='year', columns='month', values='runoff', aggfunc='mean')

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.1f')
plt.title('Average Discharge During Flood Events by Month and Year')
plt.xlabel('Month')
plt.ylabel('Year')
plt.show()