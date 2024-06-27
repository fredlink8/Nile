import pandas as pd
import matplotlib.pyplot as plt

# Load the uploaded Excel file
file_path = 'E:/HYDRO/NILE2024/export3.xlsx'  # Update this path to the location of your file
excel_data = pd.ExcelFile(file_path)

# Load the data from the first sheet
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Plotting the data
plt.figure(figsize=(14, 7))
plt.plot(df['date'], df['Calibrated modeled Runoff'], label='Calibrated modeled Runoff', color='blue')
plt.plot(df['date'], df['observed runoff'], label='Observed Runoff', color='orange')

# Adding titles and labels
plt.title('Runoff Comparison Over Time')
plt.xlabel('Date')
plt.ylabel('Runoff')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
