import pandas as pd
from scipy.stats import gumbel_r
import numpy as np

# Load the new Excel file from the specified path
file_path_new = 'E:\\HYDRO\\NILE2024\\export7_93_2023.xlsx'
df_new = pd.read_excel(file_path_new, sheet_name='Sheet1')

# Rename columns to match previous analysis
df_new.columns = ['date', 'Calibrated modeled Runoff']

# Extract year from the date
df_new['year'] = pd.to_datetime(df_new['date']).dt.year

# Extract the runoff data and the dates
runoff_data = df_new['Calibrated modeled Runoff'].values
dates = df_new['date'].values

# Fit a Gumbel distribution to the data
params = gumbel_r.fit(runoff_data)

# Calculate the return periods
sorted_indices = np.argsort(runoff_data)[::-1]
sorted_runoff = runoff_data[sorted_indices]
sorted_dates = dates[sorted_indices]
rank = np.arange(1, len(runoff_data) + 1)
return_periods = 1 / (1 - gumbel_r.cdf(sorted_runoff, *params))

# Create a DataFrame with the results
results = pd.DataFrame({
    'Date': sorted_dates,
    'Runoff': sorted_runoff,
    'Rank': rank,
    'Return Period (years)': return_periods
})

# Save the results to an Excel file
output_file_path = 'E:\\HYDRO\\NILE2024\\Gumbel_Return_Periods_Runoff_Data.xlsx'
results.to_excel(output_file_path, index=False)

# Display the results
print(results)

# Print the path to the saved file
print(f"Table saved to: {output_file_path}")
#Gumbel distribution