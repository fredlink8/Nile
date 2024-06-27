import pandas as pd

# Load the Excel file from the correct file path
file_path_new = 'E:\\HYDRO\\NILE2024\\export7_93_2023.xlsx'
xls_new = pd.ExcelFile(file_path_new)

# Load the data from the first sheet
df_new = pd.read_excel(file_path_new, sheet_name='Sheet1')

# Rename columns to match previous analysis
df_new.columns = ['date', 'Calibrated modeled Runoff']

# Extract year from the date
df_new['year'] = pd.to_datetime(df_new['date']).dt.year

# Rearrange columns to match the previous format
runoff_data_new = df_new[['year', 'Calibrated modeled Runoff', 'date']].dropna()

# Sort runoff values in descending order and calculate the ranks
runoff_data_sorted_new = runoff_data_new.sort_values(by='Calibrated modeled Runoff', ascending=False).reset_index(drop=True)
runoff_data_sorted_new['Rank'] = runoff_data_sorted_new.index + 1

# Calculate the return periods
n_new = len(runoff_data_sorted_new)
runoff_data_sorted_new['Return Period (years)'] = (n_new + 1) / runoff_data_sorted_new['Rank']

# Save the resulting dataframe to an Excel file
output_file_path = 'E:\\HYDRO\\NILE2024\\Adjusted_Return_Periods_Runoff_Data.xlsx'
runoff_data_sorted_new.to_excel(output_file_path, index=False)

# Display the table using standard print
print(runoff_data_sorted_new)

# Display the path to the saved file
print(f"Table saved to: {output_file_path}")
#Weibul distribution