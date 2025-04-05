# import pandas as pd

# # Load the monthly data CSV
# monthly_data = pd.read_csv('T20YIEM.csv')

# # Example function to convert monthly data to quarterly data (starting at the beginning of each quarter)
# def convert_monthly_to_quarterly(monthly_data):
#     """
#     Convert monthly data to quarterly data by resampling to the first month of each quarter.
#     """
#     # Ensure the 'Date' column is in datetime format
#     monthly_data['Date'] = pd.to_datetime(monthly_data['Date'])
    
#     # Set the 'Date' column as the index
#     monthly_data.set_index('Date', inplace=True)
    
#     # Resample the data to quarterly frequency, taking the first value in each quarter
#     quarterly_data = monthly_data.resample('QS').mean().round(3)  # 'Q' stands for quarterly frequency
    
#     # Reset the index to have 'Date' as a column again
#     quarterly_data.reset_index(inplace=True)
    
#     return quarterly_data

# # Convert to quarterly data
# quarterly_data = convert_monthly_to_quarterly(monthly_data)

# # Print the resulting quarterly data
# print(quarterly_data)

# # Optionally, save the quarterly data to a new CSV file
# quarterly_data.to_csv('inflation_quarterly.csv', index=False)

import pandas as pd

# Load the CSV file
file_path = 'AAPL_combined_income_eps.csv'  # Replace with the path to your CSV file
data = pd.read_csv(file_path)

# Reverse the rows in the DataFrame
reversed_data = data.iloc[::-1].reset_index(drop=True)

# Save the reversed data to a new CSV file (optional)
output_file_path = 'reversed_file.csv'  # Replace with your desired output file name
reversed_data.to_csv(output_file_path, index=False)

# Print the reversed data
print(reversed_data)
