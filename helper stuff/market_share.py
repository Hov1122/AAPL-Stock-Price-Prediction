# import pandas as pd

# # Load the dataset
# data = pd.read_csv("vendor-ww-monthly-201003-202401.csv")

# # Convert all columns (except 'Date') to numeric, forcing errors to NaN
# for col in data.columns:
#     if col != 'Date':
#         data[col] = pd.to_numeric(data[col], errors='coerce')

# # Fill NaN values with 0 for sales data
# data = data.fillna(0)

# # Calculate total daily sales
# data['Total_Sales'] = data.drop(columns=['Date']).sum(axis=1)

# # Calculate Apple's market share
# data['Apple_Market_Share'] = (data['Apple'] / data['Total_Sales']) * 100

# # Select only Date and Apple's market share
# apple_market_share_data = data[['Date', 'Apple_Market_Share']]

# # Save to a new CSV file
# apple_market_share_data.to_csv("apple_market_share.csv", index=False)


# import pandas as pd

# # Load the daily and monthly datasets
# daily_data = pd.read_csv("aapl_combined_dataset.csv")
# monthly_data = pd.read_csv("apple_market_share.csv")

# #import pandas as pd


# # Ensure 'Date' is in datetime format
# daily_data['Date'] = pd.to_datetime(daily_data['Date'])
# monthly_data['Date'] = pd.to_datetime(monthly_data['Date'])

# # Set monthly 'Date' to the start of each month (if it isn't already)
# monthly_data['Date'] = monthly_data['Date'].dt.to_period('M').dt.to_timestamp()

# # Sort monthly data by date in ascending order
# monthly_data = monthly_data.sort_values(by='Date')

# # Reindex monthly data to daily format using the date range from the daily data
# date_range = pd.date_range(start=daily_data['Date'].min(), end=daily_data['Date'].max(), freq='D')
# monthly_data_daily = (
#     monthly_data.set_index('Date')
#     .reindex(date_range, method='ffill')  # forward-fill within each month
#     .bfill()                              # backfill to cover the start of the data range
#     .rename_axis('Date')
#     .reset_index()
# )

# # Merge the expanded monthly data with the daily data
# merged_data = pd.merge(daily_data, monthly_data_daily, on='Date', how='left')

# # Save the result to a new CSV file
# merged_data.to_csv("merged_daily_data.csv", index=False)

# Load the data

import pandas as pd

# Load the datasets
daily_data = pd.read_csv("aapl_combined_dataset2.csv", parse_dates=["Date"])  # Replace with your daily dataset filename
economic_data = pd.read_csv("filled_ndxt_data.csv", parse_dates=["Date"])

# Sort the data by Date
daily_data = daily_data.sort_values("Date")
economic_data = economic_data.sort_values("Date")

new_column = economic_data.rename(columns={"Open": "nxtd_open"})["nxtd_open"]

# Append the column to df1
combined_data = pd.concat([daily_data, new_column.reset_index(drop=True)], axis=1)


# # Step 1: Forward-fill the economic data to match daily frequency
# # Reindex economic_data to daily frequency (assuming it's at least monthly)
# date_range = pd.date_range(start=daily_data['Date'].min(), end=daily_data['Date'].max(), freq='D')
# economic_data.set_index('Date', inplace=True)

# # Forward-fill missing values for the economic data
# economic_data_daily = economic_data.reindex(date_range, method='ffill')

# # Reset the index so 'Date' is a column again
# economic_data_daily = economic_data_daily.reset_index().rename(columns={'index': 'Date'})

# # Step 2: Merge the daily dataset with the forward-filled economic data
# combined_data = pd.merge(daily_data, economic_data_daily, on='Date', how='left')

# # Step 3: Fill any remaining NaN values (optional)
# combined_data.fillna(0, inplace=True)

# # Output the result
# print(combined_data.isna().sum())  # Check for any remaining missing values
combined_data.to_csv("aapl_combined_dataset3.csv", index=False)

print("Data has been saved to combined_daily_economic_data.csv")
