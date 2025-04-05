# import pandas as pd

# # Load the data
# eps_data = pd.read_csv("AAPL_combined_income_eps.csv", parse_dates=["Date"])
# historical_data = pd.read_csv("filled_AAPL_historical_data.csv", parse_dates=["Date"])
# sentiment_data = pd.read_csv("AAPL_sentiment_data.csv", parse_dates=["Date"])

# # Sort the data by Date
# eps_data = eps_data.sort_values("Date")
# historical_data = historical_data.sort_values("Date")
# sentiment_data = sentiment_data.sort_values("Date")

# # Step 1: Create a new column 'Quarter_Start' to identify the first date of each quarter
# eps_data['Quarter_Start'] = eps_data['Date'] - pd.DateOffset(months=3) + pd.DateOffset(days=1)

# # Step 2: Create a column to identify the end date of the quarter (quarter end)
# eps_data['Quarter_End'] = eps_data['Date']

# # Step 3: We will now merge the data based on which quarter the 'Date' in historical_data falls into
# # We need to assign the correct 'Gross_Profit', 'Revenue', 'Net_Income', 'EPS' values to each day within the quarter.

# # Initialize a list to collect merged data
# merged_data = []

# # Loop through the historical data and match each date to the appropriate quarter
# for index, row in historical_data.iterrows():
#     # Find the correct quarter data for the current row's date
#     quarter_data = eps_data[(eps_data['Quarter_Start'] <= row['Date']) & (eps_data['Quarter_End'] >= row['Date'])]
    
#     # If a match is found, add the values to the row
#     if not quarter_data.empty:
#         # Get the first row (since there's only one row per quarter)
#         quarter_row = quarter_data.iloc[0]
#         # Concatenate the financial data with the current row
#         merged_row = pd.concat([row, quarter_row[['Gross_Profit', 'Revenue', 'Net_Income', 'EPS']]])
#         merged_data.append(merged_row)

# # Convert the merged data back to a DataFrame
# final_data = pd.DataFrame(merged_data)

# # Step 4: Merge sentiment data
# final_data = pd.merge(final_data, sentiment_data, on="Date", how="left")

# # Step 5: Fill any remaining NaNs (sentiment or financial) with 0
# final_data['Sentiment_Score'].fillna(0, inplace=True)
# final_data[['Gross_Profit', 'Revenue', 'Net_Income', 'EPS']] = final_data[['Gross_Profit', 'Revenue', 'Net_Income', 'EPS']].fillna(0)

# # Output the results
# print(final_data.isna().sum())
# final_data.to_csv("aapl_combined_dataset.csv", index=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import os

# Load the data
# data = pd.read_csv("aapl_combined_dataset.csv", parse_dates=['Date'], index_col='Date')
data = pd.read_csv("csv_2023/aapl_combined_dataset3.csv", parse_dates=['Date'])
sent_data = pd.read_csv("csv_2023/aggregated_daily_no_weekends.csv", parse_dates=['Date'])
data = data.drop(columns=['Sentiment_Score'])
sent_data = sent_data.drop(columns=['mean_sentiment'])
data = data[data['Date'] >= '2022-03-01']

data = pd.merge(data, sent_data, on='Date')
data.set_index('Date', inplace=True)
# if 'Sentiment_Score' in data.columns:
#     # Round up the Sentiment_Score column to 4 decimal places
#     data['Sentiment_Score'] = data['Sentiment_Score'].round(4)

# ❯ python3 tt.py
# Selected Features based on correlation: Index(['Open', 'High', 'Low', 'Close', 'Adj Close', 'CPIAUCSL', 'EPS',
#        'Gross_Profit', 'Net_Income', 'Revenue', 'FEDFUNDS', 'Sentiment_Score',
#        'Volume', 'UNRATE', 'Apple_Market_Share'],
#       dtype='object')

# Mean Squared Error: 0.013894838815556391
#                Feature    Importance
# 4            Adj Close  4.055469e-01
# 3                Close  3.910400e-01
# 2                  Low  1.178428e-01
# 1                 High  7.350398e-02
# 0                 Open  1.152567e-02
# 5             CPIAUCSL  4.573177e-04
# 13              UNRATE  7.827018e-05
# 8           Net_Income  1.249028e-06
# 12              Volume  8.901901e-07
# 9              Revenue  7.806385e-07
# 6                  EPS  5.346379e-07
# 7         Gross_Profit  5.116235e-07
# 10            FEDFUNDS  4.701984e-07
# 14  Apple_Market_Share  4.322201e-07
# 11     Sentiment_Score  2.058647e-07
#Sentiment_Score,Apple_Market_Share,CPIAUCSL,FEDFUNDS,GDP,UNRAT
#Date,Open,High,Low,Close,Adj Close,Volume,Gross_Profit,Revenue,Net_Income,EPS,Sentiment_Score,Apple_Market_Share,CPIAUCSL,FEDFUNDS,GDP,UNRATE
data = data.drop(columns=["EPS","FEDFUNDS","Gross_Profit","Revenue", "Net_Income", "UNRATE", "CPIAUCSL"])
#sax 6.96
#aranc economical 3.26
#aranc economical, market_share 5.77
#aranc economical, market_share, sentiment 4.06
#market share ov 4.91

data.index.freq = 'B'
# Preprocess the data
# Handle missing values (if any)
data.fillna(method='ffill', inplace=True)

# Feature Scaling
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences
def create_sequences(dataset, look_back=60):
  X, y = [], []
  for i in range(len(dataset) - look_back - 1):
    X.append(dataset[i:(i+look_back), :])
    y.append(dataset[i + look_back, 0])  # Predict the 'Open' price
  return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)

# Split into training and testing sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test, y_train, y_test = X[0:train_size,:], X[train_size:len(X),:], y[0:train_size], y[train_size:len(X)]
y_test_unscaled = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_data.shape[1] - 1))), axis=1))[:, 0]

# Check if the model file exists
model_file = 'stock_prediction_model.h5'
if os.path.exists(model_file):
    # Load the saved model
    model = load_model(model_file)
    print("Loaded existing model.")
else:
    # Build and train the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=50, batch_size=32)
    model.save(model_file)
    print("Model trained and saved.")

# Check the shape before reshaping
print(X_test.shape)
predicted_stock_price = model.predict(X_test)
# Assuming the scaler was fitted on data with 11 features, reshape X_test appropriately
X_test = X_test.reshape((X_test.shape[0], -1))  # Flatten into 2D if necessary

# Ensure that you are only passing the correctly shaped data for inverse scaling
# Reshape predicted stock prices and target values to align with the scaler's original input shape (11 features)

# Inverse scaling for predicted stock prices
inv_predicted_stock_price = np.concatenate((predicted_stock_price, X_test[:, 1:]), axis=1)

# Ensure that inv_predicted_stock_price only has the features for scaling
print(f"inv_predicted_stock_price shape before inverse transform: {inv_predicted_stock_price.shape}")

# Inverse transform only on the relevant features (those that the scaler was trained on)
inv_predicted_stock_price = scaler.inverse_transform(inv_predicted_stock_price[:, :11])  # Use only the first 11 features
inv_predicted_stock_price = inv_predicted_stock_price[:, 0]  # Extract only the predicted values

# Inverse scaling for actual prices
y_test = y_test.reshape((len(y_test), 1))  # Ensure y_test has the correct shape

# Concatenate y_test with relevant X_test columns
inv_y = np.concatenate((y_test, X_test[:, 1:]), axis=1)

# Inverse transform only on the relevant features (those that the scaler was trained on)
inv_y = scaler.inverse_transform(inv_y[:, :11])  # Use only the first 11 features
inv_y = inv_y[:, 0]  # Extract the actual values
# bias = np.mean(inv_predicted_stock_price - inv_y)
# inv_predicted_stock_price -= bias
# predicted_open_price = inv_y
# print(inv_y)
# print(inv_predicted_stock_price)
dates_test = data.index[-len(y_test):]


actual_df = pd.DataFrame({'Open_Price': inv_y}, index=dates_test)
predicted_df = pd.DataFrame({'Open_Price': inv_predicted_stock_price}, index=dates_test)

# Plot the actual and predicted stock prices
plt.figure(figsize=(12, 6))

# Plot actual stock prices
plt.plot(actual_df.index, actual_df['Open_Price'], color='black', label='Actual Stock Price', linewidth=2)

# Plot predicted stock prices
plt.plot(predicted_df.index, predicted_df['Open_Price'], color='green', label='Predicted Stock Price', linestyle='--', linewidth=2)

# Title and labels
plt.title('Stock Price Prediction', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Stock Price (USD)', fontsize=12)

# Add a legend
plt.legend(loc='upper left')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Add gridlines for better readability
plt.grid(True)

# Save the plot as a .png file
plt.tight_layout()  # Adjust layout to avoid clipping
plt.savefig("stock_price_prediction_valid.png", dpi=300)  # Save with a higher resolution


# Additional evaluation metrics (optional)
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

mse = root_mean_squared_error(inv_y, inv_predicted_stock_price)
mae = mean_absolute_error(inv_y, inv_predicted_stock_price)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
# Calculate the Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((inv_y - inv_predicted_stock_price) / inv_y)) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

