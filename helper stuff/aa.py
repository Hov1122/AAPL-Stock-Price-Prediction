# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler

# # Step 1: Load the new dataset
# new_data = pd.read_csv("aapl_combined_dataset.csv", parse_dates=['Date'], index_col='Date')

# # Step 2: Preprocess the data (assuming the data is similar to the training set)
# new_data.fillna(method='ffill', inplace=True)  # Handling missing values

# # Scaling the data using the same scaler that was used for training
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(new_data)

# # Step 3: Prepare the data for prediction
# def create_sequences(dataset, look_back=60):
#     X = []
#     for i in range(len(dataset) - look_back):
#         X.append(dataset[i:(i + look_back), :])  # Create sequences
#     return np.array(X)

# look_back = 60
# X_new = create_sequences(scaled_data, look_back)

# # Step 4: Load the pre-trained model
# model = load_model("stock_prediction_model.h5")

# # Step 5: Make predictions
# predicted_stock_price = model.predict(X_new)

# # Step 6: Inverse transform the predictions (to get the original scale)
# # Flatten X_new to match predicted_stock_price's dimensions (2D)
# X_new_flat = X_new[:, :, 0]  # Take only the first feature (Open) for scaling

# # Concatenate the predicted stock price with the relevant part of X_new
# inv_predicted_stock_price = np.concatenate((predicted_stock_price, X_new_flat), axis=1)

# # Inverse transform only on the relevant features (those that the scaler was trained on)
# # Use only the first 11 features for inverse transform
# inv_predicted_stock_price = scaler.inverse_transform(inv_predicted_stock_price[:, :11])  # Only the first 11 features

# inv_predicted_stock_price = inv_predicted_stock_price[:, 0]  # Extract only the predicted 'Open' values

# # Step 7: Inverse scaling for actual stock prices (y_test from the previous data)
# y_test = new_data['Open'].values[-len(predicted_stock_price):]  # Ensure y_test corresponds to predicted data

# # Since 'y_test' is a single target column ('Open'), we don't need to concatenate it with features like X_new
# y_test_reshaped = y_test.reshape((len(y_test), 1))  # Ensure y_test has the correct shape

# # Use scaler to inverse transform the 'Open' values only
# y_test_scaled = scaler.transform(y_test_reshaped)  # Scale the actual values using the same scaler
# inv_y = scaler.inverse_transform(y_test_scaled)  # Inverse transform the actual 'Open' values

# # Step 8: Get the corresponding dates for the predictions
# dates_test = new_data.index[-len(y_test):]  # Dates should align with the length of y_test

# # Prepare the results for saving or further analysis
# predictions_df = pd.DataFrame({
#     'Date': dates_test,
#     'Actual_Open_Price': inv_y.flatten(),
#     'Predicted_Open_Price': inv_predicted_stock_price
# })

# # Set Date as index
# predictions_df.set_index('Date', inplace=True)

# # Step 9: Save the predictions to a CSV file
# predictions_df.to_csv("predicted_stock_prices.csv")

# print("Predictions saved to 'predicted_stock_prices.csv'")inverse_transform(inv_y[:, :11])  # Use only the first 11 features
# inv_y = inv_y[:, 0]  # Extract the actual 'Open' values

# # Step 8: Get the corresponding dates for the predictions
# dates_test = new_data.index[-len(y_test):]  # Dates should align with the length of y_test

# # Prepare the results for saving or further analysis
# predictions_df = pd.DataFrame({
#     'Date': dates_test,
#     'Actual_Open_Price': inv_y,
#     'Predicted_Open_Price': inv_predicted_stock_price
# })

# # Set Date as index
# predictions_df.set_index('Date', inplace=True)

# # Step 9: Save the predictions to a CSV file
# predictions_df.to_csv("predicted_stock_prices.csv")

# print("Predictions saved to 'predicted_stock_prices.csv'")


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential, load_model
# from keras.layers import LSTM, Dense, Dropout
# import os

# # Function to load data
# def load_and_prepare_data(file_path):
#     data = pd.read_csv(file_path, parse_dates=['Date'])
#     data.set_index('Date', inplace=True)
#     data.fillna(method='ffill', inplace=True)
#     return data

# # Function to create sequences
# def create_sequences(dataset, look_back=60):
#     X, y = [], []
#     for i in range(len(dataset) - look_back - 1):
#         X.append(dataset[i:(i + look_back), :])
#         y.append(dataset[i + look_back, 0])  # Predict the 'Open' price
#     return np.array(X), np.array(y)

# # Function to plot results
# def plot_predictions(actual, predicted, dates):
#     plt.figure(figsize=(12, 6))
#     plt.plot(dates, actual, color='black', label='Actual Stock Price', linewidth=2)
#     plt.plot(dates, predicted, color='green', label='Predicted Stock Price', linestyle='--', linewidth=2)
#     plt.title('Stock Price Prediction', fontsize=16)
#     plt.xlabel('Date', fontsize=12)
#     plt.ylabel('Stock Price (USD)', fontsize=12)
#     plt.legend(loc='upper left')
#     plt.xticks(rotation=45)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("stock_price_prediction.png", dpi=300)
#     plt.show()

# # Main function for prediction
# def predict_stock_price(file_path, model_file='stock_prediction_model.h5'):
#     data = load_and_prepare_data(file_path)

#     scaler = MinMaxScaler(feature_range=(-1, 1))
#     scaled_data = scaler.fit_transform(data)

#     X, y = create_sequences(scaled_data)
#     train_size = int(len(X) * 0.8)
#     X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

#     if os.path.exists(model_file):
#         model = load_model(model_file)
#         print("Loaded existing model.")
#     else:
#         model = Sequential([
#             LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
#             Dropout(0.2),
#             LSTM(50),
#             Dropout(0.2),
#             Dense(1)
#         ])
#         model.compile(optimizer='adam', loss='mean_squared_error')
#         model.fit(X_train, y_train, epochs=10, batch_size=32)
#         model.save(model_file)
#         print("Model trained and saved.")

#     predicted_prices = model.predict(X_test)
#     y_test_unscaled = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_data.shape[1] - 1))), axis=1))[:, 0]
#     # Reshape X_test to 2D for concatenation
#     X_test_flattened = X_test.reshape(X_test.shape[0], -1)  # Flatten to 2D

#     # Concatenate predicted prices with the flattened X_test and inverse scale
# # Pad predicted_prices with zeros to match scaler's expected shape
#     padded_predictions = np.concatenate(
#         (predicted_prices, np.zeros((predicted_prices.shape[0], 15))), axis=1
#     )

#     # Inverse transform and extract the first column (Open price)
#     predicted_prices_unscaled = scaler.inverse_transform(padded_predictions)[:, 0]
#     dates_test = data.index[-len(y_test):]
#     plot_predictions(y_test_unscaled, predicted_prices_unscaled, dates_test)

# # Run the prediction function
# predict_stock_price('aapl_combined_dataset2.csv')




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Load the trained model
model = load_model('stock_prediction_model.h5')

# Load historical data and known features for 2024
historical_data = pd.read_csv("aapl_combined_dataset2.csv", parse_dates=['Date'])
historical_data.set_index('Date', inplace=True)
historical_data.index.freq = 'B'
historical_data.fillna(method='ffill', inplace=True)

# Load known features (sentiment, economic data) for 2024
future_features = pd.read_csv("AAPL_sentiment_2024.csv", parse_dates=['Date'])
future_features.set_index('Date', inplace=True)

# Combine historical and future data
data = pd.concat([historical_data, future_features], axis=0)

# Feature scaling
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(data)

# Define look-back period (same as in training)
look_back = 60

# Get the last known sequence to start forecasting
last_sequence = scaled_data[-(look_back + len(future_features)): -len(future_features)]

# Initialize a list to store predictions
future_predictions = []

# Loop through each day in the future features
for i in range(len(future_features)):
    # Create input for the model (last look_back days)
    X_input = last_sequence.reshape((1, look_back, scaled_data.shape[1]))

    # Predict the next day's 'Open' price
    predicted_price = model.predict(X_input)[0][0]

    # Store the prediction
    future_predictions.append(predicted_price)

    # Update the sequence with known features and predicted price
    next_features = scaled_data[look_back + i]
    new_sequence = np.concatenate(([predicted_price], next_features[1:]))  # Replace 'Open' with predicted value
    last_sequence = np.vstack((last_sequence[1:], new_sequence))

# Inverse transform predictions
padded_predictions = np.concatenate((np.array(future_predictions).reshape(-1, 1), future_features.iloc[:, 1:].values), axis=1)
predicted_prices_unscaled = scaler.inverse_transform(padded_predictions)[:, 0]

# Create a DataFrame for predictions
future_dates = future_features.index
predicted_df = pd.DataFrame({'Predicted_Open_Price': predicted_prices_unscaled}, index=future_dates)

# Plot predictions
plt.figure(figsize=(12, 6))
plt.plot(predicted_df.index, predicted_df['Predicted_Open_Price'], color='blue', label='Predicted Stock Price (2024)')
plt.title('Stock Price Prediction for 2024 with Sentiment & Economic Data', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Open Price (USD)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("predicted_2024_with_features.png", dpi=300)
plt.show()


