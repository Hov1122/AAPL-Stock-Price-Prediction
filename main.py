import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso
from statsmodels.tsa.arima.model import ARIMA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Conv1D, MaxPooling1D, Flatten
from datetime import datetime, timedelta
from fetch_data import get_data
import config


class StockPricePredictor:
    NN_MODELS = ['FNN', 'CNN', 'GRU', 'LSTM']
    REG_MODELS = ['ARIMA', 'Linear regression', 'Random forest', 'Decision tree', 'Lasso']

    def __init__(self, stock_file, sentiment_file=None):
        """
        Initialize StockPricePredictor with stock price data and sentiment data.
        """
        # Load stock data
        self.stock_data = pd.read_csv(stock_file, parse_dates=['Date'])
        self.use_sentiment = sentiment_file is not None
    
        # Ensure 'Date' column is in datetime format
        self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'], errors='coerce')

        # Should really not do this but to test with and without sentiment we need to have the same dataset.
        self.stock_data = self.stock_data[self.stock_data['Date'] >= '2022-03-01']
        self.loaded_gru = 0
        self.loaded_lstm = 0
        self.loaded_fnn = 0
        self.loaded_cnn = 0
        
        # Load sentiment data only if available
        if self.use_sentiment:
            self.sentiment_data = pd.read_csv(sentiment_file, parse_dates=['Date'])

            # Ensure that the 'Date' column in sentiment data is in datetime format
            self.sentiment_data['Date'] = pd.to_datetime(self.sentiment_data['Date'], errors='coerce')

            first_available_date = self.sentiment_data['Date'].iloc[0]
            self.stock_data = self.stock_data[self.stock_data['Date'] >= first_available_date]

            # Merge sentiment data with stock data (based on Date)
            self.data = pd.merge(self.stock_data, self.sentiment_data, on='Date', how='left')

            # Fill missing sentiment scores (if any) with 0 or other preferred value
            self.data['Sentiment_Score'].fillna(0, inplace=True)
        else:
            self.data = self.stock_data

        # Normalize features (both with and without sentiment)
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Ndx_Open']
        self.target_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Ndx_Open']
        self.target = 'Open'
        if self.use_sentiment:
            self.features.append('Sentiment_Score')
            self.target_columns.append('Sentiment_Score')
        
        self.output_units = len(self.target_columns)
        self.preprocess_data()

        print("✅ Data Loaded and Merged with Sentiment Scores" if self.use_sentiment else "✅ Data Loaded without Sentiment Scores")

    def preprocess_data(self):
        """Prepares data for training by scaling features."""
       
        # Normalize features (both with and without sentiment)
        self.scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler() 
        self.X_scaled = self.scaler.fit_transform(self.data[self.features])
        self.y_scaled = self.data[self.target]

    def split_data(self, model_type, test_size=0.2, random_state=42):
        """Splits data while keeping test indices in sync with original dataset."""
        if model_type in StockPricePredictor.NN_MODELS:
            # self.y_scaled = self.y_scaler.fit_transform(self.data[['Open']])
            self.y_scaled = self.y_scaler.fit_transform(self.data[self.target_columns])

        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y_scaled, test_size=test_size, random_state=random_state, shuffle=False
        )
        X_test_indices = self.data['Date'].iloc[-len(X_test):]  # Get dates for the test set

        return X_train, X_test, y_train, y_test, X_test_indices

    def load_linear_regression(self):
        return LinearRegression()
    
    def load_lasso(self):
        return Lasso(alpha=0.2)
    
    def load_arima(self, p, i, q):
        self.open_series = self.data['Open'].copy()
        dates = pd.to_datetime(self.data['Date'])
        self.open_series.index = dates
        return ARIMA(self.open_series, order=(p, i, q))

    def load_decision_tree(self):
        return DecisionTreeRegressor()

    def load_random_forest(self):
        return RandomForestRegressor(n_estimators=200, random_state=42)

    def load_LSTM(self):
        """Load or train an LSTM model."""
        model_suffix = "_with_sentiment" if self.use_sentiment else "_without_sentiment"
        model_file = f"stock_prediction_model_lstm{model_suffix}.keras"
        self.loaded_lstm = 0
    
        if os.path.exists(model_file):
            print(f"{model_file}")
            model = load_model(model_file, compile=True)  # Load without recompiling
            print(f"✅ Loaded existing LSTM model: {model_file}")
            self.loaded_lstm = 1
            return model

        # If the model doesn't exist, create a new one
        model = Sequential([
            LSTM(units=100, return_sequences=True, input_shape=(1, self.X_scaled.shape[1])),
            Dropout(0.2),
            LSTM(units=100),
            Dropout(0.2),
            Dense(units=self.output_units)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        print(f"🚀 New LSTM model created")
        return model

    def load_GRU(self):
        """Load or train a GRU model."""
        model_suffix = "_with_sentiment" if self.use_sentiment else "_without_sentiment"
        model_file = f"stock_prediction_model_gru{model_suffix}.keras"
        self.loaded_gru = 0

        if os.path.exists(model_file):
            model = load_model(model_file, compile=True)  # Load without recompiling
            print(f"✅ Loaded existing GRU model: {model_file}")
            self.loaded_gru = 1
            return model

        # If the model doesn't exist, create a new one
        model = Sequential([
            GRU(units=100, return_sequences=True, input_shape=(1, self.X_scaled.shape[1])),
            Dropout(0.2),
            GRU(units=60),
            Dropout(0.4),
            Dense(units=self.output_units)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        print(f"🚀 New GRU model created")
        return model
    
    def load_FNN(self):
        """Load or train a Feedforward Neural Network (FNN) model."""
        model_suffix = "_with_sentiment" if self.use_sentiment else "_without_sentiment"
        model_file = f"stock_prediction_model_fnn{model_suffix}.keras"
        self.loaded_fnn = 0

        if os.path.exists(model_file):
            model = load_model(model_file, compile=True)  # Load without recompiling
            print(f"✅ Loaded existing FNN model: {model_file}")
            self.loaded_fnn = 1
            return model

        # If the model doesn't exist, create a new one
        model = Sequential([
            Dense(units=150, activation='relu', input_dim=self.X_scaled.shape[1]),
            Dropout(0.2),
            Dense(units=160, activation='relu'),
            Dropout(0.3),
            Dense(units=self.output_units)  # Output layer
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        print(f"🚀 New FNN model created")
        return model
    
    def load_CNN(self):
        """Load or train a Convolutional Neural Network (CNN) model."""
        model_suffix = "_with_sentiment" if self.use_sentiment else "_without_sentiment"
        model_file = f"stock_prediction_model_cnn{model_suffix}.keras"
        self.loaded_cnn = 0

        if os.path.exists(model_file):
            model = load_model(model_file, compile=True)  # Load without recompiling
            print(f"✅ Loaded existing CNN model: {model_file}")
            self.loaded_cnn = 1
            return model

        # If the model doesn't exist, create a new one
        model = Sequential([
            Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(self.X_scaled.shape[1], 1)),
            MaxPooling1D(pool_size=1),  # Use pool_size=1 here
            Conv1D(filters=32, kernel_size=2, activation='relu'),
            MaxPooling1D(pool_size=1),  # Use pool_size=1 here
            Flatten(),
            Dense(units=64, activation='relu'),
            Dropout(0.2),
            Dense(units=self.output_units)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        print(f"🚀 New CNN model created")
        return model

    def train_model(self, model_type='linear_regression', p=0, i=1, q=0):
        """Trains the selected model."""
        X_train, X_test, y_train, y_test, X_test_indices = self.split_data(model_type)
        self.model_type = model_type

        if model_type == 'Linear regression':
            self.model = self.load_linear_regression()
        elif model_type == 'Lasso':
            self.model = self.load_lasso()
        elif model_type == 'ARIMA':
            self.model = self.load_arima(p,i,q).fit()
            return
        elif model_type == 'Decision tree':
            self.model = self.load_decision_tree()
        elif model_type == 'Random forest':
            self.model = self.load_random_forest()
        elif model_type == "LSTM":
            self.model = self.load_LSTM()
            X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
            X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        elif model_type == "GRU":
            self.model = self.load_GRU()
            X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
            X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        elif model_type == "FNN":
            self.model = self.load_FNN()
        elif model_type == "CNN":
            self.model = self.load_CNN()
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # CNN expects 3D input
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # CNN expects 3D input
        else:
            raise ValueError("❌ Invalid model type. Choose 'Linear regression', 'Lasso', 'Decision tree', 'Random forest', 'LSTM',"
            "'FNN', 'CNN', 'ARIMA' or 'GRU'.")

        if model_type in StockPricePredictor.REG_MODELS:
            self.model.fit(X_train, y_train)  # Convert to 1D
        elif not self.loaded_gru and not self.loaded_lstm and not self.loaded_fnn and not self.loaded_cnn:
            self.model.fit(X_train, y_train, epochs=50, batch_size=32)
            model_suffix = "_with_sentiment" if self.use_sentiment else "_without_sentiment"
            self.model.save(f"stock_prediction_model_{model_type.lower()}{model_suffix}.keras")
            print(f"✅ Model saved as stock_prediction_model_{model_type.lower()}{model_suffix}.keras")

    def predict(self, all_targets=False):
        """Make predictions using the trained model."""
        # Get the data split for testing

        if self.model_type == "ARIMA":
            y_pred = self.model.predict(start=self.open_series.index[0], end=self.open_series.index[-1])
            return  self.open_series[1:], y_pred[1:], self.open_series.index[1:]

        X_train, X_test, y_train, y_test, X_test_indices = self.split_data(self.model_type)
        if self.model_type == "LSTM" or self.model_type == "GRU":
            X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
            X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        
        y_pred = self.model.predict(X_test)

        # get actual y_test values without inversing to avoid rounding
        y_test = self.data.set_index('Date').loc[X_test_indices, 'Open'].values
        
        if self.model_type in StockPricePredictor.NN_MODELS:
        #     y_pred = self.y_scaler.inverse_transform(y_pred.reshape(-1, 1))
        # else:
            y_pred = self.y_scaler.inverse_transform(y_pred)
        if not all_targets and self.model_type not in self.REG_MODELS:
            y_pred = y_pred[:, 0]

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"📊 Model: {self.model_type}")
        print(f"Mean Squared Error: {mse}")
        print(f"R-squared: {r2}")

        # Return actual and predicted values for comparison
        
        return y_test.flatten(), y_pred.flatten(), X_test_indices.values
    
    def plot_actual_vs_predicted(self, y_test, y_pred, X_test_indices, save_name = None):
        """Plot actual vs predicted stock prices with Date on x-axis."""
        test_dates = X_test_indices  # Use directly from split_data()

        y_test = np.array(y_test).flatten()
        y_pred = np.array(y_pred).flatten()

        plt.figure(figsize=(10, 6))
        plt.plot(test_dates, y_test, label='Actual Open Price', color='blue', marker='o')
        plt.plot(test_dates, y_pred, label='Predicted Open Price', linestyle='dashed', color='red', marker='x')

        plt.title(f'Actual vs Predicted Open Prices {self.model_type}')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if save_name:
            plt.savefig(save_name)
        else:
            plt.show()

    @staticmethod
    def get_errors(y_test, y_pred):

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {"mse": mse, "r2": r2}

    @staticmethod
    def compare_models(m1, m2, save_name = None):
        y_test1, y_pred1, test_indices1 = m1.predict()
        y_test2, y_pred2, test_indices2 = m2.predict()
        mse_1, r2_1 = list(StockPricePredictor.get_errors(y_test1, y_pred1).values())
        mse_2, r2_2 = list(StockPricePredictor.get_errors(y_test2, y_pred2).values())
        
        """Plot actual vs predicted stock prices for both models."""
        plt.figure(figsize=(12, 8))

        # Model without sentiment
        plt.subplot(2, 1, 1)
        plt.plot(test_indices1, y_test1, label='Actual (No Sentiment)', color='blue', marker='o')
        plt.plot(test_indices1, y_pred1, label='Predicted (No Sentiment)', linestyle='dashed', color='red', marker='x')
        plt.title(f'Model {m1.model_type} {"with " if m1.use_sentiment  else "without "} sentiment: Actual vs Predicted')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.text(test_indices1[-1], (max(y_pred1) + min(y_pred1)) / 2, f"mse:{mse_1}\nr2:{r2_1}", fontsize=12, verticalalignment='center',
                bbox = dict(facecolor = 'red', alpha = 0.5))
        plt.legend()
        
        # Model with sentiment
        plt.subplot(2, 1, 2)
        plt.plot(test_indices2, y_test2, label='Actual (With Sentiment)', color='blue', marker='o')
        plt.plot(test_indices2, y_pred2, label='Predicted (With Sentiment)', linestyle='dashed', color='red', marker='x')
        plt.title(f'Model {m2.model_type} {"with " if m2.use_sentiment  else "without "} sentiment: Actual vs Predicted')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.text(test_indices1[-1], (max(y_pred1) + min(y_pred1)) / 2, f"mse:{mse_2}\nr2:{r2_2}", fontsize=12, verticalalignment='center',
            bbox = dict(facecolor = 'red', alpha = 0.5))
        plt.legend()

        plt.tight_layout()
        if save_name:
            plt.savefig(save_name)
        else:
            plt.show()

    @staticmethod
    def plot_live_prediction(df, save_name=None):
        # Convert 'Date' to datetime if it is not already
        start_date = pd.to_datetime(config.TRAINED_UNTIL)
        today = datetime.now()
        end_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')
        df['Date'] = pd.to_datetime(df['Date'])

        # Split the data into two based on the given date range
        # before_range = df[df['Date'] < start_date]
        before_range = df[(df['Date'] >= start_date) & (df['Date'] < today.strftime('%Y-%m-%d')) ]
        current_day = df[(df['Date'] == today.strftime('%Y-%m-%d'))]
        after_range = df[df['Date'] >= end_date]


        # Plotting the graph
        plt.figure(figsize=(10, 6))

        # Plot before the range
        plt.plot(before_range['Date'], before_range['Open'], marker='o', color='b', linestyle='-', label='Available data')

        # Plot within the range
        plt.plot(current_day['Date'], current_day['Open'], marker='x', color='g', linestyle='-', label='Today')

        # Plot after the range
        plt.plot(after_range['Date'], after_range['Open'], marker='o', color='r', linestyle='-', label='Predicted stock prices')

        # Adding labels and title
        plt.xlabel('Date')
        plt.ylabel('Open Price')
        plt.title('Stock Open Price over Time')

        # Show grid
        plt.grid(True)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Display the legend
        plt.legend()

        # Display the graph
        plt.tight_layout()
        if save_name:
            plt.savefig(save_name)
        else:
            plt.show()

    @staticmethod
    def save_to_csv(model_name, sent, x_indices, y_test, y_pred):
        df = pd.DataFrame({"Date": x_indices, "price_test": y_test, "price_predicted": y_pred})
        df.to_csv(f"{model_name}_{'with' if sent else 'without'}_sentiment.csv", index=False)
    
    @staticmethod
    def get_live_predictor():
        stock_file = 'AAPL_stock_data.csv'
        predictor_no_sentiment = StockPricePredictor(stock_file)
        predictor_no_sentiment.train_model('LSTM')
        return predictor_no_sentiment

    @staticmethod
    def predict_future(predictor=None, live=False, steps=5):
        """
        Predicts future values using the last available data.

        Parameters:
        - predictor: predictorect containing the trained model, data, scalers, and necessary columns.
        - steps: Number of future time steps to predict.

        Returns:
        - Updated DataFrame with predicted values.
        """
        if predictor == "":
            predictor = StockPricePredictor.get_live_predictor()
            
        model = predictor.model
        feature_scaler, target_scaler = predictor.scaler, predictor.y_scaler
        feature_columns = predictor.features
        target_indices = predictor.target_columns
        if live:
            future_data = get_data()
        else:
            future_data = predictor.data

        last_row = future_data.iloc[-1:].copy()  # Start with the last available data
        
        for _ in range(steps):
            input_scaled = feature_scaler.transform(last_row[feature_columns].values)
            input_scaled = input_scaled.reshape((1, 1, -1))  # Reshape for LSTM

            input_scaled = model.predict(input_scaled, True).flatten()
            # print(predicted_scaled)
            input_scaled = input_scaled.reshape(1, -1)  # Reshape back to 2D
            

            # Inverse transform to original scale
            predicted_values = target_scaler.inverse_transform(input_scaled).flatten()
            new_row = [last_row.iloc[-1, 0] + timedelta(1)] + predicted_values.tolist()
            # print(new_row)
            new_row = pd.DataFrame([new_row], columns=['Date'] + feature_columns)

            future_data = pd.concat([future_data, new_row], ignore_index=True)
            last_row = new_row  # Use latest predictions for the next step

        # print(future_data)
        return future_data

# from sklearn.model_selection import cross_val_score, TimeSeriesSplit

# def cross_validate_model_sklearn(X, y, model, n_splits=5):
#     # Initialize the model
#     # if model_type == 'Linear Regression':
#     #     model = LinearRegression()
#     # elif model_type == 'Decision Tree':
#     #     model = DecisionTreeRegressor()
#     # elif model_type == 'Random Forest':
#     #     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     # else:
#     #     raise ValueError(f"Unknown model type: {model_type}")

#     # Use TimeSeriesSplit for time series data
#     tscv = TimeSeriesSplit(n_splits=n_splits)

#     # Perform cross-validation
#     scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')

#     # Output cross-validation results
#     # print(f"Cross-validation Mean Squared Error (MSE) scores for {model_type}: {scores}")
#     print(f"Mean MSE: {-scores.mean()}")
#     print(f"Standard Deviation of MSE: {scores.std()}")

#     return scores.mean(), scores.std()

# from scikeras.wrappers import KerasRegressor
# Usage Example
def main():
    # Load stock and sentiment data
    stock_file = 'AAPL_stock_data.csv'  # Replace with actual stock data file path
    sentiment_file = 'AAPL_sentiment_data.csv'  # Replace with actual sentiment data file path

    # Create predictor objects
    predictor_no_sentiment = StockPricePredictor(stock_file)
    predictor_with_sentiment = StockPricePredictor(stock_file, sentiment_file)

    # Train both models (one without sentiment and one with sentiment)
    # predictor_no_sentiment.train_model('Random forest')
    # predictor_with_sentiment.train_model('LSTM')
    predictor_no_sentiment.train_model('Lasso')
    # predictor_with_sentiment.train_model('CNN')

    # keras_regressor = KerasRegressor(build_fn=predictor_no_sentiment, epochs=50, batch_size=32, verbose=0)
    y_test_no_sentiment, y_pred_no_sentiment, test_indices1 = predictor_no_sentiment.predict()
    predictor_no_sentiment.plot_actual_vs_predicted(y_test_no_sentiment, y_pred_no_sentiment, test_indices1)
    # cv_mean, cv_std = cross_validate_model_sklearn(predictor_no_sentiment.X_scaled, predictor_no_sentiment.y_scaled, keras_regressor)
    # predictor_no_sentiment.train_model('CNN')
    # predictor_with_sentiment.train_model('CNN')

    # # Make predictions for both models
    # y_test_no_sentiment, y_pred_no_sentiment, test_indices1 = predictor_no_sentiment.predict()
    # y_test_with_sentiment, y_pred_with_sentiment, test_indices2 = predictor_with_sentiment.predict()
    # predictor_with_sentiment.plot_actual_vs_predicted(y_test_with_sentiment, y_pred_with_sentiment, test_indices2)
    # save_to_csv(predictor_no_sentiment.model_type, predictor_no_sentiment.use_sentiment, test_indices1
    #             , y_test_no_sentiment, y_pred_no_sentiment)
    # save_to_csv(predictor_with_sentiment.model_type, predictor_with_sentiment.use_sentiment, test_indices2
    #             , y_test_with_sentiment, y_pred_with_sentiment)

    # StockPricePredictor.compare_models(predictor_no_sentiment, predictor_with_sentiment)

    # predictor_no_sentiment.train_model('LSTM')
    # predictor_no_sentiment.predict_next_10_days()

        # Example usage
   # Example usage
    # feature_cols = ["High", "Low", "Close", "Volume", "Ndx_Open"]  # Original feature set (excluding Open)
    # target_idx = 0  # Assuming Open is the first column

    # predicted_df =  StockPricePredictor.predict_future(predictor_no_sentiment, live=True, steps=10)
    # StockPricePredictor.plot_live_prediction(predicted_df)
    # print(predicted_df.tail(20))
    # predicted_future = predict_open_price(predictor_no_sentiment.model, future_df, predictor_no_sentiment.scaler, predictor_no_sentiment.y_scaler, feature_cols, target_idx)

if __name__ == "__main__":
    main()
