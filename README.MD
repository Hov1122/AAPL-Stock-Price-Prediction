# Apple Stock Price Prediction

This project is designed to predict Apple's stock price using historical stock data and sentiment data. It supports live stock price prediction and allows you to compare the performance of different machine learning models on the given dataset.

## Features

- **Stock Price Prediction**: Predict Apple's stock price based on historical stock data and sentiment data.
- **Live Stock Price Prediction**: Predict the current Apple stock price using real-time data.
- **Model Comparison**: Compare the accuracy of various machine learning models (e.g., Linear Regression, Decision Tree, LSTM) on the given dataset.
- **Customizable Configuration**: Easily change configurations like port number and model parameters via a `config.py` file.

## Requirements

To set up and run this project, follow these steps:

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone <repository_url>
cd <project_directory>
```

### 2. Set Up the Virtual Environment

Create and activate a virtual environment to isolate the project dependencies:

```
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install the Dependencies

Install all the required Python packages from requirements.txt:

```
pip3 install -r requirements.txt
```

### 4. Configure the Application

Before running the application, you may want to configure some settings like the port number in the config.py file.

```
Example of modifying config.py:
PORT = 5000  # Change to your desired port number
```

### 5. Run the Application

Now, run the application:

```
python3 app.py
This will start the server, and you can access the application on the specified port.
```

### 6. Deactivate the Virtual Environment

Once you're done working, deactivate the virtual environment:

```
deactivate
```

### Data

The project requires two types of data:

Sentiment Data: Daily sentiment scores related to Apple stock (e.g., from social media or news).

Format: CSV with columns: Date, Sentiment_Score.

Stock Data: Historical Apple stock data including daily information on:

Open, High, Low, Close, Adj Close, Volume, Ndx_Open(Nasdaq-100)

Format: CSV with columns: Date, Open, High, Low, Close, Adj Close, Volume, Ndx_Open.

### How to Upload Data

Make sure the data is in the correct format and upload it to the application as instructed in the user interface. The data should be uploaded as CSV files for the application to process and predict stock prices.

### How to Use

Upload Your Dataset: Upload the sentiment data and/or stock data files to the application interface.

### Choose a Machine Learning Model: Select the machine learning model you want to use:

Linear regression

ARIMA

Lasso

Decision tree

Random forest

LSTM

GRU

FNN

CNN

### Run Prediction

After uploading the data and selecting a model, click the "Upload and Predict" button to generate stock price predictions.

View Results: The application will show the predicted stock price and the accuracy of the selected model.

### Model Comparison

This application allows you to compare the performance of different machine learning models on your data. The accuracy of each model is displayed, and you can easily assess which model performs best for your dataset.

### Live Stock Price Prediction

In addition to predicting stock prices based on historical data, this application can also predict the live stock price using current data pulled from APIs like Yahoo Finance or Alpha Vantage.

To enable live stock price prediction, configure your API keys in config.py and make sure your app has access to the internet.

Contributing
If you would like to contribute to this project, feel free to fork the repository, make changes, and create a pull request. Please ensure that your code follows the style guide and includes tests where applicable.

Thank you for checking out this project! If you have any questions or run into issues, please feel free to open an issue or contact me directly.
