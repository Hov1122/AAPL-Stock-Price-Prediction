import yfinance as yf
import pandas as pd
import requests
from datetime import datetime
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta
import config

# FRED_API_KEY = '9e56da542a0b80ec8be4689679b2578c'
ALPHA_VANTAGE_API_KEY = config.ALPHA_API_KEY

# # List of companies' ticker symbols
# # companies = ['AAPL', 'MSFT', 'KO', 'PG', 'JPM', 'GS', 'JNJ', 'PFE', 'XOM', 'NEE']
# companies = ['AAPL']

# # Set the time period for the historical data
start_date = config.TRAINED_UNTIL
tommorow = datetime.now() + timedelta(days=1)
end_date = tommorow.strftime('%Y-%m-%d')
# ticker = "^NDX"

def get_data():
    url = f'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': config.SYMBOL,
        'apikey': ALPHA_VANTAGE_API_KEY,
        'outputsize': 'full',  # or 'compact' for a shorter range
        'datatype': 'json'
    }

    response = requests.get(url, params=params)
    data = response.json()

    # Extract time series data
    time_series = data['Time Series (Daily)']
    
    # Convert it into a DataFrame and reset the index
    df = pd.DataFrame.from_dict(time_series, orient='index').reset_index()
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']  # Rename columns

    # Convert 'Date' to datetime and columns to numeric
    df['Date'] = pd.to_datetime(df['Date'])
    df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric)

    # Filter by start and end date
    df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
    df['Ndx_Open'] = 19500 
    df = df.sort_values(by='Date', ascending=True)

    return df

    ## in case api fails use manual data for demo
    # data = {
    #     'Date': pd.to_datetime(['2025-05-20']),
    #     'Open': [230],
    #     'High': [231],
    #     'Low': [230],
    #     'Close': [230.5],
    #     'Volume': [2000000],
    #     'Ndx_Open': [19000]
    # }

    # # Create a DataFrame from the data
    # df = pd.DataFrame(data)
    
    # return df

def get_historical_stock_data(symbol="AAPL", days=30):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "Time Series (Daily)" in data:
        # Extract the last 30 days of data
        time_series = data["Time Series (Daily)"]
        dates = sorted(time_series.keys(), reverse=True)[:days]  # Get the last 'days' number of dates
        
        stock_data = []
        for date in dates:
            day_data = time_series[date]
            stock_data.append({
                "date": date,
                "open": float(day_data["1. open"]),
                "high": float(day_data["2. high"]),
                "low": float(day_data["3. low"]),
                "close": float(day_data["4. close"]),
                "volume": int(day_data["5. volume"]),
            })

        return stock_data
    else:
        return None