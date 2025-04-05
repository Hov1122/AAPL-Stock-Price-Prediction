import yfinance as yf
import pandas as pd
import requests
from datetime import datetime
from alpha_vantage.timeseries import TimeSeries

# ------------- Configuration -------------
# Replace with your FRED and Alpha Vantage API keys
FRED_API_KEY = '9e56da542a0b80ec8be4689679b2578c'
ALPHA_VANTAGE_API_KEY = 'B3RXZEK5TDRGB5E1'

# List of companies' ticker symbols
# companies = ['AAPL', 'MSFT', 'KO', 'PG', 'JPM', 'GS', 'JNJ', 'PFE', 'XOM', 'NEE']
companies = ['AAPL']

# Set the time period for the historical data
start_date = '2010-01-01'
end_date = '2025-02-10'

# Economic Indicators
indicators = {
    'GDP': 'GDP',
    'Unemployment_Rate': 'UNRATE',
    'Inflation_Rate': 'CPIAUCSL',
    'Interest_Rate': 'FEDFUNDS'
}

# ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')

# # Fetch daily stock price data for AAPL
# data, meta_data = ts.get_daily(symbol='AAPL', outputsize='full')

# # Rename columns for better readability
# data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
# data = data.sort_values(by='date')
# # Save to CSV
# data.to_csv('AAPL_stock_data.csv')

# ------------- Stock Data Collection -------------
# def fetch_and_save_stock_data(ticker, start, end):
#     stock_data = yf.download(ticker, start=start, end=end)
#     csv_filename = f"{ticker}_historical_data.csv"
#     stock_data.to_csv(csv_filename)
#     print(f"Stock data for {ticker} saved to {csv_filename}")
#     return stock_data.index

# all_dates = None

# for company in companies:
#     dates = fetch_and_save_stock_data(company, start_date, end_date)
#     if all_dates is None:
#         all_dates = dates
#     else:
#         all_dates = all_dates.union(dates)

# all_dates = pd.DataFrame(all_dates, columns=['date'])
# all_dates.set_index('date', inplace=True)

# ------------- Economic Indicators Collection -------------
def fetch_fred_data(series_id, api_key, start, end):
    url = (f"https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={series_id}&api_key={api_key}&file_type=json"
           f"&observation_start={start}&observation_end={end}")
    response = requests.get(url)
    data = response.json()
    print(data)
    df = pd.DataFrame(data['observations'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df[['value']]
    df.columns = [series_id]
    df = df.reindex(all_dates.index).ffill()
    return df

# economic_data = pd.DataFrame(index=all_dates.index)

# for name, series_id in indicators.items():
#     indicator_data = fetch_fred_data(series_id, FRED_API_KEY, start_date, end_date)
#     economic_data = economic_data.join(indicator_data, how='outer')

# economic_csv_filename = "economic_indicators.csv"
# economic_data.to_csv(economic_csv_filename)
# print(f"Economic indicators data saved to {economic_csv_filename}")

# ------------- Earnings Reports and Financial Ratios Collection -------------
def fetch_income_statement(ticker, api_key, start, end):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    
    if 'quarterlyReports' in data:
        df = pd.DataFrame(data['quarterlyReports'])
        
        # Debugging: Print available columns
        print(f"Income Statement Columns for {ticker}: {df.columns.tolist()}")
        
        # Check if the required columns exist
        required_columns = ['grossProfit', 'totalRevenue', 'netIncome']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: Missing columns {missing_columns} in income statement data for {ticker}")
            return pd.DataFrame()
        
        df['date'] = pd.to_datetime(df['fiscalDateEnding'])
        df.set_index('date', inplace=True)
        df = df[(df.index >= start) & (df.index <= end)]
        df = df[['grossProfit', 'totalRevenue', 'netIncome']]
        df.columns = ['Gross_Profit', 'Revenue', 'Net_Income']
        return df
    else:
        print(f"No income statement data found for {ticker}")
        return pd.DataFrame()

# ------------- Earnings Reports Collection -------------
def fetch_earnings_reports(ticker, api_key, start, end):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    
    if 'quarterlyEarnings' in data:
        df = pd.DataFrame(data['quarterlyEarnings'])
        
        # Debugging: Print available columns
        print(f"Earnings Report Columns for {ticker}: {df.columns.tolist()}")
        
        # Check if the required columns exist
        required_columns = ['reportedEPS']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: Missing columns {missing_columns} in earnings report data for {ticker}")
            return pd.DataFrame()
        
        df['date'] = pd.to_datetime(df['fiscalDateEnding'])
        df.set_index('date', inplace=True)
        df = df[(df.index >= start) & (df.index <= end)]
        df = df[['reportedEPS']]
        df.columns = ['EPS']
        return df
    else:
        print(f"No earnings report data found for {ticker}")
        return pd.DataFrame()
    
def fetch_sentiment_data(company, api_key, start, end):
    date_start = datetime.strptime(start, '%Y-%m-%d')
    formatted_start = date_start.strftime('%Y%m%dT%H%M')
    date_end = datetime.strptime(end, '%Y-%m-%d')
    formatted_end = date_end.strftime('%Y%m%dT%H%M')

    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={company}&time_from={formatted_start}&time_to={formatted_end}&limit=1000&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    # print(data)

    if 'feed' in data:
        df = pd.DataFrame(data['feed'])
        
        # Debugging: Print available columns
        print(f"Sentiment Data Columns for {company}: {df.columns.tolist()}")
        
        # Check if the required columns exist
        required_columns = ['overall_sentiment_score', 'time_published']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: Missing columns {missing_columns} in sentiment data for {company}")
            return pd.DataFrame()
        
        # Convert 'time_published' to datetime and set as index
        df['time_published'] = pd.to_datetime(df['time_published'])
        df.set_index('time_published', inplace=True)
        
        # Select the 'overall_sentiment_score' column and rename it
        df = df[['overall_sentiment_score']]
        df.columns = ['Sentiment_Score']
        
        # Resample to daily average sentiment score
        # daily_sentiment = df.resample('D').mean()
        
        # Drop NaN rows if there are days with no sentiment data
        # daily_sentiment.dropna(inplace=True)
        
        # return daily_sentiment
        return df
    else:
        print(f"No sentiment data found for {company}")
        return pd.DataFrame()

def append_sentiment_data(company, api_key, start, end):
    # Fetch new sentiment data
    new_sentiment_data = fetch_sentiment_data(company, api_key, start, end)
    
    if not new_sentiment_data.empty:
        # Define the CSV file path
        file_path = f"{company}_sentiment_data_all.csv"
        # file_path = f"{company}_sentiment_2024.csv"

        try:
            # Read existing data if the file exists
            existing_sentiment_data = pd.read_csv(file_path, parse_dates=['time_published'], index_col='time_published')
            # Append the new data
            combined_data = pd.concat([existing_sentiment_data, new_sentiment_data])
        except FileNotFoundError:
            # If the file doesn't exist, use the new data directly
            combined_data = new_sentiment_data

        # Resample to daily average and drop NaN rows
        # daily_sentiment = combined_data.resample('D').mean().dropna()
        daily_sentiment = combined_data.dropna()

        # Save the combined data back to the CSV file
        daily_sentiment.to_csv(file_path)
        print(f"Data for {company} appended to {file_path}")


# for company in companies:
#     # income_data = fetch_income_statement(company, ALPHA_VANTAGE_API_KEY, start_date, end_date)
#     # earnings_data = fetch_earnings_reports(company, ALPHA_VANTAGE_API_KEY, start_date, end_date)
#     # sentiment_data = fetch_sentiment_data(company, ALPHA_VANTAGE_API_KEY, start_date, end_date)

#     # if not earnings_data.empty and not income_data.empty:
#     #     combined_data = income_data.join(earnings_data, how='outer')
#     #     combined_csv_filename = f"{company}_combined_income_eps.csv"
#     #     combined_data.to_csv(combined_csv_filename)
        
#     append_sentiment_data(company, ALPHA_VANTAGE_API_KEY, start_date, end_date)
    # if not sentiment_data.empty:
    #     sentiment_csv_filename = f"{company}_sentiment_data_all.csv"
    #     sentiment_data.to_csv(sentiment_csv_filename)
    #     print(f"Sentiment data for {company} saved to {sentiment_csv_filename}")


# Function to fetch economic data and filter by date range
def fetch_economic_data(function, key, indicator_name):
    url = f'https://www.alphavantage.co/query?function={function}&apikey={key}&datatype=json'
    response = requests.get(url)
    data = response.json()
    
    if 'data' in data:
        time_series = data['data']
        df = pd.DataFrame(time_series)
        df['indicator'] = indicator_name
        
        # Ensure date is in datetime format and filter by the specified date range
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        return df
    else:
        print(f"Error fetching data for {indicator_name}")
        return pd.DataFrame()  # Return an empty DataFrame if data fetch fails

# Fetch and filter data for each economic indicator
# gdp_data = fetch_economic_data('REAL_GDP', ALPHA_VANTAGE_API_KEY, 'GDP')
# unrate_data = fetch_economic_data('UNEMPLOYMENT', ALPHA_VANTAGE_API_KEY, 'UNRATE')
# cpi_data = fetch_economic_data('CPI', ALPHA_VANTAGE_API_KEY, 'CPIAUCSL')
# fedfunds_data = fetch_economic_data('FEDERAL_FUNDS_RATE', ALPHA_VANTAGE_API_KEY, 'FEDFUNDS')

# # Respect API rate limits
# # time.sleep(12)

# # Combine and format the data
# economic_data = pd.concat([gdp_data, unrate_data, cpi_data, fedfunds_data])
# economic_data = economic_data.pivot(index='date', columns='indicator', values='value').reset_index()
# economic_data.columns.name = None  # Clean column names

# # Save filtered data to a CSV file
# economic_data.to_csv('us_economic_indicators_filtered.csv', index=False)

# print("Data has been saved to us_economic_indicators_filtered.csv")

ticker = "^NDX"
start_date = "2010-01-01"
end_date = "2025-02-11"

# Fetch data
ndxt_data = yf.download(ticker, start=start_date, end=end_date)

# Display the first few rows
print(ndxt_data.head())

# Save to CSV if needed
# ndxt_data.to_csv("ndx_data.csv")