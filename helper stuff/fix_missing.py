import pandas as pd

def fill_missing_weekdays(csv_file):
    # Load the CSV file and parse dates
    df = pd.read_csv(csv_file, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)

    # Create a date range including only weekdays (Monday to Friday)
    full_weekday_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')  # 'B' for business days

    # Reindex DataFrame to include all weekdays
    df = df.reindex(full_weekday_range)

    # Use forward fill to fill missing values with the previous day's values
    df.fillna(method='ffill', inplace=True)

    # Reset index and rename for output
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Date'}, inplace=True)
    
    # Optional: Save to a new CSV
    output_file = 'filled_' + csv_file
    df.to_csv(output_file, index=False)
    print(f"Processed file saved as {output_file}")

# Example usage
fill_missing_weekdays('AAPL_stock_data.csv')
