# import pandas as pd

# # Load the CSV file
# file_path = "AAPL_sentiment_data.csv"  # Replace with your file path
# df = pd.read_csv(file_path)

# # Check if the column 'Sentiment_Score' exists
# if 'Sentiment_Score' in df.columns:
#     # Round up the Sentiment_Score column to 4 decimal places
#     df['Sentiment_Score'] = df['Sentiment_Score'].round(4)

#     # Save the updated DataFrame to a new CSV file
#     output_file_path = "updated_file.csv"  # Replace with your desired output file path
#     df.to_csv(output_file_path, index=False)

#     print("Updated file saved successfully!")
#     print(df.head())
# else:
#     print("The column 'Sentiment_Score' does not exist in the file.")

# import pandas as pd

# # Load the dataset
# file_path = "AAPL_sentiment_data_all.csv"  # Replace with your file path
# df = pd.read_csv(file_path)

# # Ensure 'Date' is in datetime format
# df['Date'] = pd.to_datetime(df['time_published'], format='%Y-%m-%d %H:%M:%S')

# # Extract the date part (year-month-day) for grouping
# df['Date'] = df['Date'].dt.date

# # Group by 'Date' and calculate mean and variance of Sentiment_Score
# aggregated_sentiment = df.groupby('Date')['Sentiment_Score'].agg(
#     Mean_Sentiment='mean',
#     Variance_Sentiment='var'  # Variance of sentiment scores
# ).reset_index()

# # Calculate the composed value
# aggregated_sentiment['Composed_Value'] = (
#     aggregated_sentiment['Mean_Sentiment'] * (1 + aggregated_sentiment['Variance_Sentiment'])
# )

# # Save the new dataset to a CSV file
# output_file = "aggregated_sentiment_features_with_composed.csv"  # Replace with your desired file name
# aggregated_sentiment.to_csv(output_file, index=False)

# print(f"Aggregated sentiment features with composed value saved to {output_file}")


import pandas as pd

# Load your dataset
input_file = "AAPL_sentiment_data_all.csv"  # Replace with your actual file path
data = pd.read_csv(input_file)

# Convert 'time_published' to datetime format
data['time_published'] = pd.to_datetime(data['time_published'])

# Extract only the date portion
data['Date'] = data['time_published'].dt.date

# Filter out weekends (Saturday and Sunday)
data = data[pd.to_datetime(data['Date']).dt.dayofweek < 5]

# Group by date and compute required values
aggregated_data = data.groupby('Date').agg(
    mean_sentiment=('Sentiment_Score', 'mean'),
    p_sum=('Sentiment_Score', 'sum'),
    news_count=('Sentiment_Score', 'count')
).reset_index()

# Sort in descending order by date
aggregated_data = aggregated_data.sort_values(by='Date')

# Save to a new CSV
output_file = "aggregated_daily_no_weekends.csv"  # Replace with your desired output file name
aggregated_data.to_csv(output_file, index=False)

print(f"Aggregated daily dataset (excluding weekends) saved to {output_file}.")


