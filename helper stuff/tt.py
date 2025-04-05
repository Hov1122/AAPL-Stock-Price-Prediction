import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Load Data
# data = pd.read_csv('random_aapl_stock_data.csv')
data = pd.read_csv('aapl_combined_dataset3.csv')
data['Date'] = pd.to_datetime(data['Date'])
# data = data[data['Date'] >= '2022-03-01']
data.set_index('Date', inplace=True)


# Step 2: Feature Selection
# Check correlation with target ('Close')
corr_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.savefig('correlation_matrix.png')  # Save correlation matrix plot

# Adjust correlation threshold
corr_target = corr_matrix['Open'].abs().sort_values(ascending=False)
selected_features = corr_target[corr_target > 0.1].index  # Lowered threshold to 0.1

# If no features pass the threshold, include a default set for testing
if len(selected_features) <= 1:  # Only 'Close' in selected
    selected_features = data.columns.difference(['Open'])  # Default to all other columns

print("Selected Features based on correlation:", selected_features)

# Step 3: Prepare Data for Model Training
X = data[selected_features]
y = data['Open']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Random Forest Regression Model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Step 5: Predict and Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Feature Importance Analysis
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print(feature_importance_df)

# Plotting feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
plt.title("Feature Importance in Predicting AAPL Stock 'Open' Price")
plt.show()
plt.savefig('feature_importance.png')  # Save feature importance plot
