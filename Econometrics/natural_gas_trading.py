import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


df = pd.read_csv(('/Users/edwardmattern/Documents/ng_model.csv'),parse_dates=['DATE'])




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(ng1_price['Date'], ng1_price['PRICE'], label='Actual Prices')
plt.plot(ng1_price['Date'][len(df) - len(y_test):], y_pred, label='Predicted Prices', linestyle='--')
plt.xlabel('Date')
plt.ylabel('PRICE')
plt.title('Price Forecasting with Date Features')
plt.legend()
plt.show()




