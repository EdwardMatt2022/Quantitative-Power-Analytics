import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


df = pd.read_csv(('/Users/edwardmattern/Documents/ng_model.csv'),parse_dates=['DATE'])
df.set_index('DATE',inplace=True)
df['MA7'] = df['NG1_PRICE'].rolling(window=7).mean()
df['MA30'] = df['NG1_PRICE'].rolling(window=30).mean()


df.dropna(inplace=True)




#Train-Test split for a simple model
X=df[['TAVG']]
y=df['ERCOT_DEMAND']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)

# Plotting
plt.figure(figsize=(14, 8))

# Scatter plot of Oil Price vs NG Price
plt.scatter(X, y, color='blue', label='Average Temperature', alpha=0.6)

# Plotting the regression line
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted ERCOT Demand')

# Add titles and labels
plt.title('Relationship Between Avg.Temperature and ERCOT Demand')
plt.xlabel('Temperature (F)')
plt.ylabel('ERCOT Demand (MW)')
plt.legend()
plt.grid()
plt.tight_layout()

# Show the plot
plt.show()








# Make predictions
y_pred = model.predict(X_test)









# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')





#Train-Test Split for the model

X=df[['NG1_PRICE_LAG','OIL_PRICE','COAL_PRICE','ERCOT_DEMAND','TAVG','MA7','MA30']]
y=df['NG1_PRICE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')






plt.figure(figsize=(12, 6))
plt.plot(df.index, y, label='Natural Gas Price (NG1_PRICE)', color='blue')
plt.plot(df.index, df['MA7'], label='7-Day Moving Average', color='orange')
plt.plot(df.index, df['MA30'], label='30-Day Moving Average', color='green')
plt.title('Natural Gas Prices with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD/MMBtu)')
plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(df['TAVG'], df['NG1_PRICE'], color='blue', alpha=0.7)

# Add labels and title
plt.title('Relationship Between Temperature and Natural Gas Prices')
plt.xlabel('Average Daily Temperature (°F)')
plt.ylabel('Natural Gas Spot Price (USD/MMBtu)')
plt.grid(True)

# Show the plot
plt.show()




#Trading Strategy Logic

df['Buy_Signal'] = np.where((df['NG1_PRICE'] > df['MA7']) & (df['NG1_PRICE'].shift(1) <= df['MA7'].shift(1)), 1, 0)
df['Sell_Signal'] = np.where((df['NG1_PRICE'] < df['MA7']) & (df['NG1_PRICE'].shift(1) >= df['MA7'].shift(1)), -1, 0)

plt.figure(figsize=(14, 8))

# Plot natural gas prices and moving average
plt.plot(df.index, df['NG1_PRICE'], label='Natural Gas Price (NG)', color='blue', alpha=0.6)
plt.plot(df.index, df['MA7'], label='7-Day Moving Average', color='orange', linestyle='--')

# Plot buy signals
plt.plot(df[df['Buy_Signal'] == 1].index, df['NG1_PRICE'][df['Buy_Signal'] == 1], '^', markersize=10, color='green', label='Buy Signal')

# Plot sell signals
plt.plot(df[df['Sell_Signal'] == -1].index, df['NG1_PRICE'][df['Sell_Signal'] == -1], 'v', markersize=10, color='red', label='Sell Signal')

# Add titles and labels
plt.title('Trading Strategy for Natural Gas with Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Price (USD/MMBtu)')
plt.legend()
plt.grid()
plt.tight_layout()

# Show the plot
plt.show()






