#!/usr/bin/env python
# coding: utf-8

# In[81]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[3]:


df = pd.read_csv('/Users/edwardmattern/Documents/Trading Economics/us_model_beta.csv')


# In[4]:


df['Date'] = pd.to_datetime(df['Date'])

df.set_index('Date', inplace=True)


# In[5]:


df['ERCOT'] = df['ERCOT']/1000


# In[6]:


df


# In[7]:


df.describe()


# In[8]:


df_season = df.copy(deep=True)
df_season['Month'] = df_season.index.month
mask = (df_season['Month'] >= 5) & (df_season['Month'] <= 10)
df_season['Winter'] = np.where(mask,1,0)
df_season['Summer'] = np.where(df_season['Winter'] != 1,1,0)
df_season


# In[21]:


df['TMAX'].plot(figsize=(8, 6), label='TMAX')
df['TMIN'].plot(figsize=(8, 6), label='TMIN')
df['TAVG'].plot(figsize=(8, 6), label='TAVG')
plt.xlabel('Date')
plt.ylabel('Temperature (F)')
plt.title('Texas Temperature')
plt.legend()


plt.show()


# In[40]:


plt.figure(figsize=(8,6))
df.TMAX.hist(bins=60, alpha=0.6, label='TMAX')
df.TMIN.hist(bins=60, alpha=0.6, label='TMIN')
df['TAVG'].hist(bins=60, alpha=0.8, label='TAVG')
plt.legend()
plt.show()


# In[47]:


plt.figure(figsize=(8,6))
df_season[df_season['Summer'] == 1]['TAVG'].hist(bins=60, alpha=0.8, label='Summer')
df_season[df_season['Winter'] == 1]['TAVG'].hist(bins=60, alpha=0.8, label='Winter')
plt.legend()
plt.show()


# In[31]:


df['NG1:COM'].plot(figsize=(7, 7), label='U.S. Natural Gas')
df['NGEU:COM'].plot(figsize=(7, 7), label='E.U. Natural Gas')
df['CL1:COM'].plot(figsize=(7, 7), label='Oil')
df['UXA:COM'].plot(figsize=(7, 7), label='Uranium')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.title('Commodities')
plt.legend()


plt.show()


# In[65]:


plt.figure(figsize=(8,6))
df['NG1:COM'].hist(bins=60, alpha=0.6, label='U.S. Natural Gas', color = 'red')
plt.legend()
plt.show()


# In[68]:


df['NGEU:COM'].hist(bins=60, alpha=0.8, label='E.U. Natural Gas', color = 'blue')
plt.legend()
plt.show()


# In[58]:


df['ERCOT'].plot(figsize=(8, 6), label='ERCOT')
plt.xlabel('Date')
plt.ylabel('Load (GWh)')
plt.title('ERCOT Generation')
plt.legend()


plt.show()


# In[61]:


summer_data = df_season[df_season['Summer'] == 1]['ERCOT']
winter_data = df_season[df_season['Winter'] == 1]['ERCOT']


summer_data.plot(figsize=(8, 6), label='Summer Load' , color='red')
winter_data.plot(figsize=(8, 6), label='Winter Load' , color='blue')


plt.xlabel('Date')
plt.ylabel('Load (GWh)')
plt.title('ERCOT Generation')
plt.legend()


plt.show()


# In[62]:


mean_winter_generation = df_season[df_season['Winter'] == 1]['ERCOT'].mean()
mean_summer_generation = df_season[df_season['Summer'] == 1]['ERCOT'].mean()


# In[63]:


print("Mean Winter Generation:", mean_winter_generation)
print("Mean Summer Generation:", mean_summer_generation)


# In[56]:


plt.figure(figsize=(8,6))
df_season[df_season['Summer'] == 1]['ERCOT'].hist(bins=60, alpha=0.8, label='Summer', color = 'red')
df_season[df_season['Winter'] == 1]['ERCOT'].hist(bins=60, alpha=0.8, label='Winter', color = 'blue')
plt.legend()
plt.show()


# In[78]:


correlation_matrix = df_season.corr()


plt.figure(figsize=(15, 15))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[91]:


X = df_season[['NG1:COM','NGEU:COM','BTCUSD:CUR','TSLA:US','CL1:COM','UXA:COM','RIVN:US','CPI','CO2E.KT','USAENEINF','FDTR'
       ,'UNITEDSTAOILEXP','UNITEDSTACRUOILPRO','WIND:IND','SOLAR:IND','NUCLEAR:IND','TAVG','TMIN','TMAX']]
y = df_season['ERCOT']


# In[92]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[93]:


regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train, y_train)


# In[94]:


y_pred = regressor.predict(X_test)


# In[95]:


mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[96]:


feature_importances = regressor.feature_importances_
feature_names = X.columns
indices = feature_importances.argsort()[::-1]


# In[97]:


plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), feature_importances[indices], color="b", align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45, ha="right")
plt.xlim([-1, X.shape[1]])
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()


# In[122]:


X = df_season[['NUCLEAR:IND','TAVG','TMIN','TMAX']]
Y = df_season['ERCOT']


# In[123]:


model = sm.OLS(Y, X)
results = model.fit()

print(results.summary())


# In[124]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[125]:


gb_regressor = GradientBoostingRegressor()
gb_regressor.fit(X_train, y_train)


# In[126]:


y_pred = gb_regressor.predict(X_test)


# In[127]:


mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[128]:


plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual Generation vs Predicted Generation')
plt.show()

