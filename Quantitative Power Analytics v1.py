#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


model = pd.read_csv('/Users/edwardmattern/Documents/Trading Economics/model1.csv')


# In[3]:


df = pd.DataFrame(model)
df['Date'] = pd.to_datetime(df['Date'])


# In[4]:


df


# In[5]:


plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['NG_Price'], color='red')
plt.title('Annual Natural Gas Prices (US)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True)
plt.show()


# In[39]:


plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['NGEU_Price'], color='blue')
plt.title('Annual Natural Gas Prices (EU)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True)
plt.show()


# In[41]:


plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Coal_Price'], color='black')
plt.title('Annual Coal Prices')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True)
plt.show()


# In[14]:


plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['NGUK_Price'], color='purple')
plt.title('Annual Natural Gas Prices (UK)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True)
plt.show()


# In[50]:


correlation_matrix = df.corr()


plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[15]:


plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['HDD'], color='orangered')
plt.title('Heating Degree Days')
plt.xlabel('Date')
plt.ylabel('HDD')
plt.grid(True)
plt.show()


# In[59]:


plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['CDD'], color='navy')
plt.title('Cooling Degree Days')
plt.xlabel('Date')
plt.ylabel('CDD')
plt.grid(True)
plt.show()


# In[6]:


# Random Forest Machine Learning Model
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# In[7]:


features = ['NGEU_Price', 'NGUK_Price', 'Coal_Price', 'HDD', 'CDD', 'GDP_Growth', 'Inflation_Rate', 'Interest_Rate']
target = 'NG_Price'
X = df[features]
y = df[target]


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[9]:


imputer = KNNImputer(n_neighbors=5)  
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)


# In[10]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)


# In[11]:


rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)


# In[12]:


y_pred = rf_model.predict(X_test_scaled)


# In[13]:


mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

