#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[8]:


df = pd.read_csv('/Users/edwardmattern/Documents/Trading Economics/modelv3.csv')


# In[9]:


df['Date'] = pd.to_datetime(df['Date'])

df.set_index('Date', inplace=True)


# In[10]:


df


# In[11]:


correlation_matrix = df.corr()


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[12]:


sns.pairplot(df)
plt.show()


# In[21]:


X = df[['NG1:COM','NGEU:COM','TSLA:US','CL1:COM','UXA:COM']]
Y = df['ERCOT']


# In[22]:


model = sm.OLS(Y, X)
results = model.fit()

print(results.summary())


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[25]:


gb_regressor = GradientBoostingRegressor()
gb_regressor.fit(X_train, y_train)


# In[26]:


y_pred = gb_regressor.predict(X_test)


# In[28]:


mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[30]:


plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual Generation vs Predicted Generation')
plt.show()


# In[ ]:




