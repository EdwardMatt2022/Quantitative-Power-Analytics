#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error


# In[4]:


df = pd.read_csv('/Users/edwardmattern/Documents/power_model_final.csv')


# In[5]:


df['DATE'] = pd.to_datetime(df['DATE'])

df.set_index('DATE', inplace=True)


# In[6]:


df


# In[7]:


df.describe()


# In[8]:


correlation_matrix = df.corr()
plt.figure(figsize=(16, 16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[9]:


df['LOAD_MW'].plot(figsize=(8, 6), label='Load MW')
plt.xlabel('Date')
plt.ylabel('Load (MWh)')
plt.title('ERCOT Load')
plt.legend()


plt.show()


# In[11]:


X1 = df[['AWND','PRCP','TAVG','TMAX','TMIN','CO2_PRICE','OIL_PRICE','COAL_PRICE','NG1_PRICE','LNG_PRICE','NG_STORAGE'
        ,'LMP_H','LMP_N','LMP_W','LMP_S']]
y1 = df['LOAD_MW']


# In[12]:


X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=0)


# In[13]:


model = LinearRegression()
model.fit(X1_train, y1_train)


# In[14]:


y1_pred = model.predict(X1_test)
mae = mean_absolute_error(y1_test, y1_pred)
print(f"Mean Absolute Error: {mae:.2f}")
mape = mean_absolute_percentage_error(y1_test,y1_pred)
print(f"Mean Absolute Percentage Error: {mape:.2f}")


# In[20]:


X2 = df[['TAVG','TMAX','TMIN','LMP_H','LMP_N','LMP_W','LMP_S','AWND','PRCP','COAL_PRICE','OIL_PRICE','CO2_PRICE','NG1_PRICE'
       ,'WT_FOG','WT_HFOG','WT_ICE','WT_GLAZE','WT_HAZE','WT_THNDR']]
y2 = df['LOAD_MW']


# In[21]:


X2_train, X2_test, y2_train, y2_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


regressor = GradientBoostingRegressor(random_state=42)
regressor.fit(X2_train, y2_train)


# In[23]:


y2_pred = regressor.predict(X2_test)


# In[25]:


mae = mean_absolute_error(y2_test, y2_pred)
mape = mean_absolute_percentage_error(y2_test, y2_pred)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Absolute Percentage Error: {mape:.2f}")


# In[27]:


feature_importances = regressor.feature_importances_
feature_names = X.columns
indices = feature_importances.argsort()[::-1]


# In[28]:


plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), feature_importances[indices], color="b", align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45, ha="right")
plt.xlim([-1, X.shape[1]])
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()


# In[40]:


X3 = df[['TAVG','TMAX','TMIN','LMP_H','LMP_N','LMP_W','LMP_S','AWND','PRCP','COAL_PRICE','OIL_PRICE','CO2_PRICE','NG1_PRICE'
       ,'WT_FOG','WT_HFOG','WT_ICE','WT_GLAZE','WT_HAZE','WT_THNDR']]
y3 = df['LOAD_MW']


# In[41]:


X3_train, X3_test, y3_train, y3_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[46]:


xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1, max_depth=3)


# In[47]:


xgb_model.fit(X3_train, y3_train)
y3_pred = xgb_model.predict(X3_test)


# In[48]:


mae = mean_absolute_error(y3_test, y3_pred)
mape = mean_absolute_percentage_error(y3_test, y3_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")


# In[75]:


df['TSQ'] = df['TAVG'] ** 2
print(f"Temperature Squared (TSQ): {df['TSQ']}")


# In[76]:


df['Physics_Load_Estimate'] = 0.5 * df['TAVG'] + 0.05 * df['TSQ'] + 1.2


# In[77]:


df['Residual_Load'] = df['LOAD_MW'] - df['Physics_Load_Estimate']


# In[78]:


X4 = df[['TAVG','TMAX','TMIN','LMP_H','LMP_N','LMP_W','LMP_S','AWND','PRCP','COAL_PRICE','OIL_PRICE','CO2_PRICE','NG1_PRICE'
       ,'WT_FOG','WT_HFOG','WT_ICE','WT_GLAZE','WT_HAZE','WT_THNDR']].values
y4 = df['Residual_Load'].values


# In[79]:


X4_train, X4_test, y4_train, y4_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[80]:


model4 = GradientBoostingRegressor()
model4.fit(X4_train, y4_train)


# In[81]:


y4_pred_residual = model4.predict(X4_test)


# In[82]:


df['ML_Residual_Pred'] = model4.predict(X4)  
df['Hybrid_Load_Prediction'] = df['Physics_Load_Estimate'] + df['ML_Residual_Pred']


# In[84]:


print("Mean Absolute Error:", mean_absolute_error(df['LOAD_MW'], df['Hybrid_Load_Prediction']))
print("Mean Absolute Percentage Error:", mean_absolute_percentage_error(df['LOAD_MW'], df['Hybrid_Load_Prediction']))


# In[50]:


plt.figure(figsize=(10, 6))
plt.plot(y1_test.values, label="True Load", color="blue")
plt.plot(y1_pred, label="Predicted Load", color="red", linestyle="--")
plt.title("Machine Learning Model 1 (Linear Regression))
plt.xlabel("Test Samples")
plt.ylabel("Load (MW)")
plt.legend()
plt.show()


# In[52]:


plt.figure(figsize=(10, 6))
plt.plot(y2_test.values, label="True Load", color="blue")
plt.plot(y2_pred, label="Predicted Load", color="red", linestyle="--")
plt.title("Machine Learning Model 2 (Gradient Boost)")
plt.xlabel("Test Samples")
plt.ylabel("Load (MW)")
plt.legend()
plt.show()


# In[56]:


plt.figure(figsize=(10, 6))
plt.plot(y3_test.values, label="True Load", color="blue")
plt.plot(y3_pred, label="Predicted Load", color="black", linestyle="--")
plt.title("Machine Learning Model 3 (XG Boost)")
plt.xlabel("Test Samples")
plt.ylabel("Load (MW)")
plt.legend()
plt.show()


# In[88]:


plt.figure(figsize=(12, 6))
plt.plot(df.index, df['LOAD_MW'], label='Actual Load', color='blue')
plt.plot(df.index, df['Hybrid_Load_Prediction'], label='Hybrid Model Prediction', color='gold')
plt.xlabel("Date")
plt.ylabel("Load (MW)")
plt.title("Hybrid Physics Machine Learning Model 4: Actual vs Predicted Load")
plt.legend()
plt.show()

