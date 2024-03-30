#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler


# In[3]:


df = pd.read_csv(r"C:\Users\DSC\Desktop\stock prices.csv")
df.head()


# In[4]:


df.date = pd.to_datetime(df.date)


# In[5]:


df.dtypes


# In[6]:


df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

df.drop('date', axis=1, inplace=True)


# In[7]:


df.head()


# In[8]:


df['symbol'].value_counts()


# In[9]:


for column in df.columns:
    histograph = sns.histplot(x=df[column], data=df)
    plt.xticks(rotation=90)
    plt.show()


# In[10]:


# Remove Outliers for OPEN feature
df_filtered = df[df['open'] <= 250]
df_filtered = df[df['high'] <= 250]
df_filtered = df[df['low'] <= 250]
df_filtered = df[df['close'] <= 250]
df_filtered = df[df['volume'] <= 2400000]

df_filtered.shape


# In[11]:


le = LabelEncoder()

df_filtered['symbol']= le.fit_transform(df_filtered['symbol'])


# In[12]:


df_filtered.head()


# In[13]:


for column in df.columns:
    df_filtered[column].fillna(df_filtered[column].mean(), inplace=True)

df_filtered.isnull().sum().sum()


# In[14]:


Y = df_filtered['volume']
X = df_filtered.drop('volume', axis=1)

print(X, Y)
print(X.shape, Y.shape)


# In[15]:


X.symbol = X.symbol.astype('int16')
X.open = X.open.astype('float16')
X.high = X.high.astype('float16')
X.low = X.low.astype('float16')
X.close = X.close.astype('float16')
X.year = X.year.astype('int16')
X.month = X.month.astype('int16')
X.day = X.day.astype('int16')

print(X.dtypes, Y.dtypes)


# In[16]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[17]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train, y_train)


# In[18]:


from sklearn.metrics import mean_absolute_error

y_hat = model.predict(x_test)

mean_absolute_error(y_test, y_hat)


# In[19]:


y_test.to_numpy()


# In[20]:


data = np.array([y_test.T, y_hat.T])
data = data.T

output = pd.DataFrame(data, columns=['y_test', 'y_hat'])
output.head()


# In[21]:


sns.set_style('whitegrid')
sns.lmplot(x ='y_test', y ='y_hat', data=output)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=200, random_state=0)
model.fit(x_train[0:200000], y_train[0:200000])


# In[ ]:


y_hat = model.predict(x_test)

mean_absolute_error(y_test, y_hat)


# In[ ]:


# from sklearn.linear_model import LogisticRegression

# model = LogisticRegression()

# model.fit(x_train[0:200000], y_train[0:200000])


# In[ ]:


# y_hat = model.predict(x_test)

# mean_absolute_error(y_test, y_hat)


# In[ ]:




