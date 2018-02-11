
# coding: utf-8

# In[1]:

import pandas as pd
from pandas_datareader import data, wb
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

start = datetime.datetime(2014, 6, 1)
end = datetime.datetime(2018, 1, 1)

tickers = ['AAPL']
data = data.DataReader(['AAPL'], 'yahoo', start, end)
data = data['Open']
data = data.sort_index(axis=0, ascending=True)
data.head()


# In[2]:

# Calculate daily differences
data['diff'] = data['AAPL'].diff(periods=1)


# In[4]:

# Calcultate the cumulative returns
data['cum'] = data['diff'].cumsum()


# In[6]:

#Meanreversion
# Setting position long = 1 and short = -1 based on previous day move
delta = 0.005
# If previous day price difference was less than or equal then delta, we go long
# If previous day price difference was more than or equal then delta, we go short
data['position_mr'] = np.where(data['diff'].shift(1) <= -delta,1, np.where(data['diff'].shift(1) >= delta, -1, 0))
data['result_mr'] = (data['diff'] * data['position_mr']).cumsum()


# In[8]:

# We will filter execution of our strategy by only executing if our result are above it's 200 day moving average
win =200
data['ma_mr'] = pd.rolling_mean(data['result_mr'], window=win)
filtering_mr = data['result_mr'].shift(1) > data['ma_mr'].shift(1)
data['filteredresult_mr'] = np.where(filtering_mr, data['diff'] * data['position_mr'], 0).cumsum()
# if we do not want to filter we use below line of code
# df['filteredresult_mr'] = (df['diff'] * df['position_mr']).cumsum()
data[['ma_mr','result_mr','filteredresult_mr']].plot(figsize=(10,8))


# In[9]:

# Breakout
# Setting position long = 1 and short = -1 based on previous day move
# By setting the delta to negative we are switching the strategy to Breakout
delta = -0.01
# If previous day price difference was less than or equal then delta, we go long
# If previous day price difference was more than or equal then delta, we go short
data['position_bo'] = np.where(data['diff'].shift(1) <= -delta,1, np.where(data['diff'].shift(1) >= delta, -1, 0))
data['result_bo'] = (data['diff'] * data['position_bo']).cumsum()


# In[10]:

# We will filter execution of our strategy by only executing if our result are above it's 200 day moving average
win = 200
data['ma_bo'] = pd.rolling_mean(data['result_bo'], window=win)
filtering_bo = data['result_bo'].shift(1) > data['ma_bo'].shift(1)
data['filteredresult_bo'] = np.where(filtering_bo, data['diff'] * data['position_bo'], 0).cumsum()
# df['filteredresult_bo'] = (df['diff'] * df['position_bo']).cumsum()
data[['ma_bo','result_bo','filteredresult_bo']].plot(figsize=(10,8))


# In[11]:

# Here we combine the Meanreversion and the Breakout strategy results
data['combi'] = data['filteredresult_mr'] + data['filteredresult_bo']
data[['combi','filteredresult_mr','filteredresult_bo']].plot(figsize=(10,8))


# In[13]:

print("Total return since 2000:",data['combi'][-1] * 100, "%")


# In[14]:

def sharpe(serie):
    std = serie.std()
    mean = serie.mean()
    return (mean / std) * 252 ** 0.5


# In[16]:

print("Sharpe ratios")
print("Combi", sharpe(data['combi'].diff(periods=1)))
print("Mr", sharpe(data['filteredresult_mr'].diff(periods=1)))
print("Bo", sharpe(data['filteredresult_bo'].diff(periods=1)))


# A n-day Support Resistance Breakout system with a volatility bases trailing stop-loss

# In[3]:

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from talib.abstract import *
import pandas_datareader.data as web


def price(stock, start):
    price = web.DataReader(name=stock, data_source='yahoo', start=start)['Adj Close']
    a = pd.DataFrame(price.div(price.iat[0]))
    a.columns = [stock]
    return a

def breakout(data, buyperiod=20, sellperiod=10):
    a = pd.DataFrame()
    b = list(data)
    for i in b:
        a[i] = data.iloc[:,0]
        a[i+'buysignal'] = np.where(a[i]>a[i].rolling(buyperiod).max().shift(1),1,0)
        a[i+'exitsignal'] = np.where(a[i]<a[i].rolling(sellperiod).min().shift(1),1,0)
        a[i+'marketposition'] = np.where(a[i+'buysignal']==1,1,np.where(a[i+'exitsignal']==1,0,np.where(a[i+'marketposition'].shift(1)==1,1,0)))
        a[i+'profit'] = np.where(a[i+'marketposition'].shift(1)==0,1,a[i]/a[i].shift(1))
        a.dropna()
        a[i+'profit'] = a[i+'profit'].cumprod()
    return a

a = price('SPY','2000-03-01')
print(breakout(a,20,10))


# In[2]:

def strategy_performance('AAPL')


# In[ ]:



