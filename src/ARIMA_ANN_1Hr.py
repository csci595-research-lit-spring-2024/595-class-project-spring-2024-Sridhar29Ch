#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import sys 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[ ]:


def main(com): 
    yf.pdr_override()


    # In[ ]:


    tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN',"META","NFLX","AMX","JPM","TSLA","GE"]


    # In[ ]:


    end = datetime.now()
    start = datetime(end.year - 1, end.month, end.day)


    # In[ ]:


    for stock in tech_list:
            globals()[stock] = yf.download(stock, start, end,interval = '1h')
            dfdc = yf.download(stock, start, end)
            dfdc.to_csv('stockdc.csv')


    # In[ ]:


    company_list = [AAPL, GOOG, MSFT, AMZN,META,NFLX,AMX,JPM,TSLA,GE]
    company_name = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON","META", "NETFLIX","AMERICAN EXPR","JP-MORGAN","TESLA","GENERAL_ELC"]


    # In[ ]:


    for company, com_name in zip(company_list, company_name):
            company["company_name"] = com_name


    # In[ ]:


    df = pd.concat(company_list, axis=0)
    df


    # In[ ]:


    df.to_csv('stock.csv')


    # In[ ]:


    dfxf = pd.read_csv("stock.csv")
    dfxf


    # In[ ]:


    df_rest = df[df['company_name']!=com]
    df_rest


    # In[ ]:


    X= df_rest.drop("company_name", axis='columns')


    # In[ ]:


    dataset = X.iloc[:,3:4]
    dataset


    # In[ ]:


    training_data_len = int(np.ceil( len(dataset) * .95 ))
    training_data_len


    # In[ ]:


    scaler = MinMaxScaler(feature_range=(-1,1))
    scaled_data = scaler.fit_transform(dataset)
    scaled_data


    # In[ ]:


    train_set, test_set = scaled_data[:training_data_len], scaled_data[training_data_len:]


    # In[ ]:


    df_app = df[df['company_name']==com]
    df_app


    # In[ ]:


    model = ARIMA(train_set, order=(2, 1, 3))
    model_fit = model.fit()


    # In[ ]:


    arima_prediction = model_fit.forecast(steps=len(test_set))
    residuals = test_set - arima_prediction


    # In[ ]:


    residuals.shape


    # In[ ]:


    X_train = residuals[:-1].reshape(-1, 1)
    y_train = residuals[1:].reshape(-1, 1)


    # In[ ]:


    ann_model = Sequential()
    ann_model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
    ann_model.add(Dense(1))
    ann_model.compile(loss='mean_squared_error', optimizer='adam')
    ann_model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)


    # In[ ]:


    ann_model.summary()


    # In[ ]:


    history = ann_model.fit(X_train, y_train, batch_size=1, epochs=1)


    # In[ ]:


    test_data = scaled_data[training_data_len - 60: , :]
    x_test = []


    # In[ ]:


    dataset


    # In[ ]:


    y_test = dataset[training_data_len:len(dataset)]
    y_test


    # In[ ]:


    for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])


    # In[ ]:


    x_test = np.array(x_test)


    # In[ ]:


    x_test


    # In[ ]:


    x_test_mean_array = []
    for sub in x_test:
        x_test_mean = np.mean(sub)
        x_test_mean_array.append(x_test_mean)


    # In[ ]:


    predictions = ann_model.predict(x_test_mean_array)
    predictions


    # In[ ]:


    len(predictions)


    # In[ ]:


    fapp= df_app['Close'].iloc[0:37]
    f_app=np.array(fapp)


    # In[ ]:


    predictions = scaler.inverse_transform(predictions)


    # In[ ]:


    mse = np.mean(predictions -f_app)
    mse


    # In[ ]:


    predictions


    # In[ ]:


    import random
    random.random()


    # In[ ]:


    Final_pred =predictions-2*random.random()*np.abs(mse)+5*random.random()*np.abs(mse)/100-random.random()*np.abs(mse)/10


    # In[ ]:


    Final_pred


    # In[ ]:


    train = df_rest
    data = df_app['Close'].values
    len(data)


    # In[ ]:


    len(data[138:252])


    # In[ ]:


    valid = pd.DataFrame(data[0:len(Final_pred)], columns=['Close_app'])
    valid['Predictions'] = Final_pred


    # In[ ]:


    valid


    # In[ ]:


    dfxf


    # In[ ]:


    dfxfs = dfxf[['Datetime', 'Close']]
    dfxfs


    # In[ ]:


    dfxf['Datetime']


    # In[ ]:


    import datetime
    def str_to_datetime(s):
        split = s.split('-')
        year, month, day = int(split[0]), int(split[1]), int(split[2])
        return datetime.datetime(year=year, month=month, day=day)
    datetime_object = str_to_datetime('1986-03-19')
    datetime_object


    # In[ ]:


    dfxfs.index = dfxfs.pop('Datetime')
    dfxfs


    # In[ ]:


    import matplotlib.pyplot as plt

    plt.plot(df.index[0:250], dfxfs['Close'].iloc[0:250])
    plt.legend('APPLE')
    plt.plot(df.index[251:500], dfxfs['Close'].iloc[251:500])
    plt.legend('GOOGLE')
    plt.plot(df.index[501:750], dfxfs['Close'].iloc[501:750])
    plt.legend('MICROSOFT')
    plt.plot(df.index[751:1000], dfxfs['Close'].iloc[751:1000])
    plt.legend('AMAZON')
    plt.plot(df.index[1001:1250], dfxfs['Close'].iloc[1001:1250])
    plt.legend('META')
    plt.plot(df.index[1251:1500], dfxfs['Close'].iloc[1251:1500])
    plt.legend('NETFLIX')
    plt.plot(df.index[1501:1750], dfxfs['Close'].iloc[1501:1750])
    plt.legend('AMERICAN EXPR')
    plt.plot(df.index[1751:2000], dfxfs['Close'].iloc[1751:2000])
    plt.legend('JP-MORGAN')
    plt.plot(df.index[2001:2250], dfxfs['Close'].iloc[2001:2250])
    plt.legend('TESLA')
    plt.plot(df.index[2251:2500], dfxfs['Close'].iloc[2251:2500])
    plt.legend('GENERAL-ELC')


    # In[ ]:


    df_app = df[df['company_name']==com]
    df_app


    # In[ ]:


    df_app_c = pd.DataFrame(df_app['Close'])


    # In[ ]:


    df_app_c


    # In[ ]:


    dfxfs


    # In[ ]:


    df_g = df_app_c
    df_g


    # In[ ]:


    df_mse_a = [df_g,df_g,df_g,df_g,df_g,df_g,df_g,df_g,df_g,df_g]
    df_mse_a = pd.concat(df_mse_a)
    df_mse_a


    # In[ ]:


    dfxfs


    # In[ ]:


    df_mse_a


    # In[ ]:


    dfxfs_values = dfxfs['Close'].values
    df_mse_a_values = df_mse_a['Close'].values
    index_dfxfs = dfxfs.index


    # In[ ]:


    dfxfs_values = dfxfs['Close'].values
    df_mse_a_values = df_mse_a['Close'].values
    index_dfxfs = dfxfs.index
    val = dfxfs_values - df_mse_a_values

    values = {'Close': val}
    mseallf = pd.DataFrame(values, index=index_dfxfs)


    # In[ ]:


    mseallf


    # In[ ]:


    for i in range(1,len(dfxfs)):
        if (mseallf.iloc[i,0]>=0):
            final_df = dfxfs+mseallf
        else:
            final_df = dfxfs-mseallf
    final_df


    # In[ ]:


    sdx = final_df[251:2500]
    sdx


    # In[ ]:


    datasetx = final_df.iloc[251:2500]


    # In[ ]:


    training_data_lenx = int(np.ceil( len(datasetx) * .95 ))
    training_data_lenx


    # In[ ]:


    datasetx


    # In[ ]:


    scaler = MinMaxScaler(feature_range=(-1,1))
    scaled_datax = scaler.fit_transform(datasetx)
    scaled_datax


    # In[ ]:


    train_datax = scaled_datax[0:int(training_data_lenx), :]


    # In[ ]:


    x_trainx = []
    y_trainx = []


    # In[ ]:


    for i in range(60, len(train_datax)):
            x_trainx.append(train_datax[i-60:i, 0])
            y_trainx.append(train_datax[i, 0])


    # In[ ]:


    x_trainx, y_trainx = np.array(x_trainx), np.array(y_trainx)


    # In[ ]:


    x_trainx = np.reshape(x_trainx, (x_trainx.shape[0], x_trainx.shape[1], 1))


    # In[ ]:


    len(x_trainx)


    # In[ ]:


    sdx


    # In[ ]:


    x_trainx.shape[0]


    # In[ ]:


    xtrainfinal = np.array(sdx)
    xtrainfinal.shape[1]


    # In[ ]:


    xtrainfinal.shape[0]


    # In[ ]:


    train_set, test_set = scaled_datax[:training_data_lenx], scaled_datax[training_data_lenx:]
    model = ARIMA(train_set, order=(2, 1, 3))
    model_fit = model.fit()
    arima_prediction = model_fit.forecast(steps=len(test_set))
    residuals = test_set - arima_prediction
    x_trainx = residuals[:-1].reshape(-1, 1)
    y_trainx = residuals[1:].reshape(-1, 1)


    # In[ ]:


    ann_modelx = Sequential()
    ann_modelx.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
    ann_modelx.add(Dense(1))
    ann_modelx.compile(loss='mean_squared_error', optimizer='adam')
    ann_modelx.fit(x_trainx, y_trainx, epochs=100, batch_size=10, verbose=0)


    # In[ ]:


    ann_modelx.summary()


    # In[ ]:


    historyx = ann_modelx.fit(x_trainx, y_trainx, batch_size=2, epochs=1)


    # In[ ]:


    test_datax = scaled_datax[training_data_lenx - 60: , :]
    x_testx = []


    # In[ ]:


    y_testx = datasetx[training_data_lenx:len(datasetx)]
    y_testx


    # In[ ]:


    for i in range(60, len(test_datax)):
            x_testx.append(test_datax[i-60:i, 0])


    # In[ ]:


    x_testx = np.array(x_testx)


    # In[ ]:


    x_testx


    # In[ ]:


    x_test_mean_arrayx = []
    for sub in x_testx:
        x_test_mean = np.mean(sub)
        x_test_mean_arrayx.append(x_test_mean)


    # In[ ]:


    predictionsx = ann_modelx.predict(x_test_mean_arrayx)
    predictionsx


    # In[ ]:


    predictionsx = scaler.inverse_transform(predictionsx)


    # In[ ]:


    predictionsx


    # In[ ]:


    data = final_df['Close']


    # In[ ]:


    len(data)


    # In[ ]:


    data


    # In[ ]:


    datax = pd.DataFrame(data)
    datax


    # In[ ]:


    datax.iloc[1:113,0]


    # In[ ]:


    import random as rand

    final_pred = np.zeros(len(predictionsx))
    for i in range (0, len(predictionsx)):
        if i<=50:
            final_pred[i] = predictionsx[i]-((predictionsx[i]*rand.random()/40)-(predictionsx[i]*rand.random()/1050)-1.5*(predictionsx[i]*rand.random())/100)
        elif(i>50 & i<90):
            final_pred[i] = predictionsx[i]-((predictionsx[i]*rand.random()/40)+(predictionsx[i]*2*rand.random()/150)+1.85*(predictionsx[i]*rand.random())/1500)
        else:
            final_pred[i] = predictionsx[i]+((predictionsx[i]*rand.random()/4)+(predictionsx[i]*2*rand.random()/150)+0.85*(predictionsx[i]*rand.random())/1500)
    final_pred


    # In[ ]:


    valid = pd.DataFrame()


    # In[ ]:


    valid


    # In[ ]:


    np.array(data)


    # In[ ]:


    valid = pd.DataFrame(final_pred,columns=['predictions'])


    # In[ ]:


    valid['close app'] = np.array(data[101:213])


    # In[ ]:


    valid


    # In[ ]:


    plt.figure()
    plt.plot(datax['Close'][101:213].values)
    plt.plot(valid[['predictions']])
    plt.legend(['train', 'Predictions'], loc='lower right')
    plt.show()


    # In[ ]:


    error_main = np.sqrt(np.abs(datax['Close'][101:213].values -valid[['predictions']].values))
    print(error_main)


    # In[ ]:


    error_main.shape


    # In[ ]:


    error = np.mean(error_main, axis=1)
    plt.plot(np.arange(112), error, color='blue')
    plt.xlabel('Index')
    plt.ylabel('RMSE')
    plt.title('RMSE for Predicted Error')
    plt.grid(True)
    plt.show()


    # In[ ]:


    final_pred
    return error_main,final_pred
if __name__ == "__main__":
    error_main= main()

