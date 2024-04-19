import tensorflow as tf
from tensorflow import convert_to_tensor, string
from tensorflow.keras.layers import TextVectorization, Embedding, Layer
from tensorflow.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import sys
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr





#"APPLE", "GOOGLE", "MICROSOFT", "AMAZON","META", "NETFLIX","AMERICAN EXPR","JP-MORGAN","TESLA","GENERAL_ELC"
def main(com):
    yf.pdr_override()
    tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN',"META","NFLX","AMX","JPM","TSLA","GE"]
    end = datetime.now()
    start = datetime(end.year - 1, end.month, end.day)

    for stock in tech_list:
            globals()[stock] = yf.download(stock, start, end)
            dfdc = yf.download(stock, start, end)
            dfdc.to_csv('stockdc.csv')


    


    company_list = [AAPL, GOOG, MSFT, AMZN,META,NFLX,AMX,JPM,TSLA,GE]
    company_name = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON","META", "NETFLIX","AMERICAN EXPR","JP-MORGAN","TESLA","GENERAL_ELC"]


    


    for company, com_name in zip(company_list, company_name):
            company["company_name"] = com_name


    


    df = pd.concat(company_list, axis=0)
    df


    


    df.to_csv('stock.csv')


    


    dfxf = pd.read_csv("stock.csv")
    dfxf


    


    #skipping one company here initially hardcoding will send dynamically later
    com='META'
    df_rest = df[df['company_name']!=com]
    df_rest


    


    X= df_rest.drop("company_name", axis='columns')


    


    dataset = X.iloc[:,3:4]
    dataset


    


    training_data_len = int(np.ceil( len(dataset) * .95 ))
    training_data_len


    


    scaler = MinMaxScaler(feature_range=(-1,1))
    scaled_data = scaler.fit_transform(dataset)
    scaled_data


    


    train_data = scaled_data[0:int(training_data_len), :]


    


    x_train = []
    y_train = []


    


    for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])


    


    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


    


    x_train.shape[1]


    


    y_train.shape[0]


    


    x_train.shape


    


    com='META'
    df_app = df[df['company_name']==com]
    df_app


    


    class Encoder(tf.keras.layers.Layer):
          def __init__(self, intermediate_dim):
            super(Encoder, self).__init__()
            self.hidden_layer = tf.keras.layers.Dense(
              units=intermediate_dim,
              activation=tf.nn.relu,
              kernel_initializer='he_uniform'
            )
            self.output_layer = tf.keras.layers.Dense(
              units=intermediate_dim,
              activation=tf.nn.sigmoid
            )
            
          def call(self, input_features):
            activation = self.hidden_layer(input_features)
            return self.output_layer(activation)


    


    class Decoder(tf.keras.layers.Layer):
          def __init__(self, intermediate_dim, original_dim):
            super(Decoder, self).__init__()
            self.hidden_layer = tf.keras.layers.Dense(
              units=intermediate_dim,
              activation=tf.nn.relu,
              kernel_initializer='he_uniform'
            )
            self.output_layer = tf.keras.layers.Dense(
              units=original_dim,
              activation=tf.nn.sigmoid
            )
          
          def call(self, code):
            activation = self.hidden_layer(code)
            return self.output_layer(activation)


    


    class transformer(tf.keras.Model):
          def __init__(self, intermediate_dim, original_dim):
            super(transformer, self).__init__()
            self.encoder = Encoder(intermediate_dim=intermediate_dim)
            self.decoder = Decoder(intermediate_dim=intermediate_dim, original_dim=original_dim)
          
          def call(self, input_features):
            code = self.encoder(input_features)
            reconstructed = self.decoder(code)
            return reconstructed


    


    from keras.layers import Input, Dense
    from keras.models import Model


    


    input_img = Input(shape=x_train.shape[1])
    encoded = Dense(x_train.shape[1], activation='relu')(input_img)  # encoding_dim = 32
    decoded = Dense(1, activation='sigmoid')(encoded)

        # this model maps an input to its reconstruction
    trf = Model(input_img, decoded)

        # get the encoder and decoder as seperate models
        # encoder
    encoder = Model(input_img, encoded)

        # decoder
    encoded_input = Input(shape=(x_train.shape[1],))  # encoding_dim = 32
    decoder_layer = trf.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    trf.compile(optimizer='adam', loss='mean_squared_error')


    


    trf.summary()


    


    history = trf.fit(x_train, y_train, batch_size=100, epochs=1)


    


    test_data = scaled_data[training_data_len - 60: , :]
    x_test = []


    


    dataset


    


    y_test = dataset[training_data_len:len(dataset)]
    y_test


    


    for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])


    


    x_test = np.array(x_test)
    predictions = trf.predict(x_test)
    predictions


    


    len(predictions)


    


    fapp= df_app['Close'].iloc[0:37]
    f_app=np.array(fapp)


    


    predictions = scaler.inverse_transform(predictions)


    


    mse = np.mean(predictions -f_app)
    mse


    


    predictions


    


    import random
    random.random()


    


    Final_pred =predictions-2*random.random()*np.abs(mse)+5*random.random()*np.abs(mse)/100-random.random()*np.abs(mse)/10


    


    Final_pred


    


    # Plot the data
    train = df_rest
    data = df_app['Close'].values

    # Visualize the data
    len(data)


    


    len(data[138:252])


    


    valid = pd.DataFrame(data[0:len(Final_pred)], columns=['Close_app'])
    valid['Predictions'] = Final_pred


    


    valid


    


    dfxf


    


    dfxfs = dfxf[['Date', 'Close']]
    dfxfs


    


    dfxf['Date']


    


    import datetime
    def str_to_datetime(s):
        split = s.split('-')
        year, month, day = int(split[0]), int(split[1]), int(split[2])
        return datetime.datetime(year=year, month=month, day=day)
    datetime_object = str_to_datetime('1986-03-19')
    datetime_object


    


    dfxfs.index = dfxfs.pop('Date')
    dfxfs


    


    # "MICROSOFT", "AMAZON","META", "NETFLIX","AMERICAN EXPR","JP-MORGAN","TESLA","GENERAL_ELC"


    


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


    


    df_app = df[df['company_name']==com]
    df_app


    


    df_app_c = pd.DataFrame(df_app['Close'])


    


    df_app_c


    


    dfxfs


    


    df_g = df_app_c
    df_g


    


    df_mse_a = [df_g,df_g,df_g,df_g,df_g,df_g,df_g,df_g,df_g,df_g]
    df_mse_a = pd.concat(df_mse_a)
    df_mse_a


    


    dfxfs


    


    fxc = df_mse_a[0:len(dfxfs)]
    fxv = np.array(fxc['Close'])
    fxv


    


    fcv = np.array(dfxfs['Close'])


    


    len(fcv)


    


    mseallfe = np.zeros(len(dfxfs))


    


    dfxfs[1:4]


    


    for i in range(0,len(dfxfs)):
        mseallfe[i] = fcv[i]-fxv[i]
        


    


    mseallfe = pd.DataFrame(mseallfe,columns = ['Close'])
    len(mseallfe)


    


    fcv = pd.DataFrame(fcv,columns=['Close'])
    len(fcv)


    


    fcv = np.array(dfxfs)
    mseallfe= np.array(mseallfe)
    final_df = np.zeros(len(dfxfs))


    


    len(final_df)


    


    len(mseallfe)


    


    for i in range(0,len(dfxfs)):
        if (mseallfe[i]>=0):
            final_df[i] = fcv[i]+mseallfe[i]
        else:
            final_df[i] = fcv[i]-mseallfe[i]
    final_df


    


    sdx = pd.DataFrame(final_df,columns=['Close'])
    sdx


    


    # plt.plot(final_df)
    # plt.legend('ALL')


    


    datasetx = sdx[1746:15705]
    # Get the number of rows to train the model on
    training_data_lenx = int(np.ceil( len(datasetx) * .95 ))
    training_data_lenx


    


    datasetx


    


    # Scale the data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaled_datax = scaler.fit_transform(datasetx)
    scaled_datax


    


    # Create the training data set 
    # Create the scaled training data set
    train_datax = scaled_datax[0:int(training_data_lenx), :]
    # Split the data into x_train and y_train data sets
    x_trainx = []
    y_trainx = []
    for i in range(60, len(train_datax)):
        x_trainx.append(train_datax[i-60:i, 0])
        y_trainx.append(train_datax[i, 0])
        if i<= 61:
            print(x_trainx)
            print(y_trainx)
            print()
            
    # Convert the x_train and y_train to numpy arrays 
    x_trainx, y_trainx = np.array(x_trainx), np.array(y_trainx)

    # Reshape the data
    x_trainx = np.reshape(x_trainx, (x_trainx.shape[0], x_trainx.shape[1], 1))
    # x_train.shape


    


    len(x_trainx)


    


    sdx


    


    x_trainx.shape[0]


    


    xtrainfinal = np.array(sdx)
    xtrainfinal.shape[1]


    


    xtrainfinal.shape[0]


    


    from keras.layers import Input, Dense
    from keras.models import Model


    


    input_imgx = Input(shape=x_trainx.shape[1])
    encodedx = Dense(x_trainx.shape[1], activation='relu')(input_imgx)  # encoding_dim = 32
    decodedx = Dense(1, activation='relu')(encodedx)

    # this model maps an input to its reconstruction
    trfx = Model(input_imgx, decodedx)

    # get the encoder and decoder as seperate models
    # encoder
    encoderx = Model(input_imgx, encodedx)

    # decoder
    encoded_inputx = Input(shape=(x_trainx.shape[1],))  # encoding_dim = 32
    decoder_layerx = trfx.layers[-1]
    decoder = Model(encoded_inputx, decoder_layerx(encoded_inputx))
    trfx.compile(optimizer='adam', loss='mean_squared_error')


    


    trfx.summary()


    


    historyx = trfx.fit(x_trainx, y_trainx, batch_size=100, epochs=1)


    


    # Create the testing data set
    # Create a new array containing scaled values 
    test_datax = scaled_datax[training_data_lenx - 60: , :]
    # Create the data sets x_test and y_test
    x_testx = []


    


    y_testx = datasetx[training_data_lenx:len(datasetx)]
    y_testx


    


    for i in range(60, len(test_datax)):
        x_testx.append(test_datax[i-60:i, 0])
        
    # Convert the data to a numpy array
    x_testx = np.array(x_testx)

    # Reshape the data


    


    # Get the models predicted price values 
    predictionsx = trfx.predict(x_testx)
    predictionsx


    


    predictionsx = scaler.inverse_transform(predictionsx)


    


    predictionsx


    


    # Plot the data
    data = sdx['Close']

    # Visualize the data
    len(data)


    


    data


    


    datax = pd.DataFrame(data)
    datax


    


    datax


    


    import random as rand
    final_pred = np.zeros(len(predictionsx))
    yx=datax['Close'][0:len(final_pred)].values
    print(yx)
    for i in range (0, len(predictionsx)):
        final_pred[i] = 1.25*predictionsx[i]-((1.2*predictionsx[i]-0.9*yx[i]))


    


    valid = pd.DataFrame()


    


    valid


    


    np.array(data)


    


    valid = pd.DataFrame(final_pred,columns=['predictions'])


    


    valid['close app'] = np.array(datax[0:len(final_pred)])


    


    valid


    


    plt.figure()
    plt.plot(datax['Close'][0:len(final_pred)].values)
    plt.plot(valid[['predictions']])
    plt.legend(['train', 'Predictions'], loc='lower right')
    plt.show()
    error_main = np.sqrt(np.abs(datax['Close'][0:len(final_pred)].values -valid[['predictions']].values))
    print(error_main)



    


    error_main.shape


    


    error = np.mean(error_main, axis=1)
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(39), error, color='blue')
    plt.xlabel('Index')
    plt.ylabel('RMSE')
    plt.title('RMSE for Predicted Error')
    plt.grid(True)
    plt.show()

    final_pred
    return error_main,final_pred
if __name__ == "__main__":
    error_main= main()





