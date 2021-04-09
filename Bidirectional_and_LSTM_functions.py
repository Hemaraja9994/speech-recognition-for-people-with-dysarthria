# Import Important Libararies
from keras.models import Sequential
from keras import layers
from keras.layers import Dense , Dropout , Activation , Flatten , LSTM, Input , Bidirectional , Embedding ,TimeDistributed 
from keras.optimizers import Adam
from sklearn import metrics
from sklearn.metrics import accuracy_score

def create_LSTM_model(out_lstm):
    model_LSTM = Sequential()
    model_LSTM.add(LSTM(units = out_lstm ,dropout = 0.05 , recurrent_dropout = 0.2 ,input_shape =(x_train.shape[1:])))
    model_LSTM.add(Dense(128,activation = 'softmax'))
    model_LSTM.add(Dropout(0.25))
    model_LSTM.add(Dense(1,activation = 'sigmoid'))
    model_LSTM.add(Dense(len(allLabelsInDatasetAsText),activation = 'softmax'))
    model_LSTM.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
    model_LSTM.fit(x_train,y_train,epochs = 8 ,batch_size = 64)
    model_LSTM.summary()
    return model_LSTM

def create_Bidirectional_LSTM_model(out_lstm):
    model_Bid = Sequential()
    # model_Bid.add(Embedding(9500,1000))
    model_Bid.add(Bidirectional(LSTM(units = out_lstm ,dropout = 0.05 , recurrent_dropout = 0.2 , return_sequences=True , input_shape =(x_train.shape[1:]))))
    model_Bid.add(Dropout(0.25))
    model_Bid.add(Bidirectional(LSTM(units = out_lstm ,return_sequences=True)))
    model_Bid.add(LSTM(units = 100))
    # model_Bid.add(Dense(len(allLabelsInDatasetAsText),activation = 'softmax'))
    # model_Bid.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
    # # Custom Backword Layer

    # forward_layer = Sequential()
    # forward_layer.add(LSTM(units = 50,activation='relu', return_sequences=True))
    # backward_layer = Sequential()
    # backward_layer.add(LSTM(units = 50, return_sequences=True,go_backwards=True))
    # model_Bid=Bidirectional(forward_layer, backward_layer = backward_layer,input_shape =(x_train.shape[1:]))

    # #Putting Backword layer and forword layer in Bidirectional-LSTM
    model_Bid.add(Dense(len(allLabelsInDatasetAsText),activation = 'softmax'))
    model_Bid.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
    model_Bid.fit(x_train,y_train,epochs = 4 ,batch_size = 32)
    model_Bid.summary()
    return model_Bid


    
LSTM = create_LSTM_model(out_lstm)
score = LSTM.evaluate(x_test,y_test)
print(score)



Bi_LSTM = create_Bidirectional_LSTM_model(out_lstm)
score = Bi_LSTM.evaluate(x_test,y_test)
print(score)
