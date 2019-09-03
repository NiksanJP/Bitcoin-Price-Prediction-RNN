import tensorflow as tf 
import pandas as pd 
import os 
import numpy as np 

import matplotlib.pyplot as plt

print("LOADING DATA")

#Load Data
df = pd.read_csv('bitcoinDataset.csv')
df = df[['Unix Timestamp', 'Open', 'High', 'Low', 'Close']]
df = df.sort_values('Unix Timestamp', ascending=True)
#df = df.drop('Unix Timestamp', axis=1)

#Change data from old to new from new to old instead
print(df.head())
print(df.tail())

#Convert to Numpy
df = df.to_numpy()

#Select all coluns for X and just one for Y
trainX = df
trainY = df[:, df.shape[1]-1]

rows = int(trainX.shape[0])
columns = int(trainX.shape[1])

#Drop last and first to make first to a second value and Last X value to None
trainX = trainX[:(rows-1)]
trainY = trainY[1:rows]

#Reshape to trainable values
#trainX = trainX.reshape(trainX.shape[0], 1, 5)
trainY = trainY.reshape(trainY.shape[0], 1)

#Create Decision TAB
trainY = trainX[:,df.shape[1]-1] - trainY[:,0]

#Reshape to trainable values
trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
trainY = trainY.reshape(trainY.shape[0], 1)

print(trainX.shape)
print(trainY.shape)

#Declare Broker fee
brokerFees = 2.5

trainY[trainY>brokerFees] = 1000000
trainY[trainY<-brokerFees] = 2000000
trainY[(trainY<brokerFees)&(trainY>-brokerFees)] = 3000000

#Convert 1 to 0 and 2 to 1 and 3 to 2
# 0  is sell
# 1 is buy
# 2 is do nothing
newTrainY = np.zeros((trainY.shape[0],3))
print("NEW ARAY : ", newTrainY.shape)
indexes = np.where(trainY == 1000000.)
newTrainY[indexes,:] = [1,0,0]
indexes = np.where(trainY == 2000000.)
newTrainY[indexes,:] = [0,1,0]
indexes = np.where(trainY == 3000000.)
newTrainY[indexes, :] = [0,0,1]
trainY = newTrainY

#plt.hist(trainY)
#plt.savefig('dataDistribution.png')
#plt.waitforbuttonpress()

for i in range(rows-100,rows-1):
    print(trainX[i, 0, 1:], trainY[i])
    
print(trainX.shape)
print(trainY.shape)

model = tf.keras.models.Sequential([
    #INPUT LAYER
    tf.compat.v1.keras.layers.CuDNNLSTM(128, input_shape=(trainX.shape[1:]), return_sequences = True),
    tf.keras.layers.Dropout(0.5),
    #tf.keras.layers.BatchNormalization(),
    
    #HIDDEN LAYER
    tf.compat.v1.keras.layers.CuDNNLSTM(128, return_sequences = True),
    tf.keras.layers.Dropout(0.5),
    #tf.keras.layers.BatchNormalization(),

    tf.compat.v1.keras.layers.CuDNNLSTM(128),
    tf.keras.layers.Dropout(0.2),
    #tf.keras.layers.BatchNormalization(),

    #tf.keras.layers.Dense(32, activation='relu'),
    #tf.keras.layers.Dropout(0.2),

    #tf.keras.layers.Dense(16, activation='relu'),
    #tf.keras.layers.Dropout(0.2),
    
    #OUTPUT LAYER
    tf.keras.layers.Dense(3, activation='softmax')
])


optimizer = tf.keras.optimizers.Adam()
model.compile(  optimizer=optimizer,
                loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

tensorboard_callback = tf.keras.callbacks.TensorBoard()

try:
    model.load_weights('training/bitcoinWeights.h5')
    print("Model Loaded")
except Exception:
    pass

checkPointDIR = os.path.dirname("training/training/cp.ckpt")
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkPointDIR,
                                                     save_weights_only=True,
                                                     verbose=0)

while True:
    history = model.fit(trainX, trainY, epochs = 10, batch_size = 1024, shuffle=True, validation_split=0.05, callbacks=[cp_callback])

    model.save_weights('training/bitcoinWeights.h5')