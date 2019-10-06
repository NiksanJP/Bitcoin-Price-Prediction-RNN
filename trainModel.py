import tensorflow as tf 
import pandas as pd 
import os 
import numpy as np 
import random

import matplotlib.pyplot as plt
from keras.regularizers import l1

trainX = np.load('trainTestData/trainX.npy')
trainY = np.load('trainTestData/trainY.npy')
testX = np.load('trainTestData/testX.npy')
testY = np.load('trainTestData/testY.npy')

trainX = trainX.astype(int)
trainY = trainY.astype(int)
testX = testX.astype(int)
testY = testY.astype(int)

print(trainX.shape, trainY.shape)
print(testX.shape, testY.shape)

print(np.isnan(trainX).any())
print(np.isnan(trainY).any())
print(np.isnan(testX).any())
print(np.isnan(testY).any())

print(np.any(trainX[:,:,:]==0))
print(np.any(trainY[:,:]==0))
print(np.any(testX[:,:,:]==0))
print(np.any(testY[:,:]==0))

for i in range(10):
    num = random.randint(0, trainX.shape[0])
    print(trainX[num], trainY[num])

model = tf.keras.models.Sequential([
    #INPUT LAYER
    tf.compat.v1.keras.layers.CuDNNLSTM(128, input_shape=(trainX.shape[1:]), return_sequences = True),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    
    #HIDDEN LAYER
    tf.compat.v1.keras.layers.CuDNNLSTM(128, return_sequences = True),
    #tf.keras.layers.Dropout(0.1),
    tf.keras.layers.BatchNormalization(),

    tf.compat.v1.keras.layers.CuDNNLSTM(128),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(32, activation='tanh'),
    #tf.keras.layers.Dropout(0.2),
    
    #OUTPUT LAYER
    tf.keras.layers.Dense(3, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam()
opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1.0)

model.compile(  optimizer=opt,
                loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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
    history = model.fit(trainX, trainY, epochs = 10, batch_size = 256, validation_data=(testX, testY))

    model.save_weights('training/bitcoinWeights.h5')