import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import datetime 
import os

pandas_train = pd.read_csv("train.csv")
pandas_train = pandas_train.drop(["Id"], axis=1)
columns = pandas_train.axes[1]
data_train = np.array(pandas_train)
X_train = data_train[:,:-1]
Y_train = data_train[:,-1]
Y_train = Y_train.reshape([Y_train.shape[0], 1])

Y_train = OneHotEncoder().fit_transform(Y_train)

X_train = preprocessing.scale(X_train)

iterations = 10

pandas_test = pd.read_csv("train.csv")
pandas_test = pandas_test.drop(["Id"], axis=1)
data_test = np.array(pandas_test)
X_test = data_test[:,:-1]

path = "C:/Users/Administrator/.spyder-py3/GIT1/kaggle_forest/saved_weights"
weigts_cp = "/cp.ckpt"
checkpoint_dir = os.path.dirname(path+weigts_cp)
cp = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir, save_weights_only=True)


model = Sequential()
model.add(Dense(100, input_dim=54, activation='tanh'))
model.add(Dense(200, input_dim=100, activation='tanh'))
model.add(Dense(500, input_dim=200, activation='tanh'))
model.add(Dense(50, input_dim=500, activation='tanh'))
model.add(Dense(7, input_dim=50, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='RMSProp', metrics=['accuracy'])
start_time = datetime.datetime.now()
model.load_weights(path)
model.fit(X_train, Y_train, epochs=iterations, batch_size=None, callbacks=[cp])
print("nn time execution:", datetime.datetime.now() - start_time)    

'''
Results:
iterations = 100

Layers: 1 (7 units, act = softmax)
Train Accuracy: acc: 0.7019
nn time execution: 0:00:36.850656

Layers: 2 (15 units act = tanh)(7 units, act = softmax)
Train Accuracy: acc: 0.7899
nn time execution: 0:00:36.532806

Layers: 2 (100 units act = relu)(7 units, act = softmax)
Train Accuracy: acc: 0.8575
nn time execution: 0:00:36.657545

Layers: 2 (100 units act = tanh)(7 units, act = softmax)
Train Accuracy: acc: 0.8752
nn time execution: 0:00:35.887496

Layers: 2 (200 units act = tanh)(7 units, act = softmax)
Train Accuracy: acc: 0.9024
nn time execution: 0:00:37.368917

Layers: 3 (100 units act = tanh)(50 units act = tanh)(7 units, act = softmax)
Train Accuracy: acc: 0.9501
nn time execution: 0:00:39.905188

Layers: 4 (25 units act = tanh)(100 units act = tanh)(50 units act = tanh)(7 units, act = softmax)
Train Accuracy: acc: 0.9360
nn time execution: 0:00:39.905188

Layers: 4 (100 units act = tanh)(200 units act = tanh)(50 units act = tanh)(7 units, act = softmax)
Train Accuracy: acc: 0.9917
nn time execution: 0:00:48.154137

Layers: 5 (100 units act = tanh)(200 units act = tanh)(500 units act = tanh)
(50 units act = tanh)(7 units, act = softmax)
Train Accuracy: acc: 0.9907
nn time execution: 0:01:36.006474


'''
