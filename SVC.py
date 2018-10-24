import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import preprocessing
import datetime 

pandas_train = pd.read_csv("train.csv")
pandas_train = pandas_train.drop(["Id"], axis=1)
columns = pandas_train.axes[1]
data_train = np.array(pandas_train)
X_train = data_train[:,:-1]
Y_train = data_train[:,-1]
#Y_train = Y_train.reshape([Y_train.shape[0], 1])

pandas_test = pd.read_csv("train.csv")
pandas_test = pandas_test.drop(["Id"], axis=1)
data_test = np.array(pandas_test)
X_test = data_test[:,:-1]


print("SVC without preprocessing:")
start_time = datetime.datetime.now()
clf = SVC(C=1, kernel='rbf')
clf.fit(X_train, Y_train)
print("SVC time execution(kernel='rbf' and C=1 ):",
      datetime.datetime.now() - start_time)

prediction = clf.predict(X_test)
acc_score = accuracy_score(Y_train, prediction)
print("Accuracy score = {0}".format(acc_score))

# Scaling
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

print("\nSVC with scaling:")
start_time = datetime.datetime.now()
clf = SVC(C=1, kernel='rbf')
clf.fit(X_train, Y_train)
print("SVC time execution(kernel='rbf' and C=1 ):",
      datetime.datetime.now() - start_time)

prediction = clf.predict(X_test)
acc_score = accuracy_score(Y_train, prediction)
print("Accuracy score = {0}".format(acc_score))

#Normalizing
X_train = preprocessing.normalize(X_train, norm='l2')
X_test = preprocessing.normalize(X_test, norm='l2')

print("\nSVC with normalizing:")
start_time = datetime.datetime.now()
clf = SVC(C=1, kernel='rbf')
clf.fit(X_train, Y_train)
print("SVC time execution(kernel='rbf' and C=1 ):",
      datetime.datetime.now() - start_time)

prediction = clf.predict(X_test)
acc_score = accuracy_score(Y_train, prediction)
print("Accuracy score = {0}".format(acc_score))

'''
Results:
Without preprocessing SVC time execution = 45s and accuracy = 1
With scaling SVC time execution = 6s but accuracy = 0.75
With normalization  SVC time execution = 10s accuracy = 0.61
''' 