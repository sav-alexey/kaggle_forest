import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
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


# Choosing the best learning rate
print("Choosing the best learning rate:")
learning_list = [1, 0.5, 0.3, 0.2, 0.1, 0.01]
for i in learning_list:
    clf = GradientBoostingClassifier(n_estimators=20, learning_rate=i, max_depth=3)
    clf.fit(X_train, Y_train)
    prediction = clf.predict(X_test)
    acc_score = accuracy_score(Y_train, prediction)
    print("Learing rate = {0}, accuracy score = {1}".format(i, acc_score))
# the best learning rate is 0.5
print("the best learning rate is 0.5")

# Choosing number of estimators
print("\nChoosing number of estimators:")
for i in range(10, 100, 5):
    clf = GradientBoostingClassifier(n_estimators=i, learning_rate=0.5, max_depth=5)
    clf.fit(X_train, Y_train)
    prediction = clf.predict(X_test)
    acc_score = accuracy_score(Y_train, prediction)
    print("n_estimators = {0}, accuracy score = {1}".format(i, acc_score))
# While number of estimators is growing accuracy is also increasing
print("While number of estimators is growing accuracy is also increasing")


# Choosing the best max_depth
print("\nChoosing the best max_depth:")   
for i in range(1, 6):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=i)
    clf.fit(X_train, Y_train)
    prediction = clf.predict(X_test)
    acc_score = accuracy_score(Y_train, prediction)
    print("max_depth = {0}, accuracy score = {1}".format(i, acc_score))
# For max_depth = 5, accuracy = 100%
print("For max_depth = 5, accuracy = 100%")

start_time = datetime.datetime.now()
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=5)
clf.fit(X_train, Y_train)
print("GradientBoostingClassifier time execution(n_estimators=100, learning_rate=0.5, max_depth=5):",
      datetime.datetime.now() - start_time)


'''
Results:
The best accuracy is when max_depth is equal to 5 and number of 
estomators is equal to 100
GradientBoostingClassifier time execution(n_estimators=100, learning_rate=0.5, max_depth=5) = 22s
''' 