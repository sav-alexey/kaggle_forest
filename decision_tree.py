import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import datetime 

pandas_train = pd.read_csv("train.csv")
pandas_train = pandas_train.drop(["Id"], axis=1)
columns = pandas_train.axes[1]
data_train = np.array(pandas_train)
X_train = data_train[:,:-1]
Y_train = data_train[:,-1]
Y_train = Y_train.reshape([Y_train.shape[0], 1])

pandas_test = pd.read_csv("train.csv")
pandas_test = pandas_test.drop(["Id"], axis=1)
data_test = np.array(pandas_test)
X_test = data_test[:,:-1]

# Choosing the best depth for criterion='entropy'
for i in range(3, 35):  
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=i, random_state=1)
    clf.fit(X_train, Y_train)
    prediction = clf.predict(X_test)
    acc_score = accuracy_score(Y_train, prediction)
    print("Depth = {0}, Accuracy = {1}".format(i, acc_score))
# The best depth is 25
    
# Choosing the best depth for criterion='gini'
print('')
for i in range(3, 35):  
    clf = DecisionTreeClassifier(criterion='gini', max_depth=i, random_state=1)
    clf.fit(X_train, Y_train)
    prediction = clf.predict(X_test)
    acc_score = accuracy_score(Y_train, prediction)
    print("Depth = {0}, Accuracy = {1}".format(i, acc_score))
# The best depth is 25    
    
    
# Extracting not important features
importances = clf.feature_importances_
zero_imp_columns = []
print("\nNot important features:")
for i, name in zip(importances, columns):
    if i == 0.0:
        print("{0}:{1}".format(name, i))
        zero_imp_columns.append(name)

# Deleting not important features from train data
for f in zero_imp_columns:
    pandas_train = pandas_train.drop([f], axis=1)

# Deleting not important features from test data
for f in zero_imp_columns:
    pandas_test = pandas_test.drop([f], axis=1)
 
# Time execution for criterion='entropy' and depth=25 
start_time = datetime.datetime.now()
clf = DecisionTreeClassifier(criterion='entropy', max_depth=25, random_state=1)
clf.fit(X_train, Y_train)
print("\nDecision tree time execution(criterion='entropy' and depth=25 ):",
      datetime.datetime.now() - start_time)

# Time execution for criterion='gini' and depth=32 
start_time = datetime.datetime.now()
clf = DecisionTreeClassifier(criterion='gini', max_depth=32, random_state=1)
clf.fit(X_train, Y_train)
print("\nDecision tree time execution(criterion='gini' and depth=32 ):",
      datetime.datetime.now() - start_time)








