import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import datetime 
import matplotlib.pyplot as plt

pandas_train = pd.read_csv("train.csv")
X_pandas = pandas_train.drop(["Cover_Type", "Id"], axis=1)
columns = pandas_train.axes[1]
X = np.array(X_pandas)
Y = np.array(pandas_train)[:,-1]
Y = Y.reshape([Y.shape[0], 1])

#pandas_test = pd.read_csv("test.csv")
#pandas_test = pandas_test.drop(["Id"], axis=1)
#X_test = np.array(pandas_test)

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=42)

acc_train, acc_valid = [], []
print("Choosing the best depth for criterion='entropy':")
for i in range(3, 35):  
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=i, random_state=1)
    clf.fit(X_train, Y_train)
    prediction = clf.predict(X_train)
    acc_score_train = accuracy_score(Y_train, prediction)
    acc_train.append(acc_score_train)
    prediction = clf.predict(X_valid)
    acc_score_valid = accuracy_score(Y_valid, prediction)
    acc_valid.append(acc_score_valid)
    print("""Depth = {0}, Train accuracy = {1},
          Valid accuracy = {2}""".format(i, acc_score_train, acc_score_valid))
print("The best depth is 26")

plt.plot(range(3, 35), acc_train)
plt.plot(range(3, 35), acc_valid)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('depth')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

acc_train, acc_valid = [], []    
print("\nChoosing the best depth for criterion='gini':")
for i in range(3, 35):  
    clf = DecisionTreeClassifier(criterion='gini', max_depth=i, random_state=1)
    clf.fit(X_train, Y_train)
    prediction = clf.predict(X_train)
    acc_score_train = accuracy_score(Y_train, prediction)
    acc_train.append(acc_score_train)
    prediction = clf.predict(X_valid)
    acc_score_valid = accuracy_score(Y_valid, prediction)
    acc_valid.append(acc_score_valid)
    print("""Depth = {0}, Train accuracy = {1}",
          Valid accuracy = {2}""".format(i, acc_score_train, acc_score_valid)) 
print("The best depth is 31")    

plt.plot(range(3, 35), acc_train)
plt.plot(range(3, 35), acc_valid)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('depth')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
  
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

X_pandas = pandas_train.drop(["Cover_Type", "Id"], axis=1)
columns = pandas_train.axes[1]
X = np.array(X_pandas)
Y = np.array(pandas_train)[:,-1]
Y = Y.reshape([Y.shape[0], 1])
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=42)
## Deleting not important features from test data
#for f in zero_imp_columns:
#    pandas_test = pandas_test.drop([f], axis=1)
 
# Time execution for criterion='entropy' and depth=25 
start_time = datetime.datetime.now()
clf = DecisionTreeClassifier(criterion='entropy', max_depth=26, random_state=1)
clf.fit(X_train, Y_train)
prediction = clf.predict(X_train)
acc_score_train = accuracy_score(Y_train, prediction)
prediction = clf.predict(X_valid)
acc_score_valid = accuracy_score(Y_valid, prediction)
print("""\nDecision tree time execution(criterion='entropy' and depth=26 ): {0},
      Train accuracy = {1},
      Valid accuracy = {2}""".format(datetime.datetime.now() - start_time, acc_score_train, acc_score_valid))

# Time execution for criterion='gini' and depth=32 
start_time = datetime.datetime.now()
clf = DecisionTreeClassifier(criterion='gini', max_depth=31, random_state=1)
clf.fit(X_train, Y_train)
print("""\nDecision tree time execution(criterion='gini' and depth=31 ): {0},
      Train accuracy = {1},
      Valid accuracy = {2}""".format(datetime.datetime.now() - start_time, acc_score_train, acc_score_valid))
     
'''
Results:
the best depth for criterion='entropy' = 25
Decision tree time execution(criterion='entropy' and depth=25 ): 0:00:00.177472
the best depth for criterion='gini' = 32
Decision tree time execution(criterion='gini' and depth=32 ): 0:00:00.142379
''' 

