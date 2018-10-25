import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
import datetime 

pandas_train = pd.read_csv("train.csv")
pandas_train = pandas_train.drop(["Id"], axis=1)
columns = pandas_train.axes[1]
data_train = np.array(pandas_train)
X_train = data_train[:,:-1]
Y_train = data_train[:,-1]
#Y_train = Y_train.reshape([Y_train.shape[0], 1])

X_train = preprocessing.scale(X_train)

iterations = 10000

pandas_test = pd.read_csv("train.csv")
pandas_test = pandas_test.drop(["Id"], axis=1)
data_test = np.array(pandas_test)
X_test = data_test[:,:-1]

tf.reset_default_graph() 
tf.set_random_seed(1)

X = tf.placeholder(tf.float32, [None, 54])
Y = tf.placeholder(tf.float32, [None, 7])

weights = (tf.random_normal_initializer, "random_normal")

Z1 = tf.contrib.layers.fully_connected(X, 7, activation_fn=tf.nn.relu, weights_initializer=weights[0])
A1 = tf.nn.softmax(Z1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z1, labels=Y))



#optimizer = (tf.train.GradientDescentOptimizer(learning_rate=0.01), "GradientDescentOptimizer")
#optimizer = (tf.train.AdadeltaOptimizer(learning_rate=0.01), "AdadeltaOptimizer")
optimizer = (tf.train.AdamOptimizer(learning_rate=0.01), "AdamOptimizer")
#optimizer = (tf.train.FtrlOptimizer(learning_rate=0.01), "FtrlOptimizer")
#optimizer = (tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.1), "MomentumOptimizer")
#optimizer = (tf.train.RMSPropOptimizer(learning_rate=0.01), "RMSPropOptimizer")

objective = optimizer[0].minimize(cost)

init = tf.global_variables_initializer()
one_hot = tf.one_hot(Y_train, 7, axis=1)
cost_array = np.zeros(iterations)
print("""\nNeural network (number of hidden layers = 1(7 hidden units), weights_initializer = {0}, 
                               learning_rate=0.01, optimizer = {1}) """.format(weights[1], optimizer[1]))
with tf.Session() as sess:
    sess.run(init)
    Y_train = sess.run(one_hot)
    start_time = datetime.datetime.now()
    for epoch in range(iterations):
        _, temp_cost, y_cap = sess.run([objective, cost, A1], feed_dict={X: X_train, Y: Y_train})
        cost_array[epoch] = temp_cost
        
    print("nn time execution:", datetime.datetime.now() - start_time)    
    predict_op = tf.argmax(Z1, 1)
    correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
    print("\nTrain Accuracy:", train_accuracy)
    
plt.ylabel('cost')
plt.xlabel('iterations')
plt.plot(range(iterations), cost_array)

'''
Results:

Optimizer = Gradient descent
Train Accuracy: 0.49861112

Optimizer = AdadeltaOptimizer
Train Accuracy: 0.22453703

Optimizer = AdamOptimizer
Train Accuracy: 0.6168651

Optimizer = FtrlOptimizer
Train Accuracy: 0.6429233
(So wierd optimizer!)

Optimizer = MomentumOptimizer
Train Accuracy: 0.5075397

Optimizer = RMSPropOptimizer
Train Accuracy: 0.606746

''' 





