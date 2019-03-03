import tensorflow as tf
import numpy as np
import xlrd
import matplotlib.pyplot as plt
import os
from sklearn.utils import check_random_state

n = 50
XX = np.arange(n)
rs = check_random_state(0)
YY = rs.randint(-20, 20, size = (n,)) +2.0 * XX
data = np.stack((XX,YY), axis = 1)#row order axis=0:column order
#print(data)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_epochs', 50, 'The number of epochs for training the model. Default=50')
 

W = tf.Variable(0.0)
b = tf.Variable(0.0)

def inference(X):
    
    return X * W + b

def loss(X,Y):
    
    Y_predicted = inference(X)
    
    return tf.reduce_sum(tf.squared_difference(Y,Y_predicted))/(2 * data.shape[0])
def train(loss):
    
    learning_rate = 0.0001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
train_loss = loss(X,Y)
train_op = train(train_loss)


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch_num in range(FLAGS.num_epochs):
        loss_value,_ = sess.run([train_loss,train_op],
                                feed_dict = {X: data[:,0], Y: data[:,1]})
        print('epoch %d,loss=%f'%(epoch_num+1,loss_value))
        wcoeff, bias = sess.run([W,b])
        
Input_values = data[:,0]
Labels = data[:,1]
Prediction_values = data[:,0] * wcoeff + bias

# # uncomment if plotting is desired!
plt.plot(Input_values, Labels, 'go',label='main')#'o' for scatter .'-'for line
plt.plot(Input_values, Prediction_values, '-',label='Predicted')
plt.show()

# # Saving the result.
#plt.legend()
# plt.savefig('plot.png')
# plt.close()
        
