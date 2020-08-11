import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tensorflow
from tensorflow.contrib.layers import fully_connected

#3d data 
data = make_blobs(n_samles=100, n_features=3, centers=2,random_state=101)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[0])
data_x = scaled_data[:,0]
data_y = scaled_data[:,1]
data_z = scaled_data[:,2]
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(data_x,data_y,data_z,c=[1])

#3d to 2d through linear auto encoder
num_inputs = 3
num_hidden = 2
num_outputs = num_inputs
learning_rate =0.01

X = tf.placeholder(tf.float32, shape=[None,num_inputs])

hidden = fully_connected(X,num_hidden,activation_fn=None)
outputs = fully_connected(hidden,num_outputs,activation_fn=None)

loss = tf.reduce_mean(tf.square(outputs-X))

optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_inializer()
num_steps = 100
with tf.Session as sess:
	sess.run(init)
	for iteration in range(num_steps):
		sess.run(train, feed_dict{X:scaled_data})
	output_2d = hidden.eval(feed_dict={X:scaled_data})

plt.scatter(output_2d[:,0],output_2d:,1,c=data[1])

