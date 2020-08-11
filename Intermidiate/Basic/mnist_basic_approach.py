import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as pyplot

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#flattening single image
#single_image = mnist.train.images[1].reshape(28, 28)
#plt.imshow(single_image,cmap='gist_gray')

#placeholders
x=tf.placeholders(tf.float32, shape=[None,784])
y_true = tf.placeholder(tf.float32, [None,10])

#variables
w = tf.Variable(tf.zeros([780,10]))
b = t.Variable(tf.zeros([10]))

#create graph operations
y = tf.matmul(x, w) + b

#loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_rntropy_with_logits(labels=y_true,logits=y))

#optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
train = optimizer.minimize(cross_entropy)

#create session
init = tf.global_variables_initializer()
#train data
with tf.Session() as sess:
	sess.run(init)
	for step in range(1000):
		batch_x, batch_y = mnist.train.next_batch(100)
		sess.run(train, feed_dict={x:batch_x,y_true:batch_y})

	#evaluate the model
	correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_true, 1))
	acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	print(sess.run(acc, feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))

