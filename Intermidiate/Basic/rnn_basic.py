import numpy as numpy
import tensorflow as tf
import matplotlib.pyplot as plt

class TimeSeriesData():

	def __init__(self,num_points,xmin,xmax):
		self.xmin = xmin
		self.xmax = xmax
		self.num_points = num_points
		self.resolution = (xmax-xmin)/ num_points
		self.x_data = np.linspace(xmin,xmax,num_points)
		self.y_true = np.sin(self.x_data)

	def ret_true(self, x_series):
		return np.sin(x_series)

	def next_batch(self, batch_size, steps,return_batch_ts=False):
		#grab a random starting point for eas\ch batch
		rand_start = np.random.rand(batch_size, 1)
		#convert to be on time series
		ts_start = rand_start*(self.xmax-self.xmin-(steps*self.resolution))
		#create batch timr series on x axis
		batch_ts = ts_start + np.arrange(0.0, steps+1)*self.resolution
		#create the y data for the time series x axisfrom previous step
		y_batch = np.sin(batch_ts)
		#formatting rnn
		if return_batch_ts:
			return y_batch[;,:-1].reshape(-1,steps,1), y_batch[:,1:].reshape(-1,steps,1), batch_ts
		else:
			return y_batch[;,:-1].reshape(-1,steps,1), y_batch[:,1:].reshape(-1,steps,1)

ts_data =TimeSeriesData(250,0,10)
plt.plot(ts_data.x_data, ts_data.y_true)

num_time_steps = 30
y1,y2,ts = ts_data.next_batch(1,num_time_steps,True)
#just for visualizing data
plt.plot(ts.flatten()[1:n], y2.flatten(),'*')

plt.plot(ts_data.x_data, ts_data.y_true,label='Sint(t)')
plt.plot(ts.flatten()[1:n], y2.flatten(),'*',label="single training instance")
plt.legend()
plt.tight_layout()

#Training data--Task--
train_inst = np.linspace(5, 5+ts_data.resolution*(num_time_steps+1),num_time_steps+1)
plt.title('A Training Instance')
#for given this data
plt.plot(train_inst[:-1],ts_data.ret_true(train_inst[:1]),'bo',marksize=15,alpha=0.5,label='INSTANCE')
#predicting this data shifted over one timestamp
plt.plot(train_inst[1:],ts_data(train_inst[1:]),'ko',marksize=7,label='TARGET')

#Creating Model
tf.reset_default_graph()
num_input = 1#features
num_neuron = 100
num_outputs = 1
learning_rate = 0.0001
num_train_iteration = 2000
batch_size = 1

#placeholders
X = tf.placeholders(tf.float32,[None,num_time_steps,num_input])
Y = tf.placeholders(tf.float32,[None,num_time_steps,num_outputs])

#RNN cell layer
cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(num_units=num_neuron, activation=tf.nn.relu),output_size=num_outputs)
outputs, states = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)

#loss function MSE
loss = tf.reduce_mean(tf.square(outputs-y))

#optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()

#session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
saver = tf.train.Saver()
with tf.Session(config = tf.ConfigProto(gpu_options=gpu_options)) as sess:
	sess.run(init)
	for iteration in range(num_train_iteration):
		x_batch, y_batch = ts_data.next_batch(batch_size,num_time_steps)
		sess.run(train, feed_dict = {X:x_batch,Y:y_batch})
		if iteratin % 100 == 0:
			mse = loss.eval(feed_dict={X:x_batch,Y:y_batch})
			print(iteration,"\tMSE",mse)
			saver.save(sess,"./rnn_basic")

#attempt predection 
with tf.Session() as sess:
	saver.restore(sess, "./rnn_basic")
	X_new = np.sin(np.array(train_inst[:-1].reshape(-1,num_time_steps,num_inputs)))
	Y_pred = sess.run(outputs, feed_dict={X:X_new})

plt.title("Testing Model")
#Training instance
plt.plot(train_inst[:-1,np.sin(train_inst[:-1]),"bo",marksize=15,alpha=0.5,label='Training Instance'])
#Target to predict
plt.plot(train_inst[1:],np.sin(train_inst[1:]),"ko",marksize=10,label='Target')
#Models prediction
plt.plot(train_inst[1:], y_pred[0,:,0],'r.',marksize=10,label='predection')
plt.xlabel('Time')
plt.legend()
plt.tight_layout()

#generating sequece
with tf.Session as sess:
	saver.restore(sess,"./rnn_basic")

	#seed Zeros
	zero_seq_seed = [0.0 for i in range(num_time_steps)]
	for iteration in range(len(ts_data.x_data)-num_time_steps):
		x_batch = np.array(zero_seq_seed[-num_time_steps:]).reshape(1,num_time_steps,1)
		y_pred = sess.run(outputs, feed_dict={X:x_batch})
		zero_seq_seed.append(y_pred[0,-1,0])

plt.plot(ts_data.x_data,zero_seq_seed,'b-')
plt.plot(ts_data.x_data[:num_time_steps],zero_seq_seed[:num_time_steps],'r',linewidth=3)
plt.xlabel('Time')
plt.ylabel('Y')

with tf.Session as sess:
	saver.restore(sess,"./rnn_basic")

	#seed Zeros
	training_instance = list(ts_data.y_true[:30])
	for iteration in range(len(training_instance)-num_time_steps):
		x_batch = np.array(training_instance[-num_time_steps:]).reshape(1,num_time_steps,1)
		y_pred = sess.run(outputs, feed_dict={X:x_batch})
		training_instance.append(y_pred[0,-1,0])

plt.plot(ts_data.x_data,ts_data.y_true,'b-')
plt.plot(ts_data.x_data[:num_time_steps],training_instance[:num_time_steps],'r',linewidth=3)
plt.xlabel('Time')
plt.ylabel('Y')
