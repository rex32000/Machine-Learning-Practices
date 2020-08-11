import numoy as np
import pandas
import matplotlib.pyplot as pyplot
import tensorflow as tensorflow

x_data = np.linspace(0.0, 10, 100000)
noise = np.random.randn(len(x_data))
#y = mx+b
#b = 5
y_true = (0.5 * x_data) + 5 + noise
X_df = pd.DataFrame(data=x_data, columns = ['X Data'])
y_df = pd.DataFrame(data = y_true, columns = ['Y'])

my_data = pd.concat([x_df, y_df], axis = 1)

my_data.sample(n=250).plot(kind='scatter', x='X Data', y ='Y')

batch_size = 8
m = tf.Variable(0.81)#random numbers
b = tf.Variable(0.17)
xph = tf.placeholder(tf.float32, [batch_size])
yph = tf.placeholder(tf.float32, [batch_size])

#operation /graph 
y_model = m * xph + b
error = tf.reduce_sum(tf.square(yph-ymodel))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()
with tf.Session as sess:
	sess.run(init)

	batches = 1000
	for i in range(batches):
		rand_ind = np.random.randint(len(x_data), size = batch_size)
		feed = {xph:x_data[rand_ind],yph:y_true[rand_ind]}
		sess.run(train, feed_dict = feed)
	
	model_m, model_b = sess.run([m,b])

feature_cols = [tf.feature_column.numeric_column('X'. shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_column=feature_column)

from sklearn.model_selection import train_test_split
x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_true,test_size=0.3,random_state=101)

#estimator input
input_func = tf.estimator.inputs.numpy_input_fn({'X':x_train}, y_train, batch_size=8, num_epocs= None, shuffle=True)
train_input_func = tf.estimator.inputs.numpy_input_fn({'X':x_train}, y_train, batch_size=8, num_epocs= 1000, shuffle=False)
eval_input_func = tf.estimator.inputs.numpy_input_fn({'X':x_eval}, y_eval, batch_size=8, num_epocs= 1000, shuffle=False)

estimator.train(input_fn=input_func, steps=1000)

train_matrics = estimator.evaluate(input_fn = train_input_func, steps = 1000)
eval_metrics = estimator.evaluate(input_fn=eval_input_func, steps=1000)
print('Training data metrics')
print(train_metrics)
print("eval matrics")
print(eval_metrics)
#predict
brand_new_data = np.linspace(0,10,10)
input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x':brand_new_data},shuffle=False)
list(estimator.predict(input_fn=input_fn_predict))
predictions = []
for pred in estimator.predict(input_fn=input_fn_predict):
	predictions.append(pred['predictions'])

my_data.sample(n=250).plot(kind='scatter', x='X Data', y=Y)
plt.plot(brand_new_data, predictions, 'r')

