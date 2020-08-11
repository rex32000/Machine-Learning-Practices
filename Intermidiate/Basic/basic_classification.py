import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

diabetes = pd.read_csv('pima-indians-diabetes.csv')
#normalize data
#for col name = diabetes.columns
cols_to_norm=['Number_pregnant', 'Glucose_concentration', 'Bloodpressure', 'Triceps','Insulin', 'BMI', 'Pedigree']
diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x:((x-x.min())/x.max()-x.min()))

#diabetes.columns
#create feature col and numeric col -- new varable for each col
#continous val
num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Bloodpressure')
triceps = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabtes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')
#non-continous/catagorical features--Twomain methods
assigned_group = tf.feature_column.catagorical_column_with_vocabulary_list('Goup', ['A', 'B', 'C', 'D'])

#assigned_group = tf.feature_column.catagorical_column_with_hash_bucket('Group', has_bucket_size=10)#atmost is 10 --auto method

#just to visualize
diabetes['Age'].hist(bins=20)
#continuos to bucket
age_bucket = tf.feature_column.bucketized_column(age, boundaries=[20,30,40,50,60,70,80])
feat_cols = [num_preg,plasma_gluc,dias_press,triceps, insulin,bmi,diabtes_pedigree,assigned_group,age_bucket]

#train test split
x_data= diabetes.drop('Class',axis=1)#excluding labels
labels = diabetes['Class']

x_train, x_test,y_train, y_test = train_test_split(x_data, labels, test_size = 0.30, random_state=101)
input_func = tf.estimator.inputs.pandas_input_fn(x = x_train, y= y_train, batch_size = 10,num_epochs=1000, shuffle=True)
model = tf.estimator.LinearClassifier(feature_column=feat_cols, n_classes=2)
model.train(input_fn=input_func, step=1000)
eval_input_func = tf.estimator.inputs.pandas_input_fn(x = x_test, y= y_test, batch_size = 10,num_epochs=1, shuffle=True)
results = model.evaluate(eval_input_func)
#75% accurate
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test, batch_size=10, num_epochs = 1, shuffle=False)
predictions = model.predict(pred_input_func)
my_pred = list(predictions)

#for dnn categorical/feature col to embeded col is req
dnn_model = tf.estimator.DNNClassifier(hidden_units = [10, 10, 10], feature_column=feat_cols, n_classes=2)
embedded_group_col = tf.feature_column.embedding_column(assigned_group, dimension=4)
#75% accurate

feat_cols = [num_preg,plasma_gluc,dias_press,triceps, insulin,bmi,diabtes_pedigree,embedded_group_col,age_bucket]
input_func = tf.estimator.inputs.pandas_input_fn(x_train, y_train, batch_size=10,num_epochs=1000, shuffle=True)
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,20,20,10,10], feature_column=feat_cols, n_classes=2)
dnn_model.train(input_fn=input_func,steps=1000)
eval_input_func= tf.estimatorinputs.pandas_input_fn(x=x_test,y=y_test, batch_size=1,num_epochs=1, shuffle=False)
dnn_model.evaluate(eval_input_func)
