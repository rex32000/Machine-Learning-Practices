import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neigbhors import KNeighborClassifier
from sklearn.svm import SVC



url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']
dataset = pandas.read_csv(url, names=names)

#shape
print(dataset.shape)

#head
print(dataset.head(20))

#describe
print(dataset.describe())

#distribution(class)
print(dataset.groupby('class'))

#histograms
dataset.hist()
plt.show()

#scatter-plot matrix
scatter_matrix(dataset)

#split-out validation dataset
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = seed)


#Test option and evaluation metric
scoring = 'accuracy'

models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborClassifier()))
models.append(('SVM',SVC()))

#evaluate eachmodel in turn
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits = 10,random_state = seed)
    cv_results = model_selection.cross_val_score(model, X_train,cv = kfold, scoring = scoring)
    results.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

#Make prediction onvalidation dataset

for name,model in models:
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    print(name)
    print(accuracy_score(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
    