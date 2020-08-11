#flower classification
import numpy as np
from sklearn.datasets import load_iris
from keras.utils import to_catagorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScale
from kerasmodels import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report, accuracy_score

iris = load_iris()

X = iris.data
y =iris.target
y=to_catagorical(y)
X_train, X_test, y_tain, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
scaler_object = MinMaxScale()
scaler_object.fit(X_train)
scaled_x_train = scaler_object.transform(X_train)
scaled_x_test = scaler_object.transform(X_test)

model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(scaled_x_train, y_train, epochs=150, verbose=2)
predictions = model.predict_classes(scaled_x_test)
y_test.argmax(axis=1)

print(classification_report(y_test.argmax(axis=1), predictions))
model.save('iris_model.h5')
