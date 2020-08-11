import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

df = pd.read_csv('../TextFiles/smsspamcollection.tsv', sep='\t')
df.head()
df.isnull().sum()#check for missing ddata
len(df)#num of rows

df['label'].unique()
df['label'].value_counts()

#X feature data
X = df['length','punct']
#y is our label
y = df['label']

x_train, x_test, y_tain, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
lr_model = LogisticRegression(solver='lbfgs')
lr_model.fit(x_train,y_tain)

prediction = lr_model.predict(x_test)

print(metrics.confusion_matrix(y_test, predictions))
print(metrics.classifications_report(y_test, predictions))
