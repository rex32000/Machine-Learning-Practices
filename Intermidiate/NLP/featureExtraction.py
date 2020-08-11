import numpy as np
import pandas as pd
from sklearn.mmodel_selection import train_test_split
from sklearn.feature_extraction import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_metrix, classification_report
df = pd.read_csv('../TextFiles/smsspamcollection.tsv', sep='\t')
X = df['message']
y = df['label']
X_train, X_test,  y_train, y_test = train_test_split(X,y,test_size=0.33,random_stat42) 
count_vect = CountVectorizer()
#fit the vectorizer to data--build vocab and count no. of words
# count_vect.fit(X_train)
# X_train_count = count_vect.transform(X_train)
#Transform the originaltext msg to vector
# X_train_count = count_vect.fit_transform(X_train)
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)
#combines above steps
# vectorizer = TfidfVectorizer()
# X_train_tfidf= vectorizer.fit_transform(X_train)
# clf = LinearSVC()
# clf.fit(X_train_tfidf,y_train)
# combning all the steps above

text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC())])
text_clf.fit(X_train,y_train)

predictions = text_clf.predict(X_test)
print(confusion_metrix(y_test, predictions))
print(classification_report(y_test, predictions))
