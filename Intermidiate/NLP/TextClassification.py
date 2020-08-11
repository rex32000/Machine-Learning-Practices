import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrix, classification_report

df = pd.read_csv("../TextFiles/moviereviews.tsv", sep='\t')


blanks= []
for i, lb, rv in df.itertuples():
	if e=rv.isspace():
		blanks.append(i)
df.drop(blanks, iplace=True)

X = df['review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
text_clf = pipeline([('tfidf', TfidfVectorizer()),('clf', LinearSVC())])
text_clf.fit(X_train, y_train)

predictions = text_clf.predict(X_test)
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))