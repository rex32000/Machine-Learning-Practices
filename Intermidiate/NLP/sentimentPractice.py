import nltk
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

sid = SentimentIntensityAnalyzer
a="This is a good movie"
sid.polarity_scores(a)
a= "This was the best, most awesome moviee EVER MADE!!!"
sid.polarity_scores(a)
df = pd.read_csv('amazonreviews.tsv', sep='\t')

df['label'].value_counts()
df.dropna(inplace=True)
blanks= []
for i, lb, rv in df.itertuples():
	if type(rv) == str:
		if rv.isspace():
			blanks.append(i)
df.drop(blanks, iplace=True)

df.iloc[0]['review']
sid.polarity_scores(df.iloc[0]['reviews'])

df['scores'] = df['review'].apply(lambda review: sid.polarity_scores(review))
df['compound'] = df['scores'].apply(lambda d:d['compound'])
df['comp_score'] = df['compound'].apply(lambda score:'pos' if score>=0 else 'neg')

accuracy_score(df['label'], df[comp_score])
print(classification_report(df['label'], df[comp_score]))

