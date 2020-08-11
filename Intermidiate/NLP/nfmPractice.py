import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

npr = pd.read_csv('mpr.csv')
tfidf = TfidfVectorizer(max_df=0.95, min_df=2,stop_words='english')
dtm = tfidf.fit_transform(npr['Article'])
nmf_model = NMF(n_components=7, random_state=42)
nfm_model.fit(dtm)
for index, topic in enumerate(nmf_model.components_):
	print(f"THE TOP !% TOPICS #{index}")
	print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])

npr['Topic'] = topic_results.argmax(axis=1)
