import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import random

npr = pd.read_csv('npr.csv')
cv = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
dtm = cv.fit_transform(['Article'])
LDA = LatentDirichletAllocation(n_components=7, random_state=42)
LDA.fit(dtm)

# grab vocab of words
cv.get_feature_names()#every single word
random_word_id = random.randint(0, 54777)
cv.get_feature_names()[random_word_id]
# grab the topics
LDA.components_#numpy array containing prob for each word
single_topic = LDA.components_[0]
single_topic.argsort()


# grab highest probablity words per topic
for i, topic in enumerate(LDA.components_):
	print(f"THE top 15 words for topic #{i}")
	print([cv.get_feature_names()[index] for index in topc.argsort()[-15:]])
	print('\n')
	print('\n')

topic_results = LDA.transform(dtm)
npr['Topic'] = topic_results.argmax(axis=1)
npr