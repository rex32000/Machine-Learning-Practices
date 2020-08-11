import spacy
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layes import Dense, LSTM, Embedding
from pickle import dump, load
from keras.preprocessing.sequence import pad_sequences
import random

def read_file(filepath):
	with open(filepath) as f:
		str_text = f.read()

	return str_text

nlp= spacy.load('en', disable=['parser','tagger','ner'])

nlp.max_length = 1198623

def separate_punc(doc_text):
	return [token.text.lower() for token in nlp(doc_text) if token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n']

d= read_file('moby_dick_four_chapters.txt')
tokens = separate_punc(d)

#25 words -> network predict #26 word
train_len = 25+1
text_sequences = []
for i in range(train_len, len(tokens)):
	seq = tokens[i-train_len:i]
	text_sequencces.append(seq)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
sequences =tokenizer.texts_to_sequences(text_sequences)
vocabulary_size = len(tokenizer.word_counts)
sequences=np.array(sequences)

X = sequences[:,:-1]
y = sequences[:,-1]
y = to_categorical(y, num_classes= vocabulary_size+1)
seq_len = X.shape[1]

def create_model(vocabulary_size, seq_len):
	model = Sequential()
	model.add(Embedding(vocabulary_size,seq_len,input_length=seq_len))
	model.add(LSTM(50, return_sequences=True))
	model.add(LSTM(50))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(vocabulary_size,activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer ='adam', metrics=['accuracy'])
	model.summary()
	return model

create_model(vocabulary_size+1, seq_len)
model.fit(X, y, batch_size=128, epochs=100, verbose=1)
model.save('mobyDick.h5')
dump(tokenizer,open('my_simpletokenizer','wb'))

def generate_text(model, tokenizer, seq_len, seed_text, num_gen_words):
	output_text = []
	input_text = seed_text
	for i in range(num_gen_words):
		encoded_text = tokenizer.texts_to_sequencies([input_text])[0]
		pad_encoded = pad_sequences([encoded_text], maxlength=seq_len, truncating = 'pre')
		pred_word_ind = model.predict_classes(pad_encoded, verbose=0)[0]
		pre_word = tokenizer.index_word[pred_word_ind]
		output_text.append(pred_word)

	return ' '.join(output_text)

random.seed(101)
random_pick = random.randint(0, (text_sequences))
random_seed_text = text_sequences[random_pick]

generate_text(model, tokenizer, seq_len, seed_text=seed_text, num_gen_words=25)

