import pickle
import numpy as np
from keras.models import Sequential,Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import Tokenizer
from keras.layers.embeddings import Embedding
from keras.layes import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, LSTM
import matplotlib.pyplot as plt

#list of tuples of list
with open('train_qa.txt','rb') as f:
	train_data = pickle.load(f)

with open('test_qa.txt','rb') as f:
	test_data = pickle.load(f)

all_data = test_data + train_data

#CREATING SET ----> a set is unorderd list of unique data
vocab = set()
for story,question,answer in all_data:
	vocab = vocab.union(set(story))
	vocab = vocab.union(set(question))
	vocab.add('no')
	vocab.add('yes')

vocab_len = len(vocab) + 1

#LONGEST STORY AND QUES
all_story_lens = [len(data[0]) for data in all_data]
max_story_len = max(all_story_lens)
max_ques_len = max([len(data[1]) for data in all_data])
tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_text(vocab)
#tokenizer.word_index
train_story_text = []
train_question_text = []
train_answer = []

for story,question,answer in train_data:
	train_story_text.append(story)
	train_question_text.append(question)
	train_answer.append(answer)

#converting all words to their respective index position in vocab
train_story_seq = tokenizer.text_to_sequences(train_story_text)

def vectorize_stories(data, word_index==tokenizer.word_index, max_story_len=max_story_len, max_ques_len=max_ques_len):
	#stories
	X = []
	#questions
	Xq = []
	#Y is answer
	Y = []
	for every story,question,answer in data:
		#for each story[23,14,...]
		x = [word_index[word.lower()] for word in story]
		xq = [word_index[word.lower()] for word in question]
		y = np.zeros(len(word_index)+1)
		y[word_index[answer]] = 1

		X.append(x)
		Xq.append(xq)
		Y.append(y)
	return (pad_sequences(X,maxlen=max_story_len),pad_sequences(Xq,maxlen=max_ques_len),np.array(Y))

input_train, queries_train, answer_train = vectorize_stories(train_data)
input_test, queries_test, answer_test = vectorize_stories(test_data)

#placeholders
input_seq = Input((max_story_len,))
question = Input((max_ques_len,))
#....define vocab size
vocab_size = len(vocab) + 1
#input encoders 
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim = vocab_sizee, output_dim = 64))
input_encoder_m.add(Dropout(0.3))

input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim = vocab_sizee, output_dim = max_ques_len))
input_encoder_c.add(Dropout(0.3))

question_encoder = Sequential()
question_encoder.add(Embedding(input_dim = vocab_sizee, output_dim = 64,input_length=max_ques_len))
question_encoder.add(Dropout(0.3))

#ENCODED --> ENCODER(INPUT)
input_encoded_m = input_encoder_m(input_seq)
input_encoded_c = input_encoded_c(input_seq)
question_encoded = question_encoder(question)
	
match = dot([input_encoded_m, question_encoded], axes=(2,2))
match = Activation('softmax')(match)

response = add([match,input_encoded_c])
response = Permute((2,1))(response)

answer = concatenate([respose, question_encoded])

answer =  LSTM(32)(answer)
answer = Dropout(0.5)(answer)
answer = Dense(vocab_size)(answer)
answer = Activation('softmax')(answer)

model = Model([input_seq, question], answer)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit([input_train,queries_train], answer_train, batch_size=32,epochs=100, validation_data=([input_test,queries_test], answer))

#summarize history for accuracy

plt.plot(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(histoy.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

filename = 'chatbotQA.h5'
model.save(filename)

model.load_weights('chatbotQA.h5')
pred_results = model.predicts(([input_test,queries_test]))
val_max = np.argmax(pred_results[0])
for key, val in tokenizer.word_index.items():
	if val==val_max:
		k=key
print(k)
pred_results[0][val_max]

#custom story based on vocab 
my_story = "John left the kitchen .  Sandra dropped the football in the gardern ."
my_ques = "Is the football in the gardn ?"
mydata = [(my_story.split(),my_ques.split(),'yes')]

my_story, my_ques, my_ans = vectorize_stories(mydata)
pred_results = model.predicts(([my_story,my_ques]))
val_max = np.argmax(pred_results[0])
for key, val in tokenizer.word_index.items():
	if val==val_max:
		k=key
print(k)
