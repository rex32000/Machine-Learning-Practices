import spacy
from spacy.pipeline import SentenceSegmenter
nlp = spacy.load('en_core_web_sm')

doc = nlp(u'This is the first sentence. This is another sentence. This is the last sentence')
for sent in doc.sents:#doc.sents--generator object--cannot directly indexed--span object not string
	print(sent)

doc = nlp(u'"Management is doing right things; leadersjip is doing the right thing."-Peter Drucker')
#adding a segmentation rule
def set_custom_boundaries(doc):
	for token in doc[:-1]:#excluding last word
		if token.text == ';':
			doc[token.i+1].is_sent_start = True
	return doc
nlp.add_pipe(set_custom_boundaries, before='parser')
nlp.pipe_names#[tagger, set_custom_coundaries,parser,ner]
doc4 = nlp(u'"Management is doing right things; leadersjip is doing the right thing."-Peter Drucker')
for sent in doc4.sents:
	print(sent)

#change segmentation rules
mystring = u"This is a sentence. This is another.\n\n This is a \nthird sentence."
doc = nlp(mystring)
for sent in doc.sents:
	print(sent)#doesnt want new line at period but at \n

def split_on_newlines(doc):
	start = 0
	seen_newline = False

	for word in doc:
		if seen_newline:
			yield doc[stat:word.i]
			start = word.i
			seen_newline = False
		elif word.text.startwith('\n'):
			seen_newline= True
	yield doc[start:]

sbd = SentenceSegmenter(nlp.vocab, strategy = split_on_newlines)
nlp.add_pipe(sbd)
doc = nlp(mystring)

for sent in doc.sents:
print(sent)
			



