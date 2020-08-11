import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

doc = nlp = (u"The quik brown fox jumped over the lazy dog's back")
print(doc[4].pos_)#verb ----tag_=past tense veb
for token in doc:
	print(f"{token.text:{10}} {token.pos_:{10}} {spacy.explain(token.tag_)}")

#present -- read
doc1 = nlp(u"I read books on nlp")
word = doc1[1]
print(word.text)
token = word
for token in doc:
	print(f"{token.text:{10}} {token.pos_:{10}} {spacy.explain(token.tag_)}")
#past ---read
doc1 = nlp(u"i read a book on nlp")
word = doc1[1]
print(word.text)
token = word
for token in doc:
	print(f"{token.text:{10}} {token.pos_:{10}} {spacy.explain(token.tag_)}")

#pos_count
doc = nlp = (u"The quik brown fox jumped over the lazy dog's back")
pos_counts = doc.count_by(spacy.attrs.POS)
print(pos_counts)
doc.vocab[83].text

for k,v in sorted(pos_counts.items()):
	print(f"{k}. {doc.vocab[k].text:{5}} {v}")

#tag_counts
tag_counts = doc.count_by(spacy.attrs.TAG)
for k,v in sorted(tag_counts.items()):
	print(f"{k}. {doc.vocab[k].text:{5}} {v}")

#syntactic dependencies ount
dep_counts = doc.count_by(spacy.attrs.DEP)
for k,v in sorted(dep_counts.items()):
	print(f"{k}. {doc.vocab[k].text:{5}} {v}")

#displacy
doc = nlp = (u"The quik brown fox jumped over the lazy dog's back")
options = {'distance': 110,'compct':'True','color':'yellow','bg':'red','font':'Times'}
displacy.render(doc, style='dep',jupyter=True, options=options)

#FOR LARGEWORDS
doc2 = nlp(u"this is a sentence. this is another sentence, possibly longer than another sentence")
spans = list(doc2.sents)
displacy.serve(spans, style='dep',options={'distance':110})
#opens on browser

