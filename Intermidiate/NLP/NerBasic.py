import spacy
from spacy.tokens import Span
from spacy.matcher import PhraseMatcher
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

def show_ents(doc):
	if doc.ents:
		for ent in doc.ents:
			print(ent.text + '-' + ent.label_ + '-' + str(spacy.explain(ent.label_)))
	else:
		print("No entities found")

doc = nlp(u'Hi how are you?')
show_ents(doc)#no entities found
doc = nlp(u'May i go to Washington DC next may to see the Washington Monment')
show_ents(doc)

doc = nlp(u"Can i please have 500dollars of Microsoft stocks")
show_ents(doc)

doc = nlp("Tesla to build U.K. factory for $6 million")
show_ents(doc)#tesla doesnt show up

org = doc.vocab.strings[u"ORG"]
new_ent = Span(doc,0,1,label=ORG)
doc.entslist(doc.ents) + [new_ent]
show_ents(doc)#tesla added as entity

#adding named entities --vaccum cleaner
doc = nlp(u"Our company created a brand new vaccum cleaner"
			u"This new vaccum-cleaner isthe best in show")
matcher = PhraseMatcher(nlp.vocab)
phrase_list = ['vaccum cleaner', 'vaccum-cleaner']
phrase_pattern = [nlp(text) for text in phrase_list]
matcher.add('newproduct',None,*phrase_pattern)
found_matches = matcher(doc)

PROD = doc.vocab.strings[u"PRODUCT"]
new_ents = [Span(doc, match[1],match[2],label=PROD) for match in found_matches]
doc.ents = list(doc.ents) + new_ents
show_ents(doc)

#counting ner
doc = nlp(u"Originally i paid $29.95 for this cay toy, but now it is marked to by $10")
count = len([ent for ent in doc.ents if ent.label_ == "MONEY"])

#displacy
doc = nlp(u"Over the last quarter Aplle sold nearly 20 thousand ipods for profit of $6 million"
	u"By contrast, Sony only sold 8 thousand Walkman music players.")
for sent in doc.sents:
	displacy.render(nlp(sent.text), style='ent', jupyter=True)
colors = {'ORG':'linear-gradient(90deg,#aa9cfc,fc9ce7)'}#simple colors --'red'
options= {'ents':['PRODUCT','ORG'],colors:'colors'}
displacy.render(doc, style='ent', jupyter=True,options=options)


