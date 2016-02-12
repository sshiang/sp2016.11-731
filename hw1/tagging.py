import nltk
import sys
import os
import pickle 
import pattern.en
import pattern.de
from nltk.stem.lancaster import LancasterStemmer

class Example:
	def __init__(self, german=[], eng=[]):
		self.eng = eng
		self.german = german

textFile = "data/dev-test-train.de-en" 

examples = []
count = 0
lexicon_en = {}
lexicon_ge = {}
stem_en = LancasterStemmer()
stem_ge = nltk.stem.snowball.GermanStemmer()
outputFile = "data/parse-de-en.txt"
f = open(outputFile, "w")
for line in open(textFile):
	count+=1
	lans = line.strip().split("|||")
	#german = [stem_ge.stem(x.encode("utf-8")) for x in lans[0].strip().split(" ")]
	german = lans[0].strip().split(" ")	
	parse_ge = pattern.de.parse(" ".join([x.decode("utf-8") for x in german]),tokenize=False,lemmata=True).replace("\n"," ")
	assert (len(german)==len(parse_ge.split(" "))), "length de mismatch"

	#eng = [stem_en.stem(x) for x in lans[1].strip().split(" ")]
	eng = lans[1].strip().split(" ")
	parse_en = pattern.en.parse(" ".join([x.decode("utf-8") for x in eng]),tokenize=False,lemmata=True).replace("\n"," ")
	assert (len(eng)==len(parse_en.split(" "))), "length en mismatch"	

	f.write("%s ||| %s\n"%(parse_ge.encode("utf-8"),parse_en.encode("utf-8")))
	
f.close()
