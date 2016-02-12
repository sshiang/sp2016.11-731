import nltk
import sys
import os
import pickle 
import pattern.en
import pattern.de
from nltk.stem.lancaster import LancasterStemmer
import re

class Example:
	def __init__(self, german=[], eng=[]):
		self.eng = eng
		self.german = german



stem_en = LancasterStemmer()
stem_ge = nltk.stem.snowball.GermanStemmer()

textFile = "data/dev-test-train.de-en" 
comFile = "data/compound-de-processed-slash.txt"
compounds = {}
count = 0
for line in open(comFile):
	count +=1
	seg = re.split("\t| ",line.strip())#.split("\t")
	#print line, seg

	seg = filter(None,seg)

	#print line, seg

	if len(seg) == 1:
		compounds[seg[0]] = [stem_ge.stem(seg[0].lower().decode("utf-8")).encode("utf-8")]
		continue

	if len(seg) == 2:
		compounds[seg[0]] = [stem_ge.stem(seg[0].lower().decode("utf-8")).encode("utf-8")]
		continue

	word = seg[0]
	coms = seg[1:]
	coms = filter(None,coms)

	while ";" in coms:
		coms.remove(";")

	ccc = [stem_ge.stem(x.lower().decode("utf-8")).encode("utf-8") for x in coms]
	
	'''
	if len(seg) == 2 and seg[1]==";":
		coms = [seg[0]]
	else:			
		coms = seg[1:]#.split(" ")
	#coms = filter(";",coms)
	'''
	compounds[word] =ccc# filter(None,coms)
	

examples = []
count = 0
outputFile = "data/parse-de-en-compound-nosemicolon.txt"
f = open(outputFile, "w")
for line in open(textFile):
	count+=1
	lans = line.strip().replace("/","&slash;").split("|||")
	german_com = []
	german = lans[0].strip().split(" ")	
	for word in german:
		german_com.append("/".join(compounds[word]))

	eng = lans[1].strip()

	eng = [stem_en.stem(x.lower().decode("utf-8")).encode("utf-8") for x in eng.split(" ")]

	f.write("%s ||| %s\n"%(" ".join(german_com)," ".join(eng)))
	
f.close()
