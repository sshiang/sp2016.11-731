import os
import sys
import math 
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import nltk
from nltk.stem.lancaster import LancasterStemmer
import random 
import copy
def readFile(textFile, reverse):
	aligns = []
	for line in open(textFile):
		seg = line.strip().split(" ")
		#align = {}
		align = []
		for al in seg:
			if "-" in al:
				x = al.split("-")
			if "?" in al:
				x = al.split("?")
				continue
			if reverse == True:
				ge = int(x[1])
				en = int(x[0])
			else:
				ge = int(x[0])
				en = int(x[1])

			align.append((ge,en))

			'''
			if ge not in align:
				align[ge] = [en]
			else:
				align[ge].append(en)
			'''
		aligns.append(align)
	return aligns

def intersect(list1,list2):
	inter = [x for x in list1 if x in list2]
	nointer1 = [x for x in list1 if x not in inter]
	nointer2 = [x for x in list2 if x not in inter]	
	return inter, nointer1+nointer2

def dumpAlign(f,aligns):
	for al in aligns:
		f.write("%d-%d "%(al[0],al[1]))
	f.write("\n")

def getE(g,align,length):
	for x in range(5):# True:
		e = random.randint(0, length-1)
		if (g,e) not in align:
			return e
	e = -1
	return e	

def extractExample(aligns, negative):
	global eng_parse
	examples = []
	labels = []
	# positive
	for i in range(len(aligns)):
		print i
		align = aligns[i]
		label = [] #[1 for x in range(len(align))]
		feature = []
		for al in align:
			fea = getFeature(i,al)
			feature.append(fea)
			label.append(1)

		#if negative == True:
		for al in align:
			# add negative example
			g = al[0]
			e = getE(g,align,len(eng_parse[i]))
			if e == -1:
				continue
			#print "ya"
			fea = getFeature(i,(g,e))
			label.append(0)
			feature.append(fea)

		labels+=label
		examples+=feature

	#print labels
			
	return examples, labels

def getFeature(index, al):
	global eng_compound
	global german_compound
	global eng_parse
	global german_parse
	feature = []
	feature += getPos(index,al)
	feature += getLength(index,al)
	feature += getIBM(index,al)
	feature += getPosition(index,al)
	feature += getMatch(index,al)
	#feature += getPrevious()

	return feature

def getPos(index, al):
	global eng_parse
	global german_parse
	global lexicon_eng
	global lexicon_ge

	feature = [0 for x in range(len(lexicon_eng)*len(lexicon_ge))]
	print al
	print eng_parse[index], german_parse[index]
	print eng_parse[index][al[1]], german_parse[index][al[0]]

	pos_eng = eng_parse[index][al[1]][1]
	pos_ge = german_parse[index][al[0]][1]

	feature[lexicon_eng[pos_eng]*len(lexicon_ge)+lexicon_ge[pos_ge]]=1

	return feature

def getLength(index, al):
	global eng_compound
	global german_compound
	length_eng = len(eng_compound[index][al[1]][0])
	abs_length = length_eng
	for word in german_compound[index][al[0]]:
		if math.fabs(len(word)-length_eng) < abs_length:
			abs_length = math.fabs(len(word)-length_eng)
	return [abs_length]

def getPosition(index, al):
	global eng_compound
	global german_compound
	p_eng = float(al[1])/float(len(eng_compound[index]))
	p_ge = float(al[0])/float(len(german_compound[index]))
	return [math.fabs(p_eng-p_ge)]

#def getPrevious(index, al_pre, al):

def getIBM(index, al):
	global p_ef
	global eng_parse
	global german_parse
	global stem_en
	global stem_ge
	word_en = stem_en.stem(eng_parse[index][al[1]][0].lower().decode("utf-8")).encode("utf-8")
	word_ge = stem_ge.stem(german_parse[index][al[0]][0].lower().decode("utf-8")).encode("utf-8")
	score = 0
	if word_ge in p_ef:
		if word_en in p_ef[word_ge]:
			score = p_ef[word_ge][word_en]

	#print score	
	return [score]

def getMatch(index,al):
	global eng_parse
	global german_parse
	if eng_parse[index][al[1]][0].lower() == german_parse[index][al[0]][0].lower():
		return [1]
	else:
		return [0]

def readText(textFile, parse):
	engs = []
	germans = []
	lexicon_eng = {}
	lexicon_ge = {}
	count_eng = 0
	count_ge = 0
	for line in open(textFile):
		seg = line.strip().split("|||")
		eng= [x.split("/") for x in seg[1].strip().split(" ")]
		ger = [x.split("/") for x in seg[0].strip().split(" ")]
		engs.append(eng)
		germans.append(ger)

		if parse == False:
			continue

		for x in eng:
			#print x
			if x[1] not in lexicon_eng:
				lexicon_eng[x[1]] = count_eng
				count_eng+=1
		for x in ger:
			if x[1] not in lexicon_ge:
				lexicon_ge[x[1]] = count_ge
				count_ge+=1

	return engs, germans, lexicon_eng, lexicon_ge


stem_en = LancasterStemmer()
stem_ge = nltk.stem.snowball.GermanStemmer()

parseFile = "data/parse-de-en.txt"
textFile = "data/parse-de-en-compound-nosemicolon.txt"

eng_compound, german_compound, garbage, garbage = readText(textFile, False)
eng_parse, german_parse, lexicon_eng, lexicon_ge = readText(parseFile, True)

pkl_file = open('p_ef.pkl', 'rb')
p_ef = pickle.load(pkl_file)
pkl_file.close()


forwardFile = "HMM_compound_forward.txt"
backwardFile = "HMM_compound_back.txt"
oracleFile = "data/dev.align"
forward = readFile(forwardFile,False)
backward = readFile(backwardFile,True)	
oracle = readFile(oracleFile,False)

assert (len(forward)==len(backward)), "fb length mismatch"


example_oracle, label_oracle = extractExample(oracle,True)
#example_forward, garbage = extractExample(forward,False)
#example_backward, garbage = extractExample(backward,False)
#features_oracle = extractFeatures(example_oracle)
#features_forward = extractFeatures(example_forward)
#features_backward = extractFeatures(example_backward)

print "complete feature..."

clf = LogisticRegression()
clf.fit(example_oracle, label_oracle)
#predict_forward = clf.predict(example_forward)
#predict_backward = clf.predict(example_backward)

print "complete model..."

f = open("output.txt","w")
for i in range(300):
	print i
	inter, nointer = intersect(forward[i],backward[i])
	newinter = copy.deepcopy(inter)
	for pair in nointer:
		print pair
		print len(eng_parse[i]), len(german_parse[i])
		if len(eng_parse[i])<=pair[1] or len(german_parse[i])<=pair[0]:
			continue
		fea = getFeature(i,pair)
		result = clf.predict([fea])
		#print result
		if result == [1]:
			# right label
			newinter.append(pair)
	dumpAlign(f,list(set(newinter)))
f.close()
