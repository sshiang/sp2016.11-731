import sys
import os
import math
import nltk
from nltk.stem.lancaster import LancasterStemmer
import copy
import pickle
import operator
import pattern.en
import pattern.de

class Example:
	def __init__(self, german=[], eng=[]):
		self.eng = eng
		self.german = german

def readAlignment(alignFile):
	global examples
	probMap = {}
	count = 0
	for line in open(alignFile):
		seg = line.strip().split(" ")
		for al in seg:
			if "-" in al:
				a = al.strip().split("-")
			elif "?" in al:
				a = al.strip().split("?")
			else:
				print "error"
			de_word = examples[count].german[int(a[0])]
			en_word = examples[count].eng[int(a[1])]
			if de_word not in probMap:
				probMap[de_word] = {en_word:1}
			else:
				if en_word not in probMap[de_word]:
					probMap[de_word][en_word] = 1
				else:
					probMap[de_word][en_word]+=1	
		count+=1

	for word in probMap:
		total = float(sum([probMap[word][x] for x in probMap[word]]))
		for word2 in probMap[word]:
			probMap[word][word2] = probMap[word][word2]/total
	return probMap

def process(textList):
	global comment
	new_textList = []
	for text in textList:
		seg = text.lower().split("/")
		if comment == "pos":
			new_textList.append(seg[1])
		elif comment == "lemma":
			new_textList.append(seg[-1])
		elif comment == "word":
			new_textList.append(seg[0])
		elif comment == "word_pos":
			new_textList.append("%s/%s"%(seg[0],seg[1]))
		elif comment == "lemma_pos":
			new_textList.append("%s/%s"%(seg[1],seg[-1]))
		else:
			new_textList.append(text)
	return new_textList

def readText(textFile):			
	examples = []
	count = 0
	lexicon_en = {}
	lexicon_ge = {}
	stem_en = LancasterStemmer()
	stem_ge = nltk.stem.snowball.GermanStemmer()
	for line in open(textFile):
		count+=1
		if count % 1000 == 0:
			print count
		lans = line.lower().strip().split("|||")
		#german = [stem_ge.stem(x.decode('utf-8')) for x in lans[0].strip().split(" ")]
		german = lans[0].strip().split(" ")
		german = process(german)
		for word in german:
			if word not in lexicon_ge:
				lexicon_ge[word]=1
			else:
				lexicon_ge[word]+=1
		eng = [stem_en.stem(x.decode('utf-8')) for x in lans[1].strip().split(" ")]
		#parse_en = pattern.en.parse(" ".join(eng))
		eng = lans[1].strip().split(" ")
		enfg = process(eng)
		for word in eng:
			if word not in lexicon_en:
				lexicon_en[word]=1
			else:
				lexicon_en[word]+=1
		examples.append(Example(german,eng))
	return examples, lexicon_en, lexicon_ge

def square(probs, pre_probs):
	if probs == None or pre_probs==None:
		return 1
	square = 0
	cc = 0
	for x in probs:	
		for y in probs[x]:
			cc +=1
			square += (probs[x][y] - pre_probs[x][y])**2
	return square/float(cc)
	


alignFile = "data/dev.align"
#textFile = "data/dev-test-train.de-en"
textFile = "data/parse-de-en.txt"
#textFile = "data/parse-de-en-small.txt"

comment = ""

if len(sys.argv)>=2:
	comment = sys.argv[1]
	pickleFile = "probs_%s.pkl"%sys.argv[1]
	outputFile = "result_%s.txt"%(sys.argv[1])
else:
	pickleFile = "prob.pkl"
	outputFile = "result.txt"


examples, lexicon_en, lexicon_ge = readText(textFile)
print len(lexicon_en)
print len(lexicon_ge)
#probMap = readAlignment(alignFile)

normal_value = 1.0/float(len(lexicon_en))


if os.path.exists(pickleFile):
	pkl_file = open(pickleFile, 'rb')
	probs = pickle.load(pkl_file)
	pkl_file.close()
else:
	pre_probs = None
	probs = None
	first = True
	iteration = 0
	#while (pre_probs!=probs or first==True):
	while (square(probs, pre_probs)>=10**-10):
		iteration +=1
		print iteration
		if iteration >= 100:
			break
		first = False
		total_f = {word_ge:0 for word_ge in lexicon_ge}
		count_ef = {}

		pre_probs = copy.deepcopy(probs)

		for i in range(len(examples)):# in examples:
			if i%10000==0:
				print "\t",i
			example = examples[i]
			stotal_en = {word_en:0 for word_en in example.eng}
			for word_en in example.eng:
				for word_ge in example.german:
					if probs == None:
						stotal_en[word_en]+=normal_value
					else:
						stotal_en[word_en]+=pre_probs[word_ge][word_en]	

			for word_en in example.eng:
				for word_ge in example.german:
					if probs == None:
						value = float(normal_value) / float(stotal_en[word_en])
					else:
						value = float(pre_probs[word_ge][word_en]) / float(stotal_en[word_en])
					total_f[word_ge] += value 
					if word_ge not in count_ef:
						count_ef[word_ge] = {word_en:value}
					else:
						if word_en not in count_ef[word_ge]:
							count_ef[word_ge][word_en] = value
						else:
							count_ef[word_ge][word_en] += value

		for word_ge in lexicon_ge:
			for word_en in lexicon_en:
				if word_en not in count_ef[word_ge]:
					continue
				value = float(count_ef[word_ge][word_en]) / float(total_f[word_ge])
				if probs == None:
					probs = {}
				if word_ge not in probs:
					probs[word_ge] = {word_en:value}
				else:
					probs[word_ge][word_en] = value 

		if iteration%5==0:
			iter_pickleFile = pickleFile.replace(".pkl","_%d.pkl"%(iteration))
			output = open(iter_pickleFile, 'wb')
			pickle.dump(probs, output)
			output.close()

	output = open(pickleFile, 'wb')
	pickle.dump(probs, output)
	output.close()


puncs = ["?",".",'"',"-","!","'","(",")","]","[","{","}"]
for p in puncs:
	if p in probs:
		probs[p][p] = 1


f = open(outputFile,"w")
for i in range(min(300,len(examples))):
	example = examples[i]
	for index_en in range(len(example.eng)):
		max_score = 0
		max_index = -1
		for index_ge in range(len(example.german)):
			word_en = example.eng[index_en]
			word_ge = example.german[index_ge]	
			score = probs[word_ge][word_en]
			if score > max_score:
				max_score = score
				max_index = index_ge
			elif score == max_score:
				# index first
				ratio_en = float(index_en) / float(len(example.eng))
				ratio_now = float(index_ge) / float(len(example.german))
				ratio_pre = float(max_index) / float(len(example.german))
				if ratio_now - ratio_en <= ratio_pre - ratio_en:
					max_index = index_ge
		'''
		word_ge = example.german[index_ge]
		sorted_x = sorted(probs[word_ge].items(), key=operator.itemgetter(1))[::-1]
		for item in sorted_x:
			word_en = item[0]
			if word_en in example.eng:
				index_en = example.eng.index(word_en)
				break
		'''
		f.write("%d-%d "%(max_index, index_en))
	f.write("\n")
