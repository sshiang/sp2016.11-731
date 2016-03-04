import os
import sys
import math 
import numpy as np
import nltk
import operator
from nltk.tokenize import wordpunct_tokenize
from gensim.models import word2vec
import copy

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense, Masking, Merge
from keras.datasets.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU, LSTM

def getText(textFile, ind, outFile, duplicate):
	global max_length
	global wordEmbed
	global dictionary

	f = open(outFile,"w")
	
	prev_hyp = []
	for line in open(textFile):
		hyp = wordpunct_tokenize(line.strip().split("|||")[ind].strip().decode("utf-8"))
		if hyp == prev_hyp and duplicate == False:
			continue
		prev_hyp = copy.deepcopy(hyp)	

		for x in hyp:
			word = x.encode("utf-8")

			f.write("%s "%(word))
		f.write("\n")
	f.close()

def getProb(textFile):
	count = 0
	for line in open(textFile):
		line = line.strip()
		count +=1
		if count == 5:
			return float(line.replace("test log probability: ",""))

	return 0

def getResult(textFile, ind1, ind2, outFile, outFile2):
	fout = open(outFile,"w")
	fout2 = open(outFile2, "w")
	#probs = []
	for line in open(textFile):
		hyp1 = wordpunct_tokenize(line.strip().split("|||")[ind1].strip().decode("utf-8"))
		hyp2 = wordpunct_tokenize(line.strip().split("|||")[ind2].strip().decode("utf-8"))

		f = open("temp.txt","w")
		f.write("%s\n"%" ".join([x.encode("utf-8") for x in hyp1]))
		f.close()
		os.system("~/Course/AMMML/project/FeatureAugmentedRNNToolkit/rnnlm -rnnlm ~/Course/AMMML/project/FeatureAugmentedRNNToolkit/model -test temp.txt -features-matrix ~/Course/AMMML/project/FeatureAugmentedRNNToolkit/feature.txt -independent > temp_out.txt")
		
		prob1 = getProb("temp_out.txt")
	
		f = open("temp.txt","w")
		f.write("%s\n"%" ".join([x.encode("utf-8") for x in hyp2]))
		f.close()
		os.system("~/Course/AMMML/project/FeatureAugmentedRNNToolkit/rnnlm -rnnlm ~/Course/AMMML/project/FeatureAugmentedRNNToolkit/model -test temp.txt -features-matrix ~/Course/AMMML/project/FeatureAugmentedRNNToolkit/feature.txt -independent > temp_out.txt")
			
		prob2 = getProb("temp_out.txt")

		#probs.append([prob1,prob2])
		fout.write("%f\t%f\n"%(prob1,prob2))
		fout2.write("%f\t%f\n"%(prob1/float(len(hyp1)),prob2/float(len(hyp2))))
	fout.close()
	fout2.close()
def getFeatures(textFile, ind, duplicate):
	global max_length
	global wordEmbed
	global dictionary
	f = open("features.txt","w")

	for word in dictionary:
		if word not in wordEmbed:
			f.write("%s %s\n"%(word," ".join(["0" for x in range(300)])))
		else:
			f.write("%s %s\n"%(word," ".join([str(x) for x in wordEmbed[word]])))

	f.close()


textFile = "data/train-test.hyp1-hyp2-ref"
'''
#textFile = "data/small"
word2vecFile = "/Users/eternal0815/Course/multimodal/project/GoogleNews-vectors-negative300.bin"

wordEmbed = word2vec.Word2Vec.load_word2vec_format(word2vecFile,binary=True)
	
max_length = 0

cc = 0
dictionary = {}
for line in open(textFile):
	candidates = line.strip().split("|||")
	for candidate in candidates:
		hyp = wordpunct_tokenize(candidate.strip().decode("utf-8"))
		for word in hyp:
			word = word.encode("utf-8")
			if word not in dictionary:
				dictionary[word]=cc
				cc+=1

		max_length = max(max_length, len(hyp))


getFeatures(textFile, 2, False)
'''
#getText(textFile, 2, "train.txt",False)

getResult(textFile, 0, 1, "result_rnn.txt", "result_rnn_normal.txt")

#getText(textFile, 0, "hyp1.txt",True)
#getText(textFile, 1, "hyp2.txt",True)

