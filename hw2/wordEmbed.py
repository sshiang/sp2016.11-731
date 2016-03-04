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

def getFeatures(textFile, ind, duplicate):
	global max_length
	global wordEmbed
	global dictionary
	features_all = []
	words_all = [] 
	cc = 0
	c = 0
	prev_hyp = []
	for line in open(textFile):
		cc+=1
		if cc%1000 == 0:
			print cc, c
		hyp = wordpunct_tokenize(line.strip().split("|||")[ind].strip().decode("utf-8"))
		if hyp == prev_hyp and duplicate==False:
			continue
		c += 1	
		prev_hyp = copy.deepcopy(hyp)	

		features = []
		words = []
		for x in hyp:
			word = x.encode("utf-8")
			w = [0 for x in range(len(dictionary))]
			w[dictionary[word]]=1
			if x in wordEmbed:
				feature = wordEmbed[word]
			else:
				feature = [float(0) for y in range(300)]
			features.append(feature)
			words.append(w)

		words = words[1:]

		for i in range(max_length-len(hyp)):
			words.append([0 for x in range(len(dictionary))])
			features.append([float(0) for y in range(300)])
			
		words.append([0 for x in range(len(dictionary))])

		features_all.append(features)
		words_all.append(words)

	return np.array(features_all), np.array(words_all)

def getTest(test_X, test_y, model):
	global batch
	test_y = test_y.tolist()
	result = model.predict_proba(test_X, batch_size=batch).tolist()
	probs = []
	for i in range(len(test_y)):
		count = 0
		prob = 1
		for j in range(len(test_y[i])):
			if sum(test_y[i][j]) == 0:
				continue
			else:
				ind = test_y[i][j].index(1)
				count +=1
				prob *= result[i][j][ind]
		prob = prob**(1.0/float(count))
		probs.append(prob)
	return probs


def dumpResult(f, probs):
	#f = open(path,"w")
	f.write("\n".join([str(x) for x in probs]))
	f.write("\n")
	#f.close()


def getResults(textFile, outFile, ind):
	global wordEmbed
	global dictionary	
	f = open(outFile, "w")
	for line in open(textFile):
		hyp = wordpunct_tokenize(line.strip().split("|||")[ind].strip().decode("utf-8"))
		features = []
		words = []
		for x in hyp:
			word = x.encode("utf-8")
			w = [0 for x in range(len(dictionary))]
			w[dictionary[word]]=1
			if x in wordEmbed:
				feature = wordEmbed[word]
			else:
				feature = [float(0) for y in range(300)]
			features.append(feature)
			words.append(w)

		words = words[1:]
			
		words.append([0 for x in range(len(dictionary))])

		probs = getTest(np.array([features]),np.array([words]), model)
		dumpResult(f,probs)

	f.close()

textFile = "data/train-test.hyp1-hyp2-ref"
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


train_X, train_y = getFeatures(textFile, 2, False)

print len(train_X), len(train_y)
print len(train_X[0]), len(train_y[0])
print len(train_X[0][0]), len(train_y[0][0])


n_in_out = 300
n_hidden = 100

batch = 5
epoch = 32
	
model = Sequential()

model.add(LSTM(input_dim=300, output_dim=100, return_sequences=True))
model.add(Dropout(0.5))
model.add(TimeDistributedDense(input_dim=100, output_dim=len(dictionary)))
model.add(Activation('softmax'))
#model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")
#model.compile(loss='mse', optimizer='rmsprop')

model.compile(loss="categorical_crossentropy", optimizer='adam')
model.fit(train_X, train_y, nb_epoch=epoch, batch_size=batch, show_accuracy=True)

train_X = None
train_y = None

'''
test_X, test_y = getFeatures(textFile, 0, True)
probs = getTest(test_X, test_y, model)
dumpResult("hyp1.txt",probs)

test_X, test_y = getFeatures(textFile, 1, True)
'''

getResults(textFile, "hyp1.txt", 0)
getResults(textFile, "hyp2.txt", 1)


'''
for line in open(textFile):
	hyp = wordpunct_tokenize(line.strip().split("|||")[ind].strip().decode("utf-8"))
	if hyp == prev_hyp and duplicate==False:
		continue
	prev_hyp = copy.deepcopy(hyp)	

	features = []
	words = []
	for x in hyp:
		word = x.encode("utf-8")
		w = [0 for x in range(len(dictionary))]
		w[dictionary[word]]=1
		if x in wordEmbed:
			feature = wordEmbed[word]
		else:
			feature = [float(0) for y in range(300)]
		features.append(feature)
		words.append(w)

	words = words[1:]

	for i in range(max_length-len(hyp)):
		words.append([0 for x in range(len(dictionary))])
		features.append([float(0) for y in range(300)])
		
	words.append([0 for x in range(len(dictionary))])

	probs = getTest(np.array([features]),np.array([words]), model)
	dumpResult(f,probs)

probs = getTest(test_X, test_y, model)
dumpResult("hyp2.txt",probs)
'''


