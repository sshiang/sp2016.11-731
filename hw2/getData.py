import os
import sys
from nltk.tokenize import wordpunct_tokenize
textFile = "data/train-test.hyp1-hyp2-ref"
outFile = "data/text"

cc = 0
dictionary = {}
lineDict = {}
for line in open(textFile):
	candidates = line.strip().split("|||")
	for candidate in candidates:
		hyp = wordpunct_tokenize(candidate.strip().decode("utf-8"))
		hypx = " ".join([x.encode("utf-8") for x in hyp])
		if hypx not in lineDict:
			lineDict[hypx] = 0

		for word in hyp:
			word = word.encode("utf-8")
			if word not in dictionary:
				dictionary[word]=cc
				cc+=1
		#max_length = max(max_length, len(hyp))

f = open(outFile,"w")
for line in lineDict:
	f.write(line+"\n")
f.close()
