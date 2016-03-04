import os
import sys
import math
from nltk import word_tokenize
from sklearn.linear_model import LogisticRegression
from nltk.stem.lancaster import LancasterStemmer

def ngramMatch(hyp,gold,n):
	count=0
	word_gold = " ".join(gold)

	while (len(gold)<n or len(hyp)<n):
		n-=1

	#print hyp, gold
	for i in range(len(hyp)-n):
		word_hyp = " ".join(hyp[i:i+n])
		if word_hyp in word_gold:
			count+=1
	#print count
	return count

def evaluate(count,hyp,gold,n):

	while (len(gold)<n or len(hyp)<n):
		n-=1

	precision = float(count)/float(len(gold)-n+1)
	recall = float(count)/float(len(hyp)-n+1)
	if precision == 0 and recall == 0:
		return 0

	fmeasure = float(2*precision*recall)/float(precision+recall)
	#return precision, recall, fmeasure	
	return fmeasure 

def stemming(sequence):
	global stem
	return [stem.stem(x) for x in sequence]

n = int(sys.argv[1])

stem = LancasterStemmer()

tupleFile = "data/train-test.hyp1-hyp2-ref"
goldFile = "data/train.gold"
if len(sys.argv)<=2:
	outputFile = "output.txt"
else:
	outputFile = "output_%s.txt"%(sys.argv[2])

labels = [int(line.strip()) for line in open(goldFile)]

features = []
for line in open(tupleFile):
	seg = line.strip().split("|||")
	assert (len(seg)==3), "tuple length mismatch"
	hyp1 = word_tokenize(seg[0].strip().decode("utf-8"))	#.split(" ")	
	hyp1 = stemming(hyp1)
	hyp2 = word_tokenize(seg[1].strip().decode("utf-8"))	#.split(" ")
	hyp2 = stemming(hyp2)
	gold = word_tokenize(seg[2].strip().decode("utf-8"))	#.split(" ")
	gold = stemming(gold)
	#print hyp1
	#print hyp2
	count1 = ngramMatch(hyp1,gold,n)
	count2 = ngramMatch(hyp2,gold,n)
	fmeasure1 = evaluate(count1,hyp1,gold,n)
	fmeasure2 = evaluate(count2,hyp2,gold,n)
	score = fmeasure2-fmeasure1
	features.append([fmeasure1,fmeasure2,score])
	#features.append([score])

	'''
	if score >= 0.02:
		f.write("1\n")
	elif score <= -0.02:
		f.write("-1\n")
	else:
		f.write("0\n")
	'''	
print "predicting..."

f = open(outputFile,"w")
clf = LogisticRegression()
clf.fit(features[:len(labels)], labels)
for i in range(len(features)):
	result = clf.predict([features[i]])
	f.write("%d\n"%(result[0]))
f.close()
