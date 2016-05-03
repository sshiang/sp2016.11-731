import os
import sys
import numpy as np
import cPickle
import scipy.io as sio

'''
length = 0
for line in open(sys.argv[1]):
	l = len(line.strip().split())
	if l >= length:
		length = l
print l
'''

def makeDirectory(path):
	if os.path.exists(path)==False:
		os.makedirs(path)

makeDirectory(sys.argv[1])
datapath = sys.argv[1]

lang = sys.argv[2]"src"
#lang = "tgt"

vocab = {}
c= 0
for line in open(os.path.join(datapath,"train.%s.vocab"%(lang))):
	word = line.strip()
	vocab[word]=c
	c+=1
vocabSize = c

f = open("train.%s.vocab.pickle"%(lang),"wb")
cPickle.dump(vocab, f)
f.close()

max_length = 67

dataType = ["train","dev","test"]
#dataType = ["train","dev"]
for typ in dataType:

	fc = open("%s/%s.%s.mat"%(sys.argv[2],typ,lang),"wb")

	# input
	#data_all = []
	# mask
	mask_all = []
	# output
	output_all = []
	lc = 0
	for line in open(os.path.join(datapath, "%s.%s"%(typ,lang))):
		lc +=1
		if lc % 100 == 0:
			print lc
		#data = [[0 for x in range(vocabSize)] for y in range(max_length)]
		mask = [0 for x in range(max_length)]
		output = [0 for x in range(max_length)]
		words = line.strip().split()
		for i in range(len(words)):
			word = words[i]
			mask[i] = 1
			output[i] = int(word)
			'''
			if word not in vocab:
				#data[i][vocab["unk"]]=1
				output[i][0] = int(word) #vocab["unk"]
			else:
				#data[i][vocab[word]]=1
				output[i][0] = int(word)#vocab[word]
			'''
		#data_all.append(data)
		output_all.append(output)
		mask_all.append(mask)
	#outputpath = os.path.join(sys.argv[2],"%s.%s.input.npy"%(typ,lang))
	#np.save(outputpath, data_all)
	#outputpath = os.path.join(sys.argv[2],"%s.%s.npy"%(typ,lang))
	#np.save(outputpath, np.array(output_all))
	#outputpath = os.path.join(sys.argv[2],"%s.%s.mask.npy"%(typ,lang))
	#np.save(outputpath, np.array(mask_all))


	sio.savemat(fc, dict(input=output_all, mask=mask_all))
	
