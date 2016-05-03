import os
import sys
import operator
from nltk.tokenize import WordPunctTokenizer
import nltk

def makeDirectory(path):
	if os.path.exists(path)==False:
		os.makedirs(path)

def writeId(path,outpath,lexicon):
	f = open(outpath,"w")
	fr = open(outpath+".reversed","w")
	for line in open(path):
		words = getWords(line.strip(), useCompound)
		idwords = [str(lexicon[words[i]]) if words[i] in lexicon else "0" for i in range(len(words))]
		f.write(" ".join(idwords)+"\n")
		fr.write(" ".join(idwords[::-1])+"\n")
	f.close()
	fr.close()

def getWords(line, useCompound):
	global lang
	#words = nltk.word_tokenize(line.strip().decode('utf-8'))
	#return [x.encode('utf-8') for x in words]
	
	words = line.strip().lower().split()
	return words

lang = sys.argv[1]

train_path = "/Users/eternal0815/Course/MT/project/data/train."+lang
test_path = "/Users/eternal0815/Course/MT/project/data/test."+lang
val_path = "/Users/eternal0815/Course/MT/project/data/val."+lang

#train_path = "/Users/eternal0815/Course/MT/homework/hw4/data/train."+lang
#val_path = "/Users/eternal0815/Course/MT/homework/hw4/data/dev."+lang
#test_path = "/Users/eternal0815/Course/MT/homework/hw4/data/test."+lang

outdir = sys.argv[2] # "/Users/eternal0815/Course/MT/project/nmt/output/wmt-lower/"
makeDirectory(outdir)

out_train_path = "%s/%s"%(outdir,train_path.split("/")[-1])
out_test_path = "%s/%s"%(outdir,test_path.split("/")[-1])
out_val_path = "%s/%s"%(outdir,val_path.split("/")[-1])


	

appendix = ".token"
thres_freq = 3
useCompound = False


lexicon = {"<unk>":0, "<s>":1, "</s>":2}
freq = {}
c = 3
for line in open(train_path+appendix):
	#words = line.strip().split()
	words = getWords(line.strip(), useCompound)

	for w in words:
		if w not in freq:
			freq[w]=1
		else:
			freq[w]+=1

lexicon= {"<unk>":0}
for w in freq:
	if freq[w] >= thres_freq:
		lexicon[w]=len(lexicon)

lexicon_path = "%s/%s.vocab"%(outdir,train_path.split("/")[-1])
fl = open(lexicon_path,"w")
sorted_x = sorted(lexicon.items(), key=operator.itemgetter(1))
for x in sorted_x:
	fl.write(x[0]+"\n")
fl.close()

writeId(train_path+appendix,out_train_path,lexicon)
writeId(test_path+appendix,out_test_path,lexicon)
writeId(val_path+appendix,out_val_path,lexicon)

