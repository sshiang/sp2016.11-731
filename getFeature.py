import os
import sys

def getF(fea):
	text = ""
	for i in range(len(fea)):
		text+="%d:%f "%(i+1,fea[i])
	return text

labels = [int(line.strip()) for line in open("/Users/eternal0815/Course/MT/homework/hw2/data/train.gold")]

labels_all = [0 for x in range(50339)]

for i in range(len(labels)):
	labels_all[i] = labels[i]


features = [[] for x in range(50339*2)]

for filename in os.listdir("features"):
	lineCount = 0
	for line in open("features/%s"%(filename)):
		scores = line.strip().split(",")
		score1 = float(scores[0])
		score2 = float(scores[1])
		features[2*lineCount].append(score1)
		features[2*lineCount+1].append(score2)
		lineCount+=1	

f = open("features_all.txt","w")
#for i in range(len(labels)):
for i in range(len(labels_all)):
	if labels_all[i] == -1:
		f.write("2 qid:%d %s\n"%(i+1,getF(features[2*i])))
		f.write("1 qid:%d %s\n"%(i+1,getF(features[2*i+1])))
	elif labels_all[i] == 1:
		f.write("1 qid:%d %s\n"%(i+1,getF(features[2*i])))
		f.write("2 qid:%d %s\n"%(i+1,getF(features[2*i+1])))	
	else: # =0
		f.write("1 qid:%d %s\n"%(i+1,getF(features[2*i])))
		f.write("1 qid:%d %s\n"%(i+1,getF(features[2*i+1])))
f.close()


f = open("features_train.txt","w")
for i in range(len(labels)):
	if labels_all[i] == -1:
		f.write("2 qid:%d %s\n"%(i+1,getF(features[2*i])))
		f.write("1 qid:%d %s\n"%(i+1,getF(features[2*i+1])))
	elif labels_all[i] == 1:
		f.write("1 qid:%d %s\n"%(i+1,getF(features[2*i])))
		f.write("2 qid:%d %s\n"%(i+1,getF(features[2*i+1])))	
	else: # =0
		f.write("1 qid:%d %s\n"%(i+1,getF(features[2*i])))
		f.write("1 qid:%d %s\n"%(i+1,getF(features[2*i+1])))
f.close()
