import os

predict = [float(line.strip()) for line in open("predict.txt")]

f = open("output.txt","w")
for i in range(len(predict)/2):
	if predict[2*i] > predict[2*i+1]:
		f.write("-1\n")
	elif predict[2*i] < predict[2*i+1]:
		f.write("1\n")
	else:
		f.write("0\n")
	
f.close()
