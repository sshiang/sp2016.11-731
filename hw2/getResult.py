import sys

f = open("output.txt","w")
for line in open(sys.argv[1]):
	probs = line.strip().split("\t")
	if float(probs[0])<=float(probs[1]):
		f.write("-1\n")
	else: #if float(probs[0])>float(probs[1]):
		f.write("1\n")
	#else:
	#	f.write("0\n")
f.close()
