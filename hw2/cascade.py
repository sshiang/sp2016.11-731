#files = ["output_0_10000.txt","output_10000_20000.txt","output_20000_30000.txt","output_30000_40000.txt","output_40000_50000.txt","output_50000_60000.txt"]
files = ["score_0_10000.txt","score_10000_20000.txt","score_20000_30000.txt","score_30000_40000.txt","score_40000_50000.txt","score_50000_60000.txt"]


f = open("score.txt","w")
for filename in files:
	for line in open(filename):
		f.write(line)
f.close()
