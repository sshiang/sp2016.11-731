import sys
import nltk
from nltk.stem.lancaster import LancasterStemmer

stem_en = LancasterStemmer()
stem_ge = nltk.stem.snowball.GermanStemmer()
f= open("data/de-en-compounds-space-processed.txt","w")

for line in open(sys.argv[1]):
	lans = line.lower().strip().split("|||")
	german = []
	for word in lans[0].strip().split(" "):
		word2_stem = []
		word2 = word.split("/")
		german.append(" ".join(word2))
		#for word2 in word.split("/"):
		#	word2_stem.append(stem_ge.stem(word2.decode('utf-8')))
		#german.append(" ".join(word2_stem).encode("utf-8"))
			
		
	#eng = [stem_en.stem(x.decode('utf-8')).encode("utf-8") for x in lans[1].strip().split(" ")]
	eng = lans[1].strip().split(" ")
	
	f.write("%s ||| %s\n"%(" ".join(german)," ".join(eng)))

f.close()
