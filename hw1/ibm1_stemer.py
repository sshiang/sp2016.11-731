#!/usr/bin/env python
import optparse
import sys
import numpy as np
from collections import defaultdict
from nltk.stem.snowball import SnowballStemmer
import codecs
import pickle
# CommandLine
optparser = optparse.OptionParser()
optparser.add_option("-b", "--bitext", dest="bitext", default="data/dev-test-train.de-en", help="Parallel corpus (default data/dev-test-train.de-en)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.9, type="float", help="Threshold for aligning with Dice's coef_vocicient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
optparser.add_option("-i", "--num_iterations", dest="num_iter", default=1, type="int", help="Number of EM iternations")
(opts, _) = optparser.parse_args()

sys.stderr.write("Training with IBM model 1...\n")
bitext = [[sentence.strip().split() for sentence in pair.split(' ||| ')] for pair in codecs.open(opts.bitext,encoding='utf-8')][:opts.num_sents]

# transfer to lower case
bitext = [[[x.lower() for x in sent ] for sent in bisent] for bisent in bitext]

fe_occur = defaultdict(set) # German-Engligh Co-occurance
f_voc = set() # German
e_voc = set() # Engligh

# stemmer
e_stemmer = SnowballStemmer("english")
f_stemmer = SnowballStemmer("german")
for (n, (f,e)) in enumerate(bitext):
    for idx, f_i in enumerate(f):
        f[idx] = f_stemmer.stem(f_i)
    for idx, e_i in enumerate(e):
        e[idx] = e_stemmer.stem(e_i)


# build dictionary
sys.stderr.write("Building dictionary...\n")
for (n, (f, e)) in enumerate(bitext):
    f[0:0] = ['null'] # remark: add NULL 
    for f_i in set(f):
        f_voc.add(f_i)
        for e_j in set(e):
            fe_occur[f_i].add(e_j)
            e_voc.add(e_j)
    if n % 500 == 0:
        sys.stderr.write(".")
sys.stderr.write("\n")

p_ef = dict()
for f in fe_occur.keys():
    p_ef[f] = dict()
    for e in fe_occur[f]:
        p_ef[f][e] = 1.0/len(fe_occur[f])

for it in range(opts.num_iter):
    sys.stderr.write("iteration %i:" %(it))

    # M-step
    # handle zero prob
    expect = defaultdict(lambda: defaultdict(float))
    for (n, (f, e)) in enumerate(bitext):
        #p_a_e = 1.0/(len(f)+1) # note: speed up

        for (j, e_j) in enumerate(e):
            # trick exact match
            if f_i == e_j:
                expect[f_i][e_j] +=1
            else:
            # E[F]
                norm = 0         
                for (i, f_i) in enumerate(f):
                    norm += p_ef[f_i][e_j]
                    #norm += p_ef[f_i][e_j]*p_a_e
                # E[E|F]
                for (i, f_i) in enumerate(f):
                    #expect[f_i][e_j] += (p_ef[f_i][e_j]*p_a_e)/norm
                    expect[f_i][e_j] += (p_ef[f_i][e_j])/norm
        if n % 500 == 0:
            sys.stderr.write(".")
    # E-step, update p_ef
    for f in expect.keys():
        norm = sum(expect[f][e] for e in fe_occur[f])
        for e in fe_occur[f]:
            p_ef[f][e]=expect[f][e]/norm
    sys.stderr.write("\n")


pickleFile = "p_ef.pkl"
output = open(pickleFile, 'wb')
pickle.dump(p_ef, output)
output.close()



# generate alignments
sys.stderr.write('Generating alignments...\n')
for (n, (f, e)) in enumerate(bitext):
    for (i, e_i) in enumerate(e):
        max_prob = -1
        max_ind = -1
        for (j, f_j) in enumerate(f): 
            if (p_ef[f_j][e_i] > max_prob):
                max_prob = p_ef[f_j][e_i]
                max_ind = j
            elif (p_ef[f_j][e_i] == max_prob): # tend to align diag
                if (abs(max_ind-i)> abs(j-i)):
                    max_ind = j
                    max_prob = p_ef[f_j][e_i]
        if max_ind > 0:
            max_ind = max_ind - 1 # NULL
        sys.stdout.write("%i-%i " % (max_ind,i))
    sys.stdout.write("\n")
