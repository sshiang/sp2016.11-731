#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict

import itertools
import pickle
from nltk.stem.snowball import SnowballStemmer
import codecs

optparser = optparse.OptionParser()
optparser.add_option("-b", "--bitext", dest="bitext", default="data/de-en-compounds-space-processed_new.txt", help="Parallel corpus (default data/dev-test-train.de-en)")
optparser.add_option("-c", "--bitextCompound", dest="bitextCompound", default="data/parse-de-en-compound-nosemicolon.txt", help="Parallel corpus (default data/dev-test-train.de-en)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
#bitext = [[sentence.strip().split() for sentence in pair.split(' ||| ')] for pair in open(opts.bitext)][:opts.num_sents]
bitext = [[sentence.strip().split() for sentence in pair.split(' ||| ')] for pair in codecs.open(opts.bitext,encoding='utf-8')][:opts.num_sents]
bitextCompound = [[sentence.strip().split() for sentence in pair.split(' ||| ')] for pair in codecs.open(opts.bitextCompound,encoding='utf-8')][:opts.num_sents]
not_exact_match = [',', 'die', 'am']

#HMM

PREPROCESS = False
ALIGN_NULL = True

def preProcessing(bitext):
    # transfer to lower case
    bitext = [[[x.lower() for x in sent ] for sent in bisent] for bisent in bitext]
    # stemmer
    e_stemmer = SnowballStemmer("english")
    f_stemmer = SnowballStemmer("german")
    for (n, (f,e)) in enumerate(bitext):
        for idx, f_i in enumerate(f):
            f[idx] = f_stemmer.stem(f_i)
        for idx, e_i in enumerate(e):
            e[idx] = e_stemmer.stem(e_i)

def wordMapping(f):
    wordMatch = {}
    count = 0
    for idx, f_i in enumerate(f):
        count += len(f_i.split('/'))
        for i in xrange(count-len(f_i.split('/')),count,1):
            wordMatch[i] = idx
    return wordMatch
            
             

def reverse_enumerate(iterable):
    """
    Enumerate over an iterable in reverse order while retaining proper indexes
    """
    return itertools.izip(reversed(xrange(len(iterable))), reversed(iterable))
'''
def processCompoundWord(bitext):
    for (n, (f, e)) in enumerate(bitext):
        newf = []
        for i, f_i in enumerate(f):
            words = f_i.split('/')
            for w in words:
                newf.append(w)
        f = newf
'''
            
def addNullToken(bitext):
    for (n, (f, e)) in enumerate(bitext):
        f.append("null");

def initialize(p):
    return defaultdict(lambda: p)

def forward(f, e, trans, align):
    alpha = initialize(1.0)
    for j, e_j in enumerate(e): 
        for i, f_i in enumerate(f):
            #initial condition
            if (j==0):
                alpha[(j, i)] = trans[(e_j, f_i)]
            else:
                alpha[(j, i)] =  sum(alpha[(j-1, k)]*trans[(e_j, f_i)]*align[i-k] for k, f_k in enumerate(f))
    #print sum(alpha[(len(f)-1, k)] for k, e_k in enumerate(e))
    return alpha

def backward(f, e, trans, align):
    beta = initialize(1.0)
    for j, e_j in reverse_enumerate(e):
        for i, f_i in enumerate(f):
            if j==len(e)-1:
                #beta[(j, i)] = trans[(e_j, f_i)]
                beta[(j, i)] = 1
            else:
                #beta[(j, i)] = sum(beta[(j+1, k)]*trans[(e_j, f_i)]*align[k-i] for k, f_k in enumerate(f))
                beta[(j, i)] = sum(beta[(j+1, k)]*trans[(e[j+1], f[k+1])]*align[k-i] if k+1<len(f) else beta[(j+1, k)]*align[k-i] for k, f_k in enumerate(f) )
    #print sum(beta[(0, k)] for k, e_k in enumerate(e))
    return beta
            
        

#transition probability from Model 1 
def Model1_EM(bitext, num_iter): 
    t = initialize(0.25) 
    for it in xrange(num_iter): 
        sys.stderr.write("EM iter=%d\n" % it)
        #E-step 
        count_e_f = defaultdict(float) 
        total_f = defaultdict(int) 
        for (n, (f, e)) in enumerate(bitext): 
            s_total = defaultdict(float) 
            for e_i in e: 
                for f_j in f: 
                    s_total[e_i] += t[(e_i,f_j)] 
            for e_i in e: 
                for f_j in f: 
                    if e_i == f_j and e_i not in not_exact_match:
                        count_e_f[(e_i,f_j)] += 1
                        total_f[f_j] += 1
                    else:
                        count_e_f[(e_i,f_j)] += t[(e_i,f_j)]/s_total[e_i] 
                        total_f[f_j] += t[(e_i,f_j)]/s_total[e_i] 
        #M-step 
        for (e_i,f_j) in t: 
            t[(e_i, f_j)] = count_e_f[(e_i, f_j)]/total_f[f_j] 
    return t 

def BaumWelch(bitext, NUM_ITER, trans):
    align = initialize(0.01)
    for it in xrange(NUM_ITER):
        sys.stderr.write("BaumWelch iter=%d\n" % it)
        corpus_xi = initialize(0.0)
        for (n, (f, e)) in enumerate(bitext):
            xi = {}
            alpha = forward(f, e, trans, align)    
            beta = backward(f, e, trans, align)
            for t in xrange(len(e)-1):
                sum_xi = 0.0
                for i, f_i in enumerate(f):
                    for j, f_j in enumerate(f):
                        xi[(i, j)] = alpha[(t, i)]*align[j-i]*trans[(e[t+1], f_j)]*beta[(t+1, j)]
                        sum_xi += xi[(i, j)]
                for i, f_i in enumerate(f):
                    for j, f_j in enumerate(f):
                        corpus_xi[(i, j)] += xi[(i, j)]/sum_xi

        #update
        gamma = initialize(0.0)
        for key in corpus_xi.keys():
            gamma[key[0]] += corpus_xi[key]     
        for key in corpus_xi.keys():
            jump = key[1] - key[0]
            align[jump] += corpus_xi[key]/gamma[key[0]]
        sum_align = sum(align.values())
        for key in align.keys():
            align[key] = align[key]/sum_align
    return align


def viterbi(f, e, trans, align):
    viterbi = initialize(0.0)
    backtrack = initialize(-1)
    alignment = []
    for j, e_j in enumerate(e): 
        for i, f_i in enumerate(f):
            #initial condition
            if (j==0):
                viterbi[(j, i)] = trans[(e_j, f_i)]
                backtrack[(j, i)] = i
            else:
                tmp =  [viterbi[(j-1, k)]*trans[(e_j, f_i)]*align[i-k] for k, f_k in enumerate(f)]
                viterbi[(j, i)] = max(tmp)
                backtrack[(j, i)] = tmp.index(max(tmp))
    #print backtrack

    #find max viterbi
    max_prob = 0
    for i in xrange(len(f)):
        if (viterbi[(len(e)-1,i)] >= max_prob): 
            max_prob = viterbi[(len(e)-1,i)]
            max_prob_id = i
    alignment.append(max_prob_id)
    for j in xrange(len(e)-1, 0, -1):
        alignment.append(backtrack[j, (alignment[-1])])
    alignment.reverse()
    return alignment

def dumpParameters(trans, align):
    trans = dict(trans)
    align = dict(align)
    output = open('trans_compound.pkl', 'wb')
    pickle.dump(trans, output)
    output.close()
    output = open('align_compound_stem.pkl', 'wb')
    pickle.dump(align, output)
    output.close()

def main():
    if PREPROCESS:
        preProcessing(bitext)
    if ALIGN_NULL:
        addNullToken(bitext)    
    trans = Model1_EM(bitext, 5)    
    align = BaumWelch(bitext, 3, trans)
    for (f, e), (fc, ec) in zip(bitext, bitextCompound):
        wordMatch = wordMapping(fc + ['null'])
        alignment = viterbi(f, e, trans, align)
        for i, a in enumerate(alignment):
            if ALIGN_NULL:
                if wordMatch[a] < len(fc) :
                    sys.stdout.write("%i-%i " % (wordMatch[a], i))
            else:
                sys.stdout.write("%i-%i " % (wordMatch[a], i))
        sys.stdout.write("\n")
    dumpParameters(trans,align)

if __name__ == "__main__":
    main()
