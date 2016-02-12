Team Name: TAIWAN NO. 1 
Team Member: sshiang, Fred, Bernie

Please download file first: https://drive.google.com/file/d/0B5yCs07HNjFyeVpjVkFJRGZEbDg/view?usp=sharing(file too large to upload to github)


A. Result: (Dev set)
1. Best system(Lower case, stemming, exact-match, HMM, symmetry classifier)
    Precision: 0.762712
    Recall: 0.820648
    AER: 0.210772

2. Bullpen system (Best system without classifier)
    Precision: 0.843837
    Recall: 0.397674
    AER: 0.233539

B. Usage:
1. Pre-processed FIles
1-2. python compound.py
1-3. python flatten.py data/parse-de-en-compound-nosemicolon.txt

2. python HMM_compound.py -n 100000 > HMM_compound_forward.txt
3. python HMM_compound_backward.py -n 100000> HMM_compound_back.txt
4-1. symmetry.py -f HMM_compound_forward.txt -b HMM_compoind_back.txt -m diag
4-2. python combine.py 
5. ./grade < output.txt 

C. Preprocessing
1. lower case
2. stem (Snowball)
3. compound decomposition (“/” as delimiter)

D. Alignment Algorithms
1. IBM1 model
EM as introduced in homework.
2. HMM
An implementation of [1]
3. Other improvements: 
3-1 Null in sentence
Add null in front of every sentence
3-2 Exact Match:
Direct matching and count same word in EM/HMM with no(' , ', 'die', 'am')
3-3 Forward/Backward
Generate German->English and English->German alignments for symmetry

E. Symmetry Algorithm
1. Unsupervised symmetry
    a. intersection
    b. union
    c. grow diagonal from intersection to union (rule-based)
2. Supervised symmetry
    a. Features:
        i. POS tags
        ii. length of word
        iii. position
        iv. IBM model 1 score
    b. classifier: logistic regression
    c. training data: 150 dev sentences
    d. grow diagonal from intersection. For those alignment pair in union but not intersect, we applied the classifier to see whether it should be included in the alignment results. 

F. Tools:
1. NLTK: English and German stemmer, stopwords
2. Pattern: English and German POS tagger
3. BaseForms: German compound words decomposition
4. Sklearn: Logistic regression

G. Reference:
[1] Vogel et al. 1996, HMM-Based Word Alignment in Statistical Translation.

H. Development History

method
Precision
Recall
AER (dev)
IBM1 (No preprocessing)
0.54426
0.617689
0.422871
IBM1 (null, lower, stemmer, exact match)
0.611936
0.695208
0.350789
HMM (No preprocessing)
0.648201
0.718816
0.320189
IBM1 (3+symmetry)
0.7558
0.613108
0.32068
Test no Null / compound
0.566533
0.638125
0.40142
Test no Null / lemma
0.539692
0.612403
0.42776
Test no Null  / pos
0.213878
0.235025
0.776656
no Null  / pos+lemma （combine pos and lemma）
0.530554
0.601128
0.437855
no Null / pos+lemma*lemma （multiply prob）
0.537122
0.609584
0.430442
no Null / pos*lemma （multiply prob）
0.542547
0.618041
0.423659
no Null / word(lowercase)
0.548258
0.623679
0.417981
As above / word(lowercase)*pos （multiply prob）
0.554255
0.628964
0.412303
HMM (null, lower, stemmer exact match)
0.663621
0.742424
0.301104
HMM, null, lemma, pos, exact match
0.557016
0.628259
0.411078
HMM (null, lower, stemmer,pos, exact match)
0.663621
0.742424
0.301104
HMM, stem, exact match no(' , ', 'die', 'am')
0.66962
0.749471
0.294619
HMM, stem, exact match no(' , ', 'die', 'am') + symmetry
0.715726
0.717407
0.28348
IBM1 (5+ pretrain p_e_f+p_f_e, n=5000, i=5)
0.767914
0.646934
0.296262
IBM1 (5+ pretrain p_e_f+p_f_e, n=100000, i=10)
0.786612
0.704017
0.255684
HMM, stem, exact match, compound, no(' , ', 'die', 'am')
0.702
0.781184
0.262543
HMM, stem, exact match, compound, no(' , ', 'die', 'am') + symmetry
0.714454
0.763566
0.263192
Combine 22 forward + 20 backward, ~/qq.al
0.768205
0.754405
0.238417
HMM, stem, exact match, compound, no(' , ', 'die', 'am') +dict
0.698567
0.786117
0.262168
HMM, stem, exact match, compound, no(' , ', 'die', 'am') +dict ＋sym
0.755262
0.767794
0.238831
HMM, stem, exact match, compound, no(' , ', 'die', 'am') + sym2
0.843837
0.697674
0.233539

