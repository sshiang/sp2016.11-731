Team Name: TAIWAN NO. 1 
Team Member: sshiang, Fred, Bernie

A. Result: (Dev set)
Precision: 0.544490

B. Usage:
Fred
glove vectors
Bernie:
python metoer6.py > output.txt
sshiang:
python parser.py


C. Preprocsssing
1. lower case
2. remove punctuation
3. stemming/lemmatization

D. Translation Evaluation Algorithms
0. Framentation (complete Meteor funcationalies)
1. Exact match
2. POS tag + Exact match
3. n-gram match (n=2~4)
4. stem match
5. similarity match by word embeddings (Glove)
6. Doc2Vec (treat each sentence as a doc)

E. Fusion Method and Other Tricks
0. Average late fusion of the inididual score ouput
1. MLP fusion on train-dev set
2. Grid search for (alpha, beta, gamma)
3. Rank SVM
4. Majority vote of our results

F. Tools
1. NLTK: stemmer, stopwords, POS tags
2. Glove for word embeddings

