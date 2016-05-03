package:
	1. theano:'0.8.0rc1.dev-cb1184d45cdc88e0f8bea0896eb7ea3775fe7593'
	2. lasagne:'0.2.dev1'

How to implement:
	1. python preprocess_hw4.py de
	   python preprocess_hw4.py en
	2. python analysis.py src
	   python analysis.py tgt
	3. pythn seq2seq_attn.py
		the output is in "translation_test.txt"


What I've tried:
	1. sequence to sequence prediction
		a. hidden size: 300
		b. only 1 stack LSTM. 
	2. global attention: 
		a. context vector is the weighted sum of source LSTM language model
		b. weight score: inner product of source LSTM and target LSTM
	3. one hot or posterior as input features of decoder, I get better results using one hot. 

What's interesting: 
	I forgot to update the weights in target LSTM in the beginning, but I still got 21.52 in the leader board! How amazing deep learing it is XD



