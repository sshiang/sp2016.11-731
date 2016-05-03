package:
	1. theano:'0.8.0rc1.dev-cb1184d45cdc88e0f8bea0896eb7ea3775fe7593'
	2. lasagne:'0.2.dev1'

How to implement:
	./seq2seq_attn.py
	the output is in "translation_attn.txt"

What I've tried:
	1. sequence to sequence prediction
		a. hidden size: 300
		b. only 1 stack LSTM. 
	2. global attention: 
		a. context vector is the weighted sum of source LSTM language model
		b. weight score: inner dot of source LSTM and target LSTM

What's interesting: 
	I forgot to update the weights in target LSTM in the beginning, but I still got 21.52 in the leader board! How amazing deep learing it is XD


