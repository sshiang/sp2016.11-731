Usage: ./decode7 -s [stack_size] -a [distance parameter]> [output]
(* the other paramters are the same as described in example code.)

The best score that we achieved: -4830.7803

What we tried: 
insert phrase into arbitrary place.
scan sentence both forward and backard, and select the one with highest score.
distance score for insertion point. The best parameter for distance is 0.1
future prediction: If scanning the sentence from the beginning and seeing the position i in the sentence, we use score of (i,end) sub-sentence from backward scanning. 



There are three Python programs here (`-h` for usage):

 - `./decode` a simple non-reordering (monotone) phrase-based decoder
 - `./grade` computes the model score of your output

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./decode | ./grade


The `data/` directory contains the input set to be decoded and the models

 - `data/input` is the input text

 - `data/lm` is the ARPA-format 3-gram language model

 - `data/tm` is the phrase translation model

