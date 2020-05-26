
# counting_NN
## Author: Natasha Seelam
## 2020 May 25

** Note ** (2020.05.25) The writeup is still in progress.

A toy example to learn how an RNN can count.

The following repo is inspired from a paper I read earlier this spring (https://arxiv.org/pdf/1805.04908.pdf). RNNs are known to be Turing complete when computational time is unbounded and there is infinite precision. Realistically, there are few cases wherein this case can be approximated well. 

The paper goes on to characterize a few RNN model types: namely LSTMs and GRU; particularly, it is interested in identifying counting behavior within hidden units of these models. Their first figure seems to indicate that LSTM's dedicate at least 1 hidden unit to counting behavior. They try 2 "languages", predicting (a^N b^N) and (a^N b^N c^N). The goal of the model is to classify if the sequence of letters is of this form.  

This paper was quite interesting to me, I wanted to ask a few other questions of the data. The implementation of the following architecture is as close to the reported architecture (ReLU, 10 hidden units, LSTM). As per the paper's requirements, the models are trained until they attain 100% accuracy on the "testing" set (generally termed validation in other settings).

This implementation is for educational purposes, so I make no guarantees on efficiency. 

I wanted to probe this counting behavior for several simple questions:

(1) What does the counting look like for 'negative' examples (those that don't form the a^N b^N behavior)?

(2) The models seem to dedicate at least 1 hidden node to recognizing the counting; why are the other nodes important for the solution?




