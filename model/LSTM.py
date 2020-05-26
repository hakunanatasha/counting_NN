"""
Date: 2020 May 20
Author: Natasha Seelam

Teaching an LSTM to count using sequences of a^N x b ^N.

Paper: https://arxiv.org/pdf/1805.04908.pdf

Note, the parameters are 10-dimension 1-layer LSTM. Activation - ReLU

The claims of the paper include:
(1) LSTMS can recognize 'languages' of the form a^N x b^N etc.
(2) LSTMS trained on this can *generalize* to other counting forms (i.e. N=100 -> N=256)
(3) Trained LSTMs hidden layers show this counting mechanism in one of their hidden unit.

Notes:
2020/05/25
+ Switched to one-hot-encoding of language 
+ Add pack-padded sequences for variable length

Note, I've seen some claims suggesting sorted sequences are even faster to compute on. That being said,
pytorch is now compatible without sorting so I'll leave it as is, as it's educational.

2020/05/24
+ Save train intermittently
+ The paper claimed training ends after 100% accuracy is attained on the test-set.
+ I'm using an extraordinarily slow LR so 5000 is totally overkill on the number of epochs
I noticed some weird behavior at certain losses; 

2020/05/21
+ Learning rate is too aggressive; employed a schedular to modulate it over time.
Using LambdaLR
+ Added torch random-seed

2020/05/20
+ Basic LSTM skeleton completed
+ Loss function cross-entropy
+ Note, this is a "2-character language" s.t. 1 input feature is required
for the LSTM ("is A/is not A"). This should be extended for the 3-character test.
+ *** Aside *** 2 inputs [1, 0] == a; [0, 1] == b; [0, 0] == c

"""
import os
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

# For variable length sequences
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

#Saving
import joblib as jb

# ---------------- #
# Language model (number of unique characters)
alphabet = ['a', 'b']

# Define the conversion dictionary
d = {letter: idx for idx, letter in enumerate(alphabet)}
invd = {value:key for key, value in d.items()}

# ---------------- #
# Classes/Models
def encode_sample(x, translate=d):
    """
    Encode a sequence, described by dictionary "d"
    into a compatible input tensor.

    """
    count = np.zeros(shape=(len(x), len(d)))
    t = [(i, j) for i, j in enumerate(list(map(lambda i: translate[i], x[:])))]
    count[tuple(zip(*t))]=1
    return torch.tensor(count)


def get_data(idir, 
             ptrain=0.7, 
             rseed=1234, 
             bfirst=True,
             encoder=encode_sample):
    """
    Retrieve the dataset of interest
    """
    if idir is not None:
        datadir = idir
    else:
        datadir = 'data/'


    with open(os.path.join(datadir, 'positive_lang.txt'), 'r') as f:
        Xpos = f.readlines()

    with open(os.path.join(datadir, 'negative_lang.txt'), 'r') as f:
        Xneg = f.readlines()

    Xpos, Xneg = [i.strip('\n') for i in Xpos], [i.strip('\n') for i in Xneg]

    Npos = len(Xpos)
    Nneg = len(Xneg)

    Ntrainpos = int(Npos * ptrain)
    Ntrainneg = int(Nneg * ptrain)

    Ntestpos = int(Npos - Ntrainpos)
    Ntestneg = int(Nneg - Ntrainneg)

    xtrain = Xpos[:int(Ntrainpos)] + Xneg[:int(Ntrainneg)]
    xtest  = Xpos[int(Ntrainpos):] + Xneg[int(Ntrainneg):]

    # Convert to OHE
    xtrain = tuple(map(lambda x: encode_sample(x), xtrain))
    xtest  = tuple(map(lambda x: encode_sample(x), xtest))

    xtrain_lens = list(map(lambda i: len(i), xtrain))
    xtest_lens  = list(map(lambda i: len(i), xtest))

    ytrain = torch.cat((torch.ones(Ntrainpos, dtype=torch.long), torch.zeros(Ntrainneg, dtype=torch.long)))
    ytest  = torch.cat((torch.ones(Ntestpos, dtype=torch.long), torch.zeros(Ntestneg, dtype=torch.long)))

    # Order should be [Time Steps x N_samples x N_features]
    X  = pad_sequence(xtrain, padding_value=0)
    Xt = pad_sequence(xtest, padding_value=0)

    Xpacked  = pack_padded_sequence(X, 
                                    xtrain_lens, 
                                    batch_first=bfirst, 
                                    enforce_sorted=False).float()
    
    Xtpacked = pack_padded_sequence(Xt, 
                                    xtest_lens, 
                                    batch_first=bfirst, 
                                    enforce_sorted=False).float()

    return xtrain, xtest, Xpacked, Xtpacked, ytrain, ytest


class LSTM_model(nn.Module):
    """
    Counting LSTM (RNN) Model Class.

    Args:
    input_dim - number of input features
    hidden_dim - number of hidden nodes
    batch_size - size of the batch 
    output_dim - number of output predictors
    num_layers - number of LSTM layers
    bfirst - If True, then [N_samples x N_timesteps x N_features]
    """
    def __init__(self,
                 input_dim, 
                 hidden_dim=10,
                 output_dim=2,
                 num_layers=1,
                 bfirst=False):

        super(LSTM_model, self).__init__()

        #Initialize parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.batch_first = bfirst

        #Initialize LSTM layer(s)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=self.batch_first)

        # Initialize readout/output layer (classification step)
        self.hidden2out = nn.Linear(self.hidden_dim, output_dim)

        # Softmax output to classify
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, xdata, batch_size):
        """
        2020.05.25- packed padded sequence approach

        Forward pass through NN; 
        """
        # Initialize the hidden states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_()

        # Outputs
        out, (hn, cn) = self.lstm(xdata, (h0.detach(), c0.detach()))

        #Note, hn[-1] == out[:, -1, :]
        y_out = self.hidden2out(hn[-1])
        y_pred = torch.exp(self.softmax(y_out))

        return y_out, y_pred, out, hn, cn
 

def train(model, 
          parameters, 
          learning_rate,
          lr_fxn,
          loss_fxn, 
          num_epochs, 
          x, 
          y, 
          xtest, 
          ytest,
          target_accuracy=.9999,
          savefreq=100):
    """
    Define a model, and train it.

    Inputs:
    model - type of model
    optimiser - update weights
    loss_fxn - criteria to optimize on
    num_epochs - number of training epochs
    x - training set
    y - training labels
    xtest - 'validation' set
    ytest - 'validation' outputs
    target_accuracy - end training if model performs better than this accuracy
    savefreq - number of iterations to save after
    """
    # Set up the scheduler, and parameters
    optimiser = torch.optim.SGD(parameters, lr=learning_rate)
    scheduler = LambdaLR(optimiser, lr_lambda=lr_fxn)

    losses = []

    xout, _ = pad_packed_sequence(xtrain)
    xoutt, _ = pad_packed_sequence(xtest)
    if model.batch_first:
        batch_train = xout.size(0)
        batch_test = xoutt.size(0)
    else:
        batch_train = xout.size(1)
        batch_test = xoutt.size(1)

    for t in range(num_epochs):

        yout, _, out, hn, cn = model(x, batch_train)

        loss = loss_fxn(yout, y)

        # Get the gradients
        loss.backward()

        # Update parameters
        optimiser.step()

        #ypred = torch.argmax(ypred, axis=1)
        #accuracy = 1 - torch.abs(ypred - y).sum().item()/y.size(0)

        _, ypred, _, _, _ = model(xtest, batch_test)
        ypred = torch.argmax(ypred, axis=1)
        accuracy = 1 - torch.abs(ypred - ytest).sum().item()/ytest.size(0) # Count the number mis-matched

        # Update the learning rate
        scheduler.step()

        # Print the learning rate
        lr_curr = scheduler.get_lr()[0]

        # Output loss
        print('Iteration: {}; Loss: {}; Test Accuracy: {}'.format(t+1, loss.item(), accuracy))
        #print('Iteration: {}; Loss: {}; Test Accuracy: {}; LR: {}'.format(t+1, loss.item(), accuracy, lr_curr))

        losses.append(loss.item())

        # If you reach the target criteria on test-set, end early
        if accuracy >= target_accuracy:
            break

        if (t%savefreq) == 0:
            torch.save(model, 'tmpmodel.pt')

    return model, optimiser, losses


def classify(ypred):
    """Classify the data"""
    return torch.argmax(ypred, axis=1)

def accuracy(ypred, y):
    """Calculate the accuracy"""
    yp = classify(ypred)
    accuracy = 1 - torch.abs(yp - y).sum().item()/y.size(0)
    return accuracy

def lr_rate(epoch, gamma=0.5):
    """Dynamic learning rate adjustment"""
    return 1/(1+gamma*epoch)

# ---------------- #


