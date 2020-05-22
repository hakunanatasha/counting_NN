"""
Date: 2020 May 20
Author: Natasha Seelam

Teaching an LSTM to count using sequences of a^N x b ^N.

Paper: https://arxiv.org/pdf/1805.04908.pdf

Note, the parameters are 10-dimensin 1-layer LSTM. Activation - ReLU

The claims of the paper include:
(1) LSTMS can recognize 'languages' of the form a^N x b^N etc.
(2) LSTMS trained on this can *generalize* to other counting forms (i.e. N=100 -> N=256)
(3) Trained LSTMs hidden layers show this counting mechanism in one of their hidden unit.
"""
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import numpy as np

torch.manual_seed(1)
np.random.seed(1)
# ---------------- #
# Language model (number of unique characters)
alphabet = ['a', 'b']

# Define the conversion dictionary
d = {letter: idx for idx, letter in enumerate(alphabet)}
invd = {value:key for key, value in d.items()}

# Classes/Models
def encode_sample(x, translate=d):
    """
    Encode a sequence, described by dictionary "d"
    into a compatible input tensor
    """
    return torch.tensor(list(map(lambda i: translate[i], x[:])))


def get_data(ldir, ptrain=0.7, rseed=1234, encoder=encode_sample):
    """
    Retrieve the dataset of interest
    """
    with open(ldir + 'data/positive_lang.txt', 'r') as f:
        Xpos = f.readlines()

    with open(ldir + 'data/negative_lang.txt', 'r') as f:
        Xneg = f.readlines()

    Xpos, Xneg = [i.strip('\n') for i in Xpos], [i.strip('\n') for i in Xneg]

    Npos = len(Xpos)
    Nneg = len(Xneg)

    Ntrainpos = int(Npos * ptrain)
    Ntrainneg = int(Nneg * ptrain)

    Ntestpos = int(Npos * (1- ptrain))
    Ntestneg = int(Nneg * (1- ptrain))

    xtrain = Xpos[:int(Ntrainpos)] + Xneg[:int(Ntrainneg)]
    xtest  = Xpos[int(Ntrainpos):] + Xneg[int(Ntrainneg):]

    ytrain = torch.cat((torch.ones(Ntrainpos, dtype=torch.long), torch.zeros(Ntrainneg, dtype=torch.long)))
    ytest  = torch.cat((torch.ones(Ntestpos, dtype=torch.long), torch.zeros(Ntestneg, dtype=torch.long)))

    # Order should be [Time Steps x N_samples x N_features]
    X  = torch.stack(tuple(map(lambda x: encode_sample(x), xtrain)), 0).float()
    Xt = torch.stack(tuple(map(lambda x: encode_sample(x), xtest)), 0).float()

    return xtrain, xtest, X, Xt, ytrain, ytest


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
                 bfirst=True):

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

    def forward(self, x):
        """
        Forward pass through NN; 
        """
        if len(x.size()) < 3:
            x = x.view([x.size(0), x.size(1), 1])

        if self.batch_first:
            batch_size = x.shape[0]
        else:
            batch_size = x.shape[1]

        # Initialize the hidden states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_()

        # Outputs
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

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
          target_accuracy=.9999):
    """
    Define a model, and train it.

    Inputs:
    model - type of model
    optimiser - update weights
    loss_fxn - criteria to optimize on
    num_epochs - number of training epochs
    x - training set
    y - training labels
    """
    # Set up the scheduler, and parameters
    optimiser = torch.optim.SGD(parameters, lr=learning_rate)
    scheduler = LambdaLR(optimiser, lr_lambda=lr_fxn)

    losses = []
    for t in range(num_epochs):

        yout, _, out, hn, cn = model(x)

        loss = loss_fxn(yout, y)

        # Get the gradients
        loss.backward()

        # Update parameters
        optimiser.step()

        #ypred = torch.argmax(ypred, axis=1)
        #accuracy = 1 - torch.abs(ypred - y).sum().item()/y.size(0)

        _, ypred, _, _, _ = model(xtest)
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

    return losses


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

# ---------------- #

if __name__ == '__main__':

    # User Parameters
    ldir = ''
    num_epochs = 1000
    learning_rate = 0.0001
    gamma = 0.1
    input_dim = 1
    lr_fxn = lambda x: lr_rate(x, gamma)

    sname = 'model/countingLSTM_AB.pt'
    # -------------- #
    #    DATASET
    # -------------- #

    # Load the data
    xseq_train, xseq_test, xtrain, xtest, ytrain, ytest = get_data(ldir)

    # -------------- #
    #  MODEL SETUP  
    # -------------- #

    # Initialize model, and optimiser

    model = LSTM_model(input_dim)

    # Optimizer - set to all modifiable/learnable parameters
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    # Loss Function (Categorical/X-ent)
    criterion = nn.CrossEntropyLoss()

    # -------------- #
    #  MODEL TRAIN
    # -------------- #

    # Train model
    model, optimiser, loss = train(model, 
                                   parameters, 
                                   learning_rate,
                                   lr_fxn,
                                   criterion, 
                                   num_epochs, 
                                   xtrain, 
                                   ytrain,
                                   xtest,
                                   ytest)

    # Save torch model
    torch.save(model, sname)

