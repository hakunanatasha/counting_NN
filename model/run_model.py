"""
2020 May 25

Splitting the model execution script into a new progam.
"""
import os
import sys

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import joblib as jb

sys.path.insert(0, 'model')
from LSTM import *

# Reproducibility in the model:
torch.manual_seed(1)
np.random.seed(1)

if __name__ == '__main__':

    # -------------- #
    # User Parameters
    # -------------- #
    # Language/Dataset
    N = 100 
    idir = None
    ptrain = 0.8

    # Model parameters
    hidden_dim = 10
    output_dim = 2
    num_layers = 1
    batch_first = False

    # Training parameters
    num_epochs = 5000
    learning_rate = 0.5
    gamma = 0.01
    input_dim = 2
    lr_fxn = lambda x: lr_rate(x, gamma)
    rseed = 1234

    sname = 'model/counting_' + str(N) + "_model.pt"
    # -------------- #
    #    Dataset
    # -------------- #
    # Load the data
    xseq_train, xseq_test, xtrain, xtest, ytrain, ytest = get_data(idir, ptrain, rseed, batch_first)
    print("01. Loading dataset. - Complete")
    # -------------- #
    #  Model Setup 
    # -------------- #
    # Initialize model, and optimiser

    model = LSTM_model(input_dim, bfirst=False)

    # Optimizer - set to all modifiable/learnable parameters
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    # Loss Function (Categorical/X-ent)
    criterion = nn.CrossEntropyLoss()
    print("\n02. Intializing model. - Complete")

    # -------------- #
    #  Model Training
    # -------------- #
    # Train model
    print("\n03. Training model\n")
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
    print("\n Training complete.")

    # Save torch model
    print("\n04. Saving models.")
    torch.save(model, sname)
    jb.dump(loss, 'model/losses' + str(N) + '.jb')

