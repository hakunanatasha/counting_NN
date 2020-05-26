"""
2020 May 24

The following script intends to dissect how the model learns the language type. The claim in the paper is that one of the hidden units becomes a 'counter'. 

Possible experiments:

1. What is the minimum training required to exhibit the counting behavior?

2. If you train a model on size "N", how far beyond N can it learn to generalize (i.e the N=100 model learns to count and predict on an N=256 model.). How does the performance change?

3. What are the counting patterns for non-positive cases? (i.e the sequences that are not of form a^N*b^N)
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

# Plotting
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
#Mac os bs
matplotlib.use('Qt5Agg') 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
 
# ------------- #

# User Parameters
idir = None
ptrain = 0.8
tstep = 200

mname = 'model/counting_seq' + str(tstep) + '.pt'
hidden_dim = 10
sdir = "figures/"

# Load the data
xseq_train, xseq_test, xtrain, xtest, ytrain, ytest = get_data(idir, ptrain)

# Load the model
model = torch.load(mname)


# ----------------- #
# Get an instance of a positive, or negative pred

# It will expect an Nsamples x Ntimesteps input
xpos = torch.
#xpos = xtrain[0].view(1, tstep)
#xneg = xtrain[-1].view(1, tstep)

cn = np.zeros(shape = (2, tstep, hidden_dim))
# The class output 0 ==> "Not the language". Class output 1 ==> "Language"
# Note, pytorch is a pain so i need to feed every sub-sequence to get the cellstate.
for i in range(tstep):
    xp = xpos[:, :(i+1)]
    xn = xneg[:, :(i+1)]
    _, _, _, _, cn_pos = model(xp)
    _, _, _, _, cn_neg = model(xn)
    #yp = torch.argmax(ypred, axis=1)
    cn[0, i, :] = cn_pos.view(-1).tolist()
    cn[1, i, :] = cn_neg.view(-1).tolist()

# ---------------- #
# Plot
alpha=0.5
cmap = matplotlib.cm.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 1, hidden_dim)]
legs = [Line2D([0,0,0], [0,0,0], 
        color=colors[i], 
        linestyle='-', 
        lw = 4) for i in range(hidden_dim)]

# Plot the counting behavior
plt.close('all')
f = plt.figure(figsize=(12,20))
ax = f.add_subplot(2,1,1)
ax2 = f.add_subplot(2,1,2)

for i in range(hidden_dim):
    ax.plot(np.arange(tstep)+1, cn[0, :, i], color=colors[i], linewidth=3, alpha=alpha)

for i in range(hidden_dim):
    ax2.plot(np.arange(tstep)+1, cn[1, :, i], color=colors[i], linewidth=3, alpha=alpha)

ax.set_xlabel("Sequence position")
ax.set_ylabel("Activation")
ax.legend(legs, ["Unit" + str(i+1) for i in range(hidden_dim)])
ax.set_title("Activation of Hidden Unit-Positive case")

ax2.set_xlabel("Sequence position")
ax2.set_ylabel("Activation")
ax2.legend(legs, ["Unit" + str(i+1) for i in range(hidden_dim)])
ax2.set_title("Activation of Hidden Unit-Negative case")

f.tight_layout()
f.savefig(sdir + "cell_state" + str(tstep) + ".png")

