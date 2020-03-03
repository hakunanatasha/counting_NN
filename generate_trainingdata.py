"""
2020 Mar 03

Generates the training data of a simple language.

The language will be various permutations of "a", "b" in a sequence
The true language will be a concurrent sequence of N "a" in a row,
and then N "b" in a row, as such:

ex: N = 5
'aaaaabbbbb' => True
'aaabbaaabb' => False
"""

import numpy as np
from numpy.random import rand
import os

# Define sequences

def define_seq(Nlen, pgen=[1, 0]):
    """
    Create a string where the likelihood of generating
    a certain sequence in a row is high.

    Nlen - length of sequence
    pgen - likelihood of generating 'a' for seq 1, 2
    """
    rands_a = rand(Nlen)
    rands_b = rand(Nlen)
    a_s = list(map(lambda x: 'a' if x < pgen[0] else 'b', rands_a))
    b_s = list(map(lambda x: 'a' if x < pgen[1] else 'b', rands_b))
    return "".join(a_s) + "".join(b_s)


# -------------------------- #
# Generate training data 

Nlen = 25 #1/2 length of sequence (input)

Ntrue = 5000  # Training of positive class

Nfalse = 5000 # Training of negative class

pos = list(map(lambda x: define_seq(Nlen)+"\n", range(Ntrue)))
neg = list(map(lambda x: define_seq(Nlen, pgen=[rand(1), rand(1)]) + "\n", range(Nfalse)))

# -------------------------- #
# Save data if the directory doesn't already exist
savedir = "/Users/NSEELAM/Dropbox (MIT)/Classes_2020/NS_Projects/counting_NN/"

if os.path.exists(savedir + "data") is False:
    
    os.makedirs(savedir + "data/")

    with open(savedir + "data/positive_lang.txt", "w") as f:
        f.writelines(pos)

    with open(savedir + "data/negative_lang.txt", "w") as f:
        f.writelines(neg)