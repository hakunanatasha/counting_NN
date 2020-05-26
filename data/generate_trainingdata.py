"""
2020 Mar 03

Generates the training data of a simple language.

The language will be various permutations of "a", "b" in a sequence
The true language will be a concurrent sequence of N "a" in a row,
and then N "b" in a row, as such:

ex: N = 5
'aaaaabbbbb' => True
'aaabbaaabb' => False

TODO:
Check if the negative-set accidentally gets a positive case and swap
"""

import numpy as np
from numpy.random import rand
import os

# Save data if the directory doesn't already exist
savedir = "data/"
np.random.seed(1) # Reproducibility in the language 

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

Nlen = 100 #1/2 length of sequence (input)

Ntrue = 7000  # Training of positive class

Nfalse = 7000 # Training of negative class

# Generate the "true" language
pos = list(map(lambda x: define_seq(Nlen)+"\n", range(Ntrue)))

# Generate variable length sequences for false case
neg = list(map(lambda x: define_seq(np.random.randint(1, Nlen), pgen=[0.5, 0.5]) + "\n", range(Nfalse)))

# -------------------------- #
with open(os.path.join(savedir + "positive_lang.txt"), "w") as f:
    f.writelines(pos)

with open(os.path.join(savedir + "negative_lang.txt"), "w") as f:
    f.writelines(neg)

print("Completed language data.")