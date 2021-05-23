#!/usr/bin/env python

import sys                                                  
import numpy as np  

#################### YOUR CODE HERE ###################

# helper function to emit records correctly formatted
def EMIT(*args):
    print('{}\t{},{},{},{}'.format(*args))

# Laplace Smoothing Parameters
#V = 5065.0  # Enron vocab size
#V = 4555.0 # Enron training set vocab size
V = 6.0  # China vocab size
k = 1
  
# initialize trackers [ham, spam]
docTotals = np.array([0.0,0.0])
wordTotals = np.array([0.0, 0.0])
cur_word, cur_counts = None, np.array([0,0])

# read from standard input
for line in sys.stdin:
    key, wrd, counts = line.split()
    counts = [int(c) for c in counts.split(',')]

    # store totals, add or emit counts and reset 
    if wrd == "*docTotals": 
        docTotals += counts
    elif wrd == "*wordTotals": 
        wordTotals += counts        
    elif wrd == cur_word:
        cur_counts += counts
    else:
        if cur_word:
            # LAPLACE MODIFICATION HERE 
            freq = (cur_counts + [k,k])/(wordTotals + [V,V])
            EMIT(cur_word, *tuple(cur_counts)+tuple(freq))
        cur_word, cur_counts  = wrd, np.array(counts)

# last record  (LAPLACE MODIFICATION HERE) 
freq = (cur_counts + [k,k])/(wordTotals + [V,V])
EMIT(cur_word, *tuple(cur_counts)+tuple(freq))

# class priors
priors = tuple(docTotals) + tuple(docTotals/sum(docTotals))
EMIT('ClassPriors', *priors)

#################### (END) YOUR CODE ###################