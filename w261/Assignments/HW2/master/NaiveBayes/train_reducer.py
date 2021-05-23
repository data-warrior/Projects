#!/usr/bin/env python
"""
Reducer aggregates word counts by class and emits frequencies.
INPUT:            
    partitionKey \t word \t class0_partialCount,class1_partialCount # <--- SOLUTION --->
OUTPUT:                                                             # <--- SOLUTION --->
    WORD \t ham_count,spam_count,P(ham|word),P(spam|word)           # <--- SOLUTION --->
    
INPUT:
    <specify record format here>
OUTPUT:
    <specify record format here>
    
Instructions:
    Again, you are free to design a solution however you see 
    fit as long as your final model meets our required format
    for the inference job we designed in Question 8. Please
    comment your code clearly and concisely.
    
    A few reminders: 
    1) Don't forget to emit Class Priors (with the right key).
    2) In python2: 3/4 = 0 and 3/float(4) = 0.75
"""
##################### YOUR CODE HERE ####################
import sys                                                  # <--- SOLUTION --->
import numpy as np                                          # <--- SOLUTION --->
                                                            # <--- SOLUTION --->
# helper function to emit records correctly formatted       # <--- SOLUTION --->
def EMIT(*args):                                            # <--- SOLUTION --->
    print('{}\t{},{},{},{}'.format(*args))                  # <--- SOLUTION --->
                                                            # <--- SOLUTION --->    
# initialize trackers [ham, spam]                           # <--- SOLUTION --->
docTotals = np.array([0.0,0.0])                             # <--- SOLUTION --->
wordTotals = np.array([0.0,0.0])                            # <--- SOLUTION --->
cur_word, cur_counts = None, np.array([0,0])                # <--- SOLUTION --->
                                                            # <--- SOLUTION --->
# read from standard input                                  # <--- SOLUTION --->
for line in sys.stdin:                                      # <--- SOLUTION --->
    part, wrd, counts = line.split()                              # <--- SOLUTION --->
    counts = [int(c) for c in counts.split(',')]            # <--- SOLUTION --->
                                                            # <--- SOLUTION --->    
    # store totals, add or emit counts and reset            # <--- SOLUTION ---> 
    if wrd == "*docTotals":                                 # <--- SOLUTION ---> 
        docTotals += counts                                 # <--- SOLUTION --->
    elif wrd == "*wordTotals":                              # <--- SOLUTION ---> 
        wordTotals += counts                                # <--- SOLUTION --->        
    elif wrd == cur_word:                                   # <--- SOLUTION --->
        cur_counts += counts                                # <--- SOLUTION --->
    else:                                                   # <--- SOLUTION --->
        if cur_word:                                        # <--- SOLUTION --->
            freq = cur_counts / wordTotals                  # <--- SOLUTION --->
            EMIT(cur_word, *tuple(cur_counts)+tuple(freq))  # <--- SOLUTION --->
        cur_word, cur_counts  = wrd, np.array(counts)       # <--- SOLUTION --->
                                                            # <--- SOLUTION --->
# last record                                               # <--- SOLUTION ---> 
EMIT(cur_word, *tuple(cur_counts)+tuple(cur_counts/wordTotals))  # <--- SOLUTION --->
                                                            # <--- SOLUTION --->
# class priors                                              # <--- SOLUTION ---> 
priors = tuple(docTotals)+tuple(docTotals/sum(docTotals))   # <--- SOLUTION --->
EMIT('ClassPriors', *priors)                                # <--- SOLUTION --->
##################### (END) CODE HERE ####################