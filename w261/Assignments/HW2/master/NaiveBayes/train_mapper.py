#!/usr/bin/env python
"""
Mapper reads in text documents and emits word counts by class.
INPUT:                                                    
    DocID \t true_class \t subject \t body                
OUTPUT:                                                   
    partitionKey \t word \t class0_partialCount,class1_partialCount       
    

Instructions:
    You know what this script should do, go for it!
    (As a favor to the graders, please comment your code clearly!)
    
    A few reminders:
    1) To make sure your results match ours please be sure
       to use the same tokenizing that we have provided in
       all the other jobs:
         words = re.findall(r'[a-z]+', text-to-tokenize.lower())
         
    2) Don't forget to handle the various "totals" that you need
       for your conditional probabilities and class priors.
       
Partitioning:
    In order to send the totals to each reducer, we need to implement
    a custom partitioning strategy.
    
    We will generate a list of keys based on the number of reduce tasks 
    that we read in from the environment configuration of our job.
    
    We'll prepend the partition key by hashing the word and selecting the
    appropriate key from our list. This will end up partitioning our data
    as if we'd used the word as the partition key - that's how it worked
    for the single reducer implementation. This is not necessarily "good",
    as our data could be very skewed. However, in practice, for this
    exercise it works well. The next step would be to generate a file of
    partition split points based on the distribution as we've seen in 
    previous exercises.
    
    Now that we have a list of partition keys, we can send the totals to 
    each reducer by prepending each of the keys to each total.
       
"""

import re                                                   
import sys                                                  
import numpy as np      

from operator import itemgetter
import os

#################### YOUR CODE HERE ###################
if os.getenv('mapreduce_job_reduces') == None:
    N = 1
else:
    N = int(os.getenv('mapreduce_job_reduces'))

# helper functions
def makeKeyHash(key, num_reducers = N):
    """
    Mimic the Hadoop string-hash function.
    
    key             the key that will be used for partitioning
    num_reducers    the number of reducers that will be configured
    """
    byteof = lambda char: int(format(ord(char), 'b'), 2)
    current_hash = 0
    for c in key:
        current_hash = (current_hash * 31 + byteof(c))
    return current_hash % num_reducers

# helper function
def makeKeyFile(num_reducers = N):
    # N = number of reducers
    KEYS = list(map(chr, range(ord('A'), ord('Z')+1)))[:num_reducers]
    partition_keys = sorted(KEYS, key=lambda k: makeKeyHash(k,num_reducers))

    return partition_keys


# call your helper function to get partition keys
pKeys = makeKeyFile()

# initialize class counters                                 
docTotals = np.array([0,0])                                 
wordTotals = np.array([0,0])                                
                                                            
# read from standard input                                 
for line in sys.stdin:                                    
    # parse input and tokenize
    docID, class_, subj, body = line.lower().split('\t')
    words = re.findall(r'[a-z]+', subj + ' ' + body)
    increment = [1,0] if class_ =='0' else [0,1]
                                                                
    # update class counts
    docTotals += increment
    wordTotals += np.array(increment) * len(words)
                                                               
    # emit words with a count for each class (0,1 or 1,0)
    for word in words:
        print("{}\t{}\t{},{}".format(pKeys[makeKeyHash(word)], word, *increment)) 

# finaly, emit totals with partition key for each reducer, as well order inversion
for k in pKeys:
    print("{}\t*docTotals\t{},{}".format(k,*docTotals))
    print("{}\t*wordTotals\t{},{}".format(k,*wordTotals))

#################### (END) YOUR CODE ###################