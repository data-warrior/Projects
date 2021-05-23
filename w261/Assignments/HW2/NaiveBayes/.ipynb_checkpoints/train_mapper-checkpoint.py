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
from collections import defaultdict
#################### YOUR CODE HERE ###################
num_reducers = 1
if "mapreduce_job_reduces" in os.environ:
    num_reducers = int(os.environ["mapreduce_job_reduces"])


def getPartitionKey(word, num_reducer):
    "Helper function to assign partition key alphabetically."
    ############## YOUR CODE HERE ##############

    if num_reducer == 1:
        return 'A'
    elif num_reducer == 2:
        if word[0] < chr(97+(26//num_reducer)): 
            return  'A'
        else:
            return 'B'                   
    elif num_reducer == 3:
        if word[0] < chr(97+(26//num_reducer)): 
            return  'A'
        elif word[0] < chr(97+2*(26//num_reducer)):
            return 'B'
        else:
            return 'C'
    else:
        return 'A'
                           
        
    ############## (END) YOUR CODE ##############
    
    
#def getPartitionKey(word):
#    "Helper function to assign partition key alphabetically."
#    ############## YOUR CODE HERE ##############
#    cuts = 26//num_reducers
#    
#    if word[0] < 'h': 
#        return  'A'
#    elif word[0] < 'p':
#        return  'B'
#    else:
#        return 'C'
    ############## (END) YOUR CODE ##############
    
    
    
TOTAL_WORDS = 0
TOTAL_WORDS_0 = 0
TOTAL_WORDS_1 = 0

TOTAL_DOCS = 0
TOTAL_DOCS_0 = 0
TOTAL_DOCS_1 = 0

# read from standard input
for line in sys.stdin:
    # parse input
    docID, _class, subject, body = line.split('\t')
    # tokenize
    words = re.findall(r'[a-z]+', (subject + ' ' + body).lower())
    counts = defaultdict(int)
    TOTAL_DOCS += 1
    if int(_class):
        TOTAL_DOCS_1 += 1
    else:
        TOTAL_DOCS_0 += 1
    for word in words:
        counts[word] += 1
        TOTAL_WORDS += 1
        if int(_class):
            pkey = getPartitionKey(word, num_reducers)
            print(f"{pkey}\t{word}\t{0}\t{1}")
            TOTAL_WORDS_1 += 1
        else:
            pkey = getPartitionKey(word, num_reducers)
            print(f"{pkey}\t{word}\t{1}\t{0}")
            TOTAL_WORDS_0 += 1
            
for i in range(num_reducers):
    part_name = chr(65+i)
    print(f'{part_name}\t**total_terms\t{TOTAL_WORDS_0}\t{TOTAL_WORDS_1}')

    
#print(f'A\t**total_terms\t{TOTAL_WORDS_0}\t{TOTAL_WORDS_1}')
#print(f'B\t**total_terms\t{TOTAL_WORDS_0}\t{TOTAL_WORDS_1}')
#print(f'C\t**total_terms\t{TOTAL_WORDS_0}\t{TOTAL_WORDS_1}')

print(f'A\t**total_docs\t{TOTAL_DOCS_0}\t{TOTAL_DOCS_1}')
#print(f'B\t**total_docs\t{TOTAL_DOCS_0}\t{TOTAL_DOCS_1}')
#print(f'C\t**total_docs\t{TOTAL_DOCS_0}\t{TOTAL_DOCS_1}')
############ (END) YOUR CODE #########



























#################### (END) YOUR CODE ###################