#!/usr/bin/env python

import sys
import numpy as np  

#################### YOUR CODE HERE ###################
import os

#defualt value for debugging
vocab_size = 6

#override the default value with the environ value
if "VOCAB_SIZE" in os.environ:
    vocab_size = int(os.environ["VOCAB_SIZE"])


cur_word = None
cur_pk = None

cur_count = 0
cur_count_0 = 0
cur_count_1 = 0

total_terms = 0
total_terms_0 = 0
total_terms_1 = 0

total_docs = 0
total_docs_0 = 0
total_docs_1 = 0

# read input key-value pairs from standard input
for line in sys.stdin:
    ########## UNCOMMENT & MODIFY AS NEEDED BELOW #########
    pk, key, value_0, value_1 = line.split()
#    # tally counts from current key
    if key == cur_word:
        cur_count_0 += int(value_0)
        cur_count_1 += int(value_1)
        # calculate the total number of terms for each class
        if cur_word == '**total_terms': 
            total_terms_0 += int(cur_count_0)
            total_terms_1 += int(cur_count_1)
            total_terms = total_terms_0 + total_terms_1
        # calculate total number of records for each class
        if cur_word == '**total_docs': 
            total_docs_0 += int(cur_count_0)
            total_docs_1 += int(cur_count_1)
            total_docs = total_docs_0 + total_docs_1
                        
    else:
        if cur_word:
            if cur_word == '**total_terms': 
                total_terms_0 = int(cur_count_0)
                total_terms_1 = int(cur_count_1)
                total_terms = total_terms_0 + total_terms_1
            elif cur_word == '**total_docs': 
                total_docs_0 = int(cur_count_0)
                total_docs_1 = int(cur_count_1)
                total_docs = total_docs_0 + total_docs_1
                print(f'ClassPriors\t{total_docs_0},{total_docs_1},{total_docs_0/total_docs},{total_docs_1/total_docs}')
            else:
                print(f'{cur_word}\t{cur_count_0},{cur_count_1},{(cur_count_0+1)/(total_terms_0 + vocab_size):.15f},{(cur_count_1+1)/(total_terms_1+vocab_size):.15f}')
            
#        # and start a new tally
        cur_pk, cur_word, cur_count_0, cur_count_1 = pk, key, int(value_0), int(value_1)

#
## don't forget the last record! 
print(f'{cur_word}\t{cur_count_0},{cur_count_1},{(cur_count_0+1)/(total_terms_0 + vocab_size):.15f},{(cur_count_1+1)/(total_terms_1+vocab_size):.15f}')





















#################### (END) YOUR CODE ###################