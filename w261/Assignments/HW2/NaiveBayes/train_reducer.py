#!/usr/bin/env python
"""
Reducer aggregates word counts by class and emits frequencies.

INPUT:
    partitionKey \t word \t class0_partialCount \t class1_partialCount       
OUTPUT:
    word \t class0_wordcount \t class1_wordcount \t class0_relfreq \t class1_relfreq
    
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
import sys

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
        
        if cur_word == '**total_terms': 
            total_terms_0 += int(cur_count_0)
            total_terms_1 += int(cur_count_1)
            total_terms = total_terms_0 + total_terms_1
            
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
                print(f'{cur_word}\t{cur_count_0},{cur_count_1},{cur_count_0/total_terms_0:.15f},{cur_count_1/total_terms_1:.15f}')
            
#        # and start a new tally
        cur_pk, cur_word, cur_count_0, cur_count_1 = pk, key, int(value_0), int(value_1)

#
## don't forget the last record! 
print(f'{cur_word}\t{cur_count_0},{cur_count_1},{cur_count_0/total_terms_0:.15f},{cur_count_1/total_terms_1:.15f}')
#print(f'ClassPriors\t{total_docs_0}\t{total_docs_1}\t{total_docs_0/total_docs}\t{total_docs_1/total_docs}')









##################### (END) CODE HERE ####################