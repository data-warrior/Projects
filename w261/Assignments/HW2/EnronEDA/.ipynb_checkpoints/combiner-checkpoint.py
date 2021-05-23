#!/usr/bin/env python
"""
Reducer takes words with their class and partial counts and computes totals.
INPUT:
    word \t class \t partialCount 
OUTPUT:
    word \t class \t totalCount  
"""
import re
import sys

# initialize trackers
current_word = None
spam_count, ham_count = 0,0

# read from standard input
for line in sys.stdin:
    # parse input
    word, is_spam, count = line.split('\t')
    
    

    if current_word is None or word == current_word:
        if int(is_spam):
            spam_count += int(count)
        else:
            ham_count += int(count)
    
    elif current_word is not None:
        print(f"{current_word}\t{1}\t{spam_count}")            
        print(f"{current_word}\t{0}\t{ham_count}")        
        if int(is_spam):
            spam_count = int(count)
            ham_count = 0
        else:
            ham_count = int(count)
            spam_count = 0
    current_word = word
print(f"{current_word}\t{1}\t{spam_count}")            
print(f"{current_word}\t{0}\t{ham_count}")      
############ YOUR CODE HERE #########













############ (END) YOUR CODE #########