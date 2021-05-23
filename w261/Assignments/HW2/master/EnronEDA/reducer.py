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
    
############ YOUR CODE HERE #########
    # tally counts from current key                                     # <--- SOLUTION --->
    if word == current_word:                                            # <--- SOLUTION --->
        spam_count += int(count) * int(is_spam == '1')                  # <--- SOLUTION --->
        ham_count += int(count) * int(is_spam == '0')                   # <--- SOLUTION --->
    # OR emit current total and start a new tally                       # <--- SOLUTION --->
    else:                                                               # <--- SOLUTION --->
        if current_word:                                                # <--- SOLUTION --->
            print(f'{current_word}\t{1}\t{spam_count}')                  # <--- SOLUTION --->
            print(f'{current_word}\t{0}\t{ham_count}')                   # <--- SOLUTION --->
        current_word = word                                             # <--- SOLUTION --->
        spam_count = int(count) * int(is_spam == '1')                   # <--- SOLUTION --->
        ham_count = int(count) * int(is_spam == '0')                    # <--- SOLUTION --->
# don't forget the last record!                                         # <--- SOLUTION --->
print(f'{current_word}\t{1}\t{spam_count}')                             # <--- SOLUTION --->
print(f'{current_word}\t{0}\t{ham_count}')                              # <--- SOLUTION --->
############ (END) YOUR CODE #########