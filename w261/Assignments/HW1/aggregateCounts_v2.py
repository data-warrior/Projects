#!/usr/bin/env python
"""
This script reads word counts from STDIN and aggregates
the counts for any duplicated words.

INPUT & OUTPUT FORMAT:
    word \t count
USAGE (standalone):
    python aggregateCounts_v2.py < yourCountsFile.txt

Instructions:
    For Q7 - Your solution should not use a dictionary or store anything   
             other than a single total count - just print them as soon as  
             you've added them. HINT: you've modified the framework script 
             to ensure that the input is alphabetized; how can you 
             use that to your advantage?
"""

# imports
import sys


################# YOUR CODE HERE #################

# stream over lines from Standard Input
word_register, count_register = sys.stdin.readline().split()
for line in sys.stdin:
    # extract words & counts
    word, count = line.split()
    
    if word == word_register:
        count_register = int(count_register) + int(count)
    else:
        print("{}\t{}".format(word_register,count_register))
        count_register = int(count)
        word_register = word
print("{}\t{}".format(word_register,count_register))

################ (END) YOUR CODE #################
