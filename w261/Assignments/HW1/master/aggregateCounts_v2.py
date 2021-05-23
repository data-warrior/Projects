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
current_word = None                                     # <---SOLUTION--->
current_count = 0                                       # <---SOLUTION--->
for line in sys.stdin:                                  # <---SOLUTION--->
    word, count = line.split()                          # <---SOLUTION--->
    if current_word is None or word == current_word:    # <---SOLUTION--->
        current_count += int(count)                     # <---SOLUTION--->
    elif current_word is not None:                      # <---SOLUTION--->
        print(f"{current_word}\t{current_count}")       # <---SOLUTION--->
        current_count = int(count)                      # <---SOLUTION--->
    current_word = word                                 # <---SOLUTION--->
# don't forget the last word!                           # <---SOLUTION--->
print(f"{current_word}\t{current_count}")               # <---SOLUTION--->
################ (END) YOUR CODE #################
