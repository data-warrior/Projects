#!/usr/bin/env python
"""
Mapper tokenizes and emits words with their class.
INPUT:
    ID \t SPAM \t SUBJECT \t CONTENT \n
OUTPUT:
    word \t class \t count 
"""
import re
import sys
from collections import defaultdict

# read from standard input
for line in sys.stdin:
    # parse input
    docID, _class, subject, body = line.split('\t')
    # tokenize
    words = re.findall(r'[a-z]+', subject + ' ' + body)
    
############ YOUR CODE HERE - Map ####
#    for word in words:
#        print(f"{word}\t{_class}\t{1}")
############ (END) YOUR CODE #########

############ YOUR CODE HERE - Map ####
    counts = defaultdict(int)
    for word in words:
        counts[word] += 1
    for key in counts:
        print(f"{key}\t{_class}\t{counts[key]}")
############ (END) YOUR CODE #########