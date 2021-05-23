#!/usr/bin/env python
"""
Reducer aggregates word counts by class and emits frequencies.

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
import sys
cur_word = None
cur_count = 0
cur_count_0 = 0
cur_count_1 = 0
cur_pk = ""
cur_class = 0
total = 0

# read input key-value pairs from standard input
for line in sys.stdin:
    ########## UNCOMMENT & MODIFY AS NEEDED BELOW #########
    pk, key, value, kclass = line.split()
#    # tally counts from current key

    if key == cur_word:
        #print("Enter", key, kclass, cur_word, cur_class, cur_count, cur_count_0, cur_count_1)

            
        cur_count += int(value)
        
        if int(kclass):
            cur_count_1 += int(value)
        else:
            cur_count_0 += int(value)
        
        if cur_word == '**total': 
            total += cur_count


            
        #print("Exit", key, kclass, cur_word, cur_class, cur_count, cur_count_0, cur_count_1)
    else:
        if cur_word:
            if cur_word == '**total': 
                total = int(cur_count)
                cur_count_0 = 0
                cur_count_1 = 0
            print(f'{cur_pk}\t{cur_word}\t{cur_count}\t{cur_count_0}\t{cur_count_1}')
            
#        # and start a new tally
        cur_pk, cur_word, cur_count, cur_class = pk, key, int(value), int(kclass)
    
        if cur_class:
            cur_count_1 = cur_count
            cur_count_0 = 0
        else:
            cur_count_0 = cur_count
            cur_count_1 = 0

#
## don't forget the last record! 
print(f'{cur_pk}\t{cur_word}\t{cur_count}\t{cur_count_0}\t{cur_count_1}')









##################### (END) CODE HERE ####################