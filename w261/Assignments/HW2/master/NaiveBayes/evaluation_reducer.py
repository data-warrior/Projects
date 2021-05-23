#!/usr/bin/env python
"""
Reducer to calculate precision and recall as part
of the inference phase of Naive Bayes.
INPUT:
    ID \t true_class \t P(ham|doc) \t P(spam|doc) \t predicted_class
OUTPUT:
    precision \t ##
    recall \t ##
    accuracy \t ##
    F-score \t ##
         
Instructions:
    Complete the missing code to compute these^ four
    evaluation measures for our classification task.
    
    Note: if you have no True Positives you will not 
    be able to compute the F1 score (and maybe not 
    precision/recall). Your code should handle this 
    case appropriately feel free to interpret the 
    "output format" above as a rough suggestion. It
    may be helpful to also print the counts for true
    positives, false positives, etc.
"""
import sys

# initialize counters
FP = 0.0 # false positives
FN = 0.0 # false negatives
TP = 0.0 # true positives
TN = 0.0 # true negatives

# read from STDIN
for line in sys.stdin:
    # parse input
    docID, class_, pHam, pSpam, pred = line.split()
    # emit classification results first
    print(line[:-2], class_ == pred)
    
    # then compute evaluation stats
#################### YOUR CODE HERE ###################
    if class_ == pred:                                     # <--- SOLUTION --->
        TP += int(pred == '1')                             # <--- SOLUTION --->
        TN += int(pred == '0')                             # <--- SOLUTION --->
    else:                                                  # <--- SOLUTION --->
        FP += int(pred == '1')                             # <--- SOLUTION --->
        FN += int(pred == '0')                             # <--- SOLUTION --->
                                                           # <--- SOLUTION --->
# report results                                           # <--- SOLUTION --->
print(f"Total # Documents:\t{TP + TN + FP + FN}")          # <--- SOLUTION --->
print(f"True Positives:\t{TP}")                            # <--- SOLUTION --->
print(f"True Negatives:\t{TN}")                            # <--- SOLUTION --->
print(f"False Positives:\t{FP}")                           # <--- SOLUTION --->
print(f"False Negatives:\t{FN}")                           # <--- SOLUTION --->
print(f"Accuracy\t{(TP + TN)/(TP + TN + FP + FN)}")        # <--- SOLUTION --->
if (TP + FP) != 0:                                         # <--- SOLUTION --->
    precision = TP / (TP + FP)                             # <--- SOLUTION --->
    print(f"Precision\t{precision}")                       # <--- SOLUTION --->
if (TP + FN) != 0:                                         # <--- SOLUTION --->
    recall = TP / (TP + FN)                                # <--- SOLUTION --->
    print(f"Recall\t{recall}")                             # <--- SOLUTION --->
if TP != 0:                                                # <--- SOLUTION --->
    f_score = 2 * precision * recall / (precision + recall)     # <--- SOLUTION --->
    print(f"F-Score\t{f_score}")                               # <--- SOLUTION --->
#################### (END) YOUR CODE ###################
    