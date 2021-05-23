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

total = 0

# read from STDIN
for line in sys.stdin:
    # parse input
    docID, class_, pHam, pSpam, pred = line.split()
    # emit classification results first
    print(line[:-2], class_ == pred)
    total += 1
    # then compute evaluation stats
#################### YOUR CODE HERE ###################
    if (int(class_) == 0 and int(pred) == 0):
        TN += 1
    elif (int(class_) == 1 and int(pred) == 1):
        TP += 1
    elif (int(class_) == 0 and int(pred) == 1):
        FP += 1
    else:
        FN += 1
        
accuracy = (TP + TN)/total
precision = TP/(TP + FP) 
recall = TP/(TP + FN)
fscore = (2*precision*recall)/(precision+recall)

print(f'# Document:{total:5.0f}')
print(f'True Positives:{TP:5.0f}')
print(f'True Negatives:{TN:5.0f}')
print(f'False Positives:{FP:5.0f}')
print(f'False Negatives:{FN:5.0f}')
print(f'Accuracy:{accuracy:5.2f}')
print(f'Precision:{precision:10.2f}')
print(f'Recall:{recall:10.4f}')
print(f'F-score:{fscore:10.4f}')



#################### (END) YOUR CODE ###################
    