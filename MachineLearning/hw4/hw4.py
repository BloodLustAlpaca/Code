# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 09:37:02 2019

@author: drumm
"""

#use linspace to split continuos up

import numpy as np

answer = -1/4*(np.log2(1/4))-(3/4*(np.log2(3/4)))
#print(answer)


def calc_entropy(p):
    if p!=0:
        return -p * np.log2(p)
    else:
        return 0


#def calc_entropy(listOfValues):
#    entropy = 0
#    for i in range(0,len(listOfValues)):
#        if(listOfValues[i] == 0):
#            pass
#        else:
#            entropy += -listOfValues[i]*np.log2(listOfValues[i])
#    return entropy
def feature_entropy(trues, falses):
    answer = 0
    answer = calc_entropy(trues/(trues+falses)) + calc_entropy(falses/(trues+falses))
    return answer
#print(calc_entropy(1/4)+calc_entropy(3/4))
def p_entropy(l):
    answer = 0
    for x in l:
        answer += -x*np.log2(x)
    return answer


print(feature_entropy(6,2))