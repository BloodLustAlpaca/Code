# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 09:18:10 2019

@author: drumm
"""


def amountMade(amount, rate, years, periodPerYear = 1):
    total = 0
    added = amount*periodPerYear*years
    for x in range(0,years):
        total += amount
        total += total*rate
        print("Year : " + str(x) + "\naccount value: " + str(total))
    return total-added, added

print("Interst gained, Amount Added: " + str(amountMade(651*12,.06,12)))
#
#def amountSaved(startingAmount,normalPayment,amountPaidExtra, rate, years, periodPerYear = 1):
#    total = startingAmount
#    for x in range (0,years):
#        total = (total - normalPayment - amountPaidExtra)
#        total += rate*total
#    return total
#
#print(amountSaved(200000,1090*12,0,.025,12))
        
    