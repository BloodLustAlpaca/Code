# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:54:27 2019

@author: Adric
"""

class report:
    def __init__(self,df):
        self.df = df
    def buildFrame(self):
        self.df.columns = self.df.columns.str.replace(' ', '_')
        return self.df
#gets the net income and returns an average
    def getAvgEarnings(self,number,startDate = 0):
        earnings = []
        sumEarn = 0
        for x in range(0,number):
            sumEarn += int(self.df.Earnings[x])
            earnings.append(int(self.df.Earnings[x]))
            print(self.df.Quarter_end[x])
        avg = (sumEarn/number)
        return ("{:,}".format(avg))
    
    def getDf():
        return self.df