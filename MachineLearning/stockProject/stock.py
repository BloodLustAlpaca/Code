# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:07:28 2019

@author: drumm
"""
import pandas as pd 
import numpy as np
from datetime import datetime, timedelta

class stock:
    def __init__(self, df):
        self.df = df
    
    #Calculates the simple moving average by adding each open by date and dividing by days    
    def sma(self,startDate, amountDays):
        currentDate = startDate
        counter = 0
        mvavg = 0
        daysInPast = 0
        #counter is incremented when a valid day is read, weekends will be skipped
        #date adds the days in the past for the next try
        while(counter < amountDays):
            daysInPast += 1
            try:
                value = self.df.loc[self.df.Date == currentDate , 'Open'].tolist()[0]
                mvavg += value
                counter += 1
            except:
                pass
            currentDate = (datetime.strptime(startDate, '%Y-%m-%d') + timedelta(-daysInPast)).strftime('%Y-%m-%d')
        return mvavg/amountDays
    
    #This is to get multiple moving averages amount of days is what is sent to sma, timeperiod is how long to show
    def stream_sma(self, startDate, amountDays, timePeriod ):
        currentDate = startDate
        mvgAvgArray = [] 
        for x in range(0,timePeriod):
            while((currentDate not in self.df.Date.values)):
                currentDate = (datetime.strptime(currentDate, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
            mvgAvgArray.append(self.sma(currentDate,amountDays))
            currentDate = (datetime.strptime(currentDate, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        return mvgAvgArray
    
    def compareAvg(self,stream1,stream2):
        base = stream1[0] < stream2[0]
        for x in range (0,len(stream1)):
            print(x)
            if (base == (stream1[0] > stream2[0])):
                print("crossed at: " +str(x))
                base = not base
                
            
        
                        