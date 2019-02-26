# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:07:28 2019

@author: drumm
"""
import pandas as pd 
import numpy as np
from datetime import datetime, timedelta,date
import calendar

class stock:
    def __init__(self, df):
        self.df = df
    
    #Calculates the simple moving average by adding each open by date and dividing by days    
    def sma(self,startDate, amountDays,Open_Close = 'Open'):
        currentDate = startDate
        counter = 0
        mvavg = 0
        daysInPast = 0
        #counter is incremented when a valid day is read, weekends will be skipped
        #date adds the days in the past for the next try
        while(counter < amountDays):
            daysInPast += 1
            try:
                value = self.df.loc[self.df.Date == currentDate , Open_Close].tolist()[0]
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
        base = (stream1[0] < stream2[0])
        print("BASE IS: " + str(base))
        for x in range (0,len(stream1)):
            if (base == (stream1[x] > stream2[x])):
                print("crossed at: " +str(x))
                base = not base
                
    def addDayCol(self,df):
        if('Weekday' not in df):
            weekdayArray = []
            for x in df.Date:
                weekdayArray.append(calendar.day_name[datetime.strptime(x, '%Y-%m-%d').weekday()])
            df['Weekday'] = weekdayArray
            return df
        else:
            print("Weekday array already exists")
            return df
    def addMaCol(self,df,name,period, Open_Close = 'Open'):
        print("in addmaCol")
        if(name not in df):
            maArray = []
            if(Open_Close == 'Open'):
                for x in df.Date[:period]:
                    maArray.append(float('nan'))
                for x in df.Date[period:]:
                    #print("on day: " + str(x))
                    maArray.append(self.sma(x,period))

            elif(Open_Close == 'Close'):
                for x in df.Date:
                    for x in df.Date[:period]:
                        maArray.append(float('nan'))
                for x in df.Date[period:]:
                    #print("on day: " + str(x))
                    maArray.append(self.sma(x,period,Open_Close))
            else:
                print("Open_Close incorrect Arg")
                return df
        else:
            print(str(name) + " is already in df")
            return df
        df[name] = maArray
        return df
            
        
                
            
        
                        