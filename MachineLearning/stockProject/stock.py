# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:07:28 2019

@author: drumm
"""
import pandas as pd 
import numpy as np
from datetime import datetime, timedelta,date
import calendar

#this class is my attempt at making things more clean by providing the functionality seperate from the main code and analysis
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
    
#This is to get multiple moving averages. Amount of days is what is sent to sma, timeperiod is how long of a window to evaluate
#so maybe 10 day moving average, stream it for a period of 90 days so that each day calculates the new average.
#Returns array of moving averages
    def stream_sma(self, startDate, amountDays, timePeriod ):
        currentDate = startDate
        mvgAvgArray = [] 
        for x in range(0,timePeriod):
            while((currentDate not in self.df.Date.values)):
                currentDate = (datetime.strptime(currentDate, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
            mvgAvgArray.append(self.sma(currentDate,amountDays))
            currentDate = (datetime.strptime(currentDate, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        return mvgAvgArray
    
#This returns an array of true and false for each comparison. So if it crossed twice,
#you would have all falses and two trues where it crossed. Returns the array of true falses relating to crosses
    def compareAvg(self,stream1,stream2):
        crossArray = []
        base = (stream1[0] < stream2[0])
        print("BASE IS: " + str(base))
        for x in range (0,len(stream1)):
            if (base == (stream1[x] > stream2[x])):
                print("crossed at: " +str(x))
                base = not base
                crossArray.append(True)
            else:
                crossArray.append(False)
        return crossArray

#This adds a column of days of the week to the DF and then returns the DF
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

#This adds a column of Moving averages based on the period you send it, example 10 would be 10day MA
#Returns the DF with new column added
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
    
#This adds the target column or the crossover column
#takes arguments of df and bigMa, bigMa is the biggest moving average 
#since the crossover can't count until it is calculated
#This will have Nans from 0-bigMa
#Returns DF with new column added
    def addTargetCol(self,df):
        if('Target' not in df):
            targetArray = []
            results = self.compareAvg(df.MA10[50:].values,df.MA50[50:].values)
            for x in df.Date[:50]:
                targetArray.append(float('nan'))
            results = targetArray + results
            df['Target'] = results
            return df
        else:
            print("col exisits already")
            return df

#This adds a column of days the market went up and down. It looks at the next day and compares it with the current day, if the next
#day is bigger it puts True.
#this returns the DF with added column
    def addUpDownCol(self,df):
        if('UpDown' not in df):
            upDownArr = []
            for x in range (0,len(df.Close)-1):
                if(df.Close[x+1] > df.Close[x]):
                    upDownArr.append(True)
                else:
                    upDownArr.append(False)
            upDownArr.append(float('nan'))
            df['UpDown'] = upDownArr
            return df
        else:
            print("col already exisits")
                
            
        
                        