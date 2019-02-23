# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 09:50:20 2019

@author: drumm
"""
import pandas as pd 
import numpy as np
import zipfile
from datetime import datetime, timedelta

def run():
    #ZF is the zip file will all the stocks and etfs broken up into the two folders
    zf = zipfile.ZipFile('Data.zip')
    #ETFs or Stocks with /ticker.us.txt will refence the ticker needed
    df = pd.read_csv(zf.open('ETFs/qqq.us.txt'))
    print(df.columns)
    print(df.sample(n=3))
    
    print(movAvg([1,2,3,4,5]))
    print(movAvg1("Open", "2010-10-22",10,"qqq"))
    
def movAvg1(open_or_close,startDate,days,ticker):
    zf = zipfile.ZipFile('Data.zip')
    df = pd.read_csv(zf.open('ETFs/' + ticker + '.us.txt'))
    date = startDate
    counter = 0
    mvavg = 0
    daysInFuture = 0
    
    #counter is incremented when a valid day is read, weekends will be skipped
    #date adds the days in the future for the next try
    
    while(counter < days):
        daysInFuture += 1
        try:
            value = df.loc[df.Date == date , 'Open'].tolist()[0]
            mvavg += value
            counter += 1
        except:
            pass
        date = (datetime.strptime(startDate, '%Y-%m-%d') + timedelta(daysInFuture)).strftime('%Y-%m-%d')
    return mvavg/days
def movAvg(arr):
    answer = 0
    for x in arr:
        answer += arr[x-1]
    return answer/len(arr)
if __name__ == '__main__':
    run()

    
    
    