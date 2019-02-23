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
    zf = zipfile.ZipFile('Data.zip')
    df = pd.read_csv(zf.open('ETFs/qqq.us.txt'))
    print(df.columns)
    print(df.sample(n=3))
    
    print(movAvg([1,2,3,4,5]))
    movAvg1("Open", "2010-10-22",10,"qqq")
    
def movAvg1(open_or_close,startDate,days,ticker):
    zf = zipfile.ZipFile('Data.zip')
    df = pd.read_csv(zf.open('ETFs/' + ticker + '.us.txt'))
    dfIndex = df[(df['Date'] == startDate)].index[0]
    print(df.loc[dfIndex,'Date'])
    print(type(df.loc[df.index[0],"Date"]))
    print(dfIndex)
    date = startDate
    mvavg = 0
    for x in range (0,days):
        #find how to weekdays
        #print('TEST:' + str(df.loc[df.Date == date , 'Open'].values[0]))
        #while((df.loc[df.Date == date, 'Open']).empty):
        #   date = (datetime.strptime(startDate, '%Y-%m-%d') + timedelta(days=x)).strftime('%Y-%m-%d')
        mvavg += df.loc[df.Date == date , 'Open']      #currentDf = df[df.Date == date]
        #mvavg += currentDf.Open.tolist()[0]
        date = (datetime.strptime(startDate, '%Y-%m-%d') + timedelta(days=x)).strftime('%Y-%m-%d')
        #int(np.isnan(df.loc[df.Date == date, 'Open'])
        print()
def movAvg(arr):
    answer = 0
    for x in arr:
        answer += arr[x-1]
    return answer/len(arr)
if __name__ == '__main__':
    run()

    
    
    