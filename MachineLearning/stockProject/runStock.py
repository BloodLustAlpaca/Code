# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 09:50:20 2019

@author: drumm
"""
import pandas as pd 
import numpy as np
import zipfile
from stock import stock
from datetime import datetime, timedelta

def run():
    #ZF is the zip file will all the stocks and etfs broken up into the two folders
    zf = zipfile.ZipFile('Data.zip')
    #ETFs or Stocks with /ticker.us.txt will refence the ticker needed
    df = pd.read_csv(zf.open('ETFs/qqq.us.txt'))
    qqqStock = stock(df)
    print("Moving average for period is: " + str(qqqStock.sma("2010-10-22", 10)))
    print("Moving average  per day for period of time: " + str(qqqStock.stream_sma("2010-10-22",10,20)))
    a = qqqStock.stream_sma("2008-10-22",10,250)
    b = qqqStock.stream_sma("2008-10-22",35,250)
    qqqStock.compareAvg(a,b)
    
    #prints tuples lined up
    #print(list(zip(a[175:190],b[175:190])))
if __name__ == '__main__':
    run()