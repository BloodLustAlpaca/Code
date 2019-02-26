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
    print(qqqStock.sma("2010-10-22", 10))
    print(qqqStock.stream_sma("2010-10-22",10,20))
    a = qqqStock.stream_sma("2010-10-22",10,150)
    b = qqqStock.stream_sma("2010-10-22",35,150)
    qqqStock.compareAvg(a,b)
if __name__ == '__main__':
    run()