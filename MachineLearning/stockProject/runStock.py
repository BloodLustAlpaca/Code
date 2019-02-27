# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 09:50:20 2019

@author: drumm
"""
import pandas as pd 
import numpy as np
import sklearn as skl
import zipfile
from stock import stock
from datetime import datetime, timedelta

def run():
    df = buildDf('qqq')
    df = df.drop(columns=df.columns[0])

    print(df.tail(100).to_string())
    #prints tuples lined up
    #print(list(zip(a[175:190],b[175:190])))
def buildDf(name,rebuild = False):
    #this will try to load the csv file, if it doesnt exist it raises and exceptions which then 
    #creates the csv file and frame you can force rebuild with a True parameter
    try:
        if(rebuild == False):
            df = pd.read_csv(str(name) + ".csv")
            return df
        else:
            raise Exception
    except:
        #ZF is the zip file with all the stocks and etfs broken up into the two folders
        zf = zipfile.ZipFile('Data.zip')
        df = pd.read_csv(zf.open('ETFs/' + name + '.us.txt'))
        targetStock = stock(df)
        df = targetStock.addDayCol(df)
        df = targetStock.addMaCol(df,'MA10',10)
        #df = targetStock.addMaCol(df,'MA20',20)
        df = targetStock.addMaCol(df,'MA50',50)
        df = targetStock.addTargetCol(df)
        #df = targetStock.addMaCol(df,'MA100',100)
        df.to_csv(path_or_buf=str(name) + ".csv")
        return df
    

if __name__ == '__main__':
    run()