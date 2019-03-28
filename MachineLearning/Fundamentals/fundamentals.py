# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:40:07 2019

@author: Adric
"""

" Dataset for fundamental data http://www.stockpup.com/data/"
"Data set for daily prices and volumes https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/home#_=_"
import pandas as pd
from report import report
name = "GOOG"
df = pd.read_csv(str(name) + ".csv")
googReport = report(df)
googReport.buildFrame()
print(googReport.getAvgEarnings(50))


