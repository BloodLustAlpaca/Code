# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 09:39:58 2019

@author: Adric
"""

class Account:
    def __init__(self,df):
        self.df = df
        self.balance = 100000
        self.shares = 0
        
    def buy(self,price,shares = 0):
        if(shares == 0):
            shares = int(self.balance/price)
        if(shares*price <= self.balance):
            print("Buying {} shares at ${} for a total of ${}".format(shares,price,shares*price))
            self.balance = self.balance - (shares * price)
            self.shares = self.shares + shares
            print("Shares:{} and Balance is now ${}\n".format(self.shares,self.balance))
        else:
            print("not enough in balance")
    def sell(self, price, shares = 0):
        if(shares == 0):
            shares = self.shares
        if(self.shares >= shares):
            print("Selling {} shares at ${} for a total of ${}".format(shares,price,shares*price))
            self.balance = self.balance + (shares * price)
            self.shares = self.shares - shares
            print("Shares:{} and Balance is now ${}\n".format(self.shares,self.balance))
        else:
            print("not enough shares")
    def buySellOnCross(self):
        for x in range(50,len(self.df)):
            if(self.df.loc[x,'CrossUpCrossDown'] == 1):
                self.buy(self.df.loc[x,'Open'])
            elif(self.df.loc[x,'CrossUpCrossDown'] == -1 ):
                self.sell(self.df.loc[x,'Open'])
        self.sell(self.df.loc[len(self.df)-1,'Open'])
    def buyAndHold(self):
        self.buy(self.df.loc[55,'Open'])
        self.sell(self.df.loc[len(self.df)-1,'Open'])
        
        