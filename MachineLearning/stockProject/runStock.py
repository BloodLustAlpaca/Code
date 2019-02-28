# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 09:50:20 2019

@author: drumm
"""
import pandas as pd 
import numpy as np
import collections
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import zipfile
from stock import stock
from datetime import datetime, timedelta
np.set_printoptions( threshold=np.inf,  formatter={'float_kind':'{:.2f}'.format})

def run():
    df = buildDf('qqq')
    df = df.drop(columns=df.columns[0])
    #I slice off the first 50 since they are Nan
    
    X = df[['Open','High','Low','Close','Volume','MA10','MA50']][50:].values
    y = df[['Target']][50:].values.astype(int)
    Xt = X[:X.shape[0]-500]
    yt = y[:y.shape[0]-500]
    Xp = X[X.shape[0]-500:]
    yp = y[y.shape[0]-500:]
    seed = 7
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    
    results = []
    names = []
    scoring = 'accuracy'
    for name, model in models:
    	kfold = model_selection.KFold(n_splits=10, random_state=seed)
    	cv_results = model_selection.cross_val_score(model, Xt, yt.ravel(), cv=kfold, scoring=scoring)
    	results.append(cv_results)
    	names.append(name)
    	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    	print(msg)
        
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()
    model = KNeighborsClassifier()
    model.fit(Xt,yt.ravel())
    predicted = model.predict(Xp)
    print("PREDICTION ExtraTrees IS:::::::::::::::::::::::::\n")
    print(predicted)
    print("Extra Trees actual:::::::::::::::::::")
    print(yp.reshape((1,500)))
    print(predicted.shape)
    print(collections.Counter(yp.ravel()))
#    model = KNeighborsClassifier(n_neighbors = 3)
#    model.fit(Xt,yt.ravel())
#    predicted = model.predict(Xp)
#
#    print("PREDICTION KNN IS:::::::::::::::::::::::::\n")
#    print(predicted)
#    print("KNN actual:::::::::::::::::::")
#    print(yp.reshape((1,500)))
#    print(predicted.shape)
    
    
    #print(df.tail(100).to_string())
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