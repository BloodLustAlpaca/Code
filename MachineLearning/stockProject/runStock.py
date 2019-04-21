# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 09:50:20 2019

@author: drumm
"""
import tensorflow as tf
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
import zipfile
from stock import stock
from adricsKnnTester import AdricsKNNClassifier
from account import Account
from datetime import datetime, timedelta
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import TensorBoard
np.random.seed(7)
np.set_printoptions( threshold=np.inf,  formatter={'float_kind':'{:.2f}'.format})

'''Originally I was trying to predict the moving average crossover but for the HW2 assignment I am not as far as I need to be to use 
that target. It is a very sparse target only being true for 2% of the data. This means that if I run the models on it they will almost
always be 98% right by never guessing 1. I decided to change to predict if the next day the stock would go up or down.
I havent figured out the best features for this yet and only get about 53% correct best case scenario.
'''
def run():
    df = buildDf('qqq')
    #df = buildDf('atvi')
    #This drops the first column which is an extra index
    df = df.drop(columns=df.columns[0])
    df['seq']=df.index
    import seaborn as sns
    fig,ax=plt.subplots()
    sns.regplot(x='seq',y='Close',data=df,lowess=True)
    

    #print(df.tail(50).to_string())
    df['Volatility']=(df['Close']-df['Open'])/df['Volume']
    fig,ax=plt.subplots()
    sns.heatmap(df.corr(),cmap='Blues')
#I slice off the first 50 since they are Nan
#start at 50 for MAcross as MA50 col doesnt start counting until 50
#The max is len(df)-1 because to calculate updown it reads the next date, since the next one at the end is null this avoids the error
#X are the features I want
#Y is the target UpDown which tries to predict based on previous close to close if the next day will go up or down
    X = df[['Open','High','Low','Close','Volume','MA10','MA50']][50:len(df)-1].values
    y = df[['UpDown']][50:len(df)-1].values.astype(int).ravel()
    
#This sets the training and tests automatically and makes sure features and targets are distributed well.
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)



    ###Feature selection
    print("Feature selection")
    selection = RFECV(RandomForestClassifier(),scoring='accuracy')
    selection.fit_transform(X_train,y_train)
    print(selection.support_)
    


#I make a list of models to try out, setting the seed so they are they same when ran
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
#I commented out my classifier because it takes like 15 min to run. feel free to try it out though. I didnt set the random seed so it will
#differ slightly from the KNN but if it is set, it will be the same
    #models.append(('AdricsKnn',AdricsKNNClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    #models.append(('SVM', SVC()))
    parameters = {'n_components':[None,1,2],'n_neighbors':[6,7,8,9],'K':[6,7,8], 'C':[.0001,.001,.01,.1,1,10,100],'max_depth':[3,4,5,6,7,8],'gamma':[.0001,.001,.01,.1]}
    results = []
    names = []
    
#this makes sure the distribution is balanced and sets up the results
    for name, model in models:
        param_grid = {}
        
#this takes the keys and checks to make sure it doesnt pass an incorrect one to a model.
        for k in parameters.keys():
            if k in model.get_params().keys():
                param_grid[k] = parameters[k]
        
#I do the grid search here to find the best parameters based from the parameters above
        gs = model_selection.GridSearchCV(model, param_grid,cv=5,scoring = 'accuracy')
        gs.fit(X_train,y_train)
        
#This gives the results after cross validating so that I can plot them on graph
        cv_results = model_selection.cross_val_score(gs, X_train, y_train, cv=5, scoring='accuracy')
        names.append(name)
        msg = "Model:\n%s  \n%s: %f (%f)" % (gs.best_estimator_,name, cv_results.mean(), cv_results.std())
        results.append(cv_results)
        print(msg)

#This shows and compares results on a graph
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

    
#This is one model and testing to see the results so that they can be compared and predicts with clean data
#    model = LinearDiscriminantAnalysis()
#    model.fit(X_train,y_train)
#    predicted = model.predict(X_test)
#    print("PREDICTION LDR IS:::::::::::::::::::::::::\n")
#    print(predicted)
#    print("LDR actual:::::::::::::::::::")
#    print(y_test.reshape((1,len(y_test))))
#    print(collections.Counter(y_test))

    #prints tuples lined up
    #print(list(zip(a[175:190],b[175:190])))
    
    
    #########   KERAS network
    ##Trying normalize data for keras network
    NAME= "StockPredict"
    tensorboard= TensorBoard(log_dir='logs/{}'.format(NAME))

    X_train =tf.keras.utils.normalize(X_train, axis =1)
    X_test= tf.keras.utils.normalize(X_test, axis=1)
    
    modelN = tf.keras.models.Sequential()
    modelN.add(tf.keras.layers.Flatten())
    modelN.add(tf.keras.layers.Dense(128,input_dim=7,activation=tf.nn.relu))
    modelN.add(tf.keras.layers.Dropout(.2))
    modelN.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
    modelN.add(tf.keras.layers.Dense(1,activation=tf.nn.sigmoid))
    modelN.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    modelN.fit(X_train,y_train,epochs=8, callbacks=[tensorboard], validation_split =.1)
    val_loss, val_acc = modelN.evaluate(X_test, y_test)
    print(val_loss, val_acc)
    #print(y_test)
    
    ##plots scatter matrix
    #pd.scatter_matrix(df)
    
#this will try to load the csv file, if it doesnt exist it raises and exceptions which then 
#creates the csv file and frame. you can force rebuild with a True parameter
#This makes it way faster than doing all the calculations every time.
def buildDf(name,rebuild = False):
    try:
        if(rebuild == False):
            df = pd.read_csv(str(name) + ".csv")
            return df
        else:
            raise Exception
    except:
#ZF is the zip file with all the stocks and etfs broken up into the two folders "Stocks/ for stock name or ETFs/ for etf name
#I then add the columns I think might be useful such as day of week, moving averages, targetColumn which is the cross over and the
#updown for other testing.
        zf = zipfile.ZipFile('Data.zip')
        df = pd.read_csv(zf.open("ETFs/" +name + '.us.txt'))
        targetStock = stock(df)
        print("adding day col")
        df = targetStock.addDayCol(df)
        print("adding MA10 col")
        df = targetStock.addMaCol(df,'MA10',10)
        print("adding MA50 col")
        #df = targetStock.addMaCol(df,'MA20',20)
        df = targetStock.addMaCol(df,'MA50',50)
        print("adding Target col")
        df = targetStock.addTargetCol(df)
        print("adding UpDown col")
        df = targetStock.addUpDownCol(df)
        print("adding crossUpCrossDownCols")
        df = targetStock.addCrossUpCrossDownCols(df)
        #df = targetStock.addMaCol(df,'MA100',100)
        df.to_csv(path_or_buf=str(name) + ".csv")
        return df
    

if __name__ == '__main__':
    run()