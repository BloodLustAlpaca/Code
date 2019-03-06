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
from datetime import datetime, timedelta
np.set_printoptions( threshold=np.inf,  formatter={'float_kind':'{:.2f}'.format})

'''Originally I was trying to predict the moving average crossover but for the HW2 assignment I am not as far as I need to be to use 
that target. It is a very sparse target only being true for 2% of the data. This means that if I run the models on it they will almost
always be 98% right by never guessing 1. I decided to change to predict if the next day the stock would go up or down.
I havent figured out the best features for this yet and only get about 53% correct best case scenario.
Data.zip has all of the database needed to use this program. keep it in the same file as this
'''
def run():
    #df = buildDf('qqq')
    df = buildDf('atvi')
    
#This drops the first column which is an extra index
    df = df.drop(columns=df.columns[0])
    
#I slice off the first 50 since they are Nan
#start at 50 for MAcross as MA50 col doesnt start counting until 50
#The max is len(df)-1 because to calculate updown it reads the next date, since the next one at the end is null this avoids the error
#X are the features I want
#Y is the target UpDown which tries to predict based on previous close to close if the next day will go up or down
#    X = df[['Open','High','Low','Close','Volume','MA10','MA50']][50:len(df)-1].values
#    y = df[['UpDown']][50:len(df)-1].values.astype(int).ravel()
#    
##This sets the training and tests automatically and makes sure features and targets are distributed well.
#    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
#
##I make a list of models to try out, setting the seed so they are they same when ran
#    models = []
#    models.append(('LR', LogisticRegression()))
#    models.append(('LDA', LinearDiscriminantAnalysis()))
#    models.append(('KNN', KNeighborsClassifier()))
##I commented out my classifier because it takes like 15 min to run. feel free to try it out though. I didnt set the random seed so it will
##differ slightly from the KNN but if it is set, it will be the same
#    #models.append(('AdricsKnn',AdricsKNNClassifier()))
#    models.append(('CART', DecisionTreeClassifier()))
#    models.append(('NB', GaussianNB()))
#    models.append(('SVM', SVC()))
#    parameters = {'n_neighbors':[6,7,8,9],'K':[6,7,8], 'C':[.0001,.001,.01,.1,1,10,100],'max_depth':[3,4,5,6,7,8],'gamma':[.0001,.001,.01,.1]}
#    results = []
#    names = []
#    
##this makes sure the distribution is balanced and sets up the results
#    for name, model in models:
#        param_grid = {}
#        
##this takes the keys and checks to make sure it doesnt pass an incorrect one to a model.
#        for k in parameters.keys():
#            if k in model.get_params().keys():
#                param_grid[k] = parameters[k]
#        
##I do the grid search here to find the best parameters based from the parameters above
#        gs = model_selection.GridSearchCV(model, param_grid,cv=5,scoring = 'accuracy')
#        gs.fit(X_train,y_train)
#        
##This gives the results after cross validating so that I can plot them on graph
#        cv_results = model_selection.cross_val_score(gs, X_train, y_train, cv=5, scoring='accuracy')
#        names.append(name)
#        msg = "Model:\n%s  \n%s: %f (%f)" % (gs.best_estimator_,name, cv_results.mean(), cv_results.std())
#        results.append(cv_results)
#        print(msg)
#
##This shows and compares results on a graph
#    fig = plt.figure()
#    fig.suptitle('Algorithm Comparison')
#    ax = fig.add_subplot(111)
#    plt.boxplot(results)
#    ax.set_xticklabels(names)
#    plt.show()
#
#    
##This is one model and testing to see the results so that they can be compared and predicts with clean data
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
        df = pd.read_csv(zf.open("Stocks/" +name + '.us.txt'))
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
        #df = targetStock.addMaCol(df,'MA100',100)
        df.to_csv(path_or_buf=str(name) + ".csv")
        return df
    

if __name__ == '__main__':
    run()



import numpy as np
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
                
            
        
import numpy as np
#import sklearn

# use multiple inheritance so this class can be used
# with scikit learn ensembles etc.
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
class AdricsKNNClassifier(BaseEstimator, ClassifierMixin):

#    """
#    Whatever parameters your classifier takes are initialized
#    in this python constructor
#    """
    def __init__(self,K=5):
        self.K=K

    """
    KNN doesn't really do any "training" it just uses the 
    training set data during the prediction phase, so just 
    store the data in the object during "fit" or "training"
    """
    def fit(self, X, y):
        self.X=X
        self.y=y
        return self

#    """
#    Given a testing set... return predictions
#    """
    def predict(self, X):
        # return an array of labels for each feature set in X
        return np.asarray([self._predict(self.X,rowToCompare,self.y,self.K) for rowToCompare in X])

#    """
#    Given a set of training vectors A (from training) and a
#    single feature vector b (from testing), return the label 
#    that best classifies b; get labels from Y (training labels)
#    """
    def _predict(self,X,rowToCompare,Y,k):
        # calculate distances between vector b and each vector a in A
        # get the cooresponding labels (from Y) of nearest K neighbors
        # return the label that has the most "votes" from the K neighbors

        #  hints: 
        # you can use np.linalg.norm to calculate distance between two vectors
        # you can use collections.Counter to select the label with maximum values
        # in a list of labels (votes)

        #

        # return a random choice of iris labels
        distances = [np.linalg.norm(exampleRow-rowToCompare) for exampleRow in X]
        votes = Y[np.argsort(distances)[0:k]]
        return Counter(votes).most_common(1)[0][0]
        

