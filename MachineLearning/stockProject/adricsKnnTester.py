# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 09:36:48 2019

@author: Adric
"""

import numpy as np
#import sklearn
import random

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
        return np.asarray([self._predict(self.X,b,self.y,self.K) for b in X])

#    """
#    Given a set of training vectors A (from training) and a
#    single feature vector b (from testing), return the label 
#    that best classifies b; get labels from Y (training labels)
#    """
    def _predict(self,A,b,Y,k):
        # calculate distances between vector b and each vector a in A
        # get the cooresponding labels (from Y) of nearest K neighbors
        # return the label that has the most "votes" from the K neighbors

        #  hints: 
        # you can use np.linalg.norm to calculate distance between two vectors
        # you can use collections.Counter to select the label with maximum values
        # in a list of labels (votes)

        # I'll let you write this code yourself.
        # We can look at my solution after the homework is collected

        # return a random choice of iris labels
        distances = [np.linalg.norm(a-b) for a in A]
        votes = Y[np.argsort(distances)[0:k]]
        return Counter(votes).most_common(1)[0][0]
        

print("\nTesting KNN classifier for HW 2...\n")

import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

# Split iris data in train and test data
# A random permutation, to split the data randomly
# NOTE: this is an example of train_test_split() that so
# many libraries offer (this allocates 10 to testing)
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

# Create and fit a nearest-neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train) 

preds = knn.predict(iris_X_test)
print("sklearn's KNN", preds)

# Create and fit a nearest-neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
knn = AdricsKNNClassifier(K=5)
knn.fit(iris_X_train, iris_y_train) 

preds = knn.predict(iris_X_test)
print("    Adric's KNN", preds)
