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
        

