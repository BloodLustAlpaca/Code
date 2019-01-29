# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 16:17:36 2019

@author: drumm
"""

class Neuron():
    #Where The number of inputs includes the bias input and weights are a list of weighted values
    #with the fireThresh is the number threshold that when greater than fires, and learning rate as the
    #adjustment value used to train
    def __init__(self, numOfInputs,weights,fireThresh,learningRate):
        self.target = None
        self.outVal = None
        self.numOfInputs = numOfInputs
        self.weights = weights
        self.learningRate = learningRate
        self.fireThresh = fireThresh
        self.total = 0.0
        self.listOfInputs=[]
        
    #Allows setting of input
    def setInput(self, listOfInputs):
        self.listOfInputs = listOfInputs
    
    #Runs the algorithm that calculates the output value and the total activation energy
    def output(self):
        self.total = 0.0
        for x in range(0,len(self.listOfInputs)):
            self.total += (self.weights[x]*self.listOfInputs[x])
        if self.total > self.fireThresh:
            self.outVal = 1
        else:
            self.outVal = 0
        self.train()   
        return str(self.outVal) + " The target was: " + str(self.target)
    ##This trains the neuron by adjusting the weights
    def train(self):
        if(self.target != self.outVal):
            print("\nINCORRECT\n")
            for x in range(0, len(self.weights)):
                self.weights[x] = self.weights[x] - self.learningRate*(self.outVal-self.target)*self.listOfInputs[x]
        else:
            print("\nCORRECT\n")
    #this sets the target that is correct
    def setTarget(self, target):
        self.target = target
    def getWeights(self):
        print("\nWeights are : " + str(self.weights)+ "\n")
        