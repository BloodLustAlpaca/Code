# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 16:25:29 2019

@author: drumm
"""
from neuron import Neuron
#inputs are the data you feed it the neuron inputs are the inputs from other neurons
inputs = [[-1,0,0],[-1,0,1],[-1,1,0],[-1,1,1]]
n1= Neuron(3,[-.05,-.02,.02],0,.25)

#loop for this many running through inputs
loopNum= 20
for x in range(0,loopNum):
    if(x%len(inputs) == 0):
        print("RUN: " + str(int(x/len(inputs))) + "\n==================================\n")
    n1.setInput(inputs[x%len(inputs)])
    if inputs[x%len(inputs)] == [-1,0,0]:
        n1.setTarget(0)
    else:
        n1.setTarget(1)
    print("input was: " + str(inputs[x%len(inputs)]) + " |" + str(n1.output()) + "\nactivation was: " + str(n1.total))

n1.getWeights()