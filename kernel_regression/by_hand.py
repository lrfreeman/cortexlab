from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

class GKR:

    def __init__(self, training_x_inputs, training_y_inputs, bandwith):
        self.x = training_x_inputs
        self.y = training_y_inputs
        self.b = bandwith

    '''Implement the Gaussian Kernel'''
    def gaussian_kernel(self, z):
        return (1/math.sqrt(2*math.pi))*math.exp(-0.5*z**2)

    '''Calculate weights and return prediction'''
    def predict(self, X):
        kernels = [self.gaussian_kernel((xi-X)/self.b) for xi in self.x]
        weights = [len(self.x) * (kernel/np.sum(kernels)) for kernel in kernels]
        return np.dot(weights, self.y)/len(self.x)

# gkr = GKR([10,20,30,40,50,60,70,80,90,100,110,120], [2337,2750,2301,2500,1700,2100,1100,1750,1000,1642, 2000,1932], 10)

# print(gkr.predict(30))
