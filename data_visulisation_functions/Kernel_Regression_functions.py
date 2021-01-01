from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#1-D normal Python
import math

'''Class for Gaussian Kernel Regression'''
class GKR:

    def __init__(self, x, y, b):
        self.x = x
        self.y = y
        self.b = b

    '''Implement the Gaussian Kernel'''
    def gaussian_kernel(self, z):
        return (1/math.sqrt(2*math.pi))*math.exp(-0.5*z**2)

    '''Calculate weights and return prediction'''
    def predict(self, X):
        kernels = [self.gaussian_kernel((xi-X)/self.b) for xi in self.x]
        weights = [len(self.x) * (kernel/np.sum(kernels)) for kernel in kernels]
        return np.dot(weights, self.y)/len(self.x)

    def visualize_kernels(self, precision):
        plt.figure(figsize = (10,5))
        for xi in self.x:
            x_normal = np.linspace(xi - 3*self.b, xi + 3*self.b, precision)
            y_normal = norm.pdf(x_normal, xi, self.b)
            plt.plot(x_normal, y_normal, label='Kernel at xi=' + str(xi))

        plt.ylabel('Kernel Weights wi')
        plt.xlabel('x')
        plt.legend()

    def visualize_predictions(self, precision, X):
        plt.figure(figsize = (10,5))
        max_y = 0
        for xi in self.x:
            x_normal = np.linspace(xi - 3*self.b, xi + 3*self.b, precision)
            y_normal = norm.pdf(x_normal, xi, self.b)
            max_y = max(max(y_normal), max_y)
            plt.plot(x_normal, y_normal, label='Kernel at xi=' + str(xi))

        plt.plot([X,X], [0, max_y], 'k-', lw=1,dashes=[2, 2])
        plt.ylabel('Kernel Weights wi')
        plt.xlabel('x')
        plt.legend()

    def test(self):
        y = []
        for i in self.x:
            y.append(self.predict(i))
        plt.plot(self.x, y)
        plt.ylabel('First lick times')
        plt.xlabel('Reward times')
        plt.show()

# gkr = GKR(x_data, y_data, sigmoid)
gkr = GKR([10,20,30,40,50,60,70,80,90,100,110,120], [2337,2750,2301,2500,1700,2100,1100,1750,1000,1642, 2000,1932], 10)
# gkr.test()
# print(gkr.x)

# data = [10,20,30,40,50,60,70,80,90,100,110,120]
# y = []
# for i in data:
#     y.append(gkr.predict(i))
# plt.plot(data, y)
# plt.show()
