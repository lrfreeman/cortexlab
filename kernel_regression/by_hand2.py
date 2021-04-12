# import numpy as np
# import plotly.express as px
# from statsmodels.nonparametric.kernel_regression import KernelReg as kr
# import plotly.graph_objs as go
# import pandas as pd

from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


# np.random.seed(1)
# # xwidth controls the range of x values.
# xwidth = 20
# ywidth = 20
# x = np.arange(0,xwidth,1)
# y = np.arange(0,ywidth,1)
#
# # we want to add some noise to the x values so that dont sit at regular intervals
# x_residuals = np.random.normal(scale=0.2, size=[x.shape[0]])
# # new_x is the range of x values we will be using all the way through
# new_x = x + x_residuals
# # We generate residuals for y values since we want to show some variation in the data
# num_points = x.shape[0]
# residuals = np.random.normal(scale=2.0, size=[num_points])
# # We will be using fun_y to generate y values all the way through
# fun_y = lambda x: -(x*x) + residuals
#
# # Plot the x and y values
# fig = px.scatter(x=new_x,y=fun_y(new_x), title='Figure 1:  Visualizing the generated data')
#
# a = kr(fun_y(new_x), new_x, var_type= 'c').fit()
# # print(a[0])
#
# fig2 = px.scatter(x=new_x,y=fun_y(new_x),  title='Figure 2: Statsmodels fit to generated data')
# fig2.add_trace(go.Scatter(x=new_x, y=a[0], name='Statsmodels fit',  mode='lines'))
# fig2.show()

class KernelRegression():
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth

    def gaussian_constant(self, bandwidth):
        """
        Returns the normalization constant for a gaussian curve
        """
        return 1/(bandwidth*np.sqrt(np.pi*2))

    def gaussian_exponential(self, kernel_x, xi, bandwidth):
        """
        Returns the gaussian function exponent term
        """
        num =  -np.square((xi - kernel_x))
        den = 2 * (bandwith * bandwidth)
        return num/den

    def kernel_function(self, bandwidth, kernel_x, xi):
        """
        Returns the gaussian function value. Combines the gauss_const and
        gauss_exp to get this result
        """
        constant = self.gaussian_const(h)
        gauss_val = constant * np.exp(self.gaussian_exponential(kernel_x,xi,bandwidth))
        return gauss_val
