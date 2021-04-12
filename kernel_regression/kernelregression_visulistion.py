# import DLC_Classes as CL
# import math
# import data_visulisation_functions.Kernel_Regression_functions as KR
# import numpy as np
#
# # data = CL.CortexLab('/Users/laurence/Desktop/Neuroscience/kevin_projects/data/processed_physdata/aligned_physdata_KM011_2020-03-23_probe1.mat',
# #                  '/Users/laurence/Desktop/Neuroscience/kevin_projects/data/KM011_video_timestamps/2020-03-23/face_timeStamps.mat',
# #                  '/Users/laurence/Desktop/Neuroscience/kevin_projects/data/23_faceDLC_resnet50_Master_ProjectAug13shuffle1_133500.csv')
# #
# # trial_df, spike_df = data.load_data(data.session_data)
# # first_lick_df, lick_df, df = data.compute_the_first_lick()
# # #
# # # trunc = lambda x: math.trunc(x)
# # # trial_df["Trial IDs"] = trial_df["trial_start_times"].apply(trunc)
# # #
# # # data = first_lick_df.merge(trial_df, on="Trial IDs")
# # #
# # y = np.asarray([1,2,3,4,5,6,7,8,9,10])
# # # x = np.asarray([1,2,3,4,5,6,7,8,8,10])
# # # bandwith = 0.1
# #
# #
# #
# #
# # gkr = KR.GKR([[0,10],[10,20],[30,40],[40,50],[60,70],[80,90],[90,100]],
# #           [0,0,0,0,0,0,0,0,0,0],
# #           10)
# #
# # gkr.visualize_kernels()
# """-------------------------------------------------------------"""
# from statsmodels.nonparametric.kernel_regression import KernelReg
# import numpy as np
#
# licks = np.array([[1,2],[2,3]])
# spikes = np.array([3,4])
#
# model = KernelReg(licks,spikes, var_type='c', reg_type='lc').fit()

# N-dimensional using numpy

from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt


'''Class for Gaussian Kernel Regression'''
class GKR:

    def __init__(self, x, y, b):
        self.x = np.array(x)
        self.y = np.array(y)
        self.b = b

    '''Implement the Gaussian Kernel'''
    def gaussian_kernel(self, z):
        return (1/np.sqrt(2*np.pi))*np.exp(-0.5*z**2)

    '''Calculate weights and return prediction'''
    def predict(self, X):
        kernels = np.array([self.gaussian_kernel((np.linalg.norm(xi-X))/self.b) for xi in self.x])
        weights = np.array([len(self.x) * (kernel/np.sum(kernels)) for kernel in kernels])
        return np.dot(weights.T, self.y)/len(self.x)

    def visualize_kernels(self):
        zsum = np.zeros((120,120))
        plt.figure(figsize = (10,5))
        ax = plt.axes(projection = '3d')
        for xi in self.x:
            x, y = np.mgrid[0:120:120j, 0:120:120j]
            xy = np.column_stack([x.flat, y.flat])
            z = multivariate_normal.pdf(xy, mean=xi, cov=self.b)
            z = z.reshape(x.shape)
            zsum += z

        ax.plot_surface(x,y,zsum)

        ax.set_ylabel('y')
        ax.set_xlabel('x')
        ax.set_zlabel('Kernel Weights wi')
        # plt.legend()
        plt.show()

gkr = GKR([[0,100000],[10,2000],[30,40000],[44,60000],[50,52000],[67,92000],[78,79000],[89,123000],[100,200]], [1,2,3,2,1,1,1,1, 1932000000], 10)

gkr.visualize_kernels()
