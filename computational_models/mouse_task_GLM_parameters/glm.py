#External imports
import statsmodels.api as sm
from statsmodels.formula.api import glm
import pandas as pd
import numpy as np
import glob
from matplotlib import pyplot as plt
import time

#My imports
from GLM_construct_regressors import construct_regressors
from GLM_classes import *

#Performance checks
start_time = time.time()

#----------

# # #Generate Synth data
# data = Data("s_data")
# choices = data._data_frame["left_choices"]
# # print(len(choices))
# choices = np.array(choices, dtype=np.int)
#
# #Regressors for Synthetic data
# regressors = construct_regressors(data, 10)
# regressors = pd.DataFrame(regressors)
# X = sm.add_constant(regressors, prepend=False)
# print(choices)
# # print(regressors)

#------------------------------------------

# Generate Behavioural data
data = Data("b_data")
print("Raw data", len(data._data_frame))
df = data._data_frame
print(df.describe())

"""Choices / Code to convert choices into binary to solve concat error"""
"""Only analyze trials that are non-vilations and of free choice."""
good_trials = data._data_frame.loc[(data._data_frame["free"] == 1)]
choices = np.asarray(good_trials["left_choices"])
choices = np.array(choices, dtype=np.float)
print("Good trial number", len(good_trials))

"""Reduce the regressors to only include the good trials"""
regressors = construct_regressors(data, 10)
regressors = pd.DataFrame(regressors)
index_of_good_trials = good_trials.index.values
regressors = regressors[regressors.index.isin(good_trials.index)]
X = sm.add_constant(regressors, prepend=False)
print(regressors)

#
# #___-______________________

"""GLM functions"""
model = sm.GLM(choices, X, family = sm.families.Binomial()).fit()
weights = model.params
weights = weights[:-1]
weights.index = weights.index + 1

"""Reset indexes so the trial history can be aligned"""
#Regressor split up
cherry_weights = weights[:10]

grape_weights = weights[10:20].reset_index(drop=True)
grape_weights.index += 1

both_weights = weights[20:30].reset_index(drop=True)
both_weights.index += 1

neither_weights = weights[30:40].reset_index(drop=True)
neither_weights.index += 1

"""Plotting graphs"""
plt.plot(cherry_weights, color="r", label = "\u03B2 Cherry")
plt.plot(grape_weights, color="m", label = "\u03B2 Grape")
plt.plot(both_weights, color = "b", label = "\u03B2 Both")
plt.plot(neither_weights, color= "k", label = "\u03B2 Neither")
plt.axhline(y=0, color = "k", linewidth=0.3)
plt.title("GLM to a Behavioural Dataset")
plt.xlabel("Trials in the Past")
plt.ylabel("Regression Weights")
plt.legend()
plt.show()

"""------------------------------------"""

#Print the time of the process
print("")
print("--- %s seconds ---" % (time.time() - start_time))
print("")
