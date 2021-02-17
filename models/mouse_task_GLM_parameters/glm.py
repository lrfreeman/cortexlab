#External imports
import statsmodels.api as sm
from statsmodels.formula.api import glm
import pandas as pd
import numpy as np
import glob
from matplotlib import pyplot as plt
import time

#My imports
from construct_regressors import *

#Performance checks
start_time = time.time()

"""Pull out all the session files contained within the data folder"""
sessions = glob.glob("/Users/laurence/Desktop/Neuroscience/kevin_projects/data/processed_physdata/Single_probes_for_cog_model/*.mat")

grape_parameter = []
cherry_parameter = []
both_parameter = []
neither_parameter = []

for session in range(len(sessions)):

    #Load session
    data = load_data(sessions[session])

    """Choices / Code to convert choices into binary to solve concat error"""
    """Only analyze trials that are non-vilations and of free choice."""
    good_trials = data.loc[(data["violations"] == 0)]
    good_trials = data.loc[(data["free"] == 1)]
    choices = np.asarray(good_trials["left_choices"])
    choices = np.array(choices, dtype=np.float)

    """Reduce the regressors to only include the good trials"""
    regressors = construct_regressors(data, 10)
    regressors = pd.DataFrame(regressors)
    index_of_good_trials = good_trials.index.values
    regressors = regressors[regressors.index.isin(good_trials.index)]
    X = sm.add_constant(regressors, prepend=False)

    """GLM functions"""
    model = sm.GLM(choices, X, family = sm.families.Binomial()).fit()
    weights = model.params
    weights = weights[:-1]
    weights.index = weights.index + 1

    """Reset indexes so the trial history can be aligned"""
    #Regressor split up
    cherry_weights = weights[:10]
    cherry_parameter.append(cherry_weights)

    grape_weights = weights[10:20].reset_index(drop=True)
    grape_weights.index += 1
    grape_parameter.append(grape_weights)

    both_weights = weights[20:30].reset_index(drop=True)
    both_weights.index += 1
    both_parameter.append(both_weights)

    neither_weights = weights[30:40].reset_index(drop=True)
    neither_weights.index += 1
    neither_parameter.append(neither_weights)

def avg_parameters(param):
    x = 0
    for i in range(len(param)):
        x = x + param[i]
    x = x / len(param)
    return(x)

cherry = avg_parameters(cherry_parameter)
grape = avg_parameters(grape_parameter)
both = avg_parameters(both_parameter)
neither = avg_parameters(neither_parameter)

"""Plotting graphs"""
plt.plot(cherry, color="r", label = "\u03B2 Cherry")
plt.plot(grape, color="m", label = "\u03B2 Grape")
plt.plot(both, color = "b", label = "\u03B2 Both")
plt.plot(neither, color= "k", label = "\u03B2 Neither")
plt.axhline(y=0, color = "k", linewidth=0.3)
plt.title("GLM to Mouse Dataset")
plt.xlabel("Trials in the Past")
plt.ylabel("Regression Weights")
plt.legend()
plt.show()

#Print the time of the process
print("")
print("--- %s seconds ---" % (time.time() - start_time))
print("")
