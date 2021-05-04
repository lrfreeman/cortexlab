#External imports
import statsmodels.api as sm
from statsmodels.formula.api import glm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#My imports
from construct_regressors import *

#Load data
data = load_data()
# print(data)

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
reward_seeking_weights = weights[:10]
choice_weights = weights[10:20].reset_index(drop=True)
choice_weights.index += 1
outcome_weights = weights[20:30].reset_index(drop=True)
outcome_weights.index += 1

"""Plotting graphs"""
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
ax1.plot(reward_seeking_weights)
ax1.set(title="Reward Seeking - Interaction term", ylabel="Regression Weights", xlabel="Trials in the past")
ax1.set_xlim(10,1)
ax1.set_ylim(-0.1,1.2)
ax2.plot(choice_weights)
ax2.set(title="Choice Perseveration", ylabel="Regression Weights", xlabel="Trials in the past")
ax2.set_xlim(10,1)
ax2.set_ylim(-0.1,1.2)
ax3.plot(outcome_weights)
ax3.set(title="Main Effect of Outcome", ylabel="Regression Weights", xlabel="Trials in the past")
ax3.set_xlim(10,1)
ax3.set_ylim(-0.1,1.2)
plt.show()
