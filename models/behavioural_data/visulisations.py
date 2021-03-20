"""Import Libraries"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Change system path so I can call this module with scripts from other folders within the application
sys.path.insert(1,'/Users/laurence/Desktop/Neuroscience/kevin_projects/code/mousetask/electrophysiology')
from ingest_timesync import convert_mat

#Extend dfs
# pd.set_option("display.max_rows", None, "display.max_columns", None)

file = "/Users/laurence/Desktop/Neuroscience/kevin_projects/data/processed_physdata/aligned_physdata_KM011_2020-03-21_probe0.mat"

def load_data(file):

    #Create trial df for mouse task
    # session_data = '/Users/laurence/Desktop/Neuroscience/kevin_projects/data/processed_physdata/aligned_physdata_KM011_2020-03-23_probe1.mat'
    trial_df, spike_times, cluster_IDs, cluster_types = convert_mat(file)
    data = trial_df.drop(columns=["nTrials", "trial_start_times", "reward_times"])

    #Create a single reward column as a boolean
    data["rewards"] = 0
    data["both_rewards"] = 0
    reward_data =  data.loc[(data["left_rewards"] == 1) | (data["right_rewards"] == 1)]
    both_reward_data = data.loc[(data["left_rewards"] == 1) & (data["right_rewards"] == 1)]
    reward_index = reward_data.index.values
    both_reward_data_index = both_reward_data.index.values
    data["rewards"].values[reward_index] = 1
    data["both_rewards"].values[both_reward_data_index] = 1

    return(data)

data = load_data(file)
print(data)
# n_trials = len(data)

"""Choices / Code to convert choices into binary to solve concat error"""
"""Only analyze trials that are non-vilations and of free choice."""
good_trials = data.loc[(data["violations"] == 0)]
good_trials = data.loc[(data["free"] == 1)]
choices = good_trials["left_choices"]
choices = choices.reset_index(drop=True)
choices = choices.to_frame('Choices')
right_choices = choices[choices == 0].dropna()
left_choices  = choices[choices == 1].dropna()
right_choice_trials = len(right_choices)
left_choice_trials = len(left_choices)

# #---------------------------------------------------------
#Create visulisations
fig, (ax1) = plt.subplots(nrows = 1, ncols = 1)
ax1.scatter(right_choices.index.values,right_choices, 1, label="Right Choices")
ax1.scatter(left_choices.index.values, left_choices, 1,  label="Left Choices")
plt.legend()
ax1.set(title = 'Animal Choices',
        xlabel = 'Trial Number',
        ylabel = 'Choice')


# plt.show()
