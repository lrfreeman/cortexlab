"""Import Libraries"""
import sys
import numpy as np
import pandas as pd
#Change system path so I can call this module with scripts from other folders within the application
sys.path.insert(1,'/Users/laurence/Desktop/Neuroscience/kevin_projects/code/mousetask/electrophysiology')
from ingest_timesync import convert_mat

#Extend dfs
# pd.set_option("display.max_rows", None, "display.max_columns", None)

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

#Construct Regressors
def construct_regressors(data,nBack):
    #Set empty regressors
    regressor_grape =  np.zeros((1,nBack))
    regressor_cherry = np.zeros((1,nBack))
    regressor_both = np.zeros((1,nBack))
    regressor_neither = np.zeros((1,nBack))
    nChoices = len(data)
    regressors = np.zeros((nChoices,nBack * 4))

    # Update regregressors
    for choice in range(nChoices):
        #Which side was picked
        side = data.at[choice,'left_choices']

        #Fill in the regressor
        regressors[choice,0:10] = regressor_cherry
        regressors[choice,10:20] = regressor_grape
        regressors[choice,20:30] = regressor_both
        regressors[choice,30:40] = regressor_neither

        """% These letter variables will be one if the choice_i'th letter is their letter.
        Set them to zero, so that if it's a violation trial and they don't get set,
        they'll be halfway between their set values of -1 and 1"""

        #update reward history vectors
        #Don't understand
        reward = data.at[choice,'rewards']
        grape_reward = data.at[choice, 'left_rewards']
        cherry_reward = data.at[choice, 'right_rewards']
        both_rewards = data.at[choice, 'both_rewards']
        cherry = 0
        grape = 0
        both = 0
        neither = 0

        """For the regessors you ultimately need binary values. 1 and -1 is used if __name__ == '__main__':
            preference due to multiplication of 0 being difficult. c is arbitarily set as left as -1 and right as 1.
            Reward is of course given as 1 and omission as 0. And x is the product of c and r"""
        if side == 1:
            if cherry_reward == 1 and grape_reward == 0:
                cherry = 1
                grape = 0
                both = 0
                neither = 0
            elif cherry_reward == 0 and grape_reward == 1:
                cherry = 0
                grape = 1
                both = 0
                neither = 0
            elif cherry_reward ==1 and grape_reward == 1:
                cherry = 0
                grape = 0
                both = 1
                neither = 0
            elif cherry_reward == 0 and grape_reward == 0:
                cherry = 0
                grape = 0
                both = 0
                neither = 1
            else: print("Error")

        elif side == 0:
            if cherry_reward == 1 and grape_reward == 0:
                cherry = -1
                grape = 0
                both = 0
                neither = 0
            elif cherry_reward == 0 and grape_reward == 1:
                cherry = 0
                grape = -1
                both = 0
                neither = 0
            elif cherry_reward ==1 and grape_reward == 1:
                cherry = 0
                grape = 0
                both = -1
                neither = 0
            elif cherry_reward == 0 and grape_reward == 0:
                cherry = 0
                grape = 0
                both = 0
                neither = -1
            else: print("Error")

        else:
            print("Error, sided should only be 1 or 0")

        #Add regressor to start of vector and remove last value
        regressor_grape = np.insert(regressor_grape, 0, grape)
        regressor_grape = regressor_grape[:-1]
        regressor_cherry = np.insert(regressor_cherry, 0, cherry)
        regressor_cherry = regressor_cherry[:-1]
        regressor_both = np.insert(regressor_both, 0, both)
        regressor_both = regressor_both[:-1]
        regressor_neither = np.insert(regressor_neither, 0, neither)
        regressor_neither = regressor_neither[:-1]

    return(regressors)

#Functions required to test this module
# data = load_data()
# # print(data)
# regressors = construct_regressors(data,10)
# print(regressors)
# print("Shape of Regessors:", regressors.shape)
