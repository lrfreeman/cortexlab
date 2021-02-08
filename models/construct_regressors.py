"""Import Libraries"""
import sys
import numpy as np
import pandas as pd
#Change system path so I can call this module with scripts from other folders within the application
sys.path.insert(1,'/Users/laurence/Desktop/Neuroscience/kevin_projects/code/mousetask/electrophysiology')
from ingest_timesync import convert_mat

#Extend dfs
# pd.set_option("display.max_rows", None, "display.max_columns", None)

def load_data():

    #Create trial df for mouse task
    session_data = '/Users/laurence/Desktop/Neuroscience/kevin_projects/data/processed_physdata/aligned_physdata_KM011_2020-03-23_probe1.mat'
    trial_df, spike_times, cluster_IDs, cluster_types = convert_mat(session_data)
    data = trial_df.drop(columns=["nTrials", "trial_start_times", "reward_times"])

    #Create a single reward column as a boolean
    data["rewards"] = 0
    reward_data =  data.loc[(data["left_rewards"] == 1) | (data["right_rewards"] == 1)]
    reward_index = reward_data.index.values
    data["rewards"].values[reward_index] = 1

    return(data)

#Construct Regressors
def construct_regressors(data,nBack):
    #Set empty regressors
    regressor_ch =  np.zeros((1,nBack))
    regressor_cxr = np.zeros((1,nBack))
    regressor_rew = np.zeros((1,nBack))
    nChoices = len(data)
    regressors = np.zeros((nChoices,nBack * 3))

    # Update regregressors
    for choice in range(nChoices):
        #Which side was picked
        side = data.at[choice,'left_choices']

        #Fill in the regressor
        regressors[choice,0:10] = regressor_cxr
        regressors[choice,10:20] = regressor_ch
        regressors[choice,20:30] = regressor_rew

        """% These letter variables will be one if the choice_i'th letter is their letter.
        Set them to zero, so that if it's a violation trial and they don't get set,
        they'll be halfway between their set values of -1 and 1"""

        #update reward history vectors
        #Don't understand
        reward = data.at[choice,'rewards']
        c = 0
        x = 0
        r = 0

        """For the regessors you ultimately need binary values. 1 and -1 is used if __name__ == '__main__':
            preference due to multiplication of 0 being difficult. c is arbitarily set as left as -1 and right as 1.
            Reward is of course given as 1 and omission as 0. And x is the product of c and r"""
        if side == 1:
            if reward == 1:
                c = -1
                x = -1
                r = 1
            else:
                c = -1
                x = 1
                r = -1

        elif side == 0:
            if reward == 1:
                c = 1
                x = 1
                r = 1
            else:
                c = 1
                x = -1
                r = -1

        else:
            print("Error, should only be 1 or 0")

        #Add regressor to start of vector and remove last value
        regressor_ch = np.insert(regressor_ch, 0,c)
        regressor_ch = regressor_ch[:-1]
        regressor_cxr = np.insert(regressor_cxr, 0, x)
        regressor_cxr = regressor_cxr[:-1]
        regressor_rew = np.insert(regressor_rew, 0, r)
        regressor_rew = regressor_rew[:-1]

    return(regressors)

#Functions required to test this module
# data = load_data()
# regressors = construct_regressors(data,10)
# print(regressors)
# print("Shape of Regessors:", regressors.shape)
