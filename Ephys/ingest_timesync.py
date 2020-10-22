from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import pandas as pd
import numpy as np

#Output a tuple consisting of a data frame and an array of spiketimes
def convert_mat(file):

    #Converts the matlab struct into a python dictionary with keys (['__header__', '__version__', '__globals__', 'data'])
    mat_dict = loadmat(file)

    #variables with same lenght of ntrails
    left_choices = [[row.flat[0] for row in line] for line in mat_dict["data"][0][0]["left_choices"]]
    free = [[row.flat[0] for row in line] for line in mat_dict["data"][0][0]["free"]]
    left_rewards = [[row.flat[0] for row in line] for line in mat_dict["data"][0][0]["left_rewards"]]
    right_rewards = [[row.flat[0] for row in line] for line in mat_dict["data"][0][0]["right_rewards"]]
    violations = [[row.flat[0] for row in line] for line in mat_dict["data"][0][0]["violations"]]
    reward_times = [[row.flat[0] for row in line] for line in mat_dict["data"][0][0]["rewardTimes"]]
    trial_start_times = [[row.flat[0] for row in line] for line in mat_dict["data"][0][0]["trialstartTimes"]]
    nTrials = [[row.flat[0] for row in line] for line in mat_dict["data"][0][0]["nTrials"]]

    #variables with not same lenght
    spiketimes = mat_dict["data"][0][0]["spiketimes"]
    cluster_IDs = mat_dict["data"][0][0]["cluster_ids"]

    #Convert to df
    df = pd.DataFrame(nTrials + left_choices + free + left_rewards + right_rewards + violations + reward_times + trial_start_times).T
    df.columns = ["nTrials", "left_choices","free", "left_rewards","right_rewards","violations", "reward_times", "trial_start_times"]
    return(df,spiketimes,cluster_IDs)

#Test lenghts - Passed lenght tests
# file = '/Users/laurence/Desktop/Neuroscience/mproject/Data/aligned_physdata_KM011_2020-03-20_probe0.mat'
# df, spiketimes, clusterid = convert_mat(file)
# left_reward_trials =  df.loc[(df['left_rewards'] == 1)
#                          & (df['right_rewards'] == 0)]
# print(len(left_reward_trials))
# print("Does spiketime length",len(spiketimes),"equal the length of original matlab file=",3850350)
# print("Does clusterid length",len(clusterid),"equal the length of original matlab file=",3850350)
