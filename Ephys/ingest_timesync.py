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
    # spiketimes = [[row.flat[0] for row in line] for line in mat_dict["data"][0][0]["spiketimes"]]

    #Convert to df
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', None)
    # pd.set_option('display.max_colwidth', None)
    df = pd.DataFrame(nTrials + left_choices + free + left_rewards + right_rewards + violations + reward_times + trial_start_times).T
    df.columns = ["nTrials", "left_choices","free", "left_rewards","right_rewards","violations", "reward_times", "trial_start_times"]
    return(df,spiketimes,cluster_IDs)
