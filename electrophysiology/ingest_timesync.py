from scipy.io import loadmat
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
    cluster_types = mat_dict["data"][0][0]["cluster_types"]

    #Convert to df
    df = pd.DataFrame(nTrials + left_choices + free + left_rewards + right_rewards + violations + reward_times + trial_start_times).T
    df.columns = ["nTrials", "left_choices","free", "left_rewards","right_rewards","violations", "reward_times", "trial_start_times"]
    return(df,spiketimes,cluster_IDs, cluster_types)

#Import time of each frame from video taken in rig
def import_frame_times(file):
    mat_dict = loadmat(file)
    frametimes = mat_dict["tVid"][0]
    return(frametimes)

#Creating a new function to ingest matlab file given new data fields
def ingest_mat(file):

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
    # locations_origin = [[row.flat[0] for row in line] for line in mat_dict["data"][0][0]["location_origin"]]

    #variables matched to a spike
    spiketimes = mat_dict["data"][0][0]["spiketimes"]
    cluster_IDs = mat_dict["data"][0][0]["cluster_ids"]
    cluster_types = mat_dict["data"][0][0]["cluster_types"]
    locations_origin = mat_dict["data"][0][0]["location_origin"]

    #Unit information
    locations_unit = np.asarray([[row.flat[0] for row in line] for line in mat_dict["data"][0][0]["locations_unit"]])
    brain_regions = {}
    for x in range(len(locations_unit.flatten())):
        brain_regions[x] = locations_unit.flatten()[x][-1]
    brain_regions = pd.DataFrame(brain_regions).T
    brain_regions.columns = ["regions"]

    #Convert to trial_df
    trial_df = pd.DataFrame(left_choices + free + left_rewards + right_rewards + violations + reward_times + trial_start_times).T
    trial_df.columns = ["left_choices","free", "left_rewards","right_rewards","violations", "reward_times", "trial_start_times"]

    #Convert to spike_df
    dic = {"spike_time" : spiketimes.flatten(),
           "cluster_ids" : cluster_IDs.flatten()}
    spike_df = pd.DataFrame.from_dict(dic)

    #Unit Tests
    assert len(trial_df) == nTrials[0][0], "Lenght of trial_df and num of trials don't match"

    return(trial_df, spike_df, brain_regions)
