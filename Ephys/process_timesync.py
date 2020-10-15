from ingest_timesync import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import math
import seaborn as sb

def left_bound():
    return row.left

def process_timesync_data():
    #Load_mat_file
    df, spike_times, cluster_IDs = convert_mat('/Users/laurence/Desktop/Neuroscience/mproject/Data/aligned_physdata_KM011_2020-03-20_probe0.mat')
    trial_df = df

    #Set variables
    #Spike_times is a numpy.ndarray
    trial_start_times = np.asarray(df.iloc[:,-1])
    nTrials = len(df.iloc[:,0])
    # print(trial_start_times)

    #Create new spike_time df
    spike_df =  pd.DataFrame(spike_times, columns = ["Spike_Times"])
    spike_df["Trial"] = np.nan
    spike_df["cluster_ids"] = cluster_IDs

    #Remove limitations to printing df
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    #Assign bins to each spike time
    bins = trial_start_times
    spike_df["Trial"] = pd.cut(spike_df["Spike_Times"], bins)
    # print(spike_df)

    #Create new column and substract trial start time from each spike time
    # print(type(spike_df["Trial"][386828].left))
    spike_df["Lower Bound"] = (spike_df["Trial"].apply(lambda x: x.left))
    spike_df["Lower Bound"] = spike_df["Lower Bound"].astype(float)

    #Remove all rows with NaN that occur before the trials start or after trials have finsished
    spike_df = spike_df.dropna()

    # #Test to check length of df before dropping nan columns
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print(spike_df.tail())
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    #Merge two dataframes
    #First create an ID between each df that can be used for the merge - Use trial start time
    trunc = lambda x: math.trunc(x)
    spike_df["Trial ID"] = spike_df["Lower Bound"].apply(trunc)
    df["Trial ID"] = df["trial_start_times"].apply(trunc)
    df = spike_df.merge(df, on="Trial ID")
    df = df.drop(columns=['nTrials'])

    return(df, trial_df)
