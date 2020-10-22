from ingest_timesync import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import math

def process_timesync_data():
    #Load_mat_file
    df, spike_times, cluster_IDs = convert_mat('/Users/laurence/Desktop/Neuroscience/mproject/Data/aligned_physdata_KM011_2020-03-20_probe0.mat')
    trial_df = df

    #Create new spike_time df
    spike_df =  pd.DataFrame(spike_times, columns = ["Spike_Times"])
    spike_df["Trial"] = np.nan
    spike_df["cluster_ids"] = cluster_IDs
    spike_and_cluster_df = spike_df

    #Remove limitations to printing df
    # pd.set_option("display.max_rows", None, "display.max_columns", None)

    #Assign bins to each spike time using trial start time and the cut function
    trial_start_times = np.asarray(df.iloc[:,-1])
    spike_df["Trial"] = pd.cut(spike_df["Spike_Times"], trial_start_times)

    #Create new column and substract trial start time from each spike time
    # print(type(spike_df["Trial"][386828].left))
    spike_df["Lower Bound"] = (spike_df["Trial"].apply(lambda x: x.left))
    spike_df["Lower Bound"] = spike_df["Lower Bound"].astype(float)

    #Test for spike length
    # print("The lenght of the spike df",spike_df)
    # print(spike_df[["reward_times","Spike_Times"]])

    #Remove all rows with NaN that occur before the trials start or after trials have finsished
    # print(spike_df)
    spike_df = spike_df.dropna()

    #Merge two dataframes
    #First create an ID between each df that can be used for the merge - Use trial start time
    #trunc removes numbers after decimal, used to create ID
    trunc = lambda x: math.trunc(x)
    spike_df["Trial ID"] = spike_df["Lower Bound"].apply(trunc)
    df["Trial ID"] = df["trial_start_times"].apply(trunc)
    df = spike_df.merge(df, on="Trial ID")
    return(df, trial_df, spike_and_cluster_df)

#Tests
# df, trial_df = process_timesync_data()
# print(df)
