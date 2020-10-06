from ingest_timesync import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
from tdt import read_block, epoc_filter
from numba import njit
from numba.typed import List
import sys

#Load_mat_file
df, spike_times = convert_mat('/Users/laurence/Desktop/Neuroscience/mproject/Data/aligned_physdata_KM011_2020-03-20_probe0.mat')
# print("~~~~~~~~~~~~~~~~~~~~~~")
# print(df)
# print("~~~~~~~~~~~~~~~~~~~~~~")

#Set variables
#Spike_times is a numpy.ndarray
trial_start_times = np.asarray(df.iloc[:,-1])
nTrials = len(df.iloc[:,0])
# print(trial_start_times)

#Create new spike_time df
spike_df =  pd.DataFrame(spike_times, columns = ["Spike_Times"])
spike_df["Trial"] = np.nan
# print(spike_time_df)

#Remove limitations to printing df
# pd.set_option("display.max_rows", None, "display.max_columns", None)

#Assign bins to each spike time
bins = trial_start_times
spike_df["Trial"] = pd.cut(spike_df["Spike_Times"], bins)
# print(spike_df)

#Create new column and substract trial start time from each spike time
# print(type(spike_df["Trial"][386828].left))
def left_bound():
    return row.left
spike_df["Lower Bound"] = (spike_df["Trial"].apply(lambda x: x.left))
spike_df["Lower Bound"] = spike_df["Lower Bound"].astype(float)
spike_df["Normalised Spike Times"] = spike_df["Spike_Times"] - spike_df["Lower Bound"]

#Split trial types
# left_reward_trials =
# right_reward_trials =
# no_reward =
# both_rewards =

#Peristimulus time histogram (PSTH) visualization
# bins = 100
# plt.hist(spike_df["Normalised Spike Times"], bins = bins, histtype='step')
# plt.title("Histogram where red line is reward time")
# plt.xlabel("Time from stimulus onset [s]")
# plt.ylabel("Count of Spikes")
# plt.show()
