from ingest_timesync import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import math

#Load_mat_file
df, spike_times = convert_mat('/Users/laurence/Desktop/Neuroscience/mproject/Data/aligned_physdata_KM011_2020-03-20_probe0.mat')

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
pd.set_option("display.max_rows", None, "display.max_columns", None)

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

#Remove all rows with NaN that occur before the trials start or after trials have finsished
spike_df = spike_df.dropna()

#Merge two dataframes
#First create an ID between each df that can be used for the merge - Use trial start time
trunc = lambda x: math.trunc(x)
spike_df["Trial ID"] = spike_df["Lower Bound"].apply(trunc)
df["Trial ID"] = df["trial_start_times"].apply(trunc)
df = spike_df.merge(df, on="Trial ID")
df.drop(columns=['nTrials'])

# Split trial types
left_reward_trials = df.loc[(df['left_rewards'] == 1)
                         & (df['right_rewards'] == 0)]
right_reward_trials = df.loc[(df['left_rewards'] == 0)
                         & (df['right_rewards'] == 1)]
no_reward = df.loc[(df['left_rewards'] == 0)
                         & (df['right_rewards'] == 0)]
both_rewards = df.loc[(df['left_rewards'] == 1)
                         & (df['right_rewards'] == 1)]

#Peristimulus time histogram (PSTH) visualization
bins = 100
fig, ax = plt.subplots()
ax.hist([left_reward_trials["Normalised Spike Times"],
         right_reward_trials["Normalised Spike Times"],
         no_reward["Normalised Spike Times"],
         both_rewards["Normalised Spike Times"]],
         bins = bins, histtype='step',label=["Left Reward",
         "Right Reward", "No Reward", "Both_Rewards"])
ax.legend()

# plt.hist(spike_df["Normalised Spike Times"], bins = bins, histtype='step')
plt.title("Peristimulus time histogram (PSTH)")
plt.xlabel("Time from stimulus onset [s]")
plt.ylabel("Count of Spikes")
plt.show()
