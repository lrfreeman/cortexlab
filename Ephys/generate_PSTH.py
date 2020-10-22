from process_timesync import *
from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import math

df, trial_df = process_timesync_data()

#Produce a graph for a given cell
df = df.loc[(df['cluster_ids'] == 1)]
#test
# print(df)

#Lock the PSTH to reward time
df["lock"] = df["Spike_Times"] - df["reward_times"]

#Extend data print rows
pd.set_option("display.max_rows", None, "display.max_columns", None)

#Test for why there arent positive lock times
# print(df[["lock","Spike_Times","reward_times"]])
# print(trial_df)

#Convert to deciseconds to ensure Y axis is firing rate per second / match bin size
# df["lock"] = df["lock"].apply(lambda x: x*20)

#Create bins that are decisecond
bins = np.arange(-1,3,0.2).tolist()
bins = [ round(elem, 2) for elem in bins ]
print("~~~~~~~~~~~~~~~~")
print("")
print("List of bins:",bins)
print("")
print("~~~~~~~~~~~~~~~~")

#Assign the bins to the dataframe to generate counts
df["bins_range"] = pd.cut(df["lock"], bins)
df["Bin_Id"] = (df["bins_range"].apply(lambda x: x.left))

# Split trial types
left_reward_trials =  df.loc[(df['left_rewards'] == 1)
                         & (df['right_rewards'] == 0)]
right_reward_trials = df.loc[(df['left_rewards'] == 0)
                         & (df['right_rewards'] == 1)]
no_reward =           df.loc[(df['left_rewards'] == 0)
                         & (df['right_rewards'] == 0)]
both_rewards =        df.loc[(df['left_rewards'] == 1)
                         & (df['right_rewards'] == 1)]

#Trial length
cherry_trials = len(trial_df.loc[(trial_df['left_rewards'] == 1)
                         & (trial_df['right_rewards'] == 0)])
grape_trials = len(trial_df.loc[(trial_df['left_rewards'] == 0)
                         & (trial_df['right_rewards'] == 1)])
noreward_trials = len(trial_df.loc[(trial_df['left_rewards'] == 0)
                         & (trial_df['right_rewards'] == 0)])
bothreward_trials = len(trial_df.loc[(trial_df['left_rewards'] == 1)
                         & (trial_df['right_rewards'] == 1)])

# #My own function for counting
# cherry_counts = left_reward_trials["Bin_Id"].value_counts(sort=False)
# cherry_counts = np.asarray(cherry_counts)
# cherry_c_trial = cherry_counts / cherry_trials

#Counting using np.histogram
cherry, bin_edges = np.histogram(left_reward_trials["lock"], bins=bins)
cherry = (cherry / cherry_trials)*5
cbincentres = 0.5*(bin_edges[1:]+bin_edges[:-1])
print("Cherry firing rates:",cherry)

grape, bin_edges = np.histogram(right_reward_trials["lock"], bins=bins)
grape = (grape / grape_trials)*5
gbincentres = 0.5*(bin_edges[1:]+bin_edges[:-1])
# print(grape)

norew, bin_edges = np.histogram(no_reward["lock"], bins=bins)
norew = (norew / noreward_trials)*5
nbincentres = 0.5*(bin_edges[1:]+bin_edges[:-1])
# print(norew)

bothrew, bin_edges = np.histogram(both_rewards["lock"], bins=bins)
bothrew = (bothrew / bothreward_trials)*5
bbincentres = 0.5*(bin_edges[1:]+bin_edges[:-1])

#Plot using counts
fig, ax = plt.subplots()
plt.plot(cbincentres,cherry, color='r', label="Cherry Reward")
plt.plot(gbincentres,grape, color='m', label="Grape Reward")
plt.plot(nbincentres,norew, color='k', label="No Reward")
plt.plot(bbincentres,bothrew, color='b', label="Both Reward")
ax.legend()
plt.title("PSTH for cluster ID: x")
plt.xlabel("Time from Outcome [s] (Spike Time - Reward Time)")
plt.xlim(right=3)
plt.xlim(left=-1)
plt.ylabel("Firing Rate (sp/s)")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()

#------------------------------Working seaborn code

# # Peristimulus time histogram (PSTH) visualization
# #Set binwidth to decisecond
# fig, ax = plt.subplots()
# sb.histplot(left_reward_trials["lock"],label="Cherry Reward",
#             stat="count",element="poly",fill=False,bins=bins, color='r')
# sb.histplot(right_reward_trials["lock"],label="Grape Reward",
#             stat="count",element="poly",fill=False,bins=bins, color='m')
# sb.histplot(no_reward["lock"],label="No_rewards",
#             stat="count",element="poly",fill=False,bins=bins, color='k')
# sb.histplot(both_rewards["lock"],label="Both Rewards",
#             stat="count",element="poly",fill=False,bins=bins, color='b')
# ax.legend()
# plt.title("PSTH for cluster ID: 1")
# plt.xlabel("Time from Outcome [s] (Spike Time - Reward Time)")
# plt.xlim(right=3)
# plt.xlim(left=-1)
# plt.ylabel("Firing Rate (sp/s)")
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# plt.show()
