from process_timesync import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

df, trial_df = process_timesync_data()

#Produce a graph for a given cell
df = df.loc[(df['Cluster ids'] == 1)]
print(df.head())

#Lock the PSTH to reward time
df["lock"] = df["reward_times"] - df["Spike_Times"]

# Split trial types
left_reward_trials =  df.loc[(df['left_rewards'] == 1)
                         & (df['right_rewards'] == 0)]
right_reward_trials = df.loc[(df['left_rewards'] == 0)
                         & (df['right_rewards'] == 1)]
no_reward =           df.loc[(df['left_rewards'] == 0)
                         & (df['right_rewards'] == 0)]
both_rewards =        df.loc[(df['left_rewards'] == 1)
                         & (df['right_rewards'] == 1)]

#Peristimulus time histogram (PSTH) visualization
fig, ax = plt.subplots()
sb.histplot(left_reward_trials["lock"],label="Cherry Reward",
            stat="frequency",element="poly",fill=False,binwidth=1)
sb.histplot(right_reward_trials["lock"],label="Grape Reward",
            stat="frequency",element="poly",fill=False,binwidth=1)
sb.histplot(no_reward["lock"],label="No_rewards",
            stat="frequency",element="poly",fill=False,binwidth=1)
sb.histplot(both_rewards["lock"],label="Both Rewards",
            stat="frequency",element="poly",fill=False,binwidth=1)
ax.legend()
plt.title("Peristimulus time histogram (PSTH) for each trial for cluster ID one")
plt.xlabel("Time from Outcome [s] (Reward Time - Spike Time)")
plt.xlim(right=7)
plt.xlim(left=-2)
#No longer spikes per second as bin width is 100ms need to x10
plt.ylabel("Firing Rate (sp/s)")
plt.show()
