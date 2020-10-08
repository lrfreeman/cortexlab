from process_timesync import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

df = process_timesync_data()

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
sb.distplot(left_reward_trials["Normalised Spike Times"],bins = bins,label="Left Reward",hist=False)
sb.distplot(right_reward_trials["Normalised Spike Times"],bins = bins,label="Right Reward",hist=False)
sb.distplot(no_reward["Normalised Spike Times"],bins = bins,label="No Reward",hist=False)
sb.distplot(both_rewards["Normalised Spike Times"],bins = bins,label="Both Reward",hist=False)

ax.legend()

# plt.hist(spike_df["Normalised Spike Times"], bins = bins, histtype='step')
plt.title("Peristimulus time histogram (PSTH)")
plt.xlabel("Time from stimulus onset [s]")
plt.ylabel("Count of Spikes")
plt.show()
