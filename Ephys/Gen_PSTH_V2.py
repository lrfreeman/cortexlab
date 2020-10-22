from ingest_timesync import *
from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.pyplot as plt

#Create Trial DF
trial_df, spike_times, cluster_IDs = convert_mat('/Users/laurence/Desktop/Neuroscience/mproject/Data/aligned_physdata_KM011_2020-03-20_probe0.mat')
trial_df = trial_df.drop(columns=["nTrials"])

#Create Spike and Cluster ID DF
spike_df =  pd.DataFrame(spike_times, columns = ["Spike_Times"])
spike_df["cluster_ids"] = cluster_IDs

#Create bins that are 200ms
bins = np.arange(-1,3,0.2).tolist()
bins = [ round(elem, 2) for elem in bins ]

#Extend data print rows
# pd.set_option("display.max_rows", None, "display.max_columns", None)

#####Choose a cell#######
spike_df = spike_df.loc[(spike_df["cluster_ids"] == 31)]

# Counts of spikes per trial
lock_time = {}
spike_counts = {}
for trial in range(len(trial_df)):
    lock_time[trial] = trial_df["reward_times"][trial]
    counts, bin_edges = np.histogram(spike_df["Spike_Times"]-lock_time[trial], bins=bins)
    spike_counts[trial] = counts

#Define reward types
cherry_reward_trials =  trial_df.loc[(trial_df['left_rewards'] == 1) & (trial_df['right_rewards'] == 0)]
grape_reward_trials =  trial_df.loc[(trial_df['left_rewards'] == 0) & (trial_df['right_rewards'] == 1)]
both_reward_trials =  trial_df.loc[(trial_df['left_rewards'] == 1) & (trial_df['right_rewards'] == 1)]
no_reward_trials =  trial_df.loc[(trial_df['left_rewards'] == 0) & (trial_df['right_rewards'] == 0)]

# Seperate counts per trial type - do cherry first
cherry_spike_counts = [spike_counts[x] for x in range(len(trial_df)) if list(spike_counts.keys())[x] in cherry_reward_trials.index.values]
grape_spike_counts = [spike_counts[x] for x in range(len(trial_df)) if list(spike_counts.keys())[x] in grape_reward_trials.index.values]
bothreward_spike_counts = [spike_counts[x] for x in range(len(trial_df)) if list(spike_counts.keys())[x] in both_reward_trials.index.values]
noreward_spike_counts = [spike_counts[x] for x in range(len(trial_df)) if list(spike_counts.keys())[x] in no_reward_trials.index.values]

#Calculate counts per trial type for each bin
cherry_count = pd.DataFrame(cherry_spike_counts).sum(axis=0)
grape_count = pd.DataFrame(grape_spike_counts).sum(axis=0)
both_reward_count = pd.DataFrame(bothreward_spike_counts).sum(axis=0)
no_reward_count = pd.DataFrame(noreward_spike_counts).sum(axis=0)

#Length of each trial type
num_c_trials = len(cherry_spike_counts)
num_g_trials = len(grape_spike_counts)
num_bothreward_trials = len(bothreward_spike_counts)
num_noreward_trials = len(noreward_spike_counts)

#Cal bin cbincentres
bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])

#Calculate average firing rate of a neuron per second
cherry_hertz = (cherry_count / num_c_trials) * 5
grape_hertz = (grape_count / num_g_trials) * 5
bothreward_hertz = (both_reward_count / num_bothreward_trials) * 5
noreward_hertz = (no_reward_count / num_noreward_trials) * 5

#Tests
# print("Do all trial lengths add up to total number of trials - 738:", num_c_trials+num_g_trials+num_bothreward_trials+num_noreward_trials)

#Plot PSTH
fig, ax = plt.subplots()
plt.plot(bin_centres,cherry_hertz, color='r', label="Cherry Reward")
plt.plot(bin_centres,grape_hertz, color='m', label="Grape Reward")
plt.plot(bin_centres,noreward_hertz, color='k', label="No Reward")
plt.plot(bin_centres,bothreward_hertz, color='b', label="Both Reward")
ax.legend()
plt.title("PSTH for cluster ID: x")
plt.xlabel("Time from Outcome [s] (Spike Time - Reward Time)")
plt.xlim(right=3)
plt.xlim(left=-1)
plt.ylabel("Firing Rate (sp/s)")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()
