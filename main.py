import PredictLicking.is_licking as lick
import electrophysiology.ingest_timesync as ingest
from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

#Performance checks
start_time = time.time()

#Configure the data
session_data = '/Users/laurence/Desktop/Neuroscience/mproject/data/processed_physdata/aligned_physdata_KM011_2020-03-24_probe1.mat'
frame_alignment_data = "/Users/laurence/Desktop/Neuroscience/mproject/data/KM011_video_timestamps/2020-03-24/face_timeStamps.mat"
dlc_video_csv = "/Users/laurence/Desktop/Neuroscience/mproject/data/24_3minsDLC_resnet50_Master_ProjectAug13shuffle1_133500.csv"

#Load the data
frame_times = ingest.import_frame_times(frame_alignment_data)
df = lick.generate_licking_times(frame_times, dlc_video_csv)
lick_df = lick.lick2trial(df,session_data)
total_frames = len(df)

#Function for generating a PSTH
def generate_PSTH(file,cellID):
    #Create Trial DF
    trial_df, spike_times, cluster_IDs = ingest.convert_mat(file)
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
    spike_df = spike_df.loc[(spike_df["cluster_ids"] == cellID)]

    # A function to return counts of spikes / x per trial into bins
    def lock_and_count(x_time):
        lock_time = {}
        x_counts = {}
        for trial in range(len(trial_df)):
            lock_time[trial] = trial_df["reward_times"][trial]
            counts, bin_edges = np.histogram(x_time-lock_time[trial], bins=bins)
            x_counts[trial] = counts
        return(x_counts, bin_edges)

    #Return spike
    spike_counts, bin_edges = lock_and_count(spike_df["Spike_Times"])

    #Define reward types for licking
    cherry_reward_lick_trials =  lick_df.loc[(lick_df['left_rewards'] == 1) & (lick_df['right_rewards'] == 0)]
    grape_reward_lick_trials =  lick_df.loc[(lick_df['left_rewards'] == 0) & (lick_df['right_rewards'] == 1)]
    both_reward_lick_trials =  lick_df.loc[(lick_df['left_rewards'] == 1) & (lick_df['right_rewards'] == 1)]
    no_reward_lick_trials =  lick_df.loc[(lick_df['left_rewards'] == 0) & (lick_df['right_rewards'] == 0)]

    def lick_count_by_trial_type(trial_type):

        #Segment by lick type
        cherrylick = trial_type.loc[(trial_type["Cherry Lick"] == 1)]
        grapelick = trial_type.loc[(trial_type["Grape Lick"] == 1)]
        centerlick = trial_type.loc[(trial_type["Center Lick"] == 1)]

        #Return lick counts
        cherry_lick_counts, x = lock_and_count(cherrylick["Time Licking"])
        grape_lick_counts, y = lock_and_count(grapelick["Time Licking"])
        center_lick_counts, z = lock_and_count(centerlick["Time Licking"])

        #-------------------
        return(cherry_lick_counts,grape_lick_counts,center_lick_counts)

    #Define trial type
    cherry_lick_counts, grape_lick_counts, center_lick_counts = lick_count_by_trial_type(cherry_reward_lick_trials)

    #Define reward types
    cherry_reward_trials =  trial_df.loc[(trial_df['left_rewards'] == 1) & (trial_df['right_rewards'] == 0)]
    grape_reward_trials =  trial_df.loc[(trial_df['left_rewards'] == 0) & (trial_df['right_rewards'] == 1)]
    both_reward_trials =  trial_df.loc[(trial_df['left_rewards'] == 1) & (trial_df['right_rewards'] == 1)]
    no_reward_trials =  trial_df.loc[(trial_df['left_rewards'] == 0) & (trial_df['right_rewards'] == 0)]

    # Seperate counts per trial type
    cherry_spike_counts = [spike_counts[x] for x in range(len(trial_df)) if list(spike_counts.keys())[x] in cherry_reward_trials.index.values]
    grape_spike_counts = [spike_counts[x] for x in range(len(trial_df)) if list(spike_counts.keys())[x] in grape_reward_trials.index.values]
    bothreward_spike_counts = [spike_counts[x] for x in range(len(trial_df)) if list(spike_counts.keys())[x] in both_reward_trials.index.values]
    noreward_spike_counts = [spike_counts[x] for x in range(len(trial_df)) if list(spike_counts.keys())[x] in no_reward_trials.index.values]

    #Calculate counts per trial type for each bin
    cherry_count = pd.DataFrame(cherry_spike_counts).sum(axis=0)
    grape_count = pd.DataFrame(grape_spike_counts).sum(axis=0)
    both_reward_count = pd.DataFrame(bothreward_spike_counts).sum(axis=0)
    no_reward_count = pd.DataFrame(noreward_spike_counts).sum(axis=0)

    #Calculate lick counts per trial type for each bin
    cherry_lick_count = pd.DataFrame(cherry_lick_counts).sum(axis=1)
    grape_lick_count = pd.DataFrame(grape_lick_counts).sum(axis=1)
    center_lick_count = pd.DataFrame(center_lick_counts).sum(axis=1)

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

    #Calculate average licking rate per trial - % of licks for each frame
    avg_cherry_lick = (cherry_lick_count / total_frames)*100
    avg_grape_lick = (grape_lick_count / total_frames)*100
    avg_center_lick = (center_lick_count / total_frames)*100

    #Plot subplots
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(bin_centres,cherry_hertz, color='r', label="Cherry Reward")
    ax1.plot(bin_centres,grape_hertz, color='m', label="Grape Reward")
    ax1.plot(bin_centres,noreward_hertz, color='k', label="No Reward")
    ax1.plot(bin_centres,bothreward_hertz, color='b', label="Both Reward")
    ax1.legend(loc='upper right')
    ax1.set(title="PSTH for cluster ID 1", ylabel="Firing Rates (sp/s)")

    #Licking subplot
    ax2.plot(bin_centres, avg_cherry_lick, color='r', label="Lick of cherry spout")
    ax2.plot(bin_centres, avg_grape_lick, color='m', label="Lick of grape spout")
    ax2.plot(bin_centres, avg_center_lick, color='k', label="Lick center of spouts")
    ax2.set(ylabel="Licking Rate (lick/s)",xlabel="Time from Outcome [s] (Spike Time - Reward Time)", title="Cherry Reward Trials")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.legend(loc='upper right')

    #Show plot
    plt.show()

#Function to generate Spike Raster
def generate_raster(file,cellID):
    #Create Trial DF
    trial_df, spike_times, cluster_IDs = convert_mat(file)
    trial_df = trial_df.drop(columns=["nTrials"])

    #Create Spike and Cluster ID DF
    spike_df =  pd.DataFrame(spike_times, columns = ["Spike_Times"])
    spike_df["cluster_ids"] = cluster_IDs

    #####Choose a cell#######
    spike_df = spike_df.loc[(spike_df["cluster_ids"] == cellID)]

    #Generate lock times
    lock_time = {}
    trial_spikes = {}
    for trial in range(len(trial_df)):
        lock_time[trial] = trial_df["reward_times"][trial]
        trial_spikes[trial] = (spike_df["Spike_Times"] - lock_time[trial])
    lockspikedf = pd.DataFrame(trial_spikes)
    data = lockspikedf.transpose().values.tolist()

    #Convert to np array as faster operation speed
    data = np.array(data)

    #Generate raster - Very slow - can I optimise the code? As the google colab wasnt that slow
    plt.eventplot(data[0], color=".2")
    plt.xlim(right=3)
    plt.xlim(left=-1)
    plt.xlabel("Time (s)")
    plt.yticks([])
    plt.show()

#Generate the visulations
generate_PSTH(session_data,1)
# generate_raster(session_data,1)

#Print the time of the process
print("--- %s seconds ---" % (time.time() - start_time))
