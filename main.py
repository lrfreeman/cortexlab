import PredictLicking.is_licking as lick
import electrophysiology.ingest_timesync as ingest
from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

#Extend data print rows
# pd.set_option("display.max_rows", None, "display.max_columns", None)

#Performance checks
start_time = time.time()

#Configure the data - 24th Session
session_data = '/Users/laurence/Desktop/Neuroscience/mproject/data/processed_physdata/aligned_physdata_KM011_2020-03-24_probe1.mat'
frame_alignment_data = "/Users/laurence/Desktop/Neuroscience/mproject/data/KM011_video_timestamps/2020-03-24/face_timeStamps.mat"
dlc_video_csv = "/Users/laurence/Desktop/Neuroscience/mproject/data/24_faceDLC_resnet50_Master_ProjectAug13shuffle1_133500.csv"

#Global Variables
#Create bins that are 200ms
bins = np.arange(-1,3,0.2).tolist()
bins = [ round(elem, 2) for elem in bins ]

#Load the data
frame_times = ingest.import_frame_times(frame_alignment_data)
df = lick.generate_licking_times(frame_times, dlc_video_csv)
lick_df = lick.map_lick_to_trial_type(df,session_data)
total_frames = len(df)

#Load data for PSTH by generating trial df and spike df
def load_data_for_PSTH(session_data):
    #Create Trial DF
    trial_df, spike_times, cluster_IDs = ingest.convert_mat(session_data)
    trial_df = trial_df.drop(columns=["nTrials"])

    #Create Spike and Cluster ID DF
    spike_df =  pd.DataFrame(spike_times, columns = ["Spike_Times"])
    spike_df["cluster_ids"] = cluster_IDs
    return(trial_df,spike_df)
trial_df, spike_df = load_data_for_PSTH(session_data)

#A function to split lick data or spike data by trial type
def split_data_by_trial_type(data_frame):
    cherry_trial =  data_frame.loc[(data_frame['left_rewards'] == 1) & (data_frame['right_rewards'] == 0)]
    grape_trial =  data_frame.loc[(data_frame['left_rewards'] == 0) & (data_frame['right_rewards'] == 1)]
    both_reward_trials =  data_frame.loc[(data_frame['left_rewards'] == 1) & (data_frame['right_rewards'] == 1)]
    no_reward_trials =  data_frame.loc[(data_frame['left_rewards'] == 0) & (data_frame['right_rewards'] == 0)]
    return(cherry_trial,grape_trial,both_reward_trials,no_reward_trials)

#Split licking data by trial type to calculate total frames inorder to normalise licking visulisations
cherry_reward_lick_trials,grape_reward_lick_trials,both_reward_lick_trials,no_reward_lick_trials = split_data_by_trial_type(lick_df)

#Split data by trial type so spike data can be split by reward
cherry_reward_trials, grape_reward_trials, both_reward_trials, no_reward_trials = split_data_by_trial_type(trial_df)

# A function to return counts of spikes / x per trial into bins and lock to stimlus onset
def lock_and_count(time,bins,trial_df):
    lock_time = {}
    x_counts = {}
    for trial in range(len(trial_df)):
        lock_time[trial] = trial_df["reward_times"][trial]
        counts, bin_edges = np.histogram(time-lock_time[trial], bins=bins)
        x_counts[trial] = counts
    return(x_counts, bin_edges)

# #Function to see licking by trial type
# def lick_count_by_trial_type(trial_type):
#
#     #Segment by lick type
#     cherrylick = trial_type.loc[(trial_type["Cherry Lick"] == 1)]
#     print("Number of cherry licks for this trial type", len(cherrylick.values))
#     grapelick = trial_type.loc[(trial_type["Grape Lick"] == 1)]
#     print("Number of grape licks for this trial type", len(grapelick.values))
#     centerlick = trial_type.loc[(trial_type["Center Lick"] == 1)]
#
#     #Return lick counts
#     cherry_lick_counts, x = lock_and_count(cherrylick["Time Licking"],bins,trial_df)
#     grape_lick_counts, y = lock_and_count(grapelick["Time Licking"],bins,trial_df)
#     center_lick_counts, z = lock_and_count(centerlick["Time Licking"],bins,trial_df)
#
#     #-------------------
#     return(cherry_lick_counts,grape_lick_counts,center_lick_counts)

#Function for generating a PSTH
def generate_PSTH(trial_df,spike_df,cellID):

    #####Choose a cell#######
    spike_df = spike_df.loc[(spike_df["cluster_ids"] == cellID)]

    #Segment by lick type - new test
    cherry_licks_only = df.loc[(df["Cherry Lick"] == 1)]
    grape_licks_only = df.loc[(df["Grape Lick"] == 1)]
    center_licks_only = df.loc[(df["Center Lick"] == 1)]

    #Return spike counts and bin edges for a set of bins for a given trial data frame
    spike_counts, bin_edges = lock_and_count(spike_df["Spike_Times"],bins,trial_df)

    #New test
    new_cherry_lick_counts, bin_edges = lock_and_count(cherry_licks_only["Time Licking"],bins,trial_df)
    new_grape_lick_counts, bin_edges = lock_and_count(grape_licks_only["Time Licking"],bins,trial_df)
    new_center_lick_counts, bin_edges = lock_and_count(center_licks_only["Time Licking"],bins,trial_df)

    #Cal bin cbincentres
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])

    #Define trial type
    # cherry_lick_counts, grape_lick_counts, center_lick_counts = lick_count_by_trial_type(trial_type)

    # Seperate counts per trial type
    cherry_spike_counts = [spike_counts[x] for x in range(len(trial_df)) if list(spike_counts.keys())[x] in cherry_reward_trials.index.values]
    grape_spike_counts = [spike_counts[x] for x in range(len(trial_df)) if list(spike_counts.keys())[x] in grape_reward_trials.index.values]
    bothreward_spike_counts = [spike_counts[x] for x in range(len(trial_df)) if list(spike_counts.keys())[x] in both_reward_trials.index.values]
    noreward_spike_counts = [spike_counts[x] for x in range(len(trial_df)) if list(spike_counts.keys())[x] in no_reward_trials.index.values]

    #Trial type licks - new test
    cherry_trial_cherry_licks = [new_cherry_lick_counts[x] for x in range(len(trial_df)) if list(new_cherry_lick_counts.keys())[x] in cherry_reward_trials.index.values]
    cherry_trial_grape_licks = [new_grape_lick_counts[x] for x in range(len(trial_df)) if list(new_grape_lick_counts.keys())[x] in cherry_reward_trials.index.values]
    grape_trial_cherry_licks = [new_cherry_lick_counts[x] for x in range(len(trial_df)) if list(new_cherry_lick_counts.keys())[x] in grape_reward_trials.index.values]
    grape_trial_grape_licks = [new_grape_lick_counts[x] for x in range(len(trial_df)) if list(new_grape_lick_counts.keys())[x] in grape_reward_trials.index.values]

    #Calculate licks for each trial type and reward
    cherry_trial_cherry_licks_count = pd.DataFrame(cherry_trial_cherry_licks).sum(axis=0)
    cherry_trial_grape_licks_count = pd.DataFrame(cherry_trial_grape_licks).sum(axis=0)
    grape_trial_cherry_licks_count = pd.DataFrame(grape_trial_cherry_licks).sum(axis=0)
    grape_trial_grape_licks_count = pd.DataFrame(grape_trial_grape_licks).sum(axis=0)

    #Calculate counts per trial type for each bin
    cherry_count = pd.DataFrame(cherry_spike_counts).sum(axis=0)
    grape_count = pd.DataFrame(grape_spike_counts).sum(axis=0)
    both_reward_count = pd.DataFrame(bothreward_spike_counts).sum(axis=0)
    no_reward_count = pd.DataFrame(noreward_spike_counts).sum(axis=0)

    # #Calculate lick counts per trial type for each bin
    # cherry_lick_count = pd.DataFrame(cherry_lick_counts).sum(axis=1)
    # grape_lick_count = pd.DataFrame(grape_lick_counts).sum(axis=1)
    # center_lick_count = pd.DataFrame(center_lick_counts).sum(axis=1)

    # print(cherry_lick_count)

    #--------------------------------------------------------------------------

    #Calculate average firing rate of a neuron per second
    cherry_hertz = (cherry_count / len(cherry_spike_counts)) * 5
    grape_hertz = (grape_count / len(grape_spike_counts)) * 5
    bothreward_hertz = (both_reward_count / len(bothreward_spike_counts)) * 5
    noreward_hertz = (no_reward_count / len(noreward_spike_counts)) * 5

    #Calculate average licking rate per trial - % of licks for each frame
    # avg_cherry_lick = (cherry_lick_count / len(trial_type)*100)
    # avg_grape_lick = (grape_lick_count / len(trial_type)*100)
    # avg_cherry_lick = (cherry_lick_count / total_frames)*100
    # avg_grape_lick = (grape_lick_count / total_frames)*100
    # avg_center_lick = (center_lick_count / total_frames)*100
    normalised_cherry_trial_cherry_licks = cherry_trial_cherry_licks_count  / len(cherry_reward_lick_trials.values) * 100
    normalised_cherry_trial_grape_licks  = cherry_trial_grape_licks_count   / len(cherry_reward_lick_trials.values) * 100
    normalised_grape_trial_cherry_licks  = grape_trial_cherry_licks_count   / len(grape_reward_lick_trials.values) * 100
    normalised_grape_trial_grape_licks   = grape_trial_grape_licks_count    / len(grape_reward_lick_trials.values) * 100

    #--------------------------------------------------------------------------

    #Plot subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    ax1.plot(bin_centres,cherry_hertz, color='r', label="Cherry Reward")
    ax1.plot(bin_centres,grape_hertz, color='m', label="Grape Reward")
    ax1.plot(bin_centres,noreward_hertz, color='k', label="No Reward")
    ax1.plot(bin_centres,bothreward_hertz, color='b', label="Both Reward")
    ax1.legend(loc='upper right')
    ax1.set(title="PSTH for cluster ID 1", ylabel="Firing Rates (sp/s)")

    #Licking subplot
    ax2.plot(bin_centres, normalised_cherry_trial_cherry_licks, color='r', label="Lick of cherry spout")
    ax2.plot(bin_centres, normalised_cherry_trial_grape_licks, color='m', label="Lick of grape spout")
    # ax2.plot(bin_centres, avg_center_lick, color='k', label="Lick center of spouts")
    ax2.set(ylabel="Percentage of frames licking",xlabel="Time from Outcome [s] (Spike Time - Reward Time)", title="Cherry Reward Trials")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.legend(loc='upper right')

    #Licking subplot
    ax3.plot(bin_centres, normalised_grape_trial_cherry_licks, color='r', label="Lick of cherry spout")
    ax3.plot(bin_centres, normalised_grape_trial_grape_licks, color='m', label="Lick of grape spout")
    # ax2.plot(bin_centres, avg_center_lick, color='k', label="Lick center of spouts")
    ax3.set(ylabel="Percentage of frames licking",xlabel="Time from Outcome [s] (Spike Time - Reward Time)", title="Grape Reward Trials")
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.legend(loc='upper right')

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
    # plt.show()

#Generate the visulations
generate_PSTH(trial_df,spike_df,1)

#----------------------------------------------------------

#Tests
print("")
print("#############")
print("````````````")
print("Length of cherry data frame out of generate lick times", len(df[df["Cherry Lick"] == 1].values))
print("Length of grape data frame out of generate lick times", len(df[df["Grape Lick"] == 1].values))
print("````````````")
print("Length of cherry data frame out of mapped lick times", len(lick_df[lick_df["Cherry Lick"] == 1].values))
print("Length of grape data frame out of mapped lick times", len(lick_df[lick_df["Grape Lick"] == 1].values))
print("````````````")
print("len of cherry trials", len(cherry_reward_trials))
print("len of grape trials", len(grape_reward_trials))
print("````````````")
print("#############")
print("")


#Print the time of the process
print("")
print("--- %s seconds ---" % (time.time() - start_time))
print("")
