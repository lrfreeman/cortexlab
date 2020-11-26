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

#Create bins that are 200ms
bins = np.arange(-1,3,0.2).tolist()
bins = [ round(elem, 2) for elem in bins ]

#Load the data
frame_times = ingest.import_frame_times(frame_alignment_data)
df = lick.generate_licking_times(frame_times, dlc_video_csv)
lick_df = lick.map_lick_to_trial_type(df,session_data)
total_frames = len(df)

#test for generating first lick
first_lick_df = lick.compute_1st_lick(lick_df)

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

# A function to return counts of spikes / x per trial into bins and lock to first lick
def lock_and_count(time,bins,trial_df):
    lock_time = {}
    x_counts = {}
    for trial in range(len(first_lick_df)):
        lock_time[trial] = first_lick_df["First Lick Times"][trial]
        counts, bin_edges = np.histogram(time-lock_time[trial], bins=bins)
        x_counts[trial] = counts
    return(x_counts, bin_edges)

#Copy function and adding first lick df logic
def count_to_trial(trial_type, data_counts):
    count = [data_counts[x] for x in range(len(first_lick_df)) if list(data_counts.keys())[x] in trial_type.index.values]
    return(count)

#Function for generating a PSTH
def generate_PSTH(trial_df,spike_df,cellID):

    #####Choose a cell#######
    spike_df = spike_df.loc[(spike_df["cluster_ids"] == cellID)]

    #Segment by lick type - new test
    cherry_licks_only = df.loc[(df["Cherry Lick"] == 1)]
    grape_licks_only = df.loc[(df["Grape Lick"] == 1)]
    center_licks_only = df.loc[(df["Center Lick"] == 1)]

    #Return spike counts and bin edges for a set of bins for a given trial data frame
    spike_counts, bin_edges = lock_and_count(spike_df["Spike_Times"],bins,first_lick_df)

    #New test
    new_cherry_lick_counts, bin_edges = lock_and_count(cherry_licks_only["Time Licking"],bins,first_lick_df)
    new_grape_lick_counts, bin_edges = lock_and_count(grape_licks_only["Time Licking"],bins,first_lick_df)
    new_center_lick_counts, bin_edges = lock_and_count(center_licks_only["Time Licking"],bins,first_lick_df)

    #Cal bin cbincentres
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])

    # Seperate counts per trial type
    cherry_spike_counts = [spike_counts[x] for x in range(len(first_lick_df)) if list(spike_counts.keys())[x] in cherry_reward_trials.index.values]
    grape_spike_counts = [spike_counts[x] for x in range(len(first_lick_df)) if list(spike_counts.keys())[x] in grape_reward_trials.index.values]
    bothreward_spike_counts = [spike_counts[x] for x in range(len(first_lick_df)) if list(spike_counts.keys())[x] in both_reward_trials.index.values]
    noreward_spike_counts = [spike_counts[x] for x in range(len(first_lick_df)) if list(spike_counts.keys())[x] in no_reward_trials.index.values]

    #Lick counts not split by trial
    total_cherry_lick_counts = pd.DataFrame(new_cherry_lick_counts).sum(axis=0)
    total_grape_lick_counts = pd.DataFrame(new_grape_lick_counts).sum(axis=0)
    total_center_lick_counts = pd.DataFrame(new_center_lick_counts).sum(axis=0)

    #Trial type licks - new test
    cherry_trial_cherry_licks = count_to_trial(cherry_reward_trials, new_cherry_lick_counts)
    cherry_trial_grape_licks = count_to_trial(cherry_reward_trials, new_grape_lick_counts)
    grape_trial_cherry_licks = count_to_trial(grape_reward_trials, new_cherry_lick_counts)
    grape_trial_grape_licks = count_to_trial(grape_reward_trials, new_grape_lick_counts)
    both_reward_trial_cherry_licks = count_to_trial(both_reward_trials, new_cherry_lick_counts)
    both_reward_trial_grape_licks = count_to_trial(both_reward_trials, new_grape_lick_counts)
    no_reward_trial_cherry_licks = count_to_trial(no_reward_trials, new_cherry_lick_counts)
    no_reward_trial_grape_licks = count_to_trial(no_reward_trials, new_grape_lick_counts)

    #Calculate licks for each trial type and reward
    cherry_trial_cherry_licks_count = pd.DataFrame(cherry_trial_cherry_licks).sum(axis=0)
    cherry_trial_grape_licks_count = pd.DataFrame(cherry_trial_grape_licks).sum(axis=0)
    grape_trial_cherry_licks_count = pd.DataFrame(grape_trial_cherry_licks).sum(axis=0)
    grape_trial_grape_licks_count = pd.DataFrame(grape_trial_grape_licks).sum(axis=0)
    both_reward_trial_cherry_licks = pd.DataFrame(both_reward_trial_cherry_licks).sum(axis=0)
    both_reward_trial_grape_licks = pd.DataFrame(both_reward_trial_grape_licks).sum(axis=0)
    no_reward_trial_cherry_licks = pd.DataFrame(no_reward_trial_cherry_licks).sum(axis=0)
    no_reward_trial_grape_licks = pd.DataFrame(no_reward_trial_grape_licks).sum(axis=0)

    #Calculate spike counts per trial type for each bin
    cherry_count = pd.DataFrame(cherry_spike_counts).sum(axis=0)
    grape_count = pd.DataFrame(grape_spike_counts).sum(axis=0)
    both_reward_count = pd.DataFrame(bothreward_spike_counts).sum(axis=0)
    no_reward_count = pd.DataFrame(noreward_spike_counts).sum(axis=0)

    #--------------------------------------------------------------------------

    #Calculate average firing rate of a neuron per second
    cherry_hertz = (cherry_count / len(cherry_spike_counts)) * 5
    grape_hertz = (grape_count / len(grape_spike_counts)) * 5
    bothreward_hertz = (both_reward_count / len(bothreward_spike_counts)) * 5
    noreward_hertz = (no_reward_count / len(noreward_spike_counts)) * 5

    #Calculate average licking rate per trial - % of licks for each frame
    normalised_cherry_trial_cherry_licks = cherry_trial_cherry_licks_count  / len(cherry_reward_lick_trials.values) * 100
    normalised_cherry_trial_grape_licks  = cherry_trial_grape_licks_count   / len(cherry_reward_lick_trials.values) * 100
    normalised_grape_trial_cherry_licks  = grape_trial_cherry_licks_count   / len(grape_reward_lick_trials.values) * 100
    normalised_grape_trial_grape_licks   = grape_trial_grape_licks_count    / len(grape_reward_lick_trials.values) * 100
    normalised_both_reward_trial_cherry_licks = both_reward_trial_cherry_licks / len(both_reward_lick_trials.values) * 100
    normalised_both_reward_trial_grape_licks = both_reward_trial_grape_licks / len(both_reward_lick_trials.values) * 100
    normalised_no_reward_trial_cherry_licks = no_reward_trial_cherry_licks / len(no_reward_lick_trials.values) * 100
    normalised_no_reward_trial_grape_licks = no_reward_trial_grape_licks / len(no_reward_lick_trials.values) * 100
    normalised_total_cherry_licks = total_cherry_lick_counts / len(cherry_reward_trials) * 100
    normalised_total_grape_licks = total_grape_lick_counts / len(grape_reward_trials) * 100

    #--------------------------------------------------------------------------

    #Plot subplots
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(bin_centres,cherry_hertz, color='r', label="Cherry Reward")
    ax1.plot(bin_centres,grape_hertz, color='m', label="Grape Reward")
    ax1.plot(bin_centres,noreward_hertz, color='k', label="No Reward")
    ax1.plot(bin_centres,bothreward_hertz, color='b', label="Both Reward")
    ax1.legend(loc='upper right')
    ax1.set(title="PSTH for cluster ID 1", ylabel="Firing Rates (sp/s)")

    # #Licking subplot
    # ax2.plot(bin_centres, normalised_total_cherry_licks, color='r', label="Lick of cherry spout")
    # ax2.plot(bin_centres, normalised_total_grape_licks, color='m', label="Lick of grape spout")
    # # ax2.plot(bin_centres, avg_center_lick, color='k', label="Lick center of spouts")
    # ax2.set(ylabel="Perc. frames licking", title="Licking")
    # ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    # ax2.legend(loc='upper right')
    # ax2.set_ylim([0, 18])

    # #Licking subplot
    # ax2.plot(bin_centres, normalised_cherry_trial_cherry_licks, color='r', label="Lick of cherry spout")
    # ax2.plot(bin_centres, normalised_cherry_trial_grape_licks, color='m', label="Lick of grape spout")
    # # ax2.plot(bin_centres, avg_center_lick, color='k', label="Lick center of spouts")
    # ax2.set(ylabel="Perc. frames licking", title="Cherry Reward Trials")
    # ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    # ax2.legend(loc='upper right')
    # ax2.set_ylim([0, 18])
    #
    # #Licking subplot
    # ax3.plot(bin_centres, normalised_grape_trial_cherry_licks, color='r', label="Lick of cherry spout")
    # ax3.plot(bin_centres, normalised_grape_trial_grape_licks, color='m', label="Lick of grape spout")
    # # ax2.plot(bin_centres, avg_center_lick, color='k', label="Lick center of spouts")
    # ax3.set(ylabel="Perc. frames licking", title="Grape Reward Trials")
    # ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
    # ax3.legend(loc='upper right')
    # ax3.set_ylim([0, 18])
    #
    # #Licking subplot
    # ax4.plot(bin_centres, normalised_both_reward_trial_cherry_licks, color='r', label="Lick of cherry spout")
    # ax4.plot(bin_centres, normalised_both_reward_trial_grape_licks, color='m', label="Lick of grape spout")
    # # ax2.plot(bin_centres, avg_center_lick, color='k', label="Lick center of spouts")
    # ax4.set(ylabel="Perc. frames licking", title="Both Reward Trials")
    # ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax4.yaxis.set_major_locator(MaxNLocator(integer=True))
    # ax4.legend(loc='upper right')
    # ax4.set_ylim([0, 18])
    #
    # #Licking subplot
    # ax5.plot(bin_centres, normalised_no_reward_trial_cherry_licks, color='r', label="Lick of cherry spout")
    # ax5.plot(bin_centres, normalised_no_reward_trial_grape_licks, color='m', label="Lick of grape spout")
    # # ax2.plot(bin_centres, avg_center_lick, color='k', label="Lick center of spouts")
    # ax5.set(ylabel="Perc. frames licking",xlabel="Time from Outcome [s] (Spike Time - Reward Time)", title="No Reward Trials")
    # ax5.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax5.yaxis.set_major_locator(MaxNLocator(integer=True))
    # ax5.legend(loc='upper right')
    # ax5.set_ylim([0, 18])

    #Show plot
    plt.show()

#Function to generate Spike Raster
def generate_raster(file,cellID):
    #Load data
    trial_df, spike_df = load_data_for_PSTH(session_data)

    #####Choose a cell#######
    spike_df = spike_df.loc[(spike_df["cluster_ids"] == cellID)]

    #Return spike counts and bin edges for a set of bins for a given trial data frame
    spike_counts, bin_edges = lock_and_count(spike_df["Spike_Times"],bins,trial_df)

    # Seperate counts per trial type
    cherry_spike_counts = count_to_trial(cherry_reward_trials, spike_counts)
    grape_spike_counts = count_to_trial(grape_reward_trials, spike_counts)
    both_reward_spike_counts = count_to_trial(both_reward_trials, spike_counts)
    no_reward_spike_counts = count_to_trial(no_reward_trials, spike_counts)

    cherry_spike_counts = np.array(cherry_spike_counts)
    print(cherry_spike_counts.shape)

    # data = df.transpose().values.tolist()

    # #Convert to np array as faster operation speed
    # data = np.array(data)

    #Generate raster - Very slow - can I optimise the code? As the google colab wasnt that slow
    plt.eventplot(cherry_spike_counts, color=".2")
    plt.xlim(right=3)
    plt.xlim(left=-1)
    plt.xlabel("Time (s)")
    plt.yticks([])
    plt.show()

#Generate the visulations
generate_PSTH(trial_df,spike_df,56)
# generate_raster(session_data,1)

#----------------------------------------------------------

# #Tests
# print("")
# print("#############")
# print("````````````")
# print("Length of cherry data frame out of generate lick times", len(df[df["Cherry Lick"] == 1].values))
# print("Length of grape data frame out of generate lick times", len(df[df["Grape Lick"] == 1].values))
# print("````````````")
# print("Length of cherry data frame out of mapped lick times", len(lick_df[lick_df["Cherry Lick"] == 1].values))
# print("Length of grape data frame out of mapped lick times", len(lick_df[lick_df["Grape Lick"] == 1].values))
# print("````````````")
# print("len of cherry trials", len(cherry_reward_trials))
# print("len of grape trials",  len(grape_reward_trials))
# print("````````````")
# print("#############")
# print("")


#Print the time of the process
print("")
print("--- %s seconds ---" % (time.time() - start_time))
print("")
