import PredictLicking.is_licking as lick
import matplotlib.backends.backend_pdf
import electrophysiology.ingest_timesync as ingest
from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import seaborn as sns

#Extend data print rows
# pd.set_option("display.max_rows", None, "display.max_columns", None)

#Performance checks
start_time = time.time()

#-------------------------------------------------------

# #Configure the data - 24th Session
# session_data = '/Users/laurence/Desktop/Neuroscience/mproject/data/processed_physdata/aligned_physdata_KM011_2020-03-24_probe1.mat'
# frame_alignment_data = "/Users/laurence/Desktop/Neuroscience/mproject/data/KM011_video_timestamps/2020-03-24/face_timeStamps.mat"
# dlc_video_csv = "/Users/laurence/Desktop/Neuroscience/mproject/data/24_faceDLC_resnet50_Master_ProjectAug13shuffle1_133500.csv"

#Configure the data - 23th Session
session_data = '/Users/laurence/Desktop/Neuroscience/mproject/data/processed_physdata/aligned_physdata_KM011_2020-03-23_probe1.mat'
frame_alignment_data = "/Users/laurence/Desktop/Neuroscience/mproject/data/KM011_video_timestamps/2020-03-23/face_timeStamps.mat"
dlc_video_csv = "/Users/laurence/Desktop/Neuroscience/mproject/data/23_faceDLC_resnet50_Master_ProjectAug13shuffle1_133500.csv"

# #Configure the data - 20th Session
# session_data = '/Users/laurence/Desktop/Neuroscience/mproject/data/processed_physdata/aligned_physdata_KM011_2020-03-20_probe0.mat'

#-------------------------------------------------------

#Create bins that are 200ms
bins = np.arange(-1,3,0.2).tolist()
bins = [ round(elem, 2) for elem in bins ]

#Load the data
frame_times = ingest.import_frame_times(frame_alignment_data)
df = lick.generate_licking_times(frame_times, dlc_video_csv)
lick_df = lick.map_lick_to_trial_type(df,session_data)
total_frames = len(df)

#Load data for PSTH by generating trial df and spike df
def load_data_for_graphs(session_data):
    #Create Trial DF
    trial_df, spike_times, cluster_IDs, cluster_types = ingest.convert_mat(session_data)
    trial_df = trial_df.drop(columns=["nTrials"])

    num_of_cluster_types = len(cluster_types[0])

    #Create Spike and Cluster ID DF
    spike_df =  pd.DataFrame(spike_times, columns = ["Spike_Times"])
    spike_df["cluster_ids"] = cluster_IDs
    return(trial_df,spike_df, num_of_cluster_types)
trial_df, spike_df, num_of_cluster_types = load_data_for_graphs(session_data)

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

#A function to return counts of spikes / licks per trial into bins and lock to reward time
def lock_and_count(time,bins,df):
    lock_time = {}
    x_counts = {}
    for trial in range(len(trial_df)):
        lock_time[trial] = trial_df["reward_times"][trial]
        counts, bin_edges = np.histogram(time-lock_time[trial], bins=bins)
        x_counts[trial] = counts
    return(x_counts, bin_edges)

#Count things and map them to trials
def count_to_trial(trial_type, data_counts):
    count = [data_counts[x] for x in range(len(trial_df)) if list(data_counts.keys())[x] in trial_type.index.values]
    return(count)

#Raster locked to reward - Returns a dictionary with key trial number and value locked spike times
def lock_and_sort_for_raster(time,trial_df):
    lock_time = {}
    trial_spike_times = {}
    for trial in range(len(trial_df)):
        lock_time[trial] = trial_df["reward_times"][trial]
        trial_spike_times[trial] = time-lock_time[trial]
    return(trial_spike_times)

# #Function to generate Spike Raster
# def generate_raster(trial_df, spike_df, cellID):
#
#     #####Choose a cell#######
#     spike_df = spike_df.loc[(spike_df["cluster_ids"] == cellID)]
#
#     #Generate spikes for each trial
#     trial_spike_times = lock_and_sort_for_raster(spike_df["Spike_Times"],trial_df)
#
#     # Seperate spikes per trial type
#     cherrySpikeValues = count_to_trial(cherry_reward_trials, trial_spike_times)
#     grapeSpikeValues = count_to_trial(grape_reward_trials, trial_spike_times)
#     bothRewardSpikeValues = count_to_trial(both_reward_trials, trial_spike_times)
#     noRewardSpikeValues = count_to_trial(no_reward_trials, trial_spike_times)
#
#     #SO that we can create a correspondding colour length for event plot
#     lenOfCherryTrials = len(cherrySpikeValues)
#     lenOfGrapeTrials = len(grapeSpikeValues)
#     lenOfBothRewardTrials = len(bothRewardSpikeValues)
#     lenOfNoRewardTrials = len(noRewardSpikeValues)
#
#     #convert to np array
#     cherrySpikeValues = np.asarray(cherrySpikeValues)
#     grapeSpikeValues = np.asarray(grapeSpikeValues)
#     bothRewardSpikeValues = np.asarray(bothRewardSpikeValues)
#     noRewardSpikeValues = np.asarray(noRewardSpikeValues)
#
#     #Concaternate arrays
#     spikes = np.concatenate((cherrySpikeValues,grapeSpikeValues,bothRewardSpikeValues,noRewardSpikeValues))
#
#     #Create colorCodes
#     colorCodesCherry = [[1,0,0]] * lenOfCherryTrials
#     colorCodesGrape = [[1,0,1]] * lenOfGrapeTrials
#     colorCodesBothReward = [[0,0,1]] * lenOfBothRewardTrials
#     colorCodesNoReward = [[0,0,0]] * lenOfNoRewardTrials
#     colorCodes = colorCodesCherry + colorCodesGrape + colorCodesBothReward + colorCodesNoReward
#
#     return(colorCodes, spikes)

#Function to generate Spike Raster
def new_generate_raster(trial_df, spike_df, cellID):

    #####Choose a cell#######
    spike_df = spike_df.loc[(spike_df["cluster_ids"] == cellID)]

    #Generate spikes for each trial
    trial_spike_times = lock_and_sort_for_raster(spike_df["Spike_Times"],trial_df)

    # Seperate spikes per trial type
    cherrySpikeValues = count_to_trial(cherry_reward_trials, trial_spike_times)
    grapeSpikeValues = count_to_trial(grape_reward_trials, trial_spike_times)
    bothRewardSpikeValues = count_to_trial(both_reward_trials, trial_spike_times)
    noRewardSpikeValues = count_to_trial(no_reward_trials, trial_spike_times)

    #SO that we can create a correspondding colour length for event plot
    lenOfCherryTrials = len(cherrySpikeValues)
    lenOfGrapeTrials = len(grapeSpikeValues)
    lenOfBothRewardTrials = len(bothRewardSpikeValues)
    lenOfNoRewardTrials = len(noRewardSpikeValues)

    #convert to np array
    cherrySpikeValues = np.asarray(cherrySpikeValues)
    grapeSpikeValues = np.asarray(grapeSpikeValues)
    bothRewardSpikeValues = np.asarray(bothRewardSpikeValues)
    noRewardSpikeValues = np.asarray(noRewardSpikeValues)

    def prepare_data_for_scatter(trial_index_modifier, trial_type_spike_values, len_of_trial_type):
        dic_of_dfs = {}
        for trial in range(len_of_trial_type):
            dic_of_dfs[trial] = pd.DataFrame(trial_type_spike_values[trial], columns=["spikes"])
            dic_of_dfs[trial].index = ([trial + trial_index_modifier]) * trial_type_spike_values.shape[1]
        x = []
        y = []
        for trial in range(len(dic_of_dfs)):
            df = dic_of_dfs[trial]
            x.extend(df["spikes"].values)
            y.extend(df.index.to_numpy())
        return(x,y)

    m1 = 0
    m2 = lenOfCherryTrials
    m3 = m2 + lenOfGrapeTrials
    m4 = m3 + lenOfBothRewardTrials

    cherryx, cherryy = prepare_data_for_scatter(m1, cherrySpikeValues, lenOfCherryTrials)
    grapex, grapey = prepare_data_for_scatter(m2, grapeSpikeValues, lenOfGrapeTrials)
    bothx, bothy = prepare_data_for_scatter(m3, bothRewardSpikeValues, lenOfBothRewardTrials)
    nox, noy = prepare_data_for_scatter(m4, noRewardSpikeValues, lenOfNoRewardTrials)

    # fig, ax = plt.subplots(1, sharex=True)
    # plt.scatter(cherryx,cherryy, marker = "|", color='r')
    # plt.scatter(grapex,grapey, marker = "|", color='m')
    # plt.scatter(bothx,bothy, marker = "|", color='b')
    # plt.scatter(nox,noy, marker = "|", color='k')
    # ax.set_xlim(right=3)
    # ax.set_xlim(left=-1)
    # plt.show()

    # #Concaternate arrays
    # spikes = np.concatenate((cherrySpikeValues,grapeSpikeValues,bothRewardSpikeValues,noRewardSpikeValues))
    #
    # #Create colorCodes
    # colorCodesCherry = [[1,0,0]] * lenOfCherryTrials
    # colorCodesGrape = [[1,0,1]] * lenOfGrapeTrials
    # colorCodesBothReward = [[0,0,1]] * lenOfBothRewardTrials
    # colorCodesNoReward = [[0,0,0]] * lenOfNoRewardTrials
    # colorCodes = colorCodesCherry + colorCodesGrape + colorCodesBothReward + colorCodesNoReward

    return(cherryx,cherryy,grapex,grapey,bothx,bothy,nox,noy)

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

    #New test for calculating lick counts - LICK predictions for spout
    cherry_lick_counts, bin_edges = lock_and_count(cherry_licks_only["Time Licking"],bins,trial_df)
    grape_lick_counts, bin_edges = lock_and_count(grape_licks_only["Time Licking"],bins,trial_df)
    center_lick_counts, bin_edges = lock_and_count(center_licks_only["Time Licking"],bins,trial_df)

    #Cal bin cbincentres
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])

    # Seperate spike counts per trial type
    cherry_spike_counts = count_to_trial(cherry_reward_trials, spike_counts)
    grape_spike_counts = count_to_trial(grape_reward_trials, spike_counts)
    bothreward_spike_counts = count_to_trial(both_reward_trials, spike_counts)
    noreward_spike_counts = count_to_trial(no_reward_trials, spike_counts)

    #Trial type licks - new test
    cherry_trial_cherry_licks = count_to_trial(cherry_reward_trials, cherry_lick_counts)
    cherry_trial_grape_licks = count_to_trial(cherry_reward_trials, grape_lick_counts)
    cherry_trial_center_licks = count_to_trial(cherry_reward_trials, center_lick_counts)

    grape_trial_cherry_licks = count_to_trial(grape_reward_trials, cherry_lick_counts)
    grape_trial_grape_licks = count_to_trial(grape_reward_trials, grape_lick_counts)
    grape_trial_center_licks = count_to_trial(grape_reward_trials, center_lick_counts)

    both_reward_trial_cherry_licks = count_to_trial(both_reward_trials, cherry_lick_counts)
    both_reward_trial_grape_licks = count_to_trial(both_reward_trials, grape_lick_counts)
    both_reward_trial_center_licks = count_to_trial(both_reward_trials, center_lick_counts)

    no_reward_trial_cherry_licks = count_to_trial(no_reward_trials, cherry_lick_counts)
    no_reward_trial_grape_licks = count_to_trial(no_reward_trials, grape_lick_counts)
    no_reward_trial_center_licks = count_to_trial(no_reward_trials, center_lick_counts)

    #Calculate licks for each trial type and reward
    cherry_trial_cherry_licks_count = pd.DataFrame(cherry_trial_cherry_licks).sum(axis=0)
    cherry_trial_grape_licks_count = pd.DataFrame(cherry_trial_grape_licks).sum(axis=0)
    cherry_trial_center_licks_count = pd.DataFrame(cherry_trial_center_licks).sum(axis=0)

    grape_trial_cherry_licks_count = pd.DataFrame(grape_trial_cherry_licks).sum(axis=0)
    grape_trial_grape_licks_count = pd.DataFrame(grape_trial_grape_licks).sum(axis=0)
    grape_trial_center_licks_count = pd.DataFrame(grape_trial_center_licks).sum(axis=0)

    both_reward_trial_cherry_licks = pd.DataFrame(both_reward_trial_cherry_licks).sum(axis=0)
    both_reward_trial_grape_licks = pd.DataFrame(both_reward_trial_grape_licks).sum(axis=0)
    both_reward_trial_center_licks = pd.DataFrame(both_reward_trial_center_licks).sum(axis=0)

    no_reward_trial_cherry_licks = pd.DataFrame(no_reward_trial_cherry_licks).sum(axis=0)
    no_reward_trial_grape_licks = pd.DataFrame(no_reward_trial_grape_licks).sum(axis=0)
    no_reward_trial_center_licks = pd.DataFrame(no_reward_trial_center_licks).sum(axis=0)

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
    spike_rates = (cherry_hertz, grape_hertz, bothreward_hertz, noreward_hertz)

    #Calculate average licking rate per trial - % of licks for each frame
    normalised_cherry_trial_cherry_licks  = cherry_trial_cherry_licks_count    / len(cherry_reward_lick_trials.values) * 100
    normalised_cherry_trial_grape_licks   = cherry_trial_grape_licks_count     / len(cherry_reward_lick_trials.values) * 100
    normalised_cherry_trial_center_licks  = cherry_trial_center_licks_count    / len(cherry_reward_lick_trials.values) * 100
    cherryTrialLicks = (normalised_cherry_trial_cherry_licks, normalised_cherry_trial_grape_licks, normalised_cherry_trial_center_licks)

    normalised_grape_trial_cherry_licks   = grape_trial_cherry_licks_count     / len(grape_reward_lick_trials.values) * 100
    normalised_grape_trial_grape_licks    = grape_trial_grape_licks_count      / len(grape_reward_lick_trials.values) * 100
    normalised_grape_trial_center_licks   = grape_trial_center_licks_count     / len(grape_reward_lick_trials.values) * 100
    grapeTrialLicks = (normalised_grape_trial_cherry_licks, normalised_grape_trial_grape_licks, normalised_grape_trial_center_licks)

    normalised_both_reward_trial_cherry_licks = both_reward_trial_cherry_licks / len(both_reward_lick_trials.values) * 100
    normalised_both_reward_trial_grape_licks  = both_reward_trial_grape_licks  / len(both_reward_lick_trials.values) * 100
    normalised_both_reward_trial_center_licks = both_reward_trial_center_licks / len(both_reward_lick_trials.values) * 100
    bothRewardLicks = (normalised_both_reward_trial_cherry_licks, normalised_both_reward_trial_grape_licks, normalised_both_reward_trial_center_licks)

    normalised_no_reward_trial_cherry_licks = no_reward_trial_cherry_licks     / len(no_reward_lick_trials.values) * 100
    normalised_no_reward_trial_grape_licks  = no_reward_trial_grape_licks      / len(no_reward_lick_trials.values) * 100
    normalised_no_reward_trial_center_licks = no_reward_trial_center_licks     / len(no_reward_lick_trials.values) * 100
    noRewardLicks = (normalised_no_reward_trial_cherry_licks, normalised_no_reward_trial_grape_licks, normalised_no_reward_trial_center_licks)

    return(bin_centres, spike_rates, cherryTrialLicks, grapeTrialLicks, bothRewardLicks, noRewardLicks)

#Function for generating a PSTH
def generate_graphs(trial_df,spike_df,cellID):

    #Load data for graphs
    cherryx,cherryy,grapex,grapey,bothx,bothy,nox,noy = new_generate_raster(trial_df,spike_df,cellID)
    # colorCodes, spikes = generate_raster(trial_df,spike_df,cellID)
    bin_centres, spike_rates, cherryTrialLicks, grapeTrialLicks, bothRewardLicks, noRewardLicks = generate_PSTH(trial_df,spike_df,cellID)

    #--------------------------------------------------------------------------
    #Outline subplots
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, sharex=True)

    #Plot PSTH
    ax1.plot(bin_centres,spike_rates[0], color='r', label="Cherry Reward")
    ax1.plot(bin_centres,spike_rates[1], color='m', label="Grape Reward" )
    ax1.plot(bin_centres,spike_rates[2], color='b', label="Both Reward"  )
    ax1.plot(bin_centres,spike_rates[3], color='k', label="No Reward"    )
    ax1.legend(loc='upper right')
    ax1.set(title="PSTH - Locked to reward", ylabel="Firing Rates (sp/s)")

    # #Plot spike Raster
    ax2.scatter(cherryx,cherryy, marker = "|", color='r', linewidths=0.1, alpha = 0.2,   s=0.2)
    ax2.scatter(grapex,grapey, marker = "|", color='m', linewidths=0.1,   alpha = 0.2,   s=0.2)
    ax2.scatter(bothx,bothy, marker = "|", color='b', linewidths=0.1,     alpha = 0.2,   s=0.2)
    ax2.scatter(nox,noy, marker = "|", color='k', linewidths=0.1,         alpha = 0.2,   s=0.2)
    ax2.set_xlim(right=3)
    ax2.set_xlim(left=-1)
    ax2.set(title="Spike Raster", xlabel="Time (s)", ylabel="Trials")

    # ax2.eventplot(spikes, color=colorCodes)
    # ax2.set_xlim(right=3)
    # ax2.set_xlim(left=-1)
    # ax2.set(title="Spike Raster", xlabel="Time (s)", ylabel="Trials")

    # #Licking subplot
    ax3.plot(bin_centres, cherryTrialLicks[0], color='r', label="Lick of cherry spout")
    ax3.plot(bin_centres, cherryTrialLicks[1], color='m', label="Lick of grape spout")
    ax3.plot(bin_centres, cherryTrialLicks[2], color='k', label="Lick center of spouts")
    ax3.set(ylabel="Perc. frames licking", title="Cherry Reward Trials")
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.legend(loc='upper right')
    ax3.set_ylim([0, 18])

    #Licking subplot
    ax4.plot(bin_centres, grapeTrialLicks[0], color='r', label="Lick of cherry spout")
    ax4.plot(bin_centres, grapeTrialLicks[1], color='m', label="Lick of grape spout")
    ax4.plot(bin_centres, grapeTrialLicks[2], color='k', label="Lick center of spouts")
    ax4.set(ylabel="Perc. frames licking", title="Grape Reward Trials")
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax4.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax4.legend(loc='upper right')
    ax4.set_ylim([0, 18])

    #Licking subplot
    ax5.plot(bin_centres, bothRewardLicks[0], color='r', label="Lick of cherry spout")
    ax5.plot(bin_centres, bothRewardLicks[1], color='m', label="Lick of grape spout")
    ax5.plot(bin_centres, bothRewardLicks[2], color='k', label="Lick center of spouts")
    ax5.set(ylabel="Perc. frames licking", title="Both Reward Trials")
    ax5.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax5.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax5.legend(loc='upper right')
    ax5.set_ylim([0, 18])

    #Licking subplot
    ax6.plot(bin_centres, noRewardLicks[0], color='r', label="Lick of cherry spout")
    ax6.plot(bin_centres, noRewardLicks[1], color='m', label="Lick of grape spout")
    ax6.plot(bin_centres, noRewardLicks[2], color='k', label="Lick center of spouts")
    ax6.set(ylabel="Perc. frames licking",xlabel="Time from Outcome [s] (Spike Time - Reward Time)", title="No Reward Trials")
    ax6.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax6.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax6.legend(loc='upper right')
    ax6.set_ylim([0, 18])

    #Show plots
    plt.show()
    return(fig)

    #---------------------------------------------------------------------------

#Generate the visulations
generate_graphs(trial_df,spike_df,1)

#Multiple viz gen
# pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")
# for x in range(10):
#     fig = generate_graphs(trial_df,spike_df,x)
#     pdf.savefig(fig)
# pdf.close()

#--------------------------------------1§--------------------
#Print the time of the process
print("")
print("--- %s seconds ---" % (time.time() - start_time))
print("")
