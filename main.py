import PredictLicking.is_licking as lick
import matplotlib.backends.backend_pdf
import electrophysiology.ingest_timesync as ingest
from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import DLC_Classes as CL

from functools import partial
import holoviews as hv
import holoviews.operation.datashader as hd
import datashader as ds
from datashader.mpl_ext import dsshow
import datashader.transfer_functions as tf

#Extend data print rows
# pd.set_option("display.max_rows", None, "display.max_columns", None)

#PSTHS - 99seconds for 350 cells

#Performance checks
start_time = time.time()

"""-----------------Load some data-------------------------------"""

data = CL.CortexLab('/Users/laurence/Desktop/Neuroscience/kevin_projects/data/processed_physdata/aligned_physdata_KM011_2020-03-23_probe1.mat',
                    '/Users/laurence/Desktop/Neuroscience/kevin_projects/data/KM011_video_timestamps/2020-03-23/face_timeStamps.mat',
                    '/Users/laurence/Desktop/Neuroscience/kevin_projects/data/23_faceDLC_resnet50_Master_ProjectAug13shuffle1_133500.csv')

trial_df, spike_df = data.load_data(data.session_data)

"""-----------------Split trials / cells-------------------------------"""
# Split data by trial type so spike data can be split by reward
cherry_reward_trials, grape_reward_trials, both_reward_trials, no_reward_trials = data.split_data_by_trial_type(trial_df)

def split_by_cell(cell_ID):
    """-----------------Lock data-------------------------------"""
    #Return spike counts and bin edges for a set of bins for a given trial data frame
    spike_counts, bin_edges, bin_centres = data.lock_and_count(spike_df.loc[(spike_df["cluster_ids"] == cell_ID)])

    """-----------------Split by trial type-------------------------------"""
    # #Seperate spikes per trial type
    cherry_spike_counts = data.count_to_trial(cherry_reward_trials, spike_counts)
    grape_spike_counts = data.count_to_trial(grape_reward_trials, spike_counts)
    both_reward_spike_counts = data.count_to_trial(both_reward_trials, spike_counts)
    no_reward_spike_counts = data.count_to_trial(no_reward_trials, spike_counts)

    """-----------------Calculate firing rates-------------------------------"""
    cherry_count = pd.DataFrame(cherry_spike_counts).sum(axis=0)
    cherry_hertz = (cherry_count / len(cherry_spike_counts)) * 5

    grape_count = pd.DataFrame(grape_spike_counts).sum(axis=0)
    grape_hertz = (grape_count / len(grape_spike_counts)) * 5

    both_reward_count = pd.DataFrame(both_reward_spike_counts).sum(axis=0)
    both_reward_hertz = (both_reward_count / len(both_reward_spike_counts)) * 5

    no_reward_count = pd.DataFrame(no_reward_spike_counts).sum(axis=0)
    no_reward_hertz = (no_reward_count / len(no_reward_spike_counts)) * 5

    #--------------------------------------------------------------------------
    #Outline subplots
    fig, (ax1) = plt.subplots(1, sharex=True)

    # #Plot PSTH
    ax1.plot(bin_centres,cherry_hertz[:-1], color='r', label="Cherry Reward")
    ax1.plot(bin_centres,grape_hertz[:-1], color='m', label="Grape Reward" )
    ax1.plot(bin_centres,both_reward_hertz[:-1], color='b', label="Both Reward"  )
    ax1.plot(bin_centres,no_reward_hertz[:-1], color='k', label="No Reward"    )
    ax1.legend(loc='upper right')
    ax1.set(title="PSTH - Locked to reward", ylabel="Firing Rates (sp/s)")
    # plt.show()
    return(fig)

"""-----------------Plot PSTH-------------------------------"""

# #Multiple viz gen
# pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")
# for x in range(data.numofcells):
#     fig = split_by_cell(x)
#     pdf.savefig(fig)
#     plt.close(fig)
# pdf.close()

"""-----------------Spike Raster-------------------------------"""
#Count things and map them to trials
def count_to_trial(trial_type, data_counts):
    count = [data_counts[x] for x in range(len(trial_df)) if list(data_counts.keys())[x] in trial_type.index.values]
    return(count)

def lock_and_sort_for_raster(time,trial_df):
    lock_time = {}
    trial_spike_times = {}
    for trial in range(len(trial_df)):
        lock_time[trial] = trial_df["reward_times"][trial]
        trial_spike_times[trial] = time-lock_time[trial]
    return(trial_spike_times)

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

    return(cherryx,cherryy,grapex,grapey,bothx,bothy,nox,noy)
cherryx,cherryy,grapex,grapey,bothx,bothy,nox,noy =  new_generate_raster(trial_df, spike_df, 4)

#Multiple viz gen
for n in range(data.numofcells):
    cherryx,cherryy,grapex,grapey,bothx,bothy,nox,noy = new_generate_raster(trial_df, spike_df, n)
    df = pd.DataFrame(dict(x=(cherryx + grapex + bothx + nox), y=(cherryy + grapey + bothy + noy)))
    if df.empty:
        print("Cell number: {} cotains an empty data frame".format(n))
        continue
    else:
        fig = dsshow(df, ds.Point('x', 'y'), aspect='auto')
        plt.colorbar(fig)
        plt.xlim(-1, 3)
        plt.savefig('/Users/laurence/Desktop/rasters/CELL{}.png'.format(n))
        plt.close()

#--------------------------------------1§--------------------
#Print the time of the process
print("")
print("--- %s seconds ---" % (time.time() - start_time))
print("")
