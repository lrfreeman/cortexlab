import utils as util
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

#Improvements to make
#1) functions being called twice affecting performance

"""-------------------PSTH Logic-----------------------------"""
def prep_data_for_PSTH(trial_df, spike_df, cell_ID):

    cherry_reward_trials, grape_reward_trials, both_reward_trials, no_reward_trials = util.split_data_by_trial_type(trial_df)
    spike_counts, bin_edges, bin_centres = util.lock_to_reward_and_count(spike_df, trial_df, cell_ID)
    cherry_spike_counts = util.count_to_trial(cherry_reward_trials, spike_counts, trial_df)
    grape_spike_counts = util.count_to_trial(grape_reward_trials, spike_counts, trial_df)
    both_reward_spike_counts = util.count_to_trial(both_reward_trials, spike_counts, trial_df)
    no_reward_spike_counts = util.count_to_trial(no_reward_trials, spike_counts, trial_df)
    return(cherry_spike_counts,
           grape_spike_counts,
           both_reward_spike_counts,
           no_reward_spike_counts,
           bin_edges,
           bin_centres)

def calculate_firing_rates_for_PSTH(trial_df, spike_df, cell_ID):
    cherry_spike_counts, grape_spike_counts, both_reward_spike_counts, no_reward_spike_counts, bin_edges, bin_centres = prep_data_for_PSTH(trial_df, spike_df, cell_ID)

    """-----------------Calculate firing rates-------------------------------"""
    #*5 because bin size is 200ms within  the lock function in utils
    cherry_count = pd.DataFrame(cherry_spike_counts).sum(axis=0)
    cherry_hertz = (cherry_count / len(cherry_spike_counts)) * 5

    grape_count = pd.DataFrame(grape_spike_counts).sum(axis=0)
    grape_hertz = (grape_count / len(grape_spike_counts)) * 5

    both_reward_count = pd.DataFrame(both_reward_spike_counts).sum(axis=0)
    both_reward_hertz = (both_reward_count / len(both_reward_spike_counts)) * 5

    no_reward_count = pd.DataFrame(no_reward_spike_counts).sum(axis=0)
    no_reward_hertz = (no_reward_count / len(no_reward_spike_counts)) * 5
    return(cherry_hertz, grape_hertz, both_reward_hertz, no_reward_hertz, bin_edges, bin_centres)

def generate_PSTH(trial_df, spike_df, cell_ID):
    cherry_hertz, grape_hertz, both_reward_hertz, no_reward_hertz, bin_edges, bin_centres = calculate_firing_rates_for_PSTH(trial_df, spike_df, cell_ID)

    #Outline subplots
    fig, (ax1) = plt.subplots(1, sharex=True)

    # #Plot PSTH
    ax1.plot(bin_centres,cherry_hertz[:-1], color='r', label="Cherry Reward")
    ax1.plot(bin_centres,grape_hertz[:-1], color='m', label="Grape Reward" )
    ax1.plot(bin_centres,both_reward_hertz[:-1], color='b', label="Both Reward")
    ax1.plot(bin_centres,no_reward_hertz[:-1], color='k', label="No Reward")
    ax1.legend(loc='upper right')
    # ax1.set(title="PSTH - Locked to reward - Cell:{} - MOs".format(cell_ID), ylabel="Firing Rates (sp/s)")
    plt.show()

"""-------------------Raster Logic---------------------------"""
def add_indexs_to_trial_df(trial_df):
    trial_df["index"] = trial_df.index
    trunc = lambda x: math.trunc(x)
    trial_df["Trial IDs"] = trial_df["trial_start_times"].apply(trunc)
    return(trial_df)

def compute_fastes_lick_by_trial(trial_df, first_lick_df):
    trial_df = add_indexs_to_trial_df(trial_df)
    first_lick_df = first_lick_df.merge(trial_df, on="Trial IDs")
    first_lick_df["lick gap"] = first_lick_df["First Lick Times"] - first_lick_df["reward_times"]
    first_lick_df = first_lick_df.drop(["left_choices", "free", "left_rewards", "right_rewards", "violations"], axis = 1)
    return(first_lick_df)

def sort_spikes_by_fastest_lick(trial_type_df, spike_counts, trial_df, first_lick_df):

    #Mapp spike data frames to trial reward type
    spike_counts_mapped_2_trial_type = [(x, spike_counts[x]) for x in range(len(trial_df)) if x in trial_type_df.index.values]
    assert len(spike_counts_mapped_2_trial_type) == len(trial_type_df.index.values), "Error when counting to trial"

    #Convert a list of tuples to a list of lifts to make mutable
    for i in range(len(spike_counts_mapped_2_trial_type)):
        spike_counts_mapped_2_trial_type[i] = list(spike_counts_mapped_2_trial_type[i])

    #Change the trial index to the lick difference - reward time minus 1st lick
    first_lick_df = compute_fastes_lick_by_trial(trial_df, first_lick_df)
    for x in range(len(spike_counts_mapped_2_trial_type)):
        spike_counts_mapped_2_trial_type[x][0] = first_lick_df.loc[first_lick_df["index"]==x, 'lick gap'].values

    #Remove trials with no first lick and sort by the trials with the fastest lick
    spike_counts_mapped_2_trial_type = [x for x in spike_counts_mapped_2_trial_type if x[0]]
    spike_counts_mapped_2_trial_type = sorted(spike_counts_mapped_2_trial_type, key=lambda x: x[0])

    #Remove first element so it becomes a list of spike dataframes
    for x in range(len(spike_counts_mapped_2_trial_type)):
        spike_counts_mapped_2_trial_type[x] = spike_counts_mapped_2_trial_type[x][1]

    return(spike_counts_mapped_2_trial_type)

def prep_data_for_raster(spike_df, trial_df, cell_ID, first_lick_df):
    spikes_mapped_to_trials = util.lock_and_sort_for_raster(spike_df, trial_df, cell_ID)
    cherry_reward_trials, grape_reward_trials, both_reward_trials, no_reward_trials = util.split_data_by_trial_type(trial_df)
    cherrySpikeValues =     sort_spikes_by_fastest_lick(cherry_reward_trials, spikes_mapped_to_trials, trial_df, first_lick_df)
    grapeSpikeValues =      sort_spikes_by_fastest_lick(grape_reward_trials, spikes_mapped_to_trials, trial_df, first_lick_df)
    bothRewardSpikeValues = sort_spikes_by_fastest_lick(both_reward_trials, spikes_mapped_to_trials, trial_df, first_lick_df)
    noRewardSpikeValues =   sort_spikes_by_fastest_lick(no_reward_trials, spikes_mapped_to_trials, trial_df, first_lick_df)
    return(cherrySpikeValues, grapeSpikeValues, bothRewardSpikeValues, noRewardSpikeValues)

def calculate_len_of_spikes_for_each_trial_type(spike_df, trial_df, cell_ID, first_lick_df):
    cherrySpikeValues, grapeSpikeValues, bothRewardSpikeValues, noRewardSpikeValues = prep_data_for_raster(spike_df, trial_df, cell_ID, first_lick_df)
    lenOfCherryTrials = len(cherrySpikeValues)
    lenOfGrapeTrials = len(grapeSpikeValues)
    lenOfBothRewardTrials = len(bothRewardSpikeValues)
    lenOfNoRewardTrials = len(noRewardSpikeValues)
    return(lenOfCherryTrials,
           lenOfGrapeTrials,
           lenOfBothRewardTrials,
           lenOfNoRewardTrials)

def generate_raster_event_plot_data(trial_df, spike_df, cell_ID, first_lick_df):

    # Seperate spikes per trial type
    cherrySpikeValues, grapeSpikeValues, bothRewardSpikeValues, noRewardSpikeValues = prep_data_for_raster(spike_df, trial_df, cell_ID, first_lick_df)

    #SO that we can create a correspondding colour length for event plot
    lenOfCherryTrials, lenOfGrapeTrials, lenOfBothRewardTrials, lenOfNoRewardTrials = calculate_len_of_spikes_for_each_trial_type(spike_df, trial_df, cell_ID, first_lick_df)

    #convert to np array
    cherrySpikeValues = np.asarray(cherrySpikeValues)
    grapeSpikeValues = np.asarray(grapeSpikeValues)
    bothRewardSpikeValues = np.asarray(bothRewardSpikeValues)
    noRewardSpikeValues = np.asarray(noRewardSpikeValues)

    #Concaternate arrays
    spikes = np.concatenate((cherrySpikeValues,grapeSpikeValues,bothRewardSpikeValues,noRewardSpikeValues))

    #Create colorCodes
    colorCodesCherry = [[1,0,0]] * lenOfCherryTrials
    colorCodesGrape = [[1,0,1]] * lenOfGrapeTrials
    colorCodesBothReward = [[0,0,1]] * lenOfBothRewardTrials
    colorCodesNoReward = [[0,0,0]] * lenOfNoRewardTrials
    colorCodes = colorCodesCherry + colorCodesGrape + colorCodesBothReward + colorCodesNoReward

    return(colorCodes, spikes)

def gen_event_plot(trial_df, spike_df, cell_ID, first_lick_df):
    colorCodes, spikes = generate_raster_event_plot_data(trial_df, spike_df, cell_ID, first_lick_df)
    #Outline subplots
    fig, (ax1) = plt.subplots(1, sharex=True)
    ax1.eventplot(spikes, color=colorCodes)
    ax1.set_xlim(right=3)
    ax1.set_xlim(left=-1)
    ax1.set(title="Spike Raster", xlabel="Time (s)", ylabel="Trials")
    plt.show()

"""-------------------None working prototype functions---------------------------"""
# def calculate_new_index(spike_df, trial_df, cell_ID):
#     cherrySpikeValues, grapeSpikeValues, bothRewardSpikeValues, noRewardSpikeValues = prep_data_for_raster(spike_df, trial_df, cell_ID)
#     #SO that we can create a correspondding colour length for event plot
#     lenOfCherryTrials = len(cherrySpikeValues)
#     lenOfGrapeTrials = len(grapeSpikeValues)
#     lenOfBothRewardTrials = len(bothRewardSpikeValues)
#     lenOfNoRewardTrials = len(noRewardSpikeValues)
#     #Mod index so you can group trial types in raster
#     index_mod_for_cherry = 0
#     index_mod_for_grape = lenOfCherryTrials
#     index_mod_for_both = index_mod_for_grape + lenOfGrapeTrials
#     index_mod_for_neither = index_mod_for_both + lenOfBothRewardTrials
#     return(index_mod_for_cherry,
#            index_mod_for_grape,
#            index_mod_for_both,
#            index_mod_for_neither)
#
# def generate_scatter_coordinates(trial_index_modifier, trial_type_spike_values, len_of_trial_type, cell_ID):
#     dic_of_dfs = {}
#     for trial in range(len_of_trial_type):
#         dic_of_dfs[trial] = pd.DataFrame(trial_type_spike_values[trial], columns=["spikes"])
#         dic_of_dfs[trial].index = ([trial + trial_index_modifier]) * trial_type_spike_values.shape[1]
#     x = []
#     y = []
#     for trial in range(len(dic_of_dfs)):
#         df = dic_of_dfs[trial]
#         x.extend(df["spikes"].values)
#         y.extend(df.index.to_numpy())
#     return(x,y)
#
# def generate_raster(spike_df, trial_df, cell_ID):
#     index_mod_for_cherry, index_mod_for_grape, index_mod_for_both, index_mod_for_neither = calculate_new_index(spike_df, trial_df, cell_ID)
#     cherrySpikeValues, grapeSpikeValues, bothRewardSpikeValues, noRewardSpikeValues = prep_data_for_raster(spike_df, trial_df, cell_ID)
#     lenOfCherryTrials, lenOfGrapeTrials, lenOfBothRewardTrials, lenOfNoRewardTrials = calculate_len_of_spikes_for_each_trial_type(spike_df, trial_df, cell_ID)
#
#     #convert to np array
#     cherrySpikeValues = np.asarray(cherrySpikeValues)
#     grapeSpikeValues = np.asarray(grapeSpikeValues)
#     bothRewardSpikeValues = np.asarray(bothRewardSpikeValues)
#     noRewardSpikeValues = np.asarray(noRewardSpikeValues)
#
#     #product coords
#     cherry_x, cherry_y = generate_scatter_coordinates(index_mod_for_cherry, cherrySpikeValues, lenOfCherryTrials, cell_ID)
#
#     #Outline subplots
#     fig, (ax1) = plt.subplots(1, sharex=True)
#
#     # #Plot PSTH
#     ax1.scatter(cherry_x,cherry_y, marker = "|", color='r', linewidths=0.1, alpha = 0.2,   s=0.2)
#     ax1.set_xlim(right=3)
#     ax1.set_xlim(left=-1)
#     plt.show()
