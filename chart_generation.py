import utils as util
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import time

#Performance checks
start_time = time.time()

#Improvements to make
#1) functions being called twice affecting performance
#2) If I remove all spikes out of the range of -1 to 3s, from original data frame,
#surely it will be much faster

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
    ax1.set(title="PSTH - Locked to reward - Cell:{}".format(cell_ID), ylabel="Firing Rates (sp/s)")
    plt.show()

"""-------------------Raster Logic---------------------------"""
class Raster:

    def __init__(self, trial_df, spike_df, first_lick_df, brain_regions):
        self.trial_df = trial_df
        self.spike_df = spike_df
        self.first_lick_df = first_lick_df

        self.add_indexs_to_trial_df()
        self.compute_fastest_lick_by_trial()
        cherry_reward_trials, grape_reward_trials, both_reward_trials, no_reward_trials = util.split_data_by_trial_type(self.trial_df)

        self.cherry_reward_trials = cherry_reward_trials
        self.grape_reward_trials = grape_reward_trials
        self.both_reward_trials = both_reward_trials
        self.no_reward_trials = no_reward_trials

        self.brain_regions = brain_regions

    def add_indexs_to_trial_df(self):
        self.trial_df["index"] = self.trial_df.index
        trunc = lambda x: math.trunc(x)
        self.trial_df["Trial IDs"] = self.trial_df["trial_start_times"].apply(trunc)

    def compute_fastest_lick_by_trial(self):
        self.first_lick_df = self.first_lick_df.merge(self.trial_df, on="Trial IDs")
        self.first_lick_df["lick gap"] = self.first_lick_df["First Lick Times"] - self.first_lick_df["reward_times"]
        self.first_lick_df = self.first_lick_df.drop(["left_choices", "free", "left_rewards", "right_rewards", "violations"], axis = 1)

    def sort_spikes_by_fastest_lick(self, trial_type_df, spike_counts):

        #Create dic. Keys are trial indexes from trial_df and values are spike dataframes
        DIC_spike_dfs_by_trial_index = {}
        for x in range(len(self.trial_df)):
            if x in trial_type_df.index.values:
                DIC_spike_dfs_by_trial_index[x] = spike_counts[x]
        assert len(DIC_spike_dfs_by_trial_index) == len(trial_type_df.index.values), "Dictionary does not match length of trials for this reward type"

        #Change keys of dic to lick time - reward = lick gap
        for x in list(DIC_spike_dfs_by_trial_index.keys()):
            new_key = self.first_lick_df.loc[self.first_lick_df["index"]==x]['lick gap'].values
            if new_key:
                DIC_spike_dfs_by_trial_index[new_key[0]] = DIC_spike_dfs_by_trial_index.pop(x)
            #Remove trials without a lick
            else: del DIC_spike_dfs_by_trial_index[x]

        #Sort dictionary
        DIC_spike_dfs_sorted_by_fastest_lick = {k: v for k, v in sorted(DIC_spike_dfs_by_trial_index.items(), key=lambda item: item[0])}

        #Create a dataframe which will act as the scatter points for overlaying first lick on the raster
        sorted_first_lick_df = pd.DataFrame(DIC_spike_dfs_sorted_by_fastest_lick.keys())

        #Name the sorted spike dataframes for the raster by trial
        sorted_spike_data_frames = DIC_spike_dfs_sorted_by_fastest_lick.values()

        return(sorted_spike_data_frames, sorted_first_lick_df)

    def prep_data_for_raster(self, cell_ID):

        spikes_mapped_to_trials = util.lock_and_sort_for_raster(self.spike_df, self.trial_df, cell_ID)

        cherrySpikeValues, sorted_fastest_lick_df_cherry_trials          = self.sort_spikes_by_fastest_lick(self.cherry_reward_trials, spikes_mapped_to_trials)
        grapeSpikeValues, sorted_fastest_lick_df_grape_trials            = self.sort_spikes_by_fastest_lick(self.grape_reward_trials, spikes_mapped_to_trials)
        bothRewardSpikeValues, sorted_fastest_lick_df_bothrewards_trials = self.sort_spikes_by_fastest_lick(self.both_reward_trials, spikes_mapped_to_trials)
        noRewardSpikeValues, sorted_fastest_lick_df_noreward_trials      = self.sort_spikes_by_fastest_lick(self.no_reward_trials, spikes_mapped_to_trials)

        #Assign fastest lick df to object for another function to processed
        self.sorted_fastest_lick_df_cherry_trials      = sorted_fastest_lick_df_cherry_trials
        self.sorted_fastest_lick_df_grape_trials       = sorted_fastest_lick_df_grape_trials
        self.sorted_fastest_lick_df_bothrewards_trials = sorted_fastest_lick_df_bothrewards_trials
        self.sorted_fastest_lick_df_noreward_trials    = sorted_fastest_lick_df_noreward_trials

        self.cell_id = cell_ID

        self.cherrySpikeValues =      list(cherrySpikeValues)
        self.grapeSpikeValues =       list(grapeSpikeValues)
        self.bothRewardSpikeValues =  list(bothRewardSpikeValues)
        self.noRewardSpikeValues =    list(noRewardSpikeValues)

        self.len_cherrySpikeValues =     len(cherrySpikeValues)
        self.len_grapeSpikeValues =      len(grapeSpikeValues)
        self.len_bothRewardSpikeValues = len(bothRewardSpikeValues)
        self.len_noRewardSpikeValues =   len(noRewardSpikeValues)

        #Check that the number of trials matches the number of first licks
        assert len(cherrySpikeValues) + len(grapeSpikeValues) + len(bothRewardSpikeValues) + len(noRewardSpikeValues) == len(self.first_lick_df.index.values), "Len of trials does not match number of 1st licks"

    def produce_overlay_licking_data_for_raster(self):
        self.licking_overlay = pd.concat([self.sorted_fastest_lick_df_cherry_trials,
                               self.sorted_fastest_lick_df_grape_trials,
                               self.sorted_fastest_lick_df_bothrewards_trials,
                               self.sorted_fastest_lick_df_noreward_trials], ignore_index=True)

    def gen_event_plot(self, cell_ID):

        print("###_/_Program commenced_/_###")
        print("Cluster id:{}".format(cell_ID))

        #Prep data
        self.prep_data_for_raster(cell_ID)
        spikes = self.cherrySpikeValues + self.grapeSpikeValues + self.bothRewardSpikeValues + self.noRewardSpikeValues

        #Create colorCodes
        colorCodesCherry = [[1,0,0]] * self.len_cherrySpikeValues
        colorCodesGrape = [[1,0,1]] * self.len_grapeSpikeValues
        colorCodesBothReward = [[0,0,1]] * self.len_bothRewardSpikeValues
        colorCodesNoReward = [[0,0,0]] * self.len_noRewardSpikeValues
        colorCodes = colorCodesCherry + colorCodesGrape + colorCodesBothReward + colorCodesNoReward

        print("")
        print("--- %s seconds until eventplot ---" % (time.time() - start_time))
        print("")

        #Outline data for time of first lick
        self.produce_overlay_licking_data_for_raster()
        x = self.licking_overlay.iloc[:,0]
        y = self.licking_overlay.index.values

        #Index brain region by cluster ID
        brain_region = self.brain_regions.loc[self.brain_regions.index.values == cell_ID]["regions"].values

        #Outline subplots
        fig, (ax1) = plt.subplots(1, sharex=True)
        ax1.eventplot(spikes, color=colorCodes)
        ax1.scatter(x,y, marker = '_', alpha = 0.3, color = 'orange')
        ax1.set_xlim(right=3)
        ax1.set_xlim(left=-1)
        ax1.set(title="Spike raster sorted by lick times. Cluster ID:{}, in region:{}".format(cell_ID, brain_region), xlabel="Time (s)", ylabel="Trials")
        ax1.margins(y=0)
        # plt.show()
        return(fig)
