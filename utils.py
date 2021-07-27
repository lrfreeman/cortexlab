import electrophysiology.ingest_timesync as matlab_convert
import deepLabCut.is_licking as lick
import fast_histogram
import numpy as np
import pandas as pd
import math

"""A class to ingest electrophysiology data, frame data and DLC data into three data frames
1) trial df
2) spike df
3) brain region df"""
class Upload_Data:

    def __init__(self, session_data, frame_alignment_data, dlc_video_csv):
        self.session_data = session_data
        self.frame_alignment_data = frame_alignment_data
        self.dlc_video_csv = dlc_video_csv
        self.load_data(self.session_data)
        self.compute_the_first_lick()

    """Generate trial_df, spike_df"""
    def load_data(self, session_data):
        trial_df, spike_df, brain_regions = matlab_convert.ingest_mat(session_data)
        self.trial_df = trial_df
        self.spike_df = spike_df
        self.brain_regions = brain_regions

    """Generate first lick data frame"""
    def compute_the_first_lick(self):
        frame_times = matlab_convert.import_frame_times(self.frame_alignment_data)
        df = lick.generate_licking_times(frame_times, self.dlc_video_csv)
        lick_df = lick.map_lick_to_trial_type(df,self.session_data)
        first_lick_df = lick.compute_1st_lick(lick_df)

        self.first_lick_df = first_lick_df
        self.lick_df = lick_df

"""Split data by trial type"""
def split_data_by_trial_type(data_frame):
    cherry_trial =  data_frame.loc[(data_frame['left_rewards'] == 1) & (data_frame['right_rewards'] == 0)]
    grape_trial =  data_frame.loc[(data_frame['left_rewards'] == 0) & (data_frame['right_rewards'] == 1)]
    both_reward_trials =  data_frame.loc[(data_frame['left_rewards'] == 1) & (data_frame['right_rewards'] == 1)]
    no_reward_trials =  data_frame.loc[(data_frame['left_rewards'] == 0) & (data_frame['right_rewards'] == 0)]
    return(cherry_trial,grape_trial,both_reward_trials,no_reward_trials)

""" Lock spikes to your event and binn for PSTH"""
def lock_to_reward_and_count(spike_df, trial_df, cell_ID):
    spike_df = spike_df.loc[(spike_df["cluster_ids"] == cell_ID)]
    ranges= [-1,3]
    bins = np.arange(-1,3,0.2).tolist()
    lock_time = {}
    spike_counts = {}
    for trial in range(len(trial_df)):
        lock_time[trial] = trial_df["reward_times"][trial]

        #Comment out for trucated PSTH
        spikes_in_trial = (spike_df.loc[spike_df["index"]==trial]["spike_time"].values)-lock_time[trial]

        #Uncomment for none truncated PSTH
        # spikes_in_trial = spike_df["spike_time"]-lock_time[trial]

        counts = fast_histogram.histogram1d(spikes_in_trial, bins=20, range=(ranges[0],ranges[1]))
        spike_counts[trial] = counts
    ignore, bin_edges = np.histogram(spike_df["spike_time"]-lock_time[trial], bins=bins)
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
    return(spike_counts, bin_edges, bin_centres)

"""Assign spike counts to a trial type by reward for PSTH"""
# Count things and map them to trials
def count_to_trial(trial_type_df, spike_counts, trial_df):
    spike_counts_mapped_2_trial_type = [spike_counts[x] for x in range(len(trial_df)) if x in trial_type_df.index.values]
    assert len(spike_counts_mapped_2_trial_type) == len(trial_type_df.index.values), "Error when counting to trial"
    return(spike_counts_mapped_2_trial_type)

"""Just lock spikes for a raster"""
def lock_and_sort_for_raster(spike_df,trial_df, cell_ID):
    spike_df = spike_df.loc[(spike_df["cluster_ids"] == cell_ID)]
    lock_time = {}
    trial_spike_times = {}

    for trial in range(len(trial_df)):
        lock_time[trial] = trial_df["reward_times"][trial]

        #Comment out for trucated raster
        trial_spike_times[trial] = (spike_df.loc[spike_df["index"]==trial]["spike_time"].values)-lock_time[trial]

        #Uncomment for none truncated raster
        # trial_spike_times[trial] = spike_df["spike_time"]-lock_time[trial]

        #Remove spikes outside of time window for performance
        df = trial_spike_times[trial]
        trial_spike_times[trial] = df[(df > -1) & (df < 5)]

    # assert len(trial_spike_times) == len(trial_df), "Lenght of dictionary does not match number of trials"
    return(trial_spike_times)

""" Lock spikes to your event and binn for PSTH - lick histo"""
#Function broken as outputting more licks than possible
def lock_to_reward_and_count_licks(licking_type, trial_df):
    ranges= [-1,3]
    bins = np.arange(-1,3,0.2).tolist()
    lock_time = {}
    lick_counts = {}

    for trial in range(len(trial_df)):
        lock_time[trial] = trial_df["reward_times"][trial]

        #Comment out for trucated
        # licks_for_trial = (licking_type.loc[licking_type["index"]==trial]['Time Licking'].values) - lock_time[trial]

        #Uncomment for none truncated
        licks_for_trial = licking_type["Time Licking"]-lock_time[trial]

        counts = fast_histogram.histogram1d(licks_for_trial, bins=20, range=(ranges[0],ranges[1]))
        lick_counts[trial] = counts
    ignore, bin_edges = np.histogram(licking_type["Time Licking"]-lock_time[trial], bins=bins)
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
    lick_counts = pd.DataFrame(lick_counts).T.sum(axis=0)
    return(lick_counts, bin_edges, bin_centres)

"""Assign licks to each trial"""
def licks_to_trial_locked_to_reward(lick_df, trial_df):
    x = trial_df[["trial_start_times", "index"]]
    new_lick_df = lick_df.merge(x, how = 'left', on = "trial_start_times")
    assert len(new_lick_df) == len(lick_df), "Merge error"

    licks_to_trial = {}
    lock_time = {}
    for trial in range(len(trial_df)):
        lock_time[trial] = trial_df["reward_times"][trial]
        licks_to_trial[trial] = (new_lick_df.loc[new_lick_df["index"]==trial]['Time Licking'].values) - lock_time[trial]

    #Unit test
    flat_list = [item for sublist in licks_to_trial.values() for item in sublist]
    assert len(flat_list) == len(lick_df), "Count of licks wrong"

    return(licks_to_trial)

"""truncate trials"""
def index_spikesPoints_to_trial(spike_df, trial_df):

    trial_df["start_time_bins"] = pd.cut(trial_df["trial_start_times"], bins=trial_df["trial_start_times"])
    spike_df["start_time_bins"] = pd.cut(spike_df["spike_time"], bins=trial_df["trial_start_times"])
    spike_df = spike_df.copy()
    spike_df = spike_df.dropna()
    spike_df["trunc_time_bins"] = (spike_df["start_time_bins"].apply(lambda x: x.left))
    trunc = lambda x: math.trunc(x)
    spike_df["Trial IDs"] = spike_df["trunc_time_bins"].apply(trunc)
    x = trial_df[["Trial IDs", "index"]]
    spike_df = spike_df.merge(x, how = "left", on = "Trial IDs")

    return(spike_df)
