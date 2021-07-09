import electrophysiology.ingest_timesync as matlab_convert
import deepLabCut.is_licking as lick
import fast_histogram
import numpy as np
import pandas as pd

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
        counts = fast_histogram.histogram1d(spike_df["spike_time"]-lock_time[trial], bins=20, range=(ranges[0],ranges[1]))
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
        trial_spike_times[trial] = spike_df["spike_time"]-lock_time[trial]

        #Remove spikes outside of time window for performance
        df = trial_spike_times[trial]
        trial_spike_times[trial] = df[(df > -1) & (df < 5)]
    return(trial_spike_times)

""" Lock spikes to your event and binn for PSTH"""
def lock_to_reward_and_count_licks(licking_type, trial_df):
    ranges= [-1,3]
    bins = np.arange(-1,3,0.2).tolist()
    lock_time = {}
    lick_counts = {}
    for trial in range(len(trial_df)):
        lock_time[trial] = trial_df["reward_times"][trial]
        counts = fast_histogram.histogram1d(licking_type["Time Licking"]-lock_time[trial], bins=20, range=(ranges[0],ranges[1]))
        lick_counts[trial] = counts
    ignore, bin_edges = np.histogram(licking_type["Time Licking"]-lock_time[trial], bins=bins)
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
    lick_counts = pd.DataFrame(lick_counts).sum(axis=1)
    return(lick_counts, bin_edges, bin_centres)
