import electrophysiology.ingest_timesync as matlab_convert
import fast_histogram
import numpy as np

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

    """Generate trial_df, spike_df"""
    def load_data(self, session_data):
        trial_df, spike_df, brain_regions = matlab_convert.ingest_mat(session_data)
        self.trial_df = trial_df
        self.spike_df = spike_df
        self.brain_regions = brain_regions

"""Split data by trial type"""
def split_data_by_trial_type(data_frame):
    cherry_trial =  data_frame.loc[(data_frame['left_rewards'] == 1) & (data_frame['right_rewards'] == 0)]
    grape_trial =  data_frame.loc[(data_frame['left_rewards'] == 0) & (data_frame['right_rewards'] == 1)]
    both_reward_trials =  data_frame.loc[(data_frame['left_rewards'] == 1) & (data_frame['right_rewards'] == 1)]
    no_reward_trials =  data_frame.loc[(data_frame['left_rewards'] == 0) & (data_frame['right_rewards'] == 0)]
    return(cherry_trial,grape_trial,both_reward_trials,no_reward_trials)

""" Lock spikes to your event """
def lock_to_reward_and_count(spike_df, trial_df):
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

"""Assign spike counts to a trial type by reward"""
#Count things and map them to trials
def count_to_trial(trial_type_df, spike_counts):
    spike_counts_mapped_2_trial_type = [spike_counts[x] for x in range(len(trial_type_df)) if x in trial_type_df.index.values]
    return(spike_counts_mapped_2_trial_type)
