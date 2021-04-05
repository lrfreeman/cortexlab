import PredictLicking.is_licking as lick
import matplotlib.backends.backend_pdf
import electrophysiology.ingest_timesync as ingest
from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import fast_histogram
from numba import njit

"""Class for project"""
class CortexLab:

    def __init__(self, session_data, frame_alignment_data, dlc_video_csv):
        self.session_data = session_data
        self.frame_alignment_data = frame_alignment_data
        self.dlc_video_csv = dlc_video_csv

    """Generate trial_df, spike_df"""
    def load_data(self, session_data):
        trial_df, spike_times, cluster_IDs, cluster_types = ingest.convert_mat(session_data)
        trial_df = trial_df.drop(columns=["nTrials"])
        spike_df =  pd.DataFrame(spike_times, columns = ["Spike_Times"])
        spike_df["cluster_ids"] = cluster_IDs
        self.numofcells = len(cluster_types[0])
        self.spike_df = spike_df
        self.df = trial_df
        return(trial_df,spike_df)

    """Generate first lick data frame"""
    def compute_the_first_lick(self):
        frame_times = ingest.import_frame_times(self.frame_alignment_data)
        df = lick.generate_licking_times(frame_times, self.dlc_video_csv)
        lick_df = lick.map_lick_to_trial_type(df,self.session_data)
        first_lick_df = lick.compute_1st_lick(lick_df)
        return(first_lick_df, lick_df, df)

    """Split data by trial type"""
    def split_data_by_trial_type(self, data_frame):
        cherry_trial =  data_frame.loc[(data_frame['left_rewards'] == 1) & (data_frame['right_rewards'] == 0)]
        grape_trial =  data_frame.loc[(data_frame['left_rewards'] == 0) & (data_frame['right_rewards'] == 1)]
        both_reward_trials =  data_frame.loc[(data_frame['left_rewards'] == 1) & (data_frame['right_rewards'] == 1)]
        no_reward_trials =  data_frame.loc[(data_frame['left_rewards'] == 0) & (data_frame['right_rewards'] == 0)]
        return(cherry_trial,grape_trial,both_reward_trials,no_reward_trials)

    """ Lock spikes to your event """
    def lock_and_count(self, spike_df):

        ranges=[-1,3]
        # ranges = np.asarray(ranges).astype(np.float64)
        bins = np.arange(-1,3,0.2).tolist()

        #Logic
        lock_time = {}
        x_counts = {}
        for trial in range(len(self.df)):
            lock_time[trial] = self.df["reward_times"][trial]
            h = fast_histogram.histogram1d(spike_df["Spike_Times"]-lock_time[trial], bins=20, range=(ranges[0],ranges[1]))
            x_counts[trial] = h
        ignore, bin_edges = np.histogram(spike_df["Spike_Times"]-lock_time[trial], bins=bins)
        bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
        return(x_counts, bin_edges, bin_centres)

    """Split counts by trial type"""
    #Count things and map them to trials
    def count_to_trial(self, trial_type, spike_counts):
        spike_counts_mapped_2_trial_type = [spike_counts[x] for x in range(len(self.df)) if x in trial_type.index.values]
        return(spike_counts_mapped_2_trial_type)
