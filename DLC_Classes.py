import PredictLicking.is_licking as lick
import electrophysiology.ingest_timesync as ingest
import numpy as np
import pandas as pd
import fast_histogram

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
        self.firstlick_df = first_lick_df
        self.lick_df = lick_df
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

    def binned_firing_rate_calculations(self, cell_ID):

        cherry_reward_trials, grape_reward_trials, both_reward_trials, no_reward_trials = self.split_data_by_trial_type(self.df)

        """-----------------Lock data-------------------------------"""
        #Return spike counts and bin edges for a set of bins for a given trial data frame
        spike_counts, bin_edges, bin_centres = self.lock_and_count(self.spike_df.loc[(self.spike_df["cluster_ids"] == cell_ID)])

        """-----------------Split by trial type-------------------------------"""
        # #Seperate spikes per trial type
        cherry_spike_counts = self.count_to_trial(cherry_reward_trials, spike_counts)
        grape_spike_counts = self.count_to_trial(grape_reward_trials, spike_counts)
        both_reward_spike_counts = self.count_to_trial(both_reward_trials, spike_counts)
        no_reward_spike_counts = self.count_to_trial(no_reward_trials, spike_counts)

        """-----------------Calculate firing rates-------------------------------"""
        cherry_count = pd.DataFrame(cherry_spike_counts).sum(axis=0)
        cherry_hertz = (cherry_count / len(cherry_spike_counts)) * 5

        grape_count = pd.DataFrame(grape_spike_counts).sum(axis=0)
        grape_hertz = (grape_count / len(grape_spike_counts)) * 5

        both_reward_count = pd.DataFrame(both_reward_spike_counts).sum(axis=0)
        both_reward_hertz = (both_reward_count / len(both_reward_spike_counts)) * 5

        no_reward_count = pd.DataFrame(no_reward_spike_counts).sum(axis=0)
        no_reward_hertz = (no_reward_count / len(no_reward_spike_counts)) * 5

        firing_rates = cherry_count + grape_count + both_reward_count + no_reward_count

        return(firing_rates)

    def binned_licking_calculations(self):

        #Split licking data by trial type to calculate total frames inorder to normalise licking visulisations
        cherry_reward_lick_trials,grape_reward_lick_trials,both_reward_lick_trials,no_reward_lick_trials = self.split_data_by_trial_type(self.lick_df)

        """ Lock spikes to your event """
        def lock_and_count_for_licks(lick_df):

            ranges=[-1,3]
            # ranges = np.asarray(ranges).astype(np.float64)
            bins = np.arange(-1,3,0.2).tolist()

            #Logic
            lock_time = {}
            x_counts = {}
            for trial in range(len(self.df)):
                lock_time[trial] = self.df["reward_times"][trial]
                h = fast_histogram.histogram1d(lick_df["Time Licking"]-lock_time[trial], bins=20, range=(ranges[0],ranges[1]))
                x_counts[trial] = h
            ignore, bin_edges = np.histogram(lick_df["Time Licking"]-lock_time[trial], bins=bins)
            bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
            return(x_counts, bin_edges, bin_centres)

        #Segment by lick type - new test
        cherry_licks_only = self.lick_df.loc[(self.lick_df["Cherry Lick"] == 1)]
        grape_licks_only = self.lick_df.loc[(self.lick_df["Grape Lick"] == 1)]
        center_licks_only = self.lick_df.loc[(self.lick_df["Center Lick"] == 1)]

        """-----------------Lock data-------------------------------"""
        #Return spike counts and bin edges for a set of bins for a given trial data frame
        cherry_lick_counts, bin_edges, bin_centres = lock_and_count_for_licks(cherry_licks_only)
        grape_lick_counts, bin_edges, bin_centres = lock_and_count_for_licks(grape_licks_only)
        center_lick_counts, bin_edges, bin_centres = lock_and_count_for_licks(center_licks_only)


        """-----------------Split by trial type-------------------------------"""
        #Trial type licks - new test
        cherry_trial_cherry_licks = self.count_to_trial(cherry_reward_lick_trials, cherry_lick_counts)
        cherry_trial_grape_licks = self.count_to_trial(cherry_reward_lick_trials, grape_lick_counts)
        cherry_trial_center_licks = self.count_to_trial(cherry_reward_lick_trials, center_lick_counts)

        grape_trial_cherry_licks = self.count_to_trial(grape_reward_lick_trials, cherry_lick_counts)
        grape_trial_grape_licks = self.count_to_trial(grape_reward_lick_trials, grape_lick_counts)
        grape_trial_center_licks = self.count_to_trial(grape_reward_lick_trials, center_lick_counts)

        both_reward_trial_cherry_licks = self.count_to_trial(both_reward_lick_trials, cherry_lick_counts)
        both_reward_trial_grape_licks = self.count_to_trial(both_reward_lick_trials, grape_lick_counts)
        both_reward_trial_center_licks = self.count_to_trial(both_reward_lick_trials, center_lick_counts)

        no_reward_trial_cherry_licks = self.count_to_trial(no_reward_lick_trials, cherry_lick_counts)
        no_reward_trial_grape_licks = self.count_to_trial(no_reward_lick_trials, grape_lick_counts)
        no_reward_trial_center_licks = self.count_to_trial(no_reward_lick_trials, center_lick_counts)

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

        licks = normalised_cherry_trial_cherry_licks + normalised_cherry_trial_grape_licks + normalised_cherry_trial_center_licks
        licks = licks + normalised_grape_trial_cherry_licks + normalised_grape_trial_grape_licks + normalised_grape_trial_center_licks
        licks = licks + normalised_both_reward_trial_cherry_licks + normalised_both_reward_trial_grape_licks + normalised_both_reward_trial_center_licks
        licks = licks + normalised_no_reward_trial_cherry_licks + normalised_no_reward_trial_grape_licks + normalised_no_reward_trial_center_licks

        return(licks)
