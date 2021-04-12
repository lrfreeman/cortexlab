import sys
sys.path.insert(1,'/Users/laurence/Desktop/Neuroscience/kevin_projects/code/mousetask/electrophsyiology')
import pandas as pd
import electrophysiology.ingest_timesync as ingest #Conversion of matlab to python
import numpy as np
import PredictLicking.is_licking as lick #DLC code to predict licking
import fast_histogram

class KernelRegression:
    def __init__(self, binned_session_data, kernel_window_range):
        self.bin_edges = binned_session_data["bin_edges"]
        self.spike_counts = binned_session_data["spike_counts"]
        self.k_window = kernel_window_range
        print("Kernel Regression object initialized")

    def kernel_function(self, t_prime):
        index = (self.bin_edges - (self.absolute_time + t_prime)).abs().idxmin()
        spike_count = self.spike_counts[index]
        return(spike_count)

    def kronecker_delta_func(self,tao,t_prime):
        if((self.absolute_time - tao) == t_prime):
            print("Absolute Time:", self.absolute_time)
            print("Reward Time:", tao)
            print("T-prime:", t_prime)
            print("Difference:", self.absolute_time - tao)
            print("Kronecker function: True")
            return(1)
        elif((self.absolute_time - tao) != t_prime):
            # print("Absolute Time:", self.absolute_time)
            # print("Reward Time:", tao)
            # print("T-prime:", t_prime)
            # print("Difference:", self.absolute_time - tao)
            # print("Kronecker function: False")
            return(0)
        else:print("Impossible logic occured within the kronecker function")

    def kernel_window_summation(self):
        spike_counts = 0
        for t_prime in self.k_window:
            t_prime = round(t_prime, 2)
            spike_counts += np.sum(self.kronecker_delta_func(self.tao,t_prime) * self.kernel_function(t_prime))
            kronecker = self.kronecker_delta_func(self.tao,t_prime)
            self.kronecker = kronecker
            #Don't iterate if kronecker is false
            if self.kronecker == 0:
                break
        return(spike_counts)

    def event_summation(self, event_times, absolute_time, tao_index):
        self.event_time = event_times
        self.absolute_time = absolute_time
        spike_counts = 0
        self.tao = event_times[tao_index]
        spike_counts = self.kernel_window_summation()
        return(spike_counts)

class ProcessData:
    def __init__(self, session_data, frame_alignment_data,
                 dlc_video_csv):
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
        self.trial_df = trial_df
        return(trial_df,spike_df)

    """Bin the session"""
    def bin_the_session(self, bin_size):
        bins = np.arange(self.trial_df["trial_start_times"][0],
                         self.trial_df["trial_start_times"].iloc[-1],
                         bin_size)
        self.bins = bins
        return(bins)

    """Generate first lick data frame"""
    def produce_licking_data(self):
        frame_times = ingest.import_frame_times(self.frame_alignment_data)
        df = lick.generate_licking_times(frame_times, self.dlc_video_csv)
        lick_df = lick.map_lick_to_trial_type(df,self.session_data)
        first_lick_df = lick.compute_1st_lick(lick_df)
        self.firstlick_df = first_lick_df
        self.lick_df = lick_df
        return(first_lick_df, lick_df, df)

    """ Lock spikes to your event """
    def histogram(self, spike_df, bins, cell_ID):

        spike_df = spike_df.loc[(spike_df["cluster_ids"] == cell_ID)]

        #Logic for fast histogram
        session_counts = fast_histogram.histogram1d(spike_df["Spike_Times"], bins=len(self.bins), range=(bins[0],bins[-1]))
        session_counts = pd.DataFrame(session_counts)
        session_counts = session_counts.rename(columns={0: "spike_counts"})

        #Logic to calculate bin centres
        ignore, bin_edges = np.histogram(spike_df["Spike_Times"], bins=bins)
        bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
        return(session_counts, bin_edges, bin_centres)
