import sys

#Ensure the KR class is in the python path
sys.path.insert(1,'/Users/laurence/Desktop/Neuroscience/kevin_projects/code/mousetask')
sys.path.insert(1,'/Users/laurence/Desktop/Neuroscience/kevin_projects/code/mousetask/deepLabCut')

import electrophysiology.ingest_timesync as ingest
import pandas as pd
import numpy as np
import is_licking as lick


class ProcessData:

    def __init__(self, session_data, frame_alignment_data, dlc_video_csv):
        self.session_data = session_data
        self.frame_alignment_data = frame_alignment_data
        self.dlc_video_csv = dlc_video_csv

    """Generate trial_df, spike_df"""
    def load_data(self, session_data):
        trial_df, spike_times, cluster_IDs, cluster_types = ingest.convert_mat(session_data)
        trial_df = trial_df.drop(columns=["nTrials"])

        spike_dic = {"Spike_Times": [spike_times], "cluster_ids": [cluster_types]}
        spike_df =  pd.DataFrame(spike_dic)

        self.spike_times = spike_times
        self.spike_df = spike_df
        self.df = trial_df

        return(trial_df,spike_df)

    def bin_the_session(self, time_bin):
        bins = np.arange(0,np.max(self.spike_times),time_bin).tolist()
        return(bins)

    """Generate first lick data frame"""
    def compute_the_first_lick(self):
        frame_times = ingest.import_frame_times(self.frame_alignment_data)
        df = lick.generate_licking_times(frame_times, self.dlc_video_csv)
        lick_df = lick.map_lick_to_trial_type(df,self.session_data)
        first_lick_df = lick.compute_1st_lick(lick_df)
        return(first_lick_df)
