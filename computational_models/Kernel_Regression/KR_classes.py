import sys

# #Ensure the KR class is in the python path
sys.path.insert(1,'/Users/laurencefreeman/Documents/cortexlab')
sys.path.insert(1,'/Users/laurencefreeman/Documents/cortexlab/deepLabCut')

import electrophysiology.ingest_timesync as ingest
import pandas as pd
import numpy as np
import is_licking as lick
import fast_histogram

class ProcessData:

    def __init__(self, session_data, frame_alignment_data, dlc_video_csv):
        self.session_data = session_data
        self.frame_alignment_data = frame_alignment_data
        self.dlc_video_csv = dlc_video_csv

    """Generate trial_df, spike_df"""
    def load_data(self, session_data):
        trial_df, spike_times, cluster_IDs, cluster_types = ingest.convert_mat(session_data)
        trial_df = trial_df.drop(columns=["nTrials"])
        spike_df = pd.DataFrame(spike_times, columns =["Spike_times"])
        spike_df["cluster_ids"] = cluster_IDs
        return(trial_df,spike_df)

    def bin_the_session(self, time_bin, end_point):
        bins = np.arange(0, end_point, time_bin).tolist()
        self.bins = bins
        return(bins)

    """Generate first lick data frame"""
    def compute_the_first_lick(self):
        frame_times = ingest.import_frame_times(self.frame_alignment_data)
        df = lick.generate_licking_times(frame_times, self.dlc_video_csv)
        lick_df = lick.map_lick_to_trial_type(df,self.session_data)
        first_lick_df = lick.compute_1st_lick(lick_df)
        return(first_lick_df, lick_df)

class Generate_Synth_Data:
    def __init__(self,
                 end_time_of_session,
                 synthetic_trial_number,
                 number_of_artificial_spikes):
        self.end_time_of_session = end_time_of_session
        self.synthetic_trial_num = synthetic_trial_number
        self.num_of_artificial_spikes = number_of_artificial_spikes

    #Generate random times uniformly across the session
    def generate_event_times(self):
        return(np.random.default_rng().uniform(0, self.end_time_of_session, self.synthetic_trial_num))

    #Generate an array of synthetic spike ID's of zero
    def generate_unit(self):
        return(np.asarray(np.random.randint(1, size=(self.num_of_artificial_spikes,1))))

    def generate_spikes(self, event_center):

        """Create synethic spike times for the X variable"""
        # #The below code makes a list of lists for spike times such as: [[23s], [34s], [35s]] over a mean of reward time
        x = 0 # Create a counter
        spike_times = []
        while x < self.num_of_artificial_spikes: #Loop until spikes = lenght
            for time in event_center:
                y = []
                y.insert(0, np.random.default_rng().normal(loc = time, scale = 0.2)) #Create a spike with time as the mean
                spike_times.insert(x, y) #Here, 2nd arg is inserted to the list at the 1st arg index using the counter
                x += 1
        spike_times = np.asarray(spike_times)
        spike_times = spike_times[:self.num_of_artificial_spikes]

        return(spike_times)

class Analyze_data:
    def __init__(self,
                 Cell_ID,
                 spike_df,
                 range_back,
                 range_forward,
                 shift_back,
                 shift_total,
                 trial_df):
        self.cell_id = Cell_ID
        self.spike_df = spike_df
        self.shift_back = shift_back
        self.range_back = range_back
        self.range_forward = range_forward
        self.shift_total = shift_total
        self.trial_df = trial_df

    def histogram(self, bins, end_time):
        spike_df = self.spike_df.loc[(self.spike_df["cluster_ids"] == self.cell_id)]
        session_counts = fast_histogram.histogram1d(spike_df["spike_times"], bins=len(bins), range=(0,end_time))
        session_counts = pd.DataFrame(session_counts)
        session_counts = session_counts.rename(columns={0: "spike_counts"})
        session_counts = np.asarray(session_counts)

        self.bins = bins

        """Plot spike distribution for checking synethic data"""
        # plt.figure()
        # plt.plot(bins, Y)
        # plt.title("Histogram of synethic spikes")
        # plt.show()

        return(session_counts)

    #This function is not used and replaced with my own design matrix
    def parker_design_matrix(self, kernel):
        map_event_to_bin = np.digitize(kernel, bins) #Return the indices of the bins to which each value in input array belongs.
        bin_vector = np.zeros((len(bins), 1)) #Creates a vector of zeros of lenghh bin

        #Iterate through binned event indexes and assign that bin a value of 1 to represent the event
        for x in map_event_to_bin:
            bin_vector[x] = 1  #Creates a vector of 1's and 0's where 1's refers to the event time index of a bin

        i = 1 #start the index at 1 as to not break the looping logic
        shifted_vector = bin_vector[shift_back + shift_forward + 1 - i:]
        design_matrix = pd.DataFrame(shifted_vector) #Creates a dataframe of a shifted event vector with column header 0

        # Create an iterator for the below loop starting at 2 considering the first index was used above to create the df
        kernel_window_len_interator = np.arange(2,shift_total + 1,1)

        for i in kernel_window_len_interator: #Starting at 2 because the first bin is used in the skeleton dataframe
            shifted_vector = bin_vector[shift_back + shift_forward + 1 - i:-i + 1]
            design_matrix[str(i - 1)] = shifted_vector #Adds to the dataframe for the size of the kernel window, each column is a param

        return (design_matrix)

    #Under construction doesn't match other script yet
    def gen_design_matrix(self, kernel_event):

        map_event_to_bin = np.digitize(kernel_event, self.bins) #Return the indices of the bins to which each value in input array belongs.
        bin_vector = np.zeros((len(self.bins), 1)) #Creates a vector of zeros of lenght bin

        #Iterate through binned event indexes and assign that bin a value of 1 to represent the event
        for x in map_event_to_bin:
            bin_vector[x - 1] = 1  #Creates a vector of 1's and 0's where 1's refers to the event time index of a bin

        shifted_vector = bin_vector[self.shift_total - self.shift_back : - self.shift_back]
        design_matrix = pd.DataFrame(shifted_vector, columns = [str(self.shift_back)]) #Creates a dataframe of a shifted event vector with column header 0

        shifted_vector = bin_vector[self.shift_total:]

        for i in self.range_back:
            shifted_vector = bin_vector[self.shift_total - i: -i]
            design_matrix["-" + str(i)] = shifted_vector #Adds to the dataframe for the size of the kernel window, each column is a param

        shifted_vector = bin_vector[self.shift_total:]
        design_matrix["0"] = shifted_vector

        for i in self.range_forward:
            shifted_vector = bin_vector[i : -self.shift_total + i]
            design_matrix[str(i)] = shifted_vector #Adds to the dataframe for the size of the kernel window, each column is a param

        return (design_matrix)
