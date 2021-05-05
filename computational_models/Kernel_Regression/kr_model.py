#Import libaries
import KR_classes as KR
import numpy as np
import pandas as pd
import fast_histogram
import statsmodels.api as sm
from statsmodels.formula.api import glm
import matplotlib.pyplot as plt
from scipy import interpolate
import random
import seaborn as sns

"""Load data"""
data = KR.ProcessData(session_data = '/Users/laurence/Desktop/Neuroscience/kevin_projects/data/processed_physdata/aligned_physdata_KM011_2020-03-23_probe1.mat',
                      frame_alignment_data = '/Users/laurence/Desktop/Neuroscience/kevin_projects/data/KM011_video_timestamps/2020-03-23/face_timeStamps.mat',
                      dlc_video_csv = '/Users/laurence/Desktop/Neuroscience/kevin_projects/data/23_faceDLC_resnet50_Master_ProjectAug13shuffle1_133500.csv')
trial_df, spike_df = data.load_data(data.session_data)
reward_times = np.asarray(trial_df["reward_times"])
trial_start_times = np.asarray(trial_df["trial_start_times"])
spike_times = np.load("/Users/laurence/Desktop/spike_times.npy")
spike_clusters = np.load("/Users/laurence/Desktop/spike_clusters.npy")

#Need to introduce licking function into class
# first_lick_df, df, lick_df = data.produce_1st_lick()
# first_lick_times = np.asarray(first_lick_df["First Lick Times"])

"""Free Parameters"""
time_bin = 0.2 # time bin size for data in seconds
sampling_rate = 30000 # sampling rate of probe in Hz
cell_ID = 0 #Define the unit to filter spike times on
synthetic_trial_number = 500

#Defines how big kernels should be - must be intergers
shift_back = 15 #1.5seconds as bin count is set to 0.1 above
# shift_back = 5 #0.5seconds as bin count is set to 0.1 above
# shift_forward = 15 #1.5seconds as bin count is set to 0.1 above
shift_forward = 30 #3seconds as bin count is set to 0.1 above
shift_total = shift_back + shift_forward
kernal_window_range = np.arange(-shift_back,shift_forward,1).tolist()

"""Configure data"""
spike_times = spike_times / sampling_rate # (assumed sampling rate of 30000 Hz?)
bins = data.bin_the_session(time_bin)
end_time = np.max(spike_times) + 10  # End-point in seconds of the time period you're interested in

"""Create synethic reward times"""
reward_times = np.random.default_rng().uniform(0, end_time, synthetic_trial_number)

# print("Synthetic reward times", reward_times)

"""Create synethic lick times"""
#Create lick times distributed across reward times
#Add one second so as to always be after reward
# first_lick_times = [np.random.default_rng().normal(loc = time, scale = 0.2) + 1 for time in reward_times]

#Create lick times randomly
first_lick_times = np.random.default_rng().uniform(0, end_time, synthetic_trial_number)

# print("Synthetic lick times", first_lick_times)

"""Outline the kernels"""
kernels = [reward_times, first_lick_times] # Each item should be an event kernel

"""Create synethic spike times for X"""
length = 100000 #Create an artificial number of spikes
spike_clusters = np.asarray(np.random.randint(2, size=(length,1))) #Create a vector of 0's and 1's / 2x units

#The below code makes a list of lists for spike times such as: [[23s], [34s], [35s]] over a mean of reward time
x = 0 # Create a counter
spike_times = []
while x < length: #Loop until spikes = lenght
    for time in reward_times:
        y = []
        y.insert(0, np.random.default_rng().normal(loc = time, scale = 0.1)) #Create a spike with time as the mean
        spike_times.insert(x, y) #Here, 2nd arg is inserted to the list at the 1st arg index using the counter
        x += 1

spike_times = np.asarray(spike_times)
spike_times = spike_times[0:length]

"""Create the Y output"""
spike_data = pd.DataFrame(spike_times, columns = ["spike_times"])
spike_data['cluster_ids'] = spike_clusters

def histogram(spike_df, cell_ID):

    spike_df = spike_df.loc[(spike_df["cluster_ids"] == cell_ID)]

    #Logic for fast histogram
    session_counts = fast_histogram.histogram1d(spike_df["spike_times"], bins=len(bins), range=(0,end_time))
    session_counts = pd.DataFrame(session_counts)
    session_counts = session_counts.rename(columns={0: "spike_counts"})

    return(session_counts)

"""Define  Y""" # Spike counts assigned to each bin

Y = np.asarray(histogram(spike_data, cell_ID))

"""Plot spike distribution for checking synethic data"""
# plt.figure()
# plt.plot(bins, Y)
# plt.title("Histogram of synethic spikes")
# plt.show()

"""Create design matrix"""

def create_design_matrix(kernel):

        map_event_to_bin = np.digitize(kernel, bins) #Return the indices of the bins to which each value in input array belongs.
        bin_vector = np.zeros((len(bins), 1)) #Creates a vector of zeros of lenghh bin

        #Some length off by one bug where the last bin sends an index error so Ive excluded it with -2 index
        for x in map_event_to_bin[:-2]:
            bin_vector[x] = 1  #Creates a vector of 1's and 0's where 1's refers to the event time index of a bin

        i = 1 #start the index at 1 as to not break the looping logic
        shifted_vector = bin_vector[shift_back + shift_forward + 1 - i:-i]
        kernel_window_len_interator = np.arange(2,shift_total,1)
        design_matrix = pd.DataFrame(shifted_vector) #Creates a dataframe of a shifted event vector with column header 0

        for i in kernel_window_len_interator: #Starting at 2 because the first bin is used in the skeleton dataframe
            shifted_vector = bin_vector[shift_back + shift_forward + 1 - i:-i]
            design_matrix[str(i - 1)] = shifted_vector #Adds to the dataframe for the size of the kernel window, each column is a param

        return (design_matrix)

# design_matrix = create_design_matrix(reward_times)
design_matrix_reward = create_design_matrix(reward_times)
design_matrix_lick = create_design_matrix(first_lick_times)
design_matrix = pd.concat([design_matrix_reward, design_matrix_lick], axis=1)

"""Calculate coefficient matrix"""
R1 = np.corrcoef(design_matrix_reward, design_matrix_lick, rowvar=False)
sns.heatmap(R1, cmap = "GnBu", yticklabels=False, xticklabels=False)
plt.title("Correlation coefficient matrix measuring covariance between regressors")
plt.show()

"""Set up design matrix to run within sm.GLM"""
X = sm.add_constant(design_matrix, prepend=False)

"""Calculate the delta between x and Y caused by differing kernel size"""
delta = Y.shape[0] - X.shape[0]

"""Code to run sm.GLM"""
model = sm.GLM(Y[delta:], X, family = sm.families.Gaussian()).fit() #Remove the first 46 elements of Y to match the shape of the design matrix
summary_of_model = model.summary()
weights = model.params
weights = np.asarray(weights) #Constant is at the endd of the vector

"""Change x axis to bin centers"""
kernal_window_range = np.asarray(kernal_window_range)
bin_centres = 0.5*(kernal_window_range[1:] + kernal_window_range[:-1]) / 10 # to convert into seconds

"""""Plotting logic"""
plt.figure()
plt.plot(bin_centres, weights[:(shift_total - 1)], label = 'Reward Kernel') # cut last weight off ddue to use of bin centres
# plt.plot(bin_centres, weights[shift_total - 1: -1], label = 'First Lick Kernel') # cut constant and last weighht off
plt.xlabel("Kernel Window (seconds)", fontsize = 12)
plt.ylabel("Coefficients", fontsize = 12)
# plt.title("Kernel Regression: cluster ID {} ".format(cell_ID), fontsize = 12)
plt.legend()
plt.axvline(x=0, color = "r", linewidth=0.9)
plt.axhline(y=0, color = "k", linewidth=0.9)
plt.show()

"""Print important data points"""
print("")
print("Lenghth of bins", len(bins))
print("Design Matrix Shape:", design_matrix.shape)
print("Lenght of Y", len(Y[46:]))
print("Lenght of Weights", len(weights))
# print("Correlation coefficient matrix", R1)
print("")

#Tests
# assert design_matrix.shape == (len(bins), shift_total), "Design Matrix is an incorrect shape"
