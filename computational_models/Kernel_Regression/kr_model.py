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
import sys

"""Load data"""
data = KR.ProcessData(session_data = '/Users/laurencefreeman/Documents/thesis_data/processed_physdata/aligned_physdata_KM011_2020-03-23_probe1.mat',
                      frame_alignment_data = '/Users/laurencefreeman/Documents/thesis_data/KM011_video_timestamps/2020-03-23/face_timeStamps.mat',
                      dlc_video_csv = '/Users/laurencefreeman/Documents/thesis_data/23_faceDLC_resnet50_Master_ProjectAug13shuffle1_133500.csv')
trial_df, spike_df = data.load_data(data.session_data)
reward_times = np.asarray(trial_df["reward_times"])
trial_start_times = np.asarray(trial_df["trial_start_times"])
spike_times = np.asarray(spike_df["Spike_times"])
spike_clusters = np.asarray(spike_df["cluster_ids"])
first_lick_df = data.compute_the_first_lick()
first_lick_times = np.asarray(first_lick_df["First Lick Times"])

print("The data has been loaded")

"""Free Parameters"""
time_bin = 0.1 # time bin size for data in seconds
sampling_rate = 30000 # sampling rate of probe in Hz
cell_ID = 3 #Define the unit to filter spike times on
synthetic_trial_number = 800
length = 100000 #Create an artificial number of spikes of lenght....
shift_back = 15 #15 would be 1.5 seconds if bin count is set to 0.1 above - How far should the kernel look back?
shift_forward = 30 #30 would be 3 seconds if bin count is set to 0.1 above - How far should the kernel look forwards?
shift_total = shift_back + shift_forward
kernel_window_range = np.arange(-shift_back,shift_forward,1).tolist()
range_back = np.sort(np.arange(1,shift_back, 1))[::-1]
range_forward = np.arange(1,shift_forward + 1, 1)

print("The free parameters have been set")

"""Configure data"""
end_time = np.max(spike_times) + 5  # End-point in seconds of the time period you're interested in
# end_time = 5000 #Use if using synethic data
bins = data.bin_the_session(time_bin, end_time)

"""Instantiate synthic class and generated synethic data"""
# synth_data =       KR.Generate_Synth_Data(end_time, synthetic_trial_number, length)
# reward_times   =   synth_data.generate_event_times() #Generate event times uniformly across the session
# # first_lick_times = synth_data.generate_event_times() #Use this line to make uncouple lick times from reward
# first_lick_times = [np.random.default_rng().normal(loc = time, scale = 0.1 ) for time in reward_times] #Create lick times distributed across reward times
# spike_clusters =   synth_data.generate_unit() #Generate an array of 0's length of spikes

"""Create synethic spike times for X"""
def create_spikes(num_of_spikes):
    # #The below code makes a list of lists for spike times such as: [[23s], [34s], [35s]] over a mean of reward time
    x = 0 # Create a counter
    spike_times = []
    while x < num_of_spikes: #Loop until spikes = lenght
        for time in reward_times:
            y = []
            y.insert(0, np.random.default_rng().normal(loc = time, scale = 0.2)) #Create a spike with time as the mean
            spike_times.insert(x, y) #Here, 2nd arg is inserted to the list at the 1st arg index using the counter
            x += 1
            assert x <= num_of_spikes, "loop broken"

    spike_times = np.asarray(spike_times)
    spike_times = spike_times[:length]

    return(spike_times)
# spike_times = create_spikes(length)

"""_Create the Y output_"""
spike_data = pd.DataFrame(spike_times, columns = ["spike_times"])
spike_data['cluster_ids'] = spike_clusters

"""_Analyze the  data_"""
analysis_object = KR.Analyze_data(cell_ID,
                                  spike_data,
                                  range_back,
                                  range_forward,
                                  shift_back,
                                  shift_total,
                                  trial_df)

print("The analysis object has been created")

"""_Calculate the number of spikes in each bin to produce the dependent Y variable for the regression_"""
Y = analysis_object.histogram(bins, end_time) # Spike counts assigned to each bin
Y = Y[shift_total:-shift_total] # truncate spike signal to fit x-mat - #Check with Kevin

"""_Create design matrix_"""
def gen_design_matrix(kernel):

        map_event_to_bin = np.digitize(kernel, bins) #Return the indices of the bins to which each value in input array belongs.
        bin_vector = np.zeros((len(bins), 1)) #Creates a vector of zeros of lenghh bin

        #Iterate through binned event indexes and assign that bin a value of 1 to represent the event
        for x in map_event_to_bin:
            bin_vector[x - 1] = 1  #Creates a vector of 1's and 0's where 1's refers to the event time index of a bin

        shifted_vector = bin_vector[shift_total - 15: -15 - shift_total] # new - second  segment should be shif back -1 so -15 - 1 = -16
        design_matrix = pd.DataFrame(shifted_vector, columns = ["-15"]) #Creates a dataframe of a shifted event vector with column header 0

        for i in range_back:
            shifted_vector = bin_vector[shift_total - i: -i - shift_total]
            design_matrix["-" + str(i)] = shifted_vector #Adds to the dataframe for the size of the kernel window, each column is a param

        shifted_vector = bin_vector[shift_total:-shift_total]
        design_matrix["0"] = shifted_vector

        for i in range_forward:
            shifted_vector = bin_vector[i + shift_total : - shift_total + i]
            design_matrix[str(i)] = shifted_vector #Adds to the dataframe for the size of the kernel window, each column is a param

        return (design_matrix)

design_matrix_reward_new = gen_design_matrix(reward_times)
design_matrix_lick_new =  gen_design_matrix(first_lick_times)
design_matrix = pd.concat([design_matrix_reward_new, design_matrix_lick_new], axis=1)
# design_matrix = design_matrix_reward_new

print("The design matrix has been created with shape:", design_matrix.shape)

"""Calculate coefficient matrix"""
# R1 = np.corrcoef(design_matrix_reward_new, design_matrix_lick_new, rowvar=False)
# sns.heatmap(R1, cmap = "GnBu", yticklabels=False, xticklabels=False)
# plt.title("Correlation coefficient matrix")
# plt.show()

"""Set up design matrix to run within sm.GLM"""
X = sm.add_constant(design_matrix, prepend=False) #prepend=False means a column of ones has been appended to the end for the constant

"""Code to run sm.GLM"""
model = sm.GLM(Y, X, family = sm.families.Gaussian()).fit()
summary_of_model = model.summary()
# print(summary_of_model)
weights = model.params
weights = np.asarray(weights) #Constant is at the endd of the vector

print("Model parameters have been analysed, and weights are now available")

"""Prediction code"""
# predicted_x = model.predict()

"""Change x axis to seconds"""
kernel_window_range = np.asarray(kernel_window_range) / 10 # to convert into seconds

"""""Plotting logic"""
plt.figure()
plt.plot(kernel_window_range, weights[:shift_total], label = 'Reward Kernel')
plt.plot(kernel_window_range, weights[shift_total + 1:-2], label = 'Lick Kernel') # Cut last weight off as it's the y intercept
plt.xlabel("Kernel Window (seconds)", fontsize = 12)
plt.ylabel("Coefficients", fontsize = 12)
plt.title("Kernel Regression: cluster ID {} ".format(cell_ID), fontsize = 12)
plt.legend()
plt.axvline(x=0, color = "r", linewidth=0.9)
plt.axhline(y=0, color = "k", linewidth=0.9)
plt.show()
