#Import sys library to deal with my poor package setup
import sys

#Ensure the KR class is in the python path
sys.path.insert(1,'/Users/laurence/Desktop/Neuroscience/kevin_projects/code/mousetask/kernel_regression_kush')
sys.path.insert(1,'/Users/laurence/Desktop/Neuroscience/kevin_projects/code/mousetask')

#Import libaries
import kr_classes as KR
import numpy as np
import pandas as pd
import fast_histogram
import statsmodels.api as sm
from statsmodels.formula.api import glm
import matplotlib.pyplot as plt

# np.set_printoptions(threshold=sys.maxsize)

"""Load data"""
data = KR.ProcessData(session_data = '/Users/laurence/Desktop/Neuroscience/kevin_projects/data/processed_physdata/aligned_physdata_KM011_2020-03-23_probe1.mat',
                      frame_alignment_data = '/Users/laurence/Desktop/Neuroscience/kevin_projects/data/KM011_video_timestamps/2020-03-23/face_timeStamps.mat',
                      dlc_video_csv = '/Users/laurence/Desktop/Neuroscience/kevin_projects/data/23_faceDLC_resnet50_Master_ProjectAug13shuffle1_133500.csv')
trial_df, spike_df = data.load_data(data.session_data)
reward_times = np.asarray(trial_df["reward_times"])
trial_start_times = np.asarray(trial_df["trial_start_times"])
spike_times = np.load("/Users/laurence/Desktop/spike_times.npy")
spike_clusters = np.load("/Users/laurence/Desktop/spike_clusters.npy")
first_lick_df, lick_df, df = data.produce_licking_data()
first_lick_times = np.asarray(first_lick_df["First Lick Times"])

"""Free Parameters"""
time_bin = 0.1 # time bin size for data in seconds
sampling_rate = 30000 # sampling rate of probe in Hz
cell_ID = 0

#Defines how big kernels should be
shift_back = 15 #1.5seconds as bin count is set to 0.1 above
shift_forward = 30 #3seconds as bin count is set to 0.1 above
shift_total = shift_back + shift_forward
kernal_window_range = np.arange(-shift_back,shift_forward,1).tolist()

"""Configure data"""
spike_times = spike_times / sampling_rate # (assumed sampling rate of 30000 Hz?)
bins = data.bin_the_session(time_bin)
end_time = np.max(spike_times) + 10  # End-point in seconds of the time period you're interested in
kernels = [reward_times, first_lick_times] # Each item should be an event kernel

"""Create synethic X"""
length = 64209750 #Create an artificial number of spikes
spike_clusters = np.asarray(np.random.randint(2, size=(length,1))) #Create a vector of 0's and 1's / 2x units

x = 0 # Create a counter
spike_times = []
while x < length: #Loop until spikes = lenght
    for time in reward_times:
        y = []
        y.insert(0, np.random.normal(loc = time, scale = 0.1)) #Create a spike with time as the mean
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

Y = np.asarray(histogram(spike_data, cell_ID))

"""Create design matrix"""

design_matrix_dic = {}

def create_design_matrix(kernels):

    for z in range(len(kernels)):

        map_event_to_bin = np.digitize(kernels[z], bins) #Return the indices of the bins to which each value in input array belongs.
        bin_vector = np.zeros((len(bins), 1)) #Creates a vector of zeros of lenghh bin

        #Some length off by one bug where the last bin sends an index error so Ive excluded it with -2 index
        for x in map_event_to_bin[:-2]:
            bin_vector[x] = 1  #Creates a vector of 1's and 0's where 1's refers to the event time index of a bin

        #Create the design matrix
        i = 1 #start the index at 1 as to not break the looping logic
        shifted_vector = bin_vector[shift_back + shift_forward + 1 - i:-i]
        list = np.arange(2,46,1)
        design_matrix = pd.DataFrame(shifted_vector) #Creates a dataframe of a shifted event vector with column header 0

        for i in list: #Starting at 2
            shifted_vector = bin_vector[shift_back + shift_forward + 1 - i:-i]
            design_matrix[str(i - 1)] = shifted_vector #Adds to the dataframe for the size of the kernel window, each column is a param

        design_matrix_dic[str(z)] = design_matrix

    return (design_matrix_dic)

design_matrix_dic = create_design_matrix(kernels)
design_matrix = design_matrix_dic["0"]
design_matrix = pd.concat([design_matrix_dic["0"], design_matrix_dic["1"]], axis=1)
print("New dm", design_matrix.shape)

print(design_matrix_dic["0"])
print(design_matrix_dic["1"])

print(design_matrix)

"""Set up design matrix to run within sm.GLM"""
X = sm.add_constant(design_matrix, prepend=False)

"""Code to run sm.GLM"""
model = sm.GLM(Y[46:], X, family = sm.families.Gaussian()).fit() #Remove the first 46 elements of Y to match the shape of the design matrix
summary_of_model = model.summary()
print(summary_of_model)
weights = model.params
weights = np.asarray(weights) #Constant is at the endd of the vector
print(weights)

"""Change x axis to bin centers"""
kernal_window_range = np.asarray(kernal_window_range)
bin_centres = 0.5*(kernal_window_range[1:] + kernal_window_range[:-1]) / 10 # to convert into seconds

"""""Plotting logic"""
plt.figure()
plt.plot(bin_centres, weights[:44], label = 'Reward Kernel') # cut last weight off ddue to use of bin centres
plt.plot(bin_centres, weights[44:-3], label = 'First Lick Kernel') # cut constant and last weighht off
plt.xlabel("Kernel Window (seconds)", fontsize = 12)
plt.ylabel("Coefficients", fontsize = 12)
plt.title("Kernel Regression: cluster ID {} ".format(cell_ID), fontsize = 12)
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
print("")


#Tests
# assert design_matrix.shape == (len(bins), shift_total), "Design Matrix is an incorrect shape"
