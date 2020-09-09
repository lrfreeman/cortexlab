from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Converts the matlab struct into a python dictionary with keys (['__header__', '__version__', '__globals__', 'data'])
mat_dict = loadmat('/Users/laurence/Desktop/Neuroscience/mproject/Data/aligned_physdata_KM011_2020-03-20_probe0.mat')

#variables with same lenght of ntrails
left_choices = [[row.flat[0] for row in line] for line in mat_dict["data"][0][0]["left_choices"]]
free = [[row.flat[0] for row in line] for line in mat_dict["data"][0][0]["free"]]
left_rewards = [[row.flat[0] for row in line] for line in mat_dict["data"][0][0]["left_rewards"]]
right_rewards = [[row.flat[0] for row in line] for line in mat_dict["data"][0][0]["right_rewards"]]
violations = [[row.flat[0] for row in line] for line in mat_dict["data"][0][0]["violations"]]
reward_times = [[row.flat[0] for row in line] for line in mat_dict["data"][0][0]["rewardTimes"]]
trial_start_times = [[row.flat[0] for row in line] for line in mat_dict["data"][0][0]["trialstartTimes"]]
nTrials = [[row.flat[0] for row in line] for line in mat_dict["data"][0][0]["nTrials"]]

#variables with not same lenght
spiketimes = mat_dict["data"][0][0]["spiketimes"]
# spiketimes = [[row.flat[0] for row in line] for line in mat_dict["data"][0][0]["spiketimes"]]
# print(spiketimes)

#Convert to df
df = pd.DataFrame(left_choices + free + left_rewards + right_rewards + violations + reward_times + trial_start_times).T
df.columns = ["left_choices","free", "left_rewards","right_rewards","violations", "reward_times", "trial_start_times"]
# print(df)

#In neuroscience the peristimulus time histogram (PSTH) is used to visualize the
#timing of neuronal spiking relative to a given stimulus. To do so one divides the stimulus period
#into a defined number of bins and counts the spikes which occur in each bin for a given trial.
#Make PSTH for a given variable - Peristimulus time histogram
bin_width = 1000
#histogram function takes in bins / bin width, time between two stimuli and spike times
hist, bins = np.histogram(spiketimes, bins = bin_width)
plt.hist(spiketimes, bins)
plt.title("Histogram")
plt.xlabel("Spike Times [ms?]")
plt.ylabel("Count of Spikes")
plt.show()
