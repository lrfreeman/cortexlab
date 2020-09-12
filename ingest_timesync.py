from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Output a tuple consisting of a data frame and an array of spiketimes
def convert_mat(file):

    #Converts the matlab struct into a python dictionary with keys (['__header__', '__version__', '__globals__', 'data'])
    mat_dict = loadmat(file)

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

    #Convert to df
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', None)
    # pd.set_option('display.max_colwidth', None)
    df = pd.DataFrame(nTrials + left_choices + free + left_rewards + right_rewards + violations + reward_times + trial_start_times).T
    df.columns = ["nTrials", "left_choices","free", "left_rewards","right_rewards","violations", "reward_times", "trial_start_times"]
    return(df,spiketimes)

df, spike_times = convert_mat('/Users/laurence/Desktop/Neuroscience/mproject/Data/aligned_physdata_KM011_2020-03-20_probe0.mat')
# print(df.iloc[:,-1])

#Assign spike times to trials
trial_start_times = df.iloc[:,-1]
nTrials = len(df.iloc[:,0])
mapped_spikes = np.asarray([])
trial_spikes = np.asarray([])

# Epoch filter - currently just for first trial as really slow
for n in range(1):
    for i in spike_times:
        #Remove spike times pre first trial
        if(i < trial_start_times[0]):
            continue
        #Select spike times within each trial range
        if(i >= trial_start_times[n] and i < trial_start_times[n+1]):
            trial_spikes = np.append(trial_spikes,[i])
        else:
            continue
    mapped_spikes = np.append(mapped_spikes, trial_spikes)
    # print(mapped_spikes)
for i in range(len(mapped_spikes)):
    mapped_spikes[i] = mapped_spikes[i] - trial_start_times[0]

#Peristimulus time histogram (PSTH) visualization
triggers = np.array(df.iloc[:,-1])
reward_times = np.array(df.iloc[:,-2])
bins = 10
plt.hist(mapped_spikes, bins = bins, histtype='step')
plt.axvline((reward_times[0]-trial_start_times[0]), color="r")
plt.title("Histogram where red line is reward time")
plt.xlabel("Time from stimulus onset [s]")
plt.ylabel("Count of Spikes")
plt.show()
