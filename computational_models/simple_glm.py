from Kernel_Regression import KR_classes as KR
import numpy as np
import pandas as pd
import fast_histogram
import statsmodels.api as sm
from statsmodels.formula.api import glm
import matplotlib.pyplot as plt
from scipy import interpolate
import random
import seaborn as sns

#Left reward is cherry, #right reward is grape
# pd.set_option('display.max_columns', None)  # or 1000
# pd.set_option('display.max_rows', None)  # or 1000
# pd.set_option('display.max_colwidth', None)  # or 199

"""Load data"""
data = KR.ProcessData(session_data = '/Users/laurencefreeman/Documents/thesis_data/processed_physdata/aligned_physdata_KM011_2020-03-23_probe1.mat',
                      frame_alignment_data = '/Users/laurencefreeman/Documents/thesis_data/KM011_video_timestamps/2020-03-23/face_timeStamps.mat',
                      dlc_video_csv = '/Users/laurencefreeman/Documents/thesis_data/23_faceDLC_resnet50_Master_ProjectAug13shuffle1_133500.csv')
trial_df, spike_df = data.load_data(data.session_data)
reward_times = np.asarray(trial_df["reward_times"])
trial_start_times = np.asarray(trial_df["trial_start_times"])
spike_times = np.asarray(spike_df["Spike_times"])
spike_clusters = np.asarray(spike_df["cluster_ids"])
first_lick_df, lick_df = data.compute_the_first_lick()
first_lick_times = np.asarray(first_lick_df["First Lick Times"])

def lock_and_count(spike_df, trial_df, bin_number, ranges, cell_id):
    lock_time = {}
    spike_counts = {}
    spike_df = spike_df.loc[(spike_df["cluster_ids"] == cell_id)]
    for trial in range(len(trial_df)):
        lock_time[trial] = trial_df["reward_times"][trial]
        spike_counts[trial] = fast_histogram.histogram1d(spike_df["Spike_times"]-lock_time[trial], bins=bin_number, range=(ranges[0],ranges[1]))
    return(pd.DataFrame(list(spike_counts.values())))
spike_matrix = lock_and_count(spike_df, trial_df, 30, [-1,3], 1)

def lock_and_count_lick(lick_df, trial_df, bin_number, ranges):
    lock_time = {}
    spike_counts = {}
    for trial in range(len(trial_df)):
        lock_time[trial] = trial_df["reward_times"][trial]
        spike_counts[trial] = fast_histogram.histogram1d(lick_df["Time Licking"]-lock_time[trial], bins=bin_number, range=(ranges[0],ranges[1]))
    return(pd.DataFrame(list(spike_counts.values())))
lick_matrix = lock_and_count_lick(lick_df, trial_df, 30, [-1,3])

def gen_reward_matrix(trial_df):
    reward_matrix = trial_df
    reward_matrix["both"] = 0
    reward_matrix["both"].loc[(reward_matrix['left_rewards'] == 1) & (reward_matrix['right_rewards'] == 1)] = 1
    reward_matrix["neither"] = 0
    reward_matrix["neither"].loc[(reward_matrix['left_rewards'] == 0) & (reward_matrix['right_rewards'] == 0)] = 1
    reward_matrix = reward_matrix.rename(columns={"left_rewards": "cherry", "right_rewards": "grape"})
    #Remove none-free choice trials
    # reward_matrix = reward_matrix.loc[(reward_matrix['free'] == 1)
    reward_matrix = reward_matrix.drop(["left_choices", "violations", "reward_times", "trial_start_times", "free"], axis = 1)
    return(reward_matrix)

def single_bin_design_matrix(bin_index):
    reward_matrix = gen_reward_matrix(trial_df)
    reward_matrix["licks"] = lick_matrix.iloc[:,bin_index]
    return(reward_matrix)

def LL_calc(index):
    dm = single_bin_design_matrix(index)
    Y = spike_matrix.iloc[:,index]
    X = sm.add_constant(dm, prepend=False)
    model = sm.OLS(Y, X)
    results = model.fit()
    #print(results.summary())
    LL = results.llf
    return(LL)

LL = {}
for x in range(10):
    LL[x] = LL_calc(x)

"""""Plotting logic"""
plt.figure()
plt.plot(LL.keys(), LL.values(), label = "1 out of 3 variant models [lick + rewards]")
plt.xlabel("Bins", fontsize = 12)
plt.ylabel("LL", fontsize = 12)
plt.legend()
plt.show()
