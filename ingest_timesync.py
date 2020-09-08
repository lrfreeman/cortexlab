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

#variables with not same lenght
spiketimes = mat_dict["data"][0][0]["spiketimes"]
# spiketimes = np.spiketimes

#Convert to df
df = pd.DataFrame(left_choices + free + left_rewards + right_rewards + violations + reward_times + trial_start_times).T
df.columns = ["left_choices","free", "left_rewards","right_rewards","violations", "reward_times", "trial_start_times"]
# print(df)

#Make PSTH for a given variable
# df.plot(x="reward_times",y="left_rewards")
# # plt.ylabel("Proability")
# plt.show()
