import DLC_Classes as CL
import math
import data_visulisation_functions.Kernel_Regression_functions as KR
import numpy as np

data = CL.CortexLab('/Users/laurence/Desktop/Neuroscience/mproject/data/processed_physdata/aligned_physdata_KM011_2020-03-23_probe1.mat',
                 '/Users/laurence/Desktop/Neuroscience/mproject/data/KM011_video_timestamps/2020-03-23/face_timeStamps.mat',
                 '/Users/laurence/Desktop/Neuroscience/mproject/data/23_faceDLC_resnet50_Master_ProjectAug13shuffle1_133500.csv')

trial_df, spike_df = data.load_data(data.session_data)
first_lick_df, lick_df, df = data.compute_the_first_lick()

trunc = lambda x: math.trunc(x)
trial_df["Trial IDs"] = trial_df["trial_start_times"].apply(trunc)

data = first_lick_df.merge(trial_df, on="Trial IDs")

y = np.asarray(data["First Lick Times"])
x = np.asarray(data["reward_times"])
bandwith = 0.1
gkr = KR.GKR(x,y,bandwith)
gkr.test()

print(x)
print(y)
