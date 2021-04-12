import sys
sys.path.insert(1,'/Users/laurence/Desktop/Neuroscience/kevin_projects/code/mousetask')
import kr_classes as KR
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go

#Extend data print rows
# pd.set_option("display.max_rows", None, "display.max_columns", None)

#Free Parameters
bin_size = 0.2
kernel_window_range = np.arange(-1,2.2,0.2).tolist()

"""Load data"""
data = KR.ProcessData(session_data = '/Users/laurence/Desktop/Neuroscience/kevin_projects/data/processed_physdata/aligned_physdata_KM011_2020-03-23_probe1.mat',
                      frame_alignment_data = '/Users/laurence/Desktop/Neuroscience/kevin_projects/data/KM011_video_timestamps/2020-03-23/face_timeStamps.mat',
                      dlc_video_csv = '/Users/laurence/Desktop/Neuroscience/kevin_projects/data/23_faceDLC_resnet50_Master_ProjectAug13shuffle1_133500.csv')
trial_df, spike_df = data.load_data(data.session_data)

"""Process the data"""
bins = data.bin_the_session(bin_size)
absolute_time = np.arange(trial_df["trial_start_times"][0],
                          trial_df["trial_start_times"].iloc[-1],
                          0.02).tolist()


#Look at licking code and figure out the difference between the last two df's
first_lick_df, lick_df, df = data.produce_licking_data()
binned_session, bin_edges, bin_centres = data.histogram(spike_df, bins, 2)
binned_session["bin_edges"] = bin_edges

"""Commence Kernel Regression"""
kernel_object = KR.KernelRegression(binned_session, kernel_window_range)

reward_spikes_at_time = {}
tao_index = 0
for event in range(len(trial_df["reward_times"])):
    tao_index+= 1
    print("Tao_Index:", tao_index)
    for time in absolute_time:
        time = round(time, 2)
        # if (time - trial_df["reward_times"][tao_index] > -3):
        #     continue
        reward_spikes_at_time[time] = kernel_object.event_summation(round(trial_df["reward_times"],2), time, tao_index)
        #If time has gone past the possible kernel window move to next behavioural event
        if (time > (2.1 + kernel_object.tao)):
            break
