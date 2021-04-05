import numpy as np
import numpy.random as nr
from datoviz import canvas, run, colormap

import PredictLicking.is_licking as lick
import matplotlib.backends.backend_pdf
import electrophysiology.ingest_timesync as ingest
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import pandas as pd
import time
import seaborn as sns
import DLC_Classes as CL

data = CL.CortexLab('/Users/laurence/Desktop/Neuroscience/kevin_projects/data/processed_physdata/aligned_physdata_KM011_2020-03-23_probe1.mat',
                    '/Users/laurence/Desktop/Neuroscience/kevin_projects/data/KM011_video_timestamps/2020-03-23/face_timeStamps.mat',
                    '/Users/laurence/Desktop/Neuroscience/kevin_projects/data/23_faceDLC_resnet50_Master_ProjectAug13shuffle1_133500.csv')

trial_df, spike_df = data.load_data(data.session_data)
print(spike_df)

#Raster locked to reward - Returns a dictionary with key trial number and value locked spike times
def lock_and_sort_for_raster(time,trial_df):
    lock_time = {}
    trial_spike_times = {}
    for trial in range(len(trial_df)):
        lock_time[trial] = trial_df["reward_times"][trial]
        trial_spike_times[trial] = time-lock_time[trial]
    return(trial_spike_times)

spike_df = spike_df.loc[(spike_df["cluster_ids"] == 1)]
spike_times = np.asarray(spike_df["Spike_Times"])
spike_clusters = np.asarray(spike_df["cluster_ids"])

c = canvas(show_fps=False)
panel = c.panel(controller='axes')
visual = panel.visual('point')
N = len(spike_times)
numcells = len(np.unique(spike_clusters))
pos = np.c_[spike_times, np.zeros(N), np.zeros(N)]
# color = colormap(spike_clusters.astype(np.float64).flatten(), cmap='glasbey', alpha=.5)

#Loop function
for cell in range(3):
    visual.data('pos', pos)
    # visual.data('color', color)
    visual.data('ms', np.array([2.]))
    run(screenshot=f"{cell}screenshot.png",n_frames =5)
