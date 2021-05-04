import DLC_Classes as CL
import numpy as np
import fast_histogram
import pandas as pd
import matplotlib.pyplot as plt


"""Load data"""
data = CL.CortexLab(session_data = '/Users/laurence/Desktop/Neuroscience/kevin_projects/data/processed_physdata/aligned_physdata_KM011_2020-03-23_probe1.mat',
                      frame_alignment_data = '/Users/laurence/Desktop/Neuroscience/kevin_projects/data/KM011_video_timestamps/2020-03-23/face_timeStamps.mat',
                      dlc_video_csv = '/Users/laurence/Desktop/Neuroscience/kevin_projects/data/23_faceDLC_resnet50_Master_ProjectAug13shuffle1_133500.csv')
trial_df, spike_df = data.load_data(data.session_data)

"""Gen synth data"""
length = 6420975
spike_clusters = np.asarray(np.random.randint(2, size=(length,1)))

reward_times = trial_df["reward_times"]

x = 0
spike_times = []
while x < length:
    for time in reward_times:
        y = []
        y.insert(0, np.random.normal(loc = time, scale = 0.1))
        spike_times.insert(x, y)
        x += 1

spike_times = np.asarray(spike_times)
spike_times = spike_times[0:length]

ranges=[-1,3]
bins = np.arange(-1,3,0.2).tolist()
lock_time = {}
x_counts = {}

for trial in range(len(reward_times)):
    lock_time[trial] = reward_times[trial]
    h = fast_histogram.histogram1d(spike_times-lock_time[trial], bins=20, range=(ranges[0],ranges[1]))
    x_counts[trial] = h
ignore, bin_edges = np.histogram(spike_df["Spike_Times"]-lock_time[trial], bins=bins)
bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])

spikecount = pd.DataFrame(x_counts).sum(axis=1)

#Test single cell
plt.figure()
plt.plot(bin_centres, spikecount[:-1])
# plt.xlabel("Kernel Window (Seconds)", fontsize = 12)
# plt.ylabel("Coefficients", fontsize = 12)
plt.title("psth for syncth data {} ", fontsize = 12)
plt.axvline(x=0, color = "r", linewidth=0.9)
plt.axhline(y=0, color = "k", linewidth=0.9)
plt.show()
