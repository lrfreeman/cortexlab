import deepLabCut.DLC_Classes as CL
import time
import fast_histogram
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from matplotlib.gridspec import GridSpec

#Performance checks
start_time = time.time()

"""--------------------Upload Data------------------"""

data = CL.CortexLab('/Users/laurencefreeman/Documents/thesis_data/processed_physdata/aligned_physdata_KM011_2020-03-23_probe1.mat',
                    '/Users/laurencefreeman/Documents/thesis_data/KM011_video_timestamps/2020-03-23/face_timeStamps.mat',
                    '/Users/laurencefreeman/Documents/thesis_data/23_faceDLC_resnet50_Master_ProjectAug13shuffle1_133500.csv')

trial_df, spike_df = data.load_data(data.session_data)

"""-----------------Compute lick data-------------------------------"""
first_lick_df, lick_df, df = data.compute_the_first_lick()
licks = lick_df["Time Licking"]

"""-------------------Lock function--------------------------------"""

def lock_and_count_spike(spike_df, trial_df, cell_id, bin_range):
    ranges=bin_range
    lock_time = {}
    spike_counts = {}
    spike_df = spike_df.loc[(spike_df["cluster_ids"] == cell_id)]
    for trial in range(len(trial_df)):
        lock_time[trial] = trial_df["reward_times"][trial]
        h = fast_histogram.histogram1d(spike_df["Spike_Times"]-lock_time[trial], bins=1, range=(ranges[0],ranges[1]))
        spike_counts[trial] = h

    return(list(spike_counts.values()))

def lock_and_count(time_of_interest,spike_df, trial_df, cell_id, bin_range):
    ranges=bin_range
    lock_time = {}
    spike_counts = {}
    spike_df = spike_df.loc[(spike_df["cluster_ids"] == cell_id)]
    for trial in range(len(trial_df)):
        lock_time[trial] = trial_df["reward_times"][trial]
        h = fast_histogram.histogram1d(time_of_interest-lock_time[trial], bins=1, range=(ranges[0],ranges[1]))
        spike_counts[trial] = h

    return(list(spike_counts.values()))

"""-------------------free params-----------------------------------"""
bin_range = [-1,1]

"""-------------------------Data viz----------------------------"""

def produce_multi_charts_scatter(cell_id):
    """-------------------Product spike counts---------------------------"""
    spike_counts = lock_and_count_spike(spike_df, trial_df, cell_id, bin_range)
    spike_counts  = [item for sublist in spike_counts for item in sublist]

    """-------------------Product lick counts---------------------------"""
    lick_counts = lock_and_count(licks, spike_df, trial_df, cell_id, bin_range)

    """-------------------Create df-------------------------------------"""
    results = pd.DataFrame(lick_counts, columns = ["lick_counts"])
    results["spike_counts"] = spike_counts

    """-------------------Create viz-------------------------------------"""
    y = results["spike_counts"]
    x = results["lick_counts"]
    # fig = plt.figure()
    # plt.scatter(x, y ,label = "Individual trials", marker = "x")
    # plt.xlabel("Number of licks", fontsize = 12)
    # plt.ylabel("Number of spikes", fontsize = 12)
    # plt.title("For each trial of cluster ID {}, lick count vs spike count".format(cell_id), fontsize = 12)
    # plt.legend()
    # start with a square Figure
    m, b = np.polyfit(x, y, 1)
    fig = plt.figure()
    gs = GridSpec(4,4)
    ax_joint = fig.add_subplot(gs[1:4,0:3])
    ax_marg_x = fig.add_subplot(gs[0,0:3])
    ax_marg_y = fig.add_subplot(gs[1:4,3])
    ax_joint.scatter(x,y,marker = "x", label = "Individual trials")
    ax_joint.plot(x, m*x + b, "r-")
    ax_marg_x.hist(x)
    ax_marg_y.hist(y,orientation="horizontal")
    ax_marg_x.set_title("For each trial of cluster ID {}, lick count vs spike count".format(cell_id), fontsize = 12)

    # Turn off tick labels on marginals
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    # Set labels on joint
    ax_joint.set_xlabel('Number of licks')
    ax_joint.set_ylabel('Number of spikes')

    return(fig)

# Multiple viz gen
pdf = matplotlib.backends.backend_pdf.PdfPages("lick_vs_spike.pdf")
for cell_id in range(30):
    fig = produce_multi_charts_scatter(cell_id)
    pdf.savefig(fig)
    plt.close(fig)
pdf.close()

#Print the time of the process
print("")
print("--- %s seconds ---" % (time.time() - start_time))
print("")
