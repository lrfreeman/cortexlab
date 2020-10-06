import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spykes.plot.neurovis import NeuroVis
from spykes.ml.neuropop import NeuroPop
from spykes.utils import train_test_split
from ingest_timesync import *

df, spike_times = convert_mat('/Users/laurence/Desktop/Neuroscience/mproject/Data/aligned_physdata_KM011_2020-03-20_probe0.mat')

neuron_PMd = NeuroVis(spike_times, name='PMd %d' % 1)
neuron_PMd.get_raster(event='left_rewards', df=df['left_rewards'])
neuron_PMd.get_psth(event='left_rewards', df=df)


# #Peristimulus time histogram (PSTH) visualization
# triggers = np.array(df.iloc[:,-1])
# reward_times = np.array(df.iloc[:,-2])
# bins = 10
# plt.hist(mapped_spikes, bins = bins, histtype='step')
# plt.axvline((reward_times[0]-trial_start_times[0]), color="r")
# plt.title("Histogram where red line is reward time")
# plt.xlabel("Time from stimulus onset [s]")
# plt.ylabel("Count of Spikes")
# plt.show()
