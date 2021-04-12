from by_hand import GKR
import sys
sys.path.insert(1,'/Users/laurence/Desktop/Neuroscience/kevin_projects/code/mousetask')
import DLC_Classes as CL
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go

#Load data
data = CL.CortexLab('/Users/laurence/Desktop/Neuroscience/kevin_projects/data/processed_physdata/aligned_physdata_KM011_2020-03-23_probe1.mat',
                    '/Users/laurence/Desktop/Neuroscience/kevin_projects/data/KM011_video_timestamps/2020-03-23/face_timeStamps.mat',
                    '/Users/laurence/Desktop/Neuroscience/kevin_projects/data/23_faceDLC_resnet50_Master_ProjectAug13shuffle1_133500.csv')
trial_df, spike_df = data.load_data(data.session_data)
first_lick_df, lick_df, df = data.compute_the_first_lick()

#Bins are hard coded for the above function in the class, the below is to use to assign licks to a bin not for bucketing of spikes
bins = np.arange(-1,3,0.2).tolist()

#Calculate licking data and spike into bins
firing_rates = data.binned_firing_rate_calculations(1)
firing_rates = pd.DataFrame(firing_rates)
firing_rates["bins"] = np.array(bins)
firing_rates = firing_rates.rename(columns={0: "firing_rates"})
licking_rates = data.binned_licking_calculations()
firing_rates["licking_rates"] = licking_rates
training_data = firing_rates
print(training_data)

#Conduct Kernel Regression
gkr = GKR(training_data["licking_rates"], training_data["firing_rates"], 10)
# for x in range(100):
#     print(gkr.predict(x))

# # Plot the x and y values
fig = px.scatter(x=training_data["licking_rates"],y=training_data["firing_rates"], title='Figure 1:  Visualizing the generated data')
fig.show()
