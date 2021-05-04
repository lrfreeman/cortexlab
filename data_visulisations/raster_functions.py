import matplotlib.pyplot as plt

def new_generate_raster(trial_df, spike_df, cellID):

    #####Choose a cell#######
    spike_df = spike_df.loc[(spike_df["cluster_ids"] == cellID)]

    #Generate spikes for each trial
    trial_spike_times = lock_and_sort_for_raster(spike_df["Spike_Times"],trial_df)

    # Seperate spikes per trial type
    cherrySpikeValues = count_to_trial(cherry_reward_trials, trial_spike_times)
    grapeSpikeValues = count_to_trial(grape_reward_trials, trial_spike_times)
    bothRewardSpikeValues = count_to_trial(both_reward_trials, trial_spike_times)
    noRewardSpikeValues = count_to_trial(no_reward_trials, trial_spike_times)

    #SO that we can create a correspondding colour length for event plot
    lenOfCherryTrials = len(cherrySpikeValues)
    lenOfGrapeTrials = len(grapeSpikeValues)
    lenOfBothRewardTrials = len(bothRewardSpikeValues)
    lenOfNoRewardTrials = len(noRewardSpikeValues)

    #convert to np array
    cherrySpikeValues = np.asarray(cherrySpikeValues)
    grapeSpikeValues = np.asarray(grapeSpikeValues)
    bothRewardSpikeValues = np.asarray(bothRewardSpikeValues)
    noRewardSpikeValues = np.asarray(noRewardSpikeValues)

    def prepare_data_for_scatter(trial_index_modifier, trial_type_spike_values, len_of_trial_type):
        dic_of_dfs = {}
        for trial in range(len_of_trial_type):
            dic_of_dfs[trial] = pd.DataFrame(trial_type_spike_values[trial], columns=["spikes"])
            dic_of_dfs[trial].index = ([trial + trial_index_modifier]) * trial_type_spike_values.shape[1]
        x = []
        y = []
        for trial in range(len(dic_of_dfs)):
            df = dic_of_dfs[trial]
            x.extend(df["spikes"].values)
            y.extend(df.index.to_numpy())
        return(x,y)

    m1 = 0
    m2 = lenOfCherryTrials
    m3 = m2 + lenOfGrapeTrials
    m4 = m3 + lenOfBothRewardTrials

    cherryx, cherryy = prepare_data_for_scatter(m1, cherrySpikeValues, lenOfCherryTrials)
    grapex, grapey = prepare_data_for_scatter(m2, grapeSpikeValues, lenOfGrapeTrials)
    bothx, bothy = prepare_data_for_scatter(m3, bothRewardSpikeValues, lenOfBothRewardTrials)
    nox, noy = prepare_data_for_scatter(m4, noRewardSpikeValues, lenOfNoRewardTrials)

    return(cherryx,cherryy,grapex,grapey,bothx,bothy,nox,noy)

#Raster locked to reward - Returns a dictionary with key trial number and value locked spike times
def lock_and_sort_for_raster(time,trial_df):
    lock_time = {}
    trial_spike_times = {}
    for trial in range(len(trial_df)):
        lock_time[trial] = trial_df["reward_times"][trial]
        trial_spike_times[trial] = time-lock_time[trial]
    return(trial_spike_times)

# Function to generate Spike Raster
def generate_raster(trial_df, spike_df, cellID):

    #####Choose a cell#######
    spike_df = spike_df.loc[(spike_df["cluster_ids"] == cellID)]

    #Generate spikes for each trial
    trial_spike_times = lock_and_sort_for_raster(spike_df["Spike_Times"],trial_df)

    # Seperate spikes per trial type
    cherrySpikeValues = count_to_trial(cherry_reward_trials, trial_spike_times)
    grapeSpikeValues = count_to_trial(grape_reward_trials, trial_spike_times)
    bothRewardSpikeValues = count_to_trial(both_reward_trials, trial_spike_times)
    noRewardSpikeValues = count_to_trial(no_reward_trials, trial_spike_times)

    #SO that we can create a correspondding colour length for event plot
    lenOfCherryTrials = len(cherrySpikeValues)
    lenOfGrapeTrials = len(grapeSpikeValues)
    lenOfBothRewardTrials = len(bothRewardSpikeValues)
    lenOfNoRewardTrials = len(noRewardSpikeValues)

    #convert to np array
    cherrySpikeValues = np.asarray(cherrySpikeValues)
    grapeSpikeValues = np.asarray(grapeSpikeValues)
    bothRewardSpikeValues = np.asarray(bothRewardSpikeValues)
    noRewardSpikeValues = np.asarray(noRewardSpikeValues)

    #Concaternate arrays
    spikes = np.concatenate((cherrySpikeValues,grapeSpikeValues,bothRewardSpikeValues,noRewardSpikeValues))

    #Create colorCodes
    colorCodesCherry = [[1,0,0]] * lenOfCherryTrials
    colorCodesGrape = [[1,0,1]] * lenOfGrapeTrials
    colorCodesBothReward = [[0,0,1]] * lenOfBothRewardTrials
    colorCodesNoReward = [[0,0,0]] * lenOfNoRewardTrials
    colorCodes = colorCodesCherry + colorCodesGrape + colorCodesBothReward + colorCodesNoReward

    return(colorCodes, spikes)

colorCodes, spikes = generate_raster(trial_df,spike_df,cellID)

#Plot spike Raster
ax2.scatter(cherryx,cherryy, marker = "|", color='r', linewidths=0.1, alpha = 0.2,   s=0.2)
ax2.scatter(grapex,grapey, marker = "|", color='m', linewidths=0.1,   alpha = 0.2,   s=0.2)
ax2.scatter(bothx,bothy, marker = "|", color='b', linewidths=0.1,     alpha = 0.2,   s=0.2)
ax2.scatter(nox,noy, marker = "|", color='k', linewidths=0.1,         alpha = 0.2,   s=0.2)
ax2.set_xlim(right=3)
ax2.set_xlim(left=-1)
ax2.set(title="Spike Raster", xlabel="Time (s)", ylabel="Trials")
