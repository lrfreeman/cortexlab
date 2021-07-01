import utils as util
import matplotlib.pyplot as plt
import pandas as pd

"""PSTH Logic"""
def prep_data_for_PSTH(trial_df, spike_df):
    cherry_reward_trials, grape_reward_trials, both_reward_trials, no_reward_trials = util.split_data_by_trial_type(trial_df)
    spike_counts, bin_edges, bin_centres = util.lock_to_reward_and_count(spike_df, trial_df)
    cherry_spike_counts = util.count_to_trial(cherry_reward_trials, spike_counts)
    grape_spike_counts = util.count_to_trial(grape_reward_trials, spike_counts)
    both_reward_spike_counts = util.count_to_trial(both_reward_trials, spike_counts)
    no_reward_spike_counts = util.count_to_trial(no_reward_trials, spike_counts)
    return(cherry_spike_counts,
           grape_spike_counts,
           both_reward_spike_counts,
           no_reward_spike_counts,
           bin_edges,
           bin_centres)

def calculate_firing_rates_for_PSTH(trial_df, spike_df):
    cherry_spike_counts, grape_spike_counts, both_reward_spike_counts, no_reward_spike_counts, bin_edges, bin_centres = prep_data_for_PSTH(trial_df, spike_df)

    """-----------------Calculate firing rates-------------------------------"""
    #*5 because bin size is 200ms within  the lock function in utils
    cherry_count = pd.DataFrame(cherry_spike_counts).sum(axis=0)
    cherry_hertz = (cherry_count / len(cherry_spike_counts)) * 5

    grape_count = pd.DataFrame(grape_spike_counts).sum(axis=0)
    grape_hertz = (grape_count / len(grape_spike_counts)) * 5

    both_reward_count = pd.DataFrame(both_reward_spike_counts).sum(axis=0)
    both_reward_hertz = (both_reward_count / len(both_reward_spike_counts)) * 5

    no_reward_count = pd.DataFrame(no_reward_spike_counts).sum(axis=0)
    no_reward_hertz = (no_reward_count / len(no_reward_spike_counts)) * 5
    return(cherry_hertz, grape_hertz, both_reward_hertz, no_reward_hertz, bin_edges, bin_centres)

def generate_PSTH(trial_df, spike_df):
    cherry_hertz, grape_hertz, both_reward_hertz, no_reward_hertz, bin_edges, bin_centres = calculate_firing_rates_for_PSTH(trial_df, spike_df)

    #Outline subplots
    fig, (ax1) = plt.subplots(1, sharex=True)

    # #Plot PSTH
    ax1.plot(bin_centres,cherry_hertz[:-1], color='r', label="Cherry Reward")
    ax1.plot(bin_centres,grape_hertz[:-1], color='m', label="Grape Reward" )
    ax1.plot(bin_centres,both_reward_hertz[:-1], color='b', label="Both Reward")
    ax1.plot(bin_centres,no_reward_hertz[:-1], color='k', label="No Reward")
    ax1.legend(loc='upper right')
    # ax1.set(title="PSTH - Locked to reward - Cell:{} - MOs".format(cell_ID), ylabel="Firing Rates (sp/s)")
    plt.show()
