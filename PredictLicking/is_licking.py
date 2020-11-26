#This script is uses to decide whether the mouse is licking or not, and which spout
import matplotlib.pyplot as plt
import electrophysiology.ingest_timesync as ingest
import PredictLicking.process_tongue_data as process
import PredictLicking.is_licking as lick
import numpy as np
import pandas as pd
from decimal import *
import math

#Extend data print rows
# pd.set_option("display.max_rows", None, "display.max_columns", None)
# pd.set_option("display.precision", 10)

def is_licking(csv_path):

    #Change file name to run function
    df = process.process_data_spout(csv_path)
    df = df.set_index('Frames')
    # df_len = process.process_data_spout(csv_path)
    df = df.astype(float)

    # print(df.iloc[282])

    # Filter for any two features predicted at 99%--------------------------
    # df = df.loc[(df['C_T_L'] >= 0.99) & (df['F_E_L'] >= 0.99)
    #            |(df['C_T_L'] >= 0.99) & (df['RE_T_L'] >= 0.99)
    #            |(df['C_T_L'] >= 0.99) & (df['LE_T_L'] >= 0.99)
    #            |(df['LE_T_L'] >= 0.99) & (df['RE_T_L'] >= 0.99)
    #            |(df['LE_T_L'] >= 0.99) & (df['F_E_L'] >= 0.99)
    #            |(df['RE_T_L'] >= 0.99) & (df['F_E_L'] >= 0.99)]
    df = df.loc[(df['C_T_L'] >= 0.99)
               &(df['RE_T_L'] >= 0.99)
               &(df['LE_T_L'] >= 0.99)]

    frame_is_licking = df.index.values
    frame_is_licking = np.array([int(x) for x in frame_is_licking])

    #Print how many frames the mouse is licking
    # print("The total frames the mouse is licking",len(df.values))
    print("The total frames the mouse is licking",len(frame_is_licking))
    return(df,frame_is_licking)

def is_licking_spout(df, csv):
    #Create a dataframe with frames where mouse is licking
    df, frame_is_licking = lick.is_licking(csv)

    #Find the middle X coord between cherry and grape spouts
    avg_LRight_GS_X = df["LR_GS_X"].mean()
    avg_RLeft_CS_X = df["RL_CS_X"].mean()
    Center_X = (avg_RLeft_CS_X - avg_LRight_GS_X) / 2
    Center_X = Center_X + avg_LRight_GS_X

    #Is licking left / grape spout
    is_licking_grape =  df.loc[(df['RE_T_X'] < df["LR_GS_X"])
                              |(df['C_T_X']  < df["LC_GS_X"])
                              |(df['LE_T_X'] < df["LC_GS_X"])
                              |(df['RE_T_X'] < Center_X)]

    #Is licking right / cherry spout
    is_licking_cherry = df.loc[(df['LE_T_X'] > df["RL_CS_X"])
                              |(df['C_T_X']  > df["RC_CS_X"])
                              |(df['RE_T_X'] > df["RC_CS_X"])
                              |(df['LE_T_X'] > Center_X)]

    #Convert data frame into values array
    frames_licking_cherry = is_licking_cherry.index.values
    frames_licking_grape = is_licking_grape.index.values
    frames_licking_cherry = np.array([int(x) for x in frames_licking_cherry])
    frames_licking_grape = np.array([int(x) for x in frames_licking_grape])

    #Is licking neither spout
    #Convert to lists to use remove function
    centre_lick = set(frame_is_licking.flat) - set(frames_licking_cherry.flat) - set(frames_licking_grape.flat)
    centre_lick = np.array([int(x) for x in centre_lick])
    return(frames_licking_cherry,frames_licking_grape,centre_lick)

def generate_licking_times(frametimes,dlc_csv):
    df, frame_is_licking = lick.is_licking(dlc_csv)
    cherry_frames, grape_frames, center_frames = lick.is_licking_spout(df, dlc_csv)

    print("")
    print("#############################################")
    print("Number of frames the mouse is licking cherry",      len(cherry_frames))
    print("Number of frames the mouse is licking grape",       len(grape_frames))
    print("Number of frames the mouse is licking the center",  len(center_frames))
    print("#############################################")
    print("")

    #Create three dfs for reward licking
    cherry_licking_df = pd.DataFrame(cherry_frames, columns = ["frames licking"])
    grape_licking_df = pd.DataFrame(grape_frames, columns = ["frames licking"])
    center_licking_df = pd.DataFrame(center_frames, columns = ["frames licking"])

    #turn frametimes into a df
    frametimes_df = pd.DataFrame(frametimes)

    #Merge aligned licking times to the each data frame
    cherry_licking_df = cherry_licking_df.merge(frametimes_df,
                                                how="left",left_on="frames licking",
                                                right_index=True)
    cherry_licking_df.columns = ["Frames Licking", "Time Licking"]
    grape_licking_df = grape_licking_df.merge(frametimes_df,
                                                how="left",left_on="frames licking",
                                                right_index=True)
    grape_licking_df.columns = ["Frames Licking", "Time Licking"]
    center_licking_df = center_licking_df.merge(frametimes_df,
                                                how="left",left_on="frames licking",
                                                right_index=True)
    center_licking_df.columns = ["Frames Licking", "Time Licking"]

    #Add type before merge of all three
    cherry_licking_df["Cherry Lick"] = 1
    grape_licking_df["Grape Lick"] = 1
    center_licking_df["Center Lick"] = 1
    df = pd.concat([cherry_licking_df, grape_licking_df,center_licking_df], axis = 0, ignore_index=True)
    df = df.fillna(value=0)
    df.sort_values(by="Frames Licking", ignore_index=True)

    #--------------
    return(df)

#This function assigns trial data to each licking prediction
#So that the PSTH can be filtered by trial type
def map_lick_to_trial_type(lick_df,session_data):

    # print(lick_df._is_view)
    #Use the convert_mat function to get trial_df
    trial_df, spike_times, cluster_IDs = ingest.convert_mat(session_data)

    #Assign a trial start time to each lick
    lick_df["trial_start_times"] = pd.cut(lick_df["Time Licking"], trial_df["trial_start_times"])
    lick_df["trial_start_times"] = (lick_df["trial_start_times"].apply(lambda x: x.left))

    #Make a copy to prevent the settingwithcopy warning
    #checked both dataframes on describe and seem to be identical
    lick_df = lick_df.copy()

    # #Turn Trial start time into trial ID for merging
    lick_df = lick_df.dropna() #Need to do this for the truncate function
    trunc = lambda x: math.trunc(x)
    lick_df["trial_start_times"] = (lick_df["trial_start_times"].apply(trunc))
    lick_df = lick_df.rename(columns={'trial_start_times': 'Trial_ID'})

    #Merge trial dataframe and lick data frame on trial ID
    trial_df["Trial_ID"] = trial_df["trial_start_times"].apply(trunc)
    lick_df = lick_df.merge(trial_df, on="Trial_ID")

    #Output a data frame with an array of licks and what trial type that lick occured in
    return(lick_df)

def compute_1st_lick(lick_df):
    #Potential bug as the number of first licks don't match total trial len
    data_frame = lick_df.sort_values(by="Time Licking", ignore_index=True)
    post_reward_lick_dict = {}
    first_lick = {}
    for row in range(len(data_frame)):
        if data_frame["Time Licking"][row] < data_frame["reward_times"][row]:
            data_frame = data_frame.drop(row, axis=0)

    for trial in data_frame["Trial_ID"]:
        indexx = data_frame[data_frame["Trial_ID"] == trial].index.values
        indexx = indexx[0]
        if data_frame["Time Licking"][indexx] >= data_frame["reward_times"][indexx]:
            first_lick[trial] = data_frame["Time Licking"][indexx]
    first_lick_df = pd.DataFrame()
    first_lick_df["Trial IDs"] = list(first_lick.keys())
    first_lick_df["First Lick Times"] = list(first_lick.values())

    print("Length of first lick data frame, should match trial len:", len(first_lick))
    print(first_lick_df)
    return(first_lick_df)
