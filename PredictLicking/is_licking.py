#Run this script with one argument in the command line. That argument should be the csv file output of DLC
#This script is uses to decide whether the mouse is licking or not, and which spout
# import credit_assignment_project.dlc
import matplotlib.pyplot as plt
import electrophysiology.ingest_timesync as ingest
import PredictLicking.process_tongue_data as process
import PredictLicking.is_licking as lick
import numpy as np
import pandas as pd
import cv2
import sys

#test

#Extend data print rows
# pd.set_option("display.max_rows", None, "display.max_columns", None)

def is_licking(csv_path):
    #Change file name to run function
    df = process.process_data_spout(csv_path)
    df_len = process.process_data_spout(csv_path)
    #Filter for any two features predicted at 99%--------------------------
    df = df.loc[(df['C_T_L'] >= 0.99) & (df['F_E_L'] >= 0.99)
               |(df['C_T_L'] >= 0.99) & (df['RE_T_L'] >= 0.99)
               |(df['C_T_L'] >= 0.99) & (df['LE_T_L'] >= 0.99)
               |(df['LE_T_L'] >= 0.99) & (df['RE_T_L'] >= 0.99)
               |(df['LE_T_L'] >= 0.99) & (df['F_E_L'] >= 0.99)
               |(df['RE_T_L'] >= 0.99) & (df['F_E_L'] >= 0.99)]
    frame_is_licking = df.iloc[:,0].values
    return(df, frame_is_licking, df_len)

def is_licking_spout(df, csv):
    #Create a dataframe with frames where mouse is licking
    df, frame_is_licking, df_len = lick.is_licking(csv)

    #Find the middle X coord between cherry and grape spouts
    avg_LRight_GS_X = df["LR_GS_X"].mean()
    avg_RLeft_CS_X = df["RL_CS_X"].mean()
    Center_X = (avg_RLeft_CS_X - avg_LRight_GS_X) / 2
    Center_X = Center_X + avg_LRight_GS_X

    #Is licking left / grape spout
    is_licking_grape = df.loc[(df['RE_T_X'] < df["LR_GS_X"])
                             |(df['C_T_X'] < df["LR_GS_X"])
                             &(df['LE_T_X'] < df["LR_GS_X"])
                             |(df['LE_T_X'] < df["LC_GS_X"])
                             |(df['RE_T_X'] < Center_X)]

    is_licking_cherry = df.loc[(df['LE_T_X'] > df["RL_CS_X"])
                             |(df['C_T_X'] > df["RL_CS_X"])
                             &(df['RE_T_X'] > df["RL_CS_X"])
                             |(df['RE_T_X'] > df["RC_CS_X"])
                             |(df['LE_T_X'] > Center_X)]
    frames_licking_cherry = is_licking_cherry.iloc[:,0].values
    frames_licking_grape = is_licking_grape.iloc[:,0].values

    #Is licking neither spout
    #Convert to lists to use remove function
    centre_lick = set(frame_is_licking.flat) - set(frames_licking_cherry.flat) - set(frames_licking_grape.flat)
    return(frames_licking_cherry,frames_licking_grape,centre_lick)

def generate_licking_times(frametimes,dlc_csv):
    df, frame_is_licking, df_len = lick.is_licking(dlc_csv)
    cherry_frames, grape_frames, center_frames = lick.is_licking_spout(df, dlc_csv)

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

def lick2trial(lick_df,session_data):
    trial_df, spike_times, cluster_IDs = ingest.convert_mat(session_data)
    bins = trial_df["trial_start_times"]
    trial_df["trial_start_times"] = trial_df["trial_start_times"].apply(lambda x :int(x))
    lick_df["trial_start_times"] = pd.cut(lick_df["Time Licking"], bins=bins)
    lick_df["trial_start_times"] = lick_df["trial_start_times"].apply(lambda x: x.left)
    lick_df = lick_df.dropna()
    lick_df["trial_start_times"] = lick_df["trial_start_times"].apply(lambda x: int(x))
    lick_df = lick_df.merge(trial_df, how="left", on="trial_start_times")
    lick_df = lick_df.drop(columns=["nTrials","violations"])
    return(lick_df)
