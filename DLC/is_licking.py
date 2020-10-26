#Run this script with one argument in the command line. That argument should be the csv file output of DLC
#This script is uses to decide whether the mouse is licking or not, and which spout
import matplotlib.pyplot as plt
from process_tongue_data import *
import numpy as np
import cv2
import sys

#Misc----------------------------------------------------------------------
test_file = "/Users/laurence/Desktop/Neuroscience/mproject/data/24_faceDLC_resnet50_Master_ProjectAug13shuffle1_200000.csv"
#Code to overlay frmes for CML
#ffmpeg -i video.mov -vf "drawtext=fontfile=Arial.ttf: text=%{n}: x=(w-tw)/2: y=h-(2*lh): fontcolor=white: box=1: boxcolor=0x00000099" -y output.mov

#Extend data print rows
# pd.set_option("display.max_rows", None, "display.max_columns", None)

def is_licking(csv_path):
    #Change file name to run function
    df = process_data_spout(csv_path)
    df_len = process_data_spout(csv_path)
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
    df, frame_is_licking, df_len = is_licking(csv)

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

def generate_licking_times():
    df, frame_is_licking, df_len = is_licking(test_file)
    cherry_frames, grape_frames, centre_lick = is_licking_spout(df, test_file)

    #adjust frames to start a 1 so there is a one off error before converting to time
    cherry_frames = [x + 1 for x in cherry_frames]
    grape_frames = [x + 1 for x in grape_frames]
    center_frames = [x + 1 for x in centre_lick]

    #Create three dfs
    cherry_licking_df = pd.DataFrame(cherry_frames, columns = ["frames licking"])
    grape_licking_df = pd.DataFrame(grape_frames, columns = ["frames licking"])
    center_licking_df = pd.DataFrame(center_frames, columns = ["frames licking"])

    # #Given 30FPS and 0.0333333 seconds per frame calculate time of lick
    cherry_licking_df["time licking"] = cherry_licking_df["frames licking"] * 0.033333333333
    grape_licking_df["time licking"] = grape_licking_df["frames licking"] * 0.033333333333
    center_licking_df["time licking"] = center_licking_df["frames licking"] * 0.033333333333
    return(cherry_licking_df, grape_licking_df, center_licking_df)
