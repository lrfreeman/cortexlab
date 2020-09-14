#Run this script with one argument in the command line. That argument should be the csv file output of DLC
#This script is uses to decide whether the mouse is licking or not, and which spout
import matplotlib.pyplot as plt
from process_tongue_data import *
import numpy as np
import cv2
import sys

#Misc----------------------------------------------------------------------
test_file = "/Users/laurence/Desktop/Neuroscience/mproject/data/Analysed_videos_CSVs_DLC/video_snippet_KM011_2020-03-19_trial90DLC_resnet50_Master_ProjectAug13shuffle1_200000.csv"
#Code to overlay frmes for CML
#ffmpeg -i video.mov -vf "drawtext=fontfile=Arial.ttf: text=%{n}: x=(w-tw)/2: y=h-(2*lh): fontcolor=white: box=1: boxcolor=0x00000099" -y output.mov

def is_licking(video_path):
    #Change file name to run function
    df = process_data_spout(video_path)
    #Filter for any two features predicted at 99%--------------------------
    df = df.loc[(df['C_T_L'] >= 0.99) & (df['F_E_L'] >= 0.99)
               |(df['C_T_L'] >= 0.99) & (df['RE_T_L'] >= 0.99)
               |(df['C_T_L'] >= 0.99) & (df['LE_T_L'] >= 0.99)
               |(df['LE_T_L'] >= 0.99) & (df['RE_T_L'] >= 0.99)
               |(df['LE_T_L'] >= 0.99) & (df['F_E_L'] >= 0.99)
               |(df['RE_T_L'] >= 0.99) & (df['F_E_L'] >= 0.99)]
    #Test
    # print(df)

    #Plot probability of tongue licking------------------------------------
    # df.plot(x='Frames',y= ['LE_T_L','RE_T_L','F_E_L','C_T_L'])
    # plt.ylabel("Proability")
    # plt.show()

    #What frames is the mouse is licking?
    frame_is_licking = df.iloc[:,0].values
    # print()
    # print("The mouse is licking at these specifc frames:")
    # print()
    # print(frame_is_licking)
    # print()
    return(df, frame_is_licking)

def is_licking_spout(df):
    #Create a dataframe with frames where mouse is licking
    df, frame_is_licking = is_licking(test_file)

    #Is licking left or right spout
    is_licking_grape = df.loc[(df['RE_T_X'] <= df["LR_GS_X"])
                             |(df['C_T_X'] <= df["LR_GS_X"])
                             &(df['LE_T_X'] <= df["LR_GS_X"])]
    frames_licking_grape = is_licking_grape.iloc[:,0].values
    print("The mouse is licking the Grape spout on these frames:", frames_licking_grape)

#Test
df, frame_is_licking = is_licking(test_file)
is_licking_spout(df)
print("The mouse is licking on these frames:", frame_is_licking)
# is_licking(sys.argv[1])
